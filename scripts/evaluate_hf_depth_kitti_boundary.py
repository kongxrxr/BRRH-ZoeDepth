#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _remove_leading_slash(path):
    return path[1:] if path.startswith("/") else path


def _edge_masks(inv_depth, threshold):
    right = (inv_depth[:, 1:] / np.clip(inv_depth[:, :-1], 1e-6, None)) > threshold
    left = (inv_depth[:, :-1] / np.clip(inv_depth[:, 1:], 1e-6, None)) > threshold
    bottom = (inv_depth[1:, :] / np.clip(inv_depth[:-1, :], 1e-6, None)) > threshold
    top = (inv_depth[:-1, :] / np.clip(inv_depth[1:, :], 1e-6, None)) > threshold
    return left, top, right, bottom


def _valid_pair_masks(valid):
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:, :] & valid[:-1, :]
    return horizontal, vertical, horizontal, vertical


def _masked_boundary_f1(pred_depth, gt_depth, valid, threshold):
    pred_inv = 1.0 / np.clip(pred_depth, 1e-6, None)
    gt_inv = 1.0 / np.clip(gt_depth, 1e-6, None)
    pred_edges = _edge_masks(pred_inv, threshold)
    gt_edges = _edge_masks(gt_inv, threshold)
    valid_pairs = _valid_pair_masks(valid)

    recalls = []
    precisions = []
    for pred_edge, gt_edge, valid_pair in zip(pred_edges, gt_edges, valid_pairs):
        pred_edge = pred_edge & valid_pair
        gt_edge = gt_edge & valid_pair
        hit = np.count_nonzero(pred_edge & gt_edge)
        recalls.append(hit / max(np.count_nonzero(gt_edge), 1))
        precisions.append(hit / max(np.count_nonzero(pred_edge), 1))

    recall = float(np.mean(recalls))
    precision = float(np.mean(precisions))
    if recall + precision == 0:
        return 0.0, precision, recall
    return 2 * recall * precision / (recall + precision), precision, recall


def si_boundary_f1_masked(pred_depth, gt_depth, valid, t_min=1.05, t_max=1.25, n=10):
    thresholds = np.linspace(t_min, t_max, n)
    weights = thresholds / thresholds.sum()
    scores, precisions, recalls = [], [], []
    for threshold in thresholds:
        f1, precision, recall = _masked_boundary_f1(pred_depth, gt_depth, valid, threshold)
        scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
    return {
        "si_boundary_f1": float(np.sum(np.array(scores) * weights)),
        "si_boundary_precision": float(np.sum(np.array(precisions) * weights)),
        "si_boundary_recall": float(np.sum(np.array(recalls) * weights)),
    }


def _pixel_boundary_from_pairs(edge_x, edge_y):
    boundary = np.zeros((edge_y.shape[0] + 1, edge_x.shape[1] + 1), dtype=bool)
    boundary[:, 1:] |= edge_x
    boundary[:, :-1] |= edge_x
    boundary[1:, :] |= edge_y
    boundary[:-1, :] |= edge_y
    return boundary


def boundary_region_metrics(pred_depth, gt_depth, valid, log_grad_threshold=0.15):
    pred = np.clip(pred_depth, 1e-6, None)
    gt = np.clip(gt_depth, 1e-6, None)
    log_pred = np.log(pred)
    log_gt = np.log(gt)

    gt_dx = log_gt[:, 1:] - log_gt[:, :-1]
    pred_dx = log_pred[:, 1:] - log_pred[:, :-1]
    valid_dx = valid[:, 1:] & valid[:, :-1]
    edge_x = valid_dx & (np.abs(gt_dx) > log_grad_threshold)
    nonedge_x = valid_dx & (np.abs(gt_dx) <= log_grad_threshold)

    gt_dy = log_gt[1:, :] - log_gt[:-1, :]
    pred_dy = log_pred[1:, :] - log_pred[:-1, :]
    valid_dy = valid[1:, :] & valid[:-1, :]
    edge_y = valid_dy & (np.abs(gt_dy) > log_grad_threshold)
    nonedge_y = valid_dy & (np.abs(gt_dy) <= log_grad_threshold)

    boundary = _pixel_boundary_from_pairs(edge_x, edge_y) & valid
    nonboundary = (~boundary) & valid

    abs_err = np.abs(pred - gt)
    sq_err = (pred - gt) ** 2
    rel_err = abs_err / gt

    def _mean_or_zero(values, mask):
        if np.count_nonzero(mask) == 0:
            return 0.0
        return float(values[mask].mean())

    def _rmse_or_zero(values, mask):
        if np.count_nonzero(mask) == 0:
            return 0.0
        return float(np.sqrt(values[mask].mean()))

    jump_losses = []
    jump_rel_losses = []
    if np.count_nonzero(edge_x) > 0:
        jump_losses.append(np.abs(pred_dx - gt_dx)[edge_x])
        jump_rel_losses.append((np.abs(pred_dx) / np.maximum(np.abs(gt_dx), 1e-6))[edge_x])
    if np.count_nonzero(edge_y) > 0:
        jump_losses.append(np.abs(pred_dy - gt_dy)[edge_y])
        jump_rel_losses.append((np.abs(pred_dy) / np.maximum(np.abs(gt_dy), 1e-6))[edge_y])

    smooth_losses = []
    if np.count_nonzero(nonedge_x) > 0:
        smooth_losses.append(np.abs(pred_dx)[nonedge_x])
    if np.count_nonzero(nonedge_y) > 0:
        smooth_losses.append(np.abs(pred_dy)[nonedge_y])

    boundary_count = int(np.count_nonzero(boundary))
    valid_count = int(np.count_nonzero(valid))
    return {
        "boundary_abs_rel": _mean_or_zero(rel_err, boundary),
        "boundary_rmse": _rmse_or_zero(sq_err, boundary),
        "nonboundary_abs_rel": _mean_or_zero(rel_err, nonboundary),
        "nonboundary_rmse": _rmse_or_zero(sq_err, nonboundary),
        "boundary_log_jump_mae": float(np.concatenate(jump_losses).mean()) if jump_losses else 0.0,
        "boundary_log_jump_ratio": float(np.concatenate(jump_rel_losses).mean()) if jump_rel_losses else 0.0,
        "nonboundary_log_smoothness": float(np.concatenate(smooth_losses).mean()) if smooth_losses else 0.0,
        "boundary_pixel_fraction": boundary_count / max(valid_count, 1),
    }


def average_dicts(dicts):
    keys = dicts[0].keys()
    return {key: float(np.mean([d[key] for d in dicts])) for key in keys}


def _make_eval_mask(gt_depth, args):
    valid = (gt_depth > args.min_depth_eval) & (gt_depth < args.max_depth_eval)
    if args.garg_crop or args.eigen_crop:
        h, w = gt_depth.shape
        crop = np.zeros_like(valid)
        if args.garg_crop:
            crop[
                int(0.40810811 * h):int(0.99189189 * h),
                int(0.03594771 * w):int(0.96405229 * w),
            ] = True
        else:
            crop[
                int(0.3324324 * h):int(0.91351351 * h),
                int(0.0359477 * w):int(0.96405229 * w),
            ] = True
        valid &= crop
    return valid


def _compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    log_10 = np.abs(np.log10(gt) - np.log10(pred)).mean()
    return {
        "a1": float(a1),
        "a2": float(a2),
        "a3": float(a3),
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "log_10": float(log_10),
        "rmse_log": float(rmse_log),
        "silog": float(silog),
        "sq_rel": float(sq_rel),
    }


def _compute_metrics(gt_depth, pred_depth, valid):
    return _compute_errors(gt_depth[valid], pred_depth[valid])


def _median_align(pred_depth, gt_depth, valid):
    pred_valid = pred_depth[valid]
    gt_valid = gt_depth[valid]
    pred_med = np.median(pred_valid)
    gt_med = np.median(gt_valid)
    if pred_med <= 1e-6 or gt_med <= 1e-6:
        return pred_depth
    return pred_depth * (gt_med / pred_med)


def _prediction_to_depth(prediction, mode, eps=1e-6):
    prediction = np.asarray(prediction, dtype=np.float32)
    if mode == "inverse":
        return 1.0 / np.clip(prediction, eps, None)
    return prediction


def _load_kitti_depth(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 256.0


def _kitti_benchmark_crop(image, depth):
    height, width = depth.shape
    top = int(height - 352)
    left = int((width - 1216) / 2)
    if hasattr(image, "crop"):
        image = image.crop((left, top, left + 1216, top + 352))
    else:
        image = image[top:top + 352, left:left + 1216]
    depth = depth[top:top + 352, left:left + 1216]
    return image, depth


def _iter_kitti_eval_samples(args):
    with Path(args.filenames_file_eval).open("r") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            if fields[1] == "None":
                continue
            image_path = Path(args.data_path_eval) / _remove_leading_slash(fields[0])
            depth_path = Path(args.gt_path_eval) / _remove_leading_slash(fields[1])
            yield image_path, depth_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model", default="LiheYoung/depth-anything-small-hf")
    parser.add_argument("--output", default="")
    parser.add_argument("--prediction_mode", choices=["depth", "inverse"], default="depth")
    parser.add_argument("--align", choices=["none", "median"], default="median")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--filenames_file_eval", default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt")
    parser.add_argument("--data_path_eval", default="/home/kxr/shortcuts/datasets/kitti/raw")
    parser.add_argument("--gt_path_eval", default="/home/kxr/shortcuts/datasets/kitti/gts")
    parser.add_argument("--min_depth_eval", type=float, default=0.001)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--garg_crop", type=int, default=1)
    parser.add_argument("--eigen_crop", type=int, default=0)
    parser.add_argument("--do_kb_crop", type=int, default=1)
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(args.hf_model)
    model = AutoModelForDepthEstimation.from_pretrained(args.hf_model).to(device).eval()
    samples = list(_iter_kitti_eval_samples(args))

    regular_metrics = []
    boundary_metrics = []
    total = len(samples) if args.max_samples <= 0 else min(args.max_samples, len(samples))

    for image_path, depth_path in tqdm(samples[:total], total=total):
        if args.max_samples > 0 and len(regular_metrics) >= args.max_samples:
            break

        image = Image.open(image_path).convert("RGB")
        depth = _load_kitti_depth(depth_path)
        if args.do_kb_crop:
            image, depth = _kitti_benchmark_crop(image, depth)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prediction = model(**inputs).predicted_depth
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=depth.shape,
            mode="bicubic",
            align_corners=False,
        ).squeeze().detach().cpu().numpy().astype(np.float32)

        pred_depth = _prediction_to_depth(prediction, args.prediction_mode)
        valid = _make_eval_mask(depth, args)
        if args.align == "median":
            pred_depth = _median_align(pred_depth, depth, valid)
        pred_depth = np.clip(pred_depth, args.min_depth_eval, args.max_depth_eval)

        regular_metrics.append(_compute_metrics(depth, pred_depth, valid))
        boundary_metrics.append({
            **si_boundary_f1_masked(pred_depth, depth, valid),
            **boundary_region_metrics(
                pred_depth,
                depth,
                valid,
                log_grad_threshold=args.boundary_log_grad_threshold,
            ),
        })

    result = {
        **average_dicts(regular_metrics),
        **average_dicts(boundary_metrics),
        "num_samples": len(regular_metrics),
        "hf_model": args.hf_model,
        "prediction_mode": args.prediction_mode,
        "align": args.align,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
