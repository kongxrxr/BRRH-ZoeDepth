#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate import infer
from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.models.model_io import load_wts
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import compute_metrics


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


def _dilate_mask(mask, radius):
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for y in range(size):
        for x in range(size):
            out |= padded[y:y + mask.shape[0], x:x + mask.shape[1]]
    return out


def _mean_or_zero(values, mask):
    if np.count_nonzero(mask) == 0:
        return 0.0
    return float(values[mask].mean())


def _rmse_or_zero(values, mask):
    if np.count_nonzero(mask) == 0:
        return 0.0
    return float(np.sqrt(values[mask].mean()))


def _edge_pixel_mask_from_log_depth(log_depth, valid, threshold):
    dx = np.abs(log_depth[:, 1:] - log_depth[:, :-1])
    valid_dx = valid[:, 1:] & valid[:, :-1]
    edge_x = valid_dx & (dx > threshold)

    dy = np.abs(log_depth[1:, :] - log_depth[:-1, :])
    valid_dy = valid[1:, :] & valid[:-1, :]
    edge_y = valid_dy & (dy > threshold)
    return _pixel_boundary_from_pairs(edge_x, edge_y) & valid


def _tolerant_edge_f1(pred_edge, gt_edge, valid, radius):
    pred_edge = pred_edge & valid
    gt_edge = gt_edge & valid
    pred_count = np.count_nonzero(pred_edge)
    gt_count = np.count_nonzero(gt_edge)
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0, 0.0, 0.0

    gt_band = _dilate_mask(gt_edge, radius) & valid
    pred_band = _dilate_mask(pred_edge, radius) & valid
    precision = np.count_nonzero(pred_edge & gt_band) / max(pred_count, 1)
    recall = np.count_nonzero(gt_edge & pred_band) / max(gt_count, 1)
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def hard_boundary_metrics(pred_depth, gt_depth, valid, log_grad_threshold=0.15,
                          top_percentiles=(5.0, 10.0), band_radii=(3, 5),
                          f1_tolerances=(1, 3, 5)):
    pred = np.clip(pred_depth, 1e-6, None)
    gt = np.clip(gt_depth, 1e-6, None)
    log_pred = np.log(pred)
    log_gt = np.log(gt)

    abs_err = np.abs(pred - gt)
    sq_err = (pred - gt) ** 2
    rel_err = abs_err / gt
    log_abs_err = np.abs(log_pred - log_gt)

    dx = np.zeros_like(gt)
    dy = np.zeros_like(gt)
    valid_dx = valid[:, 1:] & valid[:, :-1]
    valid_dy = valid[1:, :] & valid[:-1, :]
    dx[:, 1:] = np.maximum(dx[:, 1:], np.abs(log_gt[:, 1:] - log_gt[:, :-1]))
    dx[:, :-1] = np.maximum(dx[:, :-1], np.abs(log_gt[:, 1:] - log_gt[:, :-1]))
    dy[1:, :] = np.maximum(dy[1:, :], np.abs(log_gt[1:, :] - log_gt[:-1, :]))
    dy[:-1, :] = np.maximum(dy[:-1, :], np.abs(log_gt[1:, :] - log_gt[:-1, :]))
    pair_valid_pixels = np.zeros_like(valid, dtype=bool)
    pair_valid_pixels[:, 1:] |= valid_dx
    pair_valid_pixels[:, :-1] |= valid_dx
    pair_valid_pixels[1:, :] |= valid_dy
    pair_valid_pixels[:-1, :] |= valid_dy
    grad_mag = np.maximum(dx, dy)
    grad_valid = valid & pair_valid_pixels

    gt_edge = _edge_pixel_mask_from_log_depth(log_gt, valid, log_grad_threshold)
    pred_edge = _edge_pixel_mask_from_log_depth(log_pred, valid, log_grad_threshold)

    result = {}
    for pct in top_percentiles:
        if np.count_nonzero(grad_valid) == 0:
            mask = np.zeros_like(valid, dtype=bool)
        else:
            cutoff = np.percentile(grad_mag[grad_valid], 100.0 - pct)
            mask = grad_valid & (grad_mag >= cutoff)
        suffix = f"top{int(pct)}"
        result[f"{suffix}_pixel_fraction"] = np.count_nonzero(mask) / max(np.count_nonzero(valid), 1)
        result[f"{suffix}_abs_rel"] = _mean_or_zero(rel_err, mask)
        result[f"{suffix}_rmse"] = _rmse_or_zero(sq_err, mask)
        result[f"{suffix}_log_mae"] = _mean_or_zero(log_abs_err, mask)

    for radius in band_radii:
        band = _dilate_mask(gt_edge, int(radius)) & valid
        suffix = f"band{int(radius)}"
        result[f"{suffix}_pixel_fraction"] = np.count_nonzero(band) / max(np.count_nonzero(valid), 1)
        result[f"{suffix}_abs_rel"] = _mean_or_zero(rel_err, band)
        result[f"{suffix}_rmse"] = _rmse_or_zero(sq_err, band)
        result[f"{suffix}_log_mae"] = _mean_or_zero(log_abs_err, band)

    for radius in f1_tolerances:
        f1, precision, recall = _tolerant_edge_f1(pred_edge, gt_edge, valid, int(radius))
        suffix = f"edge_f1_tol{int(radius)}"
        result[suffix] = f1
        result[f"{suffix}_precision"] = precision
        result[f"{suffix}_recall"] = recall

    return result


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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default="zoedepth")
    parser.add_argument("--dataset", default="kitti")
    parser.add_argument("--config_version", default="")
    parser.add_argument("--midas_model_type", default="DPT_Hybrid")
    parser.add_argument("--img_size", default="128,256")
    parser.add_argument("--data_path_eval", default="/home/kxr/shortcuts/datasets/kitti/raw")
    parser.add_argument("--gt_path_eval", default="/home/kxr/shortcuts/datasets/kitti/gts")
    parser.add_argument("--filenames_file_eval", default="")
    parser.add_argument("--min_depth_eval", type=float, default=0.001)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--garg_crop", type=int, default=1)
    parser.add_argument("--eigen_crop", type=int, default=0)
    parser.add_argument("--use_boundary_refine", type=int, default=0)
    parser.add_argument("--boundary_refine_channels", type=int, default=32)
    parser.add_argument("--boundary_refine_scale", type=float, default=0.1)
    parser.add_argument("--boundary_refine_mode", default="scale")
    parser.add_argument("--boundary_refine_use_da_prior", type=int, default=0)
    parser.add_argument("--use_discontinuity_branch", type=int, default=0)
    parser.add_argument("--discontinuity_channels", type=int, default=32)
    parser.add_argument("--use_discontinuity_temperature", type=int, default=0)
    parser.add_argument("--discontinuity_temperature_scale", type=float, default=1.5)
    parser.add_argument("--use_frozen_da_prior", type=int, default=0)
    parser.add_argument("--frozen_da_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--frozen_da_feature_channels", type=int, default=16)
    parser.add_argument("--frozen_da_input_size", type=int, default=384)
    parser.add_argument("--frozen_da_fusion_scale", type=float, default=0.1)
    parser.add_argument("--use_frozen_da_boundary_gate", type=int, default=0)
    parser.add_argument("--frozen_da_min_gate", type=float, default=0.05)
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    parser.add_argument("--hard_boundary_top_percentiles", default="5,10")
    parser.add_argument("--hard_boundary_band_radii", default="3,5")
    parser.add_argument("--hard_boundary_f1_tolerances", default="1,3,5")
    args = parser.parse_args()

    top_percentiles = tuple(float(x) for x in args.hard_boundary_top_percentiles.split(",") if x)
    band_radii = tuple(int(x) for x in args.hard_boundary_band_radii.split(",") if x)
    f1_tolerances = tuple(int(x) for x in args.hard_boundary_f1_tolerances.split(",") if x)

    config_kwargs = dict(
        config_version=args.config_version if args.config_version else None,
        midas_model_type=args.midas_model_type,
        img_size=args.img_size,
        data_path_eval=args.data_path_eval,
        gt_path_eval=args.gt_path_eval,
        **({"filenames_file_eval": args.filenames_file_eval} if args.filenames_file_eval else {}),
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval,
        garg_crop=bool(args.garg_crop),
        eigen_crop=bool(args.eigen_crop),
        use_boundary_refine=bool(args.use_boundary_refine),
        boundary_refine_channels=args.boundary_refine_channels,
        boundary_refine_scale=args.boundary_refine_scale,
        boundary_refine_mode=args.boundary_refine_mode,
        boundary_refine_use_da_prior=bool(args.boundary_refine_use_da_prior),
        use_discontinuity_branch=bool(args.use_discontinuity_branch),
        discontinuity_channels=args.discontinuity_channels,
        use_discontinuity_temperature=bool(args.use_discontinuity_temperature),
        discontinuity_temperature_scale=args.discontinuity_temperature_scale,
        use_frozen_da_prior=bool(args.use_frozen_da_prior),
        frozen_da_model=args.frozen_da_model,
        frozen_da_feature_channels=args.frozen_da_feature_channels,
        frozen_da_input_size=args.frozen_da_input_size,
        frozen_da_fusion_scale=args.frozen_da_fusion_scale,
        use_frozen_da_boundary_gate=bool(args.use_frozen_da_boundary_gate),
        frozen_da_min_gate=args.frozen_da_min_gate,
    )
    if config_kwargs["config_version"] is None:
        config_kwargs.pop("config_version")

    config = get_config(
        args.model,
        "eval",
        args.dataset,
        **config_kwargs,
    )
    config.pretrained_resource = None

    model = build_model(config)
    model = load_wts(model, args.checkpoint)
    model = model.cuda().eval()

    loader = DepthDataLoader(config, "online_eval").data
    regular_metrics = []
    boundary_metrics = []

    for sample in tqdm(loader, total=len(loader)):
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue

        image = sample["image"].cuda()
        depth = sample["depth"].cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())

        pred = infer(model, image, dataset=sample["dataset"][0], focal=focal)
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)

        regular_metrics.append(compute_metrics(depth, pred, config=config))

        pred_np = pred.squeeze().detach().cpu().numpy()
        gt_np = depth.squeeze().detach().cpu().numpy()
        valid = (gt_np > config.min_depth_eval) & (gt_np < config.max_depth_eval)

        if config.garg_crop or config.eigen_crop:
            h, w = gt_np.shape
            eval_mask = np.zeros_like(valid)
            if config.garg_crop:
                eval_mask[int(0.40810811 * h):int(0.99189189 * h),
                          int(0.03594771 * w):int(0.96405229 * w)] = True
            else:
                eval_mask[int(0.3324324 * h):int(0.91351351 * h),
                          int(0.0359477 * w):int(0.96405229 * w)] = True
            valid = valid & eval_mask

        boundary_metrics.append({
            **si_boundary_f1_masked(pred_np, gt_np, valid),
            **boundary_region_metrics(
                pred_np, gt_np, valid,
                log_grad_threshold=args.boundary_log_grad_threshold),
            **hard_boundary_metrics(
                pred_np, gt_np, valid,
                log_grad_threshold=args.boundary_log_grad_threshold,
                top_percentiles=top_percentiles,
                band_radii=band_radii,
                f1_tolerances=f1_tolerances),
        })

    result = {
        **average_dicts(regular_metrics),
        **average_dicts(boundary_metrics),
        "num_samples": len(regular_metrics),
        "checkpoint": args.checkpoint,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
