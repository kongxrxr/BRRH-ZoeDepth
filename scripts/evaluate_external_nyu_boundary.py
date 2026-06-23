#!/usr/bin/env python
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from evaluate_hf_depth_kitti_boundary import (
    _compute_metrics,
    _median_align,
    average_dicts,
    boundary_region_metrics,
    si_boundary_f1_masked,
)
from evaluate_kitti_boundary import hard_boundary_metrics


def _remove_leading_slash(path):
    return path[1:] if path.startswith("/") else path


def _load_nyu_depth(path):
    return np.asarray(Image.open(path), dtype=np.float32) / 1000.0


def _iter_nyu_eval_samples(args):
    with Path(args.filenames_file_eval).open("r") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            image_path = Path(args.data_path_eval) / _remove_leading_slash(fields[0])
            depth_path = Path(args.gt_path_eval) / _remove_leading_slash(fields[1])
            yield image_path, depth_path


def _make_eval_mask(gt_depth, args):
    valid = (gt_depth > args.min_depth_eval) & (gt_depth < args.max_depth_eval)
    if args.eigen_crop:
        h, w = gt_depth.shape
        crop = np.zeros_like(valid)
        crop[
            int(0.3324324 * h):int(0.91351351 * h),
            int(0.0359477 * w):int(0.96405229 * w),
        ] = True
        valid &= crop
    return valid


def _prediction_to_depth(prediction, mode, eps=1e-6):
    prediction = np.asarray(prediction, dtype=np.float32)
    if mode == "inverse":
        return 1.0 / np.clip(prediction, eps, None)
    return prediction


@torch.no_grad()
def _predict_hf(args, device, image, depth_shape):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if not hasattr(_predict_hf, "state"):
        processor = AutoImageProcessor.from_pretrained(args.hf_model, local_files_only=args.local_files_only)
        model = AutoModelForDepthEstimation.from_pretrained(args.hf_model, local_files_only=args.local_files_only)
        _predict_hf.state = (processor, model.to(device).eval())

    processor, model = _predict_hf.state
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prediction = model(**inputs).predicted_depth
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=depth_shape,
        mode="bicubic",
        align_corners=False,
    ).squeeze().detach().cpu().numpy().astype(np.float32)
    return _prediction_to_depth(prediction, args.prediction_mode)


@torch.no_grad()
def _predict_depthpro(args, device, image_path, image, depth_shape):
    sys.path.insert(0, args.depthpro_src)
    import depth_pro

    if not hasattr(_predict_depthpro, "state"):
        config = dataclasses.replace(
            depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=args.depthpro_checkpoint,
        )
        model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
        _predict_depthpro.state = (model.eval(), transform)

    model, transform = _predict_depthpro.state
    try:
        dp_image, _, f_px = depth_pro.load_rgb(image_path)
    except Exception:
        dp_image, f_px = image, None
    image_tensor = transform(dp_image).to(device)
    prediction = model.infer(image_tensor, f_px=f_px)["depth"].detach().cpu().numpy().astype(np.float32)
    if prediction.shape != depth_shape:
        tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(tensor, size=depth_shape, mode="bilinear", align_corners=False).squeeze().numpy()
    return prediction


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["hf", "depthpro"], required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--hf_model", default="/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6")
    parser.add_argument("--prediction_mode", choices=["depth", "inverse"], default="depth")
    parser.add_argument("--local_files_only", type=int, default=1)
    parser.add_argument("--depthpro_src", default="/home/kxr/ml-depth-pro/src")
    parser.add_argument("--depthpro_checkpoint", default="/home/kxr/ml-depth-pro/checkpoints/depth_pro.pt")
    parser.add_argument("--align", choices=["none", "median"], default="median")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--filenames_file_eval", default="./train_test_inputs/nyudepthv2_labeled_val654_files_with_gt.txt")
    parser.add_argument("--data_path_eval", default="/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all")
    parser.add_argument("--gt_path_eval", default="/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all")
    parser.add_argument("--min_depth_eval", type=float, default=0.001)
    parser.add_argument("--max_depth_eval", type=float, default=10.0)
    parser.add_argument("--eigen_crop", type=int, default=1)
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    parser.add_argument("--hard_boundary_top_percentiles", default="5,10")
    parser.add_argument("--hard_boundary_band_radii", default="3,5")
    parser.add_argument("--hard_boundary_f1_tolerances", default="1,3,5")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    top_percentiles = tuple(float(x) for x in args.hard_boundary_top_percentiles.split(",") if x)
    band_radii = tuple(int(x) for x in args.hard_boundary_band_radii.split(",") if x)
    f1_tolerances = tuple(int(x) for x in args.hard_boundary_f1_tolerances.split(",") if x)

    samples = list(_iter_nyu_eval_samples(args))
    total = len(samples) if args.max_samples <= 0 else min(args.max_samples, len(samples))
    regular_metrics = []
    boundary_metrics = []

    for image_path, depth_path in tqdm(samples[:total], total=total):
        image = Image.open(image_path).convert("RGB")
        depth = _load_nyu_depth(depth_path)
        if args.provider == "hf":
            pred_depth = _predict_hf(args, device, image, depth.shape)
        else:
            pred_depth = _predict_depthpro(args, device, image_path, image, depth.shape)

        valid = _make_eval_mask(depth, args)
        pred_depth = np.clip(pred_depth, args.min_depth_eval, args.max_depth_eval)
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
            **hard_boundary_metrics(
                pred_depth,
                depth,
                valid,
                log_grad_threshold=args.boundary_log_grad_threshold,
                top_percentiles=top_percentiles,
                band_radii=band_radii,
                f1_tolerances=f1_tolerances,
            ),
        })

    result = {
        **average_dicts(regular_metrics),
        **average_dicts(boundary_metrics),
        "num_samples": len(regular_metrics),
        "provider": args.provider,
        "align": args.align,
        "hf_model": args.hf_model if args.provider == "hf" else "",
        "depthpro_checkpoint": args.depthpro_checkpoint if args.provider == "depthpro" else "",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
