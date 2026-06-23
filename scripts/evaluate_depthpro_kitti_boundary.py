#!/usr/bin/env python
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import depth_pro
from evaluate_hf_depth_kitti_boundary import (
    _compute_metrics,
    _iter_kitti_eval_samples,
    _kitti_benchmark_crop,
    _load_kitti_depth,
    _make_eval_mask,
    _median_align,
    average_dicts,
    boundary_region_metrics,
    si_boundary_f1_masked,
)


def _load_rgb_for_depthpro(image_path):
    try:
        image, _, f_px = depth_pro.load_rgb(image_path)
    except Exception:
        image = Image.open(image_path).convert("RGB")
        f_px = None
    return image, f_px


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/kxr/ml-depth-pro/checkpoints/depth_pro.pt")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--align", choices=["none", "median"], default="none")
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

    config = dataclasses.replace(
        depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=args.checkpoint,
    )
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model = model.eval()

    samples = list(_iter_kitti_eval_samples(args))
    total = len(samples) if args.max_samples <= 0 else min(args.max_samples, len(samples))
    regular_metrics = []
    boundary_metrics = []

    for image_path, depth_path in tqdm(samples[:total], total=total):
        image, f_px = _load_rgb_for_depthpro(image_path)
        depth = _load_kitti_depth(depth_path)
        if args.do_kb_crop:
            image, depth = _kitti_benchmark_crop(image, depth)
        image_tensor = transform(image).to(device)
        prediction = model.infer(image_tensor, f_px=f_px)
        pred_depth = prediction["depth"].detach().cpu().numpy().astype(np.float32)
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
        })

    result = {
        **average_dicts(regular_metrics),
        **average_dicts(boundary_metrics),
        "num_samples": len(regular_metrics),
        "checkpoint": args.checkpoint,
        "align": args.align,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
