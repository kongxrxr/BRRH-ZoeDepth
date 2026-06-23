#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate import infer
from scripts.evaluate_hf_depth_kitti_boundary import (
    _compute_metrics,
    average_dicts,
    boundary_region_metrics,
    si_boundary_f1_masked,
)
from scripts.evaluate_kitti_boundary import hard_boundary_metrics
from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.models.model_io import load_wts
from zoedepth.utils.config import get_config


def make_eval_mask(gt_depth, min_depth, max_depth, eigen_crop=True):
    valid = (gt_depth > min_depth) & (gt_depth < max_depth)
    if eigen_crop:
        h, w = gt_depth.shape
        crop = np.zeros_like(valid)
        crop[
            int(0.3324324 * h):int(0.91351351 * h),
            int(0.0359477 * w):int(0.96405229 * w),
        ] = True
        valid &= crop
    return valid


def pixel_boundary_from_log_depth(depth, valid, threshold):
    depth = np.clip(depth, 1e-6, None)
    log_depth = np.log(depth)
    edge = np.zeros_like(valid, dtype=bool)

    dx = np.abs(log_depth[:, 1:] - log_depth[:, :-1])
    valid_x = valid[:, 1:] & valid[:, :-1]
    edge_x = valid_x & (dx > threshold)
    edge[:, 1:] |= edge_x
    edge[:, :-1] |= edge_x

    dy = np.abs(log_depth[1:, :] - log_depth[:-1, :])
    valid_y = valid[1:, :] & valid[:-1, :]
    edge_y = valid_y & (dy > threshold)
    edge[1:, :] |= edge_y
    edge[:-1, :] |= edge_y
    return edge & valid


def build_model_from_args(args, variant):
    if variant == "baseline":
        checkpoint = args.baseline_checkpoint
        options = {
            "use_boundary_refine": False,
            "use_discontinuity_branch": False,
            "use_discontinuity_temperature": False,
            "use_frozen_da_prior": False,
        }
    elif variant == "brrh":
        checkpoint = args.brrh_checkpoint
        options = {
            "use_boundary_refine": True,
            "boundary_refine_channels": args.boundary_refine_channels,
            "boundary_refine_scale": args.boundary_refine_scale,
            "boundary_refine_mode": args.boundary_refine_mode,
            "boundary_refine_use_da_prior": bool(args.boundary_refine_use_da_prior),
            "use_discontinuity_branch": True,
            "discontinuity_channels": args.discontinuity_channels,
            "use_discontinuity_temperature": True,
            "discontinuity_temperature_scale": args.discontinuity_temperature_scale,
            "use_frozen_da_prior": True,
            "frozen_da_model": args.frozen_da_model,
            "frozen_da_feature_channels": args.frozen_da_feature_channels,
            "frozen_da_input_size": args.frozen_da_input_size,
            "frozen_da_fusion_scale": args.frozen_da_fusion_scale,
            "use_frozen_da_boundary_gate": False,
            "frozen_da_min_gate": args.frozen_da_min_gate,
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")

    config = get_config(
        args.model,
        "eval",
        args.dataset,
        config_version=args.config_version,
        midas_model_type=args.midas_model_type,
        img_size=args.img_size,
        data_path_eval=args.data_path_eval,
        gt_path_eval=args.gt_path_eval,
        filenames_file_eval=args.filenames_file_eval,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval,
        garg_crop=False,
        eigen_crop=bool(args.eigen_crop),
        **options,
    )
    config.pretrained_resource = None
    model = build_model(config)
    model = load_wts(model, checkpoint)
    return model.cuda().eval(), config


def infer_variant(model, sample, depth_shape):
    image = sample["image"].cuda()
    focal = sample.get("focal", torch.Tensor([518.8579]).cuda())
    pred = infer(model, image, dataset=sample["dataset"][0], focal=focal)
    if pred.shape[-2:] != depth_shape:
        pred = F.interpolate(pred, depth_shape, mode="bilinear", align_corners=True)
    return pred.squeeze().detach().cpu().numpy().astype(np.float32)


def metrics_for_prediction(pred, gt, valid, args):
    pred = np.clip(pred, args.min_depth_eval, args.max_depth_eval)
    return {
        **_compute_metrics(gt, pred, valid),
        **si_boundary_f1_masked(pred, gt, valid),
        **boundary_region_metrics(
            pred,
            gt,
            valid,
            log_grad_threshold=args.boundary_log_grad_threshold,
        ),
        **hard_boundary_metrics(
            pred,
            gt,
            valid,
            log_grad_threshold=args.boundary_log_grad_threshold,
            top_percentiles=(5.0, 10.0),
            band_radii=(3, 5),
            f1_tolerances=(1, 3, 5),
        ),
    }


def summarize_by_density(per_sample, model_name):
    entries = [
        {
            **sample[model_name],
            "idx": sample["idx"],
            "image_path": sample["image_path"],
            "boundary_density": sample["boundary_density"],
        }
        for sample in per_sample
    ]
    densities = np.array([entry["boundary_density"] for entry in entries], dtype=np.float32)
    q1, q2 = np.quantile(densities, [1.0 / 3.0, 2.0 / 3.0])

    groups = {
        "low": [entry for entry in entries if entry["boundary_density"] <= q1],
        "medium": [entry for entry in entries if q1 < entry["boundary_density"] <= q2],
        "high": [entry for entry in entries if entry["boundary_density"] > q2],
    }
    result = {}
    for name, group in groups.items():
        metric_group = [
            {
                key: value
                for key, value in entry.items()
                if key != "idx" and isinstance(value, (int, float, np.floating))
            }
            for entry in group
        ]
        result[name] = {
            **average_dicts(metric_group),
            "num_samples": len(group),
            "density_min": float(min(entry["boundary_density"] for entry in group)),
            "density_max": float(max(entry["boundary_density"] for entry in group)),
            "density_mean": float(np.mean([entry["boundary_density"] for entry in group])),
        }
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", default="/home/kxr/zoedepth_nyu_sync_baseline_resume_approx3ep_checkpoints/ZoeDepthv1_19-Jun_20-58-018db8113b87_latest.pt")
    parser.add_argument("--brrh_checkpoint", default="/home/kxr/zoedepth_nyu_sync_brrh_tuned_boundary_256x512_bs4_workers4_2ep_checkpoints/ZoeDepthv1_19-Jun_23-09-0c3cdbbc8dcc_latest.pt")
    parser.add_argument("--output", default="logs/nyu_boundary_density_subsets_baseline_brrh.json")
    parser.add_argument("--per_sample_output", default="logs/nyu_boundary_density_subsets_per_sample.json")
    parser.add_argument("--model", default="zoedepth")
    parser.add_argument("--dataset", default="nyu")
    parser.add_argument("--config_version", default="nyu")
    parser.add_argument("--midas_model_type", default="DPT_BEiT_L_384")
    parser.add_argument("--img_size", default="256,512")
    parser.add_argument("--data_path_eval", default="/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all")
    parser.add_argument("--gt_path_eval", default="/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all")
    parser.add_argument("--filenames_file_eval", default="./train_test_inputs/nyudepthv2_labeled_val654_files_with_gt.txt")
    parser.add_argument("--min_depth_eval", type=float, default=0.001)
    parser.add_argument("--max_depth_eval", type=float, default=10.0)
    parser.add_argument("--eigen_crop", type=int, default=1)
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--boundary_refine_channels", type=int, default=32)
    parser.add_argument("--boundary_refine_scale", type=float, default=0.015)
    parser.add_argument("--boundary_refine_mode", default="log_residual")
    parser.add_argument("--boundary_refine_use_da_prior", type=int, default=1)
    parser.add_argument("--discontinuity_channels", type=int, default=32)
    parser.add_argument("--discontinuity_temperature_scale", type=float, default=0.5)
    parser.add_argument("--frozen_da_model", default="/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6")
    parser.add_argument("--frozen_da_feature_channels", type=int, default=8)
    parser.add_argument("--frozen_da_input_size", type=int, default=384)
    parser.add_argument("--frozen_da_fusion_scale", type=float, default=0.08)
    parser.add_argument("--frozen_da_min_gate", type=float, default=0.05)
    args = parser.parse_args()

    baseline_model, config = build_model_from_args(args, "baseline")
    brrh_model, _ = build_model_from_args(args, "brrh")
    loader = DepthDataLoader(config, "online_eval").data
    total = len(loader) if args.max_samples <= 0 else min(args.max_samples, len(loader))

    per_sample = []
    for idx, sample in tqdm(enumerate(loader), total=total):
        if idx >= total:
            break
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue

        gt = sample["depth"].squeeze().numpy().astype(np.float32)
        valid = make_eval_mask(
            gt,
            args.min_depth_eval,
            args.max_depth_eval,
            eigen_crop=bool(args.eigen_crop),
        )
        gt_edge = pixel_boundary_from_log_depth(gt, valid, args.boundary_log_grad_threshold)
        boundary_density = float(np.count_nonzero(gt_edge) / max(np.count_nonzero(valid), 1))

        baseline_pred = infer_variant(baseline_model, sample, gt.shape)
        brrh_pred = infer_variant(brrh_model, sample, gt.shape)

        per_sample.append({
            "idx": idx,
            "image_path": sample.get("image_path", [""])[0],
            "depth_path": sample.get("depth_path", [""])[0],
            "boundary_density": boundary_density,
            "baseline": metrics_for_prediction(baseline_pred, gt, valid, args),
            "brrh": metrics_for_prediction(brrh_pred, gt, valid, args),
        })

    summary = {
        "num_samples": len(per_sample),
        "split_basis": "GT log-depth boundary pixel fraction, tertiles",
        "boundary_log_grad_threshold": args.boundary_log_grad_threshold,
        "baseline_checkpoint": args.baseline_checkpoint,
        "brrh_checkpoint": args.brrh_checkpoint,
        "baseline": summarize_by_density(per_sample, "baseline"),
        "brrh": summarize_by_density(per_sample, "brrh"),
    }

    improvements = {}
    for group in ("low", "medium", "high"):
        base = summary["baseline"][group]
        brrh = summary["brrh"][group]
        improvements[group] = {
            "abs_rel_delta": brrh["abs_rel"] - base["abs_rel"],
            "rmse_delta": brrh["rmse"] - base["rmse"],
            "boundary_rmse_delta": brrh["boundary_rmse"] - base["boundary_rmse"],
            "band3_abs_rel_delta": brrh["band3_abs_rel"] - base["band3_abs_rel"],
            "edge_f1_tol3_delta": brrh["edge_f1_tol3"] - base["edge_f1_tol3"],
            "edge_f1_tol5_delta": brrh["edge_f1_tol5"] - base["edge_f1_tol5"],
        }
    summary["brrh_minus_baseline"] = improvements

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    Path(args.per_sample_output).write_text(json.dumps(per_sample, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
