#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from evaluate import infer
from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.models.model_io import load_wts
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize


def depth_boundary(depth, valid, threshold):
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
    return edge


def dilate(mask, radius):
    if radius <= 0:
        return mask
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    tensor = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return tensor.squeeze().numpy() > 0


def to_rgb_image(image_tensor):
    image = image_tensor.squeeze(0).detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    if image.max() > 2:
        image = image / 255.0
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def as_rgb(array):
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def edge_overlay(rgb, edge, color):
    out = rgb.copy()
    out[edge] = (0.35 * out[edge] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def make_panel(title, image, title_h=30):
    image = Image.fromarray(as_rgb(image)).convert("RGB")
    panel = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    panel.paste(image, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((7, 8), title, fill=(0, 0, 0))
    return panel


def save_montage(path, panels, columns):
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), (245, 245, 245))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % columns) * width, (idx // columns) * height))
    canvas.save(path)


def build_baseline_model(args):
    config = get_config(
        args.model,
        "eval",
        args.dataset,
        config_version=args.config_version,
        midas_model_type=args.midas_model_type,
        img_size=args.img_size,
        data_path_eval=args.data_path_eval,
        gt_path_eval=args.gt_path_eval,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval,
        garg_crop=bool(args.garg_crop),
        eigen_crop=bool(args.eigen_crop),
        use_boundary_refine=False,
        use_discontinuity_branch=False,
        use_discontinuity_temperature=False,
        use_frozen_da_prior=False,
    )
    config.pretrained_resource = None
    model = build_model(config)
    model = load_wts(model, args.baseline_checkpoint)
    return model.cuda().eval(), config


def build_brrh_model(args):
    config = get_config(
        args.model,
        "eval",
        args.dataset,
        config_version=args.config_version,
        midas_model_type=args.midas_model_type,
        img_size=args.img_size,
        data_path_eval=args.data_path_eval,
        gt_path_eval=args.gt_path_eval,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval,
        garg_crop=bool(args.garg_crop),
        eigen_crop=bool(args.eigen_crop),
        use_boundary_refine=True,
        boundary_refine_channels=args.boundary_refine_channels,
        boundary_refine_scale=args.boundary_refine_scale,
        boundary_refine_mode=args.boundary_refine_mode,
        boundary_refine_use_da_prior=bool(args.boundary_refine_use_da_prior),
        use_discontinuity_branch=True,
        discontinuity_channels=args.discontinuity_channels,
        use_discontinuity_temperature=True,
        discontinuity_temperature_scale=args.discontinuity_temperature_scale,
        use_frozen_da_prior=True,
        frozen_da_model=args.frozen_da_model,
        frozen_da_feature_channels=args.frozen_da_feature_channels,
        frozen_da_input_size=args.frozen_da_input_size,
        frozen_da_fusion_scale=args.frozen_da_fusion_scale,
        use_frozen_da_boundary_gate=False,
        frozen_da_min_gate=args.frozen_da_min_gate,
    )
    config.pretrained_resource = None
    model = build_model(config)
    model = load_wts(model, args.brrh_checkpoint)
    return model.cuda().eval(), config


def band_absrel(pred, gt, band):
    if not band.any():
        return np.inf
    return float(np.mean(np.abs(pred[band] - gt[band]) / np.maximum(gt[band], 1e-6)))


def rel_error(pred, gt, valid):
    error = np.zeros_like(gt)
    error[valid] = np.abs(pred[valid] - gt[valid]) / np.maximum(gt[valid], 1e-6)
    return error


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", default="/home/kxr/zoedepth_kitti_checkpoints/ZoeDepthv1_29-May_09-33-5c1751bcdbf0_best.pt")
    parser.add_argument("--brrh_checkpoint", default="/home/kxr/zoedepth_kitti_brrh_scale0p24_256x512_bs4_workers8_5ep_checkpoints/ZoeDepthv1_11-Jun_16-01-d8388385a97b_latest.pt")
    parser.add_argument("--output_dir", default="figures/kitti_brrh_vs_baseline")
    parser.add_argument("--scan_samples", type=int, default=180)
    parser.add_argument("--save_top", type=int, default=10)
    parser.add_argument("--model", default="zoedepth")
    parser.add_argument("--dataset", default="kitti")
    parser.add_argument("--config_version", default="kitti")
    parser.add_argument("--midas_model_type", default="DPT_BEiT_L_384")
    parser.add_argument("--img_size", default="256,512")
    parser.add_argument("--data_path_eval", default="/home/kxr/shortcuts/datasets/kitti/raw")
    parser.add_argument("--gt_path_eval", default="/home/kxr/shortcuts/datasets/kitti/gts")
    parser.add_argument("--min_depth_eval", type=float, default=0.001)
    parser.add_argument("--max_depth_eval", type=float, default=80.0)
    parser.add_argument("--garg_crop", type=int, default=1)
    parser.add_argument("--eigen_crop", type=int, default=0)
    parser.add_argument("--boundary_refine_channels", type=int, default=32)
    parser.add_argument("--boundary_refine_scale", type=float, default=0.08)
    parser.add_argument("--boundary_refine_mode", default="log_residual")
    parser.add_argument("--boundary_refine_use_da_prior", type=int, default=1)
    parser.add_argument("--discontinuity_channels", type=int, default=32)
    parser.add_argument("--discontinuity_temperature_scale", type=float, default=0.5)
    parser.add_argument("--frozen_da_model", default="/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6")
    parser.add_argument("--frozen_da_feature_channels", type=int, default=8)
    parser.add_argument("--frozen_da_input_size", type=int, default=384)
    parser.add_argument("--frozen_da_fusion_scale", type=float, default=0.12)
    parser.add_argument("--frozen_da_min_gate", type=float, default=0.05)
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    parser.add_argument("--band_radius", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, config = build_baseline_model(args)
    brrh_model, _ = build_brrh_model(args)
    loader = DepthDataLoader(config, "online_eval").data

    candidates = []
    for idx, sample in tqdm(enumerate(loader), total=min(len(loader), args.scan_samples), desc="scan"):
        if idx >= args.scan_samples:
            break
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue

        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())

        baseline_pred = infer(baseline_model, image, dataset=sample["dataset"][0], focal=focal)
        brrh_pred = infer(brrh_model, image, dataset=sample["dataset"][0], focal=focal)
        if baseline_pred.shape[-2:] != depth.shape[-2:]:
            baseline_pred = F.interpolate(baseline_pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            brrh_pred = F.interpolate(brrh_pred, depth.shape[-2:], mode="bilinear", align_corners=True)

        gt = depth.squeeze().detach().cpu().numpy()
        baseline_np = baseline_pred.squeeze().detach().cpu().numpy()
        brrh_np = brrh_pred.squeeze().detach().cpu().numpy()
        valid = (gt > config.min_depth_eval) & (gt < config.max_depth_eval)
        gt_edge = depth_boundary(gt, valid, args.boundary_log_grad_threshold)
        band = dilate(gt_edge, args.band_radius) & valid

        baseline_band = band_absrel(baseline_np, gt, band)
        brrh_band = band_absrel(brrh_np, gt, band)
        candidates.append((baseline_band - brrh_band, idx, baseline_band, brrh_band))

    candidates.sort(reverse=True)
    chosen = candidates[: args.save_top]
    chosen_indices = {idx for _, idx, _, _ in chosen}
    score_by_idx = {idx: (score, baseline_band, brrh_band) for score, idx, baseline_band, brrh_band in chosen}

    loader = DepthDataLoader(config, "online_eval").data
    saved = 0
    for idx, sample in tqdm(enumerate(loader), total=len(loader), desc="save"):
        if idx not in chosen_indices:
            continue

        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())

        baseline_pred = infer(baseline_model, image, dataset=sample["dataset"][0], focal=focal)
        brrh_pred = infer(brrh_model, image, dataset=sample["dataset"][0], focal=focal)
        if baseline_pred.shape[-2:] != depth.shape[-2:]:
            baseline_pred = F.interpolate(baseline_pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            brrh_pred = F.interpolate(brrh_pred, depth.shape[-2:], mode="bilinear", align_corners=True)

        rgb = to_rgb_image(image)
        gt = depth.squeeze().detach().cpu().numpy()
        baseline_np = baseline_pred.squeeze().detach().cpu().numpy()
        brrh_np = brrh_pred.squeeze().detach().cpu().numpy()
        if rgb.shape[:2] != gt.shape:
            rgb = np.array(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR))

        valid = (gt > config.min_depth_eval) & (gt < config.max_depth_eval)
        gt_edge = depth_boundary(gt, valid, args.boundary_log_grad_threshold)
        baseline_edge = depth_boundary(baseline_np, valid, args.boundary_log_grad_threshold)
        brrh_edge = depth_boundary(brrh_np, valid, args.boundary_log_grad_threshold)

        gt_vis = gt.copy()
        baseline_vis = baseline_np.copy()
        brrh_vis = brrh_np.copy()
        gt_vis[~valid] = -99
        baseline_vis[~valid] = -99
        brrh_vis[~valid] = -99

        baseline_rel = rel_error(baseline_np, gt, valid)
        brrh_rel = rel_error(brrh_np, gt, valid)
        gain_map = np.maximum(baseline_rel - brrh_rel, 0.0)
        max_depth = min(config.max_depth_eval, float(np.percentile(gt[valid], 95))) if valid.any() else 80.0
        score, baseline_band, brrh_band = score_by_idx[idx]

        panels = [
            make_panel(f"RGB idx={idx}", rgb),
            make_panel("GT depth", colorize(gt_vis, 0, max_depth, cmap="magma_r")),
            make_panel(f"ZoeDepth band={baseline_band:.3f}", colorize(baseline_vis, 0, max_depth, cmap="magma_r")),
            make_panel(f"BRRH band={brrh_band:.3f}", colorize(brrh_vis, 0, max_depth, cmap="magma_r")),
            make_panel("ZoeDepth rel error", colorize(baseline_rel, 0, 0.35, cmap="inferno")),
            make_panel("BRRH rel error", colorize(brrh_rel, 0, 0.35, cmap="inferno")),
            make_panel("BRRH error reduction", colorize(gain_map, 0, 0.20, cmap="viridis")),
            make_panel(f"GT boundary gain={score:.3f}", edge_overlay(rgb, gt_edge, (0, 255, 80))),
            make_panel("ZoeDepth boundary", edge_overlay(rgb, baseline_edge, (255, 32, 32))),
            make_panel("BRRH boundary", edge_overlay(rgb, brrh_edge, (32, 128, 255))),
        ]
        save_montage(out_dir / f"{idx:06d}_baseline_vs_brrh.png", panels, columns=5)

        # Compact one-row version for paper layouts.
        paper_panels = [
            make_panel("RGB", rgb, title_h=26),
            make_panel("GT", colorize(gt_vis, 0, max_depth, cmap="magma_r"), title_h=26),
            make_panel("ZoeDepth", colorize(baseline_vis, 0, max_depth, cmap="magma_r"), title_h=26),
            make_panel("BRRH-ZoeDepth", colorize(brrh_vis, 0, max_depth, cmap="magma_r"), title_h=26),
            make_panel("ZoeDepth error", colorize(baseline_rel, 0, 0.35, cmap="inferno"), title_h=26),
            make_panel("BRRH error", colorize(brrh_rel, 0, 0.35, cmap="inferno"), title_h=26),
        ]
        save_montage(out_dir / f"{idx:06d}_paper_row.png", paper_panels, columns=6)
        Image.fromarray(edge_overlay(rgb, gt_edge, (0, 255, 80))).save(out_dir / f"{idx:06d}_gt_boundary.png")
        Image.fromarray(edge_overlay(rgb, baseline_edge, (255, 32, 32))).save(out_dir / f"{idx:06d}_baseline_boundary.png")
        Image.fromarray(edge_overlay(rgb, brrh_edge, (32, 128, 255))).save(out_dir / f"{idx:06d}_brrh_boundary.png")

        saved += 1
        if saved >= args.save_top:
            break

    (out_dir / "selected_samples.txt").write_text(
        "\n".join(
            f"idx={idx} gain={score:.6f} baseline_band_absrel={baseline_band:.6f} brrh_band_absrel={brrh_band:.6f}"
            for score, idx, baseline_band, brrh_band in chosen
        ) + "\n"
    )
    print(f"Saved {saved} KITTI BRRH-vs-baseline visualizations to {out_dir}")


if __name__ == "__main__":
    main()
