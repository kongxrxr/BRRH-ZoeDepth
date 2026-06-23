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
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    t = F.max_pool2d(t, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return t.squeeze().numpy() > 0


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


def make_panel(title, image):
    image = Image.fromarray(as_rgb(image)).convert("RGB")
    title_h = 26
    panel = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    panel.paste(image, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title, fill=(0, 0, 0))
    return panel


def save_montage(path, panels, columns=4):
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * w, rows * h), (245, 245, 245))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % columns) * w, (idx // columns) * h))
    canvas.save(path)


def build_eval_model(args, checkpoint, use_boundary_refine):
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
        use_boundary_refine=bool(use_boundary_refine),
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
    model = load_wts(model, checkpoint)
    return model.cuda().eval(), config


def band_absrel(pred, gt, band):
    if not band.any():
        return np.inf
    return float(np.mean(np.abs(pred[band] - gt[band]) / np.maximum(gt[band], 1e-6)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_checkpoint", required=True)
    parser.add_argument("--noresidual_checkpoint", required=True)
    parser.add_argument("--output_dir", default="figures/kitti_brrh_vs_noresidual")
    parser.add_argument("--scan_samples", type=int, default=120)
    parser.add_argument("--save_top", type=int, default=8)
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

    full_model, config = build_eval_model(args, args.full_checkpoint, use_boundary_refine=True)
    nores_model, _ = build_eval_model(args, args.noresidual_checkpoint, use_boundary_refine=False)
    loader = DepthDataLoader(config, "online_eval").data

    candidates = []
    for idx, sample in tqdm(enumerate(loader), total=min(len(loader), args.scan_samples)):
        if idx >= args.scan_samples:
            break
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue

        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())
        full_pred = infer(full_model, image, dataset=sample["dataset"][0], focal=focal)
        nores_pred = infer(nores_model, image, dataset=sample["dataset"][0], focal=focal)
        if full_pred.shape[-2:] != depth.shape[-2:]:
            full_pred = F.interpolate(full_pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            nores_pred = F.interpolate(nores_pred, depth.shape[-2:], mode="bilinear", align_corners=True)

        gt = depth.squeeze().detach().cpu().numpy()
        full_np = full_pred.squeeze().detach().cpu().numpy()
        nores_np = nores_pred.squeeze().detach().cpu().numpy()
        valid = (gt > config.min_depth_eval) & (gt < config.max_depth_eval)
        gt_edge = depth_boundary(gt, valid, args.boundary_log_grad_threshold)
        band = dilate(gt_edge, args.band_radius) & valid
        full_err = band_absrel(full_np, gt, band)
        nores_err = band_absrel(nores_np, gt, band)
        candidates.append((nores_err - full_err, idx, full_err, nores_err))

    candidates.sort(reverse=True)
    chosen = candidates[: args.save_top]
    chosen_indices = {idx for _, idx, _, _ in chosen}
    score_by_idx = {idx: (score, full_err, nores_err) for score, idx, full_err, nores_err in chosen}

    saved = 0
    loader = DepthDataLoader(config, "online_eval").data
    for idx, sample in tqdm(enumerate(loader), total=len(loader)):
        if idx not in chosen_indices:
            continue
        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())
        full_pred = infer(full_model, image, dataset=sample["dataset"][0], focal=focal)
        nores_pred = infer(nores_model, image, dataset=sample["dataset"][0], focal=focal)
        if full_pred.shape[-2:] != depth.shape[-2:]:
            full_pred = F.interpolate(full_pred, depth.shape[-2:], mode="bilinear", align_corners=True)
            nores_pred = F.interpolate(nores_pred, depth.shape[-2:], mode="bilinear", align_corners=True)

        rgb = to_rgb_image(image)
        gt = depth.squeeze().detach().cpu().numpy()
        full_np = full_pred.squeeze().detach().cpu().numpy()
        nores_np = nores_pred.squeeze().detach().cpu().numpy()
        if rgb.shape[:2] != gt.shape:
            rgb = np.array(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR))

        valid = (gt > config.min_depth_eval) & (gt < config.max_depth_eval)
        gt_edge = depth_boundary(gt, valid, args.boundary_log_grad_threshold)
        full_edge = depth_boundary(full_np, valid, args.boundary_log_grad_threshold)
        nores_edge = depth_boundary(nores_np, valid, args.boundary_log_grad_threshold)

        full_rel = np.zeros_like(gt)
        nores_rel = np.zeros_like(gt)
        full_rel[valid] = np.abs(full_np[valid] - gt[valid]) / np.maximum(gt[valid], 1e-6)
        nores_rel[valid] = np.abs(nores_np[valid] - gt[valid]) / np.maximum(gt[valid], 1e-6)
        gt_vis = gt.copy()
        full_vis = full_np.copy()
        nores_vis = nores_np.copy()
        gt_vis[~valid] = -99
        full_vis[~valid] = -99
        nores_vis[~valid] = -99
        max_depth = min(config.max_depth_eval, float(np.percentile(gt[valid], 95))) if valid.any() else 80.0
        score, full_err, nores_err = score_by_idx[idx]

        panels = [
            make_panel(f"RGB idx={idx}", rgb),
            make_panel("GT depth", colorize(gt_vis, 0, max_depth, cmap="magma_r")),
            make_panel(f"Full BRRH band={full_err:.3f}", colorize(full_vis, 0, max_depth, cmap="magma_r")),
            make_panel(f"No residual band={nores_err:.3f}", colorize(nores_vis, 0, max_depth, cmap="magma_r")),
            make_panel("Full rel error", colorize(full_rel, 0, 0.35, cmap="inferno")),
            make_panel("No residual rel error", colorize(nores_rel, 0, 0.35, cmap="inferno")),
            make_panel("GT boundary", edge_overlay(rgb, gt_edge, (0, 255, 80))),
            make_panel("Full pred boundary", edge_overlay(rgb, full_edge, (32, 128, 255))),
            make_panel("No residual boundary", edge_overlay(rgb, nores_edge, (255, 32, 32))),
            make_panel(f"Gain={score:.3f}", colorize(np.maximum(nores_rel - full_rel, 0), 0, 0.20, cmap="viridis")),
        ]
        save_montage(out_dir / f"{idx:06d}_brrh_vs_noresidual.png", panels, columns=5)
        saved += 1
        if saved >= args.save_top:
            break

    summary = out_dir / "selected_samples.txt"
    summary.write_text("\n".join(
        f"idx={idx} gain={score:.6f} full_band_absrel={full_err:.6f} nores_band_absrel={nores_err:.6f}"
        for score, idx, full_err, nores_err in chosen
    ) + "\n")
    print(f"Saved {saved} comparison visualizations to {out_dir}")


if __name__ == "__main__":
    main()
