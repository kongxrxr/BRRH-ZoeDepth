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


def to_rgb_image(image_tensor):
    image = image_tensor.squeeze(0).detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    if image.max() > 2:
        image = image / 255.0
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def as_rgb(array):
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def edge_overlay(rgb, edge, color=(255, 32, 32)):
    out = rgb.copy()
    out[edge] = (0.35 * out[edge] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def make_panel(title, image):
    image = Image.fromarray(as_rgb(image)).convert("RGB")
    title_h = 24
    panel = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    panel.paste(image, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(0, 0, 0))
    return panel


def save_montage(path, panels, columns=3):
    w = max(panel.width for panel in panels)
    h = max(panel.height for panel in panels)
    rows = int(np.ceil(len(panels) / columns))
    canvas = Image.new("RGB", (columns * w, rows * h), (245, 245, 245))
    for idx, panel in enumerate(panels):
        x = (idx % columns) * w
        y = (idx // columns) * h
        canvas.paste(panel, (x, y))
    canvas.save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/kitti_boundary_vis")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--start_index", type=int, default=0)
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
    parser.add_argument("--use_boundary_refine", type=int, default=1)
    parser.add_argument("--boundary_refine_channels", type=int, default=32)
    parser.add_argument("--boundary_refine_scale", type=float, default=0.1)
    parser.add_argument("--use_discontinuity_branch", type=int, default=1)
    parser.add_argument("--discontinuity_channels", type=int, default=32)
    parser.add_argument("--use_discontinuity_temperature", type=int, default=1)
    parser.add_argument("--discontinuity_temperature_scale", type=float, default=1.5)
    parser.add_argument("--use_frozen_da_prior", type=int, default=0)
    parser.add_argument("--frozen_da_model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--frozen_da_feature_channels", type=int, default=16)
    parser.add_argument("--frozen_da_input_size", type=int, default=384)
    parser.add_argument("--frozen_da_fusion_scale", type=float, default=0.1)
    parser.add_argument("--use_frozen_da_boundary_gate", type=int, default=0)
    parser.add_argument("--frozen_da_min_gate", type=float, default=0.05)
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        use_boundary_refine=bool(args.use_boundary_refine),
        boundary_refine_channels=args.boundary_refine_channels,
        boundary_refine_scale=args.boundary_refine_scale,
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
    config.pretrained_resource = None

    model = build_model(config)
    model = load_wts(model, args.checkpoint)
    model = model.cuda().eval()

    loader = DepthDataLoader(config, "online_eval").data
    saved = 0
    for idx, sample in tqdm(enumerate(loader), total=len(loader)):
        if idx < args.start_index:
            continue
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue

        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([715.0873]).cuda())
        pred = infer(model, image, dataset=sample["dataset"][0], focal=focal)
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, depth.shape[-2:], mode="bilinear", align_corners=True)

        rgb = to_rgb_image(image)
        if rgb.shape[:2] != depth.shape[-2:]:
            rgb = np.array(Image.fromarray(rgb).resize((depth.shape[-1], depth.shape[-2]), Image.BILINEAR))

        gt_np = depth.squeeze().detach().cpu().numpy()
        pred_np = pred.squeeze().detach().cpu().numpy()
        valid = (gt_np > config.min_depth_eval) & (gt_np < config.max_depth_eval)

        gt_for_vis = gt_np.copy()
        pred_for_vis = pred_np.copy()
        gt_for_vis[~valid] = -99
        pred_for_vis[~valid] = -99

        gt_edge = depth_boundary(gt_np, valid, args.boundary_log_grad_threshold)
        pred_edge = depth_boundary(pred_np, valid, args.boundary_log_grad_threshold)
        rel_error = np.zeros_like(gt_np)
        rel_error[valid] = np.abs(pred_np[valid] - gt_np[valid]) / np.maximum(gt_np[valid], 1e-6)

        max_depth = min(config.max_depth_eval, float(np.percentile(gt_np[valid], 95))) if valid.any() else 80.0
        panels = [
            make_panel("RGB", rgb),
            make_panel("GT depth", colorize(gt_for_vis, 0, max_depth, cmap="magma_r")),
            make_panel("Pred depth", colorize(pred_for_vis, 0, max_depth, cmap="magma_r")),
            make_panel("Rel error", colorize(rel_error, 0, 0.35, cmap="inferno")),
            make_panel("GT boundary", edge_overlay(rgb, gt_edge, color=(0, 255, 80))),
            make_panel("Pred boundary", edge_overlay(rgb, pred_edge, color=(255, 32, 32))),
        ]

        stem = f"{idx:06d}"
        save_montage(output_dir / f"{stem}_montage.png", panels)
        Image.fromarray(edge_overlay(rgb, gt_edge, color=(0, 255, 80))).save(output_dir / f"{stem}_gt_boundary.png")
        Image.fromarray(edge_overlay(rgb, pred_edge, color=(255, 32, 32))).save(output_dir / f"{stem}_pred_boundary.png")

        saved += 1
        if saved >= args.num_samples:
            break

    print(f"Saved {saved} KITTI boundary visualizations to {output_dir}")


if __name__ == "__main__":
    main()
