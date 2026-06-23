#!/usr/bin/env python
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def as_rgb(array):
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def colorize(array, vmin, vmax, cmap_name):
    import matplotlib.cm as cm

    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array) & (array >= vmin)
    norm = np.clip((array - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    colored = (cm.get_cmap(cmap_name)(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 255
    return colored


def make_panel(title, image, title_h=26):
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


def edge_overlay(rgb, edge, color):
    out = rgb.copy()
    out[edge] = (0.35 * out[edge] + 0.65 * np.array(color)).astype(np.uint8)
    return out


def median_align(pred, gt, valid):
    pred_med = np.median(pred[valid])
    gt_med = np.median(gt[valid])
    if pred_med <= 1e-6 or gt_med <= 1e-6:
        return pred
    return pred * (gt_med / pred_med)


def band_absrel(pred, gt, band):
    if not band.any():
        return float("nan")
    return float(np.mean(np.abs(pred[band] - gt[band]) / np.maximum(gt[band], 1e-6)))


def rel_error(pred, gt, valid):
    error = np.zeros_like(gt, dtype=np.float32)
    error[valid] = np.abs(pred[valid] - gt[valid]) / np.maximum(gt[valid], 1e-6)
    return error


@torch.no_grad()
def predict_depthpro(args, image_path, depth_shape, gt, valid):
    sys.path.insert(0, str(Path(args.depthpro_timm_path).resolve()))
    sys.path.insert(0, args.depthpro_src)
    import depth_pro

    if not hasattr(predict_depthpro, "state"):
        config = dataclasses.replace(
            depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=args.depthpro_checkpoint,
        )
        model, transform = depth_pro.create_model_and_transforms(config=config, device=torch.device(args.device))
        predict_depthpro.state = (model.eval(), transform)

    model, transform = predict_depthpro.state
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = transform(image).to(args.device)
    prediction = model.infer(image_tensor, f_px=f_px)["depth"].detach().cpu().numpy().astype(np.float32)
    if prediction.shape != depth_shape:
        tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(tensor, size=depth_shape, mode="bilinear", align_corners=False).squeeze().numpy()
    prediction = np.clip(prediction, args.min_depth_eval, args.max_depth_eval)
    if args.align == "median":
        prediction = median_align(prediction, gt, valid)
        prediction = np.clip(prediction, args.min_depth_eval, args.max_depth_eval)
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="figures/nyu_brrh_depthpro_comparison")
    parser.add_argument("--depthpro_src", default="/home/kxr/ml-depth-pro/src")
    parser.add_argument("--depthpro_checkpoint", default="/home/kxr/ml-depth-pro/checkpoints/depth_pro.pt")
    parser.add_argument("--depthpro_timm_path", default=".deps/depthpro_timm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align", choices=["none", "median"], default="median")
    parser.add_argument("--min_depth_eval", type=float, default=0.001)
    parser.add_argument("--max_depth_eval", type=float, default=10.0)
    parser.add_argument("--error_max", type=float, default=0.30)
    parser.add_argument("--gain_max", type=float, default=0.12)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    selected_path = input_dir / "selected_samples.json"
    selected = json.loads(selected_path.read_text())
    all_paper_panels = []

    for row in selected:
        idx = int(row["idx"])
        data = np.load(input_dir / "records" / f"{idx:06d}.npz")
        rgb = data["rgb"]
        gt = data["gt"]
        valid = data["valid"].astype(bool)
        band = data["band"].astype(bool)
        gt_edge = data["gt_edge"].astype(bool)
        baseline = data["baseline"]
        brrh = data["brrh"]
        depthpro = predict_depthpro(args, row["image_path"], gt.shape, gt, valid)
        row["depthpro_band_absrel"] = band_absrel(depthpro, gt, band)

        max_depth = min(args.max_depth_eval, float(np.percentile(gt[valid], 95))) if valid.any() else args.max_depth_eval
        gt_vis = np.clip(gt.copy(), args.min_depth_eval, args.max_depth_eval)
        baseline_vis = np.clip(baseline.copy(), args.min_depth_eval, args.max_depth_eval)
        brrh_vis = np.clip(brrh.copy(), args.min_depth_eval, args.max_depth_eval)
        depthpro_vis = np.clip(depthpro.copy(), args.min_depth_eval, args.max_depth_eval)

        paper_panels = [
            make_panel("RGB", rgb),
            make_panel("GT", colorize(gt_vis, 0, max_depth, "magma_r")),
            make_panel("ZoeDepth", colorize(baseline_vis, 0, max_depth, "magma_r")),
            make_panel("BRRH-ZoeDepth", colorize(brrh_vis, 0, max_depth, "magma_r")),
            make_panel("Depth Pro", colorize(depthpro_vis, 0, max_depth, "magma_r")),
        ]
        all_paper_panels.extend(paper_panels)

        full_panels = paper_panels + [
            make_panel("ZoeDepth error", colorize(rel_error(baseline, gt, valid), 0, args.error_max, "inferno"), title_h=28),
            make_panel("BRRH error", colorize(rel_error(brrh, gt, valid), 0, args.error_max, "inferno"), title_h=28),
            make_panel("Depth Pro error", colorize(rel_error(depthpro, gt, valid), 0, args.error_max, "inferno"), title_h=28),
            make_panel("GT boundary", edge_overlay(rgb, gt_edge, (0, 255, 80)), title_h=28),
            make_panel("BRRH error reduction", colorize(np.maximum(rel_error(baseline, gt, valid) - rel_error(brrh, gt, valid), 0), 0, args.gain_max, "viridis"), title_h=28),
        ]
        save_montage(input_dir / f"{idx:06d}_nyu_brrh_depthpro.png", full_panels, columns=5)
        del depthpro
        torch.cuda.empty_cache()

    save_montage(input_dir / "nyu_brrh_depthpro_top_samples.png", all_paper_panels, columns=5)
    selected_path.write_text(json.dumps(selected, indent=2) + "\n")
    print(f"Rendered Depth Pro visual comparison for {len(selected)} samples in {input_dir}")


if __name__ == "__main__":
    main()
