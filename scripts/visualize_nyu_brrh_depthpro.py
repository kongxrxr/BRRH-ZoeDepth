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
from tqdm import tqdm

from evaluate import infer
from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.models.model_io import load_wts
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize


def remove_leading_slash(path):
    return path[1:] if path.startswith("/") else path


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


def make_eval_mask(gt_depth, args):
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


def make_panel(title, image, title_h=28):
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


def band_absrel(pred, gt, band):
    if not band.any():
        return np.inf
    return float(np.mean(np.abs(pred[band] - gt[band]) / np.maximum(gt[band], 1e-6)))


def rel_error(pred, gt, valid):
    error = np.zeros_like(gt, dtype=np.float32)
    error[valid] = np.abs(pred[valid] - gt[valid]) / np.maximum(gt[valid], 1e-6)
    return error


def median_align(pred, gt, valid):
    pred_med = np.median(pred[valid])
    gt_med = np.median(gt[valid])
    if pred_med <= 1e-6 or gt_med <= 1e-6:
        return pred
    return pred * (gt_med / pred_med)


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
        filenames_file_eval=args.filenames_file_eval,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval,
        eigen_crop=bool(args.eigen_crop),
        garg_crop=False,
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
        filenames_file_eval=args.filenames_file_eval,
        min_depth_eval=args.min_depth_eval,
        max_depth_eval=args.max_depth_eval,
        eigen_crop=bool(args.eigen_crop),
        garg_crop=False,
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


@torch.no_grad()
def predict_depthpro(args, image_path, image, depth_shape, valid, gt):
    for module_name in list(sys.modules):
        if module_name == "timm" or module_name.startswith("timm."):
            del sys.modules[module_name]
    sys.path.insert(0, str(Path(args.depthpro_timm_path).resolve()))
    sys.path.insert(0, args.depthpro_src)
    import depth_pro

    if not hasattr(predict_depthpro, "state"):
        config = dataclasses.replace(
            depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT,
            checkpoint_uri=args.depthpro_checkpoint,
        )
        model, transform = depth_pro.create_model_and_transforms(config=config, device=torch.device("cuda"))
        predict_depthpro.state = (model.eval(), transform)

    model, transform = predict_depthpro.state
    try:
        dp_image, _, f_px = depth_pro.load_rgb(image_path)
    except Exception:
        dp_image, f_px = image, None
    image_tensor = transform(dp_image).cuda()
    prediction = model.infer(image_tensor, f_px=f_px)["depth"].detach().cpu().numpy().astype(np.float32)
    if prediction.shape != depth_shape:
        tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0)
        prediction = F.interpolate(tensor, size=depth_shape, mode="bilinear", align_corners=False).squeeze().numpy()
    prediction = np.clip(prediction, args.min_depth_eval, args.max_depth_eval)
    if args.depthpro_align == "median":
        prediction = median_align(prediction, gt, valid)
        prediction = np.clip(prediction, args.min_depth_eval, args.max_depth_eval)
    return prediction


def prepare_visuals(record, args):
    gt = record["gt"]
    valid = record["valid"]
    rgb = record["rgb"]
    gt_vis = gt.copy()
    gt_vis[~valid] = -99
    max_depth = min(args.max_depth_eval, float(np.percentile(gt[valid], 95))) if valid.any() else args.max_depth_eval

    panels = [
        make_panel(f"RGB idx={record['idx']}", rgb),
        make_panel("GT depth", colorize(gt_vis, 0, max_depth, cmap="magma_r")),
    ]
    paper_panels = [
        make_panel("RGB", rgb, title_h=26),
        make_panel("GT", colorize(gt_vis, 0, max_depth, cmap="magma_r"), title_h=26),
    ]

    for key, title in [
        ("baseline", "ZoeDepth"),
        ("brrh", "BRRH-ZoeDepth"),
        ("depthpro", "Depth Pro"),
    ]:
        if key not in record:
            continue
        pred = record[key]
        pred_vis = pred.copy()
        pred_vis[~valid] = -99
        band = record["band"]
        band_score = band_absrel(pred, gt, band)
        panels.append(make_panel(f"{title} band={band_score:.3f}", colorize(pred_vis, 0, max_depth, cmap="magma_r")))
        paper_panels.append(make_panel(title, colorize(pred_vis, 0, max_depth, cmap="magma_r"), title_h=26))

    for key, title in [
        ("baseline", "ZoeDepth error"),
        ("brrh", "BRRH error"),
        ("depthpro", "Depth Pro error"),
    ]:
        if key not in record:
            continue
        panels.append(make_panel(title, colorize(rel_error(record[key], gt, valid), 0, args.error_max, cmap="inferno")))

    gt_edge = record["gt_edge"]
    panels.append(make_panel("GT boundary", edge_overlay(rgb, gt_edge, (0, 255, 80))))
    if "baseline" in record and "brrh" in record:
        gain_map = np.maximum(rel_error(record["baseline"], gt, valid) - rel_error(record["brrh"], gt, valid), 0.0)
        panels.append(make_panel("BRRH error reduction", colorize(gain_map, 0, args.gain_max, cmap="viridis")))

    return panels, paper_panels


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_checkpoint", default="/home/kxr/zoedepth_nyu_sync_baseline_resume_approx3ep_checkpoints/ZoeDepthv1_19-Jun_20-58-018db8113b87_latest.pt")
    parser.add_argument("--brrh_checkpoint", default="/home/kxr/zoedepth_nyu_sync_brrh_tuned_boundary_256x512_bs4_workers4_2ep_checkpoints/ZoeDepthv1_19-Jun_23-09-0c3cdbbc8dcc_latest.pt")
    parser.add_argument("--output_dir", default="figures/nyu_brrh_depthpro_comparison")
    parser.add_argument("--scan_samples", type=int, default=80)
    parser.add_argument("--save_top", type=int, default=4)
    parser.add_argument("--include_depthpro", type=int, default=1)
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
    parser.add_argument("--depthpro_src", default="/home/kxr/ml-depth-pro/src")
    parser.add_argument("--depthpro_checkpoint", default="/home/kxr/ml-depth-pro/checkpoints/depth_pro.pt")
    parser.add_argument("--depthpro_timm_path", default=".deps/depthpro_timm")
    parser.add_argument("--depthpro_align", choices=["none", "median"], default="median")
    parser.add_argument("--boundary_log_grad_threshold", type=float, default=0.15)
    parser.add_argument("--band_radius", type=int, default=3)
    parser.add_argument("--error_max", type=float, default=0.30)
    parser.add_argument("--gain_max", type=float, default=0.12)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, config = build_baseline_model(args)
    brrh_model, _ = build_brrh_model(args)
    loader = DepthDataLoader(config, "online_eval").data

    candidates = []
    cache = {}
    for idx, sample in tqdm(enumerate(loader), total=min(len(loader), args.scan_samples), desc="scan"):
        if idx >= args.scan_samples:
            break
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue

        image = sample["image"].cuda()
        depth = sample["depth"].cuda().squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.Tensor([518.8579]).cuda())

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

        valid = make_eval_mask(gt, args)
        gt_edge = depth_boundary(gt, valid, args.boundary_log_grad_threshold)
        band = dilate(gt_edge, args.band_radius) & valid
        baseline_band = band_absrel(baseline_np, gt, band)
        brrh_band = band_absrel(brrh_np, gt, band)
        score = baseline_band - brrh_band
        image_path = Path(args.data_path_eval) / remove_leading_slash(sample["image_path"][0])
        candidates.append((score, idx, baseline_band, brrh_band))
        cache[idx] = {
            "idx": idx,
            "image_path": str(image_path),
            "rgb": rgb,
            "gt": gt,
            "valid": valid,
            "gt_edge": gt_edge,
            "band": band,
            "baseline": baseline_np,
            "brrh": brrh_np,
            "baseline_band_absrel": baseline_band,
            "brrh_band_absrel": brrh_band,
        }

    candidates.sort(reverse=True)
    chosen = [item for item in candidates[: args.save_top] if item[0] > 0]
    if not chosen:
        chosen = candidates[: args.save_top]

    del baseline_model, brrh_model, loader
    torch.cuda.empty_cache()

    record_dir = out_dir / "records"
    record_dir.mkdir(exist_ok=True)
    for _, idx, _, _ in chosen:
        record = cache[idx]
        np.savez_compressed(
            record_dir / f"{idx:06d}.npz",
            rgb=record["rgb"],
            gt=record["gt"],
            valid=record["valid"],
            gt_edge=record["gt_edge"],
            band=record["band"],
            baseline=record["baseline"],
            brrh=record["brrh"],
        )

    if args.include_depthpro:
        for _, idx, _, _ in tqdm(chosen, desc="depthpro"):
            record = cache[idx]
            image = Image.open(record["image_path"]).convert("RGB")
            record["depthpro"] = predict_depthpro(
                args,
                record["image_path"],
                image,
                record["gt"].shape,
                record["valid"],
                record["gt"],
            )

    all_paper_panels = []
    selected_lines = []
    for score, idx, baseline_band, brrh_band in chosen:
        record = cache[idx]
        panels, paper_panels = prepare_visuals(record, args)
        save_montage(out_dir / f"{idx:06d}_nyu_brrh_depthpro.png", panels, columns=5)
        save_montage(out_dir / f"{idx:06d}_paper_row.png", paper_panels, columns=len(paper_panels))
        all_paper_panels.extend(paper_panels)
        depthpro_band = band_absrel(record["depthpro"], record["gt"], record["band"]) if "depthpro" in record else float("nan")
        selected_lines.append({
            "idx": idx,
            "image_path": record["image_path"],
            "gain": score,
            "baseline_band_absrel": baseline_band,
            "brrh_band_absrel": brrh_band,
            "depthpro_band_absrel": depthpro_band,
        })

    if all_paper_panels:
        per_row = 5 if args.include_depthpro else 4
        save_montage(out_dir / "nyu_brrh_depthpro_top_samples.png", all_paper_panels, columns=per_row)

    (out_dir / "selected_samples.json").write_text(json.dumps(selected_lines, indent=2) + "\n")
    print(f"Saved {len(chosen)} NYU BRRH/DepthPro visualizations to {out_dir}")


if __name__ == "__main__":
    main()
