#!/usr/bin/env python3
"""Profile parameter counts and optional inference latency for ZoeDepth variants.

The primary purpose is reviewer-facing reproducibility: BRRH is claimed as a
lightweight residual refinement added to a ZoeDepth-style metric pipeline. This
script reports total parameters, trainable parameters, and parameters in the
added boundary/DA-prior modules. If CUDA is available and --benchmark is passed,
it also measures simple synthetic-input inference latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


def module_param_count(model: torch.nn.Module, module_name: str) -> int:
    module = getattr(model, module_name, None)
    if module is None:
        return 0
    return count_params(module, trainable_only=False)


def get_variant_config(args: argparse.Namespace, variant: str) -> Any:
    config = get_config(
        "zoedepth",
        mode="train",
        dataset=args.dataset,
        config_version=args.config_version,
    )
    config.pretrained_resource = None
    config.checkpoint = ""
    config.img_size = args.img_size
    config.midas_model_type = args.midas_model_type
    config.train_midas = args.train_midas
    config.use_pretrained_midas = False

    if variant == "baseline":
        config.use_boundary_refine = False
        config.use_discontinuity_branch = False
        config.use_discontinuity_temperature = False
        config.use_frozen_da_prior = False
        config.boundary_refine_use_da_prior = False
    elif variant == "brrh":
        config.use_boundary_refine = True
        config.boundary_refine_channels = args.boundary_refine_channels
        config.boundary_refine_scale = args.boundary_refine_scale
        config.boundary_refine_mode = "log_residual"
        config.boundary_refine_use_da_prior = True
        config.use_discontinuity_branch = True
        config.discontinuity_channels = args.discontinuity_channels
        config.use_discontinuity_temperature = True
        config.discontinuity_temperature_scale = args.discontinuity_temperature_scale
        config.use_frozen_da_prior = True
        config.frozen_da_model = args.frozen_da_model
        config.frozen_da_feature_channels = args.frozen_da_feature_channels
        config.frozen_da_input_size = args.frozen_da_input_size
        config.frozen_da_fusion_scale = args.frozen_da_fusion_scale
        config.use_frozen_da_boundary_gate = False
        config.frozen_da_min_gate = 0.05
    else:
        raise ValueError(f"unknown variant: {variant}")
    return config


def benchmark_model(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    height, width = [int(x) for x in args.img_size.split(",")]
    model = model.to(device).eval()
    x = torch.randn(args.batch_size, 3, height, width, device=device)
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = model(x)
            if use_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_ms_mean": statistics.mean(times),
        "latency_ms_median": statistics.median(times),
        "latency_ms_min": min(times),
        "latency_ms_max": max(times),
        "fps_single_batch_equiv": 1000.0 * args.batch_size / statistics.mean(times),
    }


def profile_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    config = get_variant_config(args, variant)
    model = build_model(config)

    result: dict[str, Any] = {
        "variant": variant,
        "dataset": args.dataset,
        "config_version": args.config_version,
        "img_size": args.img_size,
        "midas_model_type": args.midas_model_type,
        "train_midas": bool(args.train_midas),
        "total_params": count_params(model, trainable_only=False),
        "trainable_params": count_params(model, trainable_only=True),
        "core_params": module_param_count(model, "core"),
        "boundary_refiner_params": module_param_count(model, "boundary_refiner"),
        "discontinuity_head_params": module_param_count(model, "discontinuity_head"),
        "frozen_da_prior_params": module_param_count(model, "frozen_da_prior"),
        "frozen_da_fuser_params": module_param_count(model, "frozen_da_fuser"),
    }
    result["added_boundary_module_params"] = (
        result["boundary_refiner_params"]
        + result["discontinuity_head_params"]
        + result["frozen_da_fuser_params"]
    )
    result["added_with_frozen_da_prior_params"] = (
        result["added_boundary_module_params"] + result["frozen_da_prior_params"]
    )

    if args.benchmark:
        result.update(benchmark_model(model, args))
    return result


def fmt_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def write_markdown(results: list[dict[str, Any]], output: Path) -> None:
    by_variant = {item["variant"]: item for item in results}
    lines = [
        "# Model Complexity Profile",
        "",
        "This report is generated by:",
        "",
        "```bash",
        "/home/kxr/miniconda3/envs/zoe/bin/python scripts/profile_model_complexity.py --output reports/model_complexity_profile_latest.json --markdown reports/model_complexity_profile_latest.md",
        "```",
        "",
        "| Variant | Total params | Trainable params | Boundary refiner | Discontinuity head | DA fuser | Frozen DA prior | Added trainable boundary modules |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            "| {variant} | {total} | {trainable} | {refiner} | {disc} | {fuser} | {prior} | {added} |".format(
                variant=item["variant"],
                total=fmt_millions(item["total_params"]),
                trainable=fmt_millions(item["trainable_params"]),
                refiner=fmt_millions(item["boundary_refiner_params"]),
                disc=fmt_millions(item["discontinuity_head_params"]),
                fuser=fmt_millions(item["frozen_da_fuser_params"]),
                prior=fmt_millions(item["frozen_da_prior_params"]),
                added=fmt_millions(item["added_boundary_module_params"]),
            )
        )

    if "baseline" in by_variant and "brrh" in by_variant:
        baseline = by_variant["baseline"]
        brrh = by_variant["brrh"]
        total_delta = brrh["total_params"] - baseline["total_params"]
        trainable_delta = brrh["trainable_params"] - baseline["trainable_params"]
        lightweight_delta = brrh["added_boundary_module_params"]
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- Total-parameter delta including the frozen DA-V2 prior: {fmt_millions(total_delta)}.",
                f"- Trainable-parameter delta: {fmt_millions(trainable_delta)}.",
                f"- Added trainable BRRH/discontinuity/fusion modules: {fmt_millions(lightweight_delta)}.",
                "- The frozen DA-V2 prior should be reported separately from trainable BRRH parameters because it is used as a fixed structural guide rather than an optimized metric-depth head.",
            ]
        )

    if any("latency_ms_mean" in item for item in results):
        lines.extend(
            [
                "",
                "## Synthetic Inference Timing",
                "",
                "| Variant | Mean latency (ms) | Median latency (ms) | FPS equivalent |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in results:
            if "latency_ms_mean" in item:
                lines.append(
                    f"| {item['variant']} | {item['latency_ms_mean']:.3f} | {item['latency_ms_median']:.3f} | {item['fps_single_batch_equiv']:.3f} |"
                )
        lines.append("")
        lines.append("Timing is measured on synthetic input and should be reported as a local hardware diagnostic, not as a universal benchmark.")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kitti", choices=["kitti", "nyu"])
    parser.add_argument("--config_version", default="kitti")
    parser.add_argument("--img_size", default="256,512")
    parser.add_argument("--midas_model_type", default="DPT_BEiT_L_384")
    parser.add_argument("--train_midas", type=int, default=0)
    parser.add_argument("--frozen_da_model", default="/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6")
    parser.add_argument("--frozen_da_feature_channels", type=int, default=8)
    parser.add_argument("--frozen_da_input_size", type=int, default=384)
    parser.add_argument("--frozen_da_fusion_scale", type=float, default=0.12)
    parser.add_argument("--boundary_refine_channels", type=int, default=32)
    parser.add_argument("--boundary_refine_scale", type=float, default=0.08)
    parser.add_argument("--discontinuity_channels", type=int, default=32)
    parser.add_argument("--discontinuity_temperature_scale", type=float, default=0.5)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default="reports/model_complexity_profile_latest.json")
    parser.add_argument("--markdown", default="reports/model_complexity_profile_latest.md")
    args = parser.parse_args()

    if args.benchmark and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")

    results = [profile_variant(args, "baseline"), profile_variant(args, "brrh")]

    output = ROOT / args.output
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = ROOT / args.markdown
    write_markdown(results, markdown)

    print(f"Wrote {output}")
    print(f"Wrote {markdown}")
    for item in results:
        print(
            f"{item['variant']}: total={fmt_millions(item['total_params'])}, "
            f"trainable={fmt_millions(item['trainable_params'])}, "
            f"added_boundary={fmt_millions(item['added_boundary_module_params'])}, "
            f"frozen_da_prior={fmt_millions(item['frozen_da_prior_params'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
