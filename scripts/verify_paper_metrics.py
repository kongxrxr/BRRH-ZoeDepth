#!/usr/bin/env python3
"""Verify that key reported manuscript metrics are traceable to JSON logs.

The checker is deliberately simple and auditable. For each important table row,
it loads the authoritative JSON metric file, rounds the selected values to the
same six-decimal precision used in the paper, and checks that those values occur
in the LaTeX manuscript. This does not prove every table cell semantically, but
it catches the most common submission risk: numbers in the paper drifting away
from experiment logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper_rewriting_output" / "final_paper" / "main.tex"


@dataclass(frozen=True)
class MetricSpec:
    section: str
    label: str
    log_path: str
    metrics: tuple[str, ...]
    nested_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComplexitySpec:
    section: str
    label: str
    json_path: str
    metrics: tuple[str, ...]
    variant: str


SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "KITTI main/hard-boundary",
        "ZoeDepth baseline",
        "logs/kitti_baseline_extended_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "a1", "si_boundary_f1", "boundary_rmse"),
    ),
    MetricSpec(
        "KITTI main/hard-boundary",
        "BEiT BoundaryAlign",
        "logs/kitti_beit_boundaryalign_256x512_bs2_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "a1", "si_boundary_f1", "boundary_rmse"),
    ),
    MetricSpec(
        "KITTI main/hard-boundary",
        "BEiT + DA gate scale 0.16",
        "logs/kitti_beit_dav2gate_scale0p16_full_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "a1", "si_boundary_f1", "boundary_rmse", "edge_f1_tol3", "edge_f1_tol5", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI main/hard-boundary",
        "BEiT + DA gate scale 0.16 no gate",
        "logs/kitti_beit_dav2gate_scale0p16_nogate_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "edge_f1_tol3", "edge_f1_tol5", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI main/hard-boundary",
        "BEiT + DA gate scale 0.24 strong",
        "logs/kitti_beit_dav2gate_scale0p24_strong_bs4_workers8_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "a1", "si_boundary_f1", "boundary_rmse", "edge_f1_tol3", "edge_f1_tol5", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI main/hard-boundary",
        "BRRH scale 0.24 2epoch",
        "logs/kitti_brrh_scale0p24_bs4_workers8_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "a1", "si_boundary_f1", "boundary_rmse", "edge_f1_tol3", "edge_f1_tol5", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI main/hard-boundary",
        "BRRH scale 0.24 5epoch",
        "logs/kitti_brrh_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "a1", "si_boundary_f1", "boundary_rmse", "edge_f1_tol3", "edge_f1_tol5", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "NYU sync",
        "NYU-sync ZoeDepth baseline",
        "logs/nyu_sync_baseline_resume_approx3ep_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "a1", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3", "edge_f1_tol5"),
    ),
    MetricSpec(
        "NYU sync",
        "NYU-sync BRRH",
        "logs/nyu_sync_brrh_prevconfig_resume_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "a1", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3", "edge_f1_tol5"),
    ),
    MetricSpec(
        "NYU sync",
        "NYU-sync BRRH tuned",
        "logs/nyu_sync_brrh_tuned_boundary_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "a1", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3", "edge_f1_tol5"),
    ),
    MetricSpec(
        "NYU external",
        "DA-V2 inverse median",
        "logs/nyu_dav2_small_inverse_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "a1", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3", "edge_f1_tol5"),
    ),
    MetricSpec(
        "NYU external",
        "Depth Pro median",
        "logs/nyu_depthpro_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "a1", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3", "edge_f1_tol5"),
    ),
    MetricSpec(
        "NYU alignment",
        "DA-V2 inverse none",
        "logs/nyu_dav2_small_inverse_noalign_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3"),
    ),
    MetricSpec(
        "NYU alignment",
        "Depth Pro none",
        "logs/nyu_depthpro_noalign_val654_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without DA-V2 prior",
        "logs/kitti_brrh_nodaprior_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without temperature sharpening",
        "logs/kitti_brrh_notemperature_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without boundary-band loss",
        "logs/kitti_brrh_nobandloss_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without contrast loss",
        "logs/kitti_brrh_nocontrastloss_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without boundary losses",
        "logs/kitti_brrh_noboundaryloss_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without preservation",
        "logs/kitti_brrh_nopreserve_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
    MetricSpec(
        "KITTI ablation",
        "BRRH without residual head",
        "logs/kitti_brrh_noresidual_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json",
        ("abs_rel", "rmse", "silog", "boundary_rmse", "top5_abs_rel", "band3_abs_rel"),
    ),
)

DENSITY_SPECS: tuple[MetricSpec, ...] = tuple(
    MetricSpec(
        "NYU density",
        f"{model} {subset}",
        "logs/nyu_boundary_density_subsets_baseline_brrh.json",
        ("boundary_density", "abs_rel", "boundary_rmse", "band3_abs_rel", "edge_f1_tol3"),
        (model, subset),
    )
    for subset in ("low", "medium", "high")
    for model in ("baseline", "brrh")
)

COMPLEXITY_SPECS: tuple[ComplexitySpec, ...] = (
    ComplexitySpec(
        "Model complexity",
        "ZoeDepth baseline",
        "reports/model_complexity_profile_latest.json",
        ("total_params", "trainable_params", "boundary_refiner_params", "discontinuity_head_params", "frozen_da_fuser_params", "frozen_da_prior_params", "added_boundary_module_params"),
        "baseline",
    ),
    ComplexitySpec(
        "Model complexity",
        "BRRH-ZoeDepth",
        "reports/model_complexity_profile_latest.json",
        ("total_params", "trainable_params", "boundary_refiner_params", "discontinuity_head_params", "frozen_da_fuser_params", "frozen_da_prior_params", "added_boundary_module_params"),
        "brrh",
    ),
)


def load_metric(path: Path, nested_path: tuple[str, ...]) -> dict[str, Any]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in nested_path:
        data = data[key]
    if not isinstance(data, dict):
        raise TypeError(f"{path} nested path {nested_path} did not resolve to a dict")
    return data


def format_value(value: Any) -> str:
    if not isinstance(value, (float, int)):
        raise TypeError(f"metric value is not numeric: {value!r}")
    return f"{float(value):.6f}"


def format_param_millions(value: Any) -> str:
    if not isinstance(value, (float, int)):
        raise TypeError(f"parameter value is not numeric: {value!r}")
    return f"{float(value) / 1_000_000:.3f}M"


def verify(spec: MetricSpec, tex: str) -> tuple[list[str], list[str], list[str]]:
    missing_files: list[str] = []
    missing_metrics: list[str] = []
    missing_in_tex: list[str] = []
    path = ROOT / spec.log_path
    if not path.exists():
        missing_files.append(spec.log_path)
        return missing_files, missing_metrics, missing_in_tex
    data = load_metric(path, spec.nested_path)
    for metric in spec.metrics:
        if metric not in data:
            missing_metrics.append(f"{spec.label}:{metric}")
            continue
        rendered = format_value(data[metric])
        if rendered not in tex:
            missing_in_tex.append(f"{spec.label}:{metric}={rendered}")
    return missing_files, missing_metrics, missing_in_tex


def verify_complexity(spec: ComplexitySpec, tex: str) -> tuple[list[str], list[str], list[str]]:
    missing_files: list[str] = []
    missing_metrics: list[str] = []
    missing_in_tex: list[str] = []
    path = ROOT / spec.json_path
    if not path.exists():
        missing_files.append(spec.json_path)
        return missing_files, missing_metrics, missing_in_tex
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        missing_metrics.append(f"{spec.label}:complexity json is not a list")
        return missing_files, missing_metrics, missing_in_tex
    by_variant = {item.get("variant"): item for item in items if isinstance(item, dict)}
    data = by_variant.get(spec.variant)
    if data is None:
        missing_metrics.append(f"{spec.label}:variant={spec.variant}")
        return missing_files, missing_metrics, missing_in_tex
    for metric in spec.metrics:
        if metric not in data:
            missing_metrics.append(f"{spec.label}:{metric}")
            continue
        rendered = format_param_millions(data[metric])
        if rendered not in tex:
            missing_in_tex.append(f"{spec.label}:{metric}={rendered}")
    return missing_files, missing_metrics, missing_in_tex


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", default="", help="Optional markdown report path relative to the repository root.")
    args = parser.parse_args()

    tex = TEX_PATH.read_text(encoding="utf-8", errors="replace")
    specs = SPECS + DENSITY_SPECS

    rows: list[tuple[str, str, int, str]] = []
    all_missing_files: list[str] = []
    all_missing_metrics: list[str] = []
    all_missing_in_tex: list[str] = []

    for spec in specs:
        missing_files, missing_metrics, missing_in_tex = verify(spec, tex)
        all_missing_files.extend(missing_files)
        all_missing_metrics.extend(missing_metrics)
        all_missing_in_tex.extend(missing_in_tex)
        status = "PASS" if not (missing_files or missing_metrics or missing_in_tex) else "FAIL"
        detail_parts = []
        if missing_files:
            detail_parts.append("missing log")
        if missing_metrics:
            detail_parts.append(f"missing metrics: {len(missing_metrics)}")
        if missing_in_tex:
            detail_parts.append(f"values not found in manuscript: {len(missing_in_tex)}")
        rows.append((spec.section, spec.label, len(spec.metrics), status if not detail_parts else status + " - " + ", ".join(detail_parts)))

    for spec in COMPLEXITY_SPECS:
        missing_files, missing_metrics, missing_in_tex = verify_complexity(spec, tex)
        all_missing_files.extend(missing_files)
        all_missing_metrics.extend(missing_metrics)
        all_missing_in_tex.extend(missing_in_tex)
        status = "PASS" if not (missing_files or missing_metrics or missing_in_tex) else "FAIL"
        detail_parts = []
        if missing_files:
            detail_parts.append("missing complexity json")
        if missing_metrics:
            detail_parts.append(f"missing complexity metrics: {len(missing_metrics)}")
        if missing_in_tex:
            detail_parts.append(f"complexity values not found in manuscript: {len(missing_in_tex)}")
        rows.append((spec.section, spec.label, len(spec.metrics), status if not detail_parts else status + " - " + ", ".join(detail_parts)))

    print("Paper metric verification")
    print(f"Specs checked: {len(specs) + len(COMPLEXITY_SPECS)}")
    print(f"Missing log files: {len(all_missing_files)}")
    print(f"Missing metric keys: {len(all_missing_metrics)}")
    print(f"Logged values not found in manuscript: {len(all_missing_in_tex)}")
    print()
    for section, label, metric_count, status in rows:
        print(f"[{status}] {section} :: {label} ({metric_count} metrics)")

    if all_missing_files or all_missing_metrics or all_missing_in_tex:
        print("\nProblems:")
        for item in all_missing_files:
            print(f"- Missing log: {item}")
        for item in all_missing_metrics:
            print(f"- Missing metric: {item}")
        for item in all_missing_in_tex:
            print(f"- Value not found in manuscript: {item}")

    if args.write_report:
        report_path = ROOT / args.write_report
        lines = [
            "# Paper Metric Verification",
            "",
            "Command:",
            "",
            "```bash",
            f"python scripts/verify_paper_metrics.py --write-report {args.write_report}",
            "```",
            "",
            f"Metric specs checked: {len(specs)}",
            f"Complexity specs checked: {len(COMPLEXITY_SPECS)}",
            f"Missing log files: {len(all_missing_files)}",
            f"Missing metric keys: {len(all_missing_metrics)}",
            f"Logged values not found in manuscript: {len(all_missing_in_tex)}",
            "",
            "| Section | Label | Metrics checked | Status |",
            "|---|---|---:|---|",
        ]
        for section, label, metric_count, status in rows:
            lines.append(f"| {section} | {label} | {metric_count} | {status} |")
        if all_missing_files or all_missing_metrics or all_missing_in_tex:
            lines.extend(["", "## Problems", ""])
            for item in all_missing_files:
                lines.append(f"- Missing log: `{item}`")
            for item in all_missing_metrics:
                lines.append(f"- Missing metric: `{item}`")
            for item in all_missing_in_tex:
                lines.append(f"- Value not found in manuscript: `{item}`")
        else:
            lines.extend(["", "All checked manuscript values are traceable to configured JSON files: experiment metrics at six-decimal precision and complexity values as three-decimal million-parameter values."])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote report: {report_path}")

    return 1 if (all_missing_files or all_missing_metrics or all_missing_in_tex) else 0


if __name__ == "__main__":
    sys.exit(main())
