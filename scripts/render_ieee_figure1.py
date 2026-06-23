#!/usr/bin/env python3
"""Render Figure 1 as a DiffusionDepth-Fig.2-style overview schematic."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


BLACK = "#111111"
GRAY = "#6B7280"
LIGHT_GRAY = "#F4F5F7"
BLUE = "#BFD7F2"
BLUE_EDGE = "#2F5F9F"
GREEN = "#CFEBD8"
GREEN_EDGE = "#257448"
PURPLE = "#DCD2F4"
PURPLE_EDGE = "#6B55B7"
ORANGE = "#F8D4A6"
ORANGE_EDGE = "#B85D08"
YELLOW = "#F9E79F"
YELLOW_EDGE = "#9A7A00"
PINK = "#F6C9D0"
PINK_EDGE = "#9A3A46"


def add_box(ax, x, y, w, h, text, fc, ec, fontsize=7.2, weight="bold", linespacing=1.12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=linespacing,
        color=BLACK,
    )
    return patch


def arrow(ax, p0, p1, color=BLACK, lw=1.0, style="-", ms=8, rad=0.0):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def add_label(ax, x, y, text, fontsize=5.4, color=GRAY, weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight=weight)


def add_rgb_icon(ax, x, y, w, h):
    nx, ny = 80, 52
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)[:, None]
    img = np.zeros((ny, nx, 3), dtype=float)
    img[..., 0] = 0.25 + 0.55 * xs
    img[..., 1] = 0.35 + 0.45 * ys
    img[..., 2] = 0.70 - 0.25 * xs + 0.15 * ys
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=1)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=BLACK, linewidth=0.7, zorder=2))
    # simple road/object strokes keep it visually close to a perception-paper input thumbnail
    ax.add_patch(Polygon([[x + 0.12 * w, y], [x + 0.47 * w, y + 0.52 * h], [x + 0.58 * w, y + 0.52 * h], [x + 0.86 * w, y]], color="#6E6E6E", alpha=0.55, zorder=2))
    ax.add_patch(Rectangle((x + 0.18 * w, y + 0.45 * h), 0.18 * w, 0.18 * h, color="#D9E6F2", ec="#3B4A5A", lw=0.45, zorder=3))
    ax.add_patch(Rectangle((x + 0.62 * w, y + 0.50 * h), 0.16 * w, 0.14 * h, color="#F3D08A", ec="#5D4C25", lw=0.45, zorder=3))


def add_depth_icon(ax, x, y, w, h):
    nx, ny = 80, 52
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)[:, None]
    img = np.zeros((ny, nx, 3), dtype=float)
    img[..., 0] = 0.95 - 0.45 * ys
    img[..., 1] = 0.30 + 0.55 * (1 - ys)
    img[..., 2] = 0.18 + 0.65 * xs
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=1)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=BLACK, linewidth=0.7, zorder=2))


def add_feature_stack(ax, x, y, w, h, colors):
    offsets = [(0.018, 0.020), (0.009, 0.010), (0.0, 0.0)]
    for i, (dx, dy) in enumerate(offsets):
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor=colors[i], edgecolor=BLACK, linewidth=0.55, zorder=1 + i))
        for k in range(1, 4):
            ax.plot([x + dx + k * w / 4, x + dx + k * w / 4], [y + dy, y + dy + h], color="white", lw=0.35, alpha=0.65, zorder=2 + i)
            ax.plot([x + dx, x + dx + w], [y + dy + k * h / 4, y + dy + k * h / 4], color="white", lw=0.35, alpha=0.65, zorder=2 + i)


def render(out_pdf: Path, out_png: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.25), dpi=400)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Left input and condition construction, following the overview style of DiffusionDepth Fig. 2.
    add_rgb_icon(ax, 0.035, 0.62, 0.105, 0.18)
    add_label(ax, 0.087, 0.835, "RGB image", fontsize=6.4, color=BLACK, weight="bold")
    arrow(ax, (0.145, 0.71), (0.195, 0.71), BLACK, lw=1.0)

    add_box(ax, 0.20, 0.59, 0.13, 0.24, "ZoeDepth\nDPT encoder", BLUE, BLUE_EDGE, fontsize=7.0)
    arrow(ax, (0.33, 0.71), (0.38, 0.71), BLACK, lw=1.0)

    add_feature_stack(ax, 0.385, 0.635, 0.075, 0.13, ["#8EC3E8", "#A8D5BA", "#D7C6F2"])
    add_label(ax, 0.422, 0.835, "multi-scale\nfeatures", fontsize=5.2, color=BLACK)
    arrow(ax, (0.47, 0.71), (0.515, 0.71), BLACK, lw=1.0)

    add_box(ax, 0.52, 0.59, 0.135, 0.24, "Metric bins\n+ decoder", GREEN, GREEN_EDGE, fontsize=6.9)
    add_label(ax, 0.587, 0.625, "$D_b$", fontsize=4.9, color=GREEN_EDGE)

    # Structural prior branch.
    add_box(ax, 0.20, 0.30, 0.13, 0.17, "Frozen\nDA-V2", LIGHT_GRAY, GRAY, fontsize=7.0)
    add_feature_stack(ax, 0.385, 0.315, 0.075, 0.105, ["#DADDE2", "#C9CED6", "#B9C1CC"])
    add_label(ax, 0.374, 0.465, "structural\nprior", fontsize=5.2, color=BLACK)
    arrow(ax, (0.33, 0.385), (0.385, 0.385), GRAY, lw=0.9, style="--", ms=7)

    # Boundary condition stream.
    arrow(ax, (0.425, 0.635), (0.425, 0.49), BLACK, lw=0.9)
    add_box(ax, 0.52, 0.31, 0.135, 0.18, "Discontinuity\nhead", PURPLE, PURPLE_EDGE, fontsize=6.9)
    add_label(ax, 0.587, 0.345, "$B$", fontsize=4.9, color=PURPLE_EDGE)
    arrow(ax, (0.46, 0.385), (0.52, 0.40), GRAY, lw=0.9, style="--", ms=7)
    arrow(ax, (0.425, 0.49), (0.52, 0.43), BLACK, lw=0.9, rad=-0.12)

    add_box(ax, 0.70, 0.31, 0.13, 0.18, "Boundary\nmask", PURPLE, PURPLE_EDGE, fontsize=6.9)
    add_label(ax, 0.765, 0.345, "$M_b=\\sigma(B)$", fontsize=4.9, color=PURPLE_EDGE)
    arrow(ax, (0.655, 0.40), (0.70, 0.40), BLACK, lw=1.0)

    # Refinement target at right: base depth and boundary condition meet in BRRH.
    add_box(ax, 0.70, 0.59, 0.13, 0.24, "BRRH\nresidual\nrefinement", ORANGE, ORANGE_EDGE, fontsize=6.8)
    add_label(ax, 0.765, 0.615, "$D_t=D_b+M_b\\Delta_b$", fontsize=4.9, color=ORANGE_EDGE)
    arrow(ax, (0.655, 0.71), (0.70, 0.71), BLACK, lw=1.0)
    arrow(ax, (0.765, 0.49), (0.765, 0.59), BLACK, lw=1.0)
    arrow(ax, (0.83, 0.71), (0.885, 0.71), BLACK, lw=1.0)

    add_depth_icon(ax, 0.89, 0.62, 0.085, 0.18)
    add_label(ax, 0.933, 0.835, "metric depth", fontsize=6.4, color=BLACK, weight="bold")

    # Training-only path, styled as the lower supervision line in overview figures.
    add_box(ax, 0.055, 0.095, 0.13, 0.12, "GT depth\nvalid mask", YELLOW, YELLOW_EDGE, fontsize=6.7)
    add_box(ax, 0.29, 0.095, 0.14, 0.12, "Boundary\ntarget", PINK, PINK_EDGE, fontsize=6.7)
    add_box(ax, 0.53, 0.095, 0.15, 0.12, "Training\nlosses", LIGHT_GRAY, GRAY, fontsize=6.7)
    arrow(ax, (0.185, 0.155), (0.29, 0.155), GRAY, lw=0.85, style="--", ms=6)
    arrow(ax, (0.43, 0.155), (0.53, 0.155), GRAY, lw=0.85, style="--", ms=6)
    arrow(ax, (0.605, 0.215), (0.585, 0.31), GRAY, lw=0.85, style="--", ms=6)
    arrow(ax, (0.645, 0.215), (0.74, 0.31), GRAY, lw=0.85, style="--", ms=6)

    # Figure-style section labels.
    ax.plot([0.20, 0.655], [0.885, 0.885], color="#A9A9A9", lw=0.65)
    add_label(ax, 0.427, 0.915, "monocular condition construction", fontsize=5.8, color=BLACK)
    ax.plot([0.70, 0.975], [0.885, 0.885], color="#A9A9A9", lw=0.65)
    add_label(ax, 0.837, 0.915, "boundary-aware metric refinement", fontsize=5.8, color=BLACK)
    ax.plot([0.055, 0.68], [0.055, 0.055], color="#BFC5CE", lw=0.65, linestyle="--")
    add_label(ax, 0.37, 0.032, "training-only supervision", fontsize=5.3, color=GRAY)

    # Legend with the same economy as common conference/journal overview figures.
    arrow(ax, (0.785, 0.14), (0.835, 0.14), BLACK, lw=0.9, ms=6)
    add_label(ax, 0.895, 0.14, "inference", fontsize=5.3, color=BLACK)
    arrow(ax, (0.785, 0.095), (0.835, 0.095), GRAY, lw=0.85, style="--", ms=6)
    add_label(ax, 0.895, 0.095, "supervision", fontsize=5.3, color=GRAY)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.025)
    if out_png is not None:
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
    render(args.pdf, args.png)


if __name__ == "__main__":
    main()
