#!/usr/bin/env python3
"""Create IEEE-style framed qualitative figures from existing result grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLACK = (20, 20, 20)
WHITE = (255, 255, 255)
LIGHT = (246, 247, 249)
GRAY = (120, 128, 140)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt, fill=BLACK) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=fnt, fill=fill)


def add_border(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], width: int = 3) -> None:
    for i in range(width):
        draw.rectangle((box[0] + i, box[1] + i, box[2] - i, box[3] - i), outline=BLACK)


def render_nyu(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    header_h = max(54, h // 24)
    margin = 12
    out = Image.new("RGB", (w + 2 * margin, h + header_h + 2 * margin), WHITE)
    draw = ImageDraw.Draw(out)
    out.paste(im, (margin, margin + header_h))

    labels = ["RGB", "GT depth", "ZoeDepth", "BRRH-ZoeDepth", "Depth Pro"]
    col_w = w / 5.0
    fnt = font(max(22, w // 145), bold=True)
    for i, label in enumerate(labels):
        x0 = int(margin + i * col_w)
        x1 = int(margin + (i + 1) * col_w)
        draw.rectangle((x0, margin, x1, margin + header_h), fill=LIGHT)
        draw_center(draw, (x0, margin, x1, margin + header_h), label, fnt)
        draw.line((x0, margin, x0, h + header_h + margin), fill=WHITE, width=4)
    draw.line((margin + w, margin, margin + w, h + header_h + margin), fill=WHITE, width=4)
    for r in range(1, 3):
        y = int(margin + header_h + r * h / 3.0)
        draw.line((margin, y, margin + w, y), fill=WHITE, width=5)
    add_border(draw, (margin, margin, margin + w, margin + header_h + h), width=3)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)


def render_kitti(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    margin = 12
    header_h = max(44, h // 18)
    row_gap = 8
    out = Image.new("RGB", (w + 2 * margin, h + 2 * header_h + row_gap + 2 * margin), WHITE)
    draw = ImageDraw.Draw(out)

    half_h = h // 2
    top = im.crop((0, 0, w, half_h))
    bottom = im.crop((0, half_h, w, h))
    top_y = margin + header_h
    bottom_y = top_y + half_h + row_gap + header_h
    out.paste(top, (margin, top_y))
    out.paste(bottom, (margin, bottom_y))

    top_labels = ["RGB input", "GT depth", "Full BRRH depth"]
    bottom_labels = ["No-residual depth", "Full BRRH rel. error", "No-residual rel. error"]
    col_w = w / 3.0
    fnt = font(max(18, w // 190), bold=True)

    for row_y, labels in ((margin, top_labels), (top_y + half_h + row_gap, bottom_labels)):
        for i, label in enumerate(labels):
            x0 = int(margin + i * col_w)
            x1 = int(margin + (i + 1) * col_w)
            draw.rectangle((x0, row_y, x1, row_y + header_h), fill=LIGHT)
            draw_center(draw, (x0, row_y, x1, row_y + header_h), label, fnt)
            draw.line((x0, row_y, x0, row_y + header_h + half_h), fill=WHITE, width=4)
        draw.line((margin + w, row_y, margin + w, row_y + header_h + half_h), fill=WHITE, width=4)
        add_border(draw, (margin, row_y, margin + w, row_y + header_h + half_h), width=3)

    # Preserve the numeric band-error message in a cleaner footer.
    footer_font = font(max(16, w // 220), bold=False)
    draw.text((margin, out.height - margin - 28), "Selected KITTI sample; full BRRH band AbsRel 0.183 vs no-residual 0.198.", font=footer_font, fill=GRAY)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", required=True, type=Path)
    args = parser.parse_args()
    render_nyu(args.fig_dir / "nyu_brrh_depthpro_top_samples.png", args.fig_dir / "nyu_brrh_depthpro_top_samples_ieee.png")
    render_kitti(args.fig_dir / "brrh_noresidual_qualitative.png", args.fig_dir / "brrh_noresidual_qualitative_ieee.png")


if __name__ == "__main__":
    main()
