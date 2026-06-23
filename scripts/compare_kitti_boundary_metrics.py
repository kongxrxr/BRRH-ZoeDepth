#!/usr/bin/env python
import argparse
import json
from pathlib import Path


DEFAULT_KEYS = (
    "abs_rel",
    "rmse",
    "a1",
    "silog",
    "boundary_rmse",
    "si_boundary_f1",
    "edge_f1_tol3",
    "edge_f1_tol5",
    "top5_abs_rel",
    "band3_abs_rel",
)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    lines = ["| experiment | " + " | ".join(keys) + " |"]
    lines.append("|---|" + "|".join(["---"] * len(keys)) + "|")

    for filename in args.files:
        path = Path(filename)
        with path.open() as f:
            data = json.load(f)
        values = [fmt(data.get(key)) for key in keys]
        lines.append(f"| {path.stem} | " + " | ".join(values) + " |")

    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
