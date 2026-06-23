#!/usr/bin/env python
import argparse
from pathlib import Path


def iter_entries(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"{split_file}:{line_no} expected image and depth paths")
            yield line_no, parts


def check_split(split_file, raw_root, gt_root, limit_missing):
    total = 0
    missing = []
    for line_no, parts in iter_entries(split_file):
        total += 1
        image_path = raw_root / parts[0].lstrip("/\\")
        depth_path = None if parts[1].lower() == "none" else gt_root / parts[1].lstrip("/\\")
        if not image_path.is_file() or (depth_path is not None and not depth_path.is_file()):
            missing.append((line_no, image_path, depth_path))
            if len(missing) >= limit_missing:
                break
    return total, missing


def main():
    parser = argparse.ArgumentParser(description="Validate ZoeDepth KITTI split paths.")
    parser.add_argument("--raw-root", default="~/shortcuts/datasets/kitti/raw")
    parser.add_argument("--gt-root", default="~/shortcuts/datasets/kitti/gts")
    parser.add_argument("--train-split", default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt")
    parser.add_argument("--eval-split", default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt")
    parser.add_argument("--limit-missing", type=int, default=10)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    train_split = Path(args.train_split).expanduser().resolve()
    eval_split = Path(args.eval_split).expanduser().resolve()

    errors = []
    for name, path in (("raw root", raw_root), ("gt root", gt_root), ("train split", train_split), ("eval split", eval_split)):
        if not path.exists():
            errors.append(f"Missing {name}: {path}")

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)

    failed = False
    for label, split in (("train", train_split), ("eval", eval_split)):
        total, missing = check_split(split, raw_root, gt_root, args.limit_missing)
        print(f"{label}: checked {total} entries from {split}")
        if missing:
            failed = True
            print(f"{label}: missing files, first {len(missing)}:")
            for line_no, image_path, depth_path in missing:
                if not image_path.is_file():
                    print(f"  line {line_no}: missing image {image_path}")
                if depth_path is not None and not depth_path.is_file():
                    print(f"  line {line_no}: missing depth {depth_path}")
        else:
            print(f"{label}: ok")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
