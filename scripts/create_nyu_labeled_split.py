#!/usr/bin/env python
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", dest="all_file", default="/home/kxr/ZoeDepth/train_test_inputs/nyudepthv2_labeled_all_files_with_gt.txt")
    parser.add_argument("--train-out", default="/home/kxr/ZoeDepth/train_test_inputs/nyudepthv2_labeled_train795_files_with_gt.txt")
    parser.add_argument("--val-out", default="/home/kxr/ZoeDepth/train_test_inputs/nyudepthv2_labeled_val654_files_with_gt.txt")
    parser.add_argument("--train-count", type=int, default=795)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    lines = Path(args.all_file).read_text().splitlines()
    lines = [line + "\n" for line in lines if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(lines)

    train = lines[:args.train_count]
    val = lines[args.train_count:]
    Path(args.train_out).write_text("".join(train))
    Path(args.val_out).write_text("".join(val))
    print(f"train: {len(train)} -> {args.train_out}")
    print(f"val: {len(val)} -> {args.val_out}")


if __name__ == "__main__":
    main()
