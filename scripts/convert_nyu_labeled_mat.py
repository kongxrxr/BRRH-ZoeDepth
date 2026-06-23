#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def read_ref_string(h5_file, ref):
    arr = h5_file[ref][()]
    return "".join(chr(int(c)) for c in arr.flatten())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", default="/home/kxr/shortcuts/datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    parser.add_argument("--out-root", default="/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all")
    parser.add_argument("--split-out", default="/home/kxr/ZoeDepth/train_test_inputs/nyudepthv2_labeled_all_files_with_gt.txt")
    parser.add_argument("--focal", default="518.8579")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    rgb_dir = out_root / "rgb"
    depth_dir = out_root / "sync_depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    mat_path = Path(args.mat)
    split_path = Path(args.split_out)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(mat_path, "r") as f:
        num_samples = int(f["images"].shape[0])
        if args.limit > 0:
            num_samples = min(num_samples, args.limit)

        lines = []
        for idx in tqdm(range(num_samples), desc="Converting NYU labeled"):
            image = f["images"][idx]
            depth = f["depths"][idx]

            image = np.transpose(image, (2, 1, 0))
            depth = np.transpose(depth, (1, 0))

            rgb_rel = f"rgb/rgb_{idx:06d}.jpg"
            depth_rel = f"sync_depth/sync_depth_{idx:06d}.png"
            Image.fromarray(image).save(out_root / rgb_rel, quality=95)

            depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            Image.fromarray(depth_mm).save(out_root / depth_rel)
            lines.append(f"{rgb_rel} {depth_rel} {args.focal}\n")

    split_path.write_text("".join(lines))
    print(f"Converted {len(lines)} samples")
    print(f"Output root: {out_root}")
    print(f"Split file: {split_path}")


if __name__ == "__main__":
    main()
