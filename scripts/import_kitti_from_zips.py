#!/usr/bin/env python
import argparse
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


def split_entries(split_files):
    raw = defaultdict(set)
    depths = set()
    for split_file in split_files:
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image_rel = parts[0].lstrip("/\\")
                depth_rel = parts[1].lstrip("/\\")
                date, drive_sync = image_rel.split("/")[:2]
                drive = drive_sync.replace("_sync", "")
                raw[(date, drive)].add(image_rel)
                if depth_rel.lower() != "none":
                    depths.add(depth_rel)
    return raw, depths


def copy_member(zf, member, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with zf.open(member) as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(out_path)


def import_raw(raw_needed, zip_dir, raw_root):
    total = len(raw_needed)
    for idx, ((_, drive), members) in enumerate(sorted(raw_needed.items()), start=1):
        zip_path = zip_dir / f"{drive}_sync.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"Missing raw zip: {zip_path}")
        missing = [m for m in members if not (raw_root / m).is_file()]
        if not missing:
            print(f"[{idx}/{total}] {drive}: already imported")
            continue
        print(f"[{idx}/{total}] {drive}: importing {len(missing)} frames")
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for count, member in enumerate(sorted(missing), start=1):
                if member not in names:
                    raise RuntimeError(f"{zip_path} missing {member}")
                copy_member(zf, member, raw_root / member)
                if count % 100 == 0:
                    print(f"  {count}/{len(missing)}")


def import_depths(depths_needed, zip_dir, gt_root):
    zip_path = zip_dir / "data_depth_annotated.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing depth zip: {zip_path}")
    internal_root = gt_root / ".data_depth_annotated"
    missing = [d for d in depths_needed if not (gt_root / d).is_file()]
    if not missing:
        print("depth annotations: already imported")
        return
    print(f"depth annotations: importing {len(missing)} maps")
    imported_members = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for count, rel in enumerate(sorted(missing), start=1):
            train_name = f"train/{rel}"
            val_name = f"val/{rel}"
            if train_name in names:
                member = train_name
            elif val_name in names:
                member = val_name
            else:
                raise RuntimeError(f"{zip_path} missing depth {rel}")
            copy_member(zf, member, internal_root / member)
            imported_members.append(member)
            if count % 1000 == 0:
                print(f"  {count}/{len(missing)}")

    for member in imported_members:
        subset, drive = member.split("/", 2)[:2]
        link = gt_root / drive
        target = internal_root / subset / drive
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Import ZoeDepth KITTI files from locally downloaded zips.")
    parser.add_argument("--zip-dir", required=True, help="Directory containing *_sync.zip and data_depth_annotated.zip")
    parser.add_argument("--raw-root", default="~/shortcuts/datasets/kitti/raw")
    parser.add_argument("--gt-root", default="~/shortcuts/datasets/kitti/gts")
    parser.add_argument("--train-split", default=str(repo_root / "train_test_inputs/kitti_eigen_train_files_with_gt.txt"))
    parser.add_argument("--eval-split", default=str(repo_root / "train_test_inputs/kitti_eigen_test_files_with_gt.txt"))
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    split_files = [Path(args.train_split).expanduser().resolve(), Path(args.eval_split).expanduser().resolve()]

    if not zip_dir.is_dir():
        raise FileNotFoundError(zip_dir)

    raw_needed, depths_needed = split_entries(split_files)
    print(f"Need {sum(len(v) for v in raw_needed.values())} raw images from {len(raw_needed)} zips")
    print(f"Need {len(depths_needed)} depth maps")
    import_raw(raw_needed, zip_dir, raw_root)
    import_depths(depths_needed, zip_dir, gt_root)
    print("Import complete.")


if __name__ == "__main__":
    main()
