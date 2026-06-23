#!/usr/bin/env python
import argparse
from pathlib import Path


def human(nbytes):
    units = ["B", "K", "M", "G", "T"]
    size = float(nbytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024


def collect_targets(zip_dir, kitti_root):
    targets = []

    for duplicate in zip_dir.glob("*(1).zip"):
        base = zip_dir / duplicate.name.replace("(1)", "")
        if base.exists():
            targets.append(duplicate)

    targets.extend(zip_dir.glob("*calib*.zip"))

    downloads = kitti_root / ".downloads"
    if downloads.exists():
        targets.append(downloads)

    targets.extend(kitti_root.rglob("*.part"))

    return sorted(set(targets))


def path_size(path):
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def remove_path(path):
    if path.is_dir() and not path.is_symlink():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    else:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Remove KITTI files not needed by ZoeDepth.")
    parser.add_argument("--zip-dir", default="/mnt/c/Users/kxr/Documents/Codex/2026-05-28/wsl-kitti-zoedepth-kitti")
    parser.add_argument("--kitti-root", default="/home/kxr/shortcuts/datasets/kitti")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir)
    kitti_root = Path(args.kitti_root)
    targets = collect_targets(zip_dir, kitti_root)
    total = sum(path_size(path) for path in targets if path.exists())

    print(f"Targets: {len(targets)}")
    print(f"Reclaimable: {human(total)}")
    for path in targets:
        if path.exists():
            print(f"{human(path_size(path)):>8}  {path}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete.")
        return

    for path in targets:
        if path.exists():
            remove_path(path)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
