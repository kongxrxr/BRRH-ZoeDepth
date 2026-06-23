#!/usr/bin/env python
import argparse
import io
import time
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


KITTI_S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"


def urlopen_with_retry(req, timeout, tries=8):
    last_error = None
    for attempt in range(1, tries + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as exc:
            last_error = exc
            wait = min(30, 2 ** attempt)
            print(f"request failed ({attempt}/{tries}): {exc}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise last_error


class RangeHTTPFile(io.RawIOBase):
    def __init__(self, url, block_size=1024 * 1024):
        self.url = url
        self.block_size = block_size
        self.pos = 0
        self.cache = {}
        req = urllib.request.Request(url, method="HEAD")
        with urlopen_with_retry(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        chunks = []
        remaining = n
        while remaining:
            block_index = self.pos // self.block_size
            block_offset = self.pos % self.block_size
            block = self._get_block(block_index)
            take = min(remaining, len(block) - block_offset)
            chunks.append(block[block_offset:block_offset + take])
            self.pos += take
            remaining -= take
        return b"".join(chunks)

    def _get_block(self, block_index):
        if block_index in self.cache:
            return self.cache[block_index]
        start = block_index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        data = None
        last_error = None
        for attempt in range(1, 9):
            try:
                with urlopen_with_retry(req, timeout=120) as resp:
                    data = resp.read()
                expected = end - start + 1
                if len(data) != expected:
                    raise IOError(f"Short range read: got {len(data)} bytes, expected {expected}")
                break
            except Exception as exc:
                last_error = exc
                wait = min(30, 2 ** attempt)
                print(f"range read failed ({attempt}/8): {exc}; retrying in {wait}s", flush=True)
                time.sleep(wait)
        if data is None:
            raise last_error
        self.cache[block_index] = data
        if len(self.cache) > 64:
            self.cache.pop(next(iter(self.cache)))
        return data


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
                depths.add(depth_rel)
    return raw, depths


def extract_members_from_url(url, members, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    members = set(members)
    extracted = 0
    missing = []
    with zipfile.ZipFile(RangeHTTPFile(url)) as zf:
        names = set(zf.namelist())
        for member in sorted(members):
            if member in names:
                out_path = output_root / member
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                with zf.open(member) as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_path.replace(out_path)
                extracted += 1
                if extracted % 50 == 0:
                    print(f"  extracted {extracted}/{len(members)}", flush=True)
            else:
                missing.append(member)
    return extracted, missing


def download_raw(raw_needed, raw_root):
    total_drives = len(raw_needed)
    for idx, ((date, drive), members) in enumerate(sorted(raw_needed.items()), start=1):
        zip_name = f"{drive}_sync.zip"
        url = f"{KITTI_S3}/raw_data/{drive}/{zip_name}"
        print(f"\n[{idx}/{total_drives}] raw {drive}: {len(members)} frames", flush=True)
        missing_after_extract = [m for m in members if not (raw_root / m).is_file()]
        if not missing_after_extract:
            print("already extracted", flush=True)
            continue
        extracted, missing = extract_members_from_url(url, missing_after_extract, raw_root)
        print(f"extracted {extracted} raw images", flush=True)
        if missing:
            raise RuntimeError(f"{url} missing {len(missing)} expected images, first: {missing[:3]}")


def extract_depths(depths_needed, gt_root):
    url = f"{KITTI_S3}/data_depth_annotated.zip"
    internal_root = gt_root / ".data_depth_annotated"
    missing_depths = [d for d in depths_needed if not (gt_root / d).is_file()]
    if not missing_depths:
        print("\ndepth annotations already extracted", flush=True)
        return

    print(f"\ndepth annotations: {len(missing_depths)} files", flush=True)

    members = []
    with zipfile.ZipFile(RangeHTTPFile(url)) as zf:
        names = set(zf.namelist())
        for rel in sorted(missing_depths):
            train_name = f"train/{rel}"
            val_name = f"val/{rel}"
            if train_name in names:
                members.append(train_name)
            elif val_name in names:
                members.append(val_name)
            else:
                raise RuntimeError(f"Depth file not found in annotation zip: {rel}")

        extracted = 0
        for member in members:
            out_path = internal_root / member
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with zf.open(member) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp_path.replace(out_path)
            extracted += 1
            if extracted % 500 == 0:
                print(f"  extracted {extracted}/{len(members)} depth maps", flush=True)
    print(f"extracted {extracted} depth maps", flush=True)

    for member in members:
        subset, drive = member.split("/", 2)[:2]
        link = gt_root / drive
        target = internal_root / subset / drive
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(target, target_is_directory=True)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download only ZoeDepth KITTI Eigen split files.")
    parser.add_argument("--raw-root", default="~/shortcuts/datasets/kitti/raw")
    parser.add_argument("--gt-root", default="~/shortcuts/datasets/kitti/gts")
    parser.add_argument("--train-split", default=str(repo_root / "train_test_inputs/kitti_eigen_train_files_with_gt.txt"))
    parser.add_argument("--eval-split", default=str(repo_root / "train_test_inputs/kitti_eigen_test_files_with_gt.txt"))
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser().resolve()
    split_files = [Path(args.train_split).expanduser().resolve(), Path(args.eval_split).expanduser().resolve()]

    for split_file in split_files:
        if not split_file.is_file():
            raise FileNotFoundError(split_file)

    raw_needed, depths_needed = split_entries(split_files)
    raw_frames = sum(len(v) for v in raw_needed.values())
    print(f"Need {raw_frames} raw images from {len(raw_needed)} drives", flush=True)
    print(f"Need {len(depths_needed)} depth maps", flush=True)
    print(f"raw root: {raw_root}", flush=True)
    print(f"gt root:  {gt_root}", flush=True)
    print("Using HTTP Range reads: zip files are not stored locally.", flush=True)

    raw_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    download_raw(raw_needed, raw_root)
    extract_depths(depths_needed, gt_root)
    print("\nKITTI minimal download complete.", flush=True)


if __name__ == "__main__":
    main()
