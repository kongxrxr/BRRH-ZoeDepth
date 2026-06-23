#!/usr/bin/env python
from pathlib import Path


KITTI_S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti"


def collect_drives(repo_root):
    drives = set()
    for split in (
        repo_root / "train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        repo_root / "train_test_inputs/kitti_eigen_test_files_with_gt.txt",
    ):
        for line in split.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            drive_sync = parts[0].split("/")[1]
            drives.add(drive_sync.replace("_sync", ""))
    return sorted(drives)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "kitti_windows_download"
    out_dir.mkdir(exist_ok=True)

    urls = [f"{KITTI_S3}/data_depth_annotated.zip"]
    urls.extend(
        f"{KITTI_S3}/raw_data/{drive}/{drive}_sync.zip"
        for drive in collect_drives(repo_root)
    )

    url_file = out_dir / "kitti_urls.txt"
    ps_file = out_dir / "download_kitti.ps1"

    url_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
    ps_file.write_text(
        """$ErrorActionPreference = "Stop"
$OutDir = "D:\\KITTI_ZoeDepth_Zips"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Get-Content .\\kitti_urls.txt | ForEach-Object {
  $url = $_.Trim()
  if ($url.Length -eq 0) { return }
  $name = Split-Path $url -Leaf
  $out = Join-Path $OutDir $name
  if (Test-Path $out) {
    Write-Host "exists $out"
  } else {
    Write-Host "downloading $name"
    curl.exe -L --retry 20 --retry-delay 5 -o $out $url
  }
}
""",
        encoding="utf-8",
    )

    print(f"Wrote {len(urls)} URLs:")
    print(url_file)
    print(ps_file)
    print("Default Windows output folder in PowerShell script: D:\\KITTI_ZoeDepth_Zips")


if __name__ == "__main__":
    main()
