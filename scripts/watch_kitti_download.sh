#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/kxr/ZoeDepth"
KITTI_ROOT="/home/kxr/shortcuts/datasets/kitti"
DOWNLOAD_LOG="${KITTI_ROOT}/download.log"
WATCHDOG_LOG="${KITTI_ROOT}/watchdog.log"
SESSION="kitti_download"

mkdir -p "${KITTI_ROOT}"

is_download_running() {
  tmux has-session -t "${SESSION}" 2>/dev/null
}

start_download() {
  tmux new-session -d -s "${SESSION}" \
    "bash -lc 'set -o pipefail; cd ${REPO_DIR}; echo \"--- restart \$(date) ---\" | tee -a ${DOWNLOAD_LOG}; python -u scripts/download_kitti_minimal.py 2>&1 | tee -a ${DOWNLOAD_LOG}; code=\${PIPESTATUS[0]}; echo \"--- exit code \${code} \$(date) ---\" | tee -a ${DOWNLOAD_LOG}; exit \${code}'"
}

while true; do
  {
    echo "===== $(date) ====="
    if is_download_running; then
      echo "download session: running"
    else
      echo "download session: stopped; restarting"
      start_download
    fi
    raw_count=$(find "${KITTI_ROOT}/raw" -type f 2>/dev/null | wc -l)
    gt_count=$(find "${KITTI_ROOT}/gts" -type f 2>/dev/null | wc -l)
    echo "raw files: ${raw_count}"
    echo "gt files: ${gt_count}"
    du -sh "${KITTI_ROOT}" 2>/dev/null || true
  } >> "${WATCHDOG_LOG}" 2>&1

  sleep 1800
done
