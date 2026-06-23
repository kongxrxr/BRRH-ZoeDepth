#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"

export SAVE_DIR="${SAVE_DIR:-/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs12_after1_checkpoints}"
export LOG_FILE="${LOG_FILE:-${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs12_after1_4ep.log}"
export DONE_FILE="${DONE_FILE:-${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs12_after1_4ep.done}"

export EPOCHS="${EPOCHS:-4}"
export BS="${BS:-12}"
export WORKERS="${WORKERS:-0}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
export MEMORY_LOG_EVERY="${MEMORY_LOG_EVERY:-200}"

exec bash "${REPO_DIR}/scripts/run_kitti_scale024_strong_boundary.sh"
