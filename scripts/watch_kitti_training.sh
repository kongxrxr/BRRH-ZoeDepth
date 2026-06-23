#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_train}"
WATCH_SESSION="${WATCH_SESSION:-zoe_kitti_watch}"
CHECK_INTERVAL="${CHECK_INTERVAL:-1800}"
CHECK_MODE="${CHECK_MODE:-interval}"
EPOCHS="${EPOCHS:-5}"
BS="${BS:-4}"
WORKERS="${WORKERS:-8}"
LR="${LR:-}"
IMG_SIZE="${IMG_SIZE:-}"
TRAIN_MIDAS="${TRAIN_MIDAS:-}"
MIDAS_MODEL_TYPE="${MIDAS_MODEL_TYPE:-}"
CHECKPOINT="${CHECKPOINT:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DISABLED="${WANDB_DISABLED:-false}"
SAVE_DIR="${SAVE_DIR:-/home/kxr/zoedepth_kitti_checkpoints}"
W_EDGE="${W_EDGE:-}"
EDGE_IMAGE_WEIGHT="${EDGE_IMAGE_WEIGHT:-}"
EDGE_DEPTH_WEIGHT="${EDGE_DEPTH_WEIGHT:-}"
W_GRAD="${W_GRAD:-}"
USE_BOUNDARY_REFINE="${USE_BOUNDARY_REFINE:-}"
BOUNDARY_REFINE_CHANNELS="${BOUNDARY_REFINE_CHANNELS:-}"
BOUNDARY_REFINE_SCALE="${BOUNDARY_REFINE_SCALE:-}"
USE_DISCONTINUITY_BRANCH="${USE_DISCONTINUITY_BRANCH:-}"
DISCONTINUITY_CHANNELS="${DISCONTINUITY_CHANNELS:-}"
USE_DISCONTINUITY_TEMPERATURE="${USE_DISCONTINUITY_TEMPERATURE:-}"
DISCONTINUITY_TEMPERATURE_SCALE="${DISCONTINUITY_TEMPERATURE_SCALE:-}"
W_BOUNDARY_CLS="${W_BOUNDARY_CLS:-}"
BOUNDARY_LOG_GRAD_THRESHOLD="${BOUNDARY_LOG_GRAD_THRESHOLD:-}"
BOUNDARY_TARGET_ALPHA="${BOUNDARY_TARGET_ALPHA:-}"
BOUNDARY_POS_WEIGHT="${BOUNDARY_POS_WEIGHT:-}"
W_BOUNDARY_CONTRAST="${W_BOUNDARY_CONTRAST:-}"
W_BOUNDARY_ALIGN="${W_BOUNDARY_ALIGN:-}"
BOUNDARY_ALIGN_MAX_DISTANCE="${BOUNDARY_ALIGN_MAX_DISTANCE:-}"
BOUNDARY_ALIGN_PRED_WEIGHT="${BOUNDARY_ALIGN_PRED_WEIGHT:-}"
BOUNDARY_ALIGN_COVERAGE_WEIGHT="${BOUNDARY_ALIGN_COVERAGE_WEIGHT:-}"
W_NONBOUNDARY_SMOOTH="${W_NONBOUNDARY_SMOOTH:-}"
USE_FROZEN_DA_PRIOR="${USE_FROZEN_DA_PRIOR:-}"
FROZEN_DA_MODEL="${FROZEN_DA_MODEL:-}"
FROZEN_DA_FEATURE_CHANNELS="${FROZEN_DA_FEATURE_CHANNELS:-}"
FROZEN_DA_INPUT_SIZE="${FROZEN_DA_INPUT_SIZE:-}"
FROZEN_DA_FUSION_SCALE="${FROZEN_DA_FUSION_SCALE:-}"
USE_FROZEN_DA_BOUNDARY_GATE="${USE_FROZEN_DA_BOUNDARY_GATE:-}"
FROZEN_DA_MIN_GATE="${FROZEN_DA_MIN_GATE:-}"
LOG_FILE="${LOG_FILE:-${REPO_DIR}/logs/kitti_train_5ep_bs4.log}"
WATCH_LOG="${WATCH_LOG:-${REPO_DIR}/logs/kitti_train_watchdog.log}"
DONE_FILE="${DONE_FILE:-${REPO_DIR}/logs/kitti_train_5ep_bs4.done}"
RUNNER="${RUNNER:-${REPO_DIR}/scripts/run_kitti_train_once.sh}"
RESTART_COUNT_FILE="${RESTART_COUNT_FILE:-${REPO_DIR}/logs/kitti_train_restart_count.txt}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
CONDA_PYTHON="${CONDA_PYTHON:-/home/kxr/miniconda3/envs/zoe/bin/python}"

mkdir -p "$(dirname "${WATCH_LOG}")"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${WATCH_LOG}"
}

session_alive() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${TRAIN_SESSION}"
}

checkpoint_complete() {
  "${CONDA_PYTHON}" - "${SAVE_DIR}" "${EPOCHS}" <<'PY'
import glob
import os
import sys
import torch

save_dir = sys.argv[1]
epochs = int(sys.argv[2])
matches = glob.glob(os.path.join(save_dir, "*_latest.pt"))
if not matches:
    sys.exit(1)
latest = max(matches, key=os.path.getmtime)
try:
    ckpt = torch.load(latest, map_location="cpu")
except Exception:
    sys.exit(1)
epoch = ckpt.get("epoch", -1)
sys.exit(0 if epoch >= epochs - 1 else 1)
PY
}

restart_count() {
  if [[ -f "${RESTART_COUNT_FILE}" ]]; then
    cat "${RESTART_COUNT_FILE}"
  else
    echo 0
  fi
}

start_training() {
  local count
  count="$(restart_count)"
  if [[ "${count}" -ge "${MAX_RESTARTS}" ]]; then
    log "restart limit reached (${count}/${MAX_RESTARTS}); leaving stopped for inspection"
    return 1
  fi

  count=$((count + 1))
  echo "${count}" > "${RESTART_COUNT_FILE}"
  rm -f "${DONE_FILE}"

  log "starting ${TRAIN_SESSION}; restart_count=${count}; BS=${BS}; EPOCHS=${EPOCHS}; LR=${LR:-default}; IMG_SIZE=${IMG_SIZE:-default}; TRAIN_MIDAS=${TRAIN_MIDAS:-default}; MIDAS_MODEL_TYPE=${MIDAS_MODEL_TYPE:-default}; CHECKPOINT=${CHECKPOINT:-none}; W_EDGE=${W_EDGE:-0}; USE_BOUNDARY_REFINE=${USE_BOUNDARY_REFINE:-0}; USE_DISCONTINUITY_BRANCH=${USE_DISCONTINUITY_BRANCH:-0}; W_BOUNDARY_ALIGN=${W_BOUNDARY_ALIGN:-0}; USE_FROZEN_DA_PRIOR=${USE_FROZEN_DA_PRIOR:-0}; USE_FROZEN_DA_BOUNDARY_GATE=${USE_FROZEN_DA_BOUNDARY_GATE:-0}"
  tmux new-session -d -s "${TRAIN_SESSION}" \
    "env REPO_DIR='${REPO_DIR}' EPOCHS='${EPOCHS}' BS='${BS}' WORKERS='${WORKERS}' LR='${LR}' IMG_SIZE='${IMG_SIZE}' TRAIN_MIDAS='${TRAIN_MIDAS}' MIDAS_MODEL_TYPE='${MIDAS_MODEL_TYPE}' CHECKPOINT='${CHECKPOINT}' WANDB_MODE='${WANDB_MODE}' WANDB_DISABLED='${WANDB_DISABLED}' SAVE_DIR='${SAVE_DIR}' W_EDGE='${W_EDGE}' EDGE_IMAGE_WEIGHT='${EDGE_IMAGE_WEIGHT}' EDGE_DEPTH_WEIGHT='${EDGE_DEPTH_WEIGHT}' W_GRAD='${W_GRAD}' USE_BOUNDARY_REFINE='${USE_BOUNDARY_REFINE}' BOUNDARY_REFINE_CHANNELS='${BOUNDARY_REFINE_CHANNELS}' BOUNDARY_REFINE_SCALE='${BOUNDARY_REFINE_SCALE}' USE_DISCONTINUITY_BRANCH='${USE_DISCONTINUITY_BRANCH}' DISCONTINUITY_CHANNELS='${DISCONTINUITY_CHANNELS}' USE_DISCONTINUITY_TEMPERATURE='${USE_DISCONTINUITY_TEMPERATURE}' DISCONTINUITY_TEMPERATURE_SCALE='${DISCONTINUITY_TEMPERATURE_SCALE}' W_BOUNDARY_CLS='${W_BOUNDARY_CLS}' BOUNDARY_LOG_GRAD_THRESHOLD='${BOUNDARY_LOG_GRAD_THRESHOLD}' BOUNDARY_TARGET_ALPHA='${BOUNDARY_TARGET_ALPHA}' BOUNDARY_POS_WEIGHT='${BOUNDARY_POS_WEIGHT}' W_BOUNDARY_CONTRAST='${W_BOUNDARY_CONTRAST}' W_BOUNDARY_ALIGN='${W_BOUNDARY_ALIGN}' BOUNDARY_ALIGN_MAX_DISTANCE='${BOUNDARY_ALIGN_MAX_DISTANCE}' BOUNDARY_ALIGN_PRED_WEIGHT='${BOUNDARY_ALIGN_PRED_WEIGHT}' BOUNDARY_ALIGN_COVERAGE_WEIGHT='${BOUNDARY_ALIGN_COVERAGE_WEIGHT}' W_NONBOUNDARY_SMOOTH='${W_NONBOUNDARY_SMOOTH}' USE_FROZEN_DA_PRIOR='${USE_FROZEN_DA_PRIOR}' FROZEN_DA_MODEL='${FROZEN_DA_MODEL}' FROZEN_DA_FEATURE_CHANNELS='${FROZEN_DA_FEATURE_CHANNELS}' FROZEN_DA_INPUT_SIZE='${FROZEN_DA_INPUT_SIZE}' FROZEN_DA_FUSION_SCALE='${FROZEN_DA_FUSION_SCALE}' USE_FROZEN_DA_BOUNDARY_GATE='${USE_FROZEN_DA_BOUNDARY_GATE}' FROZEN_DA_MIN_GATE='${FROZEN_DA_MIN_GATE}' LOG_FILE='${LOG_FILE}' DONE_FILE='${DONE_FILE}' bash '${RUNNER}'"
}

log "watchdog online; checking every ${CHECK_INTERVAL}s; training session=${TRAIN_SESSION}"

sleep_until_next_hour() {
  local now next wait_seconds
  now="$(date +%s)"
  next=$(( (now / 3600 + 1) * 3600 ))
  wait_seconds=$((next - now))
  if [[ "${wait_seconds}" -lt 1 ]]; then
    wait_seconds="${CHECK_INTERVAL}"
  fi
  log "next check at $(date -d "@${next}" '+%F %T %Z')"
  sleep "${wait_seconds}"
}

while true; do
  if session_alive; then
    log "training session alive"
  elif [[ -f "${DONE_FILE}" ]]; then
    log "done marker exists; watchdog exiting"
    break
  elif checkpoint_complete; then
    touch "${DONE_FILE}"
    log "latest checkpoint reached epoch ${EPOCHS}; watchdog exiting"
    break
  else
    log "training session missing and not complete; restarting"
    start_training
  fi

  if [[ "${CHECK_MODE}" == "hourly" ]]; then
    sleep_until_next_hour
  else
    sleep "${CHECK_INTERVAL}"
  fi
done
