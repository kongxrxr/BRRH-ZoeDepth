#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_scale024_strong}"
EVAL_SESSION="${EVAL_SESSION:-zoe_kitti_scale024_strong_eval}"
SOURCE_LOG="${SOURCE_LOG:-${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs8_safe_5ep.log}"
SOURCE_CKPT_DIR="${SOURCE_CKPT_DIR:-/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs8_safe_checkpoints}"
TARGET_CKPT_DIR="${TARGET_CKPT_DIR:-/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs12_after1_checkpoints}"
TARGET_METRICS="${TARGET_METRICS:-${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p24_strong_bs12_after1_hard_boundary_metrics.json}"
MARK_FILE="${MARK_FILE:-${REPO_DIR}/logs/kitti_switch_after_epoch1_to_bs12.log}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${MARK_FILE}"
}

session_alive() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "$1"
}

latest_ckpt() {
  ls -t "${SOURCE_CKPT_DIR}"/*_latest.pt 2>/dev/null | head -n 1 || true
}

cd "${REPO_DIR}"
mkdir -p logs

log "armed: wait for Epoch: 2/5 in ${SOURCE_LOG}"
while ! grep -q "Epoch: 2/5" "${SOURCE_LOG}" 2>/dev/null; do
  if ! session_alive "${TRAIN_SESSION}"; then
    log "training session stopped before epoch 2; exiting"
    exit 1
  fi
  sleep "${CHECK_INTERVAL}"
done

detected_at="$(date +%s)"
log "Epoch 2 detected; waiting for a fresh latest checkpoint after ${detected_at}"

fresh_ckpt=""
while [[ -z "${fresh_ckpt}" ]]; do
  ckpt="$(latest_ckpt)"
  if [[ -n "${ckpt}" ]]; then
    mtime="$(stat -c %Y "${ckpt}")"
    if [[ "${mtime}" -ge "${detected_at}" ]]; then
      fresh_ckpt="${ckpt}"
      break
    fi
  fi
  if ! session_alive "${TRAIN_SESSION}"; then
    log "training session stopped before fresh checkpoint; exiting"
    exit 1
  fi
  sleep "${CHECK_INTERVAL}"
done

log "fresh checkpoint ready: ${fresh_ckpt}"
log "stopping ${TRAIN_SESSION} and restarting BS=12 for remaining 4 epochs"
tmux kill-session -t "${TRAIN_SESSION}" 2>> "${MARK_FILE}" || true
sleep 3
tmux new-session -d -s "${TRAIN_SESSION}" "cd '${REPO_DIR}' && CHECKPOINT='${fresh_ckpt}' bash scripts/run_kitti_scale024_strong_boundary_bs12_after1.sh"

if session_alive "${EVAL_SESSION}"; then
  tmux kill-session -t "${EVAL_SESSION}" 2>> "${MARK_FILE}" || true
fi
tmux new-session -d -s "${EVAL_SESSION}" "cd '${REPO_DIR}' && CKPT_DIR='${TARGET_CKPT_DIR}' OUT='${TARGET_METRICS}' bash scripts/eval_kitti_scale024_strong_after_train.sh"
log "BS=12 training and final eval watcher started"
