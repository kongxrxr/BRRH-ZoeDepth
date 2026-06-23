#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_dav2gate_014_016}"
CHECK_INTERVAL="${CHECK_INTERVAL:-1800}"
WATCH_LOG="${WATCH_LOG:-${REPO_DIR}/logs/kitti_beit_dav2gate_scale014_016_watchdog.log}"
RESTART_COUNT_FILE="${RESTART_COUNT_FILE:-${REPO_DIR}/logs/kitti_beit_dav2gate_scale014_016_restart_count.txt}"
MAX_RESTARTS="${MAX_RESTARTS:-4}"
RUNNER="${RUNNER:-${REPO_DIR}/scripts/run_kitti_dav2gate_scale_finetune_014_016.sh}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
RESUME_CHECKPOINT_014="${RESUME_CHECKPOINT_014:-}"
RESUME_EPOCHS_014="${RESUME_EPOCHS_014:-}"

done_014="${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p14_skipnan_256x512_bs1_5ep.done"
done_016="${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p16_skipnan_256x512_bs1_5ep.done"

mkdir -p "$(dirname "${WATCH_LOG}")"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${WATCH_LOG}"
}

session_alive() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${TRAIN_SESSION}"
}

restart_count() {
  if [[ -f "${RESTART_COUNT_FILE}" ]]; then
    cat "${RESTART_COUNT_FILE}"
  else
    echo 0
  fi
}

all_done() {
  [[ -f "${done_014}" && -f "${done_016}" ]]
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
  log "restarting ${TRAIN_SESSION}; restart_count=${count}; runner=${RUNNER}; BASE_CHECKPOINT=${BASE_CHECKPOINT:-default}; RESUME_CHECKPOINT_014=${RESUME_CHECKPOINT_014:-none}; RESUME_EPOCHS_014=${RESUME_EPOCHS_014:-default}"
  tmux new-session -d -s "${TRAIN_SESSION}" "cd '${REPO_DIR}' && BASE_CHECKPOINT='${BASE_CHECKPOINT}' RESUME_CHECKPOINT_014='${RESUME_CHECKPOINT_014}' RESUME_EPOCHS_014='${RESUME_EPOCHS_014}' bash '${RUNNER}'"
}

log "watchdog online; session=${TRAIN_SESSION}; interval=${CHECK_INTERVAL}s"

while true; do
  if all_done; then
    log "both scale runs completed; watchdog exiting"
    break
  fi

  if session_alive; then
    log "training session alive"
  else
    log "training session missing and sweep not complete"
    start_training
  fi

  sleep "${CHECK_INTERVAL}"
done
