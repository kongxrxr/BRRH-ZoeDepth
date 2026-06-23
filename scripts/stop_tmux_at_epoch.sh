#!/usr/bin/env bash
set -uo pipefail

LOG_FILE="${LOG_FILE:?LOG_FILE is required}"
TRAIN_SESSION="${TRAIN_SESSION:?TRAIN_SESSION is required}"
EPOCH_PATTERN="${EPOCH_PATTERN:?EPOCH_PATTERN is required}"
MARK_FILE="${MARK_FILE:-${LOG_FILE%.log}_stop_watcher.log}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

log() {
  echo "[$(date '+%F %T %Z')] $*" >> "${MARK_FILE}"
}

session_alive() {
  tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${TRAIN_SESSION}"
}

log "stop watcher armed: waiting for ${EPOCH_PATTERN}; training session=${TRAIN_SESSION}"

while true; do
  if grep -q "${EPOCH_PATTERN}" "${LOG_FILE}"; then
    log "${EPOCH_PATTERN} detected; stopping ${TRAIN_SESSION}"
    tmux kill-session -t "${TRAIN_SESSION}" 2>> "${MARK_FILE}" || true
    break
  fi

  if ! session_alive; then
    log "training session already stopped"
    break
  fi

  sleep "${CHECK_INTERVAL}"
done
