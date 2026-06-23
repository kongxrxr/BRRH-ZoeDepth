#!/usr/bin/env bash
set -uo pipefail

LOG_FILE="${LOG_FILE:?LOG_FILE is required}"
PATTERN="${PATTERN:?PATTERN is required}"
MARK_FILE="${MARK_FILE:-${LOG_FILE%.log}_stop_after_pattern.log}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"
STOP_DELAY="${STOP_DELAY:-8}"
SESSIONS="${SESSIONS:?SESSIONS is required}"

log() {
  echo "[$(date '+%F %T %Z')] $*" >> "${MARK_FILE}"
}

any_session_alive() {
  local session
  for session in ${SESSIONS}; do
    if tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${session}"; then
      return 0
    fi
  done
  return 1
}

log "armed: pattern=${PATTERN}; sessions=${SESSIONS}; log=${LOG_FILE}"

while any_session_alive; do
  if grep -q "${PATTERN}" "${LOG_FILE}"; then
    log "pattern detected; waiting ${STOP_DELAY}s before stopping sessions"
    sleep "${STOP_DELAY}"
    for session in ${SESSIONS}; do
      if tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${session}"; then
        log "stopping ${session}"
        tmux kill-session -t "${session}" 2>> "${MARK_FILE}" || true
      fi
    done
    log "stop complete"
    exit 0
  fi
  sleep "${CHECK_INTERVAL}"
done

log "all target sessions already stopped"
