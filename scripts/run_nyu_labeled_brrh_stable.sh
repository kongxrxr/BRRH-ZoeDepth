#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
cd "${REPO_DIR}"

export SAVE_DIR="${SAVE_DIR:-/home/kxr/zoedepth_nyu_labeled_brrh_stable_256x512_bs4_workers4_5ep_checkpoints}"
export LOG_FILE="${LOG_FILE:-${REPO_DIR}/logs/nyu_labeled_brrh_stable_256x512_bs4_workers4_5ep.log}"
export DONE_FILE="${DONE_FILE:-${REPO_DIR}/logs/nyu_labeled_brrh_stable_256x512_bs4_workers4_5ep.done}"

export LR="${LR:-0.000001}"
export BOUNDARY_REFINE_SCALE="${BOUNDARY_REFINE_SCALE:-0.03}"
export USE_DISCONTINUITY_TEMPERATURE="${USE_DISCONTINUITY_TEMPERATURE:-0}"
export W_BOUNDARY_CLS="${W_BOUNDARY_CLS:-0.02}"
export W_BOUNDARY_BAND="${W_BOUNDARY_BAND:-0.2}"
export W_BOUNDARY_CONTRAST="${W_BOUNDARY_CONTRAST:-0.0}"
export W_EDGE="${W_EDGE:-0.0}"
export W_NONBOUNDARY_PRESERVE="${W_NONBOUNDARY_PRESERVE:-0.1}"
export USE_AMP="${USE_AMP:-0}"

bash scripts/run_nyu_labeled_brrh.sh
