#!/usr/bin/env bash
set -euo pipefail

export CKPT_DIR="${CKPT_DIR:-/home/kxr/zoedepth_nyu_labeled_brrh_stable_256x512_bs4_workers4_5ep_checkpoints}"
export OUT="${OUT:-/home/kxr/ZoeDepth/logs/nyu_labeled_brrh_stable_val654_hard_boundary_metrics.json}"

bash /home/kxr/ZoeDepth/scripts/eval_nyu_labeled_brrh_after_train.sh
