#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p12_skipnan_256x512_bs1_checkpoints/ZoeDepthv1_06-Jun_10-57-53499d0e07cb_latest.pt}"
RESUME_CHECKPOINT_014="${RESUME_CHECKPOINT_014:-}"
RESUME_EPOCHS_014="${RESUME_EPOCHS_014:-}"
DEFAULT_EPOCHS="${EPOCHS:-5}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"

cd "${REPO_DIR}"

for scale in 0.14 0.16; do
  label="${scale/./p}"
  export SAVE_DIR="/home/kxr/zoedepth_kitti_beit_dav2gate_scale${label}_skipnan_256x512_bs1_checkpoints"
  export LOG_FILE="${REPO_DIR}/logs/kitti_beit_dav2gate_scale${label}_skipnan_256x512_bs1_5ep.log"
  export DONE_FILE="${REPO_DIR}/logs/kitti_beit_dav2gate_scale${label}_skipnan_256x512_bs1_5ep.done"
  export WANDB_DISABLED=true
  export WANDB_MODE=disabled
  export SKIP_NAN_BATCHES=1

  scale_epochs="${DEFAULT_EPOCHS}"
  scale_checkpoint="${BASE_CHECKPOINT}"
  if [[ "${scale}" == "0.14" ]]; then
    if [[ -n "${RESUME_EPOCHS_014}" ]]; then
      scale_epochs="${RESUME_EPOCHS_014}"
    fi
    if [[ -n "${RESUME_CHECKPOINT_014}" ]]; then
      scale_checkpoint="${RESUME_CHECKPOINT_014}"
    fi
  fi

  export EPOCHS="${scale_epochs}"
  export BS="${BS:-1}"
  export WORKERS="${WORKERS:-4}"
  export LR="${LR:-0.00001}"
  export IMG_SIZE=256,512
  export MIDAS_MODEL_TYPE=DPT_BEiT_L_384
  export TRAIN_MIDAS=0
  export CHECKPOINT="${scale_checkpoint}"

  export W_EDGE=0.02
  export EDGE_IMAGE_WEIGHT=0.7
  export EDGE_DEPTH_WEIGHT=0.3
  export W_GRAD=0

  export USE_BOUNDARY_REFINE=1
  export BOUNDARY_REFINE_CHANNELS=32
  export BOUNDARY_REFINE_SCALE=0.25
  export USE_DISCONTINUITY_BRANCH=1
  export DISCONTINUITY_CHANNELS=32
  export USE_DISCONTINUITY_TEMPERATURE=1
  export DISCONTINUITY_TEMPERATURE_SCALE=0.5

  export W_BOUNDARY_CLS=0.05
  export BOUNDARY_LOG_GRAD_THRESHOLD=0.15
  export BOUNDARY_TARGET_ALPHA=10.0
  export BOUNDARY_POS_WEIGHT=4.0
  export W_BOUNDARY_CONTRAST=0.03
  export W_BOUNDARY_ALIGN=0.01
  export BOUNDARY_ALIGN_MAX_DISTANCE=6.0
  export BOUNDARY_ALIGN_PRED_WEIGHT=0.5
  export BOUNDARY_ALIGN_COVERAGE_WEIGHT=0.5
  export W_NONBOUNDARY_SMOOTH=0.005

  export USE_FROZEN_DA_PRIOR=1
  export FROZEN_DA_MODEL="${FROZEN_DA_MODEL_PATH}"
  export FROZEN_DA_FEATURE_CHANNELS=8
  export FROZEN_DA_INPUT_SIZE=384
  export FROZEN_DA_FUSION_SCALE="${scale}"
  export USE_FROZEN_DA_BOUNDARY_GATE=1
  export FROZEN_DA_MIN_GATE=0.05

  if [[ -f "${DONE_FILE}" ]]; then
    echo "===== SKIP DA boundary-gate scale ${scale}; done marker exists: ${DONE_FILE} ====="
    continue
  fi

  echo "===== START DA boundary-gate scale ${scale} epochs=${EPOCHS} checkpoint=${CHECKPOINT} $(date '+%F %T %Z') ====="
  bash scripts/run_kitti_train_once.sh
  echo "===== DONE DA boundary-gate scale ${scale} $(date '+%F %T %Z') ====="
done
