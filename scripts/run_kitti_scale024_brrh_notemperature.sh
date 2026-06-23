#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p24_strong_skipnan_256x512_bs4_workers8_checkpoints/ZoeDepthv1_10-Jun_20-04-8e76f920e3fe_latest.pt}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"

cd "${REPO_DIR}"

export SAVE_DIR="${SAVE_DIR:-/home/kxr/zoedepth_kitti_brrh_notemperature_scale0p24_256x512_bs4_workers8_5ep_checkpoints}"
export LOG_FILE="${LOG_FILE:-${REPO_DIR}/logs/kitti_brrh_notemperature_scale0p24_256x512_bs4_workers8_5ep.log}"
export DONE_FILE="${DONE_FILE:-${REPO_DIR}/logs/kitti_brrh_notemperature_scale0p24_256x512_bs4_workers8_5ep.done}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export SKIP_NAN_BATCHES=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

export EPOCHS="${EPOCHS:-5}"
export BS="${BS:-4}"
export WORKERS="${WORKERS:-8}"
export LR="${LR:-0.00002}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-999999}"
export LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-999}"
export PRINT_LOSSES="${PRINT_LOSSES:-0}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-0}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export MEMORY_LOG_EVERY="${MEMORY_LOG_EVERY:-200}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
export IMG_SIZE=256,512
export MIDAS_MODEL_TYPE=DPT_BEiT_L_384
export TRAIN_MIDAS=0
export CHECKPOINT="${CHECKPOINT:-${BASE_CHECKPOINT}}"

export W_EDGE=0.02
export EDGE_IMAGE_WEIGHT=0.7
export EDGE_DEPTH_WEIGHT=0.3
export W_GRAD=0

export USE_BOUNDARY_REFINE=1
export BOUNDARY_REFINE_CHANNELS=32
export BOUNDARY_REFINE_SCALE="${BOUNDARY_REFINE_SCALE:-0.08}"
export BOUNDARY_REFINE_MODE=log_residual
export BOUNDARY_REFINE_USE_DA_PRIOR=1

export USE_DISCONTINUITY_BRANCH=1
export DISCONTINUITY_CHANNELS=32
export USE_DISCONTINUITY_TEMPERATURE=0
export DISCONTINUITY_TEMPERATURE_SCALE=1.0

export W_BOUNDARY_CLS=0.05
export W_BOUNDARY_BAND=0.5
export BOUNDARY_BAND_RADIUS=3
export BOUNDARY_LOG_GRAD_THRESHOLD=0.15
export BOUNDARY_TARGET_ALPHA=10.0
export BOUNDARY_POS_WEIGHT=4.0
export W_BOUNDARY_CONTRAST=0.15
export W_BOUNDARY_ALIGN=0.0
export BOUNDARY_ALIGN_MAX_DISTANCE=6.0
export BOUNDARY_ALIGN_PRED_WEIGHT=0.5
export BOUNDARY_ALIGN_COVERAGE_WEIGHT=0.5
export W_NONBOUNDARY_SMOOTH=0.0
export W_NONBOUNDARY_PRESERVE=0.05
export NONBOUNDARY_PRESERVE_RADIUS=5

export USE_FROZEN_DA_PRIOR=1
export FROZEN_DA_MODEL="${FROZEN_DA_MODEL_PATH}"
export FROZEN_DA_FEATURE_CHANNELS=8
export FROZEN_DA_INPUT_SIZE=384
export FROZEN_DA_FUSION_SCALE=0.12
export USE_FROZEN_DA_BOUNDARY_GATE=0
export FROZEN_DA_MIN_GATE=0.05

if [[ -f "${DONE_FILE}" ]]; then
  echo "===== SKIP BRRH no-temperature scale0.24; done marker exists: ${DONE_FILE} ====="
  exit 0
fi

echo "===== START BRRH no-temperature scale0.24 epochs=${EPOCHS} checkpoint=${CHECKPOINT} $(date '+%F %T %Z') ====="
bash scripts/run_kitti_train_once.sh
echo "===== DONE BRRH no-temperature scale0.24 $(date '+%F %T %Z') ====="
