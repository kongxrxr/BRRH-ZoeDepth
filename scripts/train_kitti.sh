#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

KITTI_RAW_ROOT="${KITTI_RAW_ROOT:-${HOME}/shortcuts/datasets/kitti/raw}"
KITTI_GT_ROOT="${KITTI_GT_ROOT:-${HOME}/shortcuts/datasets/kitti/gts}"
SAVE_DIR="${SAVE_DIR:-${HOME}/zoedepth_kitti_checkpoints}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DISABLED="${WANDB_DISABLED:-false}"
EPOCHS="${EPOCHS:-5}"
BS="${BS:-4}"
WORKERS="${WORKERS:-8}"
LR="${LR:-0.000161}"
IMG_SIZE="${IMG_SIZE:-}"
TRAIN_MIDAS="${TRAIN_MIDAS:-}"
MIDAS_MODEL_TYPE="${MIDAS_MODEL_TYPE:-}"
CHECKPOINT="${CHECKPOINT:-}"
PRETRAINED_RESOURCE="${PRETRAINED_RESOURCE:-}"
W_EDGE="${W_EDGE:-}"
EDGE_IMAGE_WEIGHT="${EDGE_IMAGE_WEIGHT:-}"
EDGE_DEPTH_WEIGHT="${EDGE_DEPTH_WEIGHT:-}"
W_GRAD="${W_GRAD:-}"
USE_BOUNDARY_REFINE="${USE_BOUNDARY_REFINE:-}"
BOUNDARY_REFINE_CHANNELS="${BOUNDARY_REFINE_CHANNELS:-}"
BOUNDARY_REFINE_SCALE="${BOUNDARY_REFINE_SCALE:-}"
BOUNDARY_REFINE_MODE="${BOUNDARY_REFINE_MODE:-}"
BOUNDARY_REFINE_USE_DA_PRIOR="${BOUNDARY_REFINE_USE_DA_PRIOR:-}"
USE_DISCONTINUITY_BRANCH="${USE_DISCONTINUITY_BRANCH:-}"
DISCONTINUITY_CHANNELS="${DISCONTINUITY_CHANNELS:-}"
USE_DISCONTINUITY_TEMPERATURE="${USE_DISCONTINUITY_TEMPERATURE:-}"
DISCONTINUITY_TEMPERATURE_SCALE="${DISCONTINUITY_TEMPERATURE_SCALE:-}"
W_BOUNDARY_CLS="${W_BOUNDARY_CLS:-}"
W_BOUNDARY_BAND="${W_BOUNDARY_BAND:-}"
BOUNDARY_BAND_RADIUS="${BOUNDARY_BAND_RADIUS:-}"
BOUNDARY_LOG_GRAD_THRESHOLD="${BOUNDARY_LOG_GRAD_THRESHOLD:-}"
BOUNDARY_TARGET_ALPHA="${BOUNDARY_TARGET_ALPHA:-}"
BOUNDARY_POS_WEIGHT="${BOUNDARY_POS_WEIGHT:-}"
W_BOUNDARY_CONTRAST="${W_BOUNDARY_CONTRAST:-}"
W_BOUNDARY_ALIGN="${W_BOUNDARY_ALIGN:-}"
BOUNDARY_ALIGN_MAX_DISTANCE="${BOUNDARY_ALIGN_MAX_DISTANCE:-}"
BOUNDARY_ALIGN_PRED_WEIGHT="${BOUNDARY_ALIGN_PRED_WEIGHT:-}"
BOUNDARY_ALIGN_COVERAGE_WEIGHT="${BOUNDARY_ALIGN_COVERAGE_WEIGHT:-}"
W_NONBOUNDARY_SMOOTH="${W_NONBOUNDARY_SMOOTH:-}"
W_NONBOUNDARY_PRESERVE="${W_NONBOUNDARY_PRESERVE:-}"
NONBOUNDARY_PRESERVE_RADIUS="${NONBOUNDARY_PRESERVE_RADIUS:-}"
USE_FROZEN_DA_PRIOR="${USE_FROZEN_DA_PRIOR:-}"
FROZEN_DA_MODEL="${FROZEN_DA_MODEL:-}"
FROZEN_DA_FEATURE_CHANNELS="${FROZEN_DA_FEATURE_CHANNELS:-}"
FROZEN_DA_INPUT_SIZE="${FROZEN_DA_INPUT_SIZE:-}"
FROZEN_DA_FUSION_SCALE="${FROZEN_DA_FUSION_SCALE:-}"
USE_FROZEN_DA_BOUNDARY_GATE="${USE_FROZEN_DA_BOUNDARY_GATE:-}"
FROZEN_DA_MIN_GATE="${FROZEN_DA_MIN_GATE:-}"
SKIP_NAN_BATCHES="${SKIP_NAN_BATCHES:-}"
VALIDATE_EVERY="${VALIDATE_EVERY:-}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-}"
PRINT_LOSSES="${PRINT_LOSSES:-1}"
PREFETCH="${PREFETCH:-}"
PIN_MEMORY="${PIN_MEMORY:-}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
MEMORY_LOG_EVERY="${MEMORY_LOG_EVERY:-}"

export WANDB_MODE
export WANDB_DISABLED
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

python scripts/check_kitti_dataset.py \
  --raw-root "${KITTI_RAW_ROOT}" \
  --gt-root "${KITTI_GT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_RESOURCE}" ]]; then
  EXTRA_ARGS+=(--pretrained_resource "${PRETRAINED_RESOURCE}")
fi
if [[ -n "${IMG_SIZE}" ]]; then
  EXTRA_ARGS+=(--img_size "${IMG_SIZE}")
fi
if [[ -n "${TRAIN_MIDAS}" ]]; then
  EXTRA_ARGS+=(--train_midas "${TRAIN_MIDAS}")
fi
if [[ -n "${MIDAS_MODEL_TYPE}" ]]; then
  EXTRA_ARGS+=(--midas_model_type "${MIDAS_MODEL_TYPE}")
fi
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${W_EDGE}" ]]; then
  EXTRA_ARGS+=(--w_edge "${W_EDGE}")
fi
if [[ -n "${EDGE_IMAGE_WEIGHT}" ]]; then
  EXTRA_ARGS+=(--edge_image_weight "${EDGE_IMAGE_WEIGHT}")
fi
if [[ -n "${EDGE_DEPTH_WEIGHT}" ]]; then
  EXTRA_ARGS+=(--edge_depth_weight "${EDGE_DEPTH_WEIGHT}")
fi
if [[ -n "${W_GRAD}" ]]; then
  EXTRA_ARGS+=(--w_grad "${W_GRAD}")
fi
if [[ -n "${USE_BOUNDARY_REFINE}" ]]; then
  EXTRA_ARGS+=(--use_boundary_refine "${USE_BOUNDARY_REFINE}")
fi
if [[ -n "${BOUNDARY_REFINE_CHANNELS}" ]]; then
  EXTRA_ARGS+=(--boundary_refine_channels "${BOUNDARY_REFINE_CHANNELS}")
fi
if [[ -n "${BOUNDARY_REFINE_SCALE}" ]]; then
  EXTRA_ARGS+=(--boundary_refine_scale "${BOUNDARY_REFINE_SCALE}")
fi
if [[ -n "${BOUNDARY_REFINE_MODE}" ]]; then
  EXTRA_ARGS+=(--boundary_refine_mode "${BOUNDARY_REFINE_MODE}")
fi
if [[ -n "${BOUNDARY_REFINE_USE_DA_PRIOR}" ]]; then
  EXTRA_ARGS+=(--boundary_refine_use_da_prior "${BOUNDARY_REFINE_USE_DA_PRIOR}")
fi
if [[ -n "${USE_DISCONTINUITY_BRANCH}" ]]; then
  EXTRA_ARGS+=(--use_discontinuity_branch "${USE_DISCONTINUITY_BRANCH}")
fi
if [[ -n "${DISCONTINUITY_CHANNELS}" ]]; then
  EXTRA_ARGS+=(--discontinuity_channels "${DISCONTINUITY_CHANNELS}")
fi
if [[ -n "${USE_DISCONTINUITY_TEMPERATURE}" ]]; then
  EXTRA_ARGS+=(--use_discontinuity_temperature "${USE_DISCONTINUITY_TEMPERATURE}")
fi
if [[ -n "${DISCONTINUITY_TEMPERATURE_SCALE}" ]]; then
  EXTRA_ARGS+=(--discontinuity_temperature_scale "${DISCONTINUITY_TEMPERATURE_SCALE}")
fi
if [[ -n "${W_BOUNDARY_CLS}" ]]; then
  EXTRA_ARGS+=(--w_boundary_cls "${W_BOUNDARY_CLS}")
fi
if [[ -n "${W_BOUNDARY_BAND}" ]]; then
  EXTRA_ARGS+=(--w_boundary_band "${W_BOUNDARY_BAND}")
fi
if [[ -n "${BOUNDARY_BAND_RADIUS}" ]]; then
  EXTRA_ARGS+=(--boundary_band_radius "${BOUNDARY_BAND_RADIUS}")
fi
if [[ -n "${BOUNDARY_LOG_GRAD_THRESHOLD}" ]]; then
  EXTRA_ARGS+=(--boundary_log_grad_threshold "${BOUNDARY_LOG_GRAD_THRESHOLD}")
fi
if [[ -n "${BOUNDARY_TARGET_ALPHA}" ]]; then
  EXTRA_ARGS+=(--boundary_target_alpha "${BOUNDARY_TARGET_ALPHA}")
fi
if [[ -n "${BOUNDARY_POS_WEIGHT}" ]]; then
  EXTRA_ARGS+=(--boundary_pos_weight "${BOUNDARY_POS_WEIGHT}")
fi
if [[ -n "${W_BOUNDARY_CONTRAST}" ]]; then
  EXTRA_ARGS+=(--w_boundary_contrast "${W_BOUNDARY_CONTRAST}")
fi
if [[ -n "${W_BOUNDARY_ALIGN}" ]]; then
  EXTRA_ARGS+=(--w_boundary_align "${W_BOUNDARY_ALIGN}")
fi
if [[ -n "${BOUNDARY_ALIGN_MAX_DISTANCE}" ]]; then
  EXTRA_ARGS+=(--boundary_align_max_distance "${BOUNDARY_ALIGN_MAX_DISTANCE}")
fi
if [[ -n "${BOUNDARY_ALIGN_PRED_WEIGHT}" ]]; then
  EXTRA_ARGS+=(--boundary_align_pred_weight "${BOUNDARY_ALIGN_PRED_WEIGHT}")
fi
if [[ -n "${BOUNDARY_ALIGN_COVERAGE_WEIGHT}" ]]; then
  EXTRA_ARGS+=(--boundary_align_coverage_weight "${BOUNDARY_ALIGN_COVERAGE_WEIGHT}")
fi
if [[ -n "${W_NONBOUNDARY_SMOOTH}" ]]; then
  EXTRA_ARGS+=(--w_nonboundary_smooth "${W_NONBOUNDARY_SMOOTH}")
fi
if [[ -n "${W_NONBOUNDARY_PRESERVE}" ]]; then
  EXTRA_ARGS+=(--w_nonboundary_preserve "${W_NONBOUNDARY_PRESERVE}")
fi
if [[ -n "${NONBOUNDARY_PRESERVE_RADIUS}" ]]; then
  EXTRA_ARGS+=(--nonboundary_preserve_radius "${NONBOUNDARY_PRESERVE_RADIUS}")
fi
if [[ -n "${USE_FROZEN_DA_PRIOR}" ]]; then
  EXTRA_ARGS+=(--use_frozen_da_prior "${USE_FROZEN_DA_PRIOR}")
fi
if [[ -n "${FROZEN_DA_MODEL}" ]]; then
  EXTRA_ARGS+=(--frozen_da_model "${FROZEN_DA_MODEL}")
fi
if [[ -n "${FROZEN_DA_FEATURE_CHANNELS}" ]]; then
  EXTRA_ARGS+=(--frozen_da_feature_channels "${FROZEN_DA_FEATURE_CHANNELS}")
fi
if [[ -n "${FROZEN_DA_INPUT_SIZE}" ]]; then
  EXTRA_ARGS+=(--frozen_da_input_size "${FROZEN_DA_INPUT_SIZE}")
fi
if [[ -n "${FROZEN_DA_FUSION_SCALE}" ]]; then
  EXTRA_ARGS+=(--frozen_da_fusion_scale "${FROZEN_DA_FUSION_SCALE}")
fi
if [[ -n "${USE_FROZEN_DA_BOUNDARY_GATE}" ]]; then
  EXTRA_ARGS+=(--use_frozen_da_boundary_gate "${USE_FROZEN_DA_BOUNDARY_GATE}")
fi
if [[ -n "${FROZEN_DA_MIN_GATE}" ]]; then
  EXTRA_ARGS+=(--frozen_da_min_gate "${FROZEN_DA_MIN_GATE}")
fi
if [[ -n "${SKIP_NAN_BATCHES}" ]]; then
  EXTRA_ARGS+=(--skip_nan_batches "${SKIP_NAN_BATCHES}")
fi
if [[ -n "${VALIDATE_EVERY}" ]]; then
  EXTRA_ARGS+=(--validate_every "${VALIDATE_EVERY}")
fi
if [[ -n "${LOG_IMAGES_EVERY}" ]]; then
  EXTRA_ARGS+=(--log_images_every "${LOG_IMAGES_EVERY}")
fi
if [[ -n "${PREFETCH}" ]]; then
  EXTRA_ARGS+=(--prefetch "${PREFETCH}")
fi
if [[ -n "${PIN_MEMORY}" ]]; then
  EXTRA_ARGS+=(--pin_memory "${PIN_MEMORY}")
fi
if [[ -n "${PERSISTENT_WORKERS}" ]]; then
  EXTRA_ARGS+=(--persistent_workers "${PERSISTENT_WORKERS}")
fi
if [[ -n "${PREFETCH_FACTOR}" ]]; then
  EXTRA_ARGS+=(--prefetch_factor "${PREFETCH_FACTOR}")
fi
if [[ -n "${GRAD_ACCUM_STEPS}" ]]; then
  EXTRA_ARGS+=(--grad_accum_steps "${GRAD_ACCUM_STEPS}")
fi
if [[ -n "${MEMORY_LOG_EVERY}" ]]; then
  EXTRA_ARGS+=(--memory_log_every "${MEMORY_LOG_EVERY}")
fi
if [[ -n "${CHECKPOINT_EVERY}" ]]; then
  EXTRA_ARGS+=(--checkpoint_every "${CHECKPOINT_EVERY}")
fi

python train_mono.py \
  -m zoedepth \
  -d kitti \
  --config_version kitti \
  --data_path "${KITTI_RAW_ROOT}" \
  --gt_path "${KITTI_GT_ROOT}" \
  --data_path_eval "${KITTI_RAW_ROOT}" \
  --gt_path_eval "${KITTI_GT_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --distributed 0 \
  --use_amp 1 \
  --print_losses "${PRINT_LOSSES}" \
  "${EXTRA_ARGS[@]}"
