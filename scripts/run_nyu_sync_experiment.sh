#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
PYTHON="${PYTHON:-/home/kxr/miniconda3/envs/zoe/bin/python}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/home/kxr/.cache/torch/hub/checkpoints/ZoeD_M12_N.pt}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"
NYU_SYNC_ROOT="${NYU_SYNC_ROOT:-/home/kxr/shortcuts/datasets/nyu_depth_v2/sync}"
NYU_EVAL_ROOT="${NYU_EVAL_ROOT:-/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all}"
NYU_TRAIN_SPLIT="${NYU_TRAIN_SPLIT:-./train_test_inputs/nyudepthv2_train_files_with_gt.txt}"
NYU_EVAL_SPLIT="${NYU_EVAL_SPLIT:-./train_test_inputs/nyudepthv2_labeled_val654_files_with_gt.txt}"

VARIANT="${VARIANT:-baseline}"
EPOCHS="${EPOCHS:-3}"
BS="${BS:-4}"
WORKERS="${WORKERS:-4}"
LR="${LR:-0.000001}"
IMG_SIZE="${IMG_SIZE:-256,512}"
USE_AMP="${USE_AMP:-0}"
TRAIN_MIDAS="${TRAIN_MIDAS:-0}"
MIDAS_MODEL_TYPE="${MIDAS_MODEL_TYPE:-DPT_BEiT_L_384}"
VALIDATE_EVERY="${VALIDATE_EVERY:-999999}"
LOG_IMAGES_EVERY="${LOG_IMAGES_EVERY:-999}"
PRINT_LOSSES="${PRINT_LOSSES:-0}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MEMORY_LOG_EVERY="${MEMORY_LOG_EVERY:-500}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SKIP_NAN_BATCHES="${SKIP_NAN_BATCHES:-1}"

cd "${REPO_DIR}"

case "${VARIANT}" in
  baseline)
    SAVE_DIR="${SAVE_DIR:-/home/kxr/zoedepth_nyu_sync_baseline_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep_checkpoints}"
    LOG_FILE="${LOG_FILE:-${REPO_DIR}/logs/nyu_sync_baseline_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep.log}"
    DONE_FILE="${DONE_FILE:-${REPO_DIR}/logs/nyu_sync_baseline_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep.done}"
    USE_BOUNDARY_REFINE=0
    USE_DISCONTINUITY_BRANCH=0
    USE_DISCONTINUITY_TEMPERATURE=0
    USE_FROZEN_DA_PRIOR=0
    W_EDGE=0
    W_BOUNDARY_CLS=0
    W_BOUNDARY_BAND=0
    W_BOUNDARY_CONTRAST=0
    W_BOUNDARY_ALIGN=0
    W_NONBOUNDARY_SMOOTH=0
    W_NONBOUNDARY_PRESERVE=0
    ;;
  brrh)
    SAVE_DIR="${SAVE_DIR:-/home/kxr/zoedepth_nyu_sync_brrh_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep_checkpoints}"
    LOG_FILE="${LOG_FILE:-${REPO_DIR}/logs/nyu_sync_brrh_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep.log}"
    DONE_FILE="${DONE_FILE:-${REPO_DIR}/logs/nyu_sync_brrh_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep.done}"
    USE_BOUNDARY_REFINE="${USE_BOUNDARY_REFINE:-1}"
    USE_DISCONTINUITY_BRANCH="${USE_DISCONTINUITY_BRANCH:-1}"
    USE_DISCONTINUITY_TEMPERATURE="${USE_DISCONTINUITY_TEMPERATURE:-0}"
    USE_FROZEN_DA_PRIOR="${USE_FROZEN_DA_PRIOR:-1}"
    W_EDGE="${W_EDGE:-0.0}"
    W_BOUNDARY_CLS="${W_BOUNDARY_CLS:-0.02}"
    W_BOUNDARY_BAND="${W_BOUNDARY_BAND:-0.2}"
    W_BOUNDARY_CONTRAST="${W_BOUNDARY_CONTRAST:-0.0}"
    W_BOUNDARY_ALIGN="${W_BOUNDARY_ALIGN:-0.0}"
    W_NONBOUNDARY_SMOOTH="${W_NONBOUNDARY_SMOOTH:-0.0}"
    W_NONBOUNDARY_PRESERVE="${W_NONBOUNDARY_PRESERVE:-0.1}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected baseline or brrh" >&2
    exit 2
    ;;
esac

BOUNDARY_REFINE_CHANNELS="${BOUNDARY_REFINE_CHANNELS:-32}"
BOUNDARY_REFINE_SCALE="${BOUNDARY_REFINE_SCALE:-0.03}"
BOUNDARY_REFINE_MODE="${BOUNDARY_REFINE_MODE:-log_residual}"
BOUNDARY_REFINE_USE_DA_PRIOR="${BOUNDARY_REFINE_USE_DA_PRIOR:-1}"
DISCONTINUITY_CHANNELS="${DISCONTINUITY_CHANNELS:-32}"
DISCONTINUITY_TEMPERATURE_SCALE="${DISCONTINUITY_TEMPERATURE_SCALE:-0.5}"
BOUNDARY_BAND_RADIUS="${BOUNDARY_BAND_RADIUS:-3}"
BOUNDARY_LOG_GRAD_THRESHOLD="${BOUNDARY_LOG_GRAD_THRESHOLD:-0.15}"
BOUNDARY_TARGET_ALPHA="${BOUNDARY_TARGET_ALPHA:-10.0}"
BOUNDARY_POS_WEIGHT="${BOUNDARY_POS_WEIGHT:-4.0}"
BOUNDARY_ALIGN_MAX_DISTANCE="${BOUNDARY_ALIGN_MAX_DISTANCE:-6.0}"
BOUNDARY_ALIGN_PRED_WEIGHT="${BOUNDARY_ALIGN_PRED_WEIGHT:-0.5}"
BOUNDARY_ALIGN_COVERAGE_WEIGHT="${BOUNDARY_ALIGN_COVERAGE_WEIGHT:-0.5}"
NONBOUNDARY_PRESERVE_RADIUS="${NONBOUNDARY_PRESERVE_RADIUS:-5}"
FROZEN_DA_MODEL="${FROZEN_DA_MODEL:-${FROZEN_DA_MODEL_PATH}}"
FROZEN_DA_FEATURE_CHANNELS="${FROZEN_DA_FEATURE_CHANNELS:-8}"
FROZEN_DA_INPUT_SIZE="${FROZEN_DA_INPUT_SIZE:-384}"
FROZEN_DA_FUSION_SCALE="${FROZEN_DA_FUSION_SCALE:-0.12}"
USE_FROZEN_DA_BOUNDARY_GATE="${USE_FROZEN_DA_BOUNDARY_GATE:-0}"
FROZEN_DA_MIN_GATE="${FROZEN_DA_MIN_GATE:-0.05}"

if [[ -f "${DONE_FILE}" ]]; then
  echo "===== SKIP NYU sync ${VARIANT}; done marker exists: ${DONE_FILE} ====="
  exit 0
fi

mkdir -p logs "${SAVE_DIR}"
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/kxr/miniconda3/envs/zoe/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "===== NYU SYNC ${VARIANT} TRAIN START $(date '+%F %T %Z') ====="
echo "EPOCHS=${EPOCHS} BS=${BS} WORKERS=${WORKERS} LR=${LR} USE_AMP=${USE_AMP}"
echo "TRAIN=${NYU_SYNC_ROOT} SPLIT=${NYU_TRAIN_SPLIT}"
echo "EVAL=${NYU_EVAL_ROOT} SPLIT=${NYU_EVAL_SPLIT}"
"${PYTHON}" train_mono.py \
  -m zoedepth \
  -d nyu \
  --data_path "${NYU_SYNC_ROOT}" \
  --gt_path "${NYU_SYNC_ROOT}" \
  --data_path_eval "${NYU_EVAL_ROOT}" \
  --gt_path_eval "${NYU_EVAL_ROOT}" \
  --filenames_file "${NYU_TRAIN_SPLIT}" \
  --filenames_file_eval "${NYU_EVAL_SPLIT}" \
  --save_dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --bs "${BS}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --distributed 0 \
  --use_amp "${USE_AMP}" \
  --print_losses "${PRINT_LOSSES}" \
  --pretrained_resource "" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --img_size "${IMG_SIZE}" \
  --train_midas "${TRAIN_MIDAS}" \
  --midas_model_type "${MIDAS_MODEL_TYPE}" \
  --validate_every "${VALIDATE_EVERY}" \
  --log_images_every "${LOG_IMAGES_EVERY}" \
  --pin_memory "${PIN_MEMORY}" \
  --persistent_workers "${PERSISTENT_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --memory_log_every "${MEMORY_LOG_EVERY}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  --skip_nan_batches "${SKIP_NAN_BATCHES}" \
  --w_edge "${W_EDGE}" \
  --edge_image_weight 0.7 \
  --edge_depth_weight 0.3 \
  --w_grad 0 \
  --use_boundary_refine "${USE_BOUNDARY_REFINE}" \
  --boundary_refine_channels "${BOUNDARY_REFINE_CHANNELS}" \
  --boundary_refine_scale "${BOUNDARY_REFINE_SCALE}" \
  --boundary_refine_mode "${BOUNDARY_REFINE_MODE}" \
  --boundary_refine_use_da_prior "${BOUNDARY_REFINE_USE_DA_PRIOR}" \
  --use_discontinuity_branch "${USE_DISCONTINUITY_BRANCH}" \
  --discontinuity_channels "${DISCONTINUITY_CHANNELS}" \
  --use_discontinuity_temperature "${USE_DISCONTINUITY_TEMPERATURE}" \
  --discontinuity_temperature_scale "${DISCONTINUITY_TEMPERATURE_SCALE}" \
  --w_boundary_cls "${W_BOUNDARY_CLS}" \
  --w_boundary_band "${W_BOUNDARY_BAND}" \
  --boundary_band_radius "${BOUNDARY_BAND_RADIUS}" \
  --boundary_log_grad_threshold "${BOUNDARY_LOG_GRAD_THRESHOLD}" \
  --boundary_target_alpha "${BOUNDARY_TARGET_ALPHA}" \
  --boundary_pos_weight "${BOUNDARY_POS_WEIGHT}" \
  --w_boundary_contrast "${W_BOUNDARY_CONTRAST}" \
  --w_boundary_align "${W_BOUNDARY_ALIGN}" \
  --boundary_align_max_distance "${BOUNDARY_ALIGN_MAX_DISTANCE}" \
  --boundary_align_pred_weight "${BOUNDARY_ALIGN_PRED_WEIGHT}" \
  --boundary_align_coverage_weight "${BOUNDARY_ALIGN_COVERAGE_WEIGHT}" \
  --w_nonboundary_smooth "${W_NONBOUNDARY_SMOOTH}" \
  --w_nonboundary_preserve "${W_NONBOUNDARY_PRESERVE}" \
  --nonboundary_preserve_radius "${NONBOUNDARY_PRESERVE_RADIUS}" \
  --use_frozen_da_prior "${USE_FROZEN_DA_PRIOR}" \
  --frozen_da_model "${FROZEN_DA_MODEL}" \
  --frozen_da_feature_channels "${FROZEN_DA_FEATURE_CHANNELS}" \
  --frozen_da_input_size "${FROZEN_DA_INPUT_SIZE}" \
  --frozen_da_fusion_scale "${FROZEN_DA_FUSION_SCALE}" \
  --use_frozen_da_boundary_gate "${USE_FROZEN_DA_BOUNDARY_GATE}" \
  --frozen_da_min_gate "${FROZEN_DA_MIN_GATE}"

touch "${DONE_FILE}"
echo "===== NYU SYNC ${VARIANT} TRAIN DONE $(date '+%F %T %Z') ====="
