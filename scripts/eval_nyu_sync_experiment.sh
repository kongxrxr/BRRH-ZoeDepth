#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
PYTHON="${PYTHON:-/home/kxr/miniconda3/envs/zoe/bin/python}"
VARIANT="${VARIANT:-baseline}"
EPOCHS="${EPOCHS:-3}"
BS="${BS:-4}"
WORKERS="${WORKERS:-4}"
NYU_EVAL_ROOT="${NYU_EVAL_ROOT:-/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all}"
NYU_EVAL_SPLIT="${NYU_EVAL_SPLIT:-${REPO_DIR}/train_test_inputs/nyudepthv2_labeled_val654_files_with_gt.txt}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"

cd "${REPO_DIR}"

case "${VARIANT}" in
  baseline)
    CKPT_DIR="${CKPT_DIR:-/home/kxr/zoedepth_nyu_sync_baseline_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep_checkpoints}"
    OUT="${OUT:-${REPO_DIR}/logs/nyu_sync_baseline_${EPOCHS}ep_val654_hard_boundary_metrics.json}"
    USE_BOUNDARY_REFINE=0
    USE_DISCONTINUITY_BRANCH=0
    USE_DISCONTINUITY_TEMPERATURE=0
    USE_FROZEN_DA_PRIOR=0
    ;;
  brrh)
    CKPT_DIR="${CKPT_DIR:-/home/kxr/zoedepth_nyu_sync_brrh_256x512_bs${BS}_workers${WORKERS}_${EPOCHS}ep_checkpoints}"
    OUT="${OUT:-${REPO_DIR}/logs/nyu_sync_brrh_${EPOCHS}ep_val654_hard_boundary_metrics.json}"
    USE_BOUNDARY_REFINE=1
    USE_DISCONTINUITY_BRANCH=1
    USE_DISCONTINUITY_TEMPERATURE=0
    USE_FROZEN_DA_PRIOR=1
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected baseline or brrh" >&2
    exit 2
    ;;
esac

ckpt="$(find "${CKPT_DIR}" -maxdepth 1 -name '*_latest.pt' -print | sort | tail -n 1)"
if [[ -z "${ckpt}" ]]; then
  echo "No latest checkpoint found in ${CKPT_DIR}" >&2
  exit 1
fi

mkdir -p logs
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/kxr/miniconda3/envs/zoe/lib:${LD_LIBRARY_PATH:-}"

echo "===== NYU SYNC ${VARIANT} VAL EVAL checkpoint=${ckpt} $(date '+%F %T %Z') ====="
"${PYTHON}" scripts/evaluate_kitti_boundary.py \
  --checkpoint "${ckpt}" \
  --out "${OUT}" \
  --model zoedepth \
  --dataset nyu \
  --midas_model_type DPT_BEiT_L_384 \
  --img_size 256,512 \
  --data_path_eval "${NYU_EVAL_ROOT}" \
  --gt_path_eval "${NYU_EVAL_ROOT}" \
  --filenames_file_eval "${NYU_EVAL_SPLIT}" \
  --min_depth_eval 0.001 \
  --max_depth_eval 10 \
  --garg_crop 0 \
  --eigen_crop 1 \
  --use_boundary_refine "${USE_BOUNDARY_REFINE}" \
  --boundary_refine_channels 32 \
  --boundary_refine_scale 0.03 \
  --boundary_refine_mode log_residual \
  --boundary_refine_use_da_prior 1 \
  --use_discontinuity_branch "${USE_DISCONTINUITY_BRANCH}" \
  --discontinuity_channels 32 \
  --use_discontinuity_temperature "${USE_DISCONTINUITY_TEMPERATURE}" \
  --discontinuity_temperature_scale 0.5 \
  --use_frozen_da_prior "${USE_FROZEN_DA_PRIOR}" \
  --frozen_da_model "${FROZEN_DA_MODEL_PATH}" \
  --frozen_da_feature_channels 8 \
  --frozen_da_input_size 384 \
  --frozen_da_fusion_scale 0.12 \
  --use_frozen_da_boundary_gate 0 \
  --frozen_da_min_gate 0.05 \
  --boundary_log_grad_threshold 0.15

echo "===== NYU SYNC ${VARIANT} VAL EVAL DONE ${OUT} $(date '+%F %T %Z') ====="
