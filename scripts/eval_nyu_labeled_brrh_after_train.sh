#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
PYTHON="${PYTHON:-/home/kxr/miniconda3/envs/zoe/bin/python}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_nyu_labeled_brrh}"
CKPT_DIR="${CKPT_DIR:-/home/kxr/zoedepth_nyu_labeled_brrh_256x512_bs4_workers4_5ep_checkpoints}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"
NYU_ROOT="${NYU_ROOT:-/home/kxr/shortcuts/datasets/nyu_depth_v2/official_splits/labeled_all}"
NYU_SPLIT="${NYU_SPLIT:-${REPO_DIR}/train_test_inputs/nyudepthv2_labeled_val654_files_with_gt.txt}"
OUT="${OUT:-${REPO_DIR}/logs/nyu_labeled_brrh_val654_hard_boundary_metrics.json}"

cd "${REPO_DIR}"
mkdir -p logs

echo "===== WAITING FOR ${TRAIN_SESSION} $(date '+%F %T %Z') ====="
while tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${TRAIN_SESSION}"; do
  sleep 60
done
echo "===== TRAIN SESSION FINISHED $(date '+%F %T %Z') ====="

ckpt="$(ls -t "${CKPT_DIR}"/*_latest.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${ckpt}" ]]; then
  echo "Missing checkpoint: ${CKPT_DIR}"
  exit 1
fi

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/kxr/miniconda3/envs/zoe/lib:${LD_LIBRARY_PATH:-}"

echo "===== NYU LABELED BRRH VAL EVAL checkpoint=${ckpt} $(date '+%F %T %Z') ====="
"${PYTHON}" scripts/evaluate_kitti_boundary.py \
  --checkpoint "${ckpt}" \
  --output "${OUT}" \
  --model zoedepth \
  --dataset nyu \
  --midas_model_type DPT_BEiT_L_384 \
  --img_size 256,512 \
  --data_path_eval "${NYU_ROOT}" \
  --gt_path_eval "${NYU_ROOT}" \
  --filenames_file_eval "${NYU_SPLIT}" \
  --min_depth_eval 0.001 \
  --max_depth_eval 10.0 \
  --garg_crop 0 \
  --eigen_crop 1 \
  --use_boundary_refine 1 \
  --boundary_refine_channels 32 \
  --boundary_refine_scale 0.08 \
  --boundary_refine_mode log_residual \
  --boundary_refine_use_da_prior 1 \
  --use_discontinuity_branch 1 \
  --discontinuity_channels 32 \
  --use_discontinuity_temperature 1 \
  --discontinuity_temperature_scale 0.5 \
  --use_frozen_da_prior 1 \
  --frozen_da_model "${FROZEN_DA_MODEL_PATH}" \
  --frozen_da_feature_channels 8 \
  --frozen_da_input_size 384 \
  --frozen_da_fusion_scale 0.12 \
  --use_frozen_da_boundary_gate 0 \
  --frozen_da_min_gate 0.05 \
  --boundary_log_grad_threshold 0.15 \
  --hard_boundary_top_percentiles 5,10 \
  --hard_boundary_band_radii 3,5 \
  --hard_boundary_f1_tolerances 1,3,5

echo "===== NYU LABELED BRRH VAL EVAL DONE ${OUT} $(date '+%F %T %Z') ====="
