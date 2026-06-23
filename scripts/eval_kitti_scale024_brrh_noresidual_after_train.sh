#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_scale024_brrh_noresidual}"
CKPT_DIR="${CKPT_DIR:-/home/kxr/zoedepth_kitti_brrh_noresidual_scale0p24_256x512_bs4_workers8_5ep_checkpoints}"
OUT="${OUT:-${REPO_DIR}/logs/kitti_brrh_noresidual_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"
PYTHON="${PYTHON:-/home/kxr/miniconda3/envs/zoe/bin/python}"

cd "${REPO_DIR}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

echo "===== WAITING FOR ${TRAIN_SESSION} $(date '+%F %T %Z') ====="
while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 60
done
echo "===== TRAIN SESSION FINISHED $(date '+%F %T %Z') ====="

ckpt="$(ls -t "${CKPT_DIR}"/*latest.pt "${CKPT_DIR}"/*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${ckpt}" ]]; then
  echo "No checkpoint found in ${CKPT_DIR}" >&2
  exit 1
fi

echo "===== HARD BOUNDARY EVAL BRRH no-residual scale0.24 checkpoint=${ckpt} $(date '+%F %T %Z') ====="
"${PYTHON}" scripts/evaluate_kitti_boundary.py \
  --checkpoint "${ckpt}" \
  --output "${OUT}" \
  --model zoedepth \
  --dataset kitti \
  --config_version kitti \
  --midas_model_type DPT_BEiT_L_384 \
  --img_size 256,512 \
  --data_path_eval /home/kxr/shortcuts/datasets/kitti/raw \
  --gt_path_eval /home/kxr/shortcuts/datasets/kitti/gts \
  --use_boundary_refine 0 \
  --boundary_refine_mode log_residual \
  --boundary_refine_use_da_prior 1 \
  --use_discontinuity_branch 1 \
  --use_discontinuity_temperature 1 \
  --discontinuity_temperature_scale 0.5 \
  --use_frozen_da_prior 1 \
  --frozen_da_model "${FROZEN_DA_MODEL_PATH}" \
  --frozen_da_feature_channels 8 \
  --frozen_da_input_size 384 \
  --frozen_da_fusion_scale 0.12 \
  --use_frozen_da_boundary_gate 0 \
  --boundary_log_grad_threshold 0.15 \
  --hard_boundary_band_radii 3,5 \
  --max_depth_eval 80
echo "===== HARD BOUNDARY EVAL DONE ${OUT} $(date '+%F %T %Z') ====="
