#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
CONDA_SH="${CONDA_SH:-/home/kxr/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-zoe}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_scale024_brrh_nodaprior}"
CKPT_DIR="${CKPT_DIR:-/home/kxr/zoedepth_kitti_brrh_nodaprior_scale0p24_256x512_bs4_workers8_5ep_checkpoints}"
OUT="${OUT:-${REPO_DIR}/logs/kitti_brrh_nodaprior_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json}"

cd "${REPO_DIR}"
mkdir -p logs

echo "===== WAITING FOR ${TRAIN_SESSION} $(date '+%F %T %Z') ====="
while tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${TRAIN_SESSION}"; do
  sleep 300
done
echo "===== TRAIN SESSION FINISHED $(date '+%F %T %Z') ====="

ckpt="$(ls -t "${CKPT_DIR}"/*_latest.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "${ckpt}" ]]; then
  echo "Missing checkpoint: ${CKPT_DIR}"
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "===== HARD BOUNDARY EVAL BRRH no-DA-prior scale0.24 checkpoint=${ckpt} $(date '+%F %T %Z') ====="
python scripts/evaluate_kitti_boundary.py \
  --checkpoint "${ckpt}" \
  --output "${OUT}" \
  --midas_model_type DPT_BEiT_L_384 \
  --img_size 256,512 \
  --use_boundary_refine 1 \
  --boundary_refine_channels 32 \
  --boundary_refine_scale 0.08 \
  --boundary_refine_mode log_residual \
  --boundary_refine_use_da_prior 0 \
  --use_discontinuity_branch 1 \
  --discontinuity_channels 32 \
  --use_discontinuity_temperature 1 \
  --discontinuity_temperature_scale 0.5 \
  --use_frozen_da_prior 0 \
  --frozen_da_feature_channels 8 \
  --frozen_da_input_size 384 \
  --frozen_da_fusion_scale 0.0 \
  --use_frozen_da_boundary_gate 0 \
  --frozen_da_min_gate 0.05 \
  --hard_boundary_top_percentiles 5,10 \
  --hard_boundary_band_radii 3,5 \
  --hard_boundary_f1_tolerances 1,3,5

echo "===== BRRH NO-DA-PRIOR SCALE 0.24 EVAL DONE $(date '+%F %T %Z') ====="
