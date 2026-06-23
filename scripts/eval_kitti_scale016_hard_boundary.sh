#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
CONDA_SH="${CONDA_SH:-/home/kxr/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-zoe}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"

cd "${REPO_DIR}"
mkdir -p logs

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

eval_variant() {
  local name="$1"
  local ckpt="$2"
  local use_gate="$3"
  local out="${REPO_DIR}/logs/${name}_hard_boundary_metrics.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint for ${name}: ${ckpt}"
    return 0
  fi
  if [[ -f "${out}" ]]; then
    echo "===== SKIP existing ${name}: ${out} ====="
    return 0
  fi

  echo "===== HARD BOUNDARY EVAL ${name} $(date '+%F %T %Z') ====="
  python scripts/evaluate_kitti_boundary.py \
    --checkpoint "${ckpt}" \
    --output "${out}" \
    --midas_model_type DPT_BEiT_L_384 \
    --img_size 256,512 \
    --use_boundary_refine 1 \
    --boundary_refine_channels 32 \
    --boundary_refine_scale 0.25 \
    --use_discontinuity_branch 1 \
    --discontinuity_channels 32 \
    --use_discontinuity_temperature 1 \
    --discontinuity_temperature_scale 0.5 \
    --use_frozen_da_prior 1 \
    --frozen_da_model "${FROZEN_DA_MODEL_PATH}" \
    --frozen_da_feature_channels 8 \
    --frozen_da_input_size 384 \
    --frozen_da_fusion_scale 0.16 \
    --use_frozen_da_boundary_gate "${use_gate}" \
    --frozen_da_min_gate 0.05 \
    --hard_boundary_top_percentiles 5,10 \
    --hard_boundary_band_radii 3,5 \
    --hard_boundary_f1_tolerances 1,3,5
}

eval_variant \
  "kitti_beit_dav2gate_scale0p16_full" \
  "/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p16_skipnan_256x512_bs1_checkpoints/ZoeDepthv1_09-Jun_19-10-fe43c4d30fdc_latest.pt" \
  1

eval_variant \
  "kitti_beit_dav2gate_scale0p16_nogate" \
  "/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p16_nogate_skipnan_256x512_bs4_checkpoints/ZoeDepthv1_09-Jun_23-29-9ddab213e681_latest.pt" \
  0

eval_variant \
  "kitti_beit_dav2gate_scale0p16_noalign" \
  "/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p16_noalign_skipnan_256x512_bs4_checkpoints/ZoeDepthv1_10-Jun_02-08-47e6622d4ded_latest.pt" \
  1

echo "===== HARD BOUNDARY EVAL DONE $(date '+%F %T %Z') ====="
