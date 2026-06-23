#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
CONDA_SH="${CONDA_SH:-/home/kxr/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-zoe}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_scale016_ablation}"
FROZEN_DA_MODEL_PATH="${FROZEN_DA_MODEL_PATH:-/home/kxr/.cache/huggingface/hub/models--LiheYoung--depth-anything-small-hf/snapshots/25216a913fa218ccb7d58cce818d52b728b6c1f6}"

cd "${REPO_DIR}"
mkdir -p logs

echo "===== WAITING FOR ${TRAIN_SESSION} $(date '+%F %T %Z') ====="
while tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq "${TRAIN_SESSION}"; do
  sleep 300
done
echo "===== TRAIN SESSION FINISHED $(date '+%F %T %Z') ====="

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

eval_variant() {
  local name="$1"
  local use_gate="$2"
  local ckpt_dir="/home/kxr/zoedepth_kitti_beit_dav2gate_scale0p16_${name}_skipnan_256x512_bs4_checkpoints"
  local ckpt
  ckpt="$(ls -t "${ckpt_dir}"/*_latest.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "Missing checkpoint for ablation=${name}: ${ckpt_dir}"
    return 0
  fi

  local out="${REPO_DIR}/logs/kitti_beit_dav2gate_scale0p16_${name}_skipnan_256x512_bs4_boundary_metrics.json"
  echo "===== EVAL ablation=${name} checkpoint=${ckpt} $(date '+%F %T %Z') ====="
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
    --frozen_da_min_gate 0.05
}

eval_variant "nogate" 0
eval_variant "noalign" 1

echo "===== SCALE 0.16 ABLATION EVAL DONE $(date '+%F %T %Z') ====="
