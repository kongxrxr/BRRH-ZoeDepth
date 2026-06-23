#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="${REPO_DIR:-/home/kxr/ZoeDepth}"
CONDA_SH="${CONDA_SH:-/home/kxr/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-zoe}"
TRAIN_SESSION="${TRAIN_SESSION:-zoe_kitti_dav2gate_014_016}"
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

for scale in 0.14 0.16; do
  label="${scale/./p}"
  ckpt_dir="/home/kxr/zoedepth_kitti_beit_dav2gate_scale${label}_skipnan_256x512_bs1_checkpoints"
  ckpt="$(ls -t "${ckpt_dir}"/*_latest.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "Missing checkpoint for scale=${scale}: ${ckpt_dir}"
    continue
  fi

  out="${REPO_DIR}/logs/kitti_beit_dav2gate_scale${label}_skipnan_256x512_bs1_boundary_metrics.json"
  echo "===== EVAL scale=${scale} checkpoint=${ckpt} $(date '+%F %T %Z') ====="
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
    --frozen_da_fusion_scale "${scale}" \
    --use_frozen_da_boundary_gate 1 \
    --frozen_da_min_gate 0.05
done

echo "===== SCALE 0.14/0.16 EVAL DONE $(date '+%F %T %Z') ====="
