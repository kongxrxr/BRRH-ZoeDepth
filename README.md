# BRRH-ZoeDepth

Boundary Residual Refinement Head (BRRH) for ZoeDepth-style metric monocular depth estimation.

This repository is prepared as the code-release package for the paper:

**Metric Depth at the Edge: Boundary Residual Refinement for Monocular Depth Estimation**

## Overview

BRRH-ZoeDepth keeps ZoeDepth as the metric-depth base and adds a lightweight boundary-aware residual head. The refinement is applied in final log-depth space:

```text
D_t = exp(log(D_b + eps) + alpha * M_b * tanh(Delta_b))
```

where `D_b` is the base metric depth, `M_b` is the predicted boundary mask, and `Delta_b` is the residual predicted by BRRH.

The repository contains:

- modified ZoeDepth source code under `zoedepth/`
- training and evaluation scripts under `scripts/`
- the IEEE Access manuscript source and PDF under `paper/`
- metric verification and evidence reports under `reports/`

Large datasets, checkpoints, cache directories, and local logs are intentionally not included.

## Environment

The original environment file is provided as:

```bash
environment.yml
```

Typical setup:

```bash
conda env create -f environment.yml
conda activate zoe
```

The local experiments used PyTorch with CUDA under WSL. Exact package versions may require adjustment for the user's CUDA/driver stack.

## Data

The experiments use public datasets:

- KITTI Eigen split
- NYU Depth V2 sync/labeled validation

Dataset files are not redistributed in this repository. See `DATA_AVAILABILITY.md`.

## Main Scripts

Representative scripts include:

```bash
scripts/run_kitti_scale024_brrh.sh
scripts/eval_kitti_scale024_brrh_after_train.sh
scripts/eval_nyu_sync_experiment.sh
scripts/evaluate_external_nyu_boundary.py
scripts/profile_model_complexity.py
```

The exact local commands used during development are preserved in `scripts/` and summarized in `reports/experiment_artifact_manifest.md`.

## Paper Artifacts

The generated manuscript artifacts are in:

```bash
paper/main.tex
paper/references.bib
paper/paper.pdf
```

## Citation

If this repository is used before formal publication, cite the manuscript title and this repository URL. A BibTeX entry can be added after publication.

## License

This repository is derived from ZoeDepth. See `LICENSE` and ensure that downstream use follows the original ZoeDepth license and third-party dataset/model terms.
