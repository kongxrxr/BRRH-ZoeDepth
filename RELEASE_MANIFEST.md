# Release Manifest

This package was prepared from the local BRRH-ZoeDepth workspace.

Included:

- `zoedepth/`: modified model and trainer source code
- `scripts/`: training, evaluation, visualization, complexity, and manuscript utility scripts
- `paper/`: IEEE Access LaTeX source, BibTeX file, and compiled PDF
- `reports/`: selected verification and evidence reports
- `environment.yml`: conda environment specification inherited from ZoeDepth
- `README.md`, `DATA_AVAILABILITY.md`, `LICENSE`

Excluded:

- KITTI and NYU dataset files
- local cache directories such as `.deps/`
- training logs and tmux logs
- generated Python bytecode
- large checkpoints and downloaded foundation model weights
- W&B runs and local output scratch files

Before public release, review:

1. Whether checkpoints should be uploaded as GitHub Releases or separate cloud links.
2. Whether any institution-specific path or private note remains in scripts or reports.
3. Whether the repository should be public or private during peer review.
