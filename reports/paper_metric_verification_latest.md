# Paper Metric Verification

Command:

```bash
python scripts/verify_paper_metrics.py --write-report reports/paper_metric_verification_latest.md
```

Metric specs checked: 27
Complexity specs checked: 2
Missing log files: 0
Missing metric keys: 0
Logged values not found in manuscript: 0

| Section | Label | Metrics checked | Status |
|---|---|---:|---|
| KITTI main/hard-boundary | ZoeDepth baseline | 6 | PASS |
| KITTI main/hard-boundary | BEiT BoundaryAlign | 6 | PASS |
| KITTI main/hard-boundary | BEiT + DA gate scale 0.16 | 10 | PASS |
| KITTI main/hard-boundary | BEiT + DA gate scale 0.16 no gate | 6 | PASS |
| KITTI main/hard-boundary | BEiT + DA gate scale 0.24 strong | 10 | PASS |
| KITTI main/hard-boundary | BRRH scale 0.24 2epoch | 10 | PASS |
| KITTI main/hard-boundary | BRRH scale 0.24 5epoch | 10 | PASS |
| NYU sync | NYU-sync ZoeDepth baseline | 7 | PASS |
| NYU sync | NYU-sync BRRH | 7 | PASS |
| NYU sync | NYU-sync BRRH tuned | 7 | PASS |
| NYU external | DA-V2 inverse median | 7 | PASS |
| NYU external | Depth Pro median | 7 | PASS |
| NYU alignment | DA-V2 inverse none | 5 | PASS |
| NYU alignment | Depth Pro none | 5 | PASS |
| KITTI ablation | BRRH without DA-V2 prior | 6 | PASS |
| KITTI ablation | BRRH without temperature sharpening | 6 | PASS |
| KITTI ablation | BRRH without boundary-band loss | 6 | PASS |
| KITTI ablation | BRRH without contrast loss | 6 | PASS |
| KITTI ablation | BRRH without boundary losses | 6 | PASS |
| KITTI ablation | BRRH without preservation | 6 | PASS |
| KITTI ablation | BRRH without residual head | 6 | PASS |
| NYU density | baseline low | 5 | PASS |
| NYU density | brrh low | 5 | PASS |
| NYU density | baseline medium | 5 | PASS |
| NYU density | brrh medium | 5 | PASS |
| NYU density | baseline high | 5 | PASS |
| NYU density | brrh high | 5 | PASS |
| Model complexity | ZoeDepth baseline | 7 | PASS |
| Model complexity | BRRH-ZoeDepth | 7 | PASS |

All checked manuscript values are traceable to configured JSON files: experiment metrics at six-decimal precision and complexity values as three-decimal million-parameter values.
