# Experiment Artifact Manifest

Date: 2026-06-20

Purpose: provide a compact reviewer-facing map from manuscript evidence to local artifacts. This complements `reports/submission_reproducibility_checklist.md`: the checklist explains how to reproduce or verify results, while this manifest explains where each submitted result comes from and what claim it supports.

## Verification Commands

Run these before sending the paper to a supervisor, journal system, or reviewer:

```bash
cd /home/kxr/ZoeDepth
bash scripts/run_submission_audit.sh
```

For a full rebuild of PDF and DOCX followed by audit:

```bash
cd /home/kxr/ZoeDepth
bash scripts/build_submission_package.sh
```

To create a clean review folder:

```bash
cd /home/kxr/ZoeDepth
bash scripts/create_submission_snapshot.sh
```

To create a transferable archive:

```bash
bash scripts/create_submission_snapshot.sh --tar
```

Expected current status: all metric, complexity, and claim-boundary checks pass; submission readiness remains attention-required until real author metadata and declaration statements are provided.

## Manuscript Artifacts

| Artifact | Path | Role |
|---|---|---|
| LaTeX manuscript | `paper_rewriting_output/final_paper/main.tex` | Authoritative paper source |
| PDF draft | `paper_rewriting_output/final_paper/paper.pdf` | Reviewer-readable draft |
| Word draft | `paper_rewriting_output/final_paper/paper.docx` | ITC/template-facing draft |
| Architecture figure | `paper_rewriting_output/final_paper/figures/boundary_zoedepth_architecture.png` | Method overview |
| NYU qualitative comparison | `paper_rewriting_output/final_paper/figures/nyu_brrh_depthpro_top_samples.png` | ZoeDepth/BRRH/Depth Pro visual diagnostic |
| KITTI no-residual qualitative | `paper_rewriting_output/final_paper/figures/brrh_noresidual_qualitative.png` | Residual-head ablation visual |

## Table And Figure Evidence Map

| Manuscript item | Main evidence files | Verification support | Safe claim |
|---|---|---|---|
| `tab:main-results` KITTI main results | `logs/kitti_brrh_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_beit_dav2gate_scale0p24_strong_bs4_workers8_hard_boundary_metrics.json`; `logs/kitti_baseline_extended_boundary_metrics.json` | `scripts/verify_paper_metrics.py` | BRRH improves several KITTI boundary-band metric errors over the previous strong boundary-gated variant. |
| `tab:hard-boundary` KITTI hard-boundary metrics | Same KITTI logs as above | `scripts/verify_paper_metrics.py` | BRRH improves Top5 AbsRel and Band3 AbsRel, but the previous strong model keeps better Edge F1. |
| `tab:nyu-sync` NYU val654 | `logs/nyu_sync_baseline_resume_approx3ep_val654_hard_boundary_metrics.json`; `logs/nyu_sync_brrh_tuned_boundary_val654_hard_boundary_metrics.json`; `logs/nyu_sync_brrh_prevconfig_resume_val654_hard_boundary_metrics.json` | `scripts/verify_paper_metrics.py` | BRRH tuned preserves global metric accuracy and improves Edge F1 on NYU. |
| `tab:nyu-density` NYU boundary-density subsets | `logs/nyu_boundary_density_subsets_baseline_brrh.json`; `logs/nyu_boundary_density_subsets_per_sample.json` | `scripts/evaluate_nyu_boundary_density_subsets.py`; `scripts/verify_paper_metrics.py` | Edge F1 improves across low-, medium-, and high-density subsets, while NYU boundary-value errors slightly worsen. |
| `tab:nyu-dav2-external` external strong baselines | `logs/nyu_dav2_small_inverse_val654_hard_boundary_metrics.json`; `logs/nyu_depthpro_val654_hard_boundary_metrics.json` | `scripts/evaluate_external_nyu_boundary.py`; `scripts/verify_paper_metrics.py` | DA-V2 and Depth Pro are strong structural references; BRRH is a metric-pipeline refinement, not a replacement. |
| `tab:alignment-sensitivity` external alignment sensitivity | `logs/nyu_dav2_small_inverse_noalign_val654_hard_boundary_metrics.json`; `logs/nyu_depthpro_noalign_val654_hard_boundary_metrics.json`; median-aligned logs above | `scripts/evaluate_external_nyu_boundary.py`; `scripts/verify_paper_metrics.py` | Edge localization is more stable than metric error under alignment changes. |
| `tab:ablation` KITTI ablations | `logs/kitti_brrh_noresidual_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_brrh_nodaprior_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_brrh_notemperature_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_brrh_nobandloss_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_brrh_nocontrastloss_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_brrh_noboundaryloss_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_brrh_nopreserve_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json` | `scripts/verify_paper_metrics.py` | The final log-depth residual head is the clearest ablation-supported component. |
| `tab:complexity` model complexity | `reports/model_complexity_profile_latest.json`; `reports/model_complexity_profile_latest.md` | `scripts/profile_model_complexity.py`; `scripts/verify_paper_metrics.py` | BRRH adds 0.081M trainable boundary-related parameters and uses a frozen 24.786M DA prior. |
| `fig:architecture` method diagram | `paper_rewriting_output/final_paper/figures/boundary_zoedepth_architecture.png` | Manual figure inspection | ZoeDepth supplies metric base; discontinuity, DA-V2 prior, and BRRH supply boundary-aware residual correction. |
| `fig:nyu-depthpro-qualitative` NYU qualitative | `paper_rewriting_output/final_paper/figures/nyu_brrh_depthpro_top_samples.png`; `figures/nyu_brrh_depthpro_comparison/selected_samples.json` | `scripts/visualize_nyu_brrh_depthpro.py`; `scripts/render_nyu_depthpro_visuals.py` | BRRH can reduce boundary-band AbsRel on selected samples, while Depth Pro remains structurally sharper in many regions. |
| `fig:brrh-noresidual-qualitative` no-residual visual | `paper_rewriting_output/final_paper/figures/brrh_noresidual_qualitative.png` | `scripts/compare_kitti_brrh_baseline_visuals.py` or stored generated artifact | The residual head changes local boundary-band errors more than global visual layout. |

## Claim Boundaries

Allowed wording:

- BRRH studies boundary-local metric reliability rather than generic visual sharpening.
- BRRH improves KITTI boundary-band metric behavior over the previous strong boundary-gated variant.
- On NYU, BRRH tuned preserves global metric accuracy and improves Edge F1, but does not improve every boundary-value metric.
- DA-V2 and Depth Pro show that structural sharpness and metric reliability should be evaluated separately.
- BRRH is lightweight in trainable parameters, while the frozen DA prior should be reported as additional frozen capacity.

Avoid wording:

- BRRH beats Depth Pro overall.
- BRRH is state of the art for boundary sharpness.
- BRRH improves all boundary metrics on NYU.
- DA-V2 is a native metric-depth baseline.
- The method validates a complete control or military panoramic surround-view system.

## Remaining Non-Technical Submission Gaps

These are intentionally not filled by scripts or experiments:

- Real author names and affiliations.
- Corresponding author name and email.
- Funding statement.
- Conflict-of-interest statement.
- Data-availability statement.
- Code-availability statement.
- Final visual check of `paper.docx` against the journal RTF template.
