# Reviewer Evidence Matrix

Date: 2026-06-20

Purpose: map every defensible BRRH-ZoeDepth claim to manuscript locations, evidence files, safe wording, and claim boundaries. Use this file when preparing reviewer responses, group-meeting slides, or final manuscript revisions.

## Core Thesis

BRRH-ZoeDepth is best defended as:

> a lightweight, bounded final log-depth residual refinement module for ZoeDepth-style metric monocular depth estimation, guided by predicted depth discontinuities and a frozen structural prior.

It should not be defended as:

> a universal sharp-depth foundation model or a method that beats Depth Pro overall.

## Evidence Matrix

| Claim | Manuscript evidence | File/log evidence | Safe wording | Do not claim |
|---|---|---|---|---|
| The problem is real and not captured by global metrics alone. | Introduction; Dataset and Metrics section | `reports/reviewer_ready_story_and_strategy.md` | Boundary regions are small but important for foreground-background separation and projected geometry. | Full-image RMSE fully measures boundary quality. |
| The story is boundary-local metric reliability, not visual sharpening. | Introduction; Dataset and Metrics; Discussion | `reports/reviewer_ready_story_and_strategy.md`; `reports/reviewer_response_playbook.md` | Boundary leakage can create non-physical intermediate depths, false ramps, and thickened contours in projected geometry. | The contribution is just making depth maps look sharper. |
| The experiment design is organized around falsifiable questions. | Dataset and Metrics section; Tables `tab:main-results`, `tab:nyu-sync`, `tab:nyu-dav2-external` | `reports/paper_metric_verification_latest.md` | KITTI tests boundary-band metric improvement, NYU tests transfer, and external baselines test the separation between structural sharpness and metric reliability. | The experiments are only a collection of unrelated benchmark tables. |
| Manuscript artifacts are traceable. | All tables and figures | `reports/experiment_artifact_manifest.md`; `reports/submission_reproducibility_checklist.md` | Each submitted table/figure has a local log, script, or generated artifact path. | Results are unsupported by local evidence. |
| Strict reviewer risks are documented. | Discussion; Limitations; reviewer support reports | `reports/strict_reviewer_precheck.md`; `reports/final_submission_gap_report.md` | The paper is defensible as a conservative small paper if claims remain calibrated. | The current evidence is enough for a SOTA or foundation-model claim. |
| BRRH is not just edge loss. | Eq. for final log-depth residual; Method; Table `tab:ablation` | `zoedepth/models/zoedepth/zoedepth_v1.py`; `logs/kitti_brrh_noresidual_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json` | BRRH changes the final prediction through a bounded residual in log-depth space. | This is merely Sobel/gradient supervision. |
| ZoeDepth remains responsible for metric scale. | Method; Complexity section | `paper_rewriting_output/final_paper/main.tex`; `reports/model_complexity_profile_latest.md` | The base metric-depth prediction is preserved and corrected locally near discontinuities. | DA-V2 replaces ZoeDepth as the metric head. |
| DA-V2 is a frozen structural prior. | Related Work; Method; External Baselines | `zoedepth/models/zoedepth/zoedepth_v1.py`; `reports/model_complexity_profile_latest.json` | DA-V2 contributes structure but is not optimized as the final metric output. | DA-V2 is a native metric-depth baseline. |
| BRRH improves KITTI boundary-band metric behavior. | Table `tab:main-results`; Table `tab:hard-boundary` | `logs/kitti_brrh_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `logs/kitti_beit_dav2gate_scale0p24_strong_bs4_workers8_hard_boundary_metrics.json` | BRRH improves RMSE, SILog, Boundary RMSE, Top5 AbsRel, and Band3 AbsRel over the previous strong boundary-gated variant. | BRRH is best on every KITTI metric. |
| BRRH does not dominate KITTI edge localization. | Table `tab:hard-boundary` | Same KITTI logs as above | BRRH improves boundary-band metric values while the previous strong model keeps better Edge F1. | BRRH gives the sharpest KITTI boundaries by Edge F1. |
| NYU shows transfer of the boundary module. | Table `tab:nyu-sync` | `logs/nyu_sync_baseline_resume_approx3ep_val654_hard_boundary_metrics.json`; `logs/nyu_sync_brrh_tuned_boundary_val654_hard_boundary_metrics.json` | BRRH preserves global metric accuracy and improves Edge F1 on NYU val654. | BRRH improves every NYU boundary metric. |
| Boundary localization and boundary-region metric error are different goals. | Tables `tab:nyu-sync`, `tab:nyu-density`, `tab:nyu-dav2-external` | `logs/nyu_boundary_density_subsets_baseline_brrh.json` | Edge F1 can improve while Boundary RMSE or Band3 AbsRel does not. | A sharper boundary map is automatically a better metric-depth map. |
| Strong external baselines were considered. | Table `tab:nyu-dav2-external`; Figure `fig:nyu-depthpro-qualitative` | `logs/nyu_dav2_small_inverse_val654_hard_boundary_metrics.json`; `logs/nyu_depthpro_val654_hard_boundary_metrics.json` | Depth Pro and DA-V2 are strong structural references; BRRH is a metric-pipeline refinement. | BRRH beats Depth Pro overall. |
| Metric errors of external models are alignment-sensitive. | Table `tab:alignment-sensitivity` | `logs/nyu_dav2_small_inverse_noalign_val654_hard_boundary_metrics.json`; `logs/nyu_depthpro_noalign_val654_hard_boundary_metrics.json` | Edge F1 remains stable while AbsRel/RMSE can change with alignment. | Median alignment is unnecessary for relative-depth comparison. |
| The key ablation supports final residual correction. | Table `tab:ablation`; Figure `fig:brrh-noresidual-qualitative` | `logs/kitti_brrh_noresidual_scale0p24_bs4_workers8_5ep_hard_boundary_metrics.json`; `figures/brrh_noresidual_qualitative.png` | Removing the residual head weakens boundary-band metric behavior. | Every auxiliary component independently gives a large gain. |
| BRRH is lightweight in trainable parameters. | Table `tab:complexity` | `reports/model_complexity_profile_latest.json`; `scripts/profile_model_complexity.py` | Added trainable BRRH/discontinuity/fusion modules contain 0.081M parameters; frozen DA prior should be reported separately. | The whole model is small or DA-V2 has no cost. |
| The control/surround-view relevance is perception-level and bounded. | Introduction; Discussion | `reports/group_meeting_presentation_2026-06-20.md`; `reports/reviewer_response_playbook.md` | BRRH can support local geometric perception, obstacle-contour understanding, point-cloud projection, or multi-view fusion as an auxiliary depth refinement stage. | The paper validates a complete military panoramic surround-view control system. |
| Limitations are explicit. | Section `Limitations and Future Work` | `reports/reviewer_response_playbook.md`; `reports/nyu_resolution_sensitivity.md` | The paper openly states supervision-quality, resolution-sensitivity, and single-mode residual limitations. | The method fully solves boundary ambiguity or low-resolution deployment. |
| The numbers are auditable. | All tables | `scripts/verify_paper_metrics.py`; `reports/paper_metric_verification_latest.md` | 27 experimental metric groups and 2 model-complexity groups, reported as 29 checked specs, are traceable to JSON evidence. | Tables were manually adjusted without logs. |
| The paper is close to submission but not final. | Submission checklist; audit reports | `scripts/run_submission_audit.sh`; `reports/submission_audit_latest.md` | The package still needs real author metadata and required declarations. | The paper is ready to submit with placeholders. |

## Claim-to-Response Shortcuts

### If reviewer says "Depth Pro is sharper"

Response:

> Yes. Depth Pro is included precisely because it is a strong high-resolution structural reference. The contribution of BRRH is not to replace Depth Pro, but to add a small trainable residual module inside a ZoeDepth-style metric pipeline. The paper therefore reports both Edge F1 and metric boundary-region errors.

Evidence:

- Table `tab:nyu-dav2-external`
- Figure `fig:nyu-depthpro-qualitative`
- `logs/nyu_depthpro_val654_hard_boundary_metrics.json`

### If reviewer says "This is just edge loss"

Response:

> The main mechanism is the bounded final log-depth residual. Boundary losses provide supervision, but the prediction itself is modified by a gated residual in final metric-depth space.

Evidence:

- Method equation
- Table `tab:ablation`
- `zoedepth/models/zoedepth/zoedepth_v1.py`

### If reviewer says "The gains are modest"

Response:

> The method is intentionally local and conservative. It is designed to reduce boundary-region metric failures without disrupting the base metric prediction. The paper avoids SOTA claims and instead reports boundary-specific metrics and limitations.

Evidence:

- Tables `tab:main-results`, `tab:hard-boundary`, `tab:ablation`
- Complexity table `tab:complexity`

### If reviewer says "NYU Boundary RMSE is not improved"

Response:

> This is acknowledged. NYU demonstrates metric preservation and Edge F1 improvement, not universal boundary-error improvement. This is why the paper separates structural edge localization from metric boundary-region error.

Evidence:

- Tables `tab:nyu-sync`, `tab:nyu-density`
- `logs/nyu_boundary_density_subsets_baseline_brrh.json`

## Recommended Paper Narrative

Use this sequence:

1. Metric MDE has strong global depth performance.
2. Boundary pixels still create foreground-background mixing.
3. RGB edges and depth discontinuities are not identical.
4. The target is boundary-local metric reliability, not only visual sharpness.
5. Foundation depth models provide strong structural priors but may not preserve the target metric pipeline.
6. BRRH keeps ZoeDepth as the metric base and applies a bounded local residual in final log-depth space.
7. KITTI supports boundary-band metric improvement.
8. NYU and external baselines show the difference between edge localization and metric boundary reliability.
9. Complexity profiling supports the lightweight trainable-refinement claim.
10. Control/surround-view relevance is framed as auxiliary perception support, not as a complete deployed system.
11. Limitations are stated directly, which keeps the paper credible rather than over-claiming.

## Final Rule

Every final claim should pass this test:

> Can it be supported by a table, figure, JSON log, or script in this repository?

If not, phrase it as a limitation, future work, or hypothesis rather than a result.
