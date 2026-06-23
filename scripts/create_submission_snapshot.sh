#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${1:-submission_snapshots/${STAMP}_brrh_zoedepth_itc}"
CREATE_TAR=0

if [[ "${OUT_DIR}" == "--tar" ]]; then
  OUT_DIR="submission_snapshots/${STAMP}_brrh_zoedepth_itc"
  CREATE_TAR=1
elif [[ "${2:-}" == "--tar" ]]; then
  CREATE_TAR=1
fi

echo "Running audit before creating snapshot..."
bash scripts/run_submission_audit.sh
audit_status="$?"

mkdir -p "$OUT_DIR"/manuscript "$OUT_DIR"/reports "$OUT_DIR"/figures "$OUT_DIR"/scripts

copy_file() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  else
    echo "Missing optional file: $src" >&2
  fi
}

copy_file paper_rewriting_output/final_paper/main.tex "$OUT_DIR/manuscript/main.tex"
copy_file paper_rewriting_output/final_paper/paper.pdf "$OUT_DIR/manuscript/paper.pdf"
copy_file paper_rewriting_output/final_paper/paper.docx "$OUT_DIR/manuscript/paper.docx"
copy_file reports/itc_cover_letter_and_highlights.md "$OUT_DIR/reports/itc_cover_letter_and_highlights.md"
copy_file reports/submission_package_readme.md "$OUT_DIR/reports/submission_package_readme.md"
copy_file reports/final_submission_gap_report.md "$OUT_DIR/reports/final_submission_gap_report.md"
copy_file reports/submission_metadata_template.md "$OUT_DIR/reports/submission_metadata_template.md"
copy_file reports/submission_metadata.sample.json "$OUT_DIR/reports/submission_metadata.sample.json"
copy_file reports/advisor_handoff_summary_cn.md "$OUT_DIR/reports/advisor_handoff_summary_cn.md"
copy_file reports/reviewer_ready_story_and_strategy.md "$OUT_DIR/reports/reviewer_ready_story_and_strategy.md"
copy_file reports/reviewer_readiness_audit.md "$OUT_DIR/reports/reviewer_readiness_audit.md"
copy_file reports/strict_reviewer_precheck.md "$OUT_DIR/reports/strict_reviewer_precheck.md"
copy_file reports/reviewer_response_playbook.md "$OUT_DIR/reports/reviewer_response_playbook.md"
copy_file reports/reviewer_evidence_matrix.md "$OUT_DIR/reports/reviewer_evidence_matrix.md"
copy_file reports/experiment_artifact_manifest.md "$OUT_DIR/reports/experiment_artifact_manifest.md"
copy_file reports/submission_reproducibility_checklist.md "$OUT_DIR/reports/submission_reproducibility_checklist.md"
copy_file reports/itc_submission_compliance_checklist.md "$OUT_DIR/reports/itc_submission_compliance_checklist.md"
copy_file reports/submission_readiness_check_latest.txt "$OUT_DIR/reports/submission_readiness_check_latest.txt"
copy_file reports/paper_metric_verification_latest.md "$OUT_DIR/reports/paper_metric_verification_latest.md"
copy_file reports/model_complexity_profile_latest.md "$OUT_DIR/reports/model_complexity_profile_latest.md"
copy_file reports/model_complexity_profile_latest.json "$OUT_DIR/reports/model_complexity_profile_latest.json"
copy_file reports/claim_boundary_check_latest.txt "$OUT_DIR/reports/claim_boundary_check_latest.txt"
copy_file reports/submission_audit_latest.md "$OUT_DIR/reports/submission_audit_latest.md"

copy_file paper_rewriting_output/final_paper/figures/boundary_zoedepth_architecture.png "$OUT_DIR/figures/boundary_zoedepth_architecture.png"
copy_file paper_rewriting_output/final_paper/figures/nyu_brrh_depthpro_top_samples.png "$OUT_DIR/figures/nyu_brrh_depthpro_top_samples.png"
copy_file paper_rewriting_output/final_paper/figures/brrh_noresidual_qualitative.png "$OUT_DIR/figures/brrh_noresidual_qualitative.png"

copy_file scripts/build_submission_package.sh "$OUT_DIR/scripts/build_submission_package.sh"
copy_file scripts/run_submission_audit.sh "$OUT_DIR/scripts/run_submission_audit.sh"
copy_file scripts/check_submission_readiness.py "$OUT_DIR/scripts/check_submission_readiness.py"
copy_file scripts/verify_paper_metrics.py "$OUT_DIR/scripts/verify_paper_metrics.py"
copy_file scripts/profile_model_complexity.py "$OUT_DIR/scripts/profile_model_complexity.py"
copy_file scripts/check_claim_boundaries.py "$OUT_DIR/scripts/check_claim_boundaries.py"
copy_file scripts/apply_submission_metadata.py "$OUT_DIR/scripts/apply_submission_metadata.py"

chmod +x "$OUT_DIR"/scripts/*.sh "$OUT_DIR"/scripts/*.py 2>/dev/null || true

cat > "$OUT_DIR/SNAPSHOT_MANIFEST.md" <<EOF
# BRRH-ZoeDepth Submission Snapshot

Created: $(date '+%Y-%m-%d %H:%M:%S %Z')

Source repository: $ROOT

Audit command:

\`\`\`bash
bash scripts/run_submission_audit.sh
\`\`\`

Audit exit code: $audit_status

Expected current interpretation:

- Exit code 0 means the package passed all automated checks.
- Exit code 1 is expected while real author metadata and submission declarations are still placeholders.
- Check \`reports/submission_audit_latest.md\` and \`reports/submission_readiness_check_latest.txt\` inside this snapshot for the exact status.

## Contents

- \`manuscript/\`: LaTeX source, PDF draft, Word draft.
- \`figures/\`: core manuscript figures.
- \`reports/\`: cover letter, evidence matrix, artifact manifest, strict reviewer precheck, advisor handoff, audit reports.
- \`scripts/\`: audit, metric verification, complexity profiling, and metadata-fill scripts.

## Final Human-Owned Steps

1. Fill real author and affiliation metadata.
2. Add truthful Funding, Conflict of Interest, Data Availability, and Code Availability statements.
3. Run \`python scripts/apply_submission_metadata.py reports/submission_metadata.completed.json\` in the source repository.
4. Run \`bash scripts/build_submission_package.sh\`.
5. Open the Word draft and check visual formatting against the ITC template.
EOF

cat > "$OUT_DIR/README_FIRST_CN.md" <<EOF
# 请先看这个文件

这是 BRRH-ZoeDepth 的论文审阅快照，用于发给导师、同门或投稿前自查。

## 推荐阅读顺序

1. \`manuscript/paper.pdf\`  
   论文正文 PDF。

2. \`reports/advisor_handoff_summary_cn.md\`  
   中文导师交接摘要：一句话故事、方法核心、主要实验结论、审稿风险。

3. \`reports/strict_reviewer_precheck.md\`  
   严苛审稿人预审：按 novelty、实验、风险、投稿完整性打分。

4. \`reports/reviewer_evidence_matrix.md\`  
   主张-证据矩阵：每个可以说的结论对应哪个表、图、日志或脚本。

5. \`reports/experiment_artifact_manifest.md\`  
   表格/图/日志/脚本对应关系，方便查证实验来源。

## 当前一句话定位

本文研究的不是“让深度图看起来更锐”，而是 ZoeDepth 这类 metric depth 模型在物体边界处的 boundary-local metric reliability：避免前景和背景深度被平均成虚假的中间深度。

## 当前自动审计状态

Audit exit code: $audit_status

- 指标追溯、模型复杂度和过强声明检查当前通过。
- 若 exit code 为 1，当前预期原因是作者信息、单位、通讯作者和 Funding/Conflict/Data/Code availability 等真实投稿信息尚未填写。
- 详细状态见 \`reports/submission_audit_latest.md\` 和 \`reports/submission_readiness_check_latest.txt\`。

## 不应过度宣称

- 不说 BRRH 全面超过 Depth Pro。
- 不说 BRRH 是 boundary sharpness SOTA。
- 不说 NYU 所有边界指标都改善。
- 不说本文验证了完整军事全景周视控制系统。

## 投稿前必须补

1. 真实作者、单位、通讯作者和邮箱。
2. Funding、Conflict of Interest、Data Availability、Code Availability。
3. 用 Word 打开 \`manuscript/paper.docx\`，对照 ITC 模板人工检查格式。
EOF

(
  cd "$OUT_DIR" || exit 2
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)

echo
echo "Snapshot created:"
echo "$OUT_DIR"

if [[ "$CREATE_TAR" -eq 1 ]]; then
  tar_path="${OUT_DIR%/}.tar.gz"
  tar -czf "$tar_path" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
  echo "Snapshot archive:"
  echo "$tar_path"
fi

if [[ "$audit_status" -ne 0 ]]; then
  echo
  echo "Audit returned attention-required status. Snapshot was still created for review."
fi

exit 0
