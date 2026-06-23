#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

mkdir -p reports

READINESS_LOG="reports/submission_readiness_check_latest.txt"
METRIC_REPORT="reports/paper_metric_verification_latest.md"
COMPLEXITY_JSON="reports/model_complexity_profile_latest.json"
COMPLEXITY_REPORT="reports/model_complexity_profile_latest.md"
CLAIM_LOG="reports/claim_boundary_check_latest.txt"
AUDIT_REPORT="reports/submission_audit_latest.md"
PYTHON_ZOE="${PYTHON_ZOE:-/home/kxr/miniconda3/envs/zoe/bin/python}"

echo "Running submission readiness check..."
python scripts/check_submission_readiness.py 2>&1 | tee "$READINESS_LOG"
readiness_status="${PIPESTATUS[0]}"

echo
echo "Running paper metric verification..."
python scripts/verify_paper_metrics.py --write-report "$METRIC_REPORT"
metric_status="$?"

echo
echo "Running model complexity profile..."
"$PYTHON_ZOE" scripts/profile_model_complexity.py --output "$COMPLEXITY_JSON" --markdown "$COMPLEXITY_REPORT"
complexity_status="$?"

echo
echo "Running claim boundary check..."
python scripts/check_claim_boundaries.py 2>&1 | tee "$CLAIM_LOG"
claim_status="${PIPESTATUS[0]}"

overall_status="PASS"
if [[ "$readiness_status" -ne 0 || "$metric_status" -ne 0 || "$complexity_status" -ne 0 || "$claim_status" -ne 0 ]]; then
  overall_status="ATTENTION_REQUIRED"
fi

cat > "$AUDIT_REPORT" <<EOF
# Submission Audit Latest

Date: $(date '+%Y-%m-%d %H:%M:%S %Z')

Command:

\`\`\`bash
bash scripts/run_submission_audit.sh
\`\`\`

Overall status: ${overall_status}

## Component Results

| Check | Exit code | Output |
|---|---:|---|
| Submission readiness | ${readiness_status} | \`${READINESS_LOG}\` |
| Paper metric verification | ${metric_status} | \`${METRIC_REPORT}\` |
| Model complexity profile | ${complexity_status} | \`${COMPLEXITY_REPORT}\` |
| Claim boundary check | ${claim_status} | \`${CLAIM_LOG}\` |

## Interpretation

- Exit code 0 means the component passed.
- A nonzero submission-readiness exit can be expected while author metadata or required submission statements are still placeholders.
- A nonzero metric-verification exit means at least one reported manuscript metric could not be traced to the configured JSON logs and should be fixed before submission.
- A nonzero complexity-profile exit means the parameter-count evidence could not be regenerated and should not be cited until fixed.
- A nonzero claim-boundary exit means the manuscript or cover-letter draft contains an over-strong claim such as SOTA or beating Depth Pro overall.

EOF

echo
echo "Wrote audit report: $AUDIT_REPORT"

if [[ "$overall_status" == "PASS" ]]; then
  exit 0
fi

exit 1
