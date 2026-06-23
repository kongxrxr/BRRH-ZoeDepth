#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ZOE="${PYTHON_ZOE:-/home/kxr/miniconda3/envs/zoe/bin/python}"
PAPERSPINE_DIR="$ROOT/paper_rewriting_output_ieee_access"
METADATA_GUARD="$PAPERSPINE_DIR/metadata_guard.md"
FINAL_REPORT="$PAPERSPINE_DIR/final_submission_readiness.md"

cd "$ROOT" || exit 2

echo "Building IEEE Access manuscript and running standard guards..."
bash scripts/build_ieee_access_paperspine.sh
build_status="$?"

echo
echo "Running strict metadata guard for final submission..."
"$PYTHON_ZOE" scripts/check_ieee_access_metadata.py --write "$METADATA_GUARD"
metadata_status="$?"

overall="PASS"
if [[ "$build_status" -ne 0 || "$metadata_status" -ne 0 ]]; then
  overall="ATTENTION_REQUIRED"
fi

cat > "$FINAL_REPORT" <<EOF
# IEEE Access Final Submission Readiness

Date: $(date '+%Y-%m-%d %H:%M:%S %Z')

Overall status: ${overall}

| Check | Exit code | Report |
|---|---:|---|
| IEEE Access build/PaperSpine/LaTeX/BibTeX | ${build_status} | \`paper_rewriting_output_ieee_access/latex_report.md\` |
| Human metadata guard | ${metadata_status} | \`paper_rewriting_output_ieee_access/metadata_guard.md\` |

## Interpretation

- \`PASS\` means the generated IEEE Access manuscript has no detected placeholder blocker.
- \`ATTENTION_REQUIRED\` usually means real author, affiliation, funding, or data/code availability information is still missing.
- DOI placeholder is treated as a warning because IEEE normally assigns DOI after acceptance.

EOF

echo
echo "Final submission readiness report: $FINAL_REPORT"

if [[ "$overall" == "PASS" ]]; then
  exit 0
fi

exit 1
