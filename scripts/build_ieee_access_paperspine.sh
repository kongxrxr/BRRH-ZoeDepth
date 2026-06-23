#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ZOE="${PYTHON_ZOE:-/home/kxr/miniconda3/envs/zoe/bin/python}"
PAPERSPINE_DIR="$ROOT/paper_rewriting_output_ieee_access"
FINAL_DIR="$PAPERSPINE_DIR/final_paper"
INTEGRITY_SCRIPT="/home/kxr/.codex/skills/paper-spine/scripts/integrity_audit.py"
ARTIFACT_SCRIPT="/home/kxr/.codex/skills/paper-spine/scripts/artifact_check.py"
LATEX_GUARD_SCRIPT="/home/kxr/.codex/skills/paper-spine/scripts/latex_guard.py"
METADATA_GUARD_SCRIPT="$ROOT/scripts/check_ieee_access_metadata.py"

cd "$ROOT" || exit 2

echo "Generating IEEE Access PaperSpine branch..."
"$PYTHON_ZOE" scripts/create_ieee_access_paperspine_version.py
gen_status="$?"
if [[ "$gen_status" -ne 0 ]]; then
  echo "IEEE Access generator failed." >&2
  exit "$gen_status"
fi

echo "Rendering IEEE-style Figure 1..."
"$PYTHON_ZOE" scripts/render_ieee_figure1.py \
  --pdf "$FINAL_DIR/figures/boundary_zoedepth_architecture_ieee.pdf" \
  --png "$FINAL_DIR/figures/boundary_zoedepth_architecture_ieee.png"
fig_status="$?"
if [[ "$fig_status" -ne 0 ]]; then
  echo "IEEE Figure 1 rendering failed." >&2
  exit "$fig_status"
fi

echo "Rendering IEEE-style qualitative figures..."
"$PYTHON_ZOE" scripts/render_ieee_qualitative_figures.py --fig-dir "$FINAL_DIR/figures"
qual_fig_status="$?"
if [[ "$qual_fig_status" -ne 0 ]]; then
  echo "IEEE qualitative figure rendering failed." >&2
  exit "$qual_fig_status"
fi

cd "$FINAL_DIR" || exit 2

echo
echo "Compiling IEEE Access LaTeX pass 1..."
pdflatex -interaction=nonstopmode main.tex
latex_status_1="$?"

echo
echo "Running BibTeX..."
bibtex main
bibtex_status="$?"

echo
echo "Compiling IEEE Access LaTeX pass 2..."
pdflatex -interaction=nonstopmode main.tex
latex_status_2="$?"

echo
echo "Compiling IEEE Access LaTeX pass 3..."
pdflatex -interaction=nonstopmode main.tex
latex_status_3="$?"

if [[ "$latex_status_1" -ne 0 || "$bibtex_status" -ne 0 || "$latex_status_2" -ne 0 || "$latex_status_3" -ne 0 ]]; then
  echo "IEEE Access pdflatex failed." >&2
  exit 1
fi

cp main.pdf paper.pdf

cd "$ROOT" || exit 2

echo
echo "Running PaperSpine integrity audit..."
"$PYTHON_ZOE" "$INTEGRITY_SCRIPT" "$PAPERSPINE_DIR" --markdown --write
integrity_status="$?"

echo
echo "Running PaperSpine artifact check..."
"$PYTHON_ZOE" "$ARTIFACT_SCRIPT" "$PAPERSPINE_DIR" --markdown --write
artifact_status="$?"

echo
echo "Running LaTeX structural guard..."
"$PYTHON_ZOE" "$LATEX_GUARD_SCRIPT" "$FINAL_DIR/main.tex" --bib "$FINAL_DIR/references.bib" --markdown > "$PAPERSPINE_DIR/latex_guard.md"
latex_guard_status="$?"

echo
echo "Running IEEE Access metadata guard..."
"$PYTHON_ZOE" "$METADATA_GUARD_SCRIPT" --write "$PAPERSPINE_DIR/metadata_guard.md"
metadata_guard_status="$?"

LOG_PATH="$FINAL_DIR/main.log"
PDF_PATH="$FINAL_DIR/paper.pdf"
critical_hits="$(rg -n "Undefined references|Citation .* undefined|Reference .* undefined|Rerun to get cross-references|Fatal error|Emergency stop|Class .* Error" "$LOG_PATH" || true)"
overfull_count="$(rg -c "Overfull" "$LOG_PATH" || true)"
underfull_count="$(rg -c "Underfull" "$LOG_PATH" || true)"
font_warning_count="$(rg -c "Font Warning" "$LOG_PATH" || true)"

compile_status="PASS"
if [[ -n "$critical_hits" ]]; then
  compile_status="ATTENTION_REQUIRED"
fi

cat > "$PAPERSPINE_DIR/latex_report.md" <<EOF
# LaTeX Report

Target: IEEE Access

Template source: \`/home/kxr/ACCESS_latex_template_20260513\`

Main source: \`paper_rewriting_output_ieee_access/final_paper/main.tex\`

PDF output: \`paper_rewriting_output_ieee_access/final_paper/paper.pdf\`

Compile status: ${compile_status}

PaperSpine integrity status: ${integrity_status}

PaperSpine artifact status: ${artifact_status}

LaTeX structural guard status: ${latex_guard_status}

BibTeX status: ${bibtex_status}

Metadata guard status: ${metadata_guard_status}

## Log Summary

- Overfull messages: ${overfull_count}
- Underfull messages: ${underfull_count}
- Font warnings: ${font_warning_count}

## Critical Hits

\`\`\`text
${critical_hits:-none}
\`\`\`

## Notes

- The IEEE Access class emits repeated overfull/underfull page-header messages with the supplied template assets; these are not fatal.
- Human metadata placeholders remain intentionally unresolved until the user provides real author, affiliation, DOI, funding, and correspondence information. See \`metadata_guard.md\`.
EOF

cat > "$PAPERSPINE_DIR/final_artifact_manifest.md" <<EOF
# Final Artifact Manifest

Date: $(date '+%Y-%m-%d %H:%M:%S %Z')

| Artifact | Path | Status |
|---|---|---|
| IEEE Access LaTeX source | \`paper_rewriting_output_ieee_access/final_paper/main.tex\` | present |
| BibTeX database | \`paper_rewriting_output_ieee_access/final_paper/references.bib\` | present |
| IEEE Access PDF | \`paper_rewriting_output_ieee_access/final_paper/paper.pdf\` | present |
| PaperSpine config | \`paper_rewriting_output_ieee_access/paper_spine_config.json\` | present |
| Writing rationale matrix | \`paper_rewriting_output_ieee_access/writing_rationale_matrix.md\` | present |
| Integrity audit | \`paper_rewriting_output_ieee_access/integrity_audit.md\` | present |
| Artifact check | \`paper_rewriting_output_ieee_access/artifact_check.md\` | present |
| LaTeX guard | \`paper_rewriting_output_ieee_access/latex_guard.md\` | present |
| Metadata guard | \`paper_rewriting_output_ieee_access/metadata_guard.md\` | present |
| LaTeX report | \`paper_rewriting_output_ieee_access/latex_report.md\` | present |

Remaining human-owned blockers:

- Real IEEE Access author and affiliation block.
- Real corresponding author email.
- Real DOI only after IEEE assignment; current value is placeholder.
- Funding/support footnote.
- Data/code availability or manuscript declarations if required by submission workflow.
EOF

echo
echo "IEEE Access artifacts:"
ls -lh "$FINAL_DIR/main.tex" "$PDF_PATH" "$PAPERSPINE_DIR/integrity_audit.md" "$PAPERSPINE_DIR/artifact_check.md" "$PAPERSPINE_DIR/latex_guard.md" "$PAPERSPINE_DIR/latex_report.md"

if [[ "$compile_status" != "PASS" || "$integrity_status" -ne 0 || "$artifact_status" -ne 0 || "$latex_guard_status" -ne 0 ]]; then
  exit 1
fi

exit 0
