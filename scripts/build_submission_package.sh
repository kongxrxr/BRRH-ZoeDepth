#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT/paper_rewriting_output/final_paper"
PYTHON_ZOE="${PYTHON_ZOE:-/home/kxr/miniconda3/envs/zoe/bin/python}"

cd "$PAPER_DIR" || exit 2

echo "Building PDF with pdflatex pass 1..."
pdflatex -interaction=nonstopmode main.tex
latex_status_1="$?"

echo
echo "Building PDF with pdflatex pass 2..."
pdflatex -interaction=nonstopmode main.tex
latex_status_2="$?"

if [[ "$latex_status_1" -ne 0 || "$latex_status_2" -ne 0 ]]; then
  echo "pdflatex failed; not continuing to submission audit." >&2
  exit 1
fi

cp main.pdf paper.pdf

echo
echo "Building DOCX draft..."
"$PYTHON_ZOE" build_word_docx.py
docx_status="$?"
if [[ "$docx_status" -ne 0 ]]; then
  echo "DOCX build failed; not continuing to submission audit." >&2
  exit 1
fi

cd "$ROOT" || exit 2

echo
echo "Running full submission audit..."
bash scripts/run_submission_audit.sh
audit_status="$?"

echo
echo "Submission package artifacts:"
ls -lh \
  paper_rewriting_output/final_paper/main.tex \
  paper_rewriting_output/final_paper/paper.pdf \
  paper_rewriting_output/final_paper/paper.docx \
  reports/submission_audit_latest.md

exit "$audit_status"

