#!/usr/bin/env python3
"""Check whether the BRRH-ZoeDepth manuscript is ready for submission.

This script is intentionally conservative: it reports a failing status when
human-owned metadata such as author affiliations or cover-letter placeholders
are still present. It does not judge scientific merit; it checks whether the
current submission package has the main files and formatting signals needed for
the final journal pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper_rewriting_output" / "final_paper"
TEX_PATH = PAPER_DIR / "main.tex"
LOG_PATH = PAPER_DIR / "main.log"
PDF_PATH = PAPER_DIR / "paper.pdf"
DOCX_PATH = PAPER_DIR / "paper.docx"

REQUIRED_REPORTS = [
    ROOT / "reports" / "reviewer_readiness_audit.md",
    ROOT / "reports" / "reviewer_ready_story_and_strategy.md",
    ROOT / "reports" / "advisor_handoff_summary_cn.md",
    ROOT / "reports" / "submission_reproducibility_checklist.md",
    ROOT / "reports" / "experiment_artifact_manifest.md",
    ROOT / "reports" / "strict_reviewer_precheck.md",
    ROOT / "reports" / "reviewer_response_playbook.md",
    ROOT / "reports" / "reviewer_evidence_matrix.md",
    ROOT / "reports" / "itc_submission_compliance_checklist.md",
    ROOT / "reports" / "itc_cover_letter_and_highlights.md",
    ROOT / "reports" / "final_submission_gap_report.md",
    ROOT / "reports" / "submission_metadata_template.md",
    ROOT / "reports" / "submission_package_readme.md",
    ROOT / "reports" / "paper_metric_verification_latest.md",
    ROOT / "reports" / "model_complexity_profile_latest.md",
]

REQUIRED_FIGURES = [
    PAPER_DIR / "figures" / "boundary_zoedepth_architecture.png",
    PAPER_DIR / "figures" / "brrh_noresidual_qualitative.png",
    PAPER_DIR / "figures" / "nyu_brrh_depthpro_top_samples.png",
]

HARD_PLACEHOLDERS = [
    "Affiliation to be completed",
    "[Corresponding author]",
    "[Author name]",
    "name@email.com",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_tex_commands(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = re.sub(r"[{}$]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def check(condition: bool, label: str, detail: str, failures: list[str], warnings: list[str], hard: bool = True) -> None:
    mark = "PASS" if condition else ("FAIL" if hard else "WARN")
    print(f"[{mark}] {label}: {detail}")
    if not condition:
        if hard:
            failures.append(label)
        else:
            warnings.append(label)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    print(f"Submission readiness check for: {ROOT}")

    check(TEX_PATH.exists(), "LaTeX source", str(TEX_PATH), failures, warnings)
    if not TEX_PATH.exists():
        return 2

    tex = read_text(TEX_PATH)
    log = read_text(LOG_PATH) if LOG_PATH.exists() else ""

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    abstract_plain = strip_tex_commands(abstract_match.group(1)) if abstract_match else ""
    abstract_chars = len(abstract_plain)
    check(
        1000 <= abstract_chars <= 1500,
        "ITC abstract length",
        f"{abstract_chars} printed-character approximation; expected 1000-1500",
        failures,
        warnings,
    )

    keywords_match = re.search(r"\\textbf\{Keywords:\}(.*?)(?:\\vspace|\\section)", tex, re.S)
    keywords = []
    if keywords_match:
        keywords_text = strip_tex_commands(keywords_match.group(1)).strip(" .")
        keywords = [item.strip() for item in keywords_text.split(",") if item.strip()]
    check(
        5 <= len(keywords) <= 7,
        "Keyword count",
        f"{len(keywords)} comma-separated keywords",
        failures,
        warnings,
    )

    cite_count = tex.count(r"\cite")
    bib_count = len(re.findall(r"\\bibitem\{", tex))
    check(cite_count > 0 and bib_count > 0, "Citations and references", f"{cite_count} citations, {bib_count} references", failures, warnings)

    figure_count = len(re.findall(r"\\begin\{figure\*?\}", tex))
    table_count = len(re.findall(r"\\begin\{table\*?\}", tex))
    check(figure_count >= 3, "Figure count", f"{figure_count} figure environments", failures, warnings)
    check(table_count >= 5, "Table count", f"{table_count} table environments", failures, warnings)

    missing_figures = [str(path.relative_to(ROOT)) for path in REQUIRED_FIGURES if not path.exists()]
    check(not missing_figures, "Required figure files", "all present" if not missing_figures else ", ".join(missing_figures), failures, warnings)

    check(PDF_PATH.exists() and PDF_PATH.stat().st_size > 100_000, "PDF artifact", str(PDF_PATH), failures, warnings)
    check(DOCX_PATH.exists() and DOCX_PATH.stat().st_size > 5_000, "DOCX artifact", str(DOCX_PATH), failures, warnings)

    missing_reports = [str(path.relative_to(ROOT)) for path in REQUIRED_REPORTS if not path.exists()]
    check(not missing_reports, "Reviewer/submission reports", "all present" if not missing_reports else ", ".join(missing_reports), failures, warnings)

    if log:
        bad_log_patterns = [
            "Undefined references",
            "Citation",
            "Rerun to get cross-references",
            "Overfull",
        ]
        bad_log_hits = [pattern for pattern in bad_log_patterns if pattern in log]
        check(not bad_log_hits, "LaTeX critical warnings", "none found" if not bad_log_hits else ", ".join(bad_log_hits), failures, warnings)
    else:
        check(False, "LaTeX log", "main.log not found; rebuild PDF before submission", failures, warnings, hard=False)

    combined_submission_text = tex
    cover_letter = ROOT / "reports" / "itc_cover_letter_and_highlights.md"
    if cover_letter.exists():
        combined_submission_text += "\n" + read_text(cover_letter)
    placeholder_hits = [item for item in HARD_PLACEHOLDERS if item in combined_submission_text]
    check(
        not placeholder_hits,
        "Human metadata placeholders",
        "none found" if not placeholder_hits else ", ".join(placeholder_hits),
        failures,
        warnings,
    )

    required_statement_terms = [
        "Funding",
        "Conflict",
        "Data availability",
        "Code availability",
    ]
    missing_terms = [term for term in required_statement_terms if term.lower() not in tex.lower()]
    check(
        not missing_terms,
        "Submission statements in manuscript",
        "all present" if not missing_terms else "missing or not explicit: " + ", ".join(missing_terms),
        failures,
        warnings,
        hard=False,
    )

    print()
    print(f"Summary: {len(failures)} hard blocker(s), {len(warnings)} warning(s).")
    if failures:
        print("Hard blockers:")
        for item in failures:
            print(f"- {item}")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"- {item}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
