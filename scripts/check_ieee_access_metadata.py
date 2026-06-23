#!/usr/bin/env python3
"""Check human-owned IEEE Access submission metadata.

This guard is intentionally narrower than the scientific audits. It does not
judge the paper's method or metrics; it only checks whether author/declaration
fields are still placeholders.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_rewriting_output_ieee_access"
FINAL = OUT / "final_paper"
TEX_PATH = FINAL / "main.tex"
METADATA_JSON = OUT / "submission_metadata.json"
TEMPLATE_JSON = OUT / "submission_metadata_template.json"

PLACEHOLDER_PATTERNS = [
    ("author", r"Author Name|First Author|Second Author", "blocker"),
    ("affiliation", r"Affiliation to be completed|Department, University, City, Country", "blocker"),
    ("email", r"email@domain\.com|first@author\.edu|second@author\.edu", "blocker"),
    ("funding", r"funding/support details|\[funding agency/project number\]", "blocker"),
    ("acknowledgment", r"\[names/institutions\]|useful discussions or computational support", "blocker"),
    ("code/data URL", r"\[repository or reviewer-link URL\]|repository or review-link URL", "blocker"),
    ("doi", r"10\.1109/ACCESS\.2026\.0000000", "warning"),
]

REQUIRED_METADATA_KEYS = [
    "doi",
    "author_latex",
    "address_latex",
    "corresponding_author_latex",
    "markboth_latex",
    "tfootnote_latex",
    "acknowledgment_latex",
    "data_code_availability_latex",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def has_placeholder(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def load_metadata() -> tuple[dict[str, str], str | None]:
    if not METADATA_JSON.exists():
        return {}, "submission_metadata.json not found; generated manuscript is using default placeholders"
    try:
        data = json.loads(read_text(METADATA_JSON))
    except json.JSONDecodeError as exc:
        return {}, f"submission_metadata.json is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return {}, "submission_metadata.json must contain a JSON object"
    return {str(k): str(v) for k, v in data.items()}, None


def make_report() -> tuple[str, int]:
    lines: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []

    lines.append("# IEEE Access Metadata Guard")
    lines.append("")
    lines.append(f"- TeX source: `{TEX_PATH}`")
    lines.append(f"- Metadata JSON: `{METADATA_JSON}`")
    lines.append(f"- Template JSON: `{TEMPLATE_JSON}`")
    lines.append("")

    if not TEX_PATH.exists():
        lines.append("Status: BLOCKED")
        lines.append("")
        lines.append(f"Missing TeX source: `{TEX_PATH}`")
        return "\n".join(lines) + "\n", 2

    tex = read_text(TEX_PATH)
    metadata, metadata_error = load_metadata()

    if metadata_error:
        warnings.append(metadata_error)

    missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if METADATA_JSON.exists() and missing_keys:
        blockers.append("submission_metadata.json is missing required key(s): " + ", ".join(missing_keys))

    lines.append("## Placeholder Scan")
    lines.append("")
    lines.append("| Field | Status | Evidence |")
    lines.append("|---|---|---|")
    for label, pattern, severity in PLACEHOLDER_PATTERNS:
        found_in_tex = has_placeholder(tex, pattern)
        found_in_metadata = any(has_placeholder(value, pattern) for value in metadata.values())
        if found_in_tex or found_in_metadata:
            if severity == "warning":
                warnings.append(f"{label} placeholder remains")
            else:
                blockers.append(f"{label} placeholder remains")
            evidence = "main.tex" if found_in_tex else "submission_metadata.json"
            status = "ATTENTION_REQUIRED"
        else:
            evidence = "no placeholder pattern found"
            status = "OK"
        lines.append(f"| {label} | {status} | {evidence} |")

    lines.append("")
    lines.append("## Required Metadata Keys")
    lines.append("")
    if not METADATA_JSON.exists():
        lines.append(
            "No completed `submission_metadata.json` exists yet. Copy "
            "`submission_metadata_template.json` to `submission_metadata.json`, fill real values, "
            "and rerun `scripts/build_ieee_access_paperspine.sh`."
        )
    else:
        lines.append("| Key | Status |")
        lines.append("|---|---|")
        for key in REQUIRED_METADATA_KEYS:
            status = "present" if key in metadata and metadata[key].strip() else "missing"
            lines.append(f"| `{key}` | {status} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if blockers:
        lines.append(f"Status: ATTENTION_REQUIRED ({len(blockers)} blocker(s))")
        lines.append("")
        for item in blockers:
            lines.append(f"- {item}")
    else:
        lines.append("Status: PASS")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for item in warnings:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n", 1 if blockers else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, help="Write the Markdown report to this path")
    parser.add_argument("--report-only", action="store_true", help="Always exit 0 after writing/printing")
    args = parser.parse_args()

    report, status = make_report()
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    if args.report_only:
        return 0
    return status


if __name__ == "__main__":
    sys.exit(main())
