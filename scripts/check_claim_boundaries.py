#!/usr/bin/env python3
"""Check submission-facing text for over-strong BRRH-ZoeDepth claims.

The manuscript deliberately avoids claiming that BRRH beats Depth Pro overall,
is state of the art, or solves all boundary problems. This checker scans the
main manuscript and cover-letter draft for those risky positive claims while
allowing explicit caveats such as "do not claim" or "does not claim".
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCAN_FILES = [
    ROOT / "paper_rewriting_output" / "final_paper" / "main.tex",
    ROOT / "reports" / "itc_cover_letter_and_highlights.md",
]


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    reason: str


RULES = [
    Rule(
        "sota_claim",
        re.compile(r"\b(state[- ]of[- ]the[- ]art|SOTA)\b", re.I),
        "Avoid claiming state-of-the-art boundary sharpness.",
    ),
    Rule(
        "beats_depthpro",
        re.compile(r"\bBRRH[- ]?ZoeDepth\b.{0,120}\b(beats?|outperforms?|surpasses?|is superior to)\b.{0,80}\bDepth Pro\b", re.I),
        "Avoid claiming BRRH beats Depth Pro overall.",
    ),
    Rule(
        "depthpro_inferior",
        re.compile(r"\bDepth Pro\b.{0,120}\b(is inferior to|is worse than|underperforms)\b.{0,80}\bBRRH", re.I),
        "Avoid framing Depth Pro as generally inferior.",
    ),
    Rule(
        "universal_solution",
        re.compile(r"\b(fully solves?|completely solves?|universal(?:ly)? solves?|solves all)\b.{0,120}\b(boundar|foreground|occlusion|leakage|sharpness)", re.I),
        "Avoid claiming the method fully solves boundary ambiguity.",
    ),
    Rule(
        "all_boundary_metrics",
        re.compile(r"\b(improves?|better|best)\b.{0,80}\b(all|every)\b.{0,80}\bboundary\b.{0,40}\b(metrics?|errors?)", re.I),
        "Avoid claiming all boundary metrics improve.",
    ),
]

NEGATION_HINTS = (
    "do not",
    "does not",
    "not claim",
    "not a claim",
    "avoid",
    "should not",
    "cannot",
    "not state",
)


def is_allowed_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 90): min(len(text), end + 90)].lower()
    return any(hint in window for hint in NEGATION_HINTS)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main() -> int:
    problems: list[str] = []

    for path in SCAN_FILES:
        if not path.exists():
            problems.append(f"Missing scan file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for rule in RULES:
            for match in rule.pattern.finditer(text):
                if is_allowed_context(text, match.start(), match.end()):
                    continue
                snippet = re.sub(r"\s+", " ", text[match.start():match.end()]).strip()
                problems.append(
                    f"{path.relative_to(ROOT)}:{line_number(text, match.start())}: "
                    f"{rule.name}: {rule.reason} Matched: {snippet}"
                )

    print("Claim boundary check")
    if problems:
        print(f"Problems: {len(problems)}")
        for item in problems:
            print(f"- {item}")
        return 1

    print("Problems: 0")
    print("No over-strong submission-facing claims detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

