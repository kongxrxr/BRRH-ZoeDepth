#!/usr/bin/env python3
"""Apply human-provided submission metadata to the manuscript files.

The script intentionally refuses placeholder-looking values. Fill a copy of
reports/submission_metadata.sample.json with real author/declaration metadata,
then run this script before rebuilding the submission package.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "paper_rewriting_output" / "final_paper" / "main.tex"
COVER_LETTER = ROOT / "reports" / "itc_cover_letter_and_highlights.md"

PLACEHOLDER_PATTERNS = [
    "Full Name",
    "Department, University",
    "University or Institute",
    "example.com",
    "[",
    "]",
    "to be completed",
]


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_metadata(data)
    return data


def require_text(data: dict, key: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise SystemExit(f"Missing required metadata field: {key}")
    reject_placeholder(value, key)
    return value


def reject_placeholder(value: str, key: str) -> None:
    lowered = value.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.lower() in lowered:
            raise SystemExit(f"Refusing placeholder-like value in {key}: {value!r}")


def validate_metadata(data: dict) -> None:
    authors = data.get("authors")
    affiliations = data.get("affiliations")
    if not isinstance(authors, list) or not authors:
        raise SystemExit("metadata.authors must be a non-empty list")
    if not isinstance(affiliations, list) or not affiliations:
        raise SystemExit("metadata.affiliations must be a non-empty list")

    affiliation_ids = {str(item.get("index", "")).strip() for item in affiliations}
    if "" in affiliation_ids:
        raise SystemExit("Each affiliation must contain a non-empty index")

    for i, author in enumerate(authors, start=1):
        for key in ("name", "affiliation_index", "email"):
            value = str(author.get(key, "")).strip()
            if not value:
                raise SystemExit(f"Author {i} is missing {key}")
            reject_placeholder(value, f"authors[{i}].{key}")
        if str(author["affiliation_index"]).strip() not in affiliation_ids:
            raise SystemExit(
                f"Author {i} uses unknown affiliation index: {author['affiliation_index']}"
            )

    for i, affiliation in enumerate(affiliations, start=1):
        text = str(affiliation.get("text", "")).strip()
        if not text:
            raise SystemExit(f"Affiliation {i} is missing text")
        reject_placeholder(text, f"affiliations[{i}].text")

    for key in (
        "corresponding_author",
        "corresponding_email",
        "funding",
        "conflict_of_interest",
        "data_availability",
        "code_availability",
    ):
        require_text(data, key)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def build_author_block(data: dict) -> str:
    author_names = []
    for author in data["authors"]:
        name = latex_escape(str(author["name"]).strip())
        aff = latex_escape(str(author["affiliation_index"]).strip())
        author_names.append(f"{name}$^{{{aff}}}$")

    affiliation_lines = []
    for affiliation in data["affiliations"]:
        idx = latex_escape(str(affiliation["index"]).strip())
        text = latex_escape(str(affiliation["text"]).strip())
        affiliation_lines.append(f"$^{{{idx}}}${text}")

    corr_name = latex_escape(require_text(data, "corresponding_author"))
    corr_email = latex_escape(require_text(data, "corresponding_email"))
    lines = [
        r"\author{",
        ", ".join(author_names) + r"\\",
        r"\\".join(affiliation_lines) + r"\\",
        f"Corresponding author: {corr_name}, {corr_email}",
        "}",
    ]
    return "\n".join(lines)


def build_declarations(data: dict) -> str:
    items = [
        ("Funding", require_text(data, "funding")),
        ("Conflict of Interest", require_text(data, "conflict_of_interest")),
        ("Data Availability", require_text(data, "data_availability")),
        ("Code Availability", require_text(data, "code_availability")),
    ]
    acknowledgements = str(data.get("acknowledgements", "")).strip()
    if acknowledgements:
        reject_placeholder(acknowledgements, "acknowledgements")
        items.append(("Acknowledgements", acknowledgements))

    body = ["\\section*{Declarations}"]
    for title, text in items:
        body.append(f"\\noindent\\textbf{{{title}.}} {latex_escape(text)}")
        body.append("")
    return "\n".join(body).strip() + "\n\n"


def update_main_tex(data: dict) -> None:
    text = MAIN_TEX.read_text(encoding="utf-8")
    author_block = build_author_block(data)

    text, count = re.subn(
        r"\\author\{kxr\\\\Affiliation to be completed\}",
        author_block,
        text,
        count=1,
    )
    if count == 0:
        text, count = re.subn(
            r"\\author\{.*?\}\n\\date\{\}",
            author_block + "\n\\date{}",
            text,
            count=1,
            flags=re.DOTALL,
        )
    if count == 0:
        raise SystemExit("Could not locate LaTeX author block to replace")

    declarations = build_declarations(data)
    if "\\section*{Declarations}" in text:
        text = re.sub(
            r"\\section\*\{Declarations\}.*?(?=\\begin\{thebibliography\})",
            declarations,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        text = text.replace("\\begin{thebibliography}{10}", declarations + "\\begin{thebibliography}{10}", 1)

    MAIN_TEX.write_text(text, encoding="utf-8")


def update_cover_letter(data: dict) -> None:
    text = COVER_LETTER.read_text(encoding="utf-8")
    corr_name = require_text(data, "corresponding_author")
    corr_email = require_text(data, "corresponding_email")

    corr_affiliation = ""
    author_by_name = {
        str(author["name"]).strip(): str(author["affiliation_index"]).strip()
        for author in data["authors"]
    }
    affiliation_by_id = {
        str(item["index"]).strip(): str(item["text"]).strip()
        for item in data["affiliations"]
    }
    aff_id = author_by_name.get(corr_name)
    if aff_id:
        corr_affiliation = affiliation_by_id.get(aff_id, "")

    text = text.replace("[Full name of corresponding author]", corr_name)
    text = text.replace("[Corresponding author]", corr_name)
    text = text.replace("[Affiliation]", corr_affiliation)
    text = text.replace("[Email]", corr_email)
    text = text.replace(
        "[Complete conflict-of-interest, funding, and data/code availability statements as required.]",
        "Funding, conflict-of-interest, data availability, and code availability statements are included in the manuscript.",
    )
    text = text.replace(
        "[Complete author, funding, conflict-of-interest, and data/code statements.]",
        "Author information, funding, conflict-of-interest, data availability, and code availability statements are included in the manuscript.",
    )
    COVER_LETTER.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_json", type=Path, help="Path to completed metadata JSON")
    args = parser.parse_args()

    data = load_metadata(args.metadata_json)
    update_main_tex(data)
    update_cover_letter(data)
    print("Applied submission metadata to:")
    print(f"- {MAIN_TEX}")
    print(f"- {COVER_LETTER}")


if __name__ == "__main__":
    main()
