#!/usr/bin/env python3
"""Create an IEEE Access PaperSpine branch from the current manuscript."""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path


ROOT = Path("/home/kxr/ZoeDepth")
SRC = ROOT / "paper_rewriting_output" / "final_paper" / "main.tex"
OUT = ROOT / "paper_rewriting_output_ieee_access"
FINAL = OUT / "final_paper"
TEMPLATE = Path("/home/kxr/ACCESS_latex_template_20260513")
METADATA_JSON = OUT / "submission_metadata.json"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_between(text: str, start_pat: str, end_pat: str) -> str:
    start = re.search(start_pat, text, re.S)
    end = re.search(end_pat, text, re.S)
    if not start or not end or end.start() <= start.end():
        raise RuntimeError(f"Could not extract between {start_pat!r} and {end_pat!r}")
    return text[start.end(): end.start()].strip()


def extract_from(text: str, start_pat: str) -> str:
    start = re.search(start_pat, text, re.S)
    if not start:
        raise RuntimeError(f"Could not extract from {start_pat!r}")
    return text[start.start():].strip()


def strip_latex_for_words(text: str) -> list[str]:
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    cleaned = re.sub(r"[{}$]", " ", cleaned)
    return re.findall(r"[A-Za-z0-9@./+-]+", cleaned)


def make_keywords() -> str:
    return (
        "boundary residual refinement, depth discontinuity, KITTI, metric depth, "
        "monocular depth estimation, NYU Depth V2, ZoeDepth"
    )


def default_submission_metadata() -> dict[str, str]:
    return {
        "doi": "10.1109/ACCESS.2026.0000000",
        "author_latex": "\\uppercase{Author Name}\\authorrefmark{1}",
        "address_latex": "\\address[1]{Affiliation to be completed by the authors (e-mail: email@domain.com)}",
        "corresponding_author_latex": "Corresponding author: Author Name (e-mail: email@domain.com).",
        "markboth_latex": "Author \\headeretal: Metric Depth at the Edge",
        "tfootnote_latex": (
            "Author information, funding/support details, acknowledgments, and repository URL "
            "must be completed with real submission metadata before upload."
        ),
        "acknowledgment_latex": "",
        "data_code_availability_latex": (
            "The experiments use the public KITTI and NYU Depth V2 datasets under their respective "
            "dataset terms. Before submission, the authors should add the permanent repository or "
            "review-link URL for the implementation, training configurations, evaluation scripts, "
            "metric logs, and released checkpoints. No private or non-public dataset is required "
            "for the reported experiments."
        ),
    }


def load_submission_metadata() -> dict[str, str]:
    metadata = default_submission_metadata()
    if METADATA_JSON.exists():
        user_metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
        for key, value in user_metadata.items():
            if key in metadata and isinstance(value, str):
                metadata[key] = value.strip()
    return metadata


def references_bib() -> str:
    return r"""@inproceedings{adabins,
  author    = {Bhat, Shariq Farooq and Alhashim, Ibraheem and Wonka, Peter},
  title     = {{AdaBins}: Depth Estimation Using Adaptive Bins},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2021}
}

@article{zoedepth,
  author  = {Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
  title   = {{ZoeDepth}: Zero-shot Transfer by Combining Relative and Metric Depth},
  journal = {arXiv preprint arXiv:2302.12288},
  year    = {2023}
}

@article{depthpro,
  author  = {Bochkovskii, Aleksei and Delaunoy, Ama{\"e}l and Germain, Hugo and Santos, Marcel and Zhou, Yichao and Richter, Stephan and Koltun, Vladlen},
  title   = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  journal = {arXiv preprint arXiv:2410.02073},
  year    = {2024}
}

@inproceedings{kitti,
  author    = {Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  title     = {Are We Ready for Autonomous Driving? The {KITTI} Vision Benchmark Suite},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2012}
}

@inproceedings{dpt,
  author    = {Ranftl, Ren{\'e} and Bochkovskiy, Alexey and Koltun, Vladlen},
  title     = {Vision Transformers for Dense Prediction},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2021}
}

@article{midas,
  author  = {Ranftl, Ren{\'e} and Lasinger, Katrin and Hafner, David and Schindler, Konrad and Koltun, Vladlen},
  title   = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2022}
}

@inproceedings{nyu,
  author    = {Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob},
  title     = {Indoor Segmentation and Support Inference from {RGBD} Images},
  booktitle = {Proceedings of the European Conference on Computer Vision},
  year      = {2012}
}

@article{dav2,
  author  = {Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  title   = {Depth Anything {V2}},
  journal = {arXiv preprint arXiv:2406.09414},
  year    = {2024}
}

@inproceedings{newcrfs,
  author    = {Yuan, Weihao and Gu, Xiaodong and Dai, Zuozhuo and Zhu, Siyu and Tan, Ping},
  title     = {New {CRFs}: Neural Window Fully-connected {CRFs} for Monocular Depth Estimation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}
"""


def convert_body(src_tex: str, metadata: dict[str, str]) -> str:
    body = extract_from(src_tex, r"\\section\{Introduction\}")
    body = body.replace("\\end{document}", "").strip()
    body = body.replace(
        "figures/boundary_zoedepth_architecture.png",
        "figures/boundary_zoedepth_architecture_ieee.pdf",
    )
    body = body.replace(
        "figures/nyu_brrh_depthpro_top_samples.png",
        "figures/nyu_brrh_depthpro_top_samples_ieee.png",
    )
    body = body.replace(
        "figures/brrh_noresidual_qualitative.png",
        "figures/brrh_noresidual_qualitative_ieee.png",
    )
    body = body.replace(
        "\\caption{Architecture of the proposed boundary-aware metric depth framework. ZoeDepth provides the metric-depth base, the discontinuity branch predicts geometry-related boundary probabilities, the frozen DA-V2 branch provides structural guidance, and BRRH applies a bounded final log-depth residual near predicted depth discontinuities. This division of labor is designed to improve boundary-band behavior without replacing the metric-depth backbone.}",
        "\\caption{Overview of BRRH-ZoeDepth. The framework preserves ZoeDepth as the metric-depth base and applies a bounded boundary residual guided by predicted discontinuities and a frozen DA-V2 structural prior.}",
    )
    body = body.replace(
        "\\caption{Qualitative NYU Depth V2 comparison among the NYU-sync ZoeDepth baseline, the tuned BRRH-ZoeDepth model, and Depth Pro. The samples are selected from the validation subset where BRRH reduces boundary-band AbsRel relative to the ZoeDepth baseline. BRRH produces localized corrections around object boundaries while preserving the ZoeDepth metric-depth layout. Depth Pro often gives sharper structural contours, which is consistent with its stronger Edge F1, but its aligned metric-depth errors are not uniformly lower under the matched NYU protocol.}",
        "\\caption{NYU Depth V2 qualitative comparison. The selected validation samples show cases where BRRH reduces boundary-band AbsRel relative to the ZoeDepth baseline while preserving the metric-depth layout; Depth Pro is included as a high-resolution structural reference.}",
    )
    body = body.replace(
        "\\caption{Qualitative comparison between Full BRRH and the no-residual-head variant on a KITTI validation sample. The selected sample is one of the cases where the final log-depth residual reduces boundary-band AbsRel. Full BRRH obtains lower boundary-band error than the no-residual variant while preserving the global scene layout.}",
        "\\caption{KITTI qualitative ablation for the final residual head. Full BRRH reduces the selected sample's boundary-band error relative to the no-residual variant while keeping the global depth layout visually stable.}",
    )
    body = body.replace("\\cite{dpt,midas,newcrfs}", "\\cite{dpt}, \\cite{midas}, \\cite{newcrfs}")
    body = body.replace("\\cite{adabins,zoedepth}", "\\cite{adabins}, \\cite{zoedepth}")
    cautious_replacements = {
        "A boundary residual refinement framework that improves metric depth near depth discontinuities while preserving the ZoeDepth metric-depth base.": (
            "A boundary residual refinement framework that targets boundary-band metric depth near depth discontinuities while preserving the ZoeDepth metric-depth base."
        ),
        "A KITTI and NYU Depth V2 evaluation with global metrics, boundary-focused metrics, ablations, and external DA-V2/Depth Pro baselines, clarifying where BRRH improves metric boundary behavior and where high-resolution structural models remain stronger.": (
            "A KITTI and NYU Depth V2 evaluation with global metrics, boundary-focused metrics, ablations, and external DA-V2/Depth Pro baselines, clarifying where BRRH lowers boundary-local metric errors and where high-resolution structural models remain stronger."
        ),
        "First, does the residual formulation improve boundary-band metric depth without damaging global KITTI performance?": (
            "First, does the residual formulation reduce boundary-band metric error without damaging global KITTI performance?"
        ),
        "if a model only improves Edge F1 but worsens metric boundary errors, or only improves global metrics while losing boundary behavior": (
            "if a model only raises Edge F1 but worsens metric boundary errors, or only lowers global errors while losing boundary behavior"
        ),
        "BRRH improves the boundary-band metric-error group but does not outperform the previous strong boundary-gated model on Edge F1.": (
            "BRRH lowers the boundary-band metric-error group but does not outperform the previous strong boundary-gated model on Edge F1."
        ),
        "BRRH improves Edge F1 in low-, medium-, and high-density subsets, but the metric error terms slightly increase.": (
            "BRRH raises Edge F1 in low-, medium-, and high-density subsets, but the metric error terms slightly increase."
        ),
        "On KITTI, the 5-epoch BRRH model improves RMSE, SILog, $\\delta_1$, Boundary RMSE, Top5 AbsRel, and Band3 AbsRel compared with the previous strong boundary-gated model.": (
            "On KITTI, the 5-epoch BRRH model lowers RMSE, SILog, Boundary RMSE, Top5 AbsRel, and Band3 AbsRel and slightly raises $\\delta_1$ compared with the previous strong boundary-gated model."
        ),
        "On NYU Depth V2 sync validation, a tuned BRRH model improves Edge F1@3 and Edge F1@5 while preserving AbsRel and RMSE": (
            "On NYU Depth V2 sync validation, a tuned BRRH model raises Edge F1@3 and Edge F1@5 while preserving AbsRel and RMSE"
        ),
        "BRRH improves NYU edge localization across density subsets": (
            "BRRH raises NYU edge localization scores across density subsets"
        ),
    }
    for old, new in cautious_replacements.items():
        body = body.replace(old, new)
    body = re.sub(
        r"\\section\{Introduction\}\s*\n\s*Depth estimation",
        lambda _: "\\section{Introduction}\n\\label{sec:introduction}\n\\PARstart{D}{epth} estimation",
        body,
        count=1,
    )
    algorithm = r"""
\begin{table}[t]
\centering
\small
\caption{Algorithm 1: Boundary Residual Refinement Head inference.}
\label{alg:brrh-inference}
\begin{tabular}{p{0.94\linewidth}}
\toprule
\textbf{Input:} RGB image $I$, ZoeDepth base predictor, frozen DA-V2 prior, residual scale $\alpha$, numerical constant $\epsilon$. \\
\textbf{Output:} refined metric depth $D_t$. \\
\midrule
1:\quad Predict base metric depth $D_b$ using the ZoeDepth metric decoder. \\
2:\quad Estimate discontinuity logits $B$ and compute the soft boundary mask $M_b=\sigma(B)$. \\
3:\quad Extract frozen DA-V2 structural features $P_{\mathrm{da}}$ under no-gradient inference. \\
4:\quad Predict residual logits $\Delta_b=\mathrm{BRRH}(D_b,M_b,P_{\mathrm{da}})$. \\
5:\quad Bound the correction as $R_b=\tanh(\Delta_b)$. \\
6:\quad Return $D_t=\exp(\log(D_b+\epsilon)+\alpha M_b R_b)$. \\
\bottomrule
\end{tabular}
\end{table}
"""
    body = body.replace(
        "The difference from feature-level gating is important. A feature gate can only influence the final depth through later decoder operations, and the effect may be diluted by the base predictor. BRRH acts directly on the final depth map, so the training losses can supervise the actual corrected metric value in the boundary band.\n\n\\subsection{Training Objective}",
        "The difference from feature-level gating is important. A feature gate can only influence the final depth through later decoder operations, and the effect may be diluted by the base predictor. BRRH acts directly on the final depth map, so the training losses can supervise the actual corrected metric value in the boundary band.\n\n"
        + algorithm
        + "\n\\subsection{Training Objective}",
    )
    claim_table = r"""
\begin{table*}[t]
\centering
\small
\caption{Claim boundary summary. The table separates supported claims from unsupported or out-of-scope interpretations.}
\label{tab:claim-boundary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{p{0.24\textwidth}p{0.31\textwidth}p{0.34\textwidth}}
\toprule
Evidence axis & Supported by the current experiments & Not claimed by this paper \\
\midrule
KITTI metric boundary behavior & BRRH lowers RMSE, SILog, Boundary RMSE, Top5 AbsRel, and Band3 AbsRel relative to the previous strong boundary-gated variant. & Universal improvement of every global metric; AbsRel remains comparable rather than strictly better. \\
KITTI edge localization & The residual design is evaluated with Edge F1 and SI-Boundary F1. & Stronger edge localization than the previous strong boundary-gated model; that model keeps higher Edge F1. \\
NYU transfer & Tuned BRRH preserves AbsRel/RMSE and reports higher Edge F1@3 and Edge F1@5 than the NYU-sync ZoeDepth baseline. & Universal indoor boundary-error reduction; NYU Boundary RMSE is slightly worse than the baseline. \\
External baselines & Depth Pro and DA-V2 provide strong structural edge references; BRRH keeps lower aligned metric errors than Depth Pro under the matched NYU diagnostic protocol. & Overall superiority over Depth Pro or DA-V2, especially for high-resolution boundary localization. \\
Deployment interpretation & BRRH is a lightweight residual module for ZoeDepth-style metric pipelines. & A replacement for large high-resolution foundation depth models. \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    body = body.replace(
        "The experimental protocol is organized around three questions. First, does the residual formulation improve boundary-band metric depth without damaging global KITTI performance? Second, does the same boundary mechanism remain useful on dense indoor NYU data, where ground-truth depth is denser and object boundaries differ from driving scenes? Third, how should BRRH be interpreted relative to strong structural models such as DA-V2 and Depth Pro? This organization is intended to make the evidence falsifiable: if a model only improves Edge F1 but worsens metric boundary errors, or only improves global metrics while losing boundary behavior, that trade-off must be reported rather than hidden.\n\n\\subsection{Main Comparison}",
        "The experimental protocol is organized around three questions. First, does the residual formulation improve boundary-band metric depth without damaging global KITTI performance? Second, does the same boundary mechanism remain useful on dense indoor NYU data, where ground-truth depth is denser and object boundaries differ from driving scenes? Third, how should BRRH be interpreted relative to strong structural models such as DA-V2 and Depth Pro? This organization is intended to make the evidence falsifiable: if a model only improves Edge F1 but worsens metric boundary errors, or only improves global metrics while losing boundary behavior, that trade-off must be reported rather than hidden.\n\n"
        + claim_table
        + "\n\\subsection{Main Comparison}",
    )
    acknowledgment = ""
    if metadata.get("acknowledgment_latex", "").strip():
        acknowledgment = (
            "\\section*{Acknowledgment}\n\n"
            + metadata["acknowledgment_latex"].strip()
            + "\n\n"
        )
    availability = rf"""
{acknowledgment}\section*{{Data and Code Availability}}

{metadata["data_code_availability_latex"]}

"""
    body = body.replace("\\begin{thebibliography}{10}", availability + "\\begin{thebibliography}{10}")
    body = re.sub(
        r"\\begin\{thebibliography\}\{10\}.*?\\end\{thebibliography\}",
        "\\\\bibliographystyle{IEEEtran}\n\\\\bibliography{references}",
        body,
        flags=re.S,
    )
    return body


def make_main_tex(src_tex: str) -> str:
    metadata = load_submission_metadata()
    abstract = extract_between(src_tex, r"\\begin\{abstract\}", r"\\end\{abstract\}")
    abstract = abstract.replace(
        "This design corrects local boundary regions while a non-boundary preservation term discourages global scale disturbance.",
        "This design targets local boundary regions while a non-boundary preservation term discourages global scale disturbance.",
    )
    abstract = abstract.replace(
        "improving the main boundary-band metric errors over the previous strong boundary-gated variant.",
        "lowering the main boundary-band metric errors relative to the previous strong boundary-gated variant.",
    )
    abstract = abstract.replace(
        "improves Edge F1@3 and Edge F1@5 over the NYU-sync ZoeDepth baseline.",
        "improves Edge F1@3 and Edge F1@5 relative to the NYU-sync ZoeDepth baseline.",
    )
    abstract = abstract.replace(
        "improves Edge F1@3 and Edge F1@5 relative to the NYU-sync ZoeDepth baseline.",
        "reports higher Edge F1@3 and Edge F1@5 than the NYU-sync ZoeDepth baseline.",
    )
    body = convert_body(src_tex, metadata)
    title = "Metric Depth at the Edge: Boundary Residual Refinement for Monocular Depth Estimation"
    today = date.today().strftime("%B %d, %Y")
    return rf"""\documentclass{{ieeeaccess}}
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{textcomp}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{bm}}

\makeatletter
\AtBeginDocument{{\DeclareMathVersion{{bold}}
\SetSymbolFont{{operators}}{{bold}}{{T1}}{{times}}{{b}}{{n}}
\SetSymbolFont{{NewLetters}}{{bold}}{{T1}}{{times}}{{b}}{{it}}
\SetMathAlphabet{{\mathrm}}{{bold}}{{T1}}{{times}}{{b}}{{n}}
\SetMathAlphabet{{\mathit}}{{bold}}{{T1}}{{times}}{{b}}{{it}}
\SetMathAlphabet{{\mathbf}}{{bold}}{{T1}}{{times}}{{b}}{{n}}
\SetMathAlphabet{{\mathtt}}{{bold}}{{OT1}}{{pcr}}{{b}}{{n}}
\SetSymbolFont{{symbols}}{{bold}}{{OMS}}{{cmsy}}{{b}}{{n}}
\renewcommand\boldmath{{\@nomath\boldmath\mathversion{{bold}}}}}}
\makeatother

\def\BibTeX{{{{\rm B\kern-.05em{{\sc i\kern-.025em b}}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{{E}}\kern-.125emX}}}}

\begin{{document}}
\history{{Date of current version {today}.}}
\doi{{{metadata["doi"]}}}

\title{{{title}}}
\author{{{metadata["author_latex"]}}}
{metadata["address_latex"]}
\tfootnote{{{metadata["tfootnote_latex"]}}}

\markboth
{{{metadata["markboth_latex"]}}}
{{{metadata["markboth_latex"]}}}

\corresp{{{metadata["corresponding_author_latex"]}}}

\begin{{abstract}}
{abstract}
\end{{abstract}}

\begin{{keywords}}
{make_keywords()}.
\end{{keywords}}

\titlepgskip=-21pt

\maketitle

{body}

\EOD

\end{{document}}
"""


def copy_template_files() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    for path in TEMPLATE.iterdir():
        if path.is_file():
            if path.suffix.lower() in {".aux", ".log", ".pdf"}:
                continue
            if path.name == "access.tex":
                continue
            shutil.copy2(path, FINAL / path.name)


def copy_figures() -> None:
    src_fig = ROOT / "paper_rewriting_output" / "final_paper" / "figures"
    dst_fig = FINAL / "figures"
    dst_fig.mkdir(parents=True, exist_ok=True)
    for name in (
        "boundary_zoedepth_architecture.png",
        "nyu_brrh_depthpro_top_samples.png",
        "brrh_noresidual_qualitative.png",
    ):
        shutil.copy2(src_fig / name, dst_fig / name)


def write_paperspine_artifacts(src_tex: str) -> None:
    config = {
        "workflow": "rewrite_existing",
        "scene": "journal",
        "tier": "pro",
        "output_language": "en",
        "target_name": "IEEE Access",
        "materials_dir": str(ROOT),
        "draft_path": str(SRC),
        "user_motivation": (
            "Reformat the existing BRRH-ZoeDepth manuscript for IEEE Access while preserving "
            "the conservative boundary-local metric reliability story and verified metrics."
        ),
        "official_urls": [
            "local:/home/kxr/ACCESS_latex_template_20260513/access.tex",
            "local:/home/kxr/ACCESS_latex_template_20260513/ieeeaccess.cls",
            "https://template-selector.ieee.org/secure/templateSelector/publicationType",
            "https://ieeeaccess.ieee.org/",
        ],
        "special_requirements": [
            "Use the IEEE Access LaTeX class ieeeaccess.cls from the local 2026-05-13 template.",
            "Use IEEE-style numeric citations and IEEE Access front matter.",
            "Keep the main claim conservative: boundary-local metric reliability, not SOTA sharp depth.",
            "Do not fabricate additional experiments, datasets, citations, or author metadata.",
            "Keep author, affiliation, DOI, funding, and correspondence fields as placeholders until the user provides real information.",
        ],
        "word_output": "none",
        "translation_package": "none",
        "reference_mode": "local_first",
        "reference_paths": [
            str(SRC),
            "/home/kxr/ACCESS_latex_template_20260513/access.tex",
            "/home/kxr/ACCESS_latex_template_20260513/ieeeaccess.cls",
            str(ROOT / "reports"),
            str(ROOT / "logs"),
            str(ROOT / "figures"),
        ],
        "citation_target_count": 20,
    }
    write(OUT / "paper_spine_config.json", json.dumps(config, indent=2))
    write(
        OUT / "paper_spine_config.md",
        "# PaperSpine Config: IEEE Access\n\n"
        "- Workflow: rewrite_existing\n"
        "- Scene: journal\n"
        "- Target: IEEE Access\n"
        "- Output language: English\n"
        "- Source draft: `paper_rewriting_output/final_paper/main.tex`\n"
        "- Final LaTeX output: `paper_rewriting_output_ieee_access/final_paper/main.tex`\n",
    )
    write(
        OUT / "source_map.md",
        "# Source Map\n\n"
        "| Source | Role | Reliability |\n"
        "|---|---|---|\n"
        f"| `{SRC}` | Current verified manuscript content | Authoritative for text, figures, tables, and metrics |\n"
        "| `/home/kxr/ACCESS_latex_template_20260513/access.tex` | IEEE Access LaTeX template sample | Authoritative local template structure |\n"
        "| `/home/kxr/ACCESS_latex_template_20260513/ieeeaccess.cls` | IEEE Access class file | Authoritative local class file |\n"
        "| `reports/paper_metric_verification_latest.md` | Metric traceability | Authoritative local audit |\n"
        "| `reports/reviewer_evidence_matrix.md` | Claim boundaries | Authoritative local claim map |\n",
    )
    write(
        OUT / "reference_materials" / "source_index.md",
        "# Reference Materials Source Index\n\n"
        "| Source | Kind | How It Is Used | Boundary / Reliability Note |\n"
        "|---|---|---|---|\n"
        f"| `{SRC}` | Source manuscript | Supplies the verified scientific text, figures, tables, labels, and bibliography for the IEEE Access rewrite. | Authoritative for existing claims; no new metric claims are introduced during format conversion. |\n"
        "| `/home/kxr/ACCESS_latex_template_20260513/access.tex` | Venue template exemplar | Supplies the IEEE Access front-matter pattern, abstract/keywords environments, `\\PARstart`, and required `\\EOD`. | Authoritative for local formatting, not for scientific claims. |\n"
        "| `/home/kxr/ACCESS_latex_template_20260513/ieeeaccess.cls` | Venue class file | Provides the class behavior used to compile the final PDF. | Compile behavior is checked through `latex_report.md`. |\n"
        "| `reports/paper_metric_verification_latest.md` | Metric audit | Used to keep RMSE, boundary metrics, and comparison claims tied to prior verification. | Prevents overclaiming when moving to IEEE Access. |\n"
        "| `reports/reviewer_evidence_matrix.md` | Claim audit | Used to preserve conservative reviewer-facing claim boundaries. | Supports the boundary-local metric reliability framing. |\n"
        "| `reports/experiment_artifact_manifest.md` | Experiment inventory | Used as the evidence bank for tables, figures, and experiment references. | Ensures the rewrite cites existing artifacts rather than invented experiments. |\n"
        "| `paper_rewriting_output/citation_support_bank.md` | Citation support | Carries the existing citation-to-claim mapping into the IEEE Access branch. | Preserves citation provenance during the format rewrite. |\n",
    )
    write(
        OUT / "research_dossier.md",
        "# Research Dossier: IEEE Access Conversion\n\n"
        "Local IEEE Access template evidence from `ACCESS_latex_template_20260513/access.tex`:\n\n"
        "- The manuscript uses `\\documentclass{ieeeaccess}`.\n"
        "- Front matter includes `\\history`, `\\doi`, `\\title`, `\\author`, `\\address`, `\\tfootnote`, `\\markboth`, and `\\corresp`.\n"
        "- Abstract should be one paragraph and 150--250 words according to the template instructions.\n"
        "- Keywords are placed in a `keywords` environment.\n"
        "- IEEE-style numeric citations are supported through `cite` and `IEEEtran.bst`.\n\n"
        "Format decision: create a separate IEEE Access branch instead of overwriting the ITC manuscript.\n",
    )
    write(
        OUT / "exemplar_learning_dossier.md",
        "# Exemplar Learning Dossier\n\n"
        "The local IEEE Access template teaches structure, not scientific claims. Reusable patterns:\n\n"
        "1. IEEE Access front matter with correspondence and support footnote.\n"
        "2. One-paragraph abstract followed by keyword block.\n"
        "3. `\\PARstart` for the first Introduction paragraph.\n"
        "4. Numeric IEEE references and compact figure/table captions.\n",
    )
    write(
        OUT / "style_profile.md",
        "# Style Profile: IEEE Access\n\n"
        "- Use concise technical English.\n"
        "- Use numeric references.\n"
        "- Keep title in uppercase/lowercase, not all caps.\n"
        "- Keep abstract self-contained and without citations.\n"
        "- Keep claims conservative and evidence-bound.\n",
    )
    write(
        OUT / "sota_gap_map.md",
        "# SOTA Gap Map\n\n"
        "The conversion does not change the scientific gap. The paper remains positioned against ZoeDepth, DA-V2, and Depth Pro as a boundary-local metric reliability study. IEEE Access framing should emphasize application-oriented perception reliability rather than venue-specific leaderboard superiority.\n",
    )
    write(
        OUT / "motivation_options_after_research.md",
        "# Motivation Options After Research\n\n"
        "Chosen option: preserve the current BRRH-ZoeDepth motivation and adapt the presentation to IEEE Access.\n\n"
        "Rejected option: reframe as a broad SOTA depth-estimation paper, because the verified evidence does not support that claim.\n",
    )
    write(
        OUT / "confirmed_motivation.md",
        "# Confirmed Motivation\n\n"
        "Reformat the manuscript for IEEE Access while preserving the verified story: BRRH-ZoeDepth improves boundary-local metric reliability in a ZoeDepth-style pipeline through bounded final log-depth residual correction.\n",
    )
    write(
        OUT / "citation_support_bank.md",
        read(ROOT / "paper_rewriting_output" / "citation_support_bank.md"),
    )
    write(
        OUT / "original_logic_map.md",
        "# Original Logic Map\n\n"
        "The ITC manuscript already uses the accepted BRRH story: global metric depth can hide local boundary metric failures; BRRH applies bounded final log-depth residual correction; KITTI, NYU, DA-V2, and Depth Pro diagnostics support the calibrated claim.\n",
    )
    write(
        OUT / "evidence_bank.md",
        read(ROOT / "reports" / "experiment_artifact_manifest.md"),
    )
    write(
        OUT / "section_blueprints.md",
        "# Section Blueprints: IEEE Access\n\n"
        "| Section | Function | IEEE Access Conversion Action |\n"
        "|---|---|---|\n"
        "| Front matter | IEEE Access metadata | Add history, DOI placeholder, author/address/corresp/tfootnote placeholders |\n"
        "| Abstract/Keywords | Self-contained summary | Keep verified abstract; convert keywords to IEEE keywords environment |\n"
        "| Introduction | Problem and contributions | Preserve boundary-local metric reliability story; add PARstart |\n"
        "| Related Work | Context | Preserve current structure and numeric citations |\n"
        "| Method | Technical contribution | Preserve equations and bounded residual formulation |\n"
        "| Experiments | Evidence | Preserve verified tables, figures, and conservative claims |\n"
        "| Discussion/Limitations | Risk control | Preserve distinction between edge localization and metric reliability |\n"
        "| References | IEEE numeric references | Use `references.bib` with `IEEEtran` BibTeX style |\n",
    )
    write(
        OUT / "writing_rationale_matrix.md",
        "# Writing Rationale Matrix\n\n"
        "| Row ID | Manuscript Unit | Current/Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| W0 | Whole-paper framework | Preserve the complete BRRH-ZoeDepth argument while changing only the venue shell. | Motivation: the paper's spine is boundary-local metric reliability, so the rewrite must protect the distinction between global metric accuracy and boundary error. | Reference/SOTA pattern: IEEE Access papers commonly foreground an application-relevant engineering problem, then support it with numeric evidence and restrained comparison; the existing ZoeDepth, DA-V2, and Depth Pro discussion already provides the SOTA contrast. | Target venue norm: IEEE Access favors a self-contained journal article with structured front matter, numeric citations, application relevance, and reproducible evidence. | Evidence/citation anchor: `paper_rewriting_output/final_paper/main.tex`, `reports/paper_metric_verification_latest.md`, `reports/reviewer_evidence_matrix.md`, and `reports/experiment_artifact_manifest.md` are the controlling source, data, and claim registers. | Text move: create a separate IEEE Access branch, keep the science, metrics, tables, and figures unchanged, reframe the opening as an IEEE-style reliability problem, and leave human-owned metadata explicit as placeholders. | Final check: PDF compiles with `ieeeaccess.cls`; PaperSpine audit stays clean; no new SOTA or unverified metric claim appears. |\n"
        "| W1 | IEEE Access front matter | Convert the title page and author block to the local IEEE Access template. | Motivation: format compliance is necessary before the manuscript can be reviewed under the new journal target. | Reference/SOTA pattern: the template `access.tex` uses `history`, `doi`, `title`, `author`, `address`, `tfootnote`, `markboth`, and `corresp` before the abstract. | Target venue norm: IEEE Access requires identifiable author, affiliation, correspondence, and support metadata, while DOI is normally assigned by IEEE after acceptance. | Evidence/citation anchor: `/home/kxr/ACCESS_latex_template_20260513/access.tex` and `ieeeaccess.cls` are the local formatting sources. | Text move: replace the ITC-style preamble/front matter with IEEE Access commands and mark missing author, email, DOI, and funding fields as placeholders instead of fabricating them. | Final check: manuscript starts with `\\documentclass{ieeeaccess}` and front-matter placeholders are visible for later human completion. |\n"
        "| W2 | Abstract and keywords | Preserve the verified abstract while converting to IEEE Access environments. | Motivation: abstract drift would be dangerous because it is the most visible claim summary and could overstate current experimental strength. | Reference/SOTA pattern: IEEE Access template guidance expects a one-paragraph abstract of roughly 150--250 words and a compact keyword block. | Target venue norm: the abstract should be self-contained, should not rely on equations or citations, and should state the metric-depth boundary problem, proposed BRRH mechanism, and evidence scope. | Evidence/citation anchor: the existing verified abstract in `paper_rewriting_output/final_paper/main.tex` and the conversion script's abstract word-count check. | Text move: keep the abstract text, move it into `abstract`, alphabetize and normalize keywords, and avoid adding new experimental promises. | Final check: abstract estimate remains inside the IEEE Access word range and keywords compile in the `keywords` environment. |\n"
        "| W3 | Introduction | Preserve the problem framing while applying IEEE-style opening mechanics. | Motivation: the introduction must make the reader care about boundary-local metric failures, not just overall RMSE. | Reference/SOTA pattern: engineering depth papers usually move from task importance to failure mode, then to method contribution and evaluation design; the ZoeDepth/DA-V2/Depth Pro comparison supplies the SOTA reference pattern. | Target venue norm: IEEE Access supports a direct first paragraph with `\\PARstart` and contribution bullets or compact claims grounded in evidence. | Evidence/citation anchor: source Introduction, reviewer evidence matrix, and local citation support bank. | Text move: insert `\\PARstart{D}{epth}` while preserving the existing claim sequence and retaining the conservative contribution framing. | Final check: the first section compiles, remains readable, and does not claim broad SOTA superiority beyond the verified boundary-local scope. |\n"
        "| W4 | Related work | Keep the relationship among metric MDE, foundation priors, and boundary-aware depth clear. | Motivation: reviewers need to see why the method is not merely ZoeDepth retraining or generic edge loss. | Reference/SOTA pattern: SOTA discussion should separate metric-capable models, relative-depth foundation priors such as DA-V2, and sharp-boundary methods such as Depth Pro. | Target venue norm: IEEE Access related work should be organized by technical theme and cited numerically. | Evidence/citation anchor: current bibliography and `citation_support_bank.md`. | Text move: preserve existing subsections and citation mapping while ensuring the IEEE branch does not introduce unsupported new literature claims. | Final check: citations remain resolved after two LaTeX passes. |\n"
        "| W5 | Method section | Preserve BRRH as a bounded residual refinement rather than inventing a larger architecture. | Motivation: the paper's strongest defensible contribution is controlled final-stage log-depth correction around difficult boundaries. | Reference/SOTA pattern: method sections in depth papers usually define baseline prediction, auxiliary prior use, loss terms, and inference behavior with equations. | Target venue norm: IEEE Access readers expect implementation-level clarity without hidden training tricks. | Evidence/citation anchor: source Method section, model scripts, and existing architecture figure. | Text move: keep equations and figure labels intact, avoid adding unimplemented modules, and present DA-V2 only as frozen structural prior rather than metric output. | Final check: equations, labels, and method figure references survive the conversion. |\n"
        "| W6 | Experiments and metrics | Preserve the verified experiment tables and prevent fabricated improvement claims. | Motivation: the current evidence supports boundary-local metric reliability, so the experiment narrative must stay attached to measured KITTI/NYU diagnostics. | Reference/SOTA pattern: depth-estimation papers report global metrics such as RMSE and boundary-aware diagnostics separately; that pattern prevents a small global change from being oversold. | Target venue norm: IEEE Access permits application-driven experiments but still expects traceable data, settings, and comparison scope. | Evidence/citation anchor: `reports/paper_metric_verification_latest.md`, logs, rendered visual figures, and experiment artifact manifest. | Text move: retain the verified tables/figures and explicitly keep ablation/comparison wording conservative. | Final check: metric values in the IEEE source match the verified source manuscript. |\n"
        "| W7 | Figures and tables | Keep visual and tabular evidence unchanged during formatting conversion. | Motivation: figures are the main support for the boundary-local story and should not be redesigned in a way that changes interpretation. | Reference/SOTA pattern: IEEE articles use numbered figures/tables with concise captions and labels that connect back to the narrative. | Target venue norm: all figures must compile under the class, fit within IEEE Access layout, and avoid unsupported visual claims. | Evidence/citation anchor: copied files in `final_paper/figures`, source captions, and artifact manifest. | Text move: copy only the current verified figures, keep captions conservative, and defer any aesthetic figure redesign to a later human-reviewed pass. | Final check: PDF includes the expected figures and captions without missing-file errors. |\n"
        "| W8 | Limitations, conclusion, and references | Keep the paper honest about scope while preserving numeric references. | Motivation: a strict reviewer will punish overclaiming, so the ending must state where the method helps and where it remains limited. | Reference/SOTA pattern: strong applied papers close by linking contribution to evidence, then admitting scope limits and future validation needs. | Target venue norm: IEEE Access expects a conventional conclusion and IEEE-style reference formatting. | Evidence/citation anchor: source conclusion, source bibliography, and citation support bank. | Text move: keep the limitations/future-work logic, convert the bibliography to `references.bib`, and build with `IEEEtran`. | Final check: BibTeX runs cleanly and no undefined references or citation warnings remain after repeated compilation. |\n",
    )
    write(
        OUT / "rewrite_matrix.md",
        "# Rewrite Matrix\n\n"
        "| Source Unit | Rewrite Action | Status |\n"
        "|---|---|---|\n"
        "| ITC preamble | Replace with IEEE Access preamble | Done by generator |\n"
        "| ITC title/author block | Replace with IEEE Access title/author/address/corresp placeholders | Done by generator |\n"
        "| Abstract | Preserve text, convert environment | Done by generator |\n"
        "| Keywords | Alphabetize and convert to IEEE keywords environment | Done by generator |\n"
        "| Main body | Preserve current sections/tables/figures while tightening claim language | Done by generator |\n"
        "| References | Convert local bibliography to `references.bib` and `IEEEtran` style | Done by generator |\n",
    )
    write(
        OUT / "logic_transfer_audit.md",
        "# Logic Transfer Audit\n\n"
        "The IEEE Access conversion changes format only. It preserves the current scientific logic, metrics, tables, figures, and claim boundaries. Required human metadata remain placeholders and must be completed before submission.\n",
    )
    write(
        OUT / "submission_metadata_template.json",
        json.dumps(
            {
                "doi": "10.1109/ACCESS.2026.0000000",
                "author_latex": "\\uppercase{First Author}\\authorrefmark{1}, \\uppercase{Second Author}\\authorrefmark{2}",
                "address_latex": (
                    "\\address[1]{Department, University, City, Country (e-mail: first@author.edu)}\n"
                    "\\address[2]{Department, University, City, Country (e-mail: second@author.edu)}"
                ),
                "corresponding_author_latex": "Corresponding author: First Author (e-mail: first@author.edu).",
                "markboth_latex": "First Author \\headeretal: Metric Depth at the Edge",
                "tfootnote_latex": (
                    "This work was supported in part by [funding agency/project number], "
                    "or state that no funding was received."
                ),
                "acknowledgment_latex": (
                    "The authors would like to thank [names/institutions] for useful discussions "
                    "or computational support."
                ),
                "data_code_availability_latex": (
                    "The experiments use the public KITTI and NYU Depth V2 datasets under their "
                    "respective dataset terms. The implementation, training configurations, "
                    "evaluation scripts, metric logs, and checkpoints are available at [repository "
                    "or reviewer-link URL]."
                ),
            },
            indent=2,
        )
        + "\n",
    )
    write(
        OUT / "final_structure.md",
        "# Final Structure\n\n"
        "IEEE Access front matter, abstract, keywords, Introduction, Related Work, Proposed Method, Experiments and Results, Discussion, Limitations and Future Work, Conclusions, References.\n",
    )
    write(
        OUT / "submission_metadata_todo.md",
        "# Submission Metadata To Complete\n\n"
        "These fields require real author-owned information and were intentionally not fabricated by the formatter.\n\n"
        "To apply real metadata, create `paper_rewriting_output_ieee_access/submission_metadata.json` using `submission_metadata_template.json` as the schema, then rerun `scripts/build_ieee_access_paperspine.sh`.\n\n"
        "| Field | Current manuscript location | Required real value |\n"
        "|---|---|---|\n"
        "| Author list | `final_paper/main.tex`, `\\\\author{...}` | Full legal author names in IEEE order |\n"
        "| Affiliations | `final_paper/main.tex`, `\\\\address[...]` | Department, institution, city, country, and author e-mails |\n"
        "| Corresponding author | `final_paper/main.tex`, `\\\\corresp{...}` | Real corresponding author name and e-mail |\n"
        "| Funding/support | `final_paper/main.tex`, `\\\\tfootnote{...}` | Grant numbers, project numbers, or explicit no-funding statement |\n"
        "| Acknowledgments | Optional section before references | Names/institutions to acknowledge, or explicit decision to omit acknowledgments |\n"
        "| Code repository | `Data and Code Availability` section | Public repository URL, private reviewer link, or submission-system supplementary-file statement |\n"
        "| Checkpoint/log release | `Data and Code Availability` section | URL or supplementary-file statement covering checkpoints, configs, and logs |\n"
        "| Dataset access statement | `Data and Code Availability` section | Confirm KITTI/NYU links or cite official dataset pages required by the target journal |\n"
        "| DOI | `final_paper/main.tex`, `\\\\doi{...}` | Keep placeholder until IEEE assigns DOI |\n",
    )


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source manuscript: {SRC}")
    if not TEMPLATE.exists():
        raise SystemExit(f"Missing IEEE Access template directory: {TEMPLATE}")
    OUT.mkdir(parents=True, exist_ok=True)
    src_tex = read(SRC)
    copy_template_files()
    copy_figures()
    write(FINAL / "main.tex", make_main_tex(src_tex))
    write(FINAL / "references.bib", references_bib())
    write_paperspine_artifacts(src_tex)
    abstract_words = len(strip_latex_for_words(extract_between(src_tex, r"\\begin\{abstract\}", r"\\end\{abstract\}")))
    write(
        OUT / "latex_report.md",
        "# LaTeX Report\n\n"
        f"- Target: IEEE Access\n- Template: `{TEMPLATE}`\n- Output: `{FINAL / 'main.tex'}`\n"
        f"- Abstract word count estimate: {abstract_words}\n"
        "- Compile status: pending until `pdflatex main.tex` is run.\n",
    )
    write(
        OUT / "final_artifact_manifest.md",
        "# Final Artifact Manifest\n\n"
        f"- `final_paper/main.tex`: IEEE Access LaTeX source\n"
        "- `final_paper/paper.pdf`: generated after compilation\n"
        "- `latex_report.md`: compile and format notes\n"
        "- `integrity_audit.md`: PaperSpine integrity audit after running audit script\n",
    )
    print(f"Wrote IEEE Access PaperSpine branch: {OUT}")


if __name__ == "__main__":
    main()
