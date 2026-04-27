"""Markdown comparison helpers for multiple ``analyze`` report packets."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


@dataclass(frozen=True)
class AnalyzeSummaryRecord:
    """Small, report-level view of one completed ``analyze`` output."""

    sample_id: str
    output_dir: str
    cancer_call: str
    cancer_type_basis: str
    purity: str
    disease_state: str
    sample_context: str
    top_therapies: tuple[str, ...]
    caveats: tuple[str, ...]


def _first_matching_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} file in {directory}")
    return matches[0]


def _section_lines(lines: list[str], heading: str) -> list[str]:
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return []
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if line.strip():
            out.append(line)
    return out


def _line_after_prefix(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _strip_markdown_prefix(line: str) -> str:
    text = line.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    return text


def _clean_table_cell(value: str) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ").strip()


def load_analyze_summary_record(output_dir: str | Path) -> AnalyzeSummaryRecord:
    """Load the summary fields needed for cross-sample comparison."""
    directory = Path(output_dir)
    summary_path = _first_matching_file(directory, "*-summary.md")
    lines = summary_path.read_text(errors="replace").splitlines()

    header = lines[0].replace("# Summary", "", 1).strip()
    sample_id = header[1:].strip() if header.startswith(":") else directory.name
    if not sample_id:
        sample_id = directory.name

    therapy_lines = [
        _strip_markdown_prefix(line)
        for line in _section_lines(lines, "## Top candidate therapies")
        if line.startswith("- **") or line.startswith("*No approved")
    ]
    caveat_lines = [
        _strip_markdown_prefix(line)
        for line in _section_lines(lines, "## Caveats")
        if line.startswith("- ")
    ]

    return AnalyzeSummaryRecord(
        sample_id=sample_id,
        output_dir=str(directory),
        cancer_call=_line_after_prefix(lines, "**Cancer call:**"),
        cancer_type_basis=_line_after_prefix(lines, "**Cancer-type basis:**"),
        purity=_line_after_prefix(lines, "**Purity:**"),
        disease_state=_line_after_prefix(lines, "**Disease state:**"),
        sample_context=_line_after_prefix(lines, "**Sample:**"),
        top_therapies=tuple(therapy_lines[:3]),
        caveats=tuple(caveat_lines),
    )


def build_analyze_comparison_markdown(
    output_dirs: list[str | Path],
    *,
    title: str = "Analyze Sample Comparison",
) -> str:
    """Build a compact Markdown comparison across completed analyze outputs."""
    records = [load_analyze_summary_record(path) for path in output_dirs]
    if len(records) < 2:
        raise ValueError("Need at least two analyze output directories to compare")

    lines = [
        f"# {title}",
        "",
        "This comparison is assembled from completed `pirlygenes analyze` summary files. "
        "It is a navigation aid, not a clinical recommendation.",
        "",
        "| Sample | Cancer call | Cancer-type basis | Purity | Disease state | Sample context |",
        "|---|---|---|---|---|---|",
    ]
    for rec in records:
        lines.append(
            "| "
            + " | ".join(
                _clean_table_cell(value)
                for value in (
                    rec.sample_id,
                    rec.cancer_call,
                    rec.cancer_type_basis,
                    rec.purity,
                    rec.disease_state,
                    rec.sample_context,
                )
            )
            + " |"
        )

    lines.extend(["", "## Therapy Shortlists", ""])
    for rec in records:
        lines.append(f"### {rec.sample_id}")
        if rec.top_therapies:
            for therapy in rec.top_therapies:
                lines.append(f"- {therapy}")
        else:
            lines.append("- No therapy shortlist was available in the summary.")
        lines.append("")

    lines.extend(
        [
            "## Cross-Sample Caveats",
            "",
            "- Do not compare raw TPMs across samples without checking assay/library differences.",
            "- Treat RNA-inferred cancer labels as hypotheses unless pathology confirms them.",
            "- Patient-facing LLM use requires diagnosis, stage, prior lines, current medications, MSI/MMR/TMB, mutations/fusions/CNVs, relevant imaging such as HER2/PSMA, and trial availability.",
        ]
    )

    unique_caveats = []
    seen = set()
    for rec in records:
        for caveat in rec.caveats:
            key = re.sub(r"\s+", " ", caveat).strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_caveats.append(caveat)
    if unique_caveats:
        lines.extend(["", "## Repeated Report Caveats", ""])
        for caveat in unique_caveats[:8]:
            lines.append(f"- {caveat}")

    return "\n".join(lines) + "\n"
