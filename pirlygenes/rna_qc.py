"""Run-level RNA quantification QC helpers.

The expression-scale checks in :mod:`pirlygenes.load_expression` validate the
gene table itself. This module looks one layer upstream when a quantifier
output directory is available, especially Salmon's ``aux_info`` JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_salmon_dirs(input_path: str | Path | None) -> list[Path]:
    if input_path is None:
        return []
    try:
        start = Path(input_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return []
    parent = start if start.is_dir() else start.parent
    out: list[Path] = []
    for candidate in (parent, parent.parent, parent / "salmon-output"):
        if candidate not in out:
            out.append(candidate)
    return out


def discover_salmon_dir(input_path: str | Path | None) -> Path | None:
    """Return the nearest Salmon output directory, if recognizable."""

    for candidate in _candidate_salmon_dirs(input_path):
        if (candidate / "aux_info" / "meta_info.json").exists():
            return candidate
        if (candidate / "quant.sf").exists() and (candidate / "aux_info").exists():
            return candidate
    return None


def _read_tpm_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    sep = "\t" if path.suffix.lower() in {".sf", ".tsv"} else ","
    try:
        df = pd.read_csv(path, sep=sep, usecols=lambda c: str(c).lower() == "tpm")
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if df.empty:
        return None
    tpm_col = next((c for c in df.columns if str(c).lower() == "tpm"), None)
    if tpm_col is None:
        return None
    tpm = pd.to_numeric(df[tpm_col], errors="coerce").fillna(0.0)
    return {
        "path": str(path),
        "total": int(len(tpm)),
        "detected_gt0": int((tpm > 0.0).sum()),
        "detected_ge1": int((tpm >= 1.0).sum()),
        "detected_ge5": int((tpm >= 5.0).sum()),
        "sum_tpm": float(tpm.sum()),
        "max_tpm": float(tpm.max()) if len(tpm) else 0.0,
    }


def _stats_from_loaded_frame(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or "TPM" not in set(df.columns):
        return None
    tpm = pd.to_numeric(df["TPM"], errors="coerce").fillna(0.0)
    return {
        "path": "loaded expression table",
        "total": int(len(tpm)),
        "detected_gt0": int((tpm > 0.0).sum()),
        "detected_ge1": int((tpm >= 1.0).sum()),
        "detected_ge5": int((tpm >= 5.0).sum()),
        "sum_tpm": float(tpm.sum()),
        "max_tpm": float(tpm.max()) if len(tpm) else 0.0,
    }


def _gene_stats(
    salmon_dir: Path | None,
    gene_df: pd.DataFrame | None,
    input_path: str | Path | None,
) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if salmon_dir is not None:
        candidates.extend(
            [
                salmon_dir / "quant.genes.sf",
                salmon_dir / "quant.gene_tpm.csv",
                salmon_dir / "gene_tpm.csv",
            ]
        )
    if input_path is not None:
        try:
            p = Path(input_path).expanduser().resolve()
            if p.name != "quant.sf":
                candidates.append(p)
        except (OSError, RuntimeError):
            pass
    for candidate in candidates:
        stats = _read_tpm_stats(candidate)
        if stats is not None:
            return stats
    return _stats_from_loaded_frame(gene_df)


def _transcript_stats(
    salmon_dir: Path | None,
    transcript_path: str | Path | None,
) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if transcript_path is not None:
        try:
            candidates.append(Path(transcript_path).expanduser().resolve())
        except (OSError, RuntimeError):
            pass
    if salmon_dir is not None:
        candidates.append(salmon_dir / "quant.sf")
    for candidate in candidates:
        stats = _read_tpm_stats(candidate)
        if stats is not None:
            return stats
    return None


def _mapping_comment(mapping_rate: float | None) -> tuple[str, list[str]]:
    if mapping_rate is None:
        return "", []
    if mapping_rate < 50.0:
        return (
            "Low; typical whole-transcriptome RNA-seq often maps much higher. "
            "Check input quality, index completeness, off-target/contaminant reads, "
            "and degradation.",
            [
                f"Salmon mapping rate is low ({mapping_rate:.1f}%). Interpret RNA-derived calls cautiously."
            ],
        )
    if mapping_rate < 70.0:
        return (
            "Below the usual range for many bulk RNA-seq runs; review input quality "
            "and index compatibility.",
            [
                f"Salmon mapping rate is below typical bulk RNA-seq expectations ({mapping_rate:.1f}%)."
            ],
        )
    return "Within a broadly typical range for compatible RNA-seq inputs.", []


def _detection_comment(stats: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not stats:
        return "", []
    ge1 = int(stats.get("detected_ge1") or 0)
    total = int(stats.get("total") or 0)
    if ge1 < 10_000:
        return (
            "Low gene-detection breadth; this can reflect degradation, low input, "
            "or a restricted assay.",
            [f"Only {ge1:,} of {total:,} genes are detected at >=1 TPM."],
        )
    if ge1 < 13_000:
        return (
            "Low-normal gene-detection breadth for whole-transcriptome bulk RNA-seq; "
            "consistent with reduced input quality or low cellularity.",
            [],
        )
    return "Gene-detection breadth is compatible with whole-transcriptome RNA-seq.", []


def _tpm_sum_comment(stats: dict[str, Any] | None, label: str) -> tuple[str, list[str]]:
    if not stats:
        return "", []
    total = float(stats.get("sum_tpm") or 0.0)
    if total <= 0:
        return "No TPM mass detected.", [f"{label} TPM sum is zero."]
    missing_frac = abs(total - 1_000_000.0) / 1_000_000.0
    if missing_frac > 0.01:
        return (
            f"TPM sum is {total:,.0f}; more than 1% away from 1,000,000. "
            "Check whether rows were filtered or values were transformed.",
            [f"{label} TPM sum is not close to 1,000,000 ({total:,.0f})."],
        )
    return f"TPM mass is near 1,000,000 ({total:,.0f}); little expression mass is missing.", []


def collect_rna_quant_qc(
    input_path: str | Path | None,
    *,
    gene_df: pd.DataFrame | None = None,
    transcript_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect Salmon run QC and expression-detection breadth.

    The return value is intentionally plain JSON-serializable data so it can be
    stored in analysis manifests and reused by reports.
    """

    salmon_dir = discover_salmon_dir(input_path)
    meta = _read_json(salmon_dir / "aux_info" / "meta_info.json") if salmon_dir else {}
    lib = _read_json(salmon_dir / "lib_format_counts.json") if salmon_dir else {}
    gene_stats = _gene_stats(salmon_dir, gene_df, input_path)
    transcript_stats = _transcript_stats(salmon_dir, transcript_path)

    warnings: list[str] = []
    mapping_rate = meta.get("percent_mapped")
    try:
        mapping_rate = float(mapping_rate) if mapping_rate is not None else None
    except (TypeError, ValueError):
        mapping_rate = None
    mapping_comment, mapping_warnings = _mapping_comment(mapping_rate)
    warnings.extend(mapping_warnings)

    detection_comment, detection_warnings = _detection_comment(gene_stats)
    warnings.extend(detection_warnings)
    gene_sum_comment, gene_sum_warnings = _tpm_sum_comment(gene_stats, "Gene")
    warnings.extend(gene_sum_warnings)
    tx_sum_comment, tx_sum_warnings = _tpm_sum_comment(transcript_stats, "Transcript")
    warnings.extend(tx_sum_warnings)

    rows: list[dict[str, str]] = []

    def add(metric: str, value: str, comment: str = "") -> None:
        if value:
            rows.append({"metric": metric, "value": value, "comment": comment})

    add("Fragments processed", _format_count(meta.get("num_processed")))
    add("Mapped fragments", _format_count(meta.get("num_mapped")))
    if mapping_rate is not None:
        add("Mapping rate", f"{mapping_rate:.1f}%", mapping_comment)
    add(
        "Decoy fragments",
        _format_count(meta.get("num_decoy_fragments")),
        "Decoys captured genome-derived or off-target fragments."
        if meta.get("num_decoy_fragments")
        else "",
    )
    add(
        "Below-threshold alignments",
        _format_count(meta.get("num_alignments_below_threshold_for_mapped_fragments_vm")),
        "Many weak hits were filtered; this supports degraded/off-target or hard-to-map input."
        if (meta.get("num_alignments_below_threshold_for_mapped_fragments_vm") or 0)
        else "",
    )
    add(
        "Filtered fragments",
        _format_count(meta.get("num_fragments_filtered_vm")),
        "Fragments removed by validation/mismatch filters."
        if meta.get("num_fragments_filtered_vm")
        else "",
    )
    lib_type = _first(meta.get("library_types")) or lib.get("expected_format")
    add("Library type", str(lib_type or ""), _library_type_comment(str(lib_type or "")))
    if meta.get("frag_length_mean") is not None:
        mean = _safe_float(meta.get("frag_length_mean"))
        sd = _safe_float(meta.get("frag_length_sd"))
        if mean is not None and sd is not None:
            add("Fragment length", f"{mean:.0f} +/- {sd:.0f} bp", "Insert size looks plausible.")
    bias = _safe_float(lib.get("strand_mapping_bias"))
    if bias is not None:
        add(
            "Strand mapping bias",
            f"{bias:.3g}",
            "No obvious strand issue." if abs(bias) < 0.01 else "Review strandedness settings.",
        )
    if gene_stats:
        add(
            "Gene detection",
            (
                f"{gene_stats['detected_gt0']:,} >0; "
                f"{gene_stats['detected_ge1']:,} >=1; "
                f"{gene_stats['detected_ge5']:,} >=5 TPM "
                f"(of {gene_stats['total']:,})"
            ),
            detection_comment,
        )
        add("Gene TPM sum", f"{gene_stats['sum_tpm']:,.0f}", gene_sum_comment)
    if transcript_stats:
        add(
            "Transcript detection",
            (
                f"{transcript_stats['detected_gt0']:,} >0; "
                f"{transcript_stats['detected_ge1']:,} >=1 TPM "
                f"(of {transcript_stats['total']:,})"
            ),
        )
        add("Transcript TPM sum", f"{transcript_stats['sum_tpm']:,.0f}", tx_sum_comment)

    summary_bits = []
    if mapping_rate is not None:
        summary_bits.append(f"Salmon mapping {mapping_rate:.1f}%")
    if gene_stats:
        summary_bits.append(
            f"{gene_stats['detected_ge1']:,}/{gene_stats['total']:,} genes >=1 TPM"
        )
    if gene_stats and abs(float(gene_stats.get("sum_tpm") or 0.0) - 1_000_000.0) <= 10_000.0:
        summary_bits.append("gene TPM sum near 1.0M")
    summary = "; ".join(summary_bits)

    return {
        "available": bool(meta or lib or gene_stats or transcript_stats),
        "source": "salmon" if (meta or lib or salmon_dir) else "expression_table",
        "salmon_dir": str(salmon_dir) if salmon_dir else None,
        "meta_info": meta,
        "lib_format_counts": lib,
        "gene_detection": gene_stats,
        "transcript_detection": transcript_stats,
        "rows": rows,
        "warnings": warnings,
        "summary": summary,
    }


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_count(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def _library_type_comment(value: str) -> str:
    lookup = {
        "ISR": "Reverse-stranded paired-end library.",
        "ISF": "Forward-stranded paired-end library.",
        "IU": "Unstranded paired-end library.",
        "SR": "Reverse-stranded single-end library.",
        "SF": "Forward-stranded single-end library.",
        "U": "Unstranded single-end library.",
    }
    return lookup.get(value, "")


def rna_quant_qc_summary_line(qc: dict[str, Any] | None) -> str:
    if not qc or not qc.get("available"):
        return ""
    summary = str(qc.get("summary") or "").strip()
    if not summary:
        return ""
    warning = ""
    warnings = qc.get("warnings") or []
    if warnings:
        warning = f"; caution: {str(warnings[0]).rstrip('.')}"
    return f"**RNA quant QC:** {summary}{warning}."


def rna_quant_qc_markdown(qc: dict[str, Any] | None, *, heading: str) -> str:
    if not qc or not qc.get("available"):
        return ""
    rows = qc.get("rows") or []
    if not rows:
        return ""
    lines = [heading, ""]
    summary = str(qc.get("summary") or "").strip()
    if summary:
        lines.append(f"- **Summary**: {summary}.")
    warnings = [str(w) for w in (qc.get("warnings") or []) if str(w).strip()]
    if warnings:
        lines.append(f"- **QC warning**: {warnings[0]}")
    lines.append("")
    lines.append("| Metric | Value | Comment |")
    lines.append("|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.get('metric', '')} | {row.get('value', '')} | {row.get('comment', '') or '-'} |"
        )
    return "\n".join(lines)
