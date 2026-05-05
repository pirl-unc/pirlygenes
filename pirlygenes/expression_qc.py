"""Expression-table QC helpers shared by loading, context, and plotting.

These helpers deliberately use symbol-level heuristics instead of a heavy
annotation dependency. The failure mode they catch is usually obvious at the
gene-symbol layer: a handful of mitochondrial or rRNA/pseudogene-like entries
consume a large fraction of TPM and distort all downstream absolute expression
values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class GeneQcClass:
    label: str
    group: str


_GENE_NA = {"", "NAN", "NONE", "NULL", "-"}


def classify_gene_qc(symbol: str | None) -> GeneQcClass:
    """Return a coarse QC class for a gene symbol.

    Groups are intentionally broad and stable:

    - ``mt_dna``: mitochondrial genome transcripts.
    - ``rrna_like``: nuclear rRNA and rRNA-pseudogene annotations that can
      dominate TPM denominators when residual rRNA/small-RNA fragments leak
      into a gene-level quantification. Pseudogenes are summarized separately
      inside this group because their failure mode is usually mapping /
      annotation / short-fragment denominator distortion rather than intact
      rRNA carryover.
    - ``ribosomal_protein`` / ``ribosomal_protein_pseudogene``: real RP
      biology/library complexity signals, not removed by rescue normalization.
    - ``small_ncrna``: other small noncoding RNA families.
    - ``other``: everything else.
    """

    raw = str(symbol or "").strip()
    upper = raw.upper()
    if upper in _GENE_NA:
        return GeneQcClass("unlabeled feature", "other")

    if upper in {"MT-RNR1", "MT-RNR2"}:
        return GeneQcClass("mitochondrial rRNA", "mt_dna")
    if upper.startswith("MT-"):
        return GeneQcClass("mitochondrial transcript", "mt_dna")

    # Common HGNC rRNA/rRNA-pseudogene symbols seen in gene-level outputs.
    # Examples: RNA5SP389, RNA5-8SP6, RNA18SP1, RNA28SP2, RNA45S5.
    if re.fullmatch(r"RNA5SP\d+", upper):
        return GeneQcClass("5S rRNA pseudogene", "rrna_like")
    if re.fullmatch(r"RNA5-8SP\d+", upper):
        return GeneQcClass("5.8S rRNA pseudogene", "rrna_like")
    if re.fullmatch(r"RNA(18S|28S|45S|5S)(P\d+|\d+|[_-].*)?", upper):
        label = {
            "RNA18S": "18S rRNA-like",
            "RNA28S": "28S rRNA-like",
            "RNA45S": "45S pre-rRNA-like",
            "RNA5S": "5S rRNA-like",
        }
        prefix = next((p for p in label if upper.startswith(p)), "RNA5S")
        return GeneQcClass(label[prefix], "rrna_like")
    if upper.startswith(("RNR", "MTRNR")):
        return GeneQcClass("rRNA-like", "rrna_like")

    # Ribosomal protein pseudogenes are informative for complexity/rRNA-like
    # contamination, but they are not themselves rRNA and should not be removed
    # by mtDNA/rRNA rescue normalization.
    if re.fullmatch(r"RP[SL]\d+[A-Z]?(P\d+|P)$", upper):
        return GeneQcClass("ribosomal protein pseudogene", "ribosomal_protein_pseudogene")
    if re.fullmatch(r"RP[SL]\d+[A-Z]?", upper) or upper.startswith("RPLP"):
        return GeneQcClass("ribosomal protein", "ribosomal_protein")

    if upper.startswith(("SNORD", "SNORA", "RNU", "Y_RNA", "MIR")):
        return GeneQcClass("small noncoding RNA", "small_ncrna")

    return GeneQcClass("protein-coding/other", "other")


def is_rescue_feature(symbol: str | None) -> bool:
    """True when a feature should be removed by mtDNA/rRNA rescue."""

    return classify_gene_qc(symbol).group in {"mt_dna", "rrna_like"}


def normalize_technical_rna_columns(
    df,
    *,
    label_col: str = "Symbol",
    value_cols: Iterable[str] | None = None,
):
    """Zero mtDNA/rRNA-like features and renormalize every expression column.

    This is the shared comparability transform for reference matrices. It
    preserves each column's total expression mass after removing technical RNA
    features, so a sample/reference comparison is not driven by different
    rRNA/mtDNA denominator burden.

    Raw-expression QC should be computed before this transform.
    """

    import pandas as pd

    if label_col not in df.columns:
        return df.copy(), {
            "applied": False,
            "reason": f"label column {label_col!r} not present",
            "columns": {},
        }
    if value_cols is None:
        value_cols = [
            c
            for c in df.columns
            if str(c).startswith(("TPM", "nTPM_", "FPKM_", "tcga_"))
        ]
    value_cols = [str(c) for c in value_cols if str(c) in df.columns]
    if not value_cols:
        return df.copy(), {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
        }

    out = df.copy()
    labels = out[label_col].fillna("").astype(str).str.strip()
    removable = labels.map(is_rescue_feature).astype(bool)
    records = {}
    any_applied = False
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        valid = vals.notna()
        removable_valid = removable & valid
        keep_valid = (~removable) & valid
        raw_sum = float(vals.sum())
        removed = float(vals[removable_valid].sum())
        remaining = raw_sum - removed
        removed_fraction = removed / raw_sum if raw_sum > 0 else 0.0
        records[col] = {
            "input_sum": raw_sum,
            "removed_tpm": removed,
            "removed_fraction": removed_fraction,
            "removed_gene_count": int(removable.sum()),
            "renormalization_factor": (
                float(raw_sum / remaining) if raw_sum > 0 and remaining > 0 else 1.0
            ),
        }
        if raw_sum <= 0 or removed <= 0 or remaining <= 0:
            continue
        scale = raw_sum / remaining
        out.loc[removable_valid, col] = 0.0
        out.loc[keep_valid, col] = vals.loc[keep_valid] * scale
        any_applied = True

    return out, {
        "applied": any_applied,
        "reason": (
            "technical RNA features zeroed and remaining expression renormalized"
            if any_applied
            else "no removable technical RNA burden"
        ),
        "columns": records,
    }


def normalize_technical_rna_long_table(
    df,
    *,
    label_col: str = "symbol",
    group_cols: Iterable[str] = ("cancer_code", "subtype"),
    value_cols: Iterable[str] = ("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
):
    """Apply technical-RNA normalization within each long-table cohort group."""

    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "groups": {}}
    if label_col not in df.columns:
        return df.copy(), {
            "applied": False,
            "reason": f"label column {label_col!r} not present",
            "groups": {},
        }
    group_cols = [str(c) for c in group_cols if str(c) in df.columns]
    value_cols = [str(c) for c in value_cols if str(c) in df.columns]
    if not group_cols or not value_cols:
        return df.copy(), {
            "applied": False,
            "reason": "missing grouping or expression columns",
            "groups": {},
        }

    out = df.copy()
    labels = out[label_col].fillna("").astype(str).str.strip()
    removable = labels.map(is_rescue_feature).astype(bool)
    group_records = {}
    any_applied = False
    grouped = out.groupby(group_cols, dropna=False).groups
    for key, idx in grouped.items():
        idx = list(idx)
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_label = "|".join(str(part) for part in key_tuple)
        group_records[key_label] = {}
        group_removable = removable.loc[idx]
        for col in value_cols:
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
            valid = vals.notna()
            removable_valid = group_removable & valid
            keep_valid = (~group_removable) & valid
            raw_sum = float(vals.sum())
            removed = float(vals[removable_valid].sum())
            remaining = raw_sum - removed
            removed_fraction = removed / raw_sum if raw_sum > 0 else 0.0
            group_records[key_label][col] = {
                "input_sum": raw_sum,
                "removed_tpm": removed,
                "removed_fraction": removed_fraction,
                "renormalization_factor": (
                    float(raw_sum / remaining)
                    if raw_sum > 0 and remaining > 0
                    else 1.0
                ),
            }
            if raw_sum <= 0 or removed <= 0 or remaining <= 0:
                continue
            scale = raw_sum / remaining
            remove_idx = removable_valid[removable_valid].index
            keep_idx = keep_valid[keep_valid].index
            out.loc[remove_idx, col] = 0.0
            out.loc[keep_idx, col] = vals.loc[keep_idx] * scale
            any_applied = True

    return out, {
        "applied": any_applied,
        "reason": (
            "technical RNA features zeroed and remaining expression renormalized"
            if any_applied
            else "no removable technical RNA burden"
        ),
        "groups": group_records,
    }


def summarize_qc_class_shares(
    gene_tpm_items: Iterable[tuple[str, float]],
) -> dict[str, object]:
    """Summarize total TPM share by QC class/group."""

    group_tpm: dict[str, float] = {}
    class_tpm: dict[str, float] = {}
    total = 0.0
    for gene, value in gene_tpm_items:
        try:
            tpm = float(value)
        except (TypeError, ValueError):
            continue
        if tpm <= 0:
            continue
        qc = classify_gene_qc(gene)
        total += tpm
        group_tpm[qc.group] = group_tpm.get(qc.group, 0.0) + tpm
        class_tpm[qc.label] = class_tpm.get(qc.label, 0.0) + tpm

    def _fraction_map(values: Mapping[str, float]) -> dict[str, float]:
        if total <= 0:
            return {key: 0.0 for key in values}
        return {
            key: round(float(val) / total, 6)
            for key, val in sorted(values.items(), key=lambda item: (-item[1], item[0]))
        }

    group_share = _fraction_map(group_tpm)
    class_share = _fraction_map(class_tpm)
    rrna_pseudogene_fraction = float(
        sum(
            val
            for key, val in class_share.items()
            if "rRNA pseudogene" in str(key)
        )
    )
    rrna_like_fraction = float(group_share.get("rrna_like", 0.0))
    mt_dna_fraction = float(group_share.get("mt_dna", 0.0))
    mitochondrial_rrna_fraction = float(class_share.get("mitochondrial rRNA", 0.0))
    nuclear_rrna_like_fraction = max(
        0.0, rrna_like_fraction - rrna_pseudogene_fraction
    )
    return {
        "total_tpm": float(total),
        "group_tpm": dict(sorted(group_tpm.items())),
        "class_tpm": dict(sorted(class_tpm.items())),
        "group_share": group_share,
        "class_share": class_share,
        "mt_dna_fraction": mt_dna_fraction,
        "mitochondrial_rrna_fraction": mitochondrial_rrna_fraction,
        "mt_non_rrna_fraction": max(0.0, mt_dna_fraction - mitochondrial_rrna_fraction),
        "rrna_like_fraction": rrna_like_fraction,
        "nuclear_rrna_like_fraction": nuclear_rrna_like_fraction,
        "rrna_pseudogene_fraction": rrna_pseudogene_fraction,
        "rrna_plus_mt_fraction": float(mt_dna_fraction + rrna_like_fraction),
    }


def technical_rna_component_phrase(summary: Mapping[str, object] | None) -> str:
    """Human-readable breakdown of mtDNA/rRNA-like TPM burden."""

    if not summary:
        return ""
    components = [
        ("rRNA pseudogene", float(summary.get("rrna_pseudogene_fraction") or 0.0)),
        ("nuclear rRNA-like", float(summary.get("nuclear_rrna_like_fraction") or 0.0)),
        ("mitochondrial rRNA", float(summary.get("mitochondrial_rrna_fraction") or 0.0)),
        ("other mtDNA", float(summary.get("mt_non_rrna_fraction") or 0.0)),
    ]
    shown = [(label, frac) for label, frac in components if frac >= 0.005]
    if not shown:
        return ""
    return ", ".join(f"{label} {frac:.0%}" for label, frac in shown)


def dominant_class_phrase(dominant: list[dict] | None) -> str:
    """Short phrase for warnings when dominant genes share one QC class."""

    rows = dominant or []
    if not rows:
        return ""
    top = rows[0]
    gene = str(top.get("gene") or "").strip()
    label = str(top.get("qc_class") or "").strip()
    if gene and label and label != "protein-coding/other":
        return f"{gene}; {label}"
    if gene:
        return gene
    return label


def expression_qc_rescue_summary_line(record: dict | None) -> str:
    """One-line report summary for mtDNA/rRNA technical normalization."""

    if not record or not record.get("applied"):
        return ""
    removed = float(record.get("removed_fraction") or 0.0)
    high_burden = bool(record.get("high_burden"))
    component_phrase = technical_rna_component_phrase(
        record.get("qc_class_shares") or {}
    )
    component_clause = f" ({component_phrase}; {removed:.0%} removed)" if component_phrase else f" ({removed:.0%} removed)"
    top_removed = record.get("top_removed_genes") or []
    top_clause = ""
    if top_removed:
        top = top_removed[0]
        gene = str(top.get("gene") or "").strip()
        qc_class = str(top.get("qc_class") or "").strip()
        share = float(top.get("share") or 0.0)
        if gene:
            top_clause = f"; top removed feature {gene}"
            if qc_class:
                top_clause += f" ({qc_class}"
                if share:
                    top_clause += f", {share:.0%} of raw TPM"
                top_clause += ")"
    prefix = (
        "**Expression QC rescue:** raw TPM was dominated by technical RNA features"
        if high_burden
        else "**Technical-RNA normalization:** mtDNA/rRNA-like features were removed for reference comparability"
    )
    return (
        f"{prefix}{component_clause}; downstream cancer, target, and pathway "
        "calculations use TPM after zeroing those features and renormalizing "
        f"the remaining genes{top_clause}."
    )
