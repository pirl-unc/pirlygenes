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
    """One-line report summary for mtDNA/rRNA rescue normalization."""

    if not record or not record.get("applied"):
        return ""
    removed = float(record.get("removed_fraction") or 0.0)
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
    return (
        "**Expression QC rescue:** raw TPM was dominated by technical RNA "
        f"features{component_clause}; downstream cancer, target, and "
        f"pathway calculations use rescued TPM after zeroing those features "
        f"and renormalizing the remaining genes{top_clause}."
    )
