"""Symbol-level + ENSG-level QC classification for expression matrices.

This module answers *which gene-symbol-level features are usable* for
downstream rescaling. The companion module
:mod:`pirlygenes.expression.normalize` consumes the classification to
zero technical-RNA rows and renormalize the remaining mass.

The classifier deliberately uses symbol-level heuristics — plus
ENSG-first lookups against the curated panels in
:mod:`pirlygenes.gene_families` — instead of a heavy annotation
dependency. The failure mode it catches is usually obvious at the
gene-symbol layer: a handful of mitochondrial or rRNA / pseudogene-
like entries consume a large fraction of TPM and distort all
downstream absolute expression values.

Per-sample QC narration (TPM-share-by-class summaries, top-K mass
concentration, rescue-summary phrasing) lives in trufflepig's
``expression_qc`` module — those are analysis-layer helpers that
consume this classifier rather than belonging next to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneQcClass:
    label: str
    group: str


_GENE_NA = {"", "NAN", "NONE", "NULL", "-"}

# Nuclear-retained, ENE-stabilized lncRNAs that survive degradation
# disproportionately and creep up as a fraction of bulk TPM. Their
# 3′ ends are processed by RNase P to a triple-helical U-rich motif
# (Element for Nuclear Expression) that confers exonuclease resistance.
# In single-cell data the artifact runs the opposite direction —
# low MALAT1 is the strongest single-gene proxy for damaged-nucleus /
# ambient-droplet cells.
#
# Refs:
#   - Brown JA et al., PNAS 2012  (MALAT1/NEAT1 ENE triple helix)
#   - Wilusz JE et al., Cell 2008  (MALAT1 3' processing → mascRNA)
#   - Hopp AK et al., bioRxiv 2024  (MALAT1 ↔ nuclear-fraction QC)
#   - Montserrat-Ayuso & Esteve-Codina, BMC Genomics 2024  (MALAT1-low
#     cell fractions in HCA / Tabula Sapiens / Tabula Muris)
#
# Held back from this default panel: KCNQ1OT1, XIST, HOTAIR — also
# nuclear-retained, but their biological signal (imprinting, Xi,
# HOX-axis) is strong enough that auto-dropping would obscure real
# biology more than it removes artifact.
_POLYA_BIAS_LNCRNA_SYMBOLS = frozenset({"MALAT1", "NEAT1"})

# The qc-groups that constitute the zero-and-renormalize *technical-RNA*
# compartment of the clean-TPM space. PUBLIC contract (#445): a consumer
# conforming a sample to clean-TPM classifies each gene with
# :func:`classify_gene_qc` and zeroes rows whose ``.group`` is in this set.
TECHNICAL_RNA_GROUPS = frozenset(
    {"mt_dna", "mt_like_pseudogene", "rrna_like", "polyadenylation_bias_lncrna"}
)
#: Deprecated private alias of :data:`TECHNICAL_RNA_GROUPS`; kept for back-compat.
_TECHNICAL_RNA_GROUPS = TECHNICAL_RNA_GROUPS

#: Fractions of the 1e6 clean-TPM budget pinned to each censored compartment in
#: the clean-TPM contract (#446). The censored block is split into two
#: separately-pinned compartments (cancerdata's 16/9 refinement, validated on
#: fresh-frozen polyA TCGA: ribosomal proteins sit at ~16% of the budget, other
#: technical RNA at ~9%), with biology getting the remaining 75% — instead of one
#: lumped 25% block. PUBLIC source of truth for the values normalize.py applies;
#: consumers import these instead of re-typing the magic numbers, and the emitted
#: reference frame records the values actually used in its metadata.
RIBOSOMAL_PROTEIN_FRACTION = 0.16
OTHER_TECHNICAL_FRACTION = 0.09
#: Combined censored budget (ribosomal + other technical). Retained for back-compat
#: and for the single-compartment paths (technical-only view, the v5 cap mode).
TECHNICAL_FRACTION = RIBOSOMAL_PROTEIN_FRACTION + OTHER_TECHNICAL_FRACTION

# Display ordering for the per-category pre-normalization QC block.
# Removed groups come first (drop-by-default), then retained groups; within
# each block, members render in the listed order so reports stay stable.
_QC_GROUP_DISPLAY_ORDER = (
    # Removed:
    "mt_dna",
    "rrna_like",
    "polyadenylation_bias_lncrna",
    "mt_like_pseudogene",
    # Retained:
    "ribosomal_protein",
    "ribosomal_protein_pseudogene",
    "small_ncrna",
    "histone",
    "hemoglobin",
    "immune_receptor",
    "other",
)

_QC_GROUP_DISPLAY_LABEL = {
    "mt_dna": "mitochondrial transcript",
    "mt_like_pseudogene": "mitochondrial pseudogene / NUMT-like",
    "rrna_like": "rRNA / rRNA-pseudogene",
    "polyadenylation_bias_lncrna": "nuclear-retained ENE-stabilized lncRNA",
    "ribosomal_protein": "ribosomal protein",
    "ribosomal_protein_pseudogene": "ribosomal protein pseudogene",
    "small_ncrna": "small noncoding RNA",
    "histone": "histone transcript",
    "hemoglobin": "hemoglobin transcript",
    "immune_receptor": "immune receptor segment",
    "other": "protein-coding / other",
}


# pirlygenes.gene_families.<family-name> → (qc_label, qc_group). The
# family naming is biological ("nuclear_retained_lncrna"); the QC
# grouping is downstream-specific (drop-by-default sets). Keeping
# the mapping in one place lets the family CSVs change in pirlygenes
# without QC semantics drifting silently.
_FAMILY_TO_QC = {
    "mitochondrial": ("mitochondrial transcript", "mt_dna"),
    "numt_pseudogene": ("mitochondrial pseudogene / NUMT-like", "mt_like_pseudogene"),
    "nuclear_retained_lncrna": (
        "nuclear-retained ENE-stabilized lncRNA",
        "polyadenylation_bias_lncrna",
    ),
    "rrna_and_pseudogene": ("rRNA / rRNA-pseudogene", "rrna_like"),
    "ribosomal_protein": ("ribosomal protein", "ribosomal_protein"),
    "ribosomal_protein_pseudogene": (
        "ribosomal protein pseudogene",
        "ribosomal_protein_pseudogene",
    ),
    "small_noncoding_rna": ("small noncoding RNA", "small_ncrna"),
    "histone": ("histone transcript", "histone"),
    "hemoglobin": ("hemoglobin transcript", "hemoglobin"),
    "immune_receptor_segment": ("immune receptor segment", "immune_receptor"),
}


# Gene-family names whose members count as "technical RNA" — the drop-
# by-default set. Derived from ``_FAMILY_TO_QC`` so the family-name
# view (used by :func:`pirlygenes.expression.filter_technical_rna`) and
# the QC-group view (used by :func:`is_rescue_feature` /
# :func:`normalize_expression`) stay in lockstep automatically.
TECHNICAL_RNA_FAMILIES = frozenset(
    family
    for family, (_label, group) in _FAMILY_TO_QC.items()
    if group in TECHNICAL_RNA_GROUPS
)
#: Deprecated private alias of :data:`TECHNICAL_RNA_FAMILIES`; kept for back-compat.
_TECHNICAL_RNA_FAMILIES = TECHNICAL_RNA_FAMILIES


def _family_to_qc_class(family: str, symbol: str | None = None) -> GeneQcClass:
    """Map a pirlygenes gene-family name to its QC class.

    When a symbol is supplied, refine the family-level label with a
    symbol-specific sub-classifier so reports keep the fine-grained
    distinctions (``mitochondrial rRNA`` for MT-RNR1/2, ``mitochondrial
    tRNA`` for MT-T*, ``5S rRNA pseudogene`` for RNA5SP*, ...). The QC
    *group* always comes from the family — the sub-classification only
    affects the user-facing label.
    """
    label, group = _FAMILY_TO_QC.get(family, ("protein-coding/other", "other"))
    if symbol:
        refined = _refine_family_label(family, str(symbol).strip().upper())
        if refined is not None:
            label = refined
    return GeneQcClass(label, group)


def _refine_family_label(family: str, upper: str) -> str | None:
    """Return a more specific human label for ``(family, symbol)`` if one
    is available, else ``None``. Family-level fallback is the default."""
    if family == "mitochondrial":
        if upper in {"MT-RNR1", "MT-RNR2"}:
            return "mitochondrial rRNA"
        # mt-tRNAs are MT-TA, MT-TC, MT-TD, ... MT-TY (22 of them; all
        # match MT-T<single letter> optionally followed by a digit).
        if re.fullmatch(r"MT-T[A-Z]\d?", upper):
            return "mitochondrial tRNA"
        return "mitochondrial transcript"
    if family == "rrna_and_pseudogene":
        if re.fullmatch(r"RNA5SP\d+", upper):
            return "5S rRNA pseudogene"
        if re.fullmatch(r"RNA5-8SP\d+", upper):
            return "5.8S rRNA pseudogene"
        for stem, label in (
            ("RNA18S", "18S rRNA-like"),
            ("RNA28S", "28S rRNA-like"),
            ("RNA45S", "45S pre-rRNA-like"),
            ("RNA5S", "5S rRNA-like"),
        ):
            if upper.startswith(stem):
                return label
    if family == "ribosomal_protein_pseudogene":
        return "ribosomal protein pseudogene"
    if family == "small_noncoding_rna":
        if upper.startswith("SNORD"):
            return "small nucleolar RNA (C/D box)"
        if upper.startswith("SNORA"):
            return "small nucleolar RNA (H/ACA box)"
        if upper.startswith("RNU"):
            return "spliceosomal snRNA"
        if upper.startswith("MIR"):
            return "microRNA"
        if "Y_RNA" in upper or upper.startswith("YR"):
            return "Y RNA"
        if upper.startswith("VTRNA"):
            return "vault RNA"
        if upper.startswith("RN7SK") or upper.startswith("RN7SL"):
            return "signal recognition particle RNA"
    return None


def classify_gene_qc(
    symbol: str | None = None,
    *,
    ensembl_id: str | None = None,
) -> GeneQcClass:
    """Return a coarse QC class for a gene by symbol and/or ENSG ID.

    Lookup order:

    1. If ``ensembl_id`` is given and the ID belongs to a curated
       :mod:`pirlygenes.gene_families` panel, use the QC group mapped
       from that family. ENSG-first lookup is stable across HGNC symbol
       renames and version-suffix drift.
    2. If ``symbol`` matches a family by symbol, use that.
    3. Otherwise fall back to the symbol-level regex below. The regex
       is the source-of-truth for family CSV regeneration.

    QC groups returned:

    - ``mt_dna``: mitochondrial genome transcripts.
    - ``mt_like_pseudogene``: NUMT-like mitochondrial pseudogenes.
    - ``rrna_like``: nuclear rRNA and rRNA-pseudogene annotations.
    - ``ribosomal_protein`` / ``ribosomal_protein_pseudogene``: real
      RP biology / library complexity signals.
    - ``small_ncrna``: snoRNA, snRNA, Y RNA, miRNA, vault RNA, SRP RNA.
    - ``histone``: non-polyadenylated replication-dependent histone mRNAs.
    - ``immune_receptor``: immunoglobulin or TCR segments.
    - ``hemoglobin``: erythroid / blood-contamination signal.
    - ``polyadenylation_bias_lncrna``: nuclear-retained ENE-stabilized
      lncRNAs (MALAT1, NEAT1) that survive degradation disproportionately.
    - ``other``: everything else.
    """
    # ENSG-first / symbol-second lookup against the curated pirlygenes
    # gene-family panels. Import lazily — pirlygenes is a runtime
    # dependency, but this fallback path keeps this module importable
    # in test fixtures that monkey-patch around pirlygenes.
    family = None
    try:
        from pirlygenes.gene_families import (
            gene_family_for_ensembl_id,
            gene_family_for_symbol,
        )
    except ImportError:
        # Only swallow ImportError — a missing pirlygenes is the
        # legitimate "fall back to regex" case. A runtime exception
        # inside pirlygenes (broken data file, schema drift, ...)
        # should propagate so the classifier doesn't silently degrade
        # to ``"other"`` for ENSG-only inputs.
        gene_family_for_ensembl_id = None  # type: ignore[assignment]
        gene_family_for_symbol = None  # type: ignore[assignment]

    if ensembl_id and gene_family_for_ensembl_id is not None:
        family = gene_family_for_ensembl_id(ensembl_id)

    if family is None and symbol and gene_family_for_symbol is not None:
        family = gene_family_for_symbol(symbol)

    raw = str(symbol or "").strip()
    upper = raw.upper()

    if family is not None:
        return _family_to_qc_class(family, symbol=upper or None)

    if upper in _GENE_NA:
        return GeneQcClass("unlabeled feature", "other")

    if upper in _POLYA_BIAS_LNCRNA_SYMBOLS:
        return GeneQcClass(
            "nuclear-retained ENE-stabilized lncRNA",
            "polyadenylation_bias_lncrna",
        )

    if upper in {"MT-RNR1", "MT-RNR2"}:
        return GeneQcClass("mitochondrial rRNA", "mt_dna")
    if upper.startswith("MT-"):
        return GeneQcClass("mitochondrial transcript", "mt_dna")
    if re.fullmatch(r"MT(RNR[12]|ATP[68]|CO[123]|CYB|ND[1-6]|ND4L)P\d+", upper):
        return GeneQcClass("mitochondrial pseudogene / NUMT-like", "mt_like_pseudogene")

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

    if re.fullmatch(r"RP[SL]\d+[A-Z]?(P\d+|P)$", upper):
        return GeneQcClass("ribosomal protein pseudogene", "ribosomal_protein_pseudogene")
    if re.fullmatch(r"RP[SL]\d+[A-Z]?", upper) or upper.startswith("RPLP"):
        return GeneQcClass("ribosomal protein", "ribosomal_protein")

    if upper.startswith(("SNORD", "SNORA", "RNU", "Y_RNA", "MIR")):
        return GeneQcClass("small noncoding RNA", "small_ncrna")

    if upper.startswith(
        ("H1-", "H2AC", "H2BC", "H3C", "H4C", "HIST1H", "HIST2H", "HIST3H", "HIST4H")
    ):
        return GeneQcClass("histone transcript", "histone")

    if re.fullmatch(r"HB(A\d?|B|D|E\d?|G\d?|M|Q\d?|Z|ZP\d?|BP\d?)", upper):
        return GeneQcClass("hemoglobin transcript", "hemoglobin")

    if re.fullmatch(
        r"(IGH[ADGME]\d*|IG[HKL][CVJ][A-Z0-9-]*|TR[ABDG][CVJ][A-Z0-9-]*)",
        upper,
    ):
        return GeneQcClass("immune receptor segment", "immune_receptor")

    return GeneQcClass("protein-coding/other", "other")


def is_rescue_feature(symbol: str | None) -> bool:
    """True when a feature should be removed by mtDNA/rRNA rescue."""
    return classify_gene_qc(symbol).group in TECHNICAL_RNA_GROUPS


__all__ = [
    "GeneQcClass",
    "TECHNICAL_RNA_GROUPS",
    "TECHNICAL_RNA_FAMILIES",
    "TECHNICAL_FRACTION",
    "RIBOSOMAL_PROTEIN_FRACTION",
    "OTHER_TECHNICAL_FRACTION",
    "_POLYA_BIAS_LNCRNA_SYMBOLS",
    "_TECHNICAL_RNA_GROUPS",
    "_TECHNICAL_RNA_FAMILIES",
    "_QC_GROUP_DISPLAY_ORDER",
    "_QC_GROUP_DISPLAY_LABEL",
    "_FAMILY_TO_QC",
    "classify_gene_qc",
    "is_rescue_feature",
]
