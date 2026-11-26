"""Active OncoRef CTA evidence and coding-gene partitions with Tsarina MS evidence (#655).

Default CTA membership, selection metadata, and manifest provenance follow
OncoRef's active :mod:`oncoref.target_selection` catalog via
:mod:`pirlygenes.cta_policy`. Historical HPA annotations (``oncoref.cta``) and
Tsarina's independent gene-level ``ms_*`` columns are attached by exact Ensembl
gene ID without redefining active admission.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from oncoref import cta
import pandas as pd
from pyensembl import EnsemblRelease
from tsarina.evidence import CTA_evidence as _current_evidence
from tsarina.gene_sets import is_coding_gene
from tsarina.partition import CTAPartitionDataFrames, CTAPartitionSets

from . import cta_policy


def _active_selection_table() -> pd.DataFrame:
    return cta_policy._active_selection_gene_table().copy()


def _active_cta_gene_ids() -> set[str]:
    return cta_policy.CTA_gene_ids()


def CTA_evidence() -> pd.DataFrame:
    """Return active OncoRef CTA decisions enriched with HPA and Tsarina MS columns.

    The row universe and ``selection_*`` / ``is_current_cta`` fields come from
    OncoRef's active :mod:`oncoref.target_selection` catalog. Historical HPA
    annotations from ``oncoref.cta`` and Tsarina's independent ``ms_*`` columns
    are left-joined by exact Ensembl gene ID; neither annotation layer alters
    active OncoRef admission. ``filtered`` remains an exact alias of
    ``passes_filters``, including missing observations. Active manifest
    provenance is attached to ``df.attrs``.
    """
    selection = _active_selection_table()
    key = "Ensembl_Gene_ID"
    if selection[key].duplicated().any():
        raise ValueError("Active OncoRef CTA selection contains duplicate Ensembl_Gene_IDs")

    historical = cta.cta_evidence().copy()
    hpa_only_cols = [c for c in historical.columns if c not in selection.columns]
    text_hpa_cols = [
        c
        for c in hpa_only_cols
        if pd.api.types.is_string_dtype(historical[c].dtype)
        or pd.api.types.is_object_dtype(historical[c].dtype)
    ]
    if key in historical.columns:
        hpa_join = historical[[key, *hpa_only_cols]]
        merged = selection.merge(hpa_join, on=key, how="left", validate="one_to_one")
    else:
        raise ValueError("Historical HPA evidence lacks Ensembl_Gene_ID column")

    missing_hpa = ~merged[key].isin(historical[key])
    for column in text_hpa_cols:
        merged.loc[missing_hpa, column] = ""
    for column in ("restriction", "restriction_confidence"):
        if column in merged.columns:
            merged[column] = merged[column].replace("", "NO_DATA")
    if "passes_filters" in merged.columns:
        merged["filtered"] = merged["passes_filters"]

    current = _current_evidence()
    if key not in current.columns:
        raise ValueError("Tsarina evidence lacks Ensembl_Gene_ID column")
    ms_columns = [column for column in current.columns if column.startswith("ms_")]
    if "ms_restriction" not in ms_columns:
        raise ValueError("Tsarina evidence lacks its MS restriction column")
    overlap = set(ms_columns) & set(merged.columns)
    if overlap:
        raise ValueError(
            "OncoRef CTA evidence unexpectedly contains MS-owned columns: "
            + ", ".join(sorted(overlap))
        )

    active_ids = set(merged.loc[merged["is_current_cta"].astype(bool), key])
    missing_active_ms = active_ids - set(current[key])
    if missing_active_ms:
        raise ValueError(
            "Tsarina evidence lacks active CTA IDs: " + ", ".join(sorted(missing_active_ms))
        )

    out = merged.merge(current[[key, *ms_columns]], on=key, how="left", validate="one_to_one")
    if "ms_restriction" in out.columns:
        out["ms_restriction"] = out["ms_restriction"].fillna("NO_MS_DATA")
    for column in ms_columns:
        if column != "ms_restriction":
            out[column] = out[column].fillna("")
    out.attrs.update(cta_policy._provenance_attrs())
    return out


def _coding_gene_names(ensembl_release: int) -> dict[str, str]:
    """Detach coding-gene identities before closing the sequence annotation."""
    with EnsemblRelease(ensembl_release) as genome:
        return {gene.gene_id: gene.gene_name for gene in genome.genes() if is_coding_gene(gene)}


def CTA_partition_dataframes(ensembl_release: int = 112) -> CTAPartitionDataFrames:
    """Partition the active OncoRef CTA catalog and an annotated coding background.

    Parameters
    ----------
    ensembl_release : int, optional
        Ensembl release defining the protein-coding background, independent of
        OncoRef's active selection reference annotation.

    Returns
    -------
    CTAPartitionDataFrames
        Active OncoRef CTAs (with ``selection_*``, HPA, and MS evidence), an
        empty ``cta_never_expressed`` compatibility frame (active CTA policy
        requires qualifying expression), and all other coding genes
        (``Symbol`` and ``Ensembl_Gene_ID``), including ``Unsafe``, ``Other``,
        ``Cancer Antigen only``, ``Reproductive Antigen only``, and unassessed
        coding loci.
    """
    evidence = CTA_evidence()
    selected = _active_cta_gene_ids()
    evidence_ids = evidence["Ensembl_Gene_ID"]
    missing = selected - set(evidence_ids)
    if missing:
        raise ValueError(
            "Active CTA evidence lacks selected IDs: " + ", ".join(sorted(missing))
        )
    coding_names = _coding_gene_names(ensembl_release)
    background_ids = set(coding_names) - selected
    background = pd.DataFrame(
        [(coding_names[gene_id], gene_id) for gene_id in sorted(background_ids)],
        columns=["Symbol", "Ensembl_Gene_ID"],
    )
    cta_frame = evidence.loc[evidence_ids.isin(selected)].reset_index(drop=True)
    cta_frame.attrs.update(cta_policy._provenance_attrs())
    never_frame = evidence.iloc[0:0].copy().reset_index(drop=True)
    never_frame.attrs.update(cta_policy._provenance_attrs())
    return CTAPartitionDataFrames(
        cta=cta_frame,
        cta_never_expressed=never_frame,
        non_cta=background,
    )


def _partition_sets(
    ensembl_release: int, column: Literal["Ensembl_Gene_ID", "Symbol"],
) -> CTAPartitionSets:
    """Derive each set view from the same evidence-bearing active partition."""
    frames = CTA_partition_dataframes(ensembl_release)
    return CTAPartitionSets(
        cta=set(frames.cta[column]),
        cta_never_expressed=set(frames.cta_never_expressed[column]),
        non_cta=set(frames.non_cta[column]),
    )


def CTA_partition_gene_ids(ensembl_release: int = 112) -> CTAPartitionSets:
    """Return the active CTA partition as sets of exact Ensembl gene IDs.

    See :func:`CTA_partition_dataframes` for population and reference semantics.
    """
    return _partition_sets(ensembl_release, "Ensembl_Gene_ID")


def CTA_partition_gene_names(ensembl_release: int = 112) -> CTAPartitionSets:
    """Return the active CTA partition as sets of gene symbols.

    See :func:`CTA_partition_dataframes` for population and reference semantics.
    Symbols can be shared by distinct loci; use IDs when locus identity matters.
    """
    return _partition_sets(ensembl_release, "Symbol")


def _normalize_axis_values(values: str | Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = {values}
    return {str(v).strip().upper() for v in values if str(v).strip()}


def CTA_by_axes(
    *,
    restriction: str | Iterable[str] | None = None,
    protein_restriction: str | Iterable[str] | None = None,
    rna_restriction: str | Iterable[str] | None = None,
    rna_restriction_level: str | Iterable[str] | None = None,
    ms_restriction: str | Iterable[str] | None = None,
    restriction_confidence: str | Iterable[str] | None = None,
    column: str = "Symbol",
    filtered_only: bool = True,
) -> set[str]:
    """Filter active OncoRef CTAs by HPA tissue-restriction and Tsarina MS axes.

    Defaults to the active OncoRef CTA set (``filtered_only=True``). Setting
    ``filtered_only=False`` evaluates across all active ``target_selection``
    genes. HPA/MS predicates are explicit secondary filters and never redefine
    OncoRef's upstream CTA admission.
    """
    df = CTA_evidence()
    if column not in df.columns or "Ensembl_Gene_ID" not in df.columns:
        return set()
    allowed_ids = _active_cta_gene_ids() if filtered_only else cta_policy.CTA_unfiltered_gene_ids()
    mask = df["Ensembl_Gene_ID"].astype(str).isin(allowed_ids)
    for col_name, wanted in (
        ("restriction", _normalize_axis_values(restriction)),
        ("protein_restriction", _normalize_axis_values(protein_restriction)),
        ("rna_restriction", _normalize_axis_values(rna_restriction)),
        ("rna_restriction_level", _normalize_axis_values(rna_restriction_level)),
        ("ms_restriction", _normalize_axis_values(ms_restriction)),
        ("restriction_confidence", _normalize_axis_values(restriction_confidence)),
    ):
        if wanted is not None and col_name in df.columns:
            mask &= df[col_name].fillna("").astype(str).str.strip().str.upper().isin(wanted)
    return {s for s in df.loc[mask, column].dropna().astype(str) if s}
