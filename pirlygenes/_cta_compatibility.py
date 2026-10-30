"""Historical OncoRef CTA membership with Tsarina's independent MS evidence.

PirlyGenes' compatibility APIs use the historical HPA panel consistently.
Tsarina's default current-selection policy does not define this population.
Its existing partition containers remain the public structural return types.
"""

from __future__ import annotations

from typing import Literal

from oncoref import cta
import pandas as pd
from pyensembl import EnsemblRelease
from tsarina.evidence import CTA_evidence as _current_evidence
from tsarina.gene_sets import is_coding_gene
from tsarina.partition import CTAPartitionDataFrames, CTAPartitionSets


def CTA_evidence() -> pd.DataFrame:
    """Return historical HPA evidence enriched with Tsarina's MS columns.

    Membership, identities and HPA annotations come from ``oncoref.cta``.
    Only ``ms_*`` columns are joined from Tsarina by exact Ensembl gene ID;
    its current-selection flags do not redefine this historical panel.
    ``filtered`` remains an alias of ``passes_filters``, including missing
    observations. The returned frame is independent of both providers' caches.
    """
    historical = cta.cta_evidence().copy()
    current = _current_evidence()
    key = "Ensembl_Gene_ID"
    missing = set(historical[key]) - set(current[key])
    if missing:
        raise ValueError("Tsarina evidence lacks historical CTA IDs: " + ", ".join(sorted(missing)))
    ms_columns = [column for column in current.columns if column.startswith("ms_")]
    if "ms_restriction" not in ms_columns:
        raise ValueError("Tsarina evidence lacks its MS restriction column")
    overlap = set(ms_columns) & set(historical.columns)
    if overlap:
        raise ValueError("Historical HPA evidence contains MS-owned columns: " + ", ".join(sorted(overlap)))
    historical["filtered"] = historical["passes_filters"]
    return historical.merge(current[[key, *ms_columns]], on=key, how="left", validate="one_to_one")


def _coding_gene_names(ensembl_release: int) -> dict[str, str]:
    """Detach coding-gene identities before closing the sequence annotation."""
    with EnsemblRelease(ensembl_release) as genome:
        return {gene.gene_id: gene.gene_name for gene in genome.genes() if is_coding_gene(gene)}


def CTA_partition_dataframes(ensembl_release: int = 112) -> CTAPartitionDataFrames:
    """Partition the historical CTA panel and an annotated coding background.

    Parameters
    ----------
    ensembl_release : int, optional
        Ensembl release defining the protein-coding background, independent of
        the historical panel's reference annotation.

    Returns
    -------
    CTAPartitionDataFrames
        Historical expressed CTAs, historical never-expressed CTAs, and all
        other coding genes. The two CTA frames retain HPA and MS evidence.
        The background contains ``Symbol`` and ``Ensembl_Gene_ID`` columns.
        Current Neither/unassigned decisions do not alter these populations.
    """
    evidence = CTA_evidence()
    selected = cta.cta_gene_ids()
    never_expressed = cta.cta_never_expressed_gene_ids()
    if selected & never_expressed:
        raise ValueError("Historical expressed and never-expressed CTA sets overlap")
    evidence_ids = evidence["Ensembl_Gene_ID"]
    missing = (selected | never_expressed) - set(evidence_ids)
    if missing:
        raise ValueError("Historical CTA evidence lacks selected IDs: " + ", ".join(sorted(missing)))
    coding_names = _coding_gene_names(ensembl_release)
    background_ids = set(coding_names) - selected - never_expressed
    background = pd.DataFrame(
        [(coding_names[gene_id], gene_id) for gene_id in sorted(background_ids)],
        columns=["Symbol", "Ensembl_Gene_ID"],
    )
    return CTAPartitionDataFrames(
        cta=evidence.loc[evidence_ids.isin(selected)].reset_index(drop=True),
        cta_never_expressed=evidence.loc[evidence_ids.isin(never_expressed)].reset_index(drop=True),
        non_cta=background,
    )


def _partition_sets(
    ensembl_release: int, column: Literal["Ensembl_Gene_ID", "Symbol"],
) -> CTAPartitionSets:
    """Derive each set view from the same evidence-bearing partition."""
    frames = CTA_partition_dataframes(ensembl_release)
    return CTAPartitionSets(
        cta=set(frames.cta[column]),
        cta_never_expressed=set(frames.cta_never_expressed[column]),
        non_cta=set(frames.non_cta[column]),
    )


def CTA_partition_gene_ids(ensembl_release: int = 112) -> CTAPartitionSets:
    """Return the historical partition as sets of exact Ensembl gene IDs.

    See :func:`CTA_partition_dataframes` for population and reference semantics.
    """
    return _partition_sets(ensembl_release, "Ensembl_Gene_ID")


def CTA_partition_gene_names(ensembl_release: int = 112) -> CTAPartitionSets:
    """Return the historical partition as sets of gene symbols.

    See :func:`CTA_partition_dataframes` for population and reference semantics.
    Symbols can be shared by distinct loci; use IDs when locus identity matters.
    """
    return _partition_sets(ensembl_release, "Symbol")
