# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .load_dataset import get_data


def get_field_from_gene_set(
    gene_set_name,
    candidate_columns,
    try_lower_case=True,
    try_upper_case=True,
    try_no_underscore=True,
    try_plural=True,
):
    if try_plural:
        candidate_columns = candidate_columns + [c + "s" for c in candidate_columns]
    if try_no_underscore:
        candidate_columns = candidate_columns + [
            c.replace("_", "") for c in candidate_columns
        ]
    if try_lower_case:
        candidate_columns = candidate_columns + [c.lower() for c in candidate_columns]
    if try_upper_case:
        candidate_columns = candidate_columns + [c.upper() for c in candidate_columns]

    df = get_data(gene_set_name)
    result = set()
    for column in candidate_columns:
        if column in df.columns:
            for x in df[column]:
                if type(x) is str:
                    result.update([xi.strip() for xi in x.split(";")])
    return result


def get_target_gene_name_set(
    name,
    candidate_symbol_columns=[
        "Tumor_Target_Symbol",
        "Symbol",
        "Gene_Name",
    ],
):
    return get_field_from_gene_set(name, candidate_symbol_columns)


def get_target_gene_id_set(
    name,
    candidate_id_columns=[
        "Tumor_Target_Ensembl_Gene_ID",
        "Tumor_Target_Ensembl_GeneID",
        "Tumor_Target_Gene_ID",
        "Tumor_Target_GeneID",
        "Target_Ensembl_Gene_ID",
        "Target_Ensembl_GeneID",
        "Target_Gene_ID",
        "Target_GeneID",
        "Ensembl_Gene_ID",
        "Ensembl_GeneID",
        "Gene_ID",
    ],
):
    return get_field_from_gene_set(name, candidate_id_columns)


# ---------- Housekeeping genes ----------
def housekeeping_gene_names(core_only=False):
    df = get_data("housekeeping-genes")
    if core_only:
        df = df[df["Category"] == "Core"]
    return set(df["Symbol"])


def housekeeping_gene_ids(core_only=False):
    df = get_data("housekeeping-genes")
    if core_only:
        df = df[df["Category"] == "Core"]
    return set(df["Ensembl_Gene_ID"])


# ---------- Surface proteins (surfaceome) ----------
def _surface_proteins_df(category=None):
    df = get_data("surface-proteins")
    if category is not None:
        df = df[df["Category"] == category]
    return df


def surface_protein_gene_names(validated_only=False):
    """Human cell surface protein gene symbols.

    Parameters
    ----------
    validated_only : bool
        If True, return only CSPA mass-spec validated proteins.
        If False (default), include ML-predicted surfaceome.
    """
    cat = "CSPA_validated" if validated_only else None
    return set(_surface_proteins_df(cat)["Symbol"])


def surface_protein_gene_ids(validated_only=False):
    """Human cell surface protein Ensembl gene IDs.

    Parameters
    ----------
    validated_only : bool
        If True, return only CSPA mass-spec validated proteins.
        If False (default), include ML-predicted surfaceome.
    """
    cat = "CSPA_validated" if validated_only else None
    df = _surface_proteins_df(cat)
    return set(df.loc[df["Ensembl_Gene_ID"].astype(str).str.startswith("ENSG"), "Ensembl_Gene_ID"])


def surface_protein_evidence():
    """Return the full surface protein DataFrame with categories and sources."""
    return get_data("surface-proteins")


# ---------- Cancer surfaceome (TCSA) ----------
def _cancer_surfaceome_df(min_cancer_types=None):
    df = get_data("cancer-surfaceome")
    if min_cancer_types is not None:
        df = df[df["num_cancer_types"] >= min_cancer_types]
    return df


def cancer_surfaceome_gene_names(min_cancer_types=None):
    """Tumor-specific surface protein gene symbols from TCSA (Lin et al. 2021).

    These are L3-tier (highest stringency) surface proteins with
    tumor-specific overexpression vs normal tissue, excluding
    immune-cell-expressed genes.

    Parameters
    ----------
    min_cancer_types : int or None
        If set, only return genes overexpressed in at least this many
        cancer types.
    """
    return set(_cancer_surfaceome_df(min_cancer_types)["Symbol"])


def cancer_surfaceome_gene_ids(min_cancer_types=None):
    """Tumor-specific surface protein Ensembl gene IDs from TCSA."""
    return set(_cancer_surfaceome_df(min_cancer_types)["Ensembl_Gene_ID"])


def cancer_surfaceome_gene_id_to_name(min_cancer_types=None):
    """Tumor-specific surface protein {Ensembl_Gene_ID: Symbol} dict from TCSA."""
    df = _cancer_surfaceome_df(min_cancer_types)
    return dict(zip(df["Ensembl_Gene_ID"], df["Symbol"]))


def cancer_surfaceome_evidence(min_cancer_types=None):
    """Full TCSA evidence DataFrame with cancer types and druggability."""
    return _cancer_surfaceome_df(min_cancer_types)


# ---------- Pan-cancer expression ----------
def pan_cancer_expression(genes=None, normalize=None, log_transform=False):
    """Expression across 50 normal tissues (nTPM) and 21 TCGA cancer types (median FPKM).

    Data from Human Protein Atlas v23. Normal tissues use consensus nTPM;
    cancer types use median FPKM across TCGA samples. Column names are
    prefixed with ``nTPM_`` or ``FPKM_`` to indicate units.

    Parameters
    ----------
    genes : iterable of str, optional
        Gene symbols or Ensembl IDs to filter to. If None, returns all
        ~3,100 genes in pirlygenes gene sets.
    normalize : str or None
        ``"percentile"`` — within-column percentile ranks (0–100).
        ``"housekeeping"`` — fold over median housekeeping expression.
        ``None`` (default) — raw values.
    log_transform : bool
        If True, apply log2(x + 1) after normalization (or to raw values
        if normalize is None). Recommended for visualization.

    Returns
    -------
    pd.DataFrame
    """
    import numpy as np

    df = get_data("pan-cancer-expression")
    if genes is not None:
        genes_upper = {str(g).upper() for g in genes}
        mask = (
            df["Ensembl_Gene_ID"].str.upper().isin(genes_upper)
            | df["Symbol"].str.upper().isin(genes_upper)
        )
        df = df[mask]

    value_cols = [c for c in df.columns if c.startswith("nTPM_") or c.startswith("FPKM_")]

    if normalize is not None:
        df = df.copy()
        if normalize == "percentile":
            for col in value_cols:
                vals = df[col].astype(float)
                df[col] = vals.rank(pct=True) * 100
        elif normalize == "housekeeping":
            hk_ids = housekeeping_gene_ids()
            hk_mask = df["Ensembl_Gene_ID"].isin(hk_ids)
            for col in value_cols:
                vals = df[col].astype(float)
                hk_median = vals[hk_mask].median()
                if hk_median > 0:
                    df[col] = vals / hk_median
                else:
                    df[col] = np.nan
        else:
            raise ValueError(f"normalize must be 'percentile', 'housekeeping', or None, got {normalize!r}")

    if log_transform:
        df = df.copy() if normalize is None else df
        for col in value_cols:
            df[col] = np.log2(df[col].astype(float) + 1)

    return df


# ---------- ADC ----------
def ADC_trial_target_gene_names():
    return get_target_gene_name_set("ADC-trials")


def ADC_trial_target_gene_ids():
    return get_target_gene_id_set("ADC-trials")


def ADC_approved_target_gene_names():
    return get_target_gene_name_set("ADC-approved")


def ADC_approved_target_gene_ids():
    return get_target_gene_id_set("ADC-approved")


def ADC_target_gene_names():
    return ADC_trial_target_gene_names().union(ADC_approved_target_gene_names())


def ADC_target_gene_ids():
    return ADC_trial_target_gene_ids().union(ADC_approved_target_gene_ids())


# ---------- TCR-T ----------
def TCR_T_trial_target_get_names():
    return get_target_gene_name_set("TCR-T-trials")


def TCR_T_trial_target_get_ids():
    return get_target_gene_id_set("TCR-T-trials")


def TCR_T_approved_target_gene_names():
    return get_target_gene_name_set("TCR-T-approved")


def TCR_T_approved_target_gene_ids():
    return get_target_gene_id_set("TCR-T-approved")


def TCR_T_target_gene_names():
    return TCR_T_trial_target_get_names().union(TCR_T_approved_target_gene_names())


def TCR_T_target_gene_ids():
    return TCR_T_trial_target_get_ids().union(TCR_T_approved_target_gene_ids())


# ---------- CAR-T ----------
def CAR_T_approved_target_gene_names():
    return get_target_gene_name_set("CAR-T-approved")


def CAR_T_approved_target_gene_ids():
    return get_target_gene_id_set("CAR-T-approved")


def CAR_T_target_gene_names():
    return CAR_T_approved_target_gene_names()


def CAR_T_target_gene_ids():
    return CAR_T_approved_target_gene_ids()


# ---------- Multispecific T-cell Engagers ----------

def _tce_filtered_df(pmhc=None):
    """Return TCE trial rows, optionally filtered by format.

    Parameters
    ----------
    pmhc : bool or None
        True  = only pMHC-targeting TCEs (TCR-based formats)
        False = only surface-antigen-targeting TCEs (antibody-based formats)
        None  = all TCEs (trials + approved pMHC TCEs)
    """
    import pandas as pd

    df = get_data("multispecific-tcell-engager-trials")
    # Include approved pMHC TCEs (e.g. tebentafusp) for pMHC=True or None
    if pmhc is not False:
        df_approved = get_data("bispecific-antibodies-approved")
        if "Format" in df_approved.columns:
            is_tcr_approved = df_approved["Format"].str.contains(
                "TCR|ImmTAC", case=False, na=False
            )
            df = pd.concat([df, df_approved[is_tcr_approved]], ignore_index=True)

    if pmhc is not None and "Format" in df.columns:
        is_tcr = df["Format"].str.contains("TCR|ImmTAC", case=False, na=False)
        df = df[is_tcr] if pmhc else df[~is_tcr]
    return df


def _extract_genes_from_df(df, candidate_columns):
    result = set()
    for column in candidate_columns:
        if column in df.columns:
            for x in df[column]:
                if type(x) is str:
                    result.update([xi.strip() for xi in x.split(";")])
    return result


def multispecific_tcell_engager_trial_target_gene_names():
    return get_target_gene_name_set("multispecific-tcell-engager-trials")


def multispecific_tcell_engager_trial_target_gene_ids():
    return get_target_gene_id_set("multispecific-tcell-engager-trials")


def multispecific_tcell_engager_target_gene_names():
    return multispecific_tcell_engager_trial_target_gene_names()


def multispecific_tcell_engager_target_gene_ids():
    return multispecific_tcell_engager_trial_target_gene_ids()


# pMHC-targeting TCEs (TCR-based: ImmTAC, TCR-scFv, scFv-TCR-Fc)
_NAME_COLS = ["Tumor_Target_Symbols", "Tumor_Target_Symbol", "Symbol", "Gene_Name"]
_ID_COLS = [
    "Tumor_Target_Gene_IDs", "Tumor_Target_Ensembl_Gene_IDs",
    "Tumor_Target_Ensembl_Gene_ID", "Ensembl_Gene_ID", "Gene_ID",
]


def pMHC_TCE_target_gene_names():
    return _extract_genes_from_df(_tce_filtered_df(pmhc=True), _NAME_COLS)


def pMHC_TCE_target_gene_ids():
    return _extract_genes_from_df(_tce_filtered_df(pmhc=True), _ID_COLS)


# Surface-antigen-targeting TCEs (antibody-based bispecifics)
def surface_TCE_target_gene_names():
    return _extract_genes_from_df(_tce_filtered_df(pmhc=False), _NAME_COLS)


def surface_TCE_target_gene_ids():
    return _extract_genes_from_df(_tce_filtered_df(pmhc=False), _ID_COLS)


# ---------- Bispecific antibodies ----------
def bispecific_antibody_approved_target_gene_names():
    return get_target_gene_name_set("bispecific-antibodies-approved")


def bispecific_antibody_approved_target_gene_ids():
    return get_target_gene_id_set("bispecific-antibodies-approved")


def bispecific_antibody_target_gene_names():
    return bispecific_antibody_approved_target_gene_names()


def bispecific_antibody_targets_gene_ids():
    return bispecific_antibody_approved_target_gene_ids()


# ---------- Radioligand therapies ----------
def radio_target_gene_names():
    return get_target_gene_name_set("radioligand-targets")


def radio_target_gene_ids():
    return get_target_gene_id_set("radioligand-targets")


def radioligand_target_gene_names():
    return radio_target_gene_names()


def radioligand_target_gene_ids():
    return radio_target_gene_ids()


# ---------- Cancer-testis antigens (CTA) ----------
def _cta_by_column(column, filtered_only=False, exclude_never_expressed=False):
    try:
        from tsarina.evidence import CTA_evidence

        df = CTA_evidence()
    except ImportError:
        from .load_dataset import get_data

        df = get_data("cancer-testis-antigens")
    mask = True
    if filtered_only and "filtered" in df.columns:
        mask = df["filtered"].astype(str).str.lower() == "true"
    if exclude_never_expressed and "never_expressed" in df.columns:
        mask = mask & ~(df["never_expressed"].astype(str).str.lower() == "true")
    subset = df[mask] if not isinstance(mask, bool) else df
    result = set()
    if column in subset.columns:
        for x in subset[column]:
            if isinstance(x, str):
                result.update([xi.strip() for xi in x.split(";")])
    return result


def CTA_gene_names():
    """CTA gene symbols: filtered AND expressed (>= 2 nTPM somewhere).

    This is the recommended default for pMHC discovery. Excludes
    never-expressed CTAs (no protein data + max RNA < 2 nTPM).
    For the full filtered set including never-expressed, use
    ``CTA_filtered_gene_names()``.
    """
    return _cta_by_column("Symbol", filtered_only=True, exclude_never_expressed=True)


def CTA_gene_ids():
    """CTA Ensembl gene IDs: filtered AND expressed."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True, exclude_never_expressed=True)


def CTA_gene_id_to_name():
    """CTA {Ensembl_Gene_ID: Symbol} dict: filtered AND expressed."""
    return dict(zip(
        _cta_by_column("Ensembl_Gene_ID", filtered_only=True, exclude_never_expressed=True),
        _cta_by_column("Symbol", filtered_only=True, exclude_never_expressed=True),
    ))


def CTA_filtered_gene_names():
    """All CTA gene symbols that pass the HPA filter (including never-expressed)."""
    return _cta_by_column("Symbol", filtered_only=True)


def CTA_filtered_gene_ids():
    """All CTA Ensembl gene IDs that pass the HPA filter (including never-expressed)."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True)


def CTA_never_expressed_gene_names():
    """CTA genes that pass filter but have no meaningful HPA expression.

    No protein data AND max RNA nTPM < 2. These are in source databases
    but lack positive evidence of tissue restriction from HPA.
    """
    return CTA_filtered_gene_names() - CTA_gene_names()


def CTA_never_expressed_gene_ids():
    """CTA Ensembl gene IDs that pass filter but have no meaningful HPA expression."""
    return CTA_filtered_gene_ids() - CTA_gene_ids()


def CTA_unfiltered_gene_names():
    """All CTA gene symbols from all source databases (unfiltered).

    This is the full CTA universe — use for excluding CTA genes from
    a non-CTA comparison set.  Any gene in this set was identified as
    a candidate CTA by at least one source database.
    """
    return get_target_gene_name_set("cancer-testis-antigens")


def CTA_unfiltered_gene_ids():
    """All CTA Ensembl gene IDs from all source databases (unfiltered)."""
    return get_target_gene_id_set("cancer-testis-antigens")


def CTA_excluded_gene_names():
    """CTA genes that FAIL the reproductive-tissue filter.

    These are candidate CTAs with evidence of somatic tissue expression.
    Use this set to exclude from a non-CTA comparison set: they should
    not be in the clean CTA set (they leak into healthy tissue) but also
    should not be in a non-CTA set (they are still CTA candidates).
    """
    return CTA_unfiltered_gene_names() - CTA_filtered_gene_names()


def CTA_excluded_gene_ids():
    """CTA Ensembl gene IDs that FAIL the reproductive-tissue filter."""
    return CTA_unfiltered_gene_ids() - CTA_filtered_gene_ids()


def CTA_evidence():
    """Return the full CTA evidence DataFrame with HPA tissue-restriction columns.

    Columns
    -------
    Symbol, Aliases, Full_Name, Function, Ensembl_Gene_ID,
    source_databases, biotype, Canonical_Transcript_ID
        Gene identity fields.
    protein_reproductive : bool or "no data"
        True if all IHC-detected tissues (excluding thymus) are in
        {testis, ovary, placenta}.
    protein_thymus : bool or "no data"
        True if protein detected in thymus.
    rna_reproductive : bool
        True if every tissue with >=1 nTPM (excluding thymus) is in
        {testis, ovary, placenta}.
    rna_thymus : bool
        True if thymus nTPM >= 1.
    protein_reliability : str
        Best HPA antibody reliability for this gene: "Enhanced",
        "Supported", "Approved", "Uncertain", or "no data".
    protein_strict_expression : str
        Semicolon-separated list of tissues where protein is detected
        (excluding thymus), or "no data" / "not detected".
    rna_reproductive_frac : float
        Fraction of total nTPM (excluding thymus) in core reproductive
        tissues, computed from raw nTPM values.
    rna_reproductive_and_thymus_frac : float
        Same but with thymus nTPM added to numerator and denominator.
    rna_deflated_reproductive_frac : float
        (1 + repro_deflated) / (1 + total_deflated) where each tissue
        is deflated via max(0, nTPM - 1).  The +1 pseudocount prevents
        0/0 for very-low-expression genes.
    rna_deflated_reproductive_and_thymus_frac : float
        Same but with thymus deflated nTPM added to the reproductive
        numerator.
    rna_80_pct_filter, rna_90_pct_filter, rna_95_pct_filter : bool
        Whether deflated reproductive fraction >= 80/90/95%.
    filtered : bool
        Final inclusion flag with tiered RNA thresholds based on protein
        antibody reliability.  True when protein is reproductive-only
        (or absent) and deflated RNA fraction meets the tier threshold:
        - Enhanced → RNA >=80%
        - Supported → RNA >=90%
        - Approved → RNA >=95%
        - Uncertain or no protein data → RNA >=99%
        Genes with protein in non-reproductive tissues always fail.
        Non-protein-coding genes (biotype != protein_coding) always fail.
    rna_max_ntpm : float
        Maximum nTPM across all tissues for this gene.
    never_expressed : bool
        True if no HPA protein data AND maximum RNA nTPM < 2.
    """
    try:
        from tsarina.evidence import CTA_evidence as _tsarina_evidence

        return _tsarina_evidence()
    except ImportError:
        from .load_dataset import get_data

        return get_data("cancer-testis-antigens")


from dataclasses import dataclass  # noqa: E402

import pandas as pd  # noqa: E402


@dataclass(frozen=True)
class CTAPartitionSets:
    """Three-way partition of protein-coding genes as sets.

    Attributes
    ----------
    cta : set[str]
        Expressed, reproductive-restricted CTAs. Source of CTA pMHCs.
    cta_never_expressed : set[str]
        CTAs from source databases with no meaningful HPA expression
        (no protein data + max RNA < 2 nTPM).
    non_cta : set[str]
        All other protein-coding genes, including CTAs that fail the
        reproductive-tissue filter (somatic expression detected).
    """

    cta: set
    cta_never_expressed: set
    non_cta: set


@dataclass(frozen=True)
class CTAPartitionDataFrames:
    """Three-way partition of protein-coding genes as DataFrames.

    Attributes
    ----------
    cta : pd.DataFrame
        Expressed, reproductive-restricted CTAs with full evidence columns.
    cta_never_expressed : pd.DataFrame
        Never-expressed CTAs with full evidence columns.
    non_cta : pd.DataFrame
        All other protein-coding genes (Symbol, Ensembl_Gene_ID).
    """

    cta: pd.DataFrame
    cta_never_expressed: pd.DataFrame
    non_cta: pd.DataFrame


def _build_partition(ensembl_release=112):
    """Shared logic for building the three-way partition."""
    from pyensembl import EnsemblRelease

    ensembl = EnsemblRelease(ensembl_release)
    evidence_df = CTA_evidence()

    all_pc_genes = {
        g.gene_id: g.gene_name
        for g in ensembl.genes()
        if g.biotype == "protein_coding"
    }
    all_pc_ids = set(all_pc_genes.keys())

    filtered_mask = evidence_df["filtered"].astype(str).str.lower() == "true"
    never_expr_mask = evidence_df["never_expressed"].astype(str).str.lower() == "true"

    cta_mask = filtered_mask & ~never_expr_mask
    never_expressed_mask = filtered_mask & never_expr_mask

    cta_ids = set(evidence_df.loc[cta_mask, "Ensembl_Gene_ID"])
    never_expressed_ids = set(evidence_df.loc[never_expressed_mask, "Ensembl_Gene_ID"])
    non_cta_ids = all_pc_ids - cta_ids - never_expressed_ids

    return all_pc_genes, evidence_df, cta_mask, never_expressed_mask, cta_ids, never_expressed_ids, non_cta_ids


def CTA_partition_gene_ids(ensembl_release=112) -> CTAPartitionSets:
    """Partition all protein-coding genes into CTA / never-expressed / non-CTA
    as sets of Ensembl gene IDs.

    CTAs that fail the reproductive-tissue filter (somatic expression)
    are included in ``non_cta``.

    Examples
    --------
    >>> p = CTA_partition_gene_ids()
    >>> "ENSG00000147381" in p.cta   # MAGEA4
    True
    >>> len(p.cta & p.non_cta)       # no overlap
    0
    """
    _, _, _, _, cta_ids, never_expressed_ids, non_cta_ids = _build_partition(ensembl_release)
    return CTAPartitionSets(
        cta=cta_ids,
        cta_never_expressed=never_expressed_ids,
        non_cta=non_cta_ids,
    )


def CTA_partition_gene_names(ensembl_release=112) -> CTAPartitionSets:
    """Partition all protein-coding genes into CTA / never-expressed / non-CTA
    as sets of gene symbols.

    CTAs that fail the reproductive-tissue filter (somatic expression)
    are included in ``non_cta``.

    Examples
    --------
    >>> p = CTA_partition_gene_names()
    >>> "MAGEA4" in p.cta
    True
    >>> "TP53" in p.non_cta
    True
    """
    all_pc_genes, evidence_df, cta_mask, never_expressed_mask, _, _, _ = _build_partition(ensembl_release)
    all_pc_names = set(all_pc_genes.values())

    cta_names = set(evidence_df.loc[cta_mask, "Symbol"])
    never_expressed_names = set(evidence_df.loc[never_expressed_mask, "Symbol"])
    non_cta_names = all_pc_names - cta_names - never_expressed_names

    return CTAPartitionSets(
        cta=cta_names,
        cta_never_expressed=never_expressed_names,
        non_cta=non_cta_names,
    )


def CTA_partition_dataframes(ensembl_release=112) -> CTAPartitionDataFrames:
    """Partition all protein-coding genes into CTA / never-expressed / non-CTA
    as DataFrames.

    The ``cta`` and ``cta_never_expressed`` DataFrames include all CTA
    evidence columns. The ``non_cta`` DataFrame has Symbol and
    Ensembl_Gene_ID columns.

    CTAs that fail the reproductive-tissue filter (somatic expression)
    are included in ``non_cta``.

    Examples
    --------
    >>> p = CTA_partition_dataframes()
    >>> "rna_deflated_reproductive_frac" in p.cta.columns
    True
    """
    all_pc_genes, evidence_df, cta_mask, never_expressed_mask, _, _, non_cta_ids = _build_partition(ensembl_release)

    non_cta_records = [
        {"Symbol": all_pc_genes[gid], "Ensembl_Gene_ID": gid}
        for gid in sorted(non_cta_ids)
        if gid in all_pc_genes
    ]

    return CTAPartitionDataFrames(
        cta=evidence_df.loc[cta_mask].copy().reset_index(drop=True),
        cta_never_expressed=evidence_df.loc[never_expressed_mask].copy().reset_index(drop=True),
        non_cta=pd.DataFrame(non_cta_records),
    )


def CTA_partition(return_type="gene_ids", ensembl_release=112):
    """Deprecated — use CTA_partition_gene_ids, CTA_partition_gene_names,
    or CTA_partition_dataframes instead."""
    if return_type == "gene_ids":
        return CTA_partition_gene_ids(ensembl_release)
    elif return_type == "gene_names":
        return CTA_partition_gene_names(ensembl_release)
    elif return_type == "dataframes":
        return CTA_partition_dataframes(ensembl_release)
    else:
        raise ValueError(f"return_type must be 'gene_ids', 'gene_names', or 'dataframes', got {return_type!r}")
