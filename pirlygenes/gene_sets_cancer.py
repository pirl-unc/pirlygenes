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


def TCR_T_target_gene_names():
    return TCR_T_trial_target_get_names()


def TCR_T_target_gene_ids():
    return TCR_T_trial_target_get_ids()


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
def multispecific_tcell_engager_trial_target_gene_names():
    return get_target_gene_name_set("multispecific-tcell-engager-trials")


def multispecific_tcell_engager_trial_target_gene_ids():
    return get_target_gene_id_set("multispecific-tcell-engager-trials")


def multispecific_tcell_engager_target_gene_names():
    return multispecific_tcell_engager_trial_target_gene_names()


def multispecific_tcell_engager_target_gene_ids():
    return multispecific_tcell_engager_trial_target_gene_ids()


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
    from .load_dataset import get_data

    return get_data("cancer-testis-antigens")


def CTA_partition(return_type="gene_ids", ensembl_release=112):
    """Partition all protein-coding genes into CTA, never-expressed CTA, and non-CTA.

    Returns three non-overlapping sets whose union is the full set of
    protein-coding genes from the given Ensembl release.

    Parameters
    ----------
    return_type : str
        What to return for each partition:
        - ``"gene_ids"`` — sets of Ensembl gene IDs (default)
        - ``"gene_names"`` — sets of gene symbols
        - ``"dataframes"`` — DataFrames with Symbol, Ensembl_Gene_ID,
          and (for CTAs) all evidence columns
    ensembl_release : int
        Ensembl release to use for the full protein-coding gene list
        (default 112).

    Returns
    -------
    dict with keys:

    ``"cta"``
        Reproductive-restricted CTAs (pass HPA filter, not never_expressed).
        Use as the source of CTA pMHCs.
    ``"cta_never_expressed"``
        CTAs from source databases with no meaningful HPA expression
        (no protein data + max RNA < 2 nTPM). These pass the filter
        on a technicality (pseudocount) but lack positive evidence of
        tissue restriction. Exclude from both CTA and non-CTA sets.
    ``"cta_excluded"``
        CTAs that fail the reproductive-tissue filter (somatic
        expression detected). Exclude from non-CTA comparison sets.
    ``"non_cta"``
        All other protein-coding genes. Use as the non-CTA comparison
        set for pMHC analysis.

    Examples
    --------
    >>> p = CTA_partition()
    >>> len(p["cta"] & p["non_cta"])  # no overlap
    0
    >>> len(p["cta"] | p["cta_never_expressed"] | p["cta_excluded"] | p["non_cta"])
    20000  # approximately — all protein-coding genes

    >>> p = CTA_partition(return_type="gene_names")
    >>> "MAGEA4" in p["cta"]
    True

    >>> p = CTA_partition(return_type="dataframes")
    >>> p["cta"].columns  # full evidence columns for CTAs
    """
    from pyensembl import EnsemblRelease

    ensembl = EnsemblRelease(ensembl_release)
    evidence_df = CTA_evidence()

    # All protein-coding genes from Ensembl
    all_pc_genes = {
        g.gene_id: g.gene_name
        for g in ensembl.genes()
        if g.biotype == "protein_coding"
    }
    all_pc_ids = set(all_pc_genes.keys())
    all_pc_names = set(all_pc_genes.values())

    # CTA partitions from evidence table
    filtered_mask = evidence_df["filtered"].astype(str).str.lower() == "true"
    never_expr_mask = evidence_df["never_expressed"].astype(str).str.lower() == "true"

    cta_mask = filtered_mask & ~never_expr_mask
    never_expressed_mask = filtered_mask & never_expr_mask
    excluded_mask = ~filtered_mask

    cta_ids = set(evidence_df.loc[cta_mask, "Ensembl_Gene_ID"])
    never_expressed_ids = set(evidence_df.loc[never_expressed_mask, "Ensembl_Gene_ID"])
    excluded_ids = set(evidence_df.loc[excluded_mask, "Ensembl_Gene_ID"])
    non_cta_ids = all_pc_ids - cta_ids - never_expressed_ids - excluded_ids

    if return_type == "gene_ids":
        return {
            "cta": cta_ids,
            "cta_never_expressed": never_expressed_ids,
            "cta_excluded": excluded_ids,
            "non_cta": non_cta_ids,
        }
    elif return_type == "gene_names":
        cta_names = set(evidence_df.loc[cta_mask, "Symbol"])
        never_expressed_names = set(evidence_df.loc[never_expressed_mask, "Symbol"])
        excluded_names = set(evidence_df.loc[excluded_mask, "Symbol"])
        non_cta_names = all_pc_names - cta_names - never_expressed_names - excluded_names
        return {
            "cta": cta_names,
            "cta_never_expressed": never_expressed_names,
            "cta_excluded": excluded_names,
            "non_cta": non_cta_names,
        }
    elif return_type == "dataframes":
        import pandas as pd

        non_cta_records = [
            {"Symbol": all_pc_genes[gid], "Ensembl_Gene_ID": gid}
            for gid in non_cta_ids
            if gid in all_pc_genes
        ]
        return {
            "cta": evidence_df.loc[cta_mask].copy().reset_index(drop=True),
            "cta_never_expressed": evidence_df.loc[never_expressed_mask].copy().reset_index(drop=True),
            "cta_excluded": evidence_df.loc[excluded_mask].copy().reset_index(drop=True),
            "non_cta": pd.DataFrame(non_cta_records),
        }
    else:
        raise ValueError(
            f"return_type must be 'gene_ids', 'gene_names', or 'dataframes', got {return_type!r}"
        )
