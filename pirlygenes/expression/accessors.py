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

"""Cancer reference expression matrices + lightweight normalization.

This module bundles the curated cross-cohort expression panels that
``pirlygenes`` ships as reference data, plus the small set of
normalization helpers needed to make them comparable across columns:

* :func:`pan_cancer_expression` — wide-form ``Symbol × tissue/cancer``
  panel: 50 HPA normal tissues (nTPM) + 33 TCGA cancer types (FPKM)
  + the per-cancer tumor-only deconvolved medians from
  :func:`tcga_deconvolved_expression` (``tcga_<CODE>`` columns).
* :func:`tcga_deconvolved_expression`,
  :func:`subtype_deconvolved_expression` — long-form tumor-only TPM
  medians (full-cohort + subtype-stratified).
* :func:`tumor_up_vs_matched_normal`,
  :func:`heme_tumor_up_vs_matched_normal` — per-cancer
  tumor-vs-tissue-of-origin enrichment panels.
* :func:`hpa_cell_type_expression` — HPA cell-type single-cell
  reference (long-form ``Symbol, cell_type, nTPM``).
* :func:`estimate_signatures` — the ESTIMATE stromal/immune signature
  gene sets (Yoshihara et al., 2013).

The normalization layer is intentionally narrow — anything that needs
per-sample QC (degradation index, FFPE rescue, library-prep
classification) lives in ``trufflepig.expression_qc``. What's here:

* :func:`normalize_to_housekeeping` — divide each column by its
  housekeeping-gene median.
* :func:`log2_transform` — log2(x + 1) over value columns.
* :func:`filter_technical_rna` — drop mtDNA / NUMT-like / rRNA-like /
  nuclear-retained-lncRNA rows by ENSG, sourced from
  :mod:`pirlygenes.gene_families` (no symbol-regex dependency).
* :func:`filter_to_genes` — subset to a caller-provided gene list.

The accessors expose ``normalize="housekeeping"``, ``log_transform=``,
and ``filter_technical_rna=`` keyword arguments that pipeline the
free functions in the expected order — for callers who prefer one
call to a chain of helpers.

Returned frames are always ``.copy()``'d from the cached CSV; callers
can mutate freely.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..gene_families import gene_family_ids
from ..gene_sets_cancer import housekeeping_gene_ids
from ..load_dataset import get_data


# ---------- column-discovery helpers ----------


_VALUE_COL_PREFIXES = ("nTPM_", "FPKM_", "tcga_", "TPM_")


def _default_value_cols(df: pd.DataFrame) -> list[str]:
    """Heuristic: wide-form expression frames use prefixed column names."""
    return [c for c in df.columns if c.startswith(_VALUE_COL_PREFIXES)]


def _resolve_id_col(df: pd.DataFrame) -> Optional[str]:
    """Find the Ensembl-ID column — wide frames use ``Ensembl_Gene_ID``,
    long frames may use ``ensembl_gene_id``."""
    for cand in ("Ensembl_Gene_ID", "ensembl_gene_id", "Ensembl_ID"):
        if cand in df.columns:
            return cand
    return None


# ---------- normalization helpers (free functions, composable) ----------


def normalize_to_housekeeping(
    df: pd.DataFrame,
    value_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Rescale each value column by the median housekeeping-gene level.

    The result is unitless: a value of 1.0 in a given column means
    "expressed at the column's housekeeping baseline". Works across
    TPM, FPKM, and nTPM units since the normalization is per-column.

    Parameters
    ----------
    df
        Expression frame with an ``Ensembl_Gene_ID`` column and one or
        more numeric value columns.
    value_cols
        Columns to rescale. If ``None``, picks columns starting with
        ``nTPM_``, ``FPKM_``, ``tcga_``, or ``TPM_``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the named columns rescaled in place.
    """
    id_col = _resolve_id_col(df)
    if id_col is None:
        raise ValueError(
            "normalize_to_housekeeping needs an Ensembl_Gene_ID column"
        )
    cols = list(value_cols) if value_cols is not None else _default_value_cols(df)
    hk_ids = housekeeping_gene_ids()
    hk_mask = df[id_col].isin(hk_ids)
    out = df.copy()
    for col in cols:
        vals = out[col].astype(float)
        hk_median = vals[hk_mask].median()
        if np.isnan(hk_median) or hk_median <= 0:
            out[col] = np.nan
        else:
            out[col] = vals / hk_median
    return out


def log2_transform(
    df: pd.DataFrame,
    value_cols: Optional[Sequence[str]] = None,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Apply ``log2(x + pseudocount)`` to expression columns.

    Useful for visualization and for damping the long right tail of
    TPM/FPKM distributions. Idempotent only if the caller tracks the
    transformed state externally.
    """
    cols = list(value_cols) if value_cols is not None else _default_value_cols(df)
    out = df.copy()
    for col in cols:
        out[col] = np.log2(out[col].astype(float) + pseudocount)
    return out


# Family groups that count as "technical RNA" — high-abundance
# transcript classes that contaminate reference panels without
# carrying biological signal. Sourced from gene_families so the set
# stays in lockstep with the curated CSVs.
_TECHNICAL_RNA_FAMILIES = (
    "mitochondrial",
    "numt_pseudogene",
    "rrna_and_pseudogene",
    "nuclear_retained_lncrna",
)


def technical_rna_gene_ids() -> set[str]:
    """Union of ENSG IDs across the technical-RNA families.

    Drives :func:`filter_technical_rna`. Exposed in case callers want
    to project onto a frame that doesn't carry ``Ensembl_Gene_ID``.
    """
    out: set[str] = set()
    for family in _TECHNICAL_RNA_FAMILIES:
        out |= gene_family_ids(family)
    return out


def filter_technical_rna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ENSG is in mtDNA / NUMT / rRNA / nuclear-retained-lncRNA.

    Returns a copy with those rows removed. Uses pirlygenes'
    ``gene_families`` CSVs as the source of truth — the regex panel
    in ``trufflepig.expression_qc.classify_gene_qc`` generates those
    CSVs, but at use-time we only need the ENSG sets.
    """
    id_col = _resolve_id_col(df)
    if id_col is None:
        raise ValueError(
            "filter_technical_rna needs an Ensembl_Gene_ID column"
        )
    drop_ids = technical_rna_gene_ids()
    return df[~df[id_col].isin(drop_ids)].reset_index(drop=True)


def filter_to_genes(
    df: pd.DataFrame,
    genes: Iterable[str],
) -> pd.DataFrame:
    """Subset rows to a caller-provided list of symbols or Ensembl IDs.

    Match is case-insensitive against both ``Symbol`` (or ``symbol``)
    and the Ensembl-ID column.
    """
    targets = {str(g).upper() for g in genes}
    id_col = _resolve_id_col(df)
    sym_col = next(
        (c for c in ("Symbol", "symbol", "Gene_Symbol") if c in df.columns),
        None,
    )
    if id_col is None and sym_col is None:
        raise ValueError(
            "filter_to_genes needs a Symbol or Ensembl_Gene_ID column"
        )
    mask = pd.Series(False, index=df.index)
    if id_col is not None:
        mask |= df[id_col].astype(str).str.upper().isin(targets)
    if sym_col is not None:
        mask |= df[sym_col].astype(str).str.upper().isin(targets)
    return df[mask].reset_index(drop=True)


def _apply_pipeline(
    df: pd.DataFrame,
    *,
    filter_technical_rna_: bool = False,
    genes: Optional[Iterable[str]] = None,
    normalize: Optional[str] = None,
    log_transform: bool = False,
    value_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Shared accessor-kwarg pipeline. Order matters: family filter →
    gene subset → cross-column normalization → log transform."""
    if filter_technical_rna_:
        df = filter_technical_rna(df)
    if genes is not None:
        df = filter_to_genes(df, genes)
    if normalize is not None:
        if normalize == "housekeeping":
            df = normalize_to_housekeeping(df, value_cols=value_cols)
        elif normalize == "percentile":
            cols = list(value_cols) if value_cols else _default_value_cols(df)
            df = df.copy()
            for col in cols:
                df[col] = df[col].astype(float).rank(pct=True) * 100
        else:
            raise ValueError(
                f"normalize must be 'housekeeping', 'percentile', or None — "
                f"got {normalize!r}"
            )
    if log_transform:
        df = log2_transform(df, value_cols=value_cols)
    return df


# ---------- accessors: pan-cancer expression ----------


@lru_cache(maxsize=1)
def _tcga_deconv_wide_cached() -> Optional[pd.DataFrame]:
    """Wide-form (``Symbol``-indexed) view of the tumor-only TPM medians.

    Returns ``None`` when ``tcga-deconvolved-expression.csv.gz`` isn't
    bundled (so callers can degrade gracefully)."""
    long = tcga_deconvolved_expression()
    if long is None or long.empty:
        return None
    wide = long.pivot_table(
        index="symbol",
        columns="cancer_code",
        values="tumor_tpm_median",
        aggfunc="median",
    )
    wide.columns = [f"tcga_{c}" for c in wide.columns]
    wide = wide.reset_index().rename(columns={"symbol": "Symbol"})
    return wide


def pan_cancer_expression(
    genes: Optional[Iterable[str]] = None,
    normalize: Optional[str] = None,
    log_transform: bool = False,
    filter_technical_rna: bool = False,
) -> pd.DataFrame:
    """Wide-form expression across HPA normal tissues + TCGA cancer types.

    50 normal tissues from HPA v23 consensus (``nTPM_<tissue>`` columns)
    plus 33 TCGA cancer types (``FPKM_<code>`` columns from HPA pathology
    + GDC/STAR reprocessing) plus per-cancer tumor-only deconvolved
    medians from :func:`tcga_deconvolved_expression` (``tcga_<code>``).

    Parameters
    ----------
    genes
        Optional iterable of gene symbols or Ensembl IDs to subset to.
    normalize
        ``"housekeeping"`` — divide each column by its housekeeping median.
        ``"percentile"`` — within-column percentile rank (0–100).
        ``None`` (default) — raw FPKM / nTPM / tumor-only TPM.
    log_transform
        Apply ``log2(x + 1)`` to value columns after any normalization.
    filter_technical_rna
        Drop mtDNA / NUMT / rRNA / nuclear-retained-lncRNA rows before
        normalization. Recommended when the goal is to compare protein-
        coding signal across cohorts.

    Returns
    -------
    pd.DataFrame
        Defensive copy — safe to mutate.
    """
    df = get_data("pan-cancer-expression")
    deconv_wide = _tcga_deconv_wide_cached()
    if deconv_wide is not None:
        df = df.merge(deconv_wide, on="Symbol", how="left")
    return _apply_pipeline(
        df,
        filter_technical_rna_=filter_technical_rna,
        genes=genes,
        normalize=normalize,
        log_transform=log_transform,
    )


def cancer_expression(
    cancer_type: str,
    genes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Housekeeping-normalized expression for a single cancer type.

    Parameters
    ----------
    cancer_type
        TCGA code or alias (e.g. ``"PRAD"``, ``"prostate"``).
    genes
        Optional gene-symbol / Ensembl-ID subset.

    Returns
    -------
    pd.DataFrame
        Columns: ``Ensembl_Gene_ID``, ``Symbol``, ``expression``
        (housekeeping-normalized, technical-RNA-filtered).
    """
    from ..gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    df = pan_cancer_expression(
        genes=genes,
        normalize="housekeeping",
        filter_technical_rna=True,
    )
    col = f"FPKM_{code}"
    if col not in df.columns:
        raise ValueError(
            f"no FPKM column for {cancer_type!r} (resolved to {code!r})"
        )
    return df[["Ensembl_Gene_ID", "Symbol", col]].rename(
        columns={col: "expression"}
    )


def cancer_enriched_genes(
    cancer_type: str,
    min_fold: float = 3.0,
    min_expression: float = 0.01,
) -> pd.DataFrame:
    """Genes enriched in one cancer type vs the pan-cancer median.

    Parameters
    ----------
    cancer_type
        TCGA code or alias.
    min_fold
        Minimum fold-change over the median of all other cancer types.
    min_expression
        Minimum housekeeping-normalized expression in the target cancer.

    Returns
    -------
    pd.DataFrame
        Columns: ``Ensembl_Gene_ID``, ``Symbol``, ``expression``,
        ``other_median``, ``fold_change``. Sorted by fold_change desc.
    """
    from ..gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    df = pan_cancer_expression(
        normalize="housekeeping",
        filter_technical_rna=True,
    )
    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")]
    target_col = f"FPKM_{code}"
    if target_col not in df.columns:
        raise ValueError(
            f"no FPKM column for {cancer_type!r} (resolved to {code!r})"
        )
    other_cols = [c for c in fpkm_cols if c != target_col]
    result = df[["Ensembl_Gene_ID", "Symbol"]].copy()
    result["expression"] = df[target_col].astype(float)
    result["other_median"] = df[other_cols].astype(float).median(axis=1)
    result["fold_change"] = (result["expression"] + 0.001) / (
        result["other_median"] + 0.001
    )
    result = result[
        (result["expression"] >= min_expression)
        & (result["fold_change"] >= min_fold)
    ].sort_values("fold_change", ascending=False)
    return result.reset_index(drop=True)


# ---------- accessors: TCGA + subtype-deconvolved tumor-only TPM ----------


def tcga_deconvolved_expression() -> Optional[pd.DataFrame]:
    """Long-form per-(symbol, TCGA code) tumor-only TPM medians.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``symbol``, ``cancer_code``, ``tumor_tpm_median``,
        ``tumor_tpm_q1``, ``tumor_tpm_q3``, ``n_samples``. ``None`` if
        the CSV isn't bundled.
    """
    try:
        return get_data("tcga-deconvolved-expression").copy()
    except ValueError:
        return None


def subtype_deconvolved_expression() -> Optional[pd.DataFrame]:
    """Long-form per-(cancer_code, subtype, symbol) tumor-only TPM medians.

    Subtype-stratified companion to :func:`tcga_deconvolved_expression`.
    Covers BRCA × PAM50, BeatAML × ELN2017, TARGET pediatric cohorts,
    SCLC, LUAD × mutation class, HNSC × HPV, Treehouse PolyA/RiboD
    public samples, GSE299759 chondrosarcoma, GSE75885 sarcoma splits,
    and more.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``symbol``, ``cancer_code``, ``subtype``,
        ``tumor_tpm_median``, ``tumor_tpm_q1``, ``tumor_tpm_q3``,
        ``n_samples``. ``None`` if the CSV isn't bundled.
    """
    try:
        return get_data("subtype-deconvolved-expression").copy()
    except ValueError:
        return None


# ---------- accessors: tumor-up vs matched normal panels ----------


def tumor_up_vs_matched_normal(
    cancer_code: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Per-cancer genes dramatically up in tumor vs matched normal tissue.

    Built offline by racing the shipped :func:`tcga_deconvolved_expression`
    tumor-only medians against the matched-tissue ``nTPM_<tissue>``
    columns in :func:`pan_cancer_expression`, filtered so each gene
    is genuinely low across *all* HPA normal tissues and specific to
    ≤ 4 cancer codes.

    Parameters
    ----------
    cancer_code
        Filter to a single TCGA code. ``None`` returns all rows.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``cancer_code``, ``matched_normal_tissue``, ``symbol``,
        ``ensembl_gene_id``, ``fold_change_vs_matched_normal``,
        ``tumor_tpm``, ``matched_normal_ntpm``, ``max_any_normal_ntpm``.
    """
    try:
        df = get_data("tumor-up-vs-matched-normal").copy()
    except ValueError:
        return None
    if cancer_code:
        df = df[df["cancer_code"] == cancer_code].reset_index(drop=True)
    return df


def heme_tumor_up_vs_matched_normal(
    cancer_code: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Heme analogue of :func:`tumor_up_vs_matched_normal`.

    DLBC vs lymph_node (mature-B normal background), LAML vs
    bone_marrow (myeloid-progenitor normal background). Filter is
    looser than the solid panel because heme tumors *are* immune
    tissue — the malignant clone shares most expression with its
    normal lineage counterpart, so treat the top hits as one signal
    among several.
    """
    try:
        df = get_data("heme-tumor-up-vs-matched-normal").copy()
    except ValueError:
        return None
    if cancer_code:
        df = df[df["cancer_code"] == cancer_code].reset_index(drop=True)
    return df


# ---------- accessors: HPA cell-type + ESTIMATE signatures ----------


def hpa_cell_type_expression() -> pd.DataFrame:
    """Human Protein Atlas single-cell consensus cell-type expression.

    Long-form: one row per (Symbol, cell_type) with consensus nTPM
    aggregated across the public HPA single-cell datasets. Useful for
    interpreting which cell type drives a sample's signal.
    """
    return get_data("hpa-cell-type-expression").copy()


def estimate_signatures() -> pd.DataFrame:
    """ESTIMATE stromal + immune gene-set signatures.

    From Yoshihara et al. 2013 (PMID:24113773). Two signature lists
    (StromalSignature, ImmuneSignature) that score a sample's stromal
    and immune-infiltrate content from bulk RNA-seq.
    """
    return get_data("estimate-signatures").copy()


__all__ = [
    # accessors
    "pan_cancer_expression",
    "cancer_expression",
    "cancer_enriched_genes",
    "tcga_deconvolved_expression",
    "subtype_deconvolved_expression",
    "tumor_up_vs_matched_normal",
    "heme_tumor_up_vs_matched_normal",
    "hpa_cell_type_expression",
    "estimate_signatures",
    # normalization
    "normalize_to_housekeeping",
    "log2_transform",
    "filter_technical_rna",
    "filter_to_genes",
    "technical_rna_gene_ids",
]
