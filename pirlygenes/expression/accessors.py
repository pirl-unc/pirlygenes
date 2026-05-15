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
  with deterministic TPM companion columns derived from those FPKM
  columns.
* :func:`hpa_cell_type_expression` — HPA cell-type single-cell
  reference (long-form ``Symbol, cell_type, nTPM``).
* :func:`estimate_signatures` — the ESTIMATE stromal/immune signature
  gene sets (Yoshihara et al., 2013).

The normalization layer is intentionally narrow — anything that
needs per-sample QC narration (degradation index, FFPE rescue,
library-prep classification) lives in trufflepig. What's here:

* :func:`normalize_to_housekeeping` — divide each column by its
  housekeeping-gene median.
* :func:`log2_transform` — log2(x + 1) over value columns.
* :func:`filter_technical_rna` — drop mtDNA / NUMT-like / rRNA-like /
  nuclear-retained-lncRNA rows by ENSG, sourced from
  :mod:`pirlygenes.gene_families` (no symbol-regex dependency).
* :func:`filter_to_genes` — subset to a caller-provided gene list.

The accessors expose ``normalize=``, ``log_transform=``, and
``drop_technical_rna=`` keyword arguments that pipeline the free
functions in the expected order — for callers who prefer one call to a
chain of helpers.

Boundary: :func:`filter_technical_rna` and the family-level filter
inside :func:`normalize_expression` (see
:mod:`pirlygenes.expression.normalize`) catch overlapping but not
identical sets of genes. ``filter_technical_rna`` uses the curated
:mod:`pirlygenes.gene_families` ENSG tables exclusively. The
``normalize_expression`` path classifies via
:func:`pirlygenes.expression.qc.classify_gene_qc`, which prefers the
same curated tables but falls back to symbol regex for genes the
tables don't yet cover (newly annotated entries, deprecated IDs).
Prefer ``normalize_expression`` when you need both the zero-and-
renormalize behavior and the wider symbol-regex coverage; prefer
``filter_technical_rna`` when you only want a row-drop on the
strictly-curated set.

Returned frames are always ``.copy()``'d from the cached CSV; callers
can mutate freely.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..gene_families import gene_family_ids
from ..gene_sets_cancer import housekeeping_gene_ids
from ..load_dataset import get_data
from .normalize import (
    add_tpm_columns_from_fpkm,
    normalize_expression,
    percentile_rank_expression,
    renormalize_to_million,
)
from .qc import _TECHNICAL_RNA_FAMILIES


# ---------- column-discovery helpers ----------


_VALUE_COL_PREFIXES = ("nTPM_", "FPKM_", "TPM_")
_PAN_ANALYSIS_VALUE_COL_PREFIXES = ("nTPM_", "TPM_")
_PAN_RAW_ANALYSIS_VALUE_COL_PREFIXES = ("nTPM_raw_", "TPM_raw_")


def _default_value_cols(df: pd.DataFrame) -> list[str]:
    """Heuristic: wide-form expression frames use prefixed column names."""
    return [
        c for c in df.columns
        if c.startswith(_VALUE_COL_PREFIXES)
        and not c.startswith(_PAN_RAW_ANALYSIS_VALUE_COL_PREFIXES)
    ]


def _pan_analysis_value_cols(df: pd.DataFrame) -> list[str]:
    """TPM-scale columns used by pan-cancer normalization presets."""
    return [
        c for c in df.columns
        if c.startswith(_PAN_ANALYSIS_VALUE_COL_PREFIXES)
        and not c.startswith(_PAN_RAW_ANALYSIS_VALUE_COL_PREFIXES)
    ]


def _pan_raw_analysis_col_name(col: str) -> str:
    """Name for preserving pre-normalization TPM/nTPM analysis values."""
    for prefix in _PAN_ANALYSIS_VALUE_COL_PREFIXES:
        if col.startswith(prefix):
            return f"{prefix}raw_{col[len(prefix):]}"
    return f"{col}_raw"


def _add_raw_pan_analysis_value_cols(
    df: pd.DataFrame,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Copy TPM/nTPM analysis columns before a normalization overwrites them."""
    out = df.copy()
    for col in value_cols:
        out[_pan_raw_analysis_col_name(col)] = out[col]
    return out


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
        ``nTPM_``, ``FPKM_``, or ``TPM_``.

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
    in ``pirlygenes.expression.qc.classify_gene_qc`` generates those
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


def _renormalize_to_million_grouped(
    df: pd.DataFrame,
    *,
    value_cols: Sequence[str],
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Within each (group_cols) partition, rescale each value column so
    its non-NaN sum is 10⁶. The whole-table version in
    :func:`renormalize_to_million` rescales globally, which collapses
    long-form per-group medians into per-row crumbs — long-form callers
    want the TPM convention enforced per cohort, not across cohorts."""
    out = df.copy()
    for col in value_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for _key, idx in out.groupby(list(group_cols), dropna=False).groups.items():
        idx = list(idx)
        for col in value_cols:
            if col not in out.columns:
                continue
            col_sum = float(out.loc[idx, col].sum())
            if col_sum <= 0:
                continue
            out.loc[idx, col] = out.loc[idx, col] * (1e6 / col_sum)
    return out


def _bundled_normalize(
    df: pd.DataFrame,
    *,
    technical_rna_normalize: bool,
    remove_noncoding: bool,
    renormalize: bool,
    label_col: str = "Symbol",
    id_col: Optional[str] = "Ensembl_Gene_ID",
    value_cols: Optional[Sequence[str]] = None,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Bundled rescaling: zero technical-RNA (optionally noncoding) rows
    and renormalize each column's remaining mass, then optionally pin
    every column to a 10⁶ total.

    Matches the kwarg surface trufflepig's local reference accessors use
    so callers can pull these transforms from pirlygenes directly.
    """
    if technical_rna_normalize or remove_noncoding:
        df, _ = normalize_expression(
            df,
            label_col=label_col,
            id_col=id_col,
            value_cols=value_cols,
            group_cols=group_cols,
            remove_noncoding=remove_noncoding,
        )
    if renormalize:
        if group_cols and value_cols:
            df = _renormalize_to_million_grouped(
                df, value_cols=value_cols, group_cols=group_cols,
            )
        else:
            df, _ = renormalize_to_million(df, value_cols=value_cols)
    return df


_VALID_NORMALIZE_PAN = ("tpm", "hk", "housekeeping", "percentile", "clean_tpm")


def _canonical_pan_normalize(normalize: Optional[str]) -> Optional[str]:
    """Normalize legacy/public tokens onto the internal short names."""
    if normalize == "housekeeping":
        return "hk"
    return normalize


def _warn_legacy_normalize_kwargs(
    used: dict[str, bool],
    stacklevel: int = 3,
) -> None:
    """Single DeprecationWarning if any of the legacy normalize kwargs is
    set. The new preset is intentionally not described as exact legacy
    behavior because it operates on TPM-scale analysis columns."""
    truthy = sorted(name for name, value in used.items() if value)
    if not truthy:
        return
    names = ", ".join(truthy)
    warnings.warn(
        f"{names} is deprecated. Use normalize=\"clean_tpm\" for "
        "the new TPM-scaled, technical-RNA-cleaned view, or compose the "
        "normalization primitives normalize_expression()/"
        "renormalize_to_million() when you need exact legacy column "
        "names or semantics. "
        "Legacy kwargs continue to work but will be removed in a future "
        "5.x release.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _apply_pipeline(
    df: pd.DataFrame,
    *,
    drop_technical_rna: bool = False,
    genes: Optional[Iterable[str]] = None,
    log_transform: bool = False,
    percentile: bool = False,
    value_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Shared accessor-kwarg pipeline. Order matters: family filter →
    gene subset → optional percentile transform → log transform."""
    if drop_technical_rna:
        df = filter_technical_rna(df)
    if genes is not None:
        df = filter_to_genes(df, genes)
    if percentile:
        df, _ = percentile_rank_expression(df, value_cols=value_cols)
    if log_transform:
        df = log2_transform(df, value_cols=value_cols)
    return df


# ---------- accessors: pan-cancer expression ----------


def pan_cancer_expression(
    genes: Optional[Iterable[str]] = None,
    normalize: Optional[str] = "clean_tpm",
    log_transform: bool = False,
    technical_rna_normalize: bool = False,
    remove_noncoding: bool = False,
    renormalize_to_million: bool = False,
    drop_technical_rna: bool = False,
) -> pd.DataFrame:
    """Wide-form expression across HPA normal tissues + TCGA cancer types.

    50 normal tissues from HPA v23 consensus (``nTPM_<tissue>`` columns)
    plus 33 TCGA cancer types from HPA pathology + GDC/STAR reprocessing
    (``FPKM_<code>`` in native units). The accessor always appends
    deterministic ``TPM_<code>`` companion columns derived from the FPKM
    columns, preserving the raw FPKM columns for provenance.

    Parameters
    ----------
    genes
        Optional iterable of gene symbols or Ensembl IDs to subset to.
    normalize
        Named preset for unit scale and technical-RNA normalization:

        - ``"clean_tpm"`` (default) — preserve pre-clean values as
          ``nTPM_raw_<tissue>`` and ``TPM_raw_<code>`` columns, then zero
          mtDNA / NUMT / rRNA / MALAT1+NEAT1 rows
          across TPM-scale analysis columns and pin each column's sum
          back to 10⁶. This is the recommended view for analysis:
          every analysis column on the same scale, technical-RNA
          denominator drift removed. Raw ``FPKM_<code>`` columns remain
          unchanged.
        - ``None`` — add ``TPM_<code>`` companion columns and otherwise
          leave values unchanged. This is the raw/provenance view:
          raw TCGA ``FPKM_<code>`` values and HPA ``nTPM_<tissue>``
          values are preserved, with no artifact-gene cleanup, HK
          scaling, percentile-rank, or log transform.
        - ``"tpm"`` — explicit alias for the raw/provenance TPM-companion
          view.
        - ``"hk"`` or ``"housekeeping"`` — divide TPM-scale analysis columns
          (``nTPM_<tissue>``, ``TPM_<code>``) by their housekeeping-gene
          median.
        - ``"percentile"`` — within-column percentile rank (0–100),
          applied to TPM-scale analysis columns.
    log_transform
        Apply ``log2(x + 1)`` to value columns after any normalization.
    technical_rna_normalize
        Deprecated since 5.2.0. Zero mtDNA / NUMT / rRNA /
        nuclear-retained-lncRNA rows and renormalize each column's
        remaining mass back to the original per-column total.
    remove_noncoding
        Deprecated since 5.2.0. Additionally zero rows with noncoding
        biotypes (keeping protein-coding, Ig, and TCR biotypes) when a
        biotype column is present.
    renormalize_to_million
        Deprecated since 5.2.0. After any zero-and-renormalize step,
        rescale every column so its non-NaN sum is exactly 10⁶.
    drop_technical_rna
        Drop mtDNA / NUMT / rRNA / nuclear-retained-lncRNA rows entirely
        (uses :func:`filter_technical_rna`). Distinct from
        ``normalize="clean_tpm"``: this removes rows,
        ``"clean_tpm"`` zeroes them in place. See Boundary note in the
        module docstring.

    Returns
    -------
    pd.DataFrame
        Defensive copy — safe to mutate.
    """
    if normalize is not None and normalize not in _VALID_NORMALIZE_PAN:
        raise ValueError(
            "normalize must be None or one of "
            f"{_VALID_NORMALIZE_PAN!r}, got {normalize!r}"
        )
    normalize = _canonical_pan_normalize(normalize)
    legacy_kwargs_used = any(
        (technical_rna_normalize, remove_noncoding, renormalize_to_million)
    )
    _warn_legacy_normalize_kwargs(
        {
            "technical_rna_normalize": technical_rna_normalize,
            "remove_noncoding": remove_noncoding,
            "renormalize_to_million": renormalize_to_million,
        },
    )

    df = get_data("pan-cancer-expression")
    df, _ = add_tpm_columns_from_fpkm(df)
    analysis_value_cols = _pan_analysis_value_cols(df)
    if normalize == "clean_tpm" and not legacy_kwargs_used:
        df = _add_raw_pan_analysis_value_cols(df, analysis_value_cols)

    do_tech_norm = technical_rna_normalize or (
        normalize == "clean_tpm" and not legacy_kwargs_used
    )
    do_renorm = renormalize_to_million or (
        normalize == "clean_tpm" and not legacy_kwargs_used
    )
    normalize_value_cols = (
        None
        if legacy_kwargs_used
        else analysis_value_cols
    )
    df = _bundled_normalize(
        df,
        technical_rna_normalize=do_tech_norm,
        remove_noncoding=remove_noncoding,
        renormalize=do_renorm,
        value_cols=normalize_value_cols,
    )
    if normalize == "hk":
        df = normalize_to_housekeeping(df, value_cols=analysis_value_cols)
    return _apply_pipeline(
        df,
        drop_technical_rna=drop_technical_rna,
        genes=genes,
        log_transform=log_transform,
        percentile=normalize == "percentile",
        value_cols=analysis_value_cols,
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
        normalize="hk",
        drop_technical_rna=True,
    )
    col = f"TPM_{code}"
    if col not in df.columns:
        raise ValueError(
            f"no TPM column for {cancer_type!r} (resolved to {code!r})"
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
        normalize="hk",
        drop_technical_rna=True,
    )
    tpm_cols = [c for c in df.columns if c.startswith("TPM_")]
    target_col = f"TPM_{code}"
    if target_col not in df.columns:
        raise ValueError(
            f"no TPM column for {cancer_type!r} (resolved to {code!r})"
        )
    other_cols = [c for c in tpm_cols if c != target_col]
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
    "hpa_cell_type_expression",
    "estimate_signatures",
    # normalization
    "normalize_to_housekeeping",
    "log2_transform",
    "filter_technical_rna",
    "filter_to_genes",
    "technical_rna_gene_ids",
]
