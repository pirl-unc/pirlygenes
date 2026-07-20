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
  panel: 50 HPA normal tissues (nTPM), 33 TCGA cancer types (observed FPKM
  provenance + deterministic TPM companions), and five TPM-only computed
  tumor rollups, with optional added normalized analysis columns.
* :func:`cancer_reference_expression` — long- or wide-form non-TCGA
  tumor reference summaries (CLL-map, MMRF, TARGET, GEO, etc.) exposed
  on a common TPM / clean-TPM contract for downstream consumers.
* :func:`tumor_up_vs_matched_normal` and
  :func:`heme_tumor_up_vs_matched_normal` — compact marker panels for
  cancer-vs-matched-normal comparisons.
* :func:`hpa_cell_type_expression` — HPA cell-type single-cell
  reference (long-form ``Symbol, cell_type, nTPM``).
* :func:`estimate_signatures` — the ESTIMATE stromal/immune signature
  gene sets (Yoshihara et al., 2013).

The normalization layer is intentionally narrow — anything that
needs per-sample QC narration (degradation index, FFPE rescue,
library-prep classification) lives in trufflepig. What's here:

* :func:`normalize_to_housekeeping` — divide each column by its
  median-of-ratios housekeeping size factor.
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

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..gene_families import gene_family_ids
from ..gene_ids import strip_version
from ..gene_names import get_alias_as_list, get_reverse_alias_as_list
from ..load_dataset import get_data
from ..version import DATA_VERSION
from .normalize import (
    add_tpm_columns_from_fpkm,
    drop_technical_genes,
    normalize_expression,
    percentile_rank_expression,
    renormalize_to_million,
    tpm_to_housekeeping_normalized,
)
from .qc import TECHNICAL_RNA_FAMILIES


# ---------- column-discovery helpers ----------


_VALUE_COL_PREFIXES = ("nTPM_", "FPKM_", "TPM_")
_PAN_ANALYSIS_VALUE_COL_PREFIXES = ("nTPM_", "TPM_")
_PAN_NORMALIZED_SUFFIXES = {
    "tpm_clean": "clean",
    "tpm_log1p": "log1p",
    "tpm_clean_log1p": "clean_log1p",
    "hk": "hk",
    "percentile": "percentile",
}
_PAN_NORMALIZE_DEPENDENCIES = {
    "tpm_clean": ("tpm",),
    "tpm_log1p": ("tpm",),
    "tpm_clean_log1p": ("tpm_clean",),
    "hk": ("tpm",),
    "percentile": ("tpm",),
}
_PAN_NORMALIZED_VALUE_COL_PREFIXES = tuple(
    f"{prefix}{suffix}_"
    for prefix in _PAN_ANALYSIS_VALUE_COL_PREFIXES
    for suffix in _PAN_NORMALIZED_SUFFIXES.values()
)
_VALUE_COL_SUFFIXES = (
    "_nTPM",
    "_FPKM",
    "_TPM",
    "_nTPM_log1p",
    "_TPM_log1p",
    "_nTPM_clean",
    "_TPM_clean",
    "_nTPM_clean_log1p",
    "_TPM_clean_log1p",
    "_nTPM_hk",
    "_TPM_hk",
    "_nTPM_percentile",
    "_TPM_percentile",
)


def _default_value_cols(df: pd.DataFrame) -> list[str]:
    """Heuristic: wide-form expression frames use prefixed column names."""
    return [
        c for c in df.columns
        if (
            c.startswith(_VALUE_COL_PREFIXES)
            or c.endswith(_VALUE_COL_SUFFIXES)
        )
    ]


def _pan_analysis_value_cols(df: pd.DataFrame) -> list[str]:
    """TPM-scale columns used by pan-cancer normalization presets."""
    return [
        c for c in df.columns
        if c.startswith(_PAN_ANALYSIS_VALUE_COL_PREFIXES)
        and not c.startswith(_PAN_NORMALIZED_VALUE_COL_PREFIXES)
    ]


def _pan_normalized_col_name(col: str, normalize: str) -> str:
    """Internal name for an added normalized TPM/nTPM analysis column."""
    if normalize == "tpm_clean_log1p":
        for prefix in ("nTPM_clean_", "TPM_clean_"):
            if col.startswith(prefix):
                return f"{prefix}log1p_{col[len(prefix):]}"
    suffix = _PAN_NORMALIZED_SUFFIXES[normalize]
    for prefix in _PAN_ANALYSIS_VALUE_COL_PREFIXES:
        if col.startswith(prefix):
            return f"{prefix}{suffix}_{col[len(prefix):]}"
    return f"{col}_{suffix}"


def _add_pan_normalized_value_cols(
    df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    value_cols: Sequence[str],
    normalize: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Add normalized TPM/nTPM analysis columns without overwriting inputs.

    Every derived column preserves its source column's availability mask.  In
    particular, ``normalize_expression(..., censored_fill="fixed_fraction")``
    fills missing inputs with zero internally; restoring the mask here keeps an
    unavailable rollup value distinct from a measured biological zero.
    """
    out = df.copy()
    target_cols = []
    for col in value_cols:
        target = _pan_normalized_col_name(col, normalize)
        out[target] = normalized_df[col].mask(df[col].isna())
        target_cols.append(target)
    return out, target_cols


def _pan_public_col_name(col: str) -> str:
    """Map internal unit-prefix columns to public entity-suffix columns."""
    prefix_to_suffix = (
        ("nTPM_clean_log1p_", "_nTPM_clean_log1p"),
        ("TPM_clean_log1p_", "_TPM_clean_log1p"),
        ("nTPM_log1p_", "_nTPM_log1p"),
        ("TPM_log1p_", "_TPM_log1p"),
        ("nTPM_percentile_", "_nTPM_percentile"),
        ("TPM_percentile_", "_TPM_percentile"),
        ("nTPM_clean_", "_nTPM_clean"),
        ("TPM_clean_", "_TPM_clean"),
        ("nTPM_hk_", "_nTPM_hk"),
        ("TPM_hk_", "_TPM_hk"),
        ("nTPM_", "_nTPM"),
        ("FPKM_", "_FPKM"),
        ("TPM_", "_TPM"),
    )
    for prefix, suffix in prefix_to_suffix:
        if col.startswith(prefix):
            return f"{col[len(prefix):]}{suffix}"
    return col


def _rename_pan_expression_columns_entity_first(df: pd.DataFrame) -> pd.DataFrame:
    """Return the public pan-cancer column schema.

    The packaged CSV and internal normalization pipeline use unit-prefix
    names. The accessor returns entity-first names for readability.
    """
    return df.rename(columns={c: _pan_public_col_name(c) for c in df.columns})


_PAN_COMPUTED_ROLLUP_MEMBERS = {
    "BTC": ("CHOL",),
    "CRC": ("COAD", "READ"),
    "NET": ("NET_PANCREAS", "NET_MIDGUT", "NET_RECTAL", "NET_LUNG"),
    "NSCLC": ("LUAD", "LUSC"),
    "SGC": ("ADCC",),
}
_PAN_ROLLUP_MEMBER_CODES = tuple(dict.fromkeys(
    member
    for members in _PAN_COMPUTED_ROLLUP_MEMBERS.values()
    for member in members
))


def _oncoref_canonicalize_gene_rows(
    df: pd.DataFrame,
    *,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """Collapse a wide linear-expression frame onto oncoref's gene-id space.

    The compatibility adapter deliberately uses oncoref's public alias resolver,
    rather than pirlygenes' proteoform-oriented sequence-identity map.  Alias and
    retired rows are summed with ``min_count=1`` before any normalization, which
    is the same rule used by oncoref's expression accessors and preserves an
    all-missing source cell as missing.
    """
    from oncoref.gene_ids import resolve_ensembl_id, unversioned

    canonical = df["Ensembl_Gene_ID"].astype(str).map(resolve_ensembl_id)
    if not canonical.duplicated().any():
        return df.assign(Ensembl_Gene_ID=canonical.to_numpy())

    original = df["Ensembl_Gene_ID"].astype(str).map(unversioned)
    is_primary = original.to_numpy() == canonical.to_numpy()
    work = df.assign(
        Ensembl_Gene_ID=canonical.to_numpy(),
        _primary=is_primary,
    )
    work = work.sort_values("_primary", ascending=False, kind="stable").drop(
        columns="_primary"
    )
    sum_cols = [col for col in value_cols if col in work.columns]
    keep_cols = [
        col
        for col in work.columns
        if col != "Ensembl_Gene_ID" and col not in sum_cols
    ]
    grouped = work.groupby("Ensembl_Gene_ID", sort=False)
    parts = []
    if keep_cols:
        parts.append(grouped[keep_cols].first())
    if sum_cols:
        parts.append(grouped[sum_cols].sum(min_count=1))
    out = pd.concat(parts, axis=1).reset_index()
    return out[list(df.columns)]


@lru_cache(maxsize=1)
def _load_pan_rollup_frame() -> pd.DataFrame:
    """Read the small persisted pan-cancer rollup artifact.

    The artifact is baked from oncoref's selected source for each member cohort
    (see ``scripts/generate_pan_cancer_expression_rollups.py``), canonicalized
    before pooling, and shipped in the wheel. This avoids both an eager scan of
    the multi-million-row reference summary and the gene-wise source fallback
    that a generic all-source cohort pivot would introduce.
    """
    rollups = get_data("pan-cancer-expression-rollups", copy=False)
    value_cols = [f"TPM_{code}" for code in _PAN_COMPUTED_ROLLUP_MEMBERS]
    missing = [
        col
        for col in ["Ensembl_Gene_ID", *value_cols]
        if col not in rollups.columns
    ]
    if missing:
        raise ValueError(
            "pan-cancer-expression-rollups has an invalid schema; missing "
            f"{missing!r}"
        )
    return _oncoref_canonicalize_gene_rows(
        rollups,
        value_cols=value_cols,
    )


def _pan_computed_rollup_frame() -> pd.DataFrame:
    """Return canonical ENSG + five persisted raw-TPM rollups."""
    return _load_pan_rollup_frame()


@lru_cache(maxsize=1)
def _pan_reference_frame() -> pd.DataFrame:
    """Canonical raw pan-cancer matrix via the oncoref compatibility adapter.

    The version-pinned pirlygenes pan matrix remains the fast persisted source.
    We canonicalize it with oncoref's delegated alias map, sum duplicate linear
    loci, derive deterministic TPM companions, and join five persisted rollups
    baked from oncoref's selected sources. This produces oncoref's canonical
    semantics without its eager multi-million-row summary scan or a runtime
    dependency on oncoref's separate expression data bundle.
    """
    raw = get_data("pan-cancer-expression", copy=False)
    id_cols = {"Ensembl_Gene_ID", "Symbol"}
    value_cols = [col for col in raw.columns if col not in id_cols]
    raw = _oncoref_canonicalize_gene_rows(raw, value_cols=value_cols)
    raw, _ = add_tpm_columns_from_fpkm(raw)
    return raw.merge(
        _pan_computed_rollup_frame(),
        on="Ensembl_Gene_ID",
        how="left",
        sort=False,
    )


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
    """Rescale each value column by its housekeeping size factor.

    The size factor is the median-of-ratios denominator: for each column,
    the median of ``sample_tpm[g] / reference_tpm[g]`` over the housekeeping
    genes. Dividing by it puts the sample on the reference-profile scale
    (a housekeeping gene reads back ~its reference TPM), not a unit baseline.
    Works across TPM, FPKM, and nTPM units since the normalization is
    per-column. This compatibility helper delegates to
    :func:`pirlygenes.expression.normalize.tpm_to_housekeeping_normalized`,
    so the HK path uses the same ENSG-first HPA-derived panel and
    median-of-ratios denominator everywhere.

    Parameters
    ----------
    df
        Expression frame with an ``Ensembl_Gene_ID`` column and one or
        more numeric value columns.
    value_cols
        Columns to rescale. If ``None``, picks columns using either the
        internal unit-prefix schema (``nTPM_``, ``FPKM_``, ``TPM_``) or
        the public entity-suffix schema (``*_nTPM``, ``*_FPKM``,
        ``*_TPM`` and added normalized suffixes).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the named columns rescaled in place. Requested
        columns that cannot be put on the housekeeping ratio scale are blanked
        to ``NaN`` rather than returned on the raw input scale.
    """
    id_col = _resolve_id_col(df)
    if id_col is None:
        raise ValueError(
            "normalize_to_housekeeping needs an Ensembl_Gene_ID column"
        )
    cols = list(value_cols) if value_cols is not None else _default_value_cols(df)
    out, record = tpm_to_housekeeping_normalized(df, id_col=id_col, value_cols=cols)
    columns = record.get("columns", {}) if isinstance(record, dict) else {}
    applied = bool(record.get("applied")) if isinstance(record, dict) else False
    failed_cols: list[str] = []
    for col in cols:
        if col not in out.columns:
            continue
        col_record = columns.get(col)
        if col_record is None:
            if not applied:
                failed_cols.append(col)
            continue
        try:
            denominator = float(col_record.get("denominator", 0.0))
        except (TypeError, ValueError):
            denominator = 0.0
        if col_record.get("applied") is False or denominator <= 0:
            failed_cols.append(col)
    if failed_cols:
        out = out.copy()
        out[failed_cols] = np.nan
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


def log1p_transform(
    df: pd.DataFrame,
    value_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Apply natural ``log1p(x)`` to expression columns."""
    cols = list(value_cols) if value_cols is not None else _default_value_cols(df)
    out = df.copy()
    for col in cols:
        out[col] = np.log1p(out[col].astype(float))
    return out


def technical_rna_gene_ids() -> set[str]:
    """Union of ENSG IDs across the technical-RNA families.

    Drives :func:`filter_technical_rna`. Exposed in case callers want
    to project onto a frame that doesn't carry ``Ensembl_Gene_ID``.
    """
    out: set[str] = set()
    for family in TECHNICAL_RNA_FAMILIES:
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
    and the Ensembl-ID column. Ensembl version suffixes are ignored, so
    ``ENSG00000146648`` and ``ENSG00000146648.17`` are equivalent.
    """
    if isinstance(genes, str):
        genes = [genes]
    targets = set()
    for gene in genes:
        name = str(gene).strip()
        targets.add(name.upper())
        targets.update(alias.upper() for alias in get_alias_as_list(name))
        targets.update(alias.upper() for alias in get_reverse_alias_as_list(name))
    ensembl_targets = {
        strip_version(target).upper()
        for target in targets
        if target.upper().startswith("ENSG")
    }
    if ensembl_targets:
        # The pan-cancer frame is keyed in oncoref's canonical ENSG space.  Keep
        # raw targets for generic/non-canonical frames, but add the delegated
        # alias-map targets so legacy and retired IDs still hit canonical rows.
        from oncoref.gene_ids import resolve_ensembl_id

        ensembl_targets.update(
            resolve_ensembl_id(target).upper()
            for target in tuple(ensembl_targets)
        )
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
        ids = df[id_col].astype(str).str.upper()
        mask |= ids.isin(targets)
        if ensembl_targets:
            mask |= ids.map(strip_version).isin(ensembl_targets)
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


_VALID_NORMALIZE_PAN = (
    "tpm",
    "tpm_clean",
    "tpm_log1p",
    "tpm_clean_log1p",
    "hk",
    "housekeeping",
    "percentile",
)
_VALID_NORMALIZE_PAN_DISPLAY = _VALID_NORMALIZE_PAN


def _canonical_pan_normalize_token(token: str) -> str:
    """Normalize public tokens onto the internal short names."""
    token = token.lower()
    if token == "housekeeping":
        return "hk"
    return token


def _resolve_pan_normalize_modes(
    normalize: Optional[str | Sequence[str]],
) -> list[str]:
    """Canonicalize ``normalize=`` into an ordered, dependency-expanded list."""
    if normalize is None:
        requested: list[str] = []
    elif isinstance(normalize, str):
        requested = [normalize]
    else:
        requested = list(normalize)

    canonical_requested: list[str] = []
    for token in requested:
        if not isinstance(token, str):
            raise ValueError(
                "normalize must be None, a string, or a sequence of strings; "
                f"got element {token!r}"
            )
        canonical = _canonical_pan_normalize_token(token)
        if canonical not in {
            "tpm",
            "tpm_clean",
            "tpm_log1p",
            "tpm_clean_log1p",
            "hk",
            "percentile",
        }:
            raise ValueError(
                "normalize must be None, a string, or a sequence containing "
                f"{_VALID_NORMALIZE_PAN_DISPLAY!r}; got {token!r}"
            )
        canonical_requested.append(canonical)

    out: list[str] = []

    def add_with_deps(mode: str) -> None:
        for dep in _PAN_NORMALIZE_DEPENDENCIES.get(mode, ()):
            add_with_deps(dep)
        if mode not in out:
            out.append(mode)

    for mode in canonical_requested:
        add_with_deps(mode)
    return out


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


# ---------- accessors: source-agnostic tumor references ----------


_REFERENCE_NORMALIZE_ALIASES = {
    "tpm": "tpm",
    "TPM": "tpm",
    "tpm_clean": "tpm_clean",
    "clean_tpm": "tpm_clean",
    "tpm_log1p": "tpm_log1p",
    "tpm_clean_log1p": "tpm_clean_log1p",
    "clean_tpm_log1p": "tpm_clean_log1p",
}
# oncoref currently normalizes explicitly empty filters to an empty set and
# then treats that set as though no filter was supplied.  Forward a guaranteed
# nonmatching cohort instead so pirlygenes retains its historical distinction
# between ``None`` (all sources) and ``[]`` (no sources).
_EMPTY_REFERENCE_SOURCE_COHORT = "__pirlygenes_explicit_empty_source_cohort__"
_REFERENCE_VALUE_COLUMNS = {
    "tpm": ("TPM_median", "TPM_q1", "TPM_q3", "TPM"),
    "tpm_clean": (
        "TPM_clean_median",
        "TPM_clean_q1",
        "TPM_clean_q3",
        "TPM_clean",
    ),
    "tpm_log1p": ("TPM_median", "TPM_q1", "TPM_q3", "TPM_log1p"),
    "tpm_clean_log1p": (
        "TPM_clean_median",
        "TPM_clean_q1",
        "TPM_clean_q3",
        "TPM_clean_log1p",
    ),
}


def _resolve_reference_normalize_modes(
    normalize: str | Sequence[str],
) -> list[str]:
    if isinstance(normalize, str):
        requested = [normalize]
    else:
        requested = list(normalize)
    out: list[str] = []
    for token in requested:
        if not isinstance(token, str):
            raise ValueError(
                "normalize must be a string or a sequence of strings; "
                f"got element {token!r}"
            )
        canonical = _REFERENCE_NORMALIZE_ALIASES.get(token)
        if canonical is None:
            canonical = _REFERENCE_NORMALIZE_ALIASES.get(token.lower())
        if canonical is None:
            raise ValueError(
                "normalize must contain one of "
                f"{tuple(_REFERENCE_NORMALIZE_ALIASES)!r}; got {token!r}"
            )
        if canonical not in out:
            out.append(canonical)
    return out


def _validate_reference_format(format: str) -> None:
    if format not in {"long", "wide"}:
        raise ValueError("format must be 'long' or 'wide'")


def _resolve_cancer_types(
    cancer_types: Optional[str | Iterable[str]],
    *,
    expand_aggregates: bool = False,
) -> list[str] | None:
    if cancer_types is None:
        return None
    from ..gene_sets_cancer import resolve_cancer_type

    if isinstance(cancer_types, str):
        requested = [cancer_types]
    else:
        requested = list(cancer_types)
    if not expand_aggregates:
        return [resolve_cancer_type(code) for code in requested]

    # Union view: a computed-aggregate code (the pan-sarcoma ``SARC`` grand
    # union, or the ``SARC_RMS`` / ``SARC_LPS`` histology rollups) expands to
    # the union of its member subtype codes. ``SARC_RMS`` / ``SARC_LPS`` are
    # aggregate-only (not registry codes), so the raw token is checked before
    # resolving; ``SARC`` resolves to itself and is also an aggregate. No
    # fabricated pooled stats — literature-curated members with no built shard
    # simply contribute no rows.
    from ..gene_sets_cancer import cohort_aggregates

    aggregates = cohort_aggregates()
    out: list[str] = []
    for code in requested:
        members = aggregates.get(str(code))
        if members is None:
            resolved = resolve_cancer_type(code)
            members = aggregates.get(resolved)
            if members is None:
                out.append(resolved)
                continue
        out.extend(members)
    return list(dict.fromkeys(out))


def _load_cancer_reference_expression() -> pd.DataFrame:
    # Read-only shared oncoref-owned view. All callers (_has_cancer_reference,
    # cancer_reference_summary, cancer_reference_expression) filter to a
    # cancer_code / gene slice and .copy() that subset before returning or
    # mutating, so the full-frame defensive copy is pure waste — and for this
    # multi-million-row table it dominated test-suite wall time (#278/#557).
    return get_data("cancer-reference-expression", copy=False)


# Identity-keyed memo of read-only views derived purely from the (shared,
# process-wide) reference frame. The frame is a singleton — get_data(copy=False)
# returns the same object every call — so any expression-value view computed
# from it is stable until the data reloads. The gene-independent availability
# manifest deliberately does not use this cache (#565).
_REFERENCE_VIEW_CACHE: dict[str, tuple] = {}


def _string_id_columns(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Cast stable identifier columns to pandas string dtype in place."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _reference_view(key: str, builder):
    """Return ``builder(reference_frame)``, memoized on the frame's identity."""
    df = _load_cancer_reference_expression()
    cached = _REFERENCE_VIEW_CACHE.get(key)
    if cached is not None and cached[0] is df:
        return cached[1]
    value = builder(df)
    _REFERENCE_VIEW_CACHE[key] = (df, value)
    return value


def _reference_code_set() -> frozenset:
    """Cached ``{cancer_code}`` set over the packaged reference frame."""
    return _reference_view(
        "reference_code_set",
        lambda df: frozenset(df["cancer_code"].astype(str)),
    )


@lru_cache(maxsize=1)
def _oncoref_summary_reference_code_set() -> frozenset:
    """Reference codes served by oncoref's all-source summary view."""
    import oncoref

    availability = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="all",
        reference_source="summary_rows_all",
        all_sources=True,
    )
    return frozenset(
        availability.loc[availability["available"], "cancer_code"].astype(str)
    )


@lru_cache(maxsize=1)
def _oncoref_artifact_reference_code_set() -> frozenset:
    """Reference codes served by oncoref's canonical percentile artifacts."""
    import oncoref

    availability = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="artifact",
        reference_source="artifact",
    )
    return frozenset(
        availability.loc[availability["available"], "cancer_code"].astype(str)
    )


@lru_cache(maxsize=1)
def _oncoref_reference_code_set() -> frozenset:
    """All codes loadable through the delegated compatibility accessor."""
    return frozenset(
        _oncoref_summary_reference_code_set()
        | _oncoref_artifact_reference_code_set()
    )


def _reference_indices_by_code() -> dict:
    """Cached ``{cancer_code: positional-row-index array}`` over the reference
    frame, so per-code slicing avoids a full-frame ``astype(str).isin`` scan."""
    return _reference_view(
        "indices_by_code",
        lambda df: {
            str(code): idx
            for code, idx in df.groupby(
                df["cancer_code"].astype(str), sort=False
            ).indices.items()
        },
    )


def _reference_slice_by_codes(
    df: pd.DataFrame,
    codes: Sequence[str],
) -> pd.DataFrame:
    """Slice the shared summary through its cached positional cohort index.

    Tests and offline callers may pass an independent fixture frame; retain the
    ordinary boolean filter for those objects. Runtime callers pass the shared
    oncoref-owned singleton, where rebuilding a multi-million-row string mask on
    every noncanonical cohort-view request is avoidable.
    """
    if df is not _load_cancer_reference_expression():
        return df[df["cancer_code"].astype(str).isin(codes)]

    index = _reference_indices_by_code()
    positions = [index[str(code)] for code in codes if str(code) in index]
    if not positions:
        return df.iloc[0:0]
    # The previous boolean mask preserved artifact row order. Keep that contract
    # even when aggregate expansion supplies codes in a different order.
    return df.iloc[np.sort(np.concatenate(positions))]


def _has_cancer_reference(code: str) -> bool:
    return code in _oncoref_reference_code_set()


def _load_cancer_expression_source_candidates() -> pd.DataFrame:
    df = get_data("cancer-expression-source-candidates")
    string_cols = [c for c in df.columns if c != "estimated_samples"]
    df[string_cols] = df[string_cols].fillna("")
    return _string_id_columns(df, "cancer_code")


def _pan_expression_codes() -> set[str]:
    df = get_data("pan-cancer-expression", copy=False)  # read-only: columns only (#278)
    return {
        str(col).removeprefix("FPKM_")
        for col in df.columns
        if str(col).startswith("FPKM_")
    }


def _registry_parent_codes(value) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [
        part.strip()
        for part in text.replace(";", ",").split(",")
        if part.strip()
    ]


def _reference_cohort_summary(code: str) -> dict[str, object]:
    refs = available_cancer_expression_references()
    summaries = _reference_cohort_summaries(refs, _pan_expression_codes())
    return summaries.get(code, {
        "source_project": "",
        "source_cohort": "",
        "n_samples": np.nan,
        "processing_pipeline": "",
    })


def _reference_cohort_summaries(
    refs: pd.DataFrame,
    pan_codes: set[str],
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for _, first in refs.drop_duplicates(subset=["cancer_code"]).iterrows():
        code = str(first.get("cancer_code", ""))
        summaries[code] = {
            "source_project": first.get("source_project", ""),
            "source_cohort": first.get("source_cohort", ""),
            "n_samples": first.get("n_samples", np.nan),
            "processing_pipeline": first.get("processing_pipeline", ""),
        }
    for code in pan_codes - set(summaries):
        summaries[code] = {
            "source_project": "TCGA/HPA",
            "source_cohort": "TCGA_XENA_TOIL",
            "n_samples": np.nan,
            "processing_pipeline": "pan_cancer_expression_tpm_clean",
        }
    return summaries


def _resolve_expression_reference_code_from_lookups(
    code: str,
    *,
    registry: pd.DataFrame,
    reference_codes: set[str],
    pan_codes: set[str],
) -> str | None:
    """Return the packaged direct or parent expression reference for a code."""

    def visit(current: str, path: set[str]) -> str | None:
        if current in path:
            return None
        if current in reference_codes or current in pan_codes:
            return current
        if current not in registry.index:
            return None
        path.add(current)
        for parent in _registry_parent_codes(registry.loc[current, "parent_code"]):
            resolved = visit(parent, path)
            if resolved is not None:
                return resolved
        return None

    return visit(code, set())


def _resolve_expression_reference_code(code: str) -> str | None:
    """Return the packaged direct or parent expression reference for a code."""
    from ..gene_sets_cancer import cancer_type_registry

    registry = cancer_type_registry().set_index("code")
    reference_codes = _oncoref_reference_code_set()
    pan_codes = _pan_expression_codes()
    return _resolve_expression_reference_code_from_lookups(
        code,
        registry=registry,
        reference_codes=reference_codes,
        pan_codes=pan_codes,
    )


def _reference_expr_value(
    df: pd.DataFrame,
    mode: str,
) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    median_col, q1_col, q3_col, label = _REFERENCE_VALUE_COLUMNS[mode]
    expr = pd.to_numeric(df[median_col], errors="coerce")
    q1 = pd.to_numeric(df[q1_col], errors="coerce")
    q3 = pd.to_numeric(df[q3_col], errors="coerce")
    if mode.endswith("_log1p"):
        expr = np.log1p(expr)
        q1 = np.log1p(q1)
        q3 = np.log1p(q3)
    return expr, q1, q3, label


def _reference_long_from_summary_frame(
    df: pd.DataFrame,
    *,
    cancer_types: Optional[str | Iterable[str]] = None,
    genes: Optional[Iterable[str]] = None,
    modes: Sequence[str] = ("tpm", "tpm_clean"),
) -> pd.DataFrame:
    """Project an oncoref-owned summary frame into pirlygenes long form.

    Canonical cohort views consume the already delegated raw summary frame
    supplied by :func:`_load_cancer_reference_expression`.  Keeping this small
    deterministic projection separate from the public oncoref call avoids a
    second multi-million-row load and lets tests inject a summary fixture while
    retaining a single empirical owner.
    """
    source = df
    codes = _resolve_cancer_types(cancer_types, expand_aggregates=True)
    if codes is not None:
        source = _reference_slice_by_codes(source, codes)
    if genes is not None:
        requested = [genes] if isinstance(genes, str) else list(genes)
        source = filter_to_genes(
            source,
            _reference_compatibility_genes(requested) or [],
        )

    id_columns = ["Ensembl_Gene_ID", "Symbol", "cancer_code", "source_cohort"]
    provenance_columns = [
        "source_project",
        "source_version",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
        "tumor_origin",
        "metastasis_site",
    ]
    keep = [
        column
        for column in id_columns + provenance_columns
        if column in source.columns
    ]
    parts: list[pd.DataFrame] = []
    for mode in modes:
        expression, q1, q3, label = _reference_expr_value(source, mode)
        part = source[keep].copy()
        part["normalization"] = label
        part["expression"] = expression
        part["q1"] = q1
        part["q3"] = q3
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=keep + ["normalization", "expression", "q1", "q3"]
        )
    return pd.concat(parts, ignore_index=True)


def available_cancer_expression_references() -> pd.DataFrame:
    """Packaged non-TCGA tumor reference cohorts available by cancer code.

    Returns one row per ``(cancer_code, source_cohort)`` with sample-count,
    processing provenance, and primary-vs-metastasis annotation. Within
    each cancer_code, rows are ordered with ``tumor_origin == 'primary'``
    first so consumers that take ``.iloc[0]`` get the canonical reference
    cohort. Downstream consumers can use this to decide which non-TCGA
    references are available without inspecting data files.

    This is a gene-independent manifest read. It combines oncoref's compact
    all-source summary availability with cohorts served only by its canonical
    percentile artifacts, then adapts that union through pirlygenes'
    compatibility registry. It never loads the multi-million-row expression
    summary or any expression values. A fresh ``.copy()`` keeps the historical
    mutation-safe contract.
    """
    return _load_available_reference_manifest().copy()


@lru_cache(maxsize=1)
def _load_available_reference_manifest() -> pd.DataFrame:
    import oncoref

    from ..gene_sets_cancer import cohort_registry_df
    from .reference_manifest import build_reference_manifest
    from .source_cohort_origin import classify_source_cohort

    summary = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="all",
        reference_source="summary_rows_all",
        all_sources=True,
    )
    artifacts = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="artifact",
        reference_source="artifact",
    )
    summary_codes = set(
        summary.loc[summary["available"], "cancer_code"].astype(str)
    )
    artifact_only = artifacts.loc[
        artifacts["available"]
        & ~artifacts["cancer_code"].astype(str).isin(summary_codes)
    ]
    availability = pd.concat([summary, artifact_only], ignore_index=True)
    manifest = build_reference_manifest(
        availability,
        availability,
        cohort_registry_df(),
        classify_source_cohort,
    )
    return _build_available_references(manifest)


def _build_available_references(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "n_samples",
        "processing_pipeline",
        "tumor_origin",
        "metastasis_site",
    ]
    present = [c for c in keep if c in df.columns]
    out = df[present].drop_duplicates()
    # Sort so primary > mixed > metastasis > everything else within each
    # cancer_code; ties broken by source_cohort. Map through object so a
    # categorical origin column never needs a synthetic category or magic-value
    # fill just to express the ordering.
    origin_priority = {"primary": 0, "mixed": 1, "metastasis": 2}
    if "tumor_origin" in out.columns:
        unknown_origin_rank = len(origin_priority)
        out = out.assign(
            _origin_rank=(
                out["tumor_origin"]
                .astype(object)
                .map(lambda value: origin_priority.get(value, unknown_origin_rank))
            ),
        )
        out = out.sort_values(
            ["cancer_code", "_origin_rank", "source_cohort"],
        ).drop(columns="_origin_rank")
    else:
        out = out.sort_values(["cancer_code", "source_cohort"])
    return _string_id_columns(out.reset_index(drop=True), "cancer_code")


def source_prefixed_references() -> pd.DataFrame:
    """Source-prefixed cohort atoms (#292): the per-``(cancer_code,
    source_cohort)`` manifest annotated with ``kind`` (pipeline family, from the
    cohort registry) and an addressable ``cohort_atom = "<kind>:<cancer_code>"``
    (e.g. ``treehouse:NET_PANCREAS``, ``geo:SARC_DDLPS``, ``beataml:LAML_ELNfav``).

    A cancer category is the **union of its source-prefixed atoms across kinds**;
    :func:`available_cancer_expression_references` gives the per-(code, cohort)
    rows and this adds the addressable atom + kind so a consumer can keep the
    sources separate and slot a new pipeline (e.g. GDC) in as a parallel ``kind``.

    **Cross-source combining rule** — absolute TPMs are NOT comparable across
    pipelines (per-gene offsets ~0.3–4.4×, r≈0.99 only on rank): pool absolute
    clean-TPM **within** one kind/cohort; combine **across** kinds in rank /
    z-space (or a per-gene calibration), never average raw TPM across kinds.
    """
    from ..gene_sets_cancer import cohort_registry_df

    refs = available_cancer_expression_references()
    reg = cohort_registry_df()
    kind = dict(zip(reg["cohort_id"].astype(str), reg["kind"].astype(str)))
    out = refs.copy()
    out["kind"] = out["source_cohort"].astype(str).map(kind).fillna("other")
    out["cohort_atom"] = out["kind"] + ":" + out["cancer_code"].astype(str)
    return out


def cancer_code_sources(cancer_code: Optional[str] = None) -> dict:
    """Cross-source rollup (#292): ``{cancer_code: {kind: [cohort_id, ...]}}`` —
    which source kinds/cohorts back each cancer code. Pass a code (alias/synonym
    accepted) for just that one; ``None`` returns the whole map. This is the
    "category rolls up across sources" view: the keys of the inner dict are the
    parallel sources a code draws from."""
    spr = source_prefixed_references()
    if cancer_code is not None:
        from ..gene_sets_cancer import resolve_cancer_type
        code = resolve_cancer_type(cancer_code)
        spr = spr[spr["cancer_code"].astype(str) == str(code)]
    out: dict = {}
    for _, r in spr.iterrows():
        out.setdefault(str(r["cancer_code"]), {}).setdefault(
            str(r["kind"]), []
        ).append(str(r["source_cohort"]))
    return out


def cancer_expression_source_candidates(
    cancer_types: Optional[str | Iterable[str]] = None,
) -> pd.DataFrame:
    """Candidate sources for missing or parent-backed expression references.

    The table is a planning/provenance surface, not an expression matrix. It
    records accession URLs, assay type, intended processing, gene-ID strategy,
    and current import status for registry codes whose direct observed cohort
    reference is still missing or should be refined.
    """
    df = _load_cancer_expression_source_candidates()
    codes = _resolve_cancer_types(cancer_types)
    if codes is not None:
        df = df[df["cancer_code"].astype(str).isin(codes)]
    return df.reset_index(drop=True)


def cancer_expression_reference_status(
    cancer_types: Optional[str | Iterable[str]] = None,
) -> pd.DataFrame:
    """Uniform expression-reference status for registry cancer codes.

    Returns one row per registry code with the packaged reference code used by
    :func:`cancer_expression`, direct/parent/TCGA status, and the best current
    acquisition candidate when a direct reference is not yet packaged.
    """
    from ..gene_sets_cancer import cancer_type_registry

    registry = cancer_type_registry()
    codes = _resolve_cancer_types(cancer_types)
    if codes is not None:
        registry = registry[registry["code"].astype(str).isin(codes)]

    candidates = _load_cancer_expression_source_candidates()
    candidate_first = (
        candidates.drop_duplicates(subset=["cancer_code"])
        .set_index("cancer_code")
        .to_dict(orient="index")
    )
    refs = available_cancer_expression_references()
    # Derive the code set from the already-loaded `refs` rather than
    # _reference_code_set() — the available-references view dedups by cohort but
    # retains every cancer_code, so this avoids a second reference-frame load.
    reference_codes = frozenset(refs["cancer_code"].astype(str))
    pan_codes = _pan_expression_codes()
    reference_summaries = _reference_cohort_summaries(refs, pan_codes)
    registry_by_code = registry.set_index("code")

    def _text(value) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value)

    rows = []
    for _, reg in registry.iterrows():
        code = str(reg["code"])
        reference_code = _resolve_expression_reference_code_from_lookups(
            code,
            registry=registry_by_code,
            reference_codes=reference_codes,
            pan_codes=pan_codes,
        )
        if code in reference_codes:
            status = "direct_reference"
        elif code in pan_codes:
            status = "tcga_pan_cancer"
        elif reference_code is not None:
            status = "parent_reference"
        else:
            status = "candidate_or_missing"

        ref_info = reference_summaries.get(reference_code, {}) if reference_code else {}
        candidate = candidate_first.get(code, {})
        rows.append({
            "cancer_code": code,
            "name": _text(reg.get("name", "")),
            "family": _text(reg.get("family", "")),
            "parent_code": _text(reg.get("parent_code", "")),
            "reference_status": status,
            "reference_code": reference_code or "",
            "reference_source_project": _text(ref_info.get("source_project", "")),
            "reference_source_cohort": _text(ref_info.get("source_cohort", "")),
            "reference_n_samples": ref_info.get("n_samples", np.nan),
            "candidate_status": _text(candidate.get("source_status", "")),
            "candidate_source_project": _text(candidate.get("source_project", "")),
            "candidate_source_cohort": _text(candidate.get("source_cohort", "")),
            "candidate_accession": _text(candidate.get("accession", "")),
            "candidate_url": _text(candidate.get("source_url", "")),
            "candidate_processing_plan": _text(candidate.get("processing_plan", "")),
        })
    return _string_id_columns(pd.DataFrame(rows).reset_index(drop=True), "cancer_code")


def _filter_cancer_code(df: pd.DataFrame, cancer_code: str | None) -> pd.DataFrame:
    if cancer_code is None:
        return df.reset_index(drop=True)
    from ..gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_code)
    return df[df["cancer_code"].astype(str).eq(code)].reset_index(drop=True)


def tumor_up_vs_matched_normal(cancer_code: str | None = None) -> pd.DataFrame:
    """Cancer-specific solid-tumor markers up vs matched normal tissue.

    The bundled table is a compact marker panel, not a full expression
    matrix. It includes one row per selected tumor-up gene with Ensembl ID,
    tumor TPM, matched-normal HPA nTPM, and broad normal-tissue guardrail
    columns used by downstream analysis packages.
    """
    return _filter_cancer_code(get_data("tumor-up-vs-matched-normal"), cancer_code)


def heme_tumor_up_vs_matched_normal(cancer_code: str | None = None) -> pd.DataFrame:
    """Heme analogue of :func:`tumor_up_vs_matched_normal`."""
    return _filter_cancer_code(
        get_data("heme-tumor-up-vs-matched-normal"),
        cancer_code,
    )


_REFERENCE_LONG_ID_COLUMNS = [
    "Ensembl_Gene_ID",
    "Symbol",
    "Proteoform_ID",
    "Member_Ensembl_Gene_IDs",
    "cancer_code",
    "source_cohort",
]
_REFERENCE_LONG_PROVENANCE_COLUMNS = [
    "source_project",
    "source_version",
    "n_samples",
    "n_detected",
    "processing_pipeline",
    "notes",
]
_REFERENCE_POOLED_PROVENANCE_COLUMNS = [
    "n_samples",
    "n_detected",
    "source_project",
    "processing_pipeline",
]
_REFERENCE_LONG_VALUE_COLUMNS = ["normalization", "expression", "q1", "q3"]


def _reference_compatibility_columns(
    *, include_provenance: bool, pool: bool
) -> list[str]:
    columns = list(_REFERENCE_LONG_ID_COLUMNS)
    if include_provenance:
        columns += (
            _REFERENCE_POOLED_PROVENANCE_COLUMNS
            if pool
            else _REFERENCE_LONG_PROVENANCE_COLUMNS
        )
    return columns + _REFERENCE_LONG_VALUE_COLUMNS


def _project_oncoref_reference_schema(
    delegated: pd.DataFrame,
    *,
    include_provenance: bool,
    pool: bool,
) -> pd.DataFrame:
    """Project oncoref's provenance superset onto pirlygenes' public schema."""
    columns = _reference_compatibility_columns(
        include_provenance=include_provenance,
        pool=pool,
    )
    out = delegated.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = (
                np.nan
                if column in {"n_samples", "n_detected", "expression", "q1", "q3"}
                else pd.NA
            )
    out = out[columns]
    return _string_id_columns(
        out,
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "source_cohort",
        "normalization",
    )


def _compatibility_availability_records(
    records: Iterable[dict], *, label: str
) -> list[dict]:
    out = []
    for record in records:
        adapted = dict(record)
        adapted["normalization"] = label
        out.append(adapted)
    return out


def _reference_compatibility_genes(
    genes: Optional[Iterable[str]],
) -> Optional[list[str]]:
    """Normalize and expand symbols before oncoref applies its filter."""
    if genes is None:
        return None
    requested = [genes] if isinstance(genes, str) else list(genes)
    expanded: list[str] = []
    for gene in requested:
        # filter_to_genes(), used by the pre-delegation implementation, made
        # symbol filters whitespace- and case-insensitive. Resolve display
        # aliases from the original spelling, then normalize every candidate
        # because oncoref's symbol match is exact.
        token = str(gene).strip()
        for candidate in (
            token,
            *get_alias_as_list(token),
            *get_reverse_alias_as_list(token),
        ):
            normalized = str(candidate).strip().upper()
            if normalized and normalized not in expanded:
                expanded.append(normalized)
    return expanded


def _reference_compatibility_source_cohorts(
    source_kind: Optional[str | Iterable[str]],
    source_cohort: Optional[str | Iterable[str]],
) -> Optional[list[str] | str]:
    """Resolve pirlygenes source-kind semantics to exact cohort filters.

    Pirlygenes' cohort registry is the compatibility authority for ``kind``.
    Resolving kinds to cohort IDs before delegation preserves that public
    contract and ensures filtering occurs before oncoref pools source rows.

    ``source_cohort`` itself is exact. In particular, the generic Treehouse
    TCGA-samples cohort and the SARC-histology cohort are distinct upstream
    identities and are never expanded into each other. Exact deprecated IDs are
    canonicalized by oncoref; an explicitly empty selection uses a nonmatching
    sentinel until oncoref#412 preserves empty filters itself.
    """
    from oncoref import canonical_cohort_id

    if source_cohort is None:
        delegated_filter = None
    elif isinstance(source_cohort, str):
        delegated_filter = canonical_cohort_id(source_cohort)
    else:
        delegated_filter = [canonical_cohort_id(value) for value in source_cohort]
    if source_kind is None:
        if source_cohort is not None:
            requested = (
                [delegated_filter]
                if isinstance(delegated_filter, str)
                else delegated_filter
            )
            if not requested or not any(str(cohort) for cohort in requested):
                return [_EMPTY_REFERENCE_SOURCE_COHORT]
        return delegated_filter

    requested_kinds = (
        [source_kind] if isinstance(source_kind, str) else list(source_kind)
    )
    requested_kinds = {str(kind) for kind in requested_kinds if str(kind)}
    from ..gene_sets_cancer import cohort_registry_df

    registry = cohort_registry_df()
    matching = registry.loc[
        registry["kind"].astype(str).isin(requested_kinds), "cohort_id"
    ].astype(str).tolist()
    allowed = matching

    if delegated_filter is not None:
        requested = (
            [delegated_filter]
            if isinstance(delegated_filter, str)
            else delegated_filter
        )
        allowed_set = set(allowed)
        allowed = [cohort for cohort in requested if cohort in allowed_set]

    return allowed or [_EMPTY_REFERENCE_SOURCE_COHORT]


def _partition_reference_codes_by_owner_view(
    cancer_types: Optional[str | Iterable[str]],
) -> tuple[Optional[list[str]], list[str]]:
    """Split requests between the compatibility summary and artifact views.

    A code stays on the historical all-source summary view whenever that view
    can serve it. Only codes absent there are delegated to the canonical
    artifact view. This preserves source-union behavior while making future
    artifact-only cohorts visible without a cohort-name allowlist.
    """
    artifact_only = (
        _oncoref_artifact_reference_code_set()
        - _oncoref_summary_reference_code_set()
    )
    if cancer_types is None:
        return None, sorted(artifact_only)
    requested = (
        [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    )
    return (
        [code for code in requested if str(code) not in artifact_only],
        [str(code) for code in requested if str(code) in artifact_only],
    )


def _artifact_record_matches_source_filter(
    record: dict,
    *,
    source_cohorts: Optional[str | Iterable[str]],
    exclude_microarray_proxy: bool,
) -> bool:
    if source_cohorts is not None:
        allowed = (
            {str(source_cohorts)}
            if isinstance(source_cohorts, str)
            else {str(value) for value in source_cohorts}
        )
        if str(record.get("source_cohort", "")) not in allowed:
            return False
    if exclude_microarray_proxy:
        source_text = " ".join(
            str(record.get(column, ""))
            for column in (
                "source_scale_class",
                "source_type",
                "processing_pipeline",
                "notes",
            )
        ).lower()
        if any(token in source_text for token in (
            "microarray", "tpm_proxy", "tpm-proxy", "tpm proxy",
        )):
            return False
    return True


def _filter_artifact_reference_source(
    frame: pd.DataFrame,
    *,
    source_cohorts: Optional[str | Iterable[str]],
    exclude_microarray_proxy: bool,
) -> pd.DataFrame:
    """Apply source-union filters to an artifact-only delegated result."""
    attrs = dict(frame.attrs)
    availability = []
    allowed_pairs = set()
    allowed_cohorts_by_code = {}
    for original in attrs.get("availability", []):
        record = dict(original)
        allowed = bool(record.get("available")) and (
            _artifact_record_matches_source_filter(
                record,
                source_cohorts=source_cohorts,
                exclude_microarray_proxy=exclude_microarray_proxy,
            )
        )
        if record.get("available") and not allowed:
            record["available"] = False
            record["missing_reason"] = "no_reference_artifact_matching_source_filter"
        if allowed:
            code = str(record.get("cancer_code", ""))
            cohort = str(record.get("source_cohort", ""))
            allowed_pairs.add((code, cohort))
            allowed_cohorts_by_code[code] = cohort
        availability.append(record)

    if frame.empty:
        out = frame.copy()
    elif "source_cohort" in frame.columns:
        pairs = pd.Series(
            list(zip(
                frame["cancer_code"].astype(str),
                frame["source_cohort"].astype(str),
            )),
            index=frame.index,
        )
        out = frame.loc[pairs.isin(allowed_pairs)].copy()
    else:
        # oncoref omits provenance columns from artifact output when
        # include_provenance=False. Pirlygenes keeps source_cohort as a stable
        # identity column, so recover it from the same availability record used
        # to decide whether the row is selectable.
        codes = frame["cancer_code"].astype(str)
        out = frame.loc[codes.isin(allowed_cohorts_by_code)].copy()
        out["source_cohort"] = (
            out["cancer_code"].astype(str).map(allowed_cohorts_by_code)
        )
    out.attrs.update(attrs)
    out.attrs["availability"] = availability
    out.attrs["missing_requests"] = [
        record for record in availability if not record.get("available")
    ]
    return out


def _artifact_sample_qc_by_code(codes: Sequence[str]) -> dict[str, str]:
    """Return each artifact's effective build-QC policy for raw recomputes."""
    import oncoref

    availability = oncoref.cancer_reference_expression_availability(
        cancer_types=codes,
        normalize="tpm_clean",
        sample_qc="artifact",
        reference_source="artifact",
    )
    valid = {"pass", "pass_or_warn", "all"}
    policies = {}
    for record in availability.loc[availability["available"]].to_dict("records"):
        policy = str(record.get("artifact_sample_qc", ""))
        if policy in valid:
            policies[str(record["cancer_code"])] = policy
    return policies


def _oncoref_reference_mode(
    *,
    cancer_types: Optional[str | Iterable[str]],
    genes: Optional[Iterable[str]],
    mode: str,
    include_provenance: bool,
    exclude_microarray_proxy: bool,
    source_kind: Optional[str | Iterable[str]],
    source_cohort: Optional[str | Iterable[str]],
    collapse_protein_identical: bool,
    collapse_cdna_identical: bool,
    pool: bool,
) -> pd.DataFrame:
    """Delegate one legacy normalization mode and adapt its labels/schema."""
    import oncoref

    # Derive both historical log views from delegated linear summaries. Besides
    # keeping one deterministic transform, this ensures identical-locus collapse
    # and pooling happen in linear space before log1p.
    delegated_mode = (
        mode.removesuffix("_log1p") if mode.endswith("_log1p") else mode
    )
    requested_genes = (
        None
        if genes is None
        else ([genes] if isinstance(genes, str) else list(genes))
    )
    requested_source_cohort = (
        None
        if source_cohort is None
        else (
            source_cohort
            if isinstance(source_cohort, str)
            else list(source_cohort)
        )
    )
    compatibility_genes = _reference_compatibility_genes(requested_genes)
    delegated_source_cohort = _reference_compatibility_source_cohorts(
        source_kind,
        requested_source_cohort,
    )
    summary_codes, artifact_codes = _partition_reference_codes_by_owner_view(
        cancer_types
    )
    delegated_parts: list[pd.DataFrame] = []
    if summary_codes is None or summary_codes:
        delegated_parts.append(oncoref.cancer_reference_expression(
            cancer_types=summary_codes,
            genes=compatibility_genes,
            normalize=delegated_mode,
            format="long",
            include_provenance=include_provenance,
            on_missing="empty",
            auto_fetch=False,
            sample_qc="all",
            reference_source="summary_rows_all",
            gene_id_style="pirlygenes",
            gene_universe="pirlygenes",
            # Resolve source kinds with pirlygenes' registry and forward exact
            # cohorts. This avoids inheriting gaps in oncoref's kind map.
            source_kind=None,
            source_cohort=delegated_source_cohort,
            exclude_microarray_proxy=exclude_microarray_proxy,
            pool=pool,
            collapse_cdna_identical=collapse_cdna_identical,
            collapse_protein_identical=collapse_protein_identical,
        ))

    if artifact_codes:
        if delegated_mode == "tpm":
            policies = _artifact_sample_qc_by_code(artifact_codes)
            artifact_groups: dict[str, list[str]] = {}
            for code in artifact_codes:
                # oncoref's strict-pass default is the conservative fallback
                # for legacy metadata without a recorded effective policy.
                artifact_groups.setdefault(policies.get(code, "pass"), []).append(code)
        else:
            artifact_groups = {"artifact": artifact_codes}

        for sample_qc, codes in artifact_groups.items():
            artifact = oncoref.cancer_reference_expression(
                cancer_types=codes,
                genes=compatibility_genes,
                normalize=delegated_mode,
                format="long",
                include_provenance=include_provenance,
                on_missing="empty",
                auto_fetch=False,
                sample_qc=sample_qc,
                reference_source="artifact",
                gene_id_style="pirlygenes",
                gene_universe="pirlygenes",
                source_kind=None,
                source_cohort=None,
                exclude_microarray_proxy=False,
                pool=False,
                collapse_cdna_identical=collapse_cdna_identical,
                collapse_protein_identical=collapse_protein_identical,
            )
            artifact = _filter_artifact_reference_source(
                artifact,
                source_cohorts=delegated_source_cohort,
                exclude_microarray_proxy=exclude_microarray_proxy,
            )
            artifact_attrs = dict(artifact.attrs)
            sample_counts = {
                str(record.get("cancer_code", "")): record.get(
                    "n_reference_samples", np.nan
                )
                for record in artifact_attrs.get("availability", [])
            }
            if include_provenance and "n_samples" in artifact.columns:
                missing_samples = artifact["n_samples"].isna()
                artifact.loc[missing_samples, "n_samples"] = (
                    artifact.loc[missing_samples, "cancer_code"]
                    .astype(str)
                    .map(sample_counts)
                )
            if pool and not artifact.empty:
                artifact["source_cohort"] = "POOLED"
                for quantile in ("q1", "q3"):
                    artifact[quantile] = np.nan
                if include_provenance and "source_project" in artifact.columns:
                    artifact["source_project"] = "pooled"
            artifact.attrs.update(artifact_attrs)
            delegated_parts.append(artifact)

    if delegated_parts:
        delegated = pd.concat(delegated_parts, ignore_index=True)
        attrs = dict(delegated_parts[0].attrs)
        attrs["availability"] = [
            record
            for part in delegated_parts
            for record in part.attrs.get("availability", [])
        ]
        attrs["missing_requests"] = [
            record
            for part in delegated_parts
            for record in part.attrs.get("missing_requests", [])
        ]
        sources = list(dict.fromkeys(
            str(part.attrs.get("reference_source", ""))
            for part in delegated_parts
            if part.attrs.get("reference_source")
        ))
        attrs["reference_source"] = "+".join(sources)
    else:
        delegated = pd.DataFrame(columns=[
            "cancer_code", "source_cohort", "expression", "q1", "q3",
        ])
        attrs = {"availability": [], "missing_requests": []}
    label = _REFERENCE_VALUE_COLUMNS[mode][3]
    delegated = delegated.copy()
    compatibility_transforms: list[str] = []
    if compatibility_genes != requested_genes:
        compatibility_transforms.append(
            "legacy gene aliases expanded before delegated filtering"
        )
    if source_kind is not None:
        compatibility_transforms.append(
            "source-kind filter resolved through pirlygenes cohort registry"
        )
    if artifact_codes:
        compatibility_transforms.append(
            "artifact-only cohorts delegated through oncoref artifact view"
        )
        if delegated_mode == "tpm":
            compatibility_transforms.append(
                "raw TPM artifact-only cohorts use artifact-recorded sample QC"
            )
    if cancer_types is not None:
        allowed_codes = {str(code) for code in cancer_types}
        delegated_codes = set(delegated["cancer_code"].astype(str))
        availability_codes = {
            str(record.get("cancer_code", ""))
            for record in attrs.get("availability", [])
        }
        if (delegated_codes | availability_codes) - allowed_codes:
            compatibility_transforms.append(
                "cancer-type request constrained to pirlygenes aggregate semantics"
            )
        delegated = delegated.loc[
            delegated["cancer_code"].astype(str).isin(allowed_codes)
        ].copy()
        for attr_name in ("availability", "missing_requests"):
            attrs[attr_name] = [
                record
                for record in attrs.get(attr_name, [])
                if str(record.get("cancer_code", "")) in allowed_codes
            ]
    delegated["normalization"] = label
    # Collapse/pool in linear space before deriving the historical raw-log view.
    if mode.endswith("_log1p"):
        for column in ("expression", "q1", "q3"):
            delegated[column] = np.log1p(
                pd.to_numeric(delegated[column], errors="coerce")
            )
        source_label = "raw TPM" if mode == "tpm_log1p" else "clean TPM"
        compatibility_transforms.append(
            f"{mode} derived with numpy.log1p from delegated {source_label}"
        )
    if pool and include_provenance and "source_project" in delegated.columns:
        delegated.loc[
            delegated["source_cohort"].astype(str).eq("POOLED"),
            "source_project",
        ] = "pooled"

    out = _project_oncoref_reference_schema(
        delegated,
        include_provenance=include_provenance,
        pool=pool,
    )
    out.attrs.update(attrs)
    availability = _compatibility_availability_records(
        attrs.get("availability", []), label=label
    )
    missing = _compatibility_availability_records(
        attrs.get("missing_requests", []), label=label
    )
    out.attrs["availability"] = availability
    out.attrs["missing_requests"] = missing
    out.attrs["delegated_to"] = "oncoref.cancer_reference_expression"
    out.attrs["reference_backend"] = "oncoref"
    out.attrs["compatibility_transforms"] = compatibility_transforms
    return out


def _merge_reference_compatibility_attrs(
    out: pd.DataFrame, parts: Sequence[pd.DataFrame]
) -> None:
    if parts:
        out.attrs.update(parts[0].attrs)
    out.attrs["availability"] = [
        record for part in parts for record in part.attrs.get("availability", [])
    ]
    out.attrs["missing_requests"] = [
        record for part in parts for record in part.attrs.get("missing_requests", [])
    ]
    out.attrs["compatibility_transforms"] = list(dict.fromkeys(
        transform
        for part in parts
        for transform in part.attrs.get("compatibility_transforms", [])
    ))
    out.attrs["delegated_to"] = "oncoref.cancer_reference_expression"
    out.attrs["reference_backend"] = "oncoref"


def _reference_wide_from_delegated_long(
    long: pd.DataFrame,
    *,
    modes: Sequence[str],
    parts: Sequence[pd.DataFrame],
    requested_codes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    # Resolve explicit requests before row-level source filtering so an empty
    # result retains its stable wide schema (for example CLL_TPM_clean even
    # when source_kind excludes CLL).  For the all-cohorts default, retain the
    # delegated availability/observed-code behavior.
    available_codes = list(dict.fromkeys(requested_codes or []))
    for part in parts:
        for record in part.attrs.get("availability", []):
            code = str(record.get("cancer_code", ""))
            if record.get("available") and code and code not in available_codes:
                available_codes.append(code)
    for code in long.get("cancer_code", pd.Series(dtype="string")).astype(str):
        if code and code not in available_codes:
            available_codes.append(code)

    wide = long[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates().copy()
    for code in available_codes:
        for mode in modes:
            label = _REFERENCE_VALUE_COLUMNS[mode][3]
            column = f"{code}_{label}"
            values = long.loc[
                long["cancer_code"].astype(str).eq(code)
                & long["normalization"].astype(str).eq(label),
                ["Ensembl_Gene_ID", "expression"],
            ].drop_duplicates(subset=["Ensembl_Gene_ID"])
            wide = wide.merge(
                values.rename(columns={"expression": column}),
                on="Ensembl_Gene_ID",
                how="left",
            )
            if column not in wide.columns:
                wide[column] = np.nan
    wide = _string_id_columns(wide, "Ensembl_Gene_ID", "Symbol")
    _merge_reference_compatibility_attrs(wide, parts)
    return wide


def cancer_reference_expression(
    cancer_types: Optional[str | Iterable[str]] = None,
    genes: Optional[Iterable[str]] = None,
    normalize: str | Sequence[str] = "tpm_clean",
    *,
    format: str = "long",
    include_provenance: bool = True,
    exclude_microarray_proxy: bool = False,
    source_kind: Optional[str | Iterable[str]] = None,
    source_cohort: Optional[str | Iterable[str]] = None,
    collapse_protein_identical: bool = False,
    collapse_cdna_identical: bool = False,
    pool: bool = False,
) -> pd.DataFrame:
    """Source-agnostic tumor expression references delegated to oncoref.

    This compatibility wrapper preserves pirlygenes' normalization labels,
    long/wide schemas, source-union semantics, provenance projection, filters,
    pooling, and identical-locus options. Summary-backed codes retain the
    historical all-source view; codes available only through oncoref's
    canonical artifacts use that view automatically. Empirical rows and
    provenance come only from :func:`oncoref.cancer_reference_expression`; no
    local expression fallback is attempted. Delegation availability, missing
    requests, and deterministic compatibility transforms are exposed in
    ``DataFrame.attrs``.

    ``source_kind`` selects union members by processing-source kind (for example
    ``"treehouse"``, ``"geo"``, ``"target"``, or ``"cllmap"``). It is not
    the sample origin: Treehouse-reprocessed TCGA rows select under
    ``source_kind="treehouse"``, not a fabricated ``"tcga"`` kind. Use
    ``source_cohort=`` for exact cohort-level selection.

    Cross-cohort unions can mix assays and gene universes. Microarray-proxy TPM
    is not magnitude-comparable across platforms; pass
    ``exclude_microarray_proxy=True`` for a pipeline-homogeneous pool. A gene
    absent from one member remains unavailable for that member, never measured
    zero. Rows therefore retain ``(cancer_code, source_cohort)`` identity unless
    the caller explicitly requests ``pool=True``.

    Parameters
    ----------
    cancer_types
        Optional registry code, alias, or iterable. Computed aggregates expand
        to the union of their subtype rows: for example, ``"SARC"`` retains
        each sarcoma member's own ``cancer_code`` and ``source_cohort``.
    genes
        Optional gene-symbol or Ensembl-ID subset. Legacy aliases and retired
        Ensembl IDs supported by the pirlygenes/oncoref migration map resolve to
        the same canonical rows.
    normalize
        One mode or a sequence of ``"tpm"``, ``"tpm_clean"``,
        ``"tpm_log1p"``, or ``"tpm_clean_log1p"``. ``"clean_tpm"`` is an
        alias for ``"tpm_clean"``.
    format
        ``"long"`` returns one row per gene/cancer/source/normalization;
        ``"wide"`` returns one row per gene with columns such as
        ``CLL_TPM_clean``.
    include_provenance
        Include source/sample/provenance columns in long-form output.
    exclude_microarray_proxy
        Exclude source rows whose magnitudes are microarray-derived TPM proxies.
    source_kind / source_cohort
        Optional processing-kind or exact-cohort filters; each accepts one value
        or an iterable.
    collapse_protein_identical / collapse_cdna_identical
        Sum identical loci once per ``(cancer_code, source_cohort)`` in linear
        space before any log transform. At most one collapse mode may be true.
        cDNA identity is the conservative read-recovery view; protein identity
        additionally combines protein-identical/cDNA-distinct loci.
    pool
        Collapse each source union to one n-sample-weighted row per
        ``(gene, cancer_code, normalization)``. Only cohorts that measured the
        gene contribute. ``source_cohort`` becomes ``"POOLED"``; ``q1`` and
        ``q3`` are ``NaN`` because quantiles cannot be reconstructed from cohort
        summaries. Pool only pipeline-comparable sources.

    Returns
    -------
    pd.DataFrame
        Defensive copy suitable for downstream mutation. ``attrs`` records the
        delegation target, availability, missing requests, and compatibility
        transforms.
    """
    modes = _resolve_reference_normalize_modes(normalize)
    _validate_reference_format(format)
    # Every mode is one delegated call. Materialize one-shot iterables once so
    # generators select the same request for every normalization stage.
    def materialize(value):
        return value if value is None or isinstance(value, str) else list(value)

    cancer_types = materialize(cancer_types)
    genes = materialize(genes)
    source_kind = materialize(source_kind)
    source_cohort = materialize(source_cohort)
    requested_codes = _resolve_cancer_types(
        cancer_types,
        expand_aggregates=True,
    )
    parts = [
        _oncoref_reference_mode(
            cancer_types=requested_codes,
            genes=genes,
            mode=mode,
            include_provenance=include_provenance,
            exclude_microarray_proxy=exclude_microarray_proxy,
            source_kind=source_kind,
            source_cohort=source_cohort,
            collapse_protein_identical=collapse_protein_identical,
            collapse_cdna_identical=collapse_cdna_identical,
            pool=pool,
        )
        for mode in modes
    ]
    if parts:
        long = pd.concat(parts, ignore_index=True)
    else:
        long = pd.DataFrame(columns=_reference_compatibility_columns(
            include_provenance=include_provenance,
            pool=pool,
        ))
    long = _string_id_columns(
        long,
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "source_cohort",
        "normalization",
    )
    _merge_reference_compatibility_attrs(long, parts)
    if format == "long":
        return long
    # Retain stable columns for directly available cohorts eliminated by a
    # row-level filter, while omitting oncoref-only grouping nodes that
    # pirlygenes historically treated as exact, unavailable codes.
    wide_requested_codes = (
        None
        if requested_codes is None
        else [
            code for code in requested_codes
            if code in _oncoref_reference_code_set()
        ]
    )
    return _reference_wide_from_delegated_long(
        long,
        modes=modes,
        parts=parts,
        requested_codes=wide_requested_codes,
    )


# ---------- accessors: unified normalization views (#319) ----------


_COHORT_VIEWS_DIR = "cancer-reference-expression-views"
_COHORT_VIEW_ID_COLS = ("Ensembl_Gene_ID", "Symbol")
_COHORT_VIEW_VALUE_FILES = {
    "tpm": "tpm.parquet",
    "clean_tpm": "clean_tpm.parquet",
}
_COHORT_VIEW_PROVENANCE_FILE = "provenance.parquet"
_ARTIFACT_MANIFEST_FILE = "_manifest.json"


class CohortExpressionViews:
    """The canonical normalization stages of a cohort reference in **one
    object**, so a consumer never re-normalizes inconsistently (#319).

    Attributes (each a gene × cohort DataFrame, ``Ensembl_Gene_ID`` + ``Symbol``
    index columns):

    * ``tpm`` — TPM-harmonized cohort summary (median).
    * ``clean_tpm`` — clean_tpm_16_9_75 (technical compartment **included**, pinned
      to the fixed fraction).
    * ``clean_tpm_biological`` — ``clean_tpm`` with the technical/ribosomal
      genes (the canonical censored-gene list) **dropped** — the
      biologically-actionable view.
    * ``provenance`` — one row per cohort: ``source_cohort``,
      ``processing_pipeline`` (records the native unit, e.g. STAR-counts→TPM),
      ``n_samples``.

    Note: the bundled references are TPM-harmonized at build time, so the
    **raw native** units (FPKM / microarray nTPM / counts) are not retained
    here — only recorded in ``provenance.processing_pipeline``. All three value
    views are on the TPM scale; the only differences are the censoring stage,
    so they are directly comparable and can't be accidentally re-normalized.
    """

    __slots__ = ("tpm", "clean_tpm", "clean_tpm_biological", "provenance")

    def __init__(self, tpm, clean_tpm, clean_tpm_biological, provenance):
        self.tpm = _object_column_index(tpm)
        self.clean_tpm = _object_column_index(clean_tpm)
        self.clean_tpm_biological = _object_column_index(clean_tpm_biological)
        self.provenance = _object_column_index(provenance)

    def __repr__(self):
        cohorts = list(self.provenance["source_cohort"]) if len(
            self.provenance) else []
        return (f"CohortExpressionViews(genes={self.tpm.shape[0]}, "
                f"cohorts={self.provenance.shape[0]}, "
                f"biological_genes={self.clean_tpm_biological.shape[0]}, "
                f"sources={cohorts[:3]}{'…' if len(cohorts) > 3 else ''})")


def _cohort_views_root():
    return _bundle_subdir(_COHORT_VIEWS_DIR)


def _artifact_manifest(root: Path) -> dict:
    path = root / _ARTIFACT_MANIFEST_FILE
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _cohort_views_present(root: Path) -> bool:
    return (
        root.exists()
        and all((root / name).exists()
                for name in _COHORT_VIEW_VALUE_FILES.values())
        and (root / _COHORT_VIEW_PROVENANCE_FILE).exists()
    )


@lru_cache(maxsize=4)
def _load_precomputed_cohort_views(root_text: str) -> tuple[pd.DataFrame, ...]:
    root = Path(root_text)
    tpm = _object_column_index(
        pd.read_parquet(root / _COHORT_VIEW_VALUE_FILES["tpm"])
    )
    clean = _object_column_index(
        pd.read_parquet(root / _COHORT_VIEW_VALUE_FILES["clean_tpm"])
    )
    provenance = _refresh_cohort_view_provenance(
        _object_column_index(
            pd.read_parquet(root / _COHORT_VIEW_PROVENANCE_FILE)
        )
    )
    return tpm, clean, provenance


def _canonicalize_source_cohort_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize exact cohort aliases through their oncoref owner.

    Existing pirlygenes data bundles can carry a provenance sidecar created
    before oncoref renamed the generic Treehouse TCGA cohort. Keeping the alias
    map upstream lets old bundles and new delegated rows expose one identity
    without another pirlygenes data release or a duplicated local map.
    """
    if "source_cohort" not in df.columns:
        return df

    from oncoref import canonical_cohort_id

    canonical = df["source_cohort"].map(
        lambda value: value if pd.isna(value) else canonical_cohort_id(value)
    )
    if canonical.equals(df["source_cohort"]):
        return df
    out = df.copy()
    out["source_cohort"] = canonical
    return out


def _refresh_cohort_view_provenance(df: pd.DataFrame) -> pd.DataFrame:
    """Refresh a bundled sidecar from oncoref's compact owner manifest.

    The precomputed value matrices remain valid across a provenance-only cohort
    rename. For matching ``(cancer_code, source_cohort)`` rows, use the current
    owner's cohort, pipeline, and sample metadata; retain sidecar values for
    custom or fixture rows absent from the owner manifest.
    """
    out = _canonicalize_source_cohort_ids(df)
    keys = ["cancer_code", "source_cohort"]
    if not set(keys) <= set(out.columns):
        return out

    refresh_cols = ["processing_pipeline", "n_samples"]
    owner = available_cancer_expression_references()[
        [*keys, *refresh_cols]
    ].drop_duplicates(keys).copy()
    # The compact owner manifest deliberately uses categorical metadata. Cast
    # only the tiny joined projection so fixture/custom sidecar values that are
    # not owner categories can be retained without mutating oncoref's cache.
    owner["processing_pipeline"] = owner["processing_pipeline"].astype(object)
    merged = out.merge(owner, on=keys, how="left", suffixes=("", "_owner"))
    for column in refresh_cols:
        owner_column = f"{column}_owner"
        if column in merged.columns:
            owner_values = merged[owner_column]
            merged[column] = owner_values.where(
                owner_values.notna(),
                merged[column],
            )
        else:
            merged[column] = merged[owner_column]
        merged = merged.drop(columns=owner_column)
    return _object_column_index(merged[out.columns])


def _cohort_value_cols(wide: pd.DataFrame) -> list[str]:
    return [c for c in wide.columns if c not in _COHORT_VIEW_ID_COLS]


def _object_column_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use ordinary object-dtype column labels for public wide matrices."""
    df.columns = pd.Index(df.columns.to_list(), dtype=object)
    return df


def _filter_canonical_view_genes(
    wide: pd.DataFrame,
    genes: Iterable[str],
) -> pd.DataFrame:
    gene_list = [str(g).strip() for g in genes if str(g).strip()]
    if not gene_list:
        return wide.iloc[0:0].reset_index(drop=True)

    # Keep the legacy symbol/alias behavior, then add canonical ENSG hits so
    # retired IDs and old symbols still match the baked canonical row.
    filtered = filter_to_genes(wide, gene_list)
    candidates: set[str] = set(gene_list)
    for gene in gene_list:
        candidates.update(get_alias_as_list(gene))
        candidates.update(get_reverse_alias_as_list(gene))

    target_ids: set[str] = set()
    from ..gene_canonicalization import canonical_gene_id

    for candidate in candidates:
        canonical = canonical_gene_id(candidate)
        if canonical is not None:
            target_ids.add(canonical)
    if target_ids:
        extra = wide[wide["Ensembl_Gene_ID"].astype(str).isin(target_ids)]
        if not extra.empty:
            filtered = pd.concat([filtered, extra], ignore_index=True)

    if filtered.empty:
        return filtered.reset_index(drop=True)
    return (
        filtered.drop_duplicates(subset=["Ensembl_Gene_ID"])
        .reset_index(drop=True)
    )


def _drop_all_missing_cohort_columns(wide: pd.DataFrame) -> pd.DataFrame:
    value_cols = _cohort_value_cols(wide)
    keep_values = [c for c in value_cols if wide[c].notna().any()]
    return wide[[*_COHORT_VIEW_ID_COLS, *keep_values]].copy()


def _drop_unmeasured_gene_rows(wide: pd.DataFrame) -> pd.DataFrame:
    """Drop genes (rows) with no value in any present cohort column — the rows
    left as pure NaN padding after a cohort/gene narrowing. With no cohort
    columns left nothing is measured, so the frame is emptied. This makes a
    narrowed slice of the full canonical matrix look like a pivot of just that
    slice (the from-reference contract), instead of carrying the whole
    all-cohort gene union."""
    value_cols = _cohort_value_cols(wide)
    if not value_cols:
        return wide.iloc[0:0].reset_index(drop=True)
    keep = wide[value_cols].notna().any(axis=1)
    return wide[keep].reset_index(drop=True)


def _select_cohort_columns(
    wide: pd.DataFrame,
    codes: list[str] | None,
) -> pd.DataFrame:
    if codes is None:
        return wide.copy()
    selected = [c for c in dict.fromkeys(codes) if c in wide.columns]
    return wide[[*_COHORT_VIEW_ID_COLS, *selected]].copy()


def _select_cohort_view_rows(
    wide: pd.DataFrame,
    *,
    protein_coding: bool,
    min_cohort_coverage: Optional[float],
) -> pd.DataFrame:
    if wide.empty or (not protein_coding and min_cohort_coverage is None):
        return wide.reset_index(drop=True)

    mask = pd.Series(True, index=wide.index)
    if protein_coding:
        from ..gene_canonicalization import canonical_gene_biotype

        mask &= wide["Ensembl_Gene_ID"].map(
            lambda e: canonical_gene_biotype(e) == "protein_coding"
        )
    if min_cohort_coverage is not None:
        cohort_cols = _cohort_value_cols(wide)
        if cohort_cols:
            coverage = wide[cohort_cols].notna().sum(axis=1) / len(cohort_cols)
            mask &= coverage >= min_cohort_coverage
    return wide[mask].reset_index(drop=True)


def _filter_cohort_view_provenance(
    provenance: pd.DataFrame,
    *,
    codes: list[str] | None,
) -> pd.DataFrame:
    out = provenance
    if codes is not None and "cancer_code" in out.columns:
        out = out[out["cancer_code"].astype(str).isin(set(codes))]
    prov_cols = ["source_cohort", "processing_pipeline", "n_samples"]
    result = (
        out[[c for c in prov_cols if c in out.columns]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Parquet/oncoref may retain a full-table Categorical vocabulary after a
    # narrow cohort slice.  Those unused categories are an encoding detail, not
    # public provenance, and can differ between equivalent artifacts.
    for column in result.select_dtypes(include="category").columns:
        result[column] = result[column].astype(object)
    return result


def _cohort_views_usable(root: Path) -> bool:
    """The precomputed views artifact is safe to use for the *canonical* path
    iff all three parquets exist and a readable manifest declares both
    canonical gene IDs and the running data version. Missing, malformed,
    stale, and unversioned manifests all take the fail-safe rebuild path."""
    if not _cohort_views_present(root):
        return False
    manifest = _artifact_manifest(root)
    return bool(
        manifest.get("canonical_gene_ids", False)
        and str(manifest.get("data_version", "")) == DATA_VERSION
    )


def _canonicalize_views_long(long: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long reference onto canonical ENSG keys (#465): sum the
    TPM-like columns and max ``n_detected`` per (cancer_code, source_cohort,
    normalization, canonical-ENSG)."""
    from ..gene_canonicalization import canonicalize_gene_table

    return canonicalize_gene_table(
        long,
        group_keys=["cancer_code", "source_cohort", "normalization"],
        value_cols=["expression", "q1", "q3"],
        max_cols=["n_detected"],
    )


def _pivot_views_long(
    long: pd.DataFrame,
    label: str,
    index_cols: list[str],
) -> pd.DataFrame:
    """Pivot one normalization stage of the long reference into a gene × cohort
    wide frame. Symbol is display metadata, never a join key (#465): when the
    long form is canonicalized we pivot on the canonical ENSG alone (so a gene
    can't fragment on residual symbol differences) and re-attach a single Symbol
    per gene afterwards."""
    sub = long[long["normalization"] == label]
    if sub.empty:
        return pd.DataFrame(columns=["Ensembl_Gene_ID", "Symbol"])
    wide = (sub.pivot_table(index=index_cols, columns="cancer_code",
                            values="expression", aggfunc="first",
                            observed=True)
            .reset_index())
    wide.columns.name = None
    wide = _object_column_index(wide)
    if "Symbol" not in wide.columns:
        symbol_by_gene = (sub.drop_duplicates("Ensembl_Gene_ID")
                          .set_index("Ensembl_Gene_ID")["Symbol"])
        wide.insert(1, "Symbol", wide["Ensembl_Gene_ID"].map(symbol_by_gene))
    return wide


def _rebuild_full_canonical_views() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the **full** canonical wide matrices (every gene × every cohort)
    plus a provenance table that still carries ``cancer_code``.

    This is the single source of truth for the canonical views. The precomputed
    artifact under ``cancer-reference-expression-views/`` is nothing but the
    on-disk serialization of this function's output (see
    ``scripts/generate_cohort_expression_views.py``), so the read path can treat
    "load artifact" and "rebuild" as interchangeable and apply one identical
    filter to either. Memoized on the reference-frame identity so a process that
    has no artifact pays this rebuild at most once, not per query.
    """

    def _build(_df: pd.DataFrame):
        long = _reference_long_from_summary_frame(_df)
        long = _canonicalize_views_long(long)
        index_cols = ["Ensembl_Gene_ID"]
        tpm = _pivot_views_long(long, "TPM", index_cols)
        clean = _pivot_views_long(long, "TPM_clean", index_cols)
        # Keep the full public availability metadata in future sidecars. The
        # lightweight reader remains compatible with older four-column
        # sidecars by filling these fields from registries (#565).
        from .reference_manifest import PUBLIC_COLUMNS

        prov_cols = list(PUBLIC_COLUMNS)
        provenance = (long[[c for c in prov_cols if c in long.columns]]
                      .drop_duplicates().reset_index(drop=True))
        return tpm, clean, provenance

    return _reference_view("full_canonical_views", _build)


def _valid_full_views(frames: tuple[pd.DataFrame, ...]) -> bool:
    """The loaded artifact must be the (tpm, clean, provenance) triple with the
    id columns on the two value matrices; anything else is treated as corrupt."""
    if not isinstance(frames, tuple) or len(frames) != 3:
        return False
    tpm, clean, provenance = frames
    if not all(isinstance(f, pd.DataFrame) for f in frames):
        return False
    for matrix in (tpm, clean):
        if not set(_COHORT_VIEW_ID_COLS) <= set(matrix.columns):
            return False
    return True


def _full_canonical_views() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The full canonical (tpm, clean_tpm, provenance) frames, served from the
    precomputed artifact when it is present and usable, else rebuilt from the
    reference. Both branches return the identical schema so a single filter
    works on either. A present-but-unreadable or schema-invalid artifact is
    never fatal — it degrades to the rebuild."""
    root = Path(_cohort_views_root())
    if _cohort_views_usable(root):
        try:
            frames = _load_precomputed_cohort_views(str(root))
        except Exception as exc:  # noqa: BLE001 — corrupt parquet / missing engine
            warnings.warn(
                f"Precomputed cohort views at {root} could not be read "
                f"({exc!r}); falling back to a full rebuild from the reference. "
                "This is much slower — check the data bundle and that a parquet "
                "engine (pyarrow) is installed.",
                RuntimeWarning, stacklevel=2,
            )
            frames = None
        else:
            if not _valid_full_views(frames):
                warnings.warn(
                    f"Precomputed cohort views at {root} have an unexpected "
                    "schema; falling back to a full rebuild from the reference. "
                    "This is much slower — the artifact may be stale or corrupt.",
                    RuntimeWarning, stacklevel=2,
                )
                _load_precomputed_cohort_views.cache_clear()
                frames = None
        if frames is not None:
            return frames
    return _rebuild_full_canonical_views()


def _apply_cohort_view_filters(
    tpm_full: pd.DataFrame,
    clean_full: pd.DataFrame,
    provenance_full: pd.DataFrame,
    cancer_types: Optional[str | Iterable[str]],
    genes: Optional[Iterable[str]],
    *,
    protein_coding: bool,
    min_cohort_coverage: Optional[float],
) -> CohortExpressionViews:
    """Slice the full canonical matrices down to a request. This is the **one**
    canonical-views filter, shared by the precomputed-artifact fast path and the
    rebuild fallback, so the two can never diverge.

    Order: select cohort columns → (optional) gene filter → drop cohorts and
    genes left all-missing by the narrowing → protein-coding / coverage row
    filter → biology-only view. Provenance is reduced to the public three
    columns and aligned to whichever cohorts survive.

    Whenever ``cancer_types`` or ``genes`` narrows the matrix we prune the
    NaN-only rows and columns the narrowing exposes, so a sliced view contains
    exactly the genes measured in the requested cohorts (matching a pivot of
    that slice) rather than the full all-cohort gene union (#474 review)."""
    codes = _resolve_cancer_types(cancer_types, expand_aggregates=True)
    tpm = _select_cohort_columns(tpm_full, codes)
    clean = _select_cohort_columns(clean_full, codes)

    provenance_codes = codes
    if genes is not None:
        gene_list = list(genes)
        tpm = _filter_canonical_view_genes(tpm, gene_list)
        clean = _filter_canonical_view_genes(clean, gene_list)

    if codes is not None or genes is not None:
        tpm = _drop_unmeasured_gene_rows(_drop_all_missing_cohort_columns(tpm))
        clean = _drop_unmeasured_gene_rows(_drop_all_missing_cohort_columns(clean))
        provenance_codes = list(dict.fromkeys(
            _cohort_value_cols(tpm) + _cohort_value_cols(clean)
        ))

    tpm = _select_cohort_view_rows(
        tpm,
        protein_coding=protein_coding,
        min_cohort_coverage=min_cohort_coverage,
    )
    clean = _select_cohort_view_rows(
        clean,
        protein_coding=protein_coding,
        min_cohort_coverage=min_cohort_coverage,
    )
    biological = drop_technical_genes(clean) if not clean.empty else clean
    return CohortExpressionViews(
        tpm,
        clean,
        biological,
        _filter_cohort_view_provenance(provenance_full, codes=provenance_codes),
    )


def _cohort_expression_views_from_reference(
    cancer_types: Optional[str | Iterable[str]] = None,
    genes: Optional[Iterable[str]] = None,
    *,
    canonicalize_genes: bool = True,
    protein_coding: bool = False,
    min_cohort_coverage: Optional[float] = None,
) -> "CohortExpressionViews":
    """Build the views straight from the long reference, filtering during the
    pivot. This is an **independent** implementation of the same contract as the
    canonical fast path: it powers the ``canonicalize_genes=False`` opt-out, and
    serves as the from-scratch oracle the canonical path is tested against.
    """
    long = _reference_long_from_summary_frame(
        _load_cancer_reference_expression(),
        cancer_types=cancer_types,
        genes=genes,
    )
    if canonicalize_genes:
        long = _canonicalize_views_long(long)
    index_cols = (["Ensembl_Gene_ID"] if canonicalize_genes
                  else ["Ensembl_Gene_ID", "Symbol"])

    tpm = _select_cohort_view_rows(
        _pivot_views_long(long, "TPM", index_cols),
        protein_coding=protein_coding,
        min_cohort_coverage=min_cohort_coverage,
    )
    clean = _select_cohort_view_rows(
        _pivot_views_long(long, "TPM_clean", index_cols),
        protein_coding=protein_coding,
        min_cohort_coverage=min_cohort_coverage,
    )
    # biological inherits clean's gene selection, then drops technical genes.
    biological = drop_technical_genes(clean) if not clean.empty else clean
    provenance = _filter_cohort_view_provenance(long, codes=None)
    return CohortExpressionViews(tpm, clean, biological, provenance)


def cohort_expression_views(
    cancer_types: Optional[str | Iterable[str]] = None,
    genes: Optional[Iterable[str]] = None,
    *,
    canonicalize_genes: bool = True,
    protein_coding: bool = False,
    min_cohort_coverage: Optional[float] = None,
) -> "CohortExpressionViews":
    """Bundle a cohort's normalization stages into one
    :class:`CohortExpressionViews` (tpm / clean_tpm / clean_tpm_biological +
    provenance) so downstream never re-normalizes inconsistently (#319).

    ``cancer_types`` / ``genes`` select cohorts and genes (aggregate codes like
    ``SARC`` expand to their subtypes). Values are the per-cohort medians.

    By default (``canonicalize_genes=True``) the result is sliced out of the
    **full canonical views** — every row keyed on one canonical ENSG so
    cross-release symbol drift cannot split a gene into several sparse rows.
    That full matrix is served from the precomputed
    ``cancer-reference-expression-views/`` artifact when present, and otherwise
    rebuilt from the reference once and memoized; either way the *same* filter
    runs, so the fast path and the fallback return identical results.
    ``canonicalize_genes=False`` opts out of canonicalization entirely and builds
    directly from the long reference (no precomputed artifact applies).

    ``protein_coding=True`` keeps only protein-coding genes (via the offline
    authority biotype), and ``min_cohort_coverage`` (0..1) keeps only genes
    measured in at least that fraction of cohorts — together they yield the dense
    coding core and skip the mostly-zero non-coding tail.
    """
    if min_cohort_coverage is not None and not 0 <= min_cohort_coverage <= 1:
        raise ValueError("min_cohort_coverage must be between 0 and 1")
    if canonicalize_genes:
        tpm_full, clean_full, provenance_full = _full_canonical_views()
        return _apply_cohort_view_filters(
            tpm_full,
            clean_full,
            provenance_full,
            cancer_types,
            genes,
            protein_coding=protein_coding,
            min_cohort_coverage=min_cohort_coverage,
        )
    return _cohort_expression_views_from_reference(
        cancer_types,
        genes,
        canonicalize_genes=False,
        protein_coding=protein_coding,
        min_cohort_coverage=min_cohort_coverage,
    )


# ---------- accessors: representative per-sample vectors (#312) ----------


def _bundle_subdir(name: str):
    """Locate a bundle shard directory: an in-repo checkout (``pirlygenes/data/…``)
    wins, else the downloaded bundle cache; the bundle is fetched if the
    directory is absent from both."""
    from pathlib import Path

    from .. import data_bundle
    from ..load_dataset import _BUNDLED_DATA_DIR

    in_repo = Path(_BUNDLED_DATA_DIR) / name
    if in_repo.exists():
        return in_repo
    cached = data_bundle.find(name)
    if cached is not None:
        return cached
    data_bundle.ensure_local()
    return data_bundle.cache_dir() / name


def available_representative_cohorts() -> list[str]:
    """Registry codes that ship a representative-samples artifact (sorted).

    Delegates to oncoref (pirlygenes#208): oncoref owns the source-matrix medoid
    selection and the representatives artifact; pirlygenes re-exports the accessor
    so trufflepig and notebooks keep a single import path.
    """
    import oncoref

    return sorted(oncoref.available_representative_cohorts())


def representative_cohort_samples(
    cancer_types: Optional[str | Iterable[str]] = None,
    *,
    k: Optional[int] = None,
    normalize: str = "tpm_clean",
    format: str = "wide",
    include_provenance: bool = False,
    canonicalize_genes: bool = True,
) -> pd.DataFrame:
    """Representative real per-sample expression vectors per cohort (#312).

    A bounded set of real joint per-sample medoid vectors per cohort, spanning
    the within-cohort variation, in the ``clean_tpm_16_9_75`` basis that matches
    the aggregate references — for the sample-level self-classification battery
    and for validating normalization / representation changes on realistic
    samples.

    Delegated to oncoref (pirlygenes#208). oncoref selects the medoids by a
    deterministic farthest-first traversal seeded at the true cohort medoid (the
    sample minimizing total distance to all others), then each subsequent pick is
    the sample farthest from those already chosen — a principled max-min coverage
    guarantee that is seed-free, always returns ``k``, and orders picks medoid
    first. (pirlygenes' former k-means++ selection was seed-dependent and could
    return fewer than ``k``.)

    Parameters
    ----------
    cancer_types
        Registry code, alias, or iterable; a computed-aggregate code expands to
        its members. ``None`` returns every cohort with representatives.
    k
        Keep at most the first ``k`` representatives per cohort (``None`` = all,
        currently up to 5), medoid first. Anonymized ``<CODE>_rep01`` …
    normalize
        ``"tpm_clean"`` (clean_tpm_16_9_75, as stored) or ``"tpm_clean_log1p"``.
    format
        ``"wide"`` → one ``Ensembl_Gene_ID`` / ``Symbol`` row per gene with one
        column per representative. ``"long"`` → one row per gene ×
        representative with ``cancer_code``; ``include_provenance=True`` adds
        ``source_cohort`` / ``source_project`` / ``n_cohort_samples``.
    canonicalize_genes
        Retained for signature compatibility. oncoref always returns the shared
        canonical ENSG space, so the wide outer join never splits a gene on
        cross-release symbol drift regardless of this flag.

    Returns
    -------
    pd.DataFrame
    """
    if normalize not in ("tpm_clean", "tpm_clean_log1p"):
        raise ValueError(
            "representative_cohort_samples normalize must be 'tpm_clean' or "
            "'tpm_clean_log1p' (the artifact ships only in clean_tpm_16_9_75)"
        )
    if format not in ("wide", "long"):
        raise ValueError("format must be 'wide' or 'long'")

    import oncoref

    out = oncoref.representative_cohort_samples(
        cancer_types,
        k=k,
        normalize=normalize,
        format=format,
        include_provenance=include_provenance,
        representative_id_style="pirlygenes",
        sample_qc="artifact",  # serve each cohort under its baked QC policy
    )
    # oncoref's long-format provenance is a superset; project to pirlygenes'
    # documented columns so the contract is unchanged for consumers.
    if format == "long" and include_provenance:
        keep = ["Ensembl_Gene_ID", "Symbol", "cancer_code", "representative_id",
                "expression", "source_cohort", "source_project", "n_cohort_samples"]
        out = out[[c for c in keep if c in out.columns]]
    return out.reset_index(drop=True)


# ---------- accessors: per-gene × cohort percentile vectors (#298) ----------


def available_percentile_cohorts() -> list[str]:
    """Cohort codes that ship a per-gene percentile-vector artifact (sorted).

    Delegated to oncoref (pirlygenes#208 / #298)."""
    import oncoref

    # Gene-level percentiles are scope-independent (oncoref's ``scope`` selects
    # gene- vs proteoform-level; the default suffices for the gene-level vector).
    return sorted(oncoref.available_percentile_cohorts())


def cohort_gene_percentiles(cancer_type, *, as_tpm: bool = True) -> pd.DataFrame:
    """Tail-weighted per-gene percentile vector for one cohort (#298).

    One row per gene (``Ensembl_Gene_ID`` + ``Symbol``) with 26 breakpoint
    columns — ``p0, p1, p5, p10 … p90, p95, p96, p97, p98, p99, p100`` — dense in
    the actionable upper tail, so a consumer can place a sample's gene as a
    **percentile rank within the cohort** instead of an absolute TPM.

    Computed on the biological clean_tpm_16_9_75 view. ``as_tpm=True`` (default)
    returns clean-TPM values, ``as_tpm=False`` the stored ``log1p`` values.

    Delegated to oncoref (pirlygenes#208); reads the cohort under its baked QC
    policy. Raises ``ValueError`` if the cohort has no per-sample data
    (summary-only cohorts have no vector — their coarse percentiles are in
    :func:`cancer_reference_expression`).
    """
    from oncoref.source_matrices import SourceMatrixError

    import oncoref

    from ..gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    try:
        # Gene-level percentiles are scope-independent (see
        # available_percentile_cohorts); no ``scope`` needed.
        df = oncoref.cohort_gene_percentiles(
            code, as_tpm=as_tpm, sample_qc="artifact", auto_fetch=True)
    except SourceMatrixError as err:
        raise ValueError(
            f"no percentile vector available for {code!r} — either it is a "
            f"summary-only cohort (no per-sample data; its coarse "
            f"p5/p10/p90/p95 are in cancer_reference_expression) or its "
            f"per-sample matrix could not be loaded. See "
            f"available_percentile_cohorts() for cohorts that ship one."
        ) from err
    return df.reset_index(drop=True)


# ---------- accessors: pan-cancer expression ----------


def pan_cancer_expression(
    genes: Optional[Iterable[str]] = None,
    normalize: Optional[str | Sequence[str]] = "tpm_clean",
    *,
    log_transform: bool = False,
    drop_technical_rna: bool = False,
    collapse_cdna_identical: bool = False,
    collapse_protein_identical: bool = False,
) -> pd.DataFrame:
    """Wide-form expression across HPA normal tissues + TCGA cancer types.

    50 normal tissues from HPA v23 consensus (``<tissue>_nTPM`` columns)
    plus 33 TCGA cancer types from HPA pathology + GDC/STAR reprocessing
    (``<code>_FPKM`` in native units with deterministic ``<code>_TPM``
    companions). Five computed tumor rollups (``BTC``/``CRC``/``NET``/
    ``NSCLC``/``SGC``) are built from sample-weighted TPM cohort medians and
    therefore have ``<code>_TPM`` but no synthetic FPKM. TPM and every requested
    analysis derivative are available uniformly across all tumor entities;
    FPKM is retained only as source provenance where it actually exists.

    An oncoref compatibility adapter canonicalizes the version-pinned local
    matrix with oncoref's alias map, collapses duplicate loci, and composes the
    TPM-only rollups from pirlygenes' selected-source rollup artifact. All
    normalization below runs locally on that canonical data, so this view stays
    identical in method to :func:`cancer_reference_expression` without an eager
    scan of oncoref's separate reference-summary bundle. See
    :func:`_pan_reference_frame`.

    Parameters
    ----------
    genes
        Optional iterable of gene symbols or Ensembl IDs to subset to.
    normalize
        Normalization mode or list of modes. Modes are additive and may be
        combined; dependencies are inserted automatically. ``"TPM"`` and
        ``"tpm"`` are equivalent.

        - ``"tpm_clean"`` (default) — start from the uniform TPM/nTPM
          analysis columns (including TPM-only computed rollups), then add
          ``<tissue>_nTPM_clean`` and ``<code>_TPM_clean`` columns with
          mtDNA / NUMT / rRNA / MALAT1+NEAT1 rows zeroed
          and each column's sum pinned back to 10⁶. This is the
          recommended view for analysis: every normalized analysis
          column on the same scale, technical-RNA denominator drift
          removed. Base ``<tissue>_nTPM`` and ``<code>_TPM`` columns,
          plus raw ``<code>_FPKM`` columns, remain unchanged.
        - ``None`` — raw/provenance view: raw TCGA ``<code>_FPKM``
          values, HPA ``<tissue>_nTPM`` values, deterministic TCGA
          ``<code>_TPM`` companions, and TPM-only computed rollups are
          preserved. No artifact-gene cleanup, HK scaling, percentile-rank,
          or log transform is applied.
        - ``"tpm"`` / ``"TPM"`` — the uniform tumor-analysis view: every
          TCGA and computed-rollup entity has ``<code>_TPM``; raw TCGA FPKM
          provenance remains available where present.
        - ``"tpm_log1p"`` — add ``<tissue>_nTPM_log1p`` and
          ``<code>_TPM_log1p`` columns using natural ``log1p`` over the
          TPM-scale analysis columns. Implies ``"tpm"``.
        - ``"hk"`` or ``"housekeeping"`` — add
          ``<tissue>_nTPM_hk`` and ``<code>_TPM_hk`` columns divided by
          their median-of-ratios housekeeping size factor. Implies ``"tpm"``.
        - ``"percentile"`` — within-column percentile rank (0–100),
          added as ``<tissue>_nTPM_percentile`` and
          ``<code>_TPM_percentile`` columns. Implies ``"tpm"``.
        - ``"tpm_clean_log1p"`` — first add clean TPM/nTPM columns, then
          add natural-log ``<tissue>_nTPM_clean_log1p`` and
          ``<code>_TPM_clean_log1p`` columns. Implies ``"tpm_clean"``.

        For example, ``normalize=["tpm_clean", "hk", "percentile"]``
        adds clean, housekeeping, and percentile columns in one call.
    log_transform
        Apply ``log2(x + 1)`` to value columns after any normalization.
    drop_technical_rna
        Drop mtDNA / NUMT / rRNA / nuclear-retained-lncRNA rows entirely
        (uses :func:`filter_technical_rna`). Distinct from
        ``normalize="tpm_clean"``: this removes rows, while
        ``"tpm_clean"`` zeroes them in added ``*_clean`` columns. See
        Boundary note in the module docstring.
    collapse_cdna_identical / collapse_protein_identical
        Collapse identical loci into one row per proteoform (same dual-identifier
        contract as :func:`cancer_reference_expression`), summed in linear space
        BEFORE any clean/log/percentile column is generated. At most one may be
        True. Regardless of these flags the result always carries the gene-view
        bridge columns ``Proteoform_ID`` (the proteoform each gene folds to — group
        by it to roll up) and ``Member_Ensembl_Gene_IDs`` (constituent ENSGs), so
        the gene/proteoform duality is uniform across accessors.

    Returns
    -------
    pd.DataFrame
        Defensive copy — safe to mutate.
    """
    normalize_modes = _resolve_pan_normalize_modes(normalize)
    if "tpm" not in normalize_modes:
        normalize_modes.insert(0, "tpm")

    # The cached compatibility view already contains deterministic TPM
    # companions for every FPKM cohort. Copy it before adding bridge and
    # normalization columns so callers can mutate independently.
    df = _pan_reference_frame().copy()

    # Proteoform duality (uniform with cancer_reference_expression): always add the
    # gene-view Proteoform_ID / Member_Ensembl_Gene_IDs bridge columns; optionally
    # collapse identical loci in LINEAR space here, BEFORE any clean/log/percentile
    # column is generated, so the summed proteoform values then normalise correctly.
    if collapse_cdna_identical and collapse_protein_identical:
        raise ValueError("set at most one of collapse_cdna_identical / "
                         "collapse_protein_identical")
    _collapse_kind = ("cdna" if collapse_cdna_identical
                      else "protein" if collapse_protein_identical else None)
    if _collapse_kind:
        from .protein_groups import collapse_wide, fold_ids, fold_symbols
        _linear = [c for c in df.columns if c.startswith(_VALUE_COL_PREFIXES)]
        df = collapse_wide(df, value_cols=_linear, kind=_collapse_kind)
        if genes is not None:   # fold the gene filter so a member-named panel hits
            genes = sorted(set(map(str, genes))
                           | set(fold_symbols(genes, kind=_collapse_kind))
                           | set(fold_ids(genes, kind=_collapse_kind)))
    else:
        from .protein_groups import add_proteoform_columns
        df = add_proteoform_columns(df)
    analysis_value_cols = _pan_analysis_value_cols(df)

    generated_value_cols: list[str] = []
    value_cols_by_mode: dict[str, list[str]] = {}
    for mode in normalize_modes:
        if mode == "tpm":
            continue
        if mode == "tpm_clean":
            # ONE clean TPM everywhere: the fixed_fraction 16/9/75 contract,
            # identical to cancer_reference_expression. (Was the legacy zero
            # drop-and-renormalize, which kept ribosomal proteins in biology and
            # inflated the biological budget to 1e6 — a different, non-comparable
            # "clean" than the reference table used.)
            normalized_df, _ = normalize_expression(
                df,
                label_col="Symbol",
                id_col="Ensembl_Gene_ID",
                value_cols=analysis_value_cols,
                censored_fill="fixed_fraction",
            )
            source_cols = analysis_value_cols
        elif mode == "tpm_log1p":
            normalized_df = log1p_transform(
                df, value_cols=analysis_value_cols,
            )
            source_cols = analysis_value_cols
        elif mode == "tpm_clean_log1p":
            clean_value_cols = value_cols_by_mode.get("tpm_clean", [])
            normalized_df = log1p_transform(
                df, value_cols=clean_value_cols,
            )
            source_cols = clean_value_cols
        elif mode == "hk":
            normalized_df = normalize_to_housekeeping(
                df, value_cols=analysis_value_cols,
            )
            source_cols = analysis_value_cols
        elif mode == "percentile":
            normalized_df, _ = percentile_rank_expression(
                df, value_cols=analysis_value_cols,
            )
            source_cols = analysis_value_cols
        else:  # pragma: no cover - guarded by _resolve_pan_normalize_modes
            continue
        df, new_cols = _add_pan_normalized_value_cols(
            df, normalized_df, source_cols, mode,
        )
        value_cols_by_mode[mode] = new_cols
        generated_value_cols.extend(new_cols)
    pipeline_value_cols = generated_value_cols or analysis_value_cols
    df = _apply_pipeline(
        df,
        drop_technical_rna=drop_technical_rna,
        genes=genes,
        log_transform=log_transform,
        percentile=False,
        value_cols=pipeline_value_cols,
    )
    return _rename_pan_expression_columns_entity_first(df)


def cancer_expression(
    cancer_type: str,
    genes: Optional[Iterable[str]] = None,
    normalize: str = "tpm_clean",
) -> pd.DataFrame:
    """Expression for a single cancer type from the best packaged reference.

    Parameters
    ----------
    cancer_type
        Registry code or alias (e.g. ``"PRAD"``, ``"prostate"``, ``"CLL"``).
    genes
        Optional gene-symbol / Ensembl-ID subset.
    normalize
        Normalization mode. Defaults to ``"tpm_clean"``. TCGA-backed
        references also support ``"hk"`` / ``"housekeeping"`` through
        :func:`pan_cancer_expression`.

    Returns
    -------
    pd.DataFrame
        Columns: ``Ensembl_Gene_ID``, ``Symbol``, ``expression``.
    """
    from ..gene_sets_cancer import resolve_cancer_type

    code = resolve_cancer_type(cancer_type)
    reference_code = _resolve_expression_reference_code(code)
    if reference_code is None:
        reference_code = code

    ref_modes = set(_REFERENCE_NORMALIZE_ALIASES.values())
    ref_mode = _REFERENCE_NORMALIZE_ALIASES.get(normalize)
    if ref_mode is None:
        ref_mode = _REFERENCE_NORMALIZE_ALIASES.get(str(normalize).lower())
    if ref_mode in ref_modes and _has_cancer_reference(reference_code):
        ref = cancer_reference_expression(
            cancer_types=[reference_code],
            genes=genes,
            normalize=ref_mode,
            include_provenance=False,
        )
        return ref[["Ensembl_Gene_ID", "Symbol", "expression"]].reset_index(
            drop=True,
        )

    pan_mode = _canonical_pan_normalize_token(str(normalize))
    df = pan_cancer_expression(
        genes=genes,
        normalize=pan_mode,
        drop_technical_rna=False,
    )
    suffix_by_mode = {
        "tpm": "TPM",
        "tpm_clean": "TPM_clean",
        "tpm_log1p": "TPM_log1p",
        "tpm_clean_log1p": "TPM_clean_log1p",
        "hk": "TPM_hk",
        "percentile": "TPM_percentile",
    }
    if pan_mode not in suffix_by_mode:
        raise ValueError(
            f"unsupported normalize mode for cancer_expression: {normalize!r}"
        )
    col = f"{reference_code}_{suffix_by_mode[pan_mode]}"
    if col not in df.columns:
        raise ValueError(
            f"no {normalize!r} expression column for {cancer_type!r} "
            f"(resolved to {code!r})"
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

    The comparison population is the original 33 source cohorts. Computed
    rollups are not independent observations and are therefore never included
    in the background. When a rollup is itself the target, its member cohorts
    are excluded from the background as well.

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
    target_col = f"{code}_TPM_hk"
    if target_col not in df.columns:
        raise ValueError(
            f"no HK-normalized TPM column for {cancer_type!r} "
            f"(resolved to {code!r})"
        )
    # Paired FPKM provenance distinguishes the 33 source cohorts from the
    # TPM-only computed rollups. Including rollups here would count source
    # cohorts more than once (for example LUAD both directly and via NSCLC).
    source_cols = [
        col
        for col in df.columns
        if col.endswith("_TPM_hk")
        and f"{col[:-len('_TPM_hk')]}_FPKM" in df.columns
    ]
    excluded_codes = {code, *_PAN_COMPUTED_ROLLUP_MEMBERS.get(code, ())}
    other_cols = [
        col
        for col in source_cols
        if col[:-len("_TPM_hk")] not in excluded_codes
    ]
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
    import oncoref

    return oncoref.hpa_cell_type_expression()


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
    "cancer_reference_expression",
    "available_cancer_expression_references",
    "cancer_expression_reference_status",
    "cancer_expression_source_candidates",
    "tumor_up_vs_matched_normal",
    "heme_tumor_up_vs_matched_normal",
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
