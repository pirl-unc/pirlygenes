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
  with optional deterministic TPM companion columns derived from those
  FPKM columns and optional added normalized analysis columns.
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

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..gene_families import gene_family_ids
from ..gene_names import get_alias_as_list, get_reverse_alias_as_list
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
    """Add normalized TPM/nTPM analysis columns without overwriting inputs."""
    out = df.copy()
    target_cols = []
    for col in value_cols:
        target = _pan_normalized_col_name(col, normalize)
        out[target] = normalized_df[col]
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
        Columns to rescale. If ``None``, picks columns using either the
        internal unit-prefix schema (``nTPM_``, ``FPKM_``, ``TPM_``) or
        the public entity-suffix schema (``*_nTPM``, ``*_FPKM``,
        ``*_TPM`` and added normalized suffixes).

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
    targets = set()
    for gene in genes:
        name = str(gene).strip()
        targets.add(name.upper())
        targets.update(alias.upper() for alias in get_alias_as_list(name))
        targets.update(alias.upper() for alias in get_reverse_alias_as_list(name))
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
) -> list[str] | None:
    if cancer_types is None:
        return None
    from ..gene_sets_cancer import resolve_cancer_type

    if isinstance(cancer_types, str):
        requested = [cancer_types]
    else:
        requested = list(cancer_types)
    return [resolve_cancer_type(code) for code in requested]


def _load_cancer_reference_expression() -> pd.DataFrame:
    return get_data("cancer-reference-expression")


def _has_cancer_reference(code: str) -> bool:
    df = _load_cancer_reference_expression()
    return code in set(df["cancer_code"].astype(str))


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


def available_cancer_expression_references() -> pd.DataFrame:
    """Packaged non-TCGA tumor reference cohorts available by cancer code.

    Returns one row per ``(cancer_code, source_cohort)`` with sample-count
    and processing provenance. Downstream consumers can use this to decide
    which non-TCGA references are available without inspecting data files.
    """
    df = _load_cancer_reference_expression()
    keep = [
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "n_samples",
        "processing_pipeline",
    ]
    present = [c for c in keep if c in df.columns]
    out = df[present].drop_duplicates().sort_values(
        ["cancer_code", "source_cohort"],
    )
    return out.reset_index(drop=True)


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


def cancer_reference_expression(
    cancer_types: Optional[str | Iterable[str]] = None,
    genes: Optional[Iterable[str]] = None,
    normalize: str | Sequence[str] = "tpm_clean",
    *,
    format: str = "long",
    include_provenance: bool = True,
) -> pd.DataFrame:
    """Source-agnostic packaged tumor expression references.

    This accessor is for non-TCGA references such as CLL-map, MMRF
    CoMMpass, TARGET, and future GEO cohorts. Values are TPM-scale
    cohort summaries; ``normalize="tpm_clean"`` is the default analysis
    view and uses per-sample technical-RNA cleanup before aggregation.

    Parameters
    ----------
    cancer_types
        Optional registry code, alias, or iterable of codes/aliases.
    genes
        Optional gene-symbol / Ensembl-ID subset.
    normalize
        One mode or a list of modes: ``"tpm"``, ``"tpm_clean"``,
        ``"tpm_log1p"``, or ``"tpm_clean_log1p"``. ``"clean_tpm"`` is
        accepted as an alias for ``"tpm_clean"``.
    format
        ``"long"`` returns one row per gene/cancer/source/normalization.
        ``"wide"`` returns one row per gene and columns like
        ``CLL_TPM_clean``.
    include_provenance
        Include source/sample/provenance columns in long-form output.

    Returns
    -------
    pd.DataFrame
        Defensive copy suitable for downstream mutation.
    """
    modes = _resolve_reference_normalize_modes(normalize)
    _validate_reference_format(format)
    df = _load_cancer_reference_expression()
    codes = _resolve_cancer_types(cancer_types)
    if codes is not None:
        df = df[df["cancer_code"].astype(str).isin(codes)]
    available_codes = list(df["cancer_code"].astype(str).drop_duplicates())
    if codes is not None:
        wide_codes = [code for code in codes if code in set(available_codes)]
    else:
        wide_codes = available_codes
    if genes is not None:
        df = filter_to_genes(df, genes)

    base_cols = ["Ensembl_Gene_ID", "Symbol", "cancer_code", "source_cohort"]
    provenance_cols = [
        "source_project",
        "source_version",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
    ]
    frames = []
    for mode in modes:
        expr, q1, q3, label = _reference_expr_value(df, mode)
        cols = list(base_cols)
        if include_provenance:
            cols += [c for c in provenance_cols if c in df.columns]
        part = df[cols].copy()
        part["normalization"] = label
        part["expression"] = expr
        part["q1"] = q1
        part["q3"] = q3
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)

    if format == "long":
        return long
    if format != "wide":
        raise ValueError("format must be 'long' or 'wide'")

    wide = long[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates().copy()
    for (code, label), group in long.groupby(["cancer_code", "normalization"]):
        col = f"{code}_{label}"
        values = group[["Ensembl_Gene_ID", "expression"]].drop_duplicates(
            subset=["Ensembl_Gene_ID"],
        )
        wide = wide.merge(
            values.rename(columns={"expression": col}),
            on="Ensembl_Gene_ID",
            how="left",
        )
    expected_value_cols = [
        f"{code}_{_REFERENCE_VALUE_COLUMNS[mode][3]}"
        for code in wide_codes
        for mode in modes
    ]
    for col in expected_value_cols:
        if col not in wide.columns:
            wide[col] = np.nan
    return wide[["Ensembl_Gene_ID", "Symbol", *expected_value_cols]]


# ---------- accessors: pan-cancer expression ----------


def pan_cancer_expression(
    genes: Optional[Iterable[str]] = None,
    normalize: Optional[str | Sequence[str]] = "tpm_clean",
    *,
    log_transform: bool = False,
    drop_technical_rna: bool = False,
) -> pd.DataFrame:
    """Wide-form expression across HPA normal tissues + TCGA cancer types.

    50 normal tissues from HPA v23 consensus (``<tissue>_nTPM`` columns)
    plus 33 TCGA cancer types from HPA pathology + GDC/STAR reprocessing
    (``<code>_FPKM`` in native units). The accessor always appends
    deterministic ``<code>_TPM`` companion columns derived from the FPKM
    columns, preserving the raw FPKM columns for provenance.

    Parameters
    ----------
    genes
        Optional iterable of gene symbols or Ensembl IDs to subset to.
    normalize
        Normalization mode or list of modes. Modes are additive and may be
        combined; dependencies are inserted automatically. ``"TPM"`` and
        ``"tpm"`` are equivalent.

        - ``"tpm_clean"`` (default) — first ensure deterministic
          ``<code>_TPM`` columns exist from ``<code>_FPKM``, then add
          ``<tissue>_nTPM_clean`` and ``<code>_TPM_clean`` columns with
          mtDNA / NUMT / rRNA / MALAT1+NEAT1 rows zeroed
          and each column's sum pinned back to 10⁶. This is the
          recommended view for analysis: every normalized analysis
          column on the same scale, technical-RNA denominator drift
          removed. Base ``<tissue>_nTPM`` and ``<code>_TPM`` columns,
          plus raw ``<code>_FPKM`` columns, remain unchanged.
        - ``None`` — raw/provenance view: raw TCGA ``<code>_FPKM``
          values and HPA ``<tissue>_nTPM`` values are preserved, while
          deterministic ``<code>_TPM`` analysis columns are generated
          from ``<code>_FPKM``. No artifact-gene cleanup, HK scaling,
          percentile-rank, or log transform is applied.
        - ``"tpm"`` / ``"TPM"`` — add missing ``<code>_TPM`` companion
          columns from ``<code>_FPKM`` while preserving raw FPKM.
        - ``"tpm_log1p"`` — add ``<tissue>_nTPM_log1p`` and
          ``<code>_TPM_log1p`` columns using natural ``log1p`` over the
          TPM-scale analysis columns. Implies ``"tpm"``.
        - ``"hk"`` or ``"housekeeping"`` — add
          ``<tissue>_nTPM_hk`` and ``<code>_TPM_hk`` columns divided by
          their housekeeping-gene median. Implies ``"tpm"``.
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

    Returns
    -------
    pd.DataFrame
        Defensive copy — safe to mutate.
    """
    normalize_modes = _resolve_pan_normalize_modes(normalize)
    if "tpm" not in normalize_modes:
        normalize_modes.insert(0, "tpm")

    df = get_data("pan-cancer-expression")
    df, _ = add_tpm_columns_from_fpkm(df)
    analysis_value_cols = _pan_analysis_value_cols(df)

    generated_value_cols: list[str] = []
    value_cols_by_mode: dict[str, list[str]] = {}
    for mode in normalize_modes:
        if mode == "tpm":
            continue
        if mode == "tpm_clean":
            normalized_df = _bundled_normalize(
                df,
                technical_rna_normalize=True,
                remove_noncoding=False,
                renormalize=True,
                value_cols=analysis_value_cols,
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
    ref_modes = set(_REFERENCE_NORMALIZE_ALIASES.values())
    ref_mode = _REFERENCE_NORMALIZE_ALIASES.get(normalize)
    if ref_mode is None:
        ref_mode = _REFERENCE_NORMALIZE_ALIASES.get(str(normalize).lower())
    if ref_mode in ref_modes and _has_cancer_reference(code):
        ref = cancer_reference_expression(
            cancer_types=[code],
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
    col = f"{code}_{suffix_by_mode[pan_mode]}"
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
    tpm_cols = [c for c in df.columns if c.endswith("_TPM_hk")]
    target_col = f"{code}_TPM_hk"
    if target_col not in df.columns:
        raise ValueError(
            f"no HK-normalized TPM column for {cancer_type!r} "
            f"(resolved to {code!r})"
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
    "cancer_reference_expression",
    "available_cancer_expression_references",
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
