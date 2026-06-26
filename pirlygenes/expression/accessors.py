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

* :func:`normalize_to_housekeeping` — divide each column by the
  active housekeeping-panel geometric mean.
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
from ..gene_names import get_alias_as_list, get_reverse_alias_as_list
from ..load_dataset import get_data
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
    """Rescale each value column by the active housekeeping-panel baseline.

    The result is unitless: a value of 1.0 in a given column means
    "expressed at the column's housekeeping baseline". Works across
    TPM, FPKM, and nTPM units since the normalization is per-column. This
    compatibility helper delegates to
    :func:`pirlygenes.expression.normalize.tpm_to_housekeeping_normalized`,
    so the HK path uses the same ENSG-first HPA-derived panel and geometric
    mean denominator everywhere.

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
    out, _ = tpm_to_housekeeping_normalized(df, id_col=id_col, value_cols=cols)
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
    # Read-only shared view. All callers (_has_cancer_reference,
    # cancer_reference_summary, cancer_reference_expression) filter to a
    # cancer_code / gene slice and .copy() that subset before returning or
    # mutating, so the full-frame defensive copy is pure waste — and for this
    # ~367 MB, ~1M-string-row table it dominated test-suite wall time (#278).
    return get_data("cancer-reference-expression", copy=False)


# Identity-keyed memo of read-only views derived purely from the (shared,
# process-wide) reference frame. The frame is a singleton — get_data(copy=False)
# returns the same object every call — so any view computed from it is stable
# until the data reloads. Keying each cache entry on the frame's *identity*
# makes it self-invalidate the moment a test monkeypatches
# _load_cancer_reference_expression to return a different frame. Without this,
# available_cancer_expression_references() factorized the ~1M-row frame once per
# cancer code, which alone was ~300 s of the serial suite (#278 follow-up).
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


def _has_cancer_reference(code: str) -> bool:
    return code in _reference_code_set()


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
    reference_codes = _reference_code_set()
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


def available_cancer_expression_references() -> pd.DataFrame:
    """Packaged non-TCGA tumor reference cohorts available by cancer code.

    Returns one row per ``(cancer_code, source_cohort)`` with sample-count,
    processing provenance, and primary-vs-metastasis annotation. Within
    each cancer_code, rows are ordered with ``tumor_origin == 'primary'``
    first so consumers that take ``.iloc[0]`` get the canonical reference
    cohort. Downstream consumers can use this to decide which non-TCGA
    references are available without inspecting data files.

    The expensive projection (drop_duplicates over the ~1M-row frame) is
    memoized on the reference frame's identity; this returns a fresh ``.copy()``
    of that cached view each call, so callers may mutate the result freely
    without corrupting the cache. The copy is cheap — the cached frame is the
    deduplicated cohort list (one row per ``(cancer_code, source_cohort)``).
    """
    return _reference_view(
        "available_references", _build_available_references
    ).copy()


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
    # Sort so primary > mixed > metastasis > everything else within
    # each cancer_code; ties broken by source_cohort.
    origin_priority = {"primary": 0, "mixed": 1, "metastasis": 2}
    if "tumor_origin" in out.columns:
        out = out.assign(
            _origin_rank=out["tumor_origin"].map(origin_priority).fillna(9),
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


def _pool_union_rows(long: pd.DataFrame, *,
                     include_provenance: bool) -> pd.DataFrame:
    """Collapse per-(gene, cancer_code, source_cohort) union rows to one
    heterogeneity-safe pooled row per (gene, cancer_code, normalization).

    Per-gene availability: only source-cohort rows with a non-NaN ``expression``
    (a cohort that measured the gene) are pooled; the central value is
    ``n_samples``-weighted and the pooled ``n_samples`` is the summed
    measuring-cohort sample count. ``q1``/``q3`` -> ``NaN`` (quantiles aren't
    recombinable from summaries). Vectorised (no per-group Python) so it is cheap
    even on the full union.
    """
    keys = ["Ensembl_Gene_ID", "Symbol", "cancer_code", "normalization"]
    # Member_Ensembl_Gene_IDs is constant per Ensembl_Gene_ID (a proteoform's
    # constituent ENSGs), so carrying it as a group key preserves it through
    # pooling without changing the grouping.
    if "Member_Ensembl_Gene_IDs" in long.columns:
        keys = keys + ["Member_Ensembl_Gene_IDs"]
    w = long["n_samples"].where(long["expression"].notna()) \
        if "n_samples" in long.columns else long["expression"].notna().astype(float)
    tmp = long.assign(_w=w, _wx=w * long["expression"])
    agg_spec = {"_wx": ("_wx", "sum"), "_w": ("_w", "sum")}
    if "n_detected" in long.columns:
        # samples detecting the gene, summed over measuring cohorts
        agg_spec["n_detected"] = ("n_detected", "sum")
    g = tmp.groupby(keys, as_index=False, sort=False).agg(**agg_spec)
    g["expression"] = g["_wx"] / g["_w"].replace(0, np.nan)
    g["n_samples"] = g["_w"]
    g["q1"] = np.nan
    g["q3"] = np.nan
    g["source_cohort"] = "POOLED"
    out_cols = ["Ensembl_Gene_ID", "Symbol", "Member_Ensembl_Gene_IDs",
                "cancer_code", "source_cohort"]
    if include_provenance:
        if "n_samples" in long.columns:
            out_cols.append("n_samples")
        if "n_detected" in g.columns:
            out_cols.append("n_detected")
        g["processing_pipeline"] = "pooled_n_weighted"
        g["source_project"] = "pooled"
        out_cols += ["source_project", "processing_pipeline"]
    out_cols += ["normalization", "expression", "q1", "q3"]
    return g[[c for c in out_cols if c in g.columns]]


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
    """Source-agnostic packaged tumor expression references.

    ``source_kind`` is the ``source:node`` cohort-grammar selector (#366): keep
    only the union members whose **processing source** is of the given kind(s) —
    one or a list of ``cohort-registry`` ``kind`` values (``treehouse``,
    ``geo``, ``target``, ``beataml``, ``cgci``, ``cllmap``, ``mmrf``,
    ``ucologne``, ``unc``, ``curated``, ``computed``). ``None`` (default) =
    ``all:`` (every source). The kind is the *processing* source, not the sample
    origin — there is deliberately **no ``tcga`` kind**: our TCGA data is
    Treehouse-reprocessed, so it selects under ``source_kind="treehouse"`` (a
    ``tcga`` kind would falsely conflate it with GDC-pipeline TCGA, which the
    package does not carry). For sample-origin / cohort-level precision (e.g.
    just the Treehouse TCGA subset) use ``source_cohort=`` with the exact
    ``cohort-registry`` cohort id(s), e.g.
    ``source_cohort="TREEHOUSE_POLYA_25_01_TCGA_SUBSET"``.

    Cross-cohort (``all:`` union) heterogeneity contract — IMPORTANT when a
    computed-aggregate code (``SARC``, ``CRC``, …) expands to many member
    cohorts:

    - **Mixed assays/pipelines.** Members may originate as microarray, FPKM,
      RPKM, or raw counts and are each converted to clean TPM **independently,
      per cohort, at build time** (see the ``processing_pipeline`` column).
      Microarray-proxy TPM (``*_microarray_tpm_proxy_*``) is NOT magnitude-
      comparable across platforms (a smaller probe universe inflates per-gene
      TPM); pass ``exclude_microarray_proxy=True`` to drop those members for a
      pipeline-homogeneous, poolable view.
    - **Different gene universes.** Members measure different gene sets (here
      ~13k–61k genes); a gene absent from a member is ``not_measurable`` for
      that cohort, **never 0**. Rows are returned **per-(gene, cancer_code,
      source_cohort)** — the reference deliberately does NOT pre-pool the union
      into one fabricated summary, so a consumer can pool correctly (per-gene
      availability mask, ``n_samples`` weighting) instead of averaging
      incomparable scales.

    This accessor is for non-TCGA references such as CLL-map, MMRF
    CoMMpass, TARGET, and future GEO cohorts. Values are TPM-scale
    cohort summaries; ``normalize="tpm_clean"`` is the default analysis
    view and uses per-sample technical-RNA cleanup before aggregation.

    Parameters
    ----------
    cancer_types
        Optional registry code, alias, or iterable of codes/aliases. A
        computed-aggregate code expands to the **union** of its member
        subtypes' rows (each row keeps its own subtype ``cancer_code`` and
        ``source_cohort``): ``"SARC"`` returns every sarcoma histology atom,
        ``"SARC_RMS"`` the four rhabdomyosarcoma subtypes, ``"SARC_LPS"`` the
        liposarcoma subtypes. No pooled summary row is fabricated — pool the
        returned subtype rows (or use the per-sample coverage path) if a single
        aggregate statistic is needed.
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
    collapse_protein_identical
        When ``True``, sum protein-identical gene loci (segmental-duplication
        paralogs, histone clusters, the CT47A cancer-testis cluster, …) into one
        row per group **per (cancer_code, source_cohort)**, in linear TPM space
        (``NaN`` members ignored). Reads split across loci that encode the
        identical protein are recombined, so the value is a faithful
        protein-abundance proxy and per-gene thresholds (e.g. CTA "ON" counting)
        aren't under-counted. Off by default (preserves the per-locus rows and
        the shipped reference semantics). See
        :func:`pirlygenes.expression.protein_groups.collapse_protein_identical_loci_long`.
    collapse_cdna_identical
        When ``True``, sum **cDNA-identical** loci (byte-identical canonical
        coding sequence) per (gene group, cancer_code, source_cohort) in linear
        TPM — the universal **read-recovery** collapse. Such loci multi-map (a
        quantifier can't assign reads between them), so each is split /
        under-counted and only the sum is reliable. A small curated override
        (``proteoform-collapse-overrides``, e.g. the CT47A antigen) force
        -collapses a few 100%-protein/cDNA-distinct groups. This is the
        principled, single-source collapse for any abundance/percentile view —
        it leaves cDNA-*distinct* paralogs alone (histone clusters stay split,
        MAGEA3 vs MAGEA6 stay distinct). Distinct from
        ``collapse_protein_identical`` (groups on protein identity, which would
        also sweep the histone clusters).
    pool
        When ``True``, collapse the ``all:``-union's per-(gene, cancer_code,
        source_cohort) rows into **one heterogeneity-safe pooled row per (gene,
        cancer_code)**. Each gene is pooled over **only the source cohorts that
        measured it** (per-gene availability — a cohort missing the gene is
        excluded, not treated as 0), with the per-cohort value **n_samples
        -weighted**. ``source_cohort`` becomes ``"POOLED"`` and the pooled
        ``n_samples`` is the summed sample count of the cohorts that measured the
        gene.

        Granularity caveat: at summary level per-gene availability is **binary at
        cohort grain** — a cohort either carries the gene (a row) or not. The
        reference ``n_samples`` is a cohort-wide constant, not per-gene, so the
        pooled ``n_samples`` is "samples in cohorts that measured the gene", NOT
        a dropout-aware per-(gene, sample) ``n_available`` (that requires the
        per-sample matrices — :class:`pirlygenes.expression.stats.PooledCohorts`
        at build time, where ``n_available`` is computed properly).

        This is the read-time, summary-level analogue of ``PooledCohorts``. The
        n-weighting is **exact for a mean** central value and an **approximation
        for a median** (quantiles cannot be recombined from per-cohort summaries
        — ``q1``/``q3`` are returned ``NaN``; for exact pooled quantiles use the
        per-sample ``PooledCohorts`` path). Only ~7 codes are multi-source today,
        so this is a no-op (one row in → one row out) for the rest. **Pool within
        a pipeline-homogeneous set** — pooling absolute TPM across microarray
        -proxy and RNA-seq members is not magnitude-comparable (pass
        ``exclude_microarray_proxy=True`` first).

    Returns
    -------
    pd.DataFrame
        Defensive copy suitable for downstream mutation.
    """
    modes = _resolve_reference_normalize_modes(normalize)
    _validate_reference_format(format)
    df = _load_cancer_reference_expression()
    codes = _resolve_cancer_types(cancer_types, expand_aggregates=True)
    idx_by_code = _reference_indices_by_code()
    if codes is not None:
        # Slice via the cached code→row-positions index instead of an
        # .astype(str).isin() scan of the full ~1M-row frame — the latter cost
        # ~15 s across the suite when called once per cancer code (#278 f/u).
        available_codes = [c for c in dict.fromkeys(codes) if c in idx_by_code]
        if available_codes:
            positions = np.concatenate([idx_by_code[c] for c in available_codes])
            df = df.iloc[positions]
        else:
            df = df.iloc[0:0]
        wide_codes = [code for code in codes if code in set(available_codes)]
    else:
        available_codes = list(idx_by_code.keys())
        wide_codes = available_codes
    if genes is not None:
        df = filter_to_genes(df, genes)
    if exclude_microarray_proxy:
        # drop cross-platform-incomparable microarray-proxy members so the
        # remaining all:-union view is pipeline-homogeneous and poolable.
        df = df[~df["processing_pipeline"].astype(str)
                .str.contains("microarray_tpm_proxy", na=False)]
    if source_kind is not None:
        # source:node selector — keep only members of the given cohort kind(s).
        kinds = {source_kind} if isinstance(source_kind, str) else set(source_kind)
        cr = get_data("cohort-registry")
        cohort_kind = dict(zip(cr["cohort_id"].astype(str),
                               cr["kind"].astype(str)))
        df = df[df["source_cohort"].astype(str).map(cohort_kind).isin(kinds)]
    if source_cohort is not None:
        # origin/cohort-level precision (e.g. the Treehouse TCGA subset).
        cohorts = ({source_cohort} if isinstance(source_cohort, str)
                   else set(source_cohort))
        df = df[df["source_cohort"].astype(str).isin(cohorts)]

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

    if collapse_protein_identical:
        from .protein_groups import collapse_protein_identical_loci_long
        long = collapse_protein_identical_loci_long(
            long,
            group_keys=["cancer_code", "source_cohort", "normalization"],
            sum_cols=["expression", "q1", "q3"],
            max_cols=("n_detected",),
        )

    if collapse_cdna_identical:
        from .protein_groups import collapse_cdna_identical_loci_long
        long = collapse_cdna_identical_loci_long(
            long,
            group_keys=["cancer_code", "source_cohort", "normalization"],
            sum_cols=["expression", "q1", "q3"],
            max_cols=("n_detected",),
        )

    if pool:
        long = _pool_union_rows(long, include_provenance=include_provenance)

    # Dual gene/proteoform identifiers on every row, so consumers can work at
    # either level: `Proteoform_ID` is the stable proteoform key each row maps to
    # (= `Ensembl_Gene_ID` on the proteoform frame; the gene's proteoform on the
    # per-ENSG gene frame), and `Member_Ensembl_Gene_IDs` is the constituent real
    # ENSGs (the gene view; = the gene's own ENSG when not folded).
    from ..gene_ids import strip_version
    from .protein_groups import canonical_to_symbol, member_to_canonical
    # Proteoform_ID uses the SAME proteoform space as the frame, so it equals
    # Ensembl_Gene_ID on a collapsed frame (cDNA xor protein) and is the matching
    # gene->proteoform bridge on the gene frame (cDNA = the matrix default).
    _kind = "protein" if collapse_protein_identical else "cdna"
    _m2c, _c2s = member_to_canonical(_kind), canonical_to_symbol(_kind)
    _ids = long["Ensembl_Gene_ID"].astype(str)
    _pid = {}
    for u in _ids.unique():               # per-id key (no dedup); idempotent on a
        s = strip_version(u)              # proteoform ID (not a member -> itself)
        _pid[u] = _c2s.get(_m2c.get(s, s), s)
    long = long.assign(Proteoform_ID=_ids.map(_pid))
    if "Member_Ensembl_Gene_IDs" not in long.columns:
        long = long.assign(Member_Ensembl_Gene_IDs=long["Ensembl_Gene_ID"])
    # keep the four identifier columns grouped + consistently ordered (so the gene
    # and proteoform frames share one schema)
    _id_cols = ["Ensembl_Gene_ID", "Symbol", "Proteoform_ID",
                "Member_Ensembl_Gene_IDs"]
    long = long[[c for c in _id_cols if c in long.columns]
                + [c for c in long.columns if c not in _id_cols]]
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
    wide = wide[["Ensembl_Gene_ID", "Symbol", *expected_value_cols]]
    return _string_id_columns(wide, "Ensembl_Gene_ID", "Symbol")


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
        self.tpm = tpm
        self.clean_tpm = clean_tpm
        self.clean_tpm_biological = clean_tpm_biological
        self.provenance = provenance

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
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _artifact_has_canonical_genes(root: Path) -> bool:
    return bool(_artifact_manifest(root).get("canonical_gene_ids"))


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
    tpm = pd.read_parquet(root / _COHORT_VIEW_VALUE_FILES["tpm"])
    clean = pd.read_parquet(root / _COHORT_VIEW_VALUE_FILES["clean_tpm"])
    provenance = pd.read_parquet(root / _COHORT_VIEW_PROVENANCE_FILE)
    return tpm, clean, provenance


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
    return (
        out[[c for c in prov_cols if c in out.columns]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _cohort_views_usable(root: Path) -> bool:
    """The precomputed views artifact is safe to use for the *canonical* path
    iff all three parquets exist and the manifest does not actively declare the
    artifact non-canonical. A missing manifest is treated as canonical (legacy
    artifacts predate the marker); an explicit ``canonical_gene_ids: false`` is
    rejected so a non-canonical bake can never masquerade as the canonical
    source."""
    if not _cohort_views_present(root):
        return False
    manifest = _artifact_manifest(root)
    if manifest and not manifest.get("canonical_gene_ids", False):
        return False
    return True


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
                            values="expression", aggfunc="first")
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
        long = cancer_reference_expression(
            None, genes=None, normalize=["tpm", "tpm_clean"],
            format="long", include_provenance=True)
        long = _canonicalize_views_long(long)
        index_cols = ["Ensembl_Gene_ID"]
        tpm = _pivot_views_long(long, "TPM", index_cols)
        clean = _pivot_views_long(long, "TPM_clean", index_cols)
        prov_cols = [
            "cancer_code", "source_cohort", "processing_pipeline", "n_samples",
        ]
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
    long = cancer_reference_expression(
        cancer_types, genes=genes, normalize=["tpm", "tpm_clean"],
        format="long", include_provenance=True)
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
    prov_cols = ["source_cohort", "processing_pipeline", "n_samples"]
    provenance = (long[[c for c in prov_cols if c in long.columns]]
                  .drop_duplicates().reset_index(drop=True))
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

_REPRESENTATIVES_DIR = "cancer-reference-expression-representatives"


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


def _available_shard_codes(root) -> list[str]:
    """Sorted cohort codes that ship a parquet shard under ``root`` (the shared
    body of the ``available_*_cohorts`` accessors). ``root`` is resolved by the
    per-artifact root function so test monkeypatches on those still apply."""
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.parquet"))


def _representatives_root():
    return _bundle_subdir(_REPRESENTATIVES_DIR)


def _percentiles_root():
    return _bundle_subdir(_PERCENTILES_DIR)


def available_representative_cohorts() -> list[str]:
    """Registry codes that ship a representative-samples shard (sorted)."""
    return _available_shard_codes(_representatives_root())


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

    The packaged cohort references are per-cohort aggregates (median /
    quantiles), so downstream can only validate classification / normalization
    against the cohort *median* — which overstates accuracy and can't
    reconstruct a physiological sample. This accessor returns a **bounded** set
    of real joint per-sample vectors per cohort — medoids spanning the
    within-cohort variation — in the same ``clean_tpm_16_9_75`` basis as the
    aggregates, for the honest sample-level self-classification battery and for
    validating normalization / representation changes on realistic samples.

    Parameters
    ----------
    cancer_types
        Registry code, alias, or iterable. A computed-aggregate code expands to
        the union of its member subtypes (e.g. ``"SARC"``). ``None`` returns
        every cohort that ships representatives. Codes without a representatives
        shard are skipped.
    k
        Keep at most the first ``k`` representatives per cohort (``None`` = all,
        currently up to 5). Representatives are anonymized (``<CODE>_rep01`` …).
    normalize
        ``"tpm_clean"`` (clean_tpm_16_9_75, as stored) or ``"tpm_clean_log1p"``
        (log1p of the stored values).
    format
        ``"wide"`` → one ``Ensembl_Gene_ID`` / ``Symbol`` row per gene with one
        column per representative (genes × samples). ``"long"`` → one row per
        gene × representative with ``cancer_code`` + ``representative_id``;
        ``include_provenance=True`` adds ``source_cohort`` / ``source_project``
        / ``n_cohort_samples``.
    canonicalize_genes
        Collapse each shard onto the shared canonical ENSG space before the
        wide outer join.  This removes same-gene rows split only by
        cross-release symbol drift.

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

    root = _representatives_root()
    canonical_baked = _artifact_has_canonical_genes(Path(root))
    available = set(available_representative_cohorts())
    if cancer_types is None:
        codes = sorted(available)
    else:
        requested = _resolve_cancer_types(cancer_types, expand_aggregates=True)
        codes = [c for c in dict.fromkeys(requested) if c in available]

    base = ["Ensembl_Gene_ID", "Symbol"]
    wide = None
    wide_parts = []
    # One canonical Symbol per ENSG, preferring a real symbol over the ENSG-as-
    # fallback that some baked shards carry (#474 review). Built while iterating
    # so the final wide frame can index on ENSG alone.
    symbol_by_gene: dict[str, str] = {}
    long_parts = []
    for code in codes:
        shard = pd.read_parquet(root / f"{code}.parquet")
        rep_cols = [c for c in shard.columns if c not in base]
        if k is not None:
            rep_cols = rep_cols[:k]
        if canonicalize_genes and not canonical_baked:
            from ..gene_canonicalization import canonicalize_gene_table
            shard = canonicalize_gene_table(
                shard,
                value_cols=rep_cols,
                source_version_col=None,
            )
        if normalize == "tpm_clean_log1p":
            shard[rep_cols] = np.log1p(shard[rep_cols].to_numpy(dtype=float))
        if format == "wide":
            part = shard[base + rep_cols]
            if canonicalize_genes:
                # Index on the canonical ENSG ONLY. The display Symbol is a
                # fallback that can differ across shards (a real symbol in one,
                # the ENSG string in another); indexing on (ENSG, Symbol) would
                # split one gene into duplicate rows. Symbol is re-attached once
                # at the end.
                for ensg, sym in zip(part["Ensembl_Gene_ID"], part["Symbol"]):
                    cur = symbol_by_gene.get(ensg)
                    if cur is None or cur == ensg:
                        symbol_by_gene[ensg] = sym
                wide_parts.append(
                    part.set_index("Ensembl_Gene_ID")[rep_cols])
            else:
                wide = part if wide is None else wide.merge(
                    part, on=base, how="outer")
        else:
            melted = shard[base + rep_cols].melt(
                id_vars=base, var_name="representative_id", value_name="expression")
            melted.insert(2, "cancer_code", code)
            long_parts.append(melted)

    if format == "wide":
        if canonicalize_genes:
            if not wide_parts:
                return pd.DataFrame(columns=base)
            combined = pd.concat(wide_parts, axis=1)
            combined.index.name = "Ensembl_Gene_ID"
            combined = combined.reset_index()
            combined.insert(
                1, "Symbol",
                combined["Ensembl_Gene_ID"].map(symbol_by_gene))
            return combined
        if wide is None:
            return pd.DataFrame(columns=base)
        return wide

    if not long_parts:
        cols = base + ["cancer_code", "representative_id", "expression"]
        return pd.DataFrame(columns=cols)
    long = pd.concat(long_parts, ignore_index=True)
    if include_provenance:
        prov_path = root / "_provenance.csv"
        if prov_path.exists():
            prov = pd.read_csv(prov_path)
            keep = ["representative_id", "source_cohort", "source_project",
                    "n_cohort_samples"]
            long = long.merge(prov[[c for c in keep if c in prov.columns]],
                              on="representative_id", how="left")
    return long


# ---------- accessors: per-gene × cohort percentile vectors (#298) ----------

_PERCENTILES_DIR = "cancer-reference-expression-percentiles"


def available_percentile_cohorts() -> list[str]:
    """Cohort codes that ship a per-gene percentile-vector shard (sorted)."""
    return _available_shard_codes(_percentiles_root())


def cohort_gene_percentiles(cancer_type, *, as_tpm: bool = True) -> pd.DataFrame:
    """Tail-weighted per-gene percentile vector for one cohort (#298).

    Returns one row per gene (``Ensembl_Gene_ID`` + ``Symbol``) with 26
    breakpoint columns — ``p0, p1, p5, p10 … p90, p95, p96, p97, p98, p99,
    p100`` — dense in the actionable upper tail. Lets a consumer place a
    sample's gene as a **percentile rank within the cohort** instead of an
    absolute TPM (the producer side of trufflepig#54).

    Computed on the **biological clean_tpm_16_9_75 view** (technical genes dropped,
    so the fixed-fraction inflation #304 doesn't apply). Stored compactly as
    ``log1p`` + float16; ``as_tpm=True`` (default) ``expm1``-restores clean-TPM
    values, ``as_tpm=False`` returns the stored log1p values. Raises if the
    cohort has no per-sample data (summary-only cohorts have no vector — their
    coarse percentiles are in :func:`cancer_reference_expression`).
    """
    from ..gene_sets_cancer import resolve_cancer_type
    code = resolve_cancer_type(cancer_type)
    shard = _percentiles_root() / f"{code}.parquet"
    if not shard.exists():
        raise ValueError(
            f"no percentile vector for {code!r} — only cohorts with per-sample "
            f"data ship one; see available_percentile_cohorts(). Summary-only "
            "cohorts expose coarse p5/p10/p90/p95 via cancer_reference_expression."
        )
    df = pd.read_parquet(shard)
    bp_cols = [c for c in df.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    df[bp_cols] = df[bp_cols].astype("float32")
    if as_tpm:
        df[bp_cols] = np.expm1(df[bp_cols])
    return df


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
          their housekeeping-panel geometric mean. Implies ``"tpm"``.
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

    df = get_data("pan-cancer-expression")
    df, _ = add_tpm_columns_from_fpkm(df)

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
