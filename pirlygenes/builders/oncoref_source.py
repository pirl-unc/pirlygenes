"""Delegate the source-matrix contract to oncoref — one entry point for every builder.

Historically each pirlygenes builder carried its own gene-ID harmonizer, which
drifted apart and re-implemented what oncoref now owns. oncoref (>=1.8.75) ships
the canonical source-matrix contract:

- ``canonicalize_source_gene_matrix`` — map any source gene id (Ensembl / HUGO /
  Entrez / transcript / synonym, mixed OK) to canonical unversioned ENSG and sum
  duplicate rows in linear space, plus a per-source-row mapping *audit*.
- ``coerce_source_expression_values`` — parse diagnostics that keep source
  *missingness* distinct from measured zero (``n_input_missing`` / ``n_parse_missing``
  / ``n_literal_zero``).
- ``sample_expression_qc_from_matrix`` — the per-sample QC manifest (detected
  genes, housekeeping floor, top-gene concentration, source-scale class), with
  RNA-seq fail gates skipped for non-linear/proxy scales (warn-only).

This module wraps those into the shapes pirlygenes builders already use
(gene-id-indexed TPM matrix → ``gene_table`` + ENSG-indexed ``values``) so a
builder swaps its local harmonizer for one call. See pirlygenes#526.

Builders share this one gene-mapping path but opt into the *optional* layers as
their input warrants: ``canonicalize_source(symbols=…)`` for HUGO rescue only
when the source carries a symbol column separate from its id (cllmap's GENCODE19),
and ``sample_qc(…)`` only where per-sample QC gating is wanted (the generic
geo-matrix build). ENSG-native single-id sources (recount3, lnen, ne) need
neither. The mapping/summing/parse-diagnostics contract is identical for all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from oncoref.expression import sample_expression_qc_from_matrix
from oncoref.expression_engine import (
    canonicalize_source_gene_matrix,
    coerce_source_expression_values,
    source_gene_mapping_stats,
)

# Read-time QC filter modes, matching oncoref's ``_SAMPLE_QC_MODES``. Summary
# stats + the per-sample parquet select samples by one of these; the persisted
# QC manifest keeps EVERY sample so a consumer can re-filter at read time.
SampleQcMode = Literal["all", "pass", "pass_or_warn"]

# oncoref source-scale classes that ARE linear-TPM comparable across cohorts;
# they resolve to linear TPM after unit normalization. An unrecognized class (a
# future microarray / rank / percentile proxy) is NON-comparable unless a source
# sets ``linear_tpm_comparable`` explicitly, so a proxy scale never silently
# inherits the RNA-seq hard-fail thresholds (oncoref#292 gates the fail checks
# behind ``linear_tpm_comparable``).
_LINEAR_SCALE_CLASSES = frozenset(
    {"linear_rnaseq_tpm", "count_derived_tpm", "log2_tpm_inverse"}
)


def default_source_scale_class(unit: str) -> str:
    """Source-scale class implied by a builder's declared input ``unit``."""
    if unit == "raw_counts":
        return "count_derived_tpm"
    if unit == "log2(TPM+1)":
        return "log2_tpm_inverse"
    return "linear_rnaseq_tpm"


def scale_class_is_linear(source_scale_class: str) -> bool:
    """True when a scale class is linear-TPM comparable (empty → assume linear)."""
    if not source_scale_class:
        return True
    return source_scale_class in _LINEAR_SCALE_CLASSES


def resolve_linear_comparable(
    source_scale_class: str, override: bool | None = None
) -> bool:
    """Effective ``linear_tpm_comparable``: explicit override, else derived."""
    if override is not None:
        return override
    return scale_class_is_linear(source_scale_class)


def source_metadata(
    *,
    unit: str = "",
    source_scale_class: str = "",
    linear_tpm_comparable: bool | None = None,
    source_cohort: str | None = None,
    source_type: str | None = None,
) -> dict[str, object]:
    """Build the ``source_metadata`` dict oncoref's sample-QC expects.

    Only ``linear_tpm_comparable`` changes QC gating; the rest are provenance
    pass-throughs. ``source_scale_class`` defaults from ``unit`` when unset.
    """
    scale = source_scale_class or (default_source_scale_class(unit) if unit else "unknown")
    linear = resolve_linear_comparable(scale, linear_tpm_comparable)
    return {
        "source_cohort": source_cohort,
        "source_type": source_type,
        "unit": unit or None,
        "source_scale_class": scale,
        "linear_tpm_comparable": linear,
        "tpm_proxy": not linear,
    }


@dataclass(frozen=True)
class CanonicalizedSource:
    """Result of delegating a source matrix's gene mapping to oncoref."""

    matrix: pd.DataFrame          # [Ensembl_Gene_ID, Symbol, *sample_cols]
    gene_table: pd.DataFrame      # [Ensembl_Gene_ID, Symbol]
    values: pd.DataFrame          # ENSG-indexed TPM (columns = sample_cols)
    sample_cols: list[str]        # uniquified sample-column names
    audit: pd.DataFrame           # per-source-row mapping audit
    mapping_stats: dict           # n_source_rows / n_resolved_rows / ...
    parse_diagnostics: pd.DataFrame  # per-value-column missing/parse/zero counts


def _uniquify(names: list[str]) -> list[str]:
    """Uniquify duplicate sample-column labels (e.g. a replicate id repeated)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _as_row_id_frame(matrix: pd.DataFrame, row_id_name: str, sample_cols: list[str]) -> pd.DataFrame:
    """A gene-id-indexed matrix → a frame with an explicit row-id column."""
    frame = matrix.copy()
    frame.columns = sample_cols
    frame = frame.reset_index()
    frame = frame.rename(columns={frame.columns[0]: row_id_name})
    frame[row_id_name] = frame[row_id_name].astype(str)
    return frame


def canonicalize_source(
    tpm_matrix: pd.DataFrame,
    *,
    row_id_name: str = "gene_id",
    high_expression_threshold: float = 1.0,
    raw_matrix: pd.DataFrame | None = None,
    symbols: "list[str] | pd.Series | None" = None,
) -> CanonicalizedSource:
    """Map a gene-id-indexed TPM matrix to canonical ENSG rows via oncoref.

    ``tpm_matrix`` is indexed by the source gene id (any supported space, mixed
    OK) with one column per sample; values are already unit-normalized TPM.
    Duplicate sample-column labels are uniquified. ``raw_matrix`` (optional, same
    orientation, PRE-numeric-coercion) is used for the parse diagnostics so
    source missingness is measured on the original strings; when omitted the
    diagnostics are computed from ``tpm_matrix`` (parse-missing will read 0).

    ``symbols`` (optional) is a per-row HUGO symbol aligned positionally to
    ``tpm_matrix`` rows; oncoref uses it to rescue rows whose id doesn't resolve
    (e.g. retired GENCODE ids with a still-current symbol). Pass a list or a
    Series whose *values* are already in row order — the index is ignored.
    """
    sample_cols = _uniquify([str(c) for c in tpm_matrix.columns])
    df = _as_row_id_frame(tpm_matrix, row_id_name, sample_cols)
    symbol_col: str | None = None
    if symbols is not None:
        values = list(symbols)
        if len(values) != len(df):
            raise ValueError(
                f"symbols length {len(values)} != matrix rows {len(df)}"
            )
        symbol_col = "__symbol__"
        df[symbol_col] = values
    matrix, audit = canonicalize_source_gene_matrix(
        df,
        row_id_col=row_id_name,
        symbol_col=symbol_col,
        value_cols=sample_cols,
        high_expression_threshold=high_expression_threshold,
    )
    mapping_stats = source_gene_mapping_stats(audit)
    # ``canonicalize_source_gene_matrix`` stashes the mapping stats + parse
    # diagnostics DataFrame in ``matrix.attrs``. Those propagate to every derived
    # frame and make pandas ``concat`` raise ("truth value of a DataFrame is
    # ambiguous") when it compares attrs. We surface both as explicit fields, so
    # drop just those two keys (not the whole attrs dict — leave any other
    # metadata a future oncoref release attaches intact) to keep frames concat-safe.
    matrix.attrs.pop("source_value_parse_diagnostics", None)
    matrix.attrs.pop("source_gene_mapping_stats", None)

    diag_source = raw_matrix if raw_matrix is not None else tpm_matrix
    diag_cols = _uniquify([str(c) for c in diag_source.columns])
    diag_frame = _as_row_id_frame(diag_source, row_id_name, diag_cols)
    _coerced, parse_diagnostics = coerce_source_expression_values(
        diag_frame, value_cols=diag_cols
    )

    gene_table = matrix[["Ensembl_Gene_ID", "Symbol"]].reset_index(drop=True)
    values = matrix.set_index("Ensembl_Gene_ID")[sample_cols]
    return CanonicalizedSource(
        matrix=matrix,
        gene_table=gene_table,
        values=values,
        sample_cols=list(sample_cols),
        audit=audit,
        mapping_stats=mapping_stats,
        parse_diagnostics=parse_diagnostics,
    )


def sample_qc(
    matrix: pd.DataFrame,
    sample_cols: list[str],
    *,
    metadata: dict[str, object],
    cancer_type: str | None = None,
    mode: SampleQcMode = "pass_or_warn",
    **thresholds: object,
) -> tuple[pd.DataFrame, list[str]]:
    """Per-sample QC manifest + the sample columns a stat build should keep.

    Returns ``(qc_manifest, kept_sample_cols)``. The manifest covers EVERY
    sample (so a consumer can re-filter at read time); ``kept_sample_cols`` is
    the subset selected by ``mode`` (``all`` / ``pass`` / ``pass_or_warn``), in
    the original column order. ``mode="all"`` keeps everything (QC audit only).
    """
    ids = [c for c in ("Ensembl_Gene_ID", "Symbol") if c in matrix.columns]
    qc = sample_expression_qc_from_matrix(
        matrix[[*ids, *sample_cols]],
        cancer_type=cancer_type,
        source_metadata=metadata,
        **thresholds,
    )
    if qc.empty or mode == "all":
        return qc, list(sample_cols)
    if mode == "pass":
        allowed = set(qc.loc[qc["sample_qc_status"] == "pass", "sample_id"].astype(str))
    else:  # pass_or_warn
        allowed = set(
            qc.loc[qc["sample_qc_status"].isin(["pass", "warn"]), "sample_id"].astype(str)
        )
    kept = [c for c in sample_cols if str(c) in allowed]
    return qc, kept
