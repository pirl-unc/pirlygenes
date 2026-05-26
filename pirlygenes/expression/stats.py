"""Shared per-cohort summary-statistic computation.

Every ``scripts/build_*_reference_expression.py`` builder produces a
per-gene-per-cohort summary from a per-sample TPM matrix; this module
defines the exact stat suite and column names so every cohort lands
in ``cancer-reference-expression.csv.gz`` with the same shape.

Stat suite (raw and ``_clean`` companions, raw applied to the input
TPM matrix and clean applied after technical-RNA features are zeroed
and the remaining mass renormalized):

- median (50th percentile)
- q1, q3 (25th, 75th percentiles)
- p5, p10, p90, p95
- min, max
- mean, std (sample standard deviation, ``ddof=1``; NaN when
  ``n_samples < 2``)

plus ``n_samples`` (total samples in the cohort) and ``n_detected``
(samples with ``TPM > 0`` for the gene, raw matrix).

Use :func:`compute_cohort_stats` from each builder's ``_summarize``
to populate the columns; ``STAT_COLUMNS`` / ``CLEAN_STAT_COLUMNS``
are the canonical column-name tuples for schema work.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


STAT_COLUMNS: tuple[str, ...] = (
    "TPM_median",
    "TPM_q1",
    "TPM_q3",
    "TPM_mean",
    "TPM_std",
    "TPM_min",
    "TPM_max",
    "TPM_p5",
    "TPM_p10",
    "TPM_p90",
    "TPM_p95",
)

CLEAN_STAT_COLUMNS: tuple[str, ...] = tuple(
    "TPM_clean_" + col.removeprefix("TPM_") for col in STAT_COLUMNS
)

COUNT_COLUMNS: tuple[str, ...] = ("n_samples", "n_detected")


IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "Ensembl_Gene_ID",
    "Symbol",
)


PROVENANCE_COLUMNS: tuple[str, ...] = (
    "cancer_code",
    "source_cohort",
    "source_project",
    "source_version",
)


METADATA_COLUMNS: tuple[str, ...] = (
    "processing_pipeline",
    "notes",
)


# Canonical column order for cancer-reference-expression.csv.gz rows.
# Every `build_*_reference_expression.py` and
# `import_cancer_specific_expression.py` writes this exact ordering so
# the on-disk schema stays uniform across cohorts. The legacy 17
# columns (median/q1/q3/mean + clean median/q1/q3 + counts +
# metadata) come first to preserve byte-stable diffs with prior
# releases; the v5.3 extension (std/min/max/p5/p10/p90/p95 + clean
# mean + clean std/min/max/p5/p10/p90/p95) is appended.
REFERENCE_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *PROVENANCE_COLUMNS,
    "TPM_median", "TPM_q1", "TPM_q3", "TPM_mean",
    "TPM_clean_median", "TPM_clean_q1", "TPM_clean_q3",
    *COUNT_COLUMNS,
    *METADATA_COLUMNS,
    # v5.3 extension — appended so existing positional consumers keep working.
    "TPM_std", "TPM_min", "TPM_max",
    "TPM_p5", "TPM_p10", "TPM_p90", "TPM_p95",
    "TPM_clean_mean", "TPM_clean_std", "TPM_clean_min", "TPM_clean_max",
    "TPM_clean_p5", "TPM_clean_p10", "TPM_clean_p90", "TPM_clean_p95",
)


def _percentile(values: pd.DataFrame, q: float) -> np.ndarray:
    return values.quantile(q, axis=1).to_numpy()


def compute_cohort_stats(
    values: pd.DataFrame,
    *,
    prefix: str = "TPM_",
) -> dict[str, np.ndarray]:
    """Return the canonical per-gene stat suite for a sample matrix.

    ``values`` is ``(n_genes, n_samples)``, indexed by gene id. Output
    keys match ``STAT_COLUMNS`` (or, when ``prefix='TPM_clean_'``,
    ``CLEAN_STAT_COLUMNS``).

    All quantile-based stats use pandas' linear-interpolation default.
    Mean and std are computed with ``axis=1``. Std uses sample stddev
    (``ddof=1``) and is NaN when ``n_samples < 2``.
    """
    median = values.median(axis=1).to_numpy()
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    mean = values.mean(axis=1).to_numpy()
    if values.shape[1] >= 2:
        std = values.std(axis=1, ddof=1).to_numpy()
    else:
        std = np.full(values.shape[0], np.nan, dtype=float)
    minimum = values.min(axis=1).to_numpy()
    maximum = values.max(axis=1).to_numpy()
    p5 = _percentile(values, 0.05)
    p10 = _percentile(values, 0.10)
    p90 = _percentile(values, 0.90)
    p95 = _percentile(values, 0.95)
    return {
        f"{prefix}median": median,
        f"{prefix}q1": q1,
        f"{prefix}q3": q3,
        f"{prefix}mean": mean,
        f"{prefix}std": std,
        f"{prefix}min": minimum,
        f"{prefix}max": maximum,
        f"{prefix}p5": p5,
        f"{prefix}p10": p10,
        f"{prefix}p90": p90,
        f"{prefix}p95": p95,
    }


def compute_count_columns(values: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return ``{n_samples, n_detected}`` for the raw matrix.

    ``n_samples`` is constant across rows (cohort cardinality);
    ``n_detected`` counts samples with strictly positive TPM per gene.
    """
    n_samples = np.full(values.shape[0], values.shape[1], dtype=int)
    n_detected = (values > 0).sum(axis=1).to_numpy()
    return {
        "n_samples": n_samples,
        "n_detected": n_detected,
    }


def assign_stats(
    df: pd.DataFrame,
    raw_values: pd.DataFrame,
    clean_values: pd.DataFrame,
) -> pd.DataFrame:
    """Populate the full stat suite on ``df`` in place + return it.

    ``df`` must already have the gene-identifier and provenance
    columns; this helper writes every ``STAT_COLUMNS`` / ``CLEAN_STAT_COLUMNS``
    / ``COUNT_COLUMNS`` entry from ``raw_values`` / ``clean_values``.
    """
    raw_stats = compute_cohort_stats(raw_values, prefix="TPM_")
    clean_stats = compute_cohort_stats(clean_values, prefix="TPM_clean_")
    counts = compute_count_columns(raw_values)
    for key, arr in {**raw_stats, **clean_stats, **counts}.items():
        df[key] = arr
    return df


def numeric_stat_columns() -> tuple[str, ...]:
    """Every numeric stat column (raw + clean), excluding counts."""
    return STAT_COLUMNS + CLEAN_STAT_COLUMNS


def round_stat_columns(
    df: pd.DataFrame,
    *,
    decimals: int = 6,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(numeric_stat_columns())
    present = [c for c in cols if c in df.columns]
    if present:
        df[present] = df[present].round(decimals)
    return df


__all__ = [
    "STAT_COLUMNS",
    "CLEAN_STAT_COLUMNS",
    "COUNT_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "PROVENANCE_COLUMNS",
    "METADATA_COLUMNS",
    "REFERENCE_COLUMNS",
    "compute_cohort_stats",
    "compute_count_columns",
    "assign_stats",
    "numeric_stat_columns",
    "round_stat_columns",
]
