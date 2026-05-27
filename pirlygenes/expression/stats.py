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


# Cohort-level tumor-source annotation (v5.4+).
#
# tumor_origin: one of ``primary`` | ``metastasis`` | ``recurrence``
#               | ``cell_line`` | ``pdx`` | ``normal_tissue`` | ``mixed``.
#               NaN means "unknown / not curated for this source yet";
#               new builders MUST set it explicitly. The backfill script
#               populates legacy shards from a curated source-to-origin
#               map.
#
# metastasis_site: free-text site when tumor_origin == 'metastasis'
#                  (e.g. ``liver``, ``brain``, ``lung``, ``bone``,
#                  ``lymph_node``). NaN otherwise.
#
# Use these to filter the per-gene reference matrix when a downstream
# tool needs strictly primary-tumor expression — e.g. tumor-specific
# computations should default to ``tumor_origin == 'primary'`` and fall
# back to mets only when no primary cohort exists for the cancer_code.
COHORT_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "tumor_origin",
    "metastasis_site",
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
    # v5.4 extension — primary vs metastasis annotation.
    *COHORT_ANNOTATION_COLUMNS,
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


def upsert_to_shard(
    summary_output,
    new_rows: pd.DataFrame,
    *,
    source_cohort: str,
    cancer_codes: list[str],
) -> pd.DataFrame:
    """Upsert ``new_rows`` into the per-source shard for ``source_cohort``.

    ``summary_output`` is the
    ``pirlygenes/data/cancer-reference-expression/`` directory. The
    per-source shard ``<dir>/<source_cohort>.csv.gz`` holds every row
    for that source; this function replaces the rows for every
    ``cancer_code`` in ``cancer_codes`` and preserves rows for every
    other cancer code in that source.

    Backwards-compat: if ``summary_output`` looks like the legacy
    single-file path
    (``pirlygenes/data/cancer-reference-expression.csv.gz``), the
    shard dir is derived from its parent so existing builders don't
    need to update their --summary-output default in one go.
    """
    from pathlib import Path as _Path

    out_path = _Path(str(summary_output))
    if out_path.suffix == ".gz" or out_path.is_file():
        shard_dir = out_path.parent / "cancer-reference-expression"
    else:
        shard_dir = out_path
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{source_cohort}.csv.gz"

    if shard_path.exists():
        existing = pd.read_csv(shard_path, low_memory=False)
        keep = ~existing["cancer_code"].astype(str).isin(cancer_codes)
        merged = pd.concat(
            [
                existing.loc[keep].reindex(columns=list(REFERENCE_COLUMNS)),
                new_rows,
            ],
            ignore_index=True,
        )
    else:
        merged = new_rows.copy()

    merged = merged.reindex(columns=list(REFERENCE_COLUMNS)).sort_values(
        ["cancer_code", "Ensembl_Gene_ID"], na_position="last",
    )
    merged.to_csv(shard_path, index=False, compression="gzip")
    return merged


__all__ = [
    "STAT_COLUMNS",
    "CLEAN_STAT_COLUMNS",
    "COUNT_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "PROVENANCE_COLUMNS",
    "METADATA_COLUMNS",
    "COHORT_ANNOTATION_COLUMNS",
    "REFERENCE_COLUMNS",
    "compute_cohort_stats",
    "compute_count_columns",
    "assign_stats",
    "numeric_stat_columns",
    "round_stat_columns",
    "upsert_to_shard",
]
