"""Reference-expression schema and pure summary-statistic helpers.

Oncoref owns source ingestion, per-sample matrices, and persisted empirical
summary rows. This module retains pirlygenes' compatibility column names and
pure in-memory statistics used by downstream consumers; it deliberately has no
writer or source-builder path.

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

``STAT_COLUMNS`` / ``CLEAN_STAT_COLUMNS`` are the canonical column-name tuples
for schema compatibility work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

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


# Valid values for the compatibility ``tumor_origin`` column. Oncoref validates
# and publishes these values at the owning data boundary.
TUMOR_ORIGIN_VALUES: frozenset[str] = frozenset({
    "primary",
    "metastasis",
    "recurrence",
    "cell_line",
    "pdx",
    "normal_tissue",
    "mixed",
})

TumorOrigin = Literal[
    "primary",
    "metastasis",
    "recurrence",
    "cell_line",
    "pdx",
    "normal_tissue",
    "mixed",
]


# Canonical compatibility column order for delegated reference-expression rows.
# The legacy 17 columns come first to preserve positional consumers; later
# extensions remain appended in release order.
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


# Availability-aware count columns for a *pooled* (cross-cohort) matrix, where
# different source cohorts measured different gene sets. ``n_available`` is the
# per-gene count that ``n_samples`` (cohort-wide constant) and ``n_detected``
# (measured AND > 0) cannot express on their own.
POOLED_COUNT_COLUMNS: tuple[str, ...] = ("n_samples", "n_available", "n_detected")


@dataclass(frozen=True)
class PooledCohorts:
    """Heterogeneity-safe pool of ragged per-cohort sample matrices.

    The single, centralized representation for cross-cohort mixing. ``values``
    spans the **union** gene set x all pooled samples; ``measured`` is a
    parallel boolean mask that is ``True`` exactly where the sample's source
    cohort carries that gene in its panel.

    Crucially the mask is built from per-cohort **row (gene) and column
    (sample) membership**, NOT from ``values.notna()``. That keeps two cases
    correct that a notna mask gets wrong:

    - a *measured-but-zero* gene (value ``0.0``) is ``measured=True``;
    - a *measured-but-dropout* gene (an intra-cohort ``NaN``) still counts in
      that cohort's ``n_available`` denominator — the cohort measured it, this
      sample just had no value — even though it's excluded from the reductions.

    Not-measured cells are ``NaN`` in ``values`` and ``False`` in ``measured``
    and are **never filled** (missing != zero). Every pooled operation routes
    through this object (``analysis_matrix`` / :meth:`counts` / :meth:`stats` /
    :meth:`summary`) so the availability semantics live in exactly one place.

    When built from **labelled** cohorts (a ``{name: matrix}`` mapping), the
    optional ``sample_cohort`` (sample-column -> cohort name) unlocks the
    per-cohort availability accounting needed to pool correctly:
    :attr:`cohort_measured` (``is_measured[gene, cohort]``),
    :attr:`n_measured_genes` (per cohort), and :attr:`n_measured_samples`
    (``[gene, cohort]`` observed-sample count — the weight for a per-gene mean).
    """

    values: pd.DataFrame
    measured: pd.DataFrame
    sample_cohort: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        # The mask can only be trusted if it shares the EXACT gene index and
        # sample columns of values, by label. This forbids ever pairing a mask
        # with a value matrix whose rows/cols are in a different order (the
        # "matched by position, not id" bug the whole design exists to avoid).
        if (self.sample_cohort is not None and len(self.values.columns)
                and not self.values.columns.equals(self.sample_cohort.index)):
            raise ValueError("sample_cohort index must match values columns")
        if not self.values.index.equals(self.measured.index):
            raise ValueError("values/measured gene index mismatch (must be "
                             "identical and identically ordered, by id)")
        if not self.values.columns.equals(self.measured.columns):
            raise ValueError("values/measured sample columns mismatch")

    @classmethod
    def from_cohorts(cls, matrices) -> "PooledCohorts":
        """Build the pool from ``(n_genes, n_samples)`` per-cohort matrices.

        ``matrices`` is either an iterable of matrices (cohorts auto-labelled
        ``cohort_0``, ``cohort_1``, …) or a ``{cohort_name: matrix}`` mapping
        (labels preserved — pass this to use the per-cohort availability views).

        Each matrix's index is its **row mask** (the genes that cohort measures)
        and its columns are its **column mask** (that cohort's samples); the
        pooled measurement mask is the OR of each cohort's ``row x column``
        block, with ``measured`` the single authority (a not-measured cell is
        ``False`` here regardless of what ``values`` holds). Sample (column)
        labels must be globally unique across inputs — prefix them by source
        cohort upstream if they collide.

        The union **gene rows are sorted canonically (lexical Ensembl id)** so
        the pool is reproducible regardless of input cohort order — there is no
        global gene order elsewhere (everything aligns by label), but a
        deterministic order keeps any persisted pooled artifact diff-stable.
        Sample **columns stay in input order** (grouped by cohort).
        """
        if isinstance(matrices, Mapping):
            items = [(str(n), m) for n, m in matrices.items()
                     if m is not None and m.shape[1] > 0]
        else:
            items = [(f"cohort_{i}", m) for i, m in enumerate(matrices)
                     if m is not None and m.shape[1] > 0]
        if not items:
            empty = pd.DataFrame()
            return cls(empty, empty.copy(), None)
        mats = [m for _, m in items]
        values = pd.concat(mats, axis=1, join="outer").sort_index()
        # Each block is all-True over its cohort's genes x samples; after an
        # outer join, a cell is present (True) iff some cohort covers it and NaN
        # otherwise, so ``.notna()`` is exactly the membership mask (bool dtype,
        # no fill/downcast).
        blocks = [pd.DataFrame(True, index=m.index, columns=m.columns)
                  for m in mats]
        measured = (
            pd.concat(blocks, axis=1, join="outer")
            .reindex(index=values.index, columns=values.columns)
            .notna()
        )
        sample_cohort = pd.Series(
            {col: name for name, m in items for col in m.columns},
        ).reindex(values.columns)
        sample_cohort.name = "cohort"
        return cls(values, measured, sample_cohort)

    @property
    def analysis_matrix(self) -> pd.DataFrame:
        """``values`` projected through the (authoritative) ``measured`` mask:
        every not-measured cell is forced to ``NaN`` so each **marginal** (per
        gene, ``axis=1``) reduction is taken only over the samples whose cohort
        measured the gene.

        For the :meth:`from_cohorts` outer-join this equals ``values`` (already
        ``NaN`` off-panel), but the projection is what makes correctness depend
        on the mask rather than on how ``values`` was filled — essential if
        ``values`` is ever a dense backing store. **Cross-sample / pairwise**
        operations (distances, correlations) must consult ``measured`` directly
        instead, since they need pairwise co-availability, not per-cell ``NaN``.
        """
        return self.values.where(self.measured)

    @property
    def gene_index(self) -> pd.Index:
        """The authoritative gene-id (Ensembl) index every result is keyed by."""
        return self.values.index

    def _require_cohorts(self) -> pd.Series:
        if self.sample_cohort is None:
            raise ValueError("per-cohort availability needs labelled cohorts — "
                             "build with PooledCohorts.from_cohorts({name: matrix})")
        return self.sample_cohort

    @property
    def cohort_measured(self) -> pd.DataFrame:
        """``is_measured[gene, cohort]`` — gene x cohort bool, ``True`` where that
        cohort's panel carries the gene (membership; independent of dropouts)."""
        sc = self._require_cohorts()
        return self.measured.T.groupby(sc, sort=False).any().T

    @property
    def n_measured_genes(self) -> pd.Series:
        """``n_measured_genes[cohort]`` — how many genes each cohort measures."""
        return self.cohort_measured.sum(axis=0)

    @property
    def n_measured_samples(self) -> pd.DataFrame:
        """``n_measured_samples[gene, cohort]`` — the count of each cohort's
        samples that **observed** the gene (non-``NaN`` after masking). This is
        the correct per-cohort, per-gene weight for combining means: a cohort's
        contribution to a gene's pooled mean is weighted by how many of its
        samples actually carried a value for that gene (dropout-aware)."""
        sc = self._require_cohorts()
        observed = self.analysis_matrix.notna()
        return observed.T.groupby(sc, sort=False).sum().T

    def counts(self) -> pd.DataFrame:
        """Gene-indexed ``{n_samples, n_available, n_detected}`` (see
        :data:`POOLED_COUNT_COLUMNS`). ``n_available`` is the per-gene measured
        (membership) count — the correct cross-cohort denominator; ``n_detected``
        counts measured-and-``> 0`` only, never treating a not-measured cell as
        a zero.
        """
        am = self.analysis_matrix
        return pd.DataFrame(
            {
                "n_samples": np.full(self.values.shape[0], self.values.shape[1],
                                     dtype=int),
                "n_available": self.measured.sum(axis=1).to_numpy(),
                "n_detected": ((am > 0) & self.measured).sum(axis=1).to_numpy(),
            },
            index=self.gene_index,
        )

    def stats(self, *, prefix: str = "TPM_") -> pd.DataFrame:
        """The :func:`compute_cohort_stats` suite as a gene-indexed DataFrame.

        Per-gene ``std`` is ``NaN`` where fewer than two samples measured the
        gene.
        """
        return pd.DataFrame(
            compute_cohort_stats(self.analysis_matrix, prefix=prefix),
            index=self.gene_index,
        )

    def summary(self, *, prefix: str = "TPM_") -> pd.DataFrame:
        """:meth:`stats` + :meth:`counts` as one **gene-indexed** DataFrame.

        The single output surface. Every column is label-aligned to
        :attr:`gene_index`, so which row is which ENSG is explicit — there is no
        bare, position-only array to mis-pair with a foreign gene list. This is
        what a future bundle-persistence step should serialise.
        """
        return pd.concat([self.stats(prefix=prefix), self.counts()], axis=1)


def align_ragged_matrices(matrices: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join ``(n_genes, n_samples)`` matrices on the gene-id index.

    Thin convenience wrapper over :meth:`PooledCohorts.from_cohorts` returning
    just the union value matrix (not-measured cells ``NaN``, never filled).
    Prefer :class:`PooledCohorts` when you also need the measurement mask.
    """
    return PooledCohorts.from_cohorts(matrices).values


def available_count_columns(values: pd.DataFrame) -> dict[str, np.ndarray]:
    """Availability counts for a single matrix where ``NaN`` means not-measured.

    Degenerate (single-panel) case of :meth:`PooledCohorts.counts`: with no
    per-cohort blocks the only mask available is ``values.notna()``. Use
    :class:`PooledCohorts` for a real cross-cohort pool, where membership and
    notna can differ (measured-but-dropout cells).
    """
    measured = values.notna()
    return {
        "n_samples": np.full(values.shape[0], values.shape[1], dtype=int),
        "n_available": measured.sum(axis=1).to_numpy(),
        "n_detected": ((values > 0) & measured).sum(axis=1).to_numpy(),
    }


def pool_cohort_samples(
    matrices: Iterable[pd.DataFrame],
    *,
    prefix: str = "TPM_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heterogeneity-safe pool of ragged per-cohort sample matrices.

    Functional front door to :class:`PooledCohorts`: a gene measured by one
    cohort but not another is summarised over just the measuring cohort's
    patients, with a per-gene ``n_available`` denominator — never imputed to
    zero, never inner-joined down to the lowest-common gene set.

    Returns ``(analysis_matrix, summary)``, both **gene-indexed** DataFrames:
    ``analysis_matrix`` is the union matrix with not-measured cells ``NaN`` and
    ``summary`` is the per-gene stat + availability-count suite.
    """
    pool = PooledCohorts.from_cohorts(matrices)
    if pool.values.empty:
        return pool.values, pd.DataFrame()
    return pool.analysis_matrix, pool.summary(prefix=prefix)


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
    "COHORT_ANNOTATION_COLUMNS",
    "TUMOR_ORIGIN_VALUES",
    "TumorOrigin",
    "REFERENCE_COLUMNS",
    "compute_cohort_stats",
    "compute_count_columns",
    "assign_stats",
    "numeric_stat_columns",
    "round_stat_columns",
]
