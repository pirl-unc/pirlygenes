"""Shared per-cohort summary-statistic computation.

Every ``scripts/build_*_reference_expression.py`` builder produces a
per-gene-per-cohort summary from a per-sample TPM matrix; this module
defines the exact stat suite and column names so every cohort lands
in ``cancer-reference-expression.csv.gz`` with the same shape.

Schema additions in v5.4 — ``tumor_origin`` (one of the values in
:data:`TUMOR_ORIGIN_VALUES`) and ``metastasis_site`` (free-text site
when ``tumor_origin == 'metastasis'``). Every builder MUST set
``tumor_origin``; :func:`upsert_to_shard` raises if it's missing or
holds an unrecognised value.

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

import warnings
from typing import Iterable, Literal

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


# Valid values for the ``tumor_origin`` column. ``upsert_to_shard``
# rejects any row whose ``tumor_origin`` falls outside this set —
# catches typos like 'metastatic' (vs 'metastasis') at write time
# rather than at downstream-analysis time. NaN is allowed only when
# explicitly opted into via ``upsert_to_shard(..., allow_unset_tumor_origin=True)``
# (used by the schema-backfill script during the v5.4 migration; new
# builders should always set it).
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


def _validate_tumor_origin(
    rows: pd.DataFrame,
    *,
    source_cohort: str,
    allow_unset: bool,
) -> None:
    """Reject rows whose ``tumor_origin`` is missing or out-of-enum.

    Catches typos like ``'metastatic'`` (vs ``'metastasis'``) and silently
    skipped ``tumor_origin = …`` assignments at write time, rather than
    at downstream-analysis time. When ``allow_unset=True``, NaN values
    pass (used by the v5.4 migration backfill); the validity check still
    runs on any non-null values.
    """
    if "tumor_origin" not in rows.columns:
        if allow_unset:
            return
        raise ValueError(
            f"upsert_to_shard({source_cohort!r}): rows are missing the "
            "'tumor_origin' column. Every builder must set this — see "
            "TUMOR_ORIGIN_VALUES for the allowed enum."
        )
    series = rows["tumor_origin"]
    null_mask = series.isna()
    if null_mask.any() and not allow_unset:
        raise ValueError(
            f"upsert_to_shard({source_cohort!r}): {int(null_mask.sum())} "
            f"of {len(rows)} rows have null tumor_origin. Set it "
            "explicitly in the builder (or pass allow_unset_tumor_origin"
            "=True if this is a legacy-data backfill)."
        )
    non_null = series.dropna().astype(str)
    invalid = sorted(set(non_null) - TUMOR_ORIGIN_VALUES)
    if invalid:
        raise ValueError(
            f"upsert_to_shard({source_cohort!r}): unrecognised "
            f"tumor_origin values {invalid}. Allowed: "
            f"{sorted(TUMOR_ORIGIN_VALUES)}."
        )


def upsert_to_shard(
    summary_output,
    new_rows: pd.DataFrame,
    *,
    source_cohort: str,
    cancer_codes: list[str],
    per_cancer_code_shards: bool = False,
    allow_unset_tumor_origin: bool = False,
) -> pd.DataFrame:
    """Upsert ``new_rows`` into the per-source shard for ``source_cohort``.

    ``summary_output`` is the
    ``pirlygenes/data/cancer-reference-expression/`` directory. By
    default the per-source shard ``<dir>/<source_cohort>.csv.gz`` holds
    every row for that source; this function replaces the rows for
    every ``cancer_code`` in ``cancer_codes`` and preserves rows for
    every other cancer code in that source.

    When ``per_cancer_code_shards=True`` the source is sharded one
    file per cancer_code at ``<dir>/<source_cohort>__<cancer_code>.csv.gz``.
    Use this for sources that span many large per-code groups and
    would otherwise push the combined file past GitHub's 100 MiB
    hard limit (Treehouse TCGA subset is the motivating case).
    The reader (``_load_shard_directory``) transparently concats all
    matching files so no consumer needs to know about the split.

    Backwards-compat: if ``summary_output`` looks like the legacy
    single-file path
    (``pirlygenes/data/cancer-reference-expression.csv.gz``), the
    shard dir is derived from its parent so existing builders don't
    need to update their --summary-output default in one go.

    Raises ``ValueError`` if ``tumor_origin`` is missing or holds a
    value outside :data:`TUMOR_ORIGIN_VALUES`. Pass
    ``allow_unset_tumor_origin=True`` to skip the non-null check (the
    v5.4 migration backfill needs this; new builders should not).
    """
    from pathlib import Path as _Path

    _validate_tumor_origin(
        new_rows,
        source_cohort=source_cohort,
        allow_unset=allow_unset_tumor_origin,
    )

    # Canonicalize cancer codes on write so a builder still emitting a
    # pre-rename code (e.g. recount3 routing → "MID_NET"/"PANNET") lands
    # under the current registry code ("NET_MIDGUT"/"NET_PANCREAS"). This
    # is the single chokepoint that keeps shards free of rename-orphan
    # rows: the cross-code upsert below then replaces the canonical rows
    # instead of leaving a stale copy under the old name. Lazy import to
    # avoid coupling the expression layer to the registry at module load.
    from ..gene_sets_cancer import canonical_cancer_code

    new_rows = new_rows.copy()
    new_rows["cancer_code"] = (
        new_rows["cancer_code"].map(canonical_cancer_code)
    )
    cancer_codes = [canonical_cancer_code(c) for c in cancer_codes]

    out_path = _Path(str(summary_output))
    if out_path.suffix == ".gz" or out_path.is_file():
        shard_dir = out_path.parent / "cancer-reference-expression"
    else:
        shard_dir = out_path
    shard_dir.mkdir(parents=True, exist_ok=True)

    new_rows = new_rows.reindex(columns=list(REFERENCE_COLUMNS))

    if per_cancer_code_shards:
        # One file per code; each file holds rows for that code only.
        # No cross-code upsert needed because there's no shared file.
        present_codes = (
            new_rows["cancer_code"].dropna().astype(str).unique().tolist()
        )

        # Defensive: catch caller typos in BOTH directions. A code in
        # cancer_codes but missing from new_rows is almost always a
        # bug (typo, dropped during filtering); a code in new_rows but
        # not in cancer_codes is usually accidental cross-contamination
        # in the input frame and worth surfacing too.
        missing = set(cancer_codes) - set(present_codes)
        if missing:
            raise ValueError(
                f"upsert_to_shard({source_cohort!r}, per_cancer_code_shards"
                f"=True): cancer_codes lists {sorted(missing)} but no "
                "rows in new_rows carry that code."
            )
        unexpected = set(present_codes) - set(cancer_codes)
        if unexpected:
            warnings.warn(
                f"upsert_to_shard({source_cohort!r}, per_cancer_code_shards"
                f"=True): new_rows contains cancer_codes {sorted(unexpected)} "
                "that were NOT listed in the cancer_codes argument. Writing "
                "them anyway, but this usually indicates accidental cross-"
                "contamination in the input frame — double-check the builder.",
                UserWarning,
                stacklevel=2,
            )

        written_frames: list[pd.DataFrame] = []
        for code in present_codes:
            code_rows = new_rows[new_rows["cancer_code"].astype(str) == code]
            code_rows = code_rows.sort_values(
                ["cancer_code", "Ensembl_Gene_ID"], na_position="last",
            )
            shard_path = shard_dir / f"{source_cohort}__{code}.csv.gz"
            code_rows.to_csv(shard_path, index=False, compression="gzip")
            written_frames.append(code_rows)
        return pd.concat(written_frames, ignore_index=True)

    shard_path = shard_dir / f"{source_cohort}.csv.gz"
    if shard_path.exists():
        existing = pd.read_csv(shard_path, low_memory=False)
        # Canonicalize the existing shard's codes too, so a stale row written
        # under a pre-rename name (e.g. an earlier build's "MID_NET") is folded
        # into the canonical namespace and matched by the (already canonical)
        # cancer_codes removal list — otherwise it survives as a rename-orphan.
        existing["cancer_code"] = existing["cancer_code"].map(canonical_cancer_code)
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


_SAMPLE_MANIFEST_SORT = ["cancer_code", "source_cohort", "sample_id"]


def upsert_samples_manifest(path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Upsert per-sample provenance rows into the shared samples manifest.

    The samples manifest
    (``pirlygenes/data/cancer-reference-expression-samples.csv.gz``) is a single
    un-sharded CSV recording which samples were included/excluded per
    ``source_cohort``. Every sample-writing builder funnels through here so the
    contract is enforced in one place (previously ~6 copy-pasted variants with
    subtly different keys and a column-stripping bug).

    Contract:
    - **Replace** all rows for every ``source_cohort`` present in ``new_rows``;
      **preserve** every other cohort's rows untouched.
    - **Union the columns** of the existing manifest and ``new_rows`` so a
      builder whose manifest carries a narrower column set never strips columns
      (e.g. ``lineage_label``) from the cohorts it does not own — the bug that
      let partial rebuilds silently corrupt foreign-cohort provenance.
    """
    from pathlib import Path as _Path

    out_path = _Path(str(path))
    new_rows = new_rows.copy()
    cohorts = set(new_rows["source_cohort"].astype(str))

    if out_path.exists():
        existing = pd.read_csv(out_path, low_memory=False)
        keep = ~existing["source_cohort"].astype(str).isin(cohorts)
        # Order-preserving union: existing columns first, then any new ones.
        cols = list(dict.fromkeys(list(existing.columns) + list(new_rows.columns)))
        out = pd.concat(
            [existing.loc[keep].reindex(columns=cols), new_rows.reindex(columns=cols)],
            ignore_index=True,
        )
    else:
        out = new_rows

    sort_cols = [c for c in _SAMPLE_MANIFEST_SORT if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


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
    "upsert_to_shard",
    "upsert_samples_manifest",
]
