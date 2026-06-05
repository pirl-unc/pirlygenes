"""Shared Treehouse-compendium builder.

The Treehouse Childhood Cancer Initiative publishes large
pan-cancer TPM compendia (PolyA, RiboD) keyed by HUGO symbol and
encoded as ``log2(TPM + 1)``. Each release contains thousands of
samples spanning ~130 disease labels, including ~10k re-processed
TCGA samples in the PolyA release.

This module owns the per-release pipeline that every Treehouse-derived
cohort uses, and the on-disk derived cache that makes re-runs cheap:

1. Read the per-release clinical table; filter to the requested
   disease labels (and optional further per-cohort filters, e.g.
   ``th_dataset_id.startswith("TCGA")`` for the TCGA-from-Treehouse
   sweep).
2. Read the per-release log2 TPM matrix once via
   ``pandas.read_csv(..., usecols=...)``, restricted to the union of
   selected samples — bounds peak memory at
   ``n_genes × n_selected_samples`` rather than the full matrix.
3. Inverse-transform log2(TPM + 1) → TPM (clamp tiny negatives at 0).
4. Harmonize the 58,581 HUGO symbols → Ensembl release IDs via
   pyensembl. The mapping is cached per release as
   ``derived/symbol_to_ensembl_<release>.parquet``.
5. Per cohort: subset, aggregate by Ensembl_Gene_ID, technical-RNA
   zero + per-sample renormalize, compute the v5.3 stat suite, write
   per-sample matrix to ``derived/<cancer_code>_per_sample_tpm.parquet``.
6. Upsert all cohorts together into
   ``pirlygenes/data/cancer-reference-expression.csv.gz`` (one read +
   one write covering every cohort in the sweep).

Cache invalidation: ``refresh_cache=True`` blows away the symbol
mapping + per-cohort matrices and re-derives.

The thin CLI wrappers in ``scripts/sweep_treehouse_*.py`` instantiate
:class:`TreehouseRelease` + :class:`TreehouseCohort` lists and call
:func:`run_sweep`.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from ..expression.normalize import clean_tpm_matrix as _clean_tpm
from ..expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
from .gene_mapping import resolve_symbol


@dataclass(frozen=True)
class TreehouseRelease:
    """One Treehouse compendium release (PolyA, RiboD, ...).

    ``tumor_origin`` defaults to ``"mixed"`` because Treehouse
    compendia combine primary, recurrence, and metastasis samples
    without per-sample staging in their clinical table. A subclass
    that can identify primary-only subsets may override this.

    ``per_cancer_code_shards`` defaults to False; set True for any
    release that emits a multi-cancer-code shard which would otherwise
    push past GitHub's 100 MiB hard limit (motivating case: the
    TCGA-via-Treehouse sweep that landed at 99.47 MiB after the v5.4
    schema and got re-sharded per cancer_code).
    """

    source_id: str
    source_cohort: str          # cancer-reference-expression source_cohort tag
    source_project: str         # e.g. "Treehouse"
    release_label: str          # human-readable, used in source_version + notes
    tpm_filename: str
    clinical_filename: str
    cache_dir: Path
    pipeline_prefix: str        # e.g. "treehouse_polya_25_01_log2tpm_to_tpm"
    tumor_origin: str = "mixed"
    per_cancer_code_shards: bool = False

    @property
    def tpm_path(self) -> Path:
        return self.cache_dir / self.tpm_filename

    @property
    def clinical_path(self) -> Path:
        return self.cache_dir / self.clinical_filename

    @property
    def derived_dir(self) -> Path:
        return self.cache_dir / "derived"


@dataclass(frozen=True)
class TreehouseCohort:
    """One cohort to build from a Treehouse release."""

    cancer_code: str
    disease_label: str
    # Optional further filter on the clinical row dict — for example
    # the TCGA-via-Treehouse sweep filters `th_dataset_id.startswith("TCGA")`.
    sample_predicate: Callable[[dict], bool] | None = None
    # Optional extra note appended to the cohort's `notes` cell. Used
    # by the TCGA sweep to flag "TCGA-only subset of N total Treehouse
    # samples for this disease".
    extra_notes: str = ""
    # Optional override of the derived per-sample-TPM cache filename
    # stem (default: cancer_code). Useful for TCGA cohorts so the
    # cache file isn't overwritten when a non-TCGA build of the same
    # cancer_code runs later.
    cache_stem: str | None = None

    @property
    def effective_cache_stem(self) -> str:
        return self.cache_stem or self.cancer_code


def _log(msg: str) -> None:
    print(f"[treehouse {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _filter_samples(
    clinical: pd.DataFrame, cohort: TreehouseCohort,
) -> list[str]:
    norm = clinical["disease"].astype(str).str.strip().str.lower()
    mask = norm.eq(cohort.disease_label.strip().lower())
    subset = clinical.loc[mask]
    if cohort.sample_predicate is not None:
        keep = subset.apply(
            lambda row: cohort.sample_predicate(row.to_dict()),
            axis=1,
        )
        subset = subset.loc[keep]
    ids = subset["th_dataset_id"].astype(str).tolist()
    return ids


def _load_clinical_buckets(
    release: TreehouseRelease,
    cohorts: list[TreehouseCohort],
) -> dict[str, list[str]]:
    clin = pd.read_csv(release.clinical_path, sep="\t")
    buckets: dict[str, list[str]] = {}
    for cohort in cohorts:
        ids = _filter_samples(clin, cohort)
        if not ids:
            raise RuntimeError(
                f"no samples matched cohort {cohort.cancer_code!r} "
                f"(disease={cohort.disease_label!r}, "
                f"sample_predicate={cohort.sample_predicate is not None})"
            )
        buckets[cohort.cancer_code] = ids
        _log(
            f"  {cohort.cancer_code}: {len(ids):>4} samples "
            f"({cohort.disease_label})"
        )
    return buckets


def _build_or_load_symbol_mapping(
    all_symbols: pd.Index,
    *,
    ensembl_release: int,
    cache_path: Path,
    refresh: bool,
    symbol_to_entrez: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Symbol → (Ensembl_Gene_ID, Symbol) mapping table, parquet-cached.

    Every symbol goes through the shared :func:`gene_mapping.resolve_symbol`
    — direct pyensembl lookup, then synonym rescue (and the Entrez chain
    when ``symbol_to_entrez`` supplies an ID). This is the *same* resolver
    the GEO-matrix and microarray builders use, so a renamed symbol like
    ``HIST1H1T``→``H1-6`` resolves identically here instead of being
    silently dropped. Shared by the Treehouse sweep (no Entrez) and the
    microarray builder (Entrez per probe).
    """
    if cache_path.exists() and not refresh:
        _log(f"loading cached symbol mapping from {cache_path}")
        return pd.read_parquet(cache_path)
    _log(
        f"building symbol → Ensembl mapping for {len(all_symbols):,} HUGO "
        f"symbols (release {ensembl_release})..."
    )
    genome = EnsemblRelease(ensembl_release)
    entrez = symbol_to_entrez or {}
    rows: list[dict[str, str]] = []
    method_counts: Counter = Counter()
    dropped = 0
    for sym in all_symbols:
        key = str(sym).strip()
        result = resolve_symbol(genome, key, entrez_id=entrez.get(key) or None)
        if result is None:
            # Unknown or ambiguous (>1 gene for the symbol) — dropped.
            dropped += 1
            continue
        ensembl_id, name, method = result
        rows.append(
            {
                "source_symbol": key,
                "Ensembl_Gene_ID": ensembl_id,
                "Symbol": name,
            }
        )
        method_counts[method] += 1
    mapping = pd.DataFrame(rows)
    breakdown = ", ".join(f"{m}={n}" for m, n in sorted(method_counts.items()))
    _log(
        f"  resolved={sum(method_counts.values())} ({breakdown}), "
        f"unresolved/ambiguous={dropped} (dropped)"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(cache_path, index=False)
    _log(f"  wrote symbol mapping cache to {cache_path}")
    return mapping


def _read_tpm_columns(
    tpm_path: Path, sample_cols: list[str],
) -> pd.DataFrame:
    with tpm_path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
    available = set(header)
    missing = [s for s in sample_cols if s not in available]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples not in TPM header; first few: "
            f"{missing[:5]}"
        )
    gene_col = header[0]
    keep = [gene_col] + sample_cols
    _log(
        f"reading TPM matrix for {len(sample_cols):,} samples (one disk "
        f"scan)..."
    )
    df = pd.read_csv(tpm_path, sep="\t", usecols=keep, low_memory=False)
    return df.set_index(gene_col)


def _inverse_log2(log2_df: pd.DataFrame) -> pd.DataFrame:
    tpm = np.power(2.0, log2_df.to_numpy()) - 1.0
    tpm[tpm < 0] = 0.0
    return pd.DataFrame(tpm, index=log2_df.index, columns=log2_df.columns)


def _aggregate_by_ensembl(
    values_by_symbol: pd.DataFrame, mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_col = values_by_symbol.index.name or "Gene"
    values_flat = values_by_symbol.reset_index().rename(
        columns={symbol_col: "source_symbol"}
    )
    sample_cols = [c for c in values_flat.columns if c != "source_symbol"]
    merged = mapping.merge(values_flat, on="source_symbol", how="inner")
    by_gene = merged.groupby(
        "Ensembl_Gene_ID", as_index=False, sort=False,
    ).agg({"Symbol": "first", **{c: "sum" for c in sample_cols}})
    gene_table = by_gene[["Ensembl_Gene_ID", "Symbol"]].copy()
    values = by_gene.set_index("Ensembl_Gene_ID")[sample_cols]
    return gene_table, values


def _build_or_load_per_cohort_tpm(
    cohort: TreehouseCohort,
    sample_ids: list[str],
    values_full: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    derived_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = derived_dir / f"{cohort.effective_cache_stem}_per_sample_tpm.parquet"
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        gene_table = cached[["Ensembl_Gene_ID", "Symbol"]].copy()
        sample_cols = [
            c for c in cached.columns if c not in ("Ensembl_Gene_ID", "Symbol")
        ]
        values = cached.set_index("Ensembl_Gene_ID")[sample_cols]
        return gene_table, values
    cohort_log2 = values_full[sample_ids]
    cohort_tpm = _inverse_log2(cohort_log2)
    gene_table, values = _aggregate_by_ensembl(cohort_tpm, mapping)
    cached = pd.concat(
        [gene_table.reset_index(drop=True), values.reset_index(drop=True)],
        axis=1,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path, index=False)
    return gene_table, values


def _summarize_cohort(
    cohort: TreehouseCohort,
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
    *,
    release: TreehouseRelease,
    ensembl_release: int,
) -> pd.DataFrame:
    source_version = (
        f"{release.release_label}; HUGO symbols harmonized to Ensembl "
        f"release {ensembl_release}; log2(TPM+1) inverse-transformed"
    )
    pipeline = f"{release.pipeline_prefix}_ensembl{ensembl_release}_clean_tpm_v3"
    notes = (
        f"Per-sample TPMs from {release.release_label}. Sample selection: "
        f"clinical.disease == '{cohort.disease_label}'. "
        f"HUGO symbols mapped to Ensembl release {ensembl_release}; "
        f"duplicate symbol mappings dropped. TPM_clean (v2) is computed "
        f"per-sample by zeroing technical-RNA + ribosomal-protein genes and "
        f"rescaling the remaining mass to 1e6 (ribosomal proteins excluded for "
        f"cross-source comparability)."
    )
    if cohort.extra_notes:
        notes = notes + " " + cohort.extra_notes

    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cohort.cancer_code
    out["source_cohort"] = release.source_cohort
    out["source_project"] = release.source_project
    out["source_version"] = source_version
    assign_stats(out, values, clean)
    out["processing_pipeline"] = pipeline
    out["notes"] = notes
    out["tumor_origin"] = release.tumor_origin
    out["metastasis_site"] = pd.NA
    return round_stat_columns(out)[list(REFERENCE_COLUMNS)]


# Thin alias kept so the existing call site keeps its descriptive
# name; the shared implementation lives in pirlygenes.expression.stats.
_upsert_many = upsert_to_shard


def run_sweep(
    release: TreehouseRelease,
    cohorts: Iterable[TreehouseCohort],
    *,
    summary_output: Path,
    ensembl_release: int = 112,
    refresh_cache: bool = False,
) -> dict[str, int]:
    """Execute the end-to-end sweep for one release × N cohorts.

    Returns a dict of cohort_code → n_samples actually built so the CLI
    wrapper can print a summary.
    """
    cohorts = list(cohorts)
    if not release.tpm_path.exists() or not release.clinical_path.exists():
        raise RuntimeError(
            f"Missing release files under {release.cache_dir}: "
            f"tpm={release.tpm_path.exists()}, "
            f"clinical={release.clinical_path.exists()}"
        )
    release.derived_dir.mkdir(parents=True, exist_ok=True)

    _log(f"release {release.source_id}: {len(cohorts)} cohorts targeted")
    buckets = _load_clinical_buckets(release, cohorts)
    all_samples = sorted({s for ids in buckets.values() for s in ids})
    _log(
        f"  union: {len(all_samples):,} distinct samples across "
        f"{len(buckets)} cohorts"
    )

    log2_values = _read_tpm_columns(release.tpm_path, all_samples)
    _log(f"  TPM frame shape: {log2_values.shape}")

    # The "_rescued" suffix invalidates the older direct-only mapping
    # caches: this build now applies the shared synonym rescue, so a stale
    # parquet would silently re-drop the renamed symbols it now recovers.
    symbol_cache = (
        release.derived_dir / f"symbol_to_ensembl_{ensembl_release}_rescued.parquet"
    )
    mapping = _build_or_load_symbol_mapping(
        log2_values.index,
        ensembl_release=ensembl_release,
        cache_path=symbol_cache,
        refresh=refresh_cache,
    )

    per_cohort_summaries: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for cohort in cohorts:
        _log(f"== {cohort.cancer_code} ({cohort.disease_label}) ==")
        sample_ids = buckets[cohort.cancer_code]
        gene_table, values = _build_or_load_per_cohort_tpm(
            cohort,
            sample_ids,
            log2_values,
            mapping,
            derived_dir=release.derived_dir,
            refresh=refresh_cache,
        )
        summary = _summarize_cohort(
            cohort,
            gene_table,
            values,
            release=release,
            ensembl_release=ensembl_release,
        )
        _log(
            f"  {cohort.cancer_code}: {len(summary):,} gene rows ready; "
            f"per-sample cache at {cohort.effective_cache_stem}_per_sample_tpm.parquet"
        )
        per_cohort_summaries.append(summary)
        counts[cohort.cancer_code] = values.shape[1]

    combined_new = pd.concat(per_cohort_summaries, ignore_index=True)
    _log(
        f"writing {len(combined_new):,} new rows across "
        f"{len(per_cohort_summaries)} cohorts into {summary_output}..."
    )
    shard_total = _upsert_many(
        Path(summary_output),
        combined_new,
        source_cohort=release.source_cohort,
        cancer_codes=[c.cancer_code for c in cohorts],
        per_cancer_code_shards=release.per_cancer_code_shards,
    )
    if release.per_cancer_code_shards:
        _log(
            f"  {len(shard_total):,} rows written across "
            f"{len(cohorts)} per-code shards "
            f"({release.source_cohort}__<CODE>.csv.gz within {summary_output})"
        )
    else:
        _log(
            f"  {len(shard_total):,} rows in shard "
            f"{release.source_cohort}.csv.gz (within {summary_output})"
        )
    return counts


__all__ = [
    "TreehouseRelease",
    "TreehouseCohort",
    "run_sweep",
]
