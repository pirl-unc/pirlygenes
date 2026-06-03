"""Public inventory of the cohort-level reference data.

Backs the ``pirlygenes data`` CLI surface. This module is the
read-side counterpart to :mod:`pirlygenes.downloads`:

- ``downloads`` describes what's *fetchable* (raw inputs from GDC /
  GEO / etc.) and how much disk those raw inputs occupy locally.
- ``data_inventory`` describes the per-gene-per-cohort summary
  tables — wherever they currently live (bundled in the wheel for
  a git checkout, or downloaded into the cache for a pip install).

Trufflepig and ad-hoc notebooks can call :func:`summarize_inventory`
directly without going through the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data_bundle
from .downloads import load_registry


_BUNDLED_REFERENCE_DIR = (
    Path(__file__).parent / "data" / "cancer-reference-expression"
)


def _active_reference_dir() -> Path:
    """Pick whichever cancer-reference-expression dir actually has files.

    Priority order matches :func:`pirlygenes.load_dataset._shard_directories`:
      1. Bundled in-wheel / git-checkout location
      2. Downloaded cache populated by :mod:`pirlygenes.data_bundle`

    Returns the bundled path as a fallback even when neither exists
    so callers always get a sensible Path to display.
    """
    if any(_BUNDLED_REFERENCE_DIR.glob("*.csv.gz")):
        return _BUNDLED_REFERENCE_DIR
    cached = data_bundle.cache_dir() / "cancer-reference-expression"
    if any(cached.glob("*.csv.gz")):
        return cached
    return _BUNDLED_REFERENCE_DIR


@dataclass(frozen=True)
class CohortRowCount:
    cancer_code: str
    source_cohort: str
    source_project: str
    n_rows: int          # one row per measured gene → this is the gene count
    n_samples: int | None
    processing_pipeline: str = ""


@dataclass(frozen=True)
class InventorySnapshot:
    """Snapshot of the cancer-reference-expression dataset state.

    ``data_path`` points at whichever on-disk directory actually
    supplies the data (bundled or downloaded; see
    :func:`_active_reference_dir`). ``data_source`` distinguishes
    the two so consumers can tell whether the user is in a dev
    checkout or a wheel install.
    """

    data_path: Path
    data_source: str          # "bundled" | "downloaded" | "missing"
    data_size_bytes: int
    total_rows: int
    unique_genes: int
    cohort_rows: tuple[CohortRowCount, ...]
    registered_sources: int
    shard_files: int = 0
    total_samples: int = 0    # summed across cohort assignments
    cancer_codes: int = 0     # distinct cancer codes


def native_unit_from_pipeline(pipeline: str) -> str:
    """Human-readable quantification method from a shard's
    ``processing_pipeline`` provenance tag (shared by the CLI's
    ``data sources`` and ``data list``)."""
    p = (pipeline or "").lower()
    table = [
        ("microarray", "microarray intensity (TPM-proxy)"),
        ("recount3", "recount3 Gencode-v26 coverage"),
        ("gdc_star_counts", "GDC STAR counts (TPM)"),
        ("star", "STAR counts (TPM)"),
        ("log2tpm", "RSEM log2(TPM+1)"),
        ("treehouse", "RSEM log2(TPM+1)"),
        ("pseudobulk", "scRNA pseudobulk (nTPM)"),
        ("ntpm", "scRNA pseudobulk (nTPM)"),
        ("rpkm", "RPKM→TPM"),
        ("fpkm", "FPKM→TPM"),
        ("raw_counts", "raw counts→TPM"),
        ("htseq", "HTSeq counts→TPM"),
    ]
    for needle, label in table:
        if needle in p:
            return label
    return "TPM"


def _coerce_int(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shard_total_bytes(shard_dir: Path) -> int:
    total = 0
    for child in shard_dir.glob("*.csv.gz"):
        try:
            total += child.stat().st_size
        except OSError:
            continue
    for child in shard_dir.glob("*.csv"):
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _shard_count(shard_dir: Path) -> int:
    return sum(1 for _ in shard_dir.glob("*.csv.gz")) + sum(
        1 for _ in shard_dir.glob("*.csv")
    )


def _classify_path(active: Path) -> str:
    """Tag the active directory as bundled / downloaded / missing."""
    if active == _BUNDLED_REFERENCE_DIR:
        return "bundled" if active.exists() else "missing"
    return "downloaded"


def summarize_inventory() -> InventorySnapshot:
    """Snapshot the cancer-reference-expression table.

    Reads every per-source shard (.csv.gz) under whichever data
    directory currently holds the data and reduces to per-(cancer_code,
    source_cohort) row counts so consumers don't pay the full read
    on every CLI invocation when only the summary is needed.

    Triggers a one-time auto-fetch from the GitHub Release if the data
    isn't local (wheel install, fresh cache).
    """
    from .load_dataset import get_data

    df = get_data("cancer-reference-expression")
    active = _active_reference_dir()
    grouped = (
        df.groupby(
            ["cancer_code", "source_cohort", "source_project"],
            dropna=False,
            sort=False,
        )
        .agg(
            n_rows=("Ensembl_Gene_ID", "size"),
            n_samples=("n_samples", "max"),
            processing_pipeline=("processing_pipeline", "first"),
        )
        .reset_index()
    )
    cohort_rows = tuple(
        CohortRowCount(
            cancer_code=str(row.cancer_code),
            source_cohort=str(row.source_cohort),
            source_project=str(row.source_project),
            n_rows=int(row.n_rows),
            n_samples=_coerce_int(row.n_samples),
            processing_pipeline=str(getattr(row, "processing_pipeline", "") or ""),
        )
        for row in grouped.sort_values(
            ["source_cohort", "cancer_code"]
        ).itertuples(index=False)
    )
    total_samples = int(
        sum(c.n_samples for c in cohort_rows if c.n_samples is not None)
    )
    return InventorySnapshot(
        data_path=active,
        data_source=_classify_path(active),
        data_size_bytes=_shard_total_bytes(active),
        total_rows=int(len(df)),
        unique_genes=int(df["Ensembl_Gene_ID"].nunique(dropna=True)),
        cohort_rows=cohort_rows,
        registered_sources=len(load_registry()),
        shard_files=_shard_count(active),
        total_samples=total_samples,
        cancer_codes=int(df["cancer_code"].astype(str).nunique()),
    )


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:6.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def render_inventory(snapshot: InventorySnapshot) -> str:
    by_source: dict[str, list[CohortRowCount]] = {}
    for row in snapshot.cohort_rows:
        by_source.setdefault(row.source_cohort, []).append(row)

    lines = [
        f"cancer-reference-expression  [{snapshot.data_source}]",
        f"  path:            {snapshot.data_path}",
        f"  size on disk:    {_format_bytes(snapshot.data_size_bytes).strip()} "
        f"across {snapshot.shard_files} per-source shards",
        f"  cancer codes:    {snapshot.cancer_codes:,}",
        f"  source cohorts:  {len(by_source):,}",
        f"  distinct genes:  {snapshot.unique_genes:,}   "
        f"(the gene universe; each cohort measures a subset)",
        f"  total samples:   {snapshot.total_samples:,}   "
        f"(summed across cohorts; a sample shared by subtype cohorts is "
        f"counted in each)",
        "",
        "Per source cohort   ·   one row per cancer code "
        "(n = tumor samples · genes measured · quantification method).",
        "Source-cohort id encodes the origin — GSE… = GEO accession; "
        "TCGA/TARGET/MMRF/Treehouse = consortium datasets.",
        "",
    ]
    for source_cohort in sorted(by_source):
        entries = sorted(by_source[source_cohort], key=lambda e: e.cancer_code)
        n_samples = sum(e.n_samples or 0 for e in entries)
        citation = entries[0].source_project
        cohort_word = "cohort" if len(entries) == 1 else "cohorts"
        lines.append(
            f"  {source_cohort}   —   {len(entries)} {cohort_word} · "
            f"{n_samples:,} samples · {citation}"
        )
        for e in entries:
            n = "n=NaN" if e.n_samples is None else f"n={e.n_samples}"
            lines.append(
                f"      {e.cancer_code:<20} {n:<7} "
                f"{e.n_rows:>7,} genes   "
                f"{native_unit_from_pipeline(e.processing_pipeline)}"
            )
    return "\n".join(lines)


__all__ = [
    "CohortRowCount",
    "InventorySnapshot",
    "summarize_inventory",
    "render_inventory",
]
