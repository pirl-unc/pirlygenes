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
    n_rows: int
    n_samples: int | None


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
        .agg(n_rows=("Ensembl_Gene_ID", "size"), n_samples=("n_samples", "max"))
        .reset_index()
    )
    cohort_rows = tuple(
        CohortRowCount(
            cancer_code=str(row.cancer_code),
            source_cohort=str(row.source_cohort),
            source_project=str(row.source_project),
            n_rows=int(row.n_rows),
            n_samples=_coerce_int(row.n_samples),
        )
        for row in grouped.sort_values(
            ["source_cohort", "cancer_code"]
        ).itertuples(index=False)
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
    lines = [
        f"cancer-reference-expression/  [{snapshot.data_source}]",
        f"  path:         {snapshot.data_path}",
        f"  size on disk: {_format_bytes(snapshot.data_size_bytes)} "
        f"across {snapshot.shard_files} per-source shards",
        f"  total rows:   {snapshot.total_rows:,}",
        f"  unique genes: {snapshot.unique_genes:,}",
        f"  cohorts:      {len(snapshot.cohort_rows)}",
        f"  registry:     {snapshot.registered_sources} sources",
        "",
        "Per-cohort row counts:",
    ]
    by_source: dict[str, list[CohortRowCount]] = {}
    for row in snapshot.cohort_rows:
        by_source.setdefault(row.source_cohort, []).append(row)
    for source_cohort in sorted(by_source):
        entries = by_source[source_cohort]
        total_rows = sum(e.n_rows for e in entries)
        lines.append(f"  {source_cohort}  ({total_rows:,} rows)")
        for entry in sorted(entries, key=lambda e: e.cancer_code):
            sample_str = (
                f"  n_samples={entry.n_samples}"
                if entry.n_samples is not None
                else "  n_samples=NaN"
            )
            lines.append(
                f"    {entry.cancer_code:<20} {entry.n_rows:>7,} rows"
                f"{sample_str}   ({entry.source_project})"
            )
    return "\n".join(lines)


__all__ = [
    "CohortRowCount",
    "InventorySnapshot",
    "summarize_inventory",
    "render_inventory",
]
