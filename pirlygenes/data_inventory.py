"""Public inventory of the bundled cohort-level reference data.

Backs the ``pirlygenes data`` CLI surface. This module is the
read-side counterpart to :mod:`pirlygenes.downloads`:

- ``downloads`` describes what's *fetchable* (raw inputs from GDC /
  GEO / etc.) and how much disk those raw inputs occupy locally.
- ``data_inventory`` describes what's *bundled* in the installed
  pirlygenes package — the per-gene-per-cohort summary tables that
  build steps write into ``pirlygenes/data/``.

Trufflepig and ad-hoc notebooks can call :func:`summarize_inventory`
directly without going through the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .downloads import load_registry


_REFERENCE_DIR = (
    Path(__file__).parent / "data" / "cancer-reference-expression"
)


@dataclass(frozen=True)
class CohortRowCount:
    cancer_code: str
    source_cohort: str
    source_project: str
    n_rows: int
    n_samples: int | None


@dataclass(frozen=True)
class InventorySnapshot:
    bundled_path: Path
    bundled_size_bytes: int
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


def summarize_inventory() -> InventorySnapshot:
    """Snapshot the bundled cancer-reference-expression table.

    Reads every per-source shard under
    ``pirlygenes/data/cancer-reference-expression/`` (one .csv.gz per
    ``source_cohort``) and reduces to per-(cancer_code, source_cohort)
    row counts so consumers don't pay the full read on every CLI
    invocation when only the summary is needed.
    """
    from .load_dataset import get_data

    df = get_data("cancer-reference-expression")
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
        bundled_path=_REFERENCE_DIR,
        bundled_size_bytes=_shard_total_bytes(_REFERENCE_DIR),
        total_rows=int(len(df)),
        unique_genes=int(df["Ensembl_Gene_ID"].nunique(dropna=True)),
        cohort_rows=cohort_rows,
        registered_sources=len(load_registry()),
        shard_files=_shard_count(_REFERENCE_DIR),
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
        f"cancer-reference-expression/: {snapshot.bundled_path}",
        f"  size on disk: {_format_bytes(snapshot.bundled_size_bytes)} "
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
