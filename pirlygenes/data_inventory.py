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

import re
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
    tumor_origin: str = ""


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


# Cross-cohort comparability classes: RNA-seq→TPM cohorts are roughly
# comparable in magnitude; microarray TPM-proxy and scRNA nTPM are NOT
# comparable to RNA-seq TPM (the most decision-relevant grouping).
def comparability_class(pipeline: str) -> tuple[str, str]:
    """Return (key, label) bucketing a cohort by cross-cohort comparability."""
    p = (pipeline or "").lower()
    if "microarray" in p:
        return ("microarray", "microarray TPM-proxy · NOT RNA-seq-comparable")
    if "pseudobulk" in p or "ntpm" in p:
        return ("scrna", "scRNA pseudobulk nTPM · NOT RNA-seq-comparable")
    return ("rnaseq", "RNA-seq → TPM · cross-cohort comparable")


def _clean_citation(source_cohort: str, source_project: str) -> str:
    """Drop the redundant '— recount3 …' suffix (the quant column covers it),
    and turn a bare 'GEO' into the study's 'Author Year' parsed from the
    source-cohort id (GSE100026_DING_2017 → 'Ding 2017')."""
    cit = re.split(r"\s+[—-]+\s+recount3", source_project)[0].strip()
    if cit.upper() in {"GEO", ""}:
        m = re.search(r"_([A-Z][A-Za-z]+)_((?:19|20)\d{2})", source_cohort)
        if m:
            return f"{m.group(1).title()} {m.group(2)}"
    return cit


def _origin_label(tumor_origin: str) -> str:
    o = (tumor_origin or "").lower()
    if o.startswith("met"):
        return "metastasis"
    if o in {"primary", ""}:
        return "primary"
    return o


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
            tumor_origin=("tumor_origin", "first"),
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
            tumor_origin=str(getattr(row, "tumor_origin", "") or ""),
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


def render_inventory(
    snapshot: InventorySnapshot,
    *,
    sort_by: str = "name",
    code_filter: str | None = None,
) -> str:
    """Render the inventory. ``sort_by`` ∈ {name, samples} orders source
    cohorts; ``code_filter`` restricts to source cohorts feeding that cancer
    code (case-insensitive)."""
    rows = list(snapshot.cohort_rows)
    if code_filter:
        wanted = code_filter.upper()
        keep = {r.source_cohort for r in rows if r.cancer_code.upper() == wanted}
        rows = [r for r in rows if r.source_cohort in keep]

    by_source: dict[str, list[CohortRowCount]] = {}
    for row in rows:
        by_source.setdefault(row.source_cohort, []).append(row)

    # ---- aggregate signal for the summary ----
    codes_to_sources: dict[str, set[str]] = {}
    class_cohorts: dict[str, set[str]] = {}
    class_samples: dict[str, int] = {}
    class_label: dict[str, str] = {}
    origin_samples: dict[str, int] = {}
    small_n = 0
    gene_counts = [r.n_rows for r in rows] or [0]
    for r in rows:
        codes_to_sources.setdefault(r.cancer_code, set()).add(r.source_cohort)
        key, label = comparability_class(r.processing_pipeline)
        class_label[key] = label
        class_cohorts.setdefault(key, set()).add(r.source_cohort)
        class_samples[key] = class_samples.get(key, 0) + (r.n_samples or 0)
        origin_samples[_origin_label(r.tumor_origin)] = (
            origin_samples.get(_origin_label(r.tumor_origin), 0) + (r.n_samples or 0)
        )
        if r.n_samples is not None and r.n_samples < 10:
            small_n += 1
    multi = sorted(c for c, s in codes_to_sources.items() if len(s) > 1)

    lines = [
        f"cancer-reference-expression  [{snapshot.data_source}]",
        f"  path:            {snapshot.data_path}",
        f"  size on disk:    {_format_bytes(snapshot.data_size_bytes).strip()} "
        f"across {snapshot.shard_files} per-source shards",
        f"  cancer codes:    {snapshot.cancer_codes:,}   "
        f"({len(multi)} fed by ≥2 sources)",
        f"  source cohorts:  {len(by_source):,}",
        f"  distinct genes:  {snapshot.unique_genes:,}   "
        f"(universe; per-cohort coverage {min(gene_counts):,}–{max(gene_counts):,})",
        f"  total samples:   {snapshot.total_samples:,}   "
        f"(summed; a sample shared by subtype cohorts is counted in each)",
        "",
        "  quantification (cross-cohort comparability):",
    ]
    for key in ("rnaseq", "microarray", "scrna"):
        if key in class_cohorts:
            n_c = len(class_cohorts[key])
            lines.append(
                f"      {class_label[key]:<46} "
                f"{n_c:>2} {'cohort ' if n_c == 1 else 'cohorts'} · "
                f"{class_samples[key]:,} samples"
            )
    _origin_order = {"primary": 0, "metastasis": 1, "mixed": 2}
    lines.append(
        "  tumor origin:    "
        + " · ".join(
            f"{o} {n:,} samples"
            for o, n in sorted(
                origin_samples.items(), key=lambda kv: (_origin_order.get(kv[0], 9), kv[0])
            )
        )
    )
    lines.append(f"  cohorts with n<10 samples: {small_n}")
    if multi and not code_filter:
        lines += [
            "",
            "Cancer codes with multiple sources (compare on the comparable scale, "
            "don't merge):",
            "  " + " · ".join(
                f"{c} ({len(codes_to_sources[c])})" for c in multi
            ),
        ]

    lines += [
        "",
        "Per source cohort   ·   one row per cancer code "
        "(n = tumor samples · genes measured · quantification).",
        "",
    ]
    if sort_by == "samples":
        order = sorted(
            by_source,
            key=lambda s: -sum(e.n_samples or 0 for e in by_source[s]),
        )
    else:
        order = sorted(by_source)
    for source_cohort in order:
        entries = sorted(by_source[source_cohort], key=lambda e: e.cancer_code)
        n_samples = sum(e.n_samples or 0 for e in entries)
        citation = _clean_citation(source_cohort, entries[0].source_project)
        origin = _origin_label(entries[0].tumor_origin)
        cohort_word = "cohort" if len(entries) == 1 else "cohorts"
        origin_str = f" · {origin}" if origin != "primary" else ""
        lines.append(
            f"  {source_cohort}   —   {len(entries)} {cohort_word} · "
            f"{n_samples:,} samples · {citation}{origin_str}"
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
