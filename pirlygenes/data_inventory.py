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

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
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
    cancer_type_name: str = ""
    reference: str = ""


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
    """The NATIVE source quantification a cohort started as, from its
    ``processing_pipeline`` tag. Every cohort is normalized to clean TPM in
    the packaged data — this names the *input* unit, not the output (so no
    '→TPM'). Shared by the CLI's ``data sources`` and ``data list``."""
    p = (pipeline or "").lower()
    table = [
        ("microarray", "microarray intensity"),
        ("recount3", "recount3 coverage"),
        ("gdc_star_counts", "STAR read counts"),
        ("star", "STAR read counts"),
        ("log2tpm", "RSEM TPM"),
        ("treehouse", "RSEM TPM"),
        ("pseudobulk", "scRNA UMI counts"),
        ("ntpm", "scRNA UMI counts"),
        ("rpkm", "RPKM"),
        ("fpkm", "FPKM"),
        ("htseq", "HTSeq read counts"),
        ("raw_counts", "read counts"),
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




def _assay(source_cohort: str, pipeline: str) -> str:
    """Library prep / platform for a cohort: microarray · scRNA ·
    ribo-depleted RNA-seq · polyA RNA-seq · RNA-seq. Only claims a specific
    bulk prep (polyA / ribo-depleted) when the source explicitly encodes it
    (Treehouse PolyA vs RiboDeplete); otherwise the honest 'RNA-seq'."""
    p, sc = (pipeline or "").lower(), source_cohort.upper()
    if "microarray" in p:
        return "microarray"
    if "pseudobulk" in p or "ntpm" in p:
        return "scRNA"
    if "RIBOD" in sc:
        return "ribo-depleted RNA-seq"
    if "POLYA" in sc:
        return "polyA RNA-seq"
    return "RNA-seq"


def _origin_label(tumor_origin: str) -> str:
    o = (tumor_origin or "").lower()
    if o.startswith("met"):
        return "metastasis"
    if o == "mixed":
        return "mixed origin"
    if o in {"primary", ""}:
        return "primary"
    return o


def _reference_for(
    source_cohort: str, source_project: str, reg_by_acc: dict, reg_by_id: dict
) -> str:
    """Always return a human-meaningful reference for a source cohort.

    Looks up the authoritative registry citation — by GEO accession, or by the
    registry source-id matched as a prefix of the normalized cohort id (so
    ``TREEHOUSE_POLYA_25_01_TCGA_SUBSET`` → the ``treehouse-polya-25-01``
    citation). Prefers a rich registry citation when ``source_project`` is just
    a bare name (e.g. 'Treehouse'); keeps a descriptive multi-word
    ``source_project`` (e.g. 'CGCI Burkitt … Project') over a bare URL; always
    falls back so the reference is never empty."""
    cleaned = re.split(r"\s+[—-]+\s+recount3", source_project)[0].strip()
    m = re.match(r"(GSE\d+|[A-Z]+\d+)", source_cohort)
    reg = reg_by_acc.get(m.group(1)) if m else None
    if not reg:
        norm = source_cohort.lower().replace("_", "-")
        cands = [rid for rid in reg_by_id if norm.startswith(rid)]
        if cands:
            reg = reg_by_id[max(cands, key=len)]
    sp_rich = bool(cleaned) and cleaned.upper() != "GEO" and len(cleaned.split()) >= 2
    reg_rich = bool(reg) and not str(reg).lower().startswith("http")
    if reg_rich and not sp_rich:
        return reg
    if cleaned and cleaned.upper() != "GEO":
        return cleaned
    if reg:
        return reg
    return m.group(1) if m else source_cohort


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


# Only the columns the summary needs — reading these 7 instead of the full
# ~33-column shard is the bulk of the speedup.
_SUMMARY_COLS = (
    "Ensembl_Gene_ID", "cancer_code", "source_cohort", "source_project",
    "n_samples", "processing_pipeline", "tumor_origin",
)
_SUMMARY_CACHE = Path.home() / ".cache" / "pirlygenes" / "inventory_summary.json"
# Bump when the cached snapshot's fields/shape change, so stale caches from an
# older code version are ignored (the shard fingerprint alone wouldn't catch a
# pure-code schema change like adding the `reference` field).
_SUMMARY_SCHEMA = "3"


def _shard_signature(paths: list[Path]) -> str:
    """Fingerprint the shard set by (name, size, mtime) — plus the summary
    schema version — so the cached summary is reused only while both are
    unchanged."""
    parts = sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in paths)
    return hashlib.md5(  # noqa: S324 (non-crypto)
        (_SUMMARY_SCHEMA + repr(parts)).encode()
    ).hexdigest()


def summarize_inventory(*, progress: bool = True) -> InventorySnapshot:
    """Snapshot the cancer-reference-expression table.

    Reads only the few columns the summary needs from each per-source shard
    (with a progress bar), and caches the result keyed on a shard fingerprint
    so repeat invocations are instant. Triggers a one-time auto-fetch from the
    GitHub Release if the data isn't local.
    """
    import pandas as pd

    from . import data_bundle
    from .load_dataset import _shard_paths

    data_bundle.ensure_local()
    active = _active_reference_dir()
    paths = _shard_paths(active)

    # cheap, environment-specific fields are always recomputed live
    env = dict(
        data_path=active,
        data_source=_classify_path(active),
        data_size_bytes=_shard_total_bytes(active),
        registered_sources=len(load_registry()),
        shard_files=_shard_count(active),
    )
    sig = _shard_signature(paths)
    cached = _read_cache(sig)
    if cached is not None:
        return _snapshot_from(cached, env)

    show = progress and sys.stderr.isatty()
    if show:
        sys.stderr.write(
            f"Summarizing {len(paths)} cohort shards to build the data overview "
            "(first run only — cached afterward)…\n"
        )
        sys.stderr.flush()
    usecols = set(_SUMMARY_COLS)
    parts, gene_ids, total_rows = [], set(), 0
    bar = paths
    try:
        from tqdm import tqdm
        bar = tqdm(
            paths, desc="reading cohort summaries", unit="shard", disable=not show,
        )
    except Exception:
        pass
    for p in bar:
        sd = pd.read_csv(p, usecols=lambda c: c in usecols, low_memory=False)
        for col in _SUMMARY_COLS:
            if col not in sd.columns:
                sd[col] = None if col == "n_samples" else ""
        total_rows += len(sd)
        gene_ids.update(sd["Ensembl_Gene_ID"].astype(str).unique())
        parts.append(
            sd.groupby(
                ["cancer_code", "source_cohort", "source_project"],
                dropna=False, sort=False,
            ).agg(
                n_rows=("Ensembl_Gene_ID", "size"),
                n_samples=("n_samples", "max"),
                processing_pipeline=("processing_pipeline", "first"),
                tumor_origin=("tumor_origin", "first"),
            ).reset_index()
        )
    # a source_cohort can span shards (per_cancer_code_shards) → re-aggregate
    grouped = (
        pd.concat(parts, ignore_index=True)
        .groupby(["cancer_code", "source_cohort", "source_project"], dropna=False, sort=False)
        .agg(
            n_rows=("n_rows", "sum"),
            n_samples=("n_samples", "max"),
            processing_pipeline=("processing_pipeline", "first"),
            tumor_origin=("tumor_origin", "first"),
        )
        .reset_index()
        .sort_values(["source_cohort", "cancer_code"])
    )
    try:
        from .gene_sets_cancer import CANCER_TYPE_NAMES
        name_for = {
            c: (CANCER_TYPE_NAMES.get(c) or "")
            for c in grouped["cancer_code"].astype(str).unique()
        }
    except Exception:
        name_for = {}
    registry = load_registry()
    reg_by_acc = {s.accession: s.citation for s in registry if s.accession and s.citation}
    reg_by_id = {s.id: s.citation for s in registry if s.id and s.citation}
    cohort_rows = tuple(
        CohortRowCount(
            cancer_code=str(row.cancer_code),
            source_cohort=str(row.source_cohort),
            source_project=str(row.source_project),
            n_rows=int(row.n_rows),
            n_samples=_coerce_int(row.n_samples),
            processing_pipeline=str(getattr(row, "processing_pipeline", "") or ""),
            tumor_origin=str(getattr(row, "tumor_origin", "") or ""),
            cancer_type_name=name_for.get(str(row.cancer_code), ""),
            reference=_reference_for(
                str(row.source_cohort), str(row.source_project),
                reg_by_acc, reg_by_id,
            ),
        )
        for row in grouped.itertuples(index=False)
    )
    computed = dict(
        total_rows=int(total_rows),
        unique_genes=len(gene_ids),
        total_samples=int(
            sum(c.n_samples for c in cohort_rows if c.n_samples is not None)
        ),
        cancer_codes=int(grouped["cancer_code"].astype(str).nunique()),
        cohort_rows=[asdict(c) for c in cohort_rows],
    )
    _write_cache(sig, computed)
    return _snapshot_from(computed, env)


def _read_cache(sig: str) -> dict | None:
    try:
        payload = json.loads(_SUMMARY_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload.get("data") if payload.get("signature") == sig else None


def _write_cache(sig: str, computed: dict) -> None:
    try:
        _SUMMARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _SUMMARY_CACHE.write_text(
            json.dumps({"signature": sig, "data": computed}), encoding="utf-8"
        )
    except OSError:
        pass


def _snapshot_from(computed: dict, env: dict) -> InventorySnapshot:
    cohort_rows = tuple(CohortRowCount(**c) for c in computed["cohort_rows"])
    return InventorySnapshot(
        total_rows=computed["total_rows"],
        unique_genes=computed["unique_genes"],
        total_samples=computed["total_samples"],
        cancer_codes=computed["cancer_codes"],
        cohort_rows=cohort_rows,
        **env,
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
        "All values are normalized to clean TPM; the 'quantification' on each "
        "source header is the NATIVE source unit (read counts, microarray "
        "intensity, …), not the output.",
        "Header: cohorts · samples · assay · genes · quantification · "
        "[metastasis] · reference.",
        "Indented rows: one per cancer code — <code> <cancer type> n=<samples>.",
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
        citation = entries[0].reference
        if len(citation) > 78:                       # truncate at a word boundary
            citation = citation[:78].rsplit(" ", 1)[0] + "…"
        origin = _origin_label(entries[0].tumor_origin)
        quant = native_unit_from_pipeline(entries[0].processing_pipeline)
        assay = _assay(source_cohort, entries[0].processing_pipeline)
        gene_set = {e.n_rows for e in entries}
        genes_vary = len(gene_set) > 1
        genes_str = (
            f"{min(gene_set):,}–{max(gene_set):,}"
            if genes_vary
            else f"{next(iter(gene_set)):,}"
        )
        # header: cohorts · samples · assay · genes · quantification · [met] · reference
        parts = [
            f"{len(entries)} {'cohort' if len(entries) == 1 else 'cohorts'}",
            f"{n_samples:,} samples",
            assay,
            f"{genes_str} genes",
            quant,
        ]
        if origin == "metastasis":                   # only the notable origin
            parts.append(origin)
        if citation:
            parts.append(citation)
        lines.append(f"  {source_cohort}   —   " + " · ".join(parts))
        for e in entries:
            n = "n=NaN" if e.n_samples is None else f"n={e.n_samples}"
            extra = f"   {e.n_rows:>7,} genes" if genes_vary else ""
            n_field = f"{n:<7}" if genes_vary else n
            name = e.cancer_type_name
            if len(name) > 40:
                name = name[:39] + "…"
            lines.append(
                f"      {e.cancer_code:<18} {name:<41} {n_field}{extra}"
            )
    return "\n".join(lines)


__all__ = [
    "CohortRowCount",
    "InventorySnapshot",
    "summarize_inventory",
    "render_inventory",
]
