"""Compatibility inspection of oncoref-owned expression data sources.

Backs the ``pirlygenes downloads`` CLI surface. Oncoref now owns both the
source registry and per-cohort source-matrix cache. Pirlygenes retains this
module so existing callers can inspect sources without carrying a second YAML
registry or a second set of builders.

The legacy ``source_cache_dir`` helper remains for fixture/custom-source
compatibility only. Canonical matrices are fetched by
``oncoref.source_matrices``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "pirlygenes"
_CACHE_ENV_VAR = "PIRLYGENES_CACHE"


def cache_root() -> Path:
    """Return the active cache root, honoring ``PIRLYGENES_CACHE``."""
    override = os.environ.get(_CACHE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_CACHE_ROOT


def source_cache_dir(source_id: str, *, category: str = "expression") -> Path:
    """Cache subdirectory a given source should use."""
    return cache_root() / category / source_id


@dataclass(frozen=True)
class ExpressionSource:
    id: str
    category: str
    cancer_codes: tuple[str, ...]
    source_type: str
    builder: str | None
    # Extra fixed CLI args appended to the builder invocation for this source
    # (e.g. ["--only", "lps"] to select one cohort group from a multi-cohort
    # microarray builder). Empty for builders that need no source-specific args.
    builder_args: tuple[str, ...]
    project_id: str | None
    accession: str | None
    url: str | None
    unit: str | None
    expected_size_gb: float | None
    citation: str | None
    special_handling: str | None
    # recount3 provenance mirrored from oncoref's source registry: the SRA
    # study id and canonical source-cohort tag. None for non-recount3 sources.
    recount3_srp: str | None = None
    source_cohort: str | None = None
    # Curated library prep / platform (what the inventory shows as 'assay'):
    # 'polyA RNA-seq' | 'ribo-depleted RNA-seq' | 'microarray' | 'scRNA' |
    # 'RNA-seq' (prep not recorded). Only set when documented — see the YAML.
    library_prep: str | None = None
    # Package that owns source-matrix/artifact rebuilding. ``None`` means the
    # local ``builder`` remains authoritative; ``oncoref`` keeps this registry
    # entry as compatibility/discovery metadata but delegates all writes.
    # Kept last and defaulted to preserve positional/direct construction.
    build_owner: str | None = None


def _coerce_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_registry(path: Path | None = None) -> list[ExpressionSource]:
    """Return the oncoref source registry as pirlygenes compatibility rows.

    ``path`` retains the historical custom-registry hook used by tests and
    private tooling. The default never reads a pirlygenes-owned registry.
    """
    if path is None:
        from oncoref.expression_registry import expression_sources

        return [
            ExpressionSource(
                id=source.id,
                category=source.category,
                cancer_codes=source.cancer_codes,
                source_type=source.source_type,
                builder=None,
                build_owner="oncoref",
                builder_args=(),
                project_id=source.project_id,
                accession=source.accession,
                url=source.url,
                unit=source.unit,
                expected_size_gb=source.expected_size_gb,
                citation=source.citation,
                special_handling=source.special_handling,
                recount3_srp=source.recount3_srp,
                source_cohort=source.source_cohort,
                library_prep=source.library_prep,
            )
            for source in expression_sources()
        ]

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    raw_sources = payload.get("sources") or []
    out: list[ExpressionSource] = []
    for entry in raw_sources:
        source = ExpressionSource(
            id=str(entry["id"]),
            category=str(entry.get("category", "expression")),
            cancer_codes=_coerce_tuple(entry.get("cancer_codes")),
            source_type=str(entry.get("source_type", "")),
            builder=_coerce_str(entry.get("builder")),
            build_owner=_coerce_str(entry.get("build_owner")),
            builder_args=_coerce_tuple(entry.get("builder_args")),
            project_id=_coerce_str(entry.get("project_id")),
            accession=_coerce_str(entry.get("accession")),
            url=_coerce_str(entry.get("url")),
            unit=_coerce_str(entry.get("unit")),
            expected_size_gb=_coerce_float(entry.get("expected_size_gb")),
            citation=_coerce_str(entry.get("citation")),
            special_handling=_coerce_str(entry.get("special_handling")),
            recount3_srp=_coerce_str(entry.get("recount3_srp")),
            source_cohort=_coerce_str(entry.get("source_cohort")),
            library_prep=_coerce_str(entry.get("library_prep")),
        )
        if source.builder and source.build_owner:
            raise ValueError(
                f"expression source {source.id!r} cannot declare both "
                "builder and build_owner"
            )
        out.append(source)
    return out


def _walk_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


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


@dataclass(frozen=True)
class CacheUsage:
    source: ExpressionSource
    on_disk_bytes: int
    cache_dir: Path

    @property
    def on_disk_human(self) -> str:
        return _format_bytes(self.on_disk_bytes)


def collect_cache_usage(
    sources: Iterable[ExpressionSource] | None = None,
) -> list[CacheUsage]:
    """Measure each source in the cache owned by its build authority.

    Supplying ``sources`` filters the returned rows; it does not change their
    ownership. Oncoref-owned rows are always measured from the selected-matrix
    cache, while true custom/local rows retain the legacy pirlygenes cache.
    """
    sources = list(sources) if sources is not None else load_registry()
    assigned_codes: dict[str, tuple[str, ...]] = {}
    owner_sources = [
        source for source in sources if source.build_owner == "oncoref"
    ]
    if owner_sources:
        from oncoref import source_matrices

        # Determine physical ownership against the complete owner registry even
        # when the caller supplied only a filtered subset. Ownership must not
        # change with presentation filtering: otherwise routed matrices can be
        # charged to an alias or disappear merely because their physical source
        # was omitted from ``sources``.
        routing_sources_by_id = {
            source.id: source for source in load_registry()
        }
        routing_sources_by_id.update({
            source.id: source for source in owner_sources
        })
        routing_sources = list(routing_sources_by_id.values())
        resolutions = {}
        for source in routing_sources:
            try:
                resolutions[source.id] = (
                    source_matrices.resolution_for_source(source.id)
                )
            except source_matrices.SourceMatrixError:
                continue

        # A declared cancer-code route can point at a matrix physically owned
        # by another source (for example tcga-acc -> the Treehouse TCGA
        # matrix). Resolve one owner for every published matrix, preferring an
        # exact physical route and using shared cohort provenance only when the
        # owner resolver omits an umbrella/subtype code.
        physical_owner: dict[str, str] = {}
        owner_by_cohort: dict[str, str] = {}
        for source in routing_sources:
            resolution = resolutions.get(source.id)
            if (
                resolution is None
                or resolution.resolution_method != "physical_source"
            ):
                continue
            for matrix in resolution.matrices:
                owner_by_cohort.setdefault(matrix.source_cohort, source.id)
            for code in resolution.codes:
                physical_owner.setdefault(code, source.id)

        for source in routing_sources:
            resolution = resolutions.get(source.id)
            if resolution is None:
                continue
            for matrix in resolution.matrices:
                owner_by_cohort.setdefault(matrix.source_cohort, source.id)

        declared_owner = {
            code: source.id
            for source in routing_sources
            for code in source.cancer_codes
        }
        codes_by_source: dict[str, list[str]] = {
            source.id: [] for source in routing_sources
        }
        registry = source_matrices.registry()
        unassigned: list[str] = []
        for row in registry.to_dict("records"):
            code = str(row["cancer_code"])
            owner = (
                physical_owner.get(code)
                or owner_by_cohort.get(str(row["source_cohort"]))
                or declared_owner.get(code)
            )
            if owner not in codes_by_source:
                unassigned.append(code)
                continue
            codes_by_source[owner].append(code)
        if unassigned:
            raise RuntimeError(
                "oncoref source matrices have no owning source route: "
                + ", ".join(sorted(unassigned))
            )

        for source in owner_sources:
            assigned_codes[source.id] = tuple(codes_by_source[source.id])

    out: list[CacheUsage] = []
    for source in sources:
        if source.build_owner == "oncoref":
            from oncoref import source_matrices

            paths = [
                source_matrices.local_path(code)
                for code in assigned_codes.get(source.id, ())
                if source_matrices.is_cached(code)
            ]
            cache_dir = source_matrices.cache_dir()
            size = sum(_walk_size_bytes(path) for path in paths)
        else:
            cache_dir = source_cache_dir(source.id, category=source.category)
            size = _walk_size_bytes(cache_dir)
        out.append(
            CacheUsage(
                source=source,
                on_disk_bytes=size,
                cache_dir=cache_dir,
            )
        )
    return out


def render_list(usages: Iterable[CacheUsage]) -> str:
    """Render the `pirlygenes downloads list` output.

    Groups by category; within each category, sorts by on-disk size
    descending so the heaviest entries are easy to find when freeing
    space.
    """
    usages = list(usages)
    by_category: dict[str, list[CacheUsage]] = {}
    for usage in usages:
        by_category.setdefault(usage.source.category, []).append(usage)

    lines: list[str] = []
    grand_total = 0
    for category in sorted(by_category):
        entries = sorted(
            by_category[category],
            key=lambda u: (-u.on_disk_bytes, u.source.id),
        )
        category_total = sum(u.on_disk_bytes for u in entries)
        grand_total += category_total
        lines.append(
            f"== {category} ({_format_bytes(category_total)} across "
            f"{len(entries)} sources) =="
        )
        for usage in entries:
            source = usage.source
            cancer = ",".join(source.cancer_codes) or "-"
            expected = (
                f" (~{source.expected_size_gb:g} GB expected)"
                if source.expected_size_gb
                else ""
            )
            lines.append(
                f"  {usage.on_disk_human:>10}  {source.id:<28} "
                f"{source.source_type:<22} {cancer}{expected}"
            )
        lines.append("")
    lines.append(
        f"Total across {sum(len(v) for v in by_category.values())} sources: "
        f"{_format_bytes(grand_total)}"
    )
    roots = sorted({str(usage.cache_dir) for usage in usages})
    lines.append(
        f"Cache root: {roots[0]}"
        if len(roots) == 1
        else f"Cache roots: {', '.join(roots)}"
    )
    return "\n".join(lines)


__all__ = [
    "ExpressionSource",
    "CacheUsage",
    "cache_root",
    "source_cache_dir",
    "load_registry",
    "collect_cache_usage",
    "render_list",
]
