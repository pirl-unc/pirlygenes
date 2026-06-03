"""Cache layout + registry-driven inspection of expression data sources.

Backs the ``pirlygenes downloads`` CLI surface. This module owns:

- The on-disk cache convention
  (``~/.cache/pirlygenes/expression/<source_id>/``, overridable via
  the ``PIRLYGENES_CACHE`` environment variable).
- Loading the data-source registry from
  ``pirlygenes/data/expression_sources.yaml``.
- Reporting per-source disk usage so callers can group by category and
  sort by size.

Write operations (fetch, prune) are not implemented yet — see
``docs/expression-data-refresh-plan.md`` milestones 5 and 6. Calling
those CLI subcommands surfaces a ``NotImplementedError`` with a
pointer to the plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_REGISTRY_PATH = Path(__file__).parent / "data" / "expression_sources.yaml"
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
    project_id: str | None
    accession: str | None
    url: str | None
    unit: str | None
    expected_size_gb: float | None
    citation: str | None
    special_handling: str | None
    # recount3 sources: the SRA study id the builder re-quantifies, and the
    # shard tag it writes (so the registry, the builder, and the on-disk
    # source_cohort all agree). None for non-recount3 sources.
    recount3_srp: str | None = None
    source_cohort: str | None = None


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
    """Parse the expression-sources YAML registry."""
    import yaml

    path = path or _REGISTRY_PATH
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    raw_sources = payload.get("sources") or []
    out: list[ExpressionSource] = []
    for entry in raw_sources:
        out.append(
            ExpressionSource(
                id=str(entry["id"]),
                category=str(entry.get("category", "expression")),
                cancer_codes=_coerce_tuple(entry.get("cancer_codes")),
                source_type=str(entry.get("source_type", "")),
                builder=_coerce_str(entry.get("builder")),
                project_id=_coerce_str(entry.get("project_id")),
                accession=_coerce_str(entry.get("accession")),
                url=_coerce_str(entry.get("url")),
                unit=_coerce_str(entry.get("unit")),
                expected_size_gb=_coerce_float(entry.get("expected_size_gb")),
                citation=_coerce_str(entry.get("citation")),
                special_handling=_coerce_str(entry.get("special_handling")),
                recount3_srp=_coerce_str(entry.get("recount3_srp")),
                source_cohort=_coerce_str(entry.get("source_cohort")),
            )
        )
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
    sources = list(sources) if sources is not None else load_registry()
    out: list[CacheUsage] = []
    for source in sources:
        cache_dir = source_cache_dir(source.id, category=source.category)
        out.append(
            CacheUsage(
                source=source,
                on_disk_bytes=_walk_size_bytes(cache_dir),
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
    lines.append(f"Cache root: {cache_root()}")
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
