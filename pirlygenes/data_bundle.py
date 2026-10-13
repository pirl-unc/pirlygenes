"""Lazy download of large data assets from this version's GitHub Release.

The wheel ships small curated panels directly (gene families, cancer-type
registry, key-gene panels, CTA list — total ~1 MB). Pirlygenes' larger derived
matrices and HPA/pan-cancer matrices are downloaded on first access from the GitHub
release matching the installed data version. Empirical per-cohort summary
rows are owned and fetched by oncoref (#557), not this bundle.

Why split: PyPI's per-file limit is 100 MiB. Bundling the full reference data
formerly pushed the wheel to 346 MB (vs. ~5 MB without it). Runtime expression
ownership is now split explicitly: oncoref serves empirical rows; pirlygenes'
version-pinned tarball retains only its purpose-specific derived matrices.

Layout:

  Bundled (in wheel, ships with pip install pirlygenes):
    pirlygenes/data/*.csv                           (small curated panels)

  Owned by oncoref:
    expression source registry
    per-sample source matrices and sample-QC manifest
    empirical cancer-reference summary rows

  Downloaded (lazy, pirlygenes bundle, cached locally):
    cohort-expression-matrices/*.parquet     (precomputed canonical matrices)
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv

  (Per-cohort medoid representatives and per-gene percentile vectors moved
  to oncoref in pirlygenes#208 — pirlygenes re-exports the accessors but no
  longer ships or downloads those artifacts.)

Cache layout (data-version-pinned; code-only upgrades reuse the cache):

  ~/.cache/pirlygenes/bundled_data/v<version>/
    .datacache-archive/bundles/<installation>/
      cohort-expression-matrices/...
      pan-cancer-expression.csv
      hpa-cell-type-expression.csv

Existing files directly inside the version directory remain readable. New
installs and explicit repairs use DataCache's immutable archive bundles, so
paths returned by ``find`` and ``ensure_local`` survive later refreshes.

The cache root is overridable via the ``PIRLYGENES_BUNDLED_DATA`` env
variable for offline / shared-cache setups.

Public API:

  cache_dir()      → version-pinned cache Path
  is_local()       → bool: every downloadable path present?
  fetch()          → download + extract from the GitHub Release
  ensure_local()   → extracted data directory; fetch if missing
  find(path)       → cached path or None
  status()         → dict summarizing local state

CLI: ``pirlygenes data {fetch, status, cache-dir}`` wraps these.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import datacache

from .version import DATA_VERSION


GITHUB_REPO = "pirl-unc/pirlygenes"
# Pinned to DATA_VERSION (not __version__) so code-only package releases
# reuse the last uploaded bundle — see version.py.
TARBALL_FILENAME = f"pirlygenes-data-v{DATA_VERSION}.tar.gz"
RELEASE_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/v{DATA_VERSION}/"
    f"{TARBALL_FILENAME}"
)


# The set of names that live in the downloadable tarball (relative to
# the cache root) and are NOT bundled in the wheel. The load_dataset
# module looks here as a fallback after checking pirlygenes/data/.
DOWNLOADABLE_PATHS: tuple[str, ...] = (
    "cohort-expression-matrices",  # precomputed canonical wide matrices
    "pan-cancer-expression.csv",
    # Public hpa_cell_type_expression delegates to oncoref (#510). Keep the old
    # file only for direct get_data("hpa-cell-type-expression") compatibility.
    "hpa-cell-type-expression.csv",
)


def cache_root() -> Path:
    """Parent of all version-pinned cache dirs (``v<version>/`` lives
    inside this). Used by :func:`list_cache_versions` and
    :func:`prune_cache` to enumerate sibling versions for cleanup."""
    override = os.environ.get("PIRLYGENES_BUNDLED_DATA")
    if override:
        # Override points at the version-pinned dir; its parent is the root.
        return Path(override).expanduser().parent
    return Path.home() / ".cache" / "pirlygenes" / "bundled_data"


def cache_dir() -> Path:
    """Version cache container; resolve members with ``find``/``ensure_local``."""
    override = os.environ.get("PIRLYGENES_BUNDLED_DATA")
    if override:
        return Path(override).expanduser()
    return cache_root() / f"v{DATA_VERSION}"


def _archive_store(root: Path) -> Path:
    # Keep managed data separate from legacy files and other cache owners.
    # Resolve aliases so they share DataCache's destination lock.
    return (root / ".datacache-archive").resolve()


def _data_dir(root: Path | None = None) -> Path:
    """Resolve one immutable DataCache bundle, or the legacy cache directory."""
    root = cache_dir() if root is None else root
    state = datacache.inspect_archive(
        _archive_store(root), RELEASE_URL, verify_files=False,
    )
    if state.status == "available":
        return Path(state.bundle)
    if state.status == "inaccessible":
        raise state.error
    if state.status == "invalid":
        raise datacache.FileValidationError(
            root, "invalid bundle cache; run `pirlygenes data fetch` to repair it",
        ) from state.error
    return root


def _is_complete(root: Path) -> bool:
    for name in DOWNLOADABLE_PATHS:
        path = root / name
        if name == "cohort-expression-matrices":
            if not path.is_dir() or not any(
                child.is_file() and child.stat().st_size > 0
                for child in path.rglob("*")
            ):
                return False
        elif not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def is_local() -> bool:
    """Every downloadable path exists in the active managed or legacy bundle."""
    return _is_complete(_data_dir())


def find(relative_path: str) -> Path | None:
    """Resolve a cached member without downloading or modifying the cache."""
    candidate = _data_dir() / relative_path
    return candidate if candidate.exists() else None


def _install(*, force: bool, verbose: bool) -> Path:
    # Historical releases lack trusted hashes. DataCache still records and
    # validates observed hashes, serializes writers, and publishes atomically.
    bundle = datacache.install_archive(
        _archive_store(cache_dir()), RELEASE_URL,
        force=force, verified=False,
        download_options={"show_progress": verbose},
    )
    if not _is_complete(bundle):
        raise datacache.FileValidationError(
            bundle, "release is missing required data; run `pirlygenes data fetch` to repair it",
        )
    if verbose:
        sys.stderr.write(f"pirlygenes: data bundle ready at {bundle}\n")
    return bundle


def fetch(*, verbose: bool = True) -> Path:
    """Explicitly refresh through DataCache; existing reader paths stay valid.

    Legacy files are retained. New downloads live in a managed archive store
    inside :func:`cache_dir`; return the extracted bundle's actual directory.
    """
    return _install(force=True, verbose=verbose)


def ensure_local(*, auto_fetch: bool = True, verbose: bool = True) -> Path:
    """Reuse local data or let DataCache install the missing archive once.

    Complete legacy caches remain readable without migration or network IO.
    ``auto_fetch=False`` is read-only and raises when data are missing.
    """
    if is_local():
        return _data_dir()
    if not auto_fetch:
        raise FileNotFoundError(
            f"pirlygenes data bundle not found at {cache_dir()}. "
            "Run `pirlygenes data fetch` to download it."
        )
    return _install(force=False, verbose=verbose)


def status() -> dict:
    """Snapshot of cache state — used by ``pirlygenes data status``."""
    root = _data_dir()
    items: dict[str, dict] = {}
    for p in DOWNLOADABLE_PATHS:
        path = root / p
        size_bytes = 0
        if path.exists():
            if path.is_dir():
                size_bytes = sum(
                    (f.stat().st_size for f in path.rglob("*") if f.is_file()),
                    start=0,
                )
            else:
                size_bytes = path.stat().st_size
        items[p] = {
            "present": path.exists(),
            "path": str(path),
            "size_bytes": size_bytes,
        }
    return {
        "data_version": DATA_VERSION,
        "cache_dir": str(cache_dir()),
        "data_dir": str(root),
        "release_url": RELEASE_URL,
        "items": items,
        "all_local": _is_complete(root),
    }


def is_downloadable(relative_path: str) -> bool:
    """True if ``relative_path`` falls under one of the downloadable roots."""
    parts = Path(relative_path).parts
    if not parts:
        return False
    first = parts[0]
    return first in DOWNLOADABLE_PATHS or relative_path in DOWNLOADABLE_PATHS


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def list_cache_versions() -> list[dict]:
    """Enumerate recognizable version-pinned bundle dirs under :func:`cache_root`.

    Only real ``vMAJOR.MINOR.PATCH`` directories containing a legacy bundle or
    a DataCache archive with both HPA/pan-cancer CSVs qualify. Unrelated,
    incomplete, and symlinked directories are left alone.

    Returns a list of ``{"version", "path", "size_bytes", "is_current"}``
    dicts, sorted by version label (lexicographic). Used by
    ``pirlygenes data prune`` to identify which dirs are upgrade
    leftovers.
    """
    root = cache_root()
    if not root.exists():
        return []
    current = cache_dir()
    out: list[dict] = []
    for child in sorted(root.iterdir()):
        # An override can put us beside unrelated user directories. A version
        # label alone does not prove ownership, and symlinks must not expose
        # external directories as pruning targets.
        if (
            child.is_symlink()
            or not child.is_dir()
            or re.fullmatch(r"v\d+\.\d+\.\d+", child.name) is None
        ):
            continue
        store = child / ".datacache-archive"
        if store.is_symlink():
            continue
        state = datacache.inspect_archive(store, verify_files=False)
        if state.status not in ("available", "missing"):
            continue
        data = Path(state.bundle) if state.status == "available" else child
        if not all(
            (data / name).is_file() and not (data / name).is_symlink()
            for name in (
                "pan-cancer-expression.csv", "hpa-cell-type-expression.csv",
            )
        ):
            continue
        out.append({
            "version": child.name,
            "path": child,
            "size_bytes": _dir_size_bytes(child),
            "is_current": child.resolve() == current.resolve(),
        })
    return out


def prune_cache(*, keep_current: bool = True, dry_run: bool = False) -> list[dict]:
    """Delete every version-pinned cache dir EXCEPT the one for the
    installed version (when ``keep_current=True``). The current dir
    is always kept by default — pass ``keep_current=False`` to also
    nuke it (e.g. for a full reset).

    With ``dry_run=True`` returns the candidate-for-deletion list
    without touching disk.

    Returns the list of dirs deleted (or planned, in dry-run mode).
    Each entry is the same shape as :func:`list_cache_versions`
    output.
    """
    candidates = []
    for entry in list_cache_versions():
        if keep_current and entry["is_current"]:
            continue
        candidates.append(entry)
    if dry_run:
        return candidates
    for entry in candidates:
        # Delete files first then empty dirs. Walk bottom-up.
        path = entry["path"]
        for child in sorted(path.rglob("*"), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    child.rmdir()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass
    return candidates


__all__ = [
    "GITHUB_REPO",
    "TARBALL_FILENAME",
    "RELEASE_URL",
    "DOWNLOADABLE_PATHS",
    "cache_root",
    "cache_dir",
    "is_local",
    "find",
    "fetch",
    "ensure_local",
    "status",
    "is_downloadable",
    "list_cache_versions",
    "prune_cache",
]
