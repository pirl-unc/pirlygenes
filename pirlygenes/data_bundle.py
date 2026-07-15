"""Lazy download of large data assets from this version's GitHub Release.

The wheel ships small curated panels directly (gene families, cancer-type
registry, key-gene panels, CTA list — total ~1 MB). The much larger
per-cohort expression summaries are downloaded on first access from the
GitHub Release matching the installed package version.

Why split: PyPI's per-file limit is 100 MiB. Bundling the full reference
data pushed the wheel to 346 MB (vs. ~5 MB without it), so we hit the
ceiling on this PR. The fix shifts the heavy assets out of the wheel
into a version-pinned downloadable tarball.

Layout:

  Bundled (in wheel, ships with pip install pirlygenes):
    pirlygenes/data/*.csv                           (small panels)
    pirlygenes/data/expression_sources.yaml         (registry)
    pirlygenes/data/cancer-reference-expression-samples.csv.gz

Downloaded (lazy, expression bundle, cached locally):
    cancer-reference-expression/*.csv.gz            (legacy manifest/status source)
    cancer-reference-expression-views/*.parquet     (precomputed canonical views)
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv

  (Per-cohort medoid representatives and per-gene percentile vectors moved
  to oncoref in pirlygenes#208 — pirlygenes re-exports the accessors but no
  longer ships or downloads those artifacts.)

Cache layout (version-pinned so upgrades trigger a re-fetch):

  ~/.cache/pirlygenes/bundled_data/v<version>/
    cancer-reference-expression/...                  (legacy manifest/status source)
    cancer-reference-expression-views/...
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv

The cache root is overridable via the ``PIRLYGENES_BUNDLED_DATA`` env
variable for offline / shared-cache setups.

Public API:

  cache_dir()      → version-pinned cache Path
  is_local()       → bool: every downloadable path present?
  fetch()          → download + extract from the GitHub Release
  ensure_local()   → fetch if missing; safe to call on every access
  find(path)       → cached path or None
  status()         → dict summarizing local state

CLI: ``pirlygenes data {fetch, status, cache-dir}`` wraps these.
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

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
    # The public expression accessor delegates to oncoref (#557). These shards
    # remain temporarily for available_cancer_expression_references(), status
    # rollups, and rebuilding the legacy canonical-view artifact.
    "cancer-reference-expression",
    "cancer-reference-expression-views",  # precomputed canonical wide views
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
    """Where the downloaded bundle lives on disk for this version."""
    override = os.environ.get("PIRLYGENES_BUNDLED_DATA")
    if override:
        return Path(override).expanduser()
    return cache_root() / f"v{DATA_VERSION}"


def is_local() -> bool:
    """Every downloadable path exists in the cache for this version."""
    root = cache_dir()
    return all((root / p).exists() for p in DOWNLOADABLE_PATHS)


def find(relative_path: str) -> Path | None:
    """Resolve a downloadable file to its on-disk cached location.

    Returns the path if cached, else None. Use :func:`ensure_local`
    first if the caller can't tolerate None.
    """
    candidate = cache_dir() / relative_path
    return candidate if candidate.exists() else None


def fetch(*, verbose: bool = True) -> Path:
    """Download + extract the bundle for this version into the cache.

    Always overwrites — safe to call to repair a corrupt cache.
    Returns the cache directory.
    """
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    if verbose:
        sys.stderr.write(
            f"pirlygenes: downloading data bundle for v{DATA_VERSION} "
            "(~350 MB, one-time)\n"
            f"  from {RELEASE_URL}\n"
            f"  to   {root}\n"
        )
        sys.stderr.flush()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(RELEASE_URL) as resp, tmp_path.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        if verbose:
            sys.stderr.write("pirlygenes: extracting...\n")
            sys.stderr.flush()
        with tarfile.open(tmp_path) as tf:
            # filter=data is Python 3.12+; fall back to the older API.
            try:
                tf.extractall(root, filter="data")
            except TypeError:
                tf.extractall(root)
    finally:
        tmp_path.unlink(missing_ok=True)
    if verbose:
        sys.stderr.write(f"pirlygenes: data bundle ready at {root}\n")
        sys.stderr.flush()
    return root


def ensure_local(*, auto_fetch: bool = True, verbose: bool = True) -> Path:
    """Make sure the bundle is present locally; download if not.

    With ``auto_fetch=False``, raises ``FileNotFoundError`` instead of
    triggering a network call — useful for read-only CLI inspection
    paths (``pirlygenes data status``) that shouldn't surprise users
    with a 340 MB download.
    """
    if is_local():
        return cache_dir()
    if not auto_fetch:
        raise FileNotFoundError(
            f"pirlygenes data bundle not found at {cache_dir()}. "
            "Run `pirlygenes data fetch` to download it."
        )
    return fetch(verbose=verbose)


def status() -> dict:
    """Snapshot of cache state — used by ``pirlygenes data status``."""
    root = cache_dir()
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
        "cache_dir": str(root),
        "release_url": RELEASE_URL,
        "items": items,
        "all_local": is_local(),
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
    """Enumerate every version-pinned cache dir under :func:`cache_root`.

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
        if not child.is_dir() or not child.name.startswith("v"):
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
