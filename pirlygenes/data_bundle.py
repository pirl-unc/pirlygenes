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

  Downloaded (lazy, ~340 MB total, cached locally):
    cancer-reference-expression/*.csv.gz            (per-cohort summaries)
    pan-cancer-expression.csv
    hpa-cell-type-expression.csv

Cache layout (version-pinned so upgrades trigger a re-fetch):

  ~/.cache/pirlygenes/bundled_data/v<version>/
    cancer-reference-expression/...
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

from .version import __version__


GITHUB_REPO = "pirl-unc/pirlygenes"
TARBALL_FILENAME = f"pirlygenes-data-v{__version__}.tar.gz"
RELEASE_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/download/v{__version__}/"
    f"{TARBALL_FILENAME}"
)


# The set of names that live in the downloadable tarball (relative to
# the cache root) and are NOT bundled in the wheel. The load_dataset
# module looks here as a fallback after checking pirlygenes/data/.
DOWNLOADABLE_PATHS: tuple[str, ...] = (
    "cancer-reference-expression",     # directory of per-source shards
    "pan-cancer-expression.csv",
    "hpa-cell-type-expression.csv",
)


def cache_dir() -> Path:
    """Where the downloaded bundle lives on disk for this version."""
    override = os.environ.get("PIRLYGENES_BUNDLED_DATA")
    if override:
        return Path(override).expanduser()
    return (
        Path.home() / ".cache" / "pirlygenes" / "bundled_data"
        / f"v{__version__}"
    )


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
            f"pirlygenes: downloading data bundle for v{__version__} "
            "(~340 MB, one-time)\n"
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
        "version": __version__,
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


__all__ = [
    "GITHUB_REPO",
    "TARBALL_FILENAME",
    "RELEASE_URL",
    "DOWNLOADABLE_PATHS",
    "cache_dir",
    "is_local",
    "find",
    "fetch",
    "ensure_local",
    "status",
    "is_downloadable",
]
