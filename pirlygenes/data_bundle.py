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
    cohort-expression-matrices/...
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

import gzip
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from filelock import FileLock

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


def _cache_lock(root: Path) -> FileLock:
    """Lock the resolved destination, including callers using symlink aliases.

    Keep the lock outside the directory replaced during publication. Never
    unlink it: waiters must continue locking the same inode.
    """
    root.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root.parent / f".{root.name}.pirlygenes.lock"))


def _backup_dir(root: Path) -> Path:
    return root.parent / f".{root.name}.pirlygenes-backup"


def _recover_cache(root: Path) -> None:
    """Recover an interrupted directory swap while holding the cache lock."""
    backup = _backup_dir(root)
    if backup.exists():
        if not root.exists():
            os.replace(backup, root)
        else:
            shutil.rmtree(backup)
    # A killed downloader cannot clean its TemporaryDirectory. No writer for
    # this destination is active while we hold its lock.
    prefix = f".{root.name}.pirlygenes-stage-"
    for child in root.parent.iterdir():
        if child.name.startswith(prefix) and child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)


def _extract_bundle(archive: Path, destination: Path) -> None:
    """Validate gzip integrity, safe tar members, and required bundle content.

    Existing releases do not publish a checksum manifest. Reading through EOF
    checks the gzip CRC/trailer even when tarfile stops at its earlier end marker.
    """
    with gzip.open(archive, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or not path.parts
                or path.parts[0] not in DOWNLOADABLE_PATHS
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe or unexpected bundle member: {member.name!r}")
        # The explicit checks also protect Python 3.9/3.10 installations that
        # predate tarfile's extraction filters. Bundles contain no links/devices.
        if hasattr(tarfile, "data_filter"):
            tf.extractall(destination, members=members, filter="data")
        else:
            tf.extractall(destination, members=members)
    for relative in DOWNLOADABLE_PATHS:
        path = destination / relative
        if relative == "cohort-expression-matrices":
            complete = path.is_dir() and any(
                child.is_file() and child.stat().st_size > 0
                for child in path.rglob("*")
            )
        else:
            complete = path.is_file() and path.stat().st_size > 0
        if not complete:
            raise ValueError(f"Incomplete data bundle: {relative}")


def _publish_bundle(staged: Path, root: Path) -> None:
    """Publish a complete directory, retaining the old cache for rollback.

    A repair briefly removes the destination between renames; it never exposes
    a partly extracted bundle. A subsequent locked reader recovers the backup
    if the process dies in that interval.
    """
    backup = _backup_dir(root)
    if root.exists():
        # Custom/shared cache directories can contain other owners' data.
        # Preserve those entries, but replace our bundle roots in their entirety.
        for child in root.iterdir():
            if child.name in DOWNLOADABLE_PATHS:
                continue
            target = staged / child.name
            if child.is_symlink():
                target.symlink_to(os.readlink(child), target_is_directory=child.is_dir())
            elif child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)
        os.replace(root, backup)
    try:
        os.replace(staged, root)
    except BaseException:
        if backup.exists():
            os.replace(backup, root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _fetch_into_cache(root: Path, *, verbose: bool) -> None:
    """Download, validate, and publish while holding the destination lock."""
    if verbose:
        sys.stderr.write(
            f"pirlygenes: downloading data bundle for v{DATA_VERSION} "
            "(~52 MB, one-time)\n"
            f"  from {RELEASE_URL}\n"
            f"  to   {root}\n"
        )
        sys.stderr.flush()
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.pirlygenes-stage-", dir=root.parent,
    ) as temporary:
        tmp_path = Path(temporary) / "bundle.tar.gz"
        staged = Path(temporary) / "data"
        staged.mkdir()
        with urllib.request.urlopen(RELEASE_URL) as resp, tmp_path.open("wb") as h:
            shutil.copyfileobj(resp, h, length=1024 * 1024)
        if verbose:
            sys.stderr.write("pirlygenes: extracting...\n")
            sys.stderr.flush()
        _extract_bundle(tmp_path, staged)
        _publish_bundle(staged, root)
    if verbose:
        sys.stderr.write(f"pirlygenes: data bundle ready at {root}\n")
        sys.stderr.flush()


def fetch(*, verbose: bool = True) -> Path:
    """Explicitly download and replace the bundle, even if already cached.

    Writers are serialized. Failed downloads/extractions leave existing data
    intact, so this remains the repair path for a corrupt cache.
    """
    root = cache_dir()
    destination = root.resolve()
    with _cache_lock(destination):
        _recover_cache(destination)
        _fetch_into_cache(destination, verbose=verbose)
    return root


def ensure_local(*, auto_fetch: bool = True, verbose: bool = True) -> Path:
    """Make sure the bundle is present locally; download if not.

    With ``auto_fetch=False``, raises ``FileNotFoundError`` instead of
    triggering a network call — useful for read-only CLI inspection
    paths (``pirlygenes data status``) that shouldn't surprise users
    with a large download.
    """
    if is_local():
        return cache_dir()
    if not auto_fetch:
        raise FileNotFoundError(
            f"pirlygenes data bundle not found at {cache_dir()}. "
            "Run `pirlygenes data fetch` to download it."
        )
    root = cache_dir()
    destination = root.resolve()
    with _cache_lock(destination):
        _recover_cache(destination)
        # Another thread/process may have populated the cache while we waited.
        if not is_local():
            _fetch_into_cache(destination, verbose=verbose)
    return root


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
    """Enumerate recognizable version-pinned bundle dirs under :func:`cache_root`.

    Only real directories with a ``vMAJOR.MINOR.PATCH`` name and both historical
    HPA/pan-cancer CSVs qualify. Unrelated, incomplete, and symlinked directories
    are left alone, including beside a custom cache location.

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
        # label alone does not prove ownership, and a symlink must never make
        # an external directory a pruning target. Both CSVs are present in
        # current and historical pirlygenes data bundles.
        if (
            child.is_symlink()
            or not child.is_dir()
            or re.fullmatch(r"v\d+\.\d+\.\d+", child.name) is None
            or not all(
                (child / name).is_file() and not (child / name).is_symlink()
                for name in (
                    "pan-cancer-expression.csv", "hpa-cell-type-expression.csv",
                )
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
