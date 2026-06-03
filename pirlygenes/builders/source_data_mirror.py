"""Mirror of small/medium build-time source data on a pinned GitHub release.

Several builders download their raw inputs from flaky or rate-limited
upstreams (NCBI GEO FTP, the Broad CLL-map host, …). To make
``pirlygenes build`` reproducible and upstream-independent — the same
motivation as the NCBI gene-data mirror in :mod:`ncbi_gene_info` — we keep a
**copy of each small/medium source file on a pinned GitHub release** and
fetch from there first, falling back to the upstream URL only if the mirror
is unavailable. Files are cached under
``~/.cache/pirlygenes/source_data/<TAG>/`` (tag-pinned, so bumping the tag
re-fetches), and CI caches that directory.

Large sources (the ~6 GB GDC STAR-count tarballs) are **not** mirrored —
they exceed sensible GitHub-release asset sizes; those stay on the local
download cache + the GDC API. This mirror is for the small/medium inputs:
the GEO heme matrices and the 78 MB CLL-map TPM TSV.

Refresh: download the upstreams, then upload them to a new dated release and
bump ``SOURCE_DATA_MIRROR_TAG`` here::

    python scripts/refresh_source_data_mirror.py --tag source-data-YYYYMMDD

Public API mirrors the recount3/ncbi helpers: :func:`fetch` returns a cached
local path, downloading from the mirror (then upstream) on a miss.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

GITHUB_REPO = "pirl-unc/pirlygenes"
SOURCE_DATA_MIRROR_TAG = "source-data-20260603"
_MIRROR_BASE = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{SOURCE_DATA_MIRROR_TAG}"
)


def cache_dir() -> Path:
    """Tag-pinned local cache dir (override via ``PIRLYGENES_SOURCE_DATA``)."""
    override = os.environ.get("PIRLYGENES_SOURCE_DATA")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "pirlygenes" / "source_data" / SOURCE_DATA_MIRROR_TAG


def mirror_url(filename: str) -> str:
    """Where ``filename`` lives on the pinned mirror release."""
    return f"{_MIRROR_BASE}/{filename}"


def _stream_to(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as h:
        shutil.copyfileobj(resp, h, length=1024 * 1024)
    tmp.replace(dest)


def fetch(
    filename: str,
    *,
    upstream_url: str,
    verbose: bool = True,
) -> Path:
    """Return a cached local path for ``filename``.

    On a cache miss, download from the GitHub mirror first; if that 404s (the
    file hasn't been mirrored yet) or errors, fall back to ``upstream_url``.
    ``filename`` is the name under which the file lives both on the mirror
    release and in the local cache.
    """
    dest = cache_dir() / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for label, url in (("mirror", mirror_url(filename)), ("upstream", upstream_url)):
        try:
            if verbose:
                sys.stderr.write(f"source-data: fetching {filename} from {label}\n")
                sys.stderr.flush()
            _stream_to(url, dest)
            return dest
        except urllib.error.HTTPError as exc:
            if label == "mirror" and exc.code == 404:
                continue  # not mirrored yet → try upstream
            if label == "upstream":
                raise
        except (urllib.error.URLError, OSError):
            if label == "upstream":
                raise
    raise RuntimeError(f"could not fetch {filename} from mirror or upstream")
