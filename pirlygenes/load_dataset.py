# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loader for bundled + downloaded data files.

Two data roots are checked in order:

  1. ``_BUNDLED_DATA_DIR`` — files shipped in the wheel (small panels,
     registries) AND files present in a git-checkout's
     ``pirlygenes/data/`` (the dev workflow keeps everything here).
  2. ``_DOWNLOADED_DATA_DIR`` — the cache populated by
     :mod:`pirlygenes.data_bundle` (large per-cohort summaries fetched
     from the GitHub Release matching the installed version).

Any file present in (1) wins over (2) — this keeps dev iteration on
the cancer-reference-expression shards working without forcing a re-
download. In a fresh wheel install, (1) will only have the small
panels and (2) supplies the heavy data.

When a callable here requests one of the
:data:`pirlygenes.data_bundle.DOWNLOADABLE_PATHS` items and it's
missing from both roots, :func:`pirlygenes.data_bundle.ensure_local`
triggers a one-time download from the GitHub Release.
"""

from pathlib import Path

import pandas as pd

from . import data_bundle

_BUNDLED_DATA_DIR = Path(__file__).parent / "data"
_DOWNLOADED_DATA_DIR = data_bundle.cache_dir()
_DATASET_PATHS = None
_CACHED_DATAFRAMES = {}


# Back-compat alias — many call sites still import _DATA_DIR.
_DATA_DIR = _BUNDLED_DATA_DIR


def _data_roots() -> list[Path]:
    """Roots checked when resolving a data file, in priority order."""
    return [_BUNDLED_DATA_DIR, _DOWNLOADED_DATA_DIR]


def _ensure_downloadable(name: str) -> None:
    """If ``name`` (a file or dir basename) maps to a downloadable
    bundle item missing from BOTH the bundled checkout AND the cache,
    fetch it. No-op when the file/dir exists in either location.

    Why we check the bundled path first: in a git-dev checkout the
    full data is sitting at ``pirlygenes/data/<name>/`` already and
    triggering a fetch from the GitHub Release that doesn't exist yet
    (when version bumps before release-creation) breaks test.sh."""
    stem_with_csv = name if name.endswith(".csv") else f"{name}.csv"
    stem = name.removesuffix(".csv").removesuffix(".gz")
    candidates = {name, stem, stem_with_csv, stem_with_csv.removesuffix(".csv")}
    for cand in candidates:
        if not data_bundle.is_downloadable(cand):
            continue
        # Bundled-checkout fast path: dev pip-installed --editable
        # or in-repo work has the file at pirlygenes/data/<cand>.
        if (_BUNDLED_DATA_DIR / cand).exists():
            return
        # Already cached — fast path hot.
        if data_bundle.find(cand) is not None:
            return
        # Neither bundled nor cached → fetch from this version's release.
        data_bundle.ensure_local()
        return


def _shard_directories() -> list[Path]:
    """Subdirectories holding sharded CSV datasets, gathered from both
    the bundled and downloaded data roots.

    A shard directory ``<root>/<name>/`` containing one or more
    ``*.csv.gz`` files acts as a single logical dataset addressable
    as ``<name>`` via :func:`get_data` — its shards are loaded and
    concatenated transparently. Used to keep individual file sizes
    under GitHub's 100 MB push limit; cancer-reference-expression is
    sharded per ``source_cohort``.
    """
    seen: dict[str, Path] = {}
    for root in _data_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if any(child.glob("*.csv")) or any(child.glob("*.csv.gz")):
                # Bundled root wins over downloaded root on name conflicts.
                seen.setdefault(child.name, child)
    return [seen[name] for name in sorted(seen)]


def _shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(list(shard_dir.glob("*.csv")) + list(shard_dir.glob("*.csv.gz")))


def get_all_csv_paths() -> list:
    """Paths to every top-level CSV file across both data roots.

    Picks up both plain ``.csv`` and gzipped ``.csv.gz`` files. Sharded
    datasets (see :func:`_shard_directories`) are not enumerated here —
    they are loaded as a single logical CSV by :func:`get_data`.

    On name conflicts, the bundled root wins over the downloaded root.
    """
    seen: dict[str, Path] = {}
    for root in _data_roots():
        if not root.exists():
            continue
        for p in sorted(list(root.glob("*.csv")) + list(root.glob("*.csv.gz"))):
            seen.setdefault(p.name, p)
    return list(seen.values())


# Pure-text provenance columns: a handful of distinct values (long strings)
# repeated across every gene row — the ~130 distinct `notes` strings alone span
# 4.8M rows (3.3 GB as object). Stored as `object` the concatenated
# cancer-reference-expression frame is ~8 GB and its cached-parquet read ~10 s;
# casting just these three to `category` drops it to ~3 GB and ~1 s — paid once
# per process (every xdist worker, script and plot run).
#
# A `category` is only a codes+dictionary *encoding* of the same strings: the
# values, NaNs, ==, .str and groupby results are byte-identical to the object
# column (asserted element-wise on representative data in test_load_dataset).
# Only operations that
# introduce a NEW category at the array level (.loc/.iloc/.at setitem, .fillna,
# .replace, reindex-fill) raise on a categorical — so this is deliberately
# limited to columns that are pure display/provenance and never computed on:
# the cohort-availability code reindex-/fillna-s on source_cohort /
# source_project / tumor_origin / cancer_code (a categorical's "no new category"
# rule would break those), so they stay object.
#
# NOTE the pooling path does `g["processing_pipeline"] = "pooled_n_weighted"`,
# a value not among the categories. That stays safe ONLY because whole-column
# scalar assignment REPLACES the column (-> a fresh object column), not an
# in-place categorical setitem; do not switch it to .loc/.fillna on these cols.
_LOW_CARDINALITY_METADATA_COLS = (
    "source_version", "processing_pipeline", "notes",
)
# Bump when the cached dtype scheme changes so stale object-dtype parquets in an
# existing ``~/.cache/pirlygenes/shard_cache/`` rebuild instead of being reused.
_SHARD_CACHE_FORMAT = 3


def _categorize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the low-cardinality provenance columns to ``category`` in place and
    return ``df`` (returned for convenient chaining at the call sites)."""
    for col in _LOW_CARDINALITY_METADATA_COLS:
        if col in df.columns and not isinstance(
                df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    return df


def _load_shard_directory(shard_dir: Path) -> pd.DataFrame:
    """Concatenate every ``*.csv[.gz]`` shard in a sharded dataset directory.

    Parsing ~70 gzipped CSVs into a multi-million-row frame is the slowest single
    step in the test suite, and a fresh process repeats it every run. So we keep a
    best-effort **parquet cache** of the concatenated frame in
    ``~/.cache/pirlygenes/shard_cache/``, keyed on a signature of the shard
    files (count + total size + newest mtime) and the cache format. A subsequent
    run reads one parquet (~7-10x faster) instead of re-parsing every gzip; the
    cache auto-invalidates when any shard changes, and any cache error silently
    falls back to the CSVs. Low-cardinality provenance columns are stored as
    ``category`` (see :data:`_LOW_CARDINALITY_METADATA_COLS`).
    """
    paths = _shard_paths(shard_dir)
    if not paths:
        raise FileNotFoundError(f"no CSV shards found under {shard_dir}")
    sig = repr((_SHARD_CACHE_FORMAT, len(paths),
                sum(p.stat().st_size for p in paths),
                max(p.stat().st_mtime_ns for p in paths)))
    cache_dir = Path.home() / ".cache" / "pirlygenes" / "shard_cache"
    cache_file = cache_dir / f"{shard_dir.name}.parquet"
    sig_file = cache_dir / f"{shard_dir.name}.sig"
    try:
        if (cache_file.exists() and sig_file.exists()
                and sig_file.read_text() == sig):
            return _categorize_metadata(pd.read_parquet(cache_file))
    except Exception:
        pass  # any cache-read problem -> rebuild from the authoritative CSVs
    df = pd.concat([pd.read_csv(str(p), low_memory=False) for p in paths],
                   ignore_index=True)
    df = _categorize_metadata(df)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        sig_file.write_text(sig)
    except Exception:
        pass  # caching is best-effort; never fail the load on a write error
    return df


def load_all_dataframes():
    """
    Generator that yields pairs of (csv_name, df) for all CSV files in the
    data directory. Gzipped files are transparently decompressed and keyed
    under their underlying ``.csv`` name so callers don't need to know
    the on-disk compression format. Sharded directories yield once as the
    full concatenated frame, keyed under ``<dirname>.csv``.

    Triggers a one-time download of any heavy data-bundle item that is missing
    from both the checkout/wheel data directory and the local cache.
    """
    for path in data_bundle.DOWNLOADABLE_PATHS:
        _ensure_downloadable(path)
    _invalidate_dataset_paths()
    for csv_path in get_all_csv_paths():
        df = pd.read_csv(str(csv_path), low_memory=False)
        key = csv_path.name.removesuffix(".gz")
        yield key, df
    for shard_dir in _shard_directories():
        yield f"{shard_dir.name}.csv", _load_shard_directory(shard_dir)


def load_all_dataframes_dict():
    """
    Dictionary of csv_file -> df for all CSV files in the data directory
    """
    return {csv_file: df for csv_file, df in load_all_dataframes()}


def _invalidate_dataset_paths() -> None:
    global _DATASET_PATHS
    _DATASET_PATHS = None


def _dataset_paths():
    """Map accepted dataset names to their on-disk CSV path or shard dir."""
    global _DATASET_PATHS
    if _DATASET_PATHS is not None:
        return _DATASET_PATHS

    paths: dict[str, Path] = {}
    for csv_path in get_all_csv_paths():
        csv_key = csv_path.name.removesuffix(".gz")
        stem_key = csv_key.removesuffix(".csv")
        for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
            paths[key] = csv_path
    for shard_dir in _shard_directories():
        stem_key = shard_dir.name
        csv_key = stem_key + ".csv"
        for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
            paths[key] = shard_dir
    _DATASET_PATHS = paths
    return paths


def get_data(name, _dataframes_dict=None, *, copy=True):
    """Load a packaged dataset as a DataFrame.

    By default returns a defensive ``.copy()`` so callers that mutate in
    place (``df["c"] = ...``, ``df.fillna(0, inplace=True)``) can't corrupt
    the shared cache. Pass ``copy=False`` to get the cached frame directly —
    only for read-only callers that filter/copy a small slice before any
    mutation. This skips the full-frame copy, which for the large
    ``cancer-reference-expression`` table (~367 MB, ~1M string rows)
    dominated test-suite time (#278).
    """
    candidates = [name, name.lower()]
    for candidate in list(candidates):
        candidates.append(candidate + ".csv")

    if _dataframes_dict is None:
        # Trigger download for downloadable items before resolving paths,
        # so the _dataset_paths cache sees the newly-extracted files.
        _ensure_downloadable(name)
        paths = _dataset_paths()

        # If the lookup misses but the requested name is a downloadable
        # item, force a path-cache rebuild after the fetch and retry once.
        miss = not any(c in paths for c in candidates)
        if miss and data_bundle.is_downloadable(name):
            data_bundle.ensure_local()
            _invalidate_dataset_paths()
            paths = _dataset_paths()

        for candidate in candidates:
            if candidate in paths:
                resolved = paths[candidate]
                if resolved.is_dir():
                    cache_key = resolved.name + ".csv"
                    if cache_key not in _CACHED_DATAFRAMES:
                        _CACHED_DATAFRAMES[cache_key] = _load_shard_directory(resolved)
                else:
                    cache_key = resolved.name.removesuffix(".gz")
                    if cache_key not in _CACHED_DATAFRAMES:
                        _CACHED_DATAFRAMES[cache_key] = pd.read_csv(
                            str(resolved), low_memory=False
                        )
                # Return a copy so callers that mutate in place (e.g. df["c"]=...,
                # df.fillna(0, inplace=True)) can't corrupt the shared cache.
                # copy=False skips this for read-only callers (#278).
                cached = _CACHED_DATAFRAMES[cache_key]
                return cached.copy() if copy else cached
        raise ValueError(f"Dataset {name} not found")

    for candidate in candidates:
        if candidate in _dataframes_dict:
            # Return a copy so callers that mutate in place (e.g. df["c"]=...,
            # df.fillna(0, inplace=True)) can't corrupt the shared cache.
            return _dataframes_dict[candidate].copy() if copy else _dataframes_dict[candidate]
    raise ValueError(f"Dataset {name} not found")
