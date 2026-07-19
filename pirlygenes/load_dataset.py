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

For pirlygenes-owned datasets, two data roots are checked in order:

  1. ``_BUNDLED_DATA_DIR`` — files shipped in the wheel (small panels,
     registries) AND files present in a git-checkout's
     ``pirlygenes/data/`` (the dev workflow keeps everything here).
  2. ``_DOWNLOADED_DATA_DIR`` — the cache populated by
     :mod:`pirlygenes.data_bundle` (large per-cohort summaries fetched
     from the GitHub Release matching the installed version).

Any file present in (1) wins over (2). Datasets owned by oncoref are explicit
exceptions: ``get_data`` delegates them even in a source checkout, so stale
pirlygenes-local mirrors can never win (#557, #514).

When a callable here requests one of the
:data:`pirlygenes.data_bundle.DOWNLOADABLE_PATHS` items and it's
missing from both roots, :func:`pirlygenes.data_bundle.ensure_local`
triggers a one-time download from the GitHub Release.
"""

from pathlib import Path

import pandas as pd

from . import data_bundle
from .reference_source_cohorts import (
    normalize_reference_source_cohort_labels as _normalize_reference_source_cohort_labels,
)

_BUNDLED_DATA_DIR = Path(__file__).parent / "data"
_DOWNLOADED_DATA_DIR = data_bundle.cache_dir()
_DATASET_PATHS = None
_CACHED_DATAFRAMES = {}

# Base-layer normalization/gene-family datasets owned by oncoref. Keep their
# historical pirlygenes get_data names as compatibility re-exports without
# shipping a second physical copy that can drift.
_ONCOREF_DATASETS = frozenset({
    "cancer-cohort-aggregates",
    "clean-tpm-censored-genes",
    "ribosomal-protein-pseudogenes",
})

# oncoref owns the complete computed-aggregate ontology.  Pirlygenes keeps its
# historical public expansion surface (the four rollups it has always exposed)
# while sourcing those rows from that authority, so newly added children such
# as SARC_MPLPS cannot silently drift here again.
_PIRLYGENES_AGGREGATE_CODES = frozenset({
    "CRC",
    "SARC_ESS",
    "SARC_LPS",
    "SARC_RMS",
})

_COMPUTED_COHORT_AGGREGATE_CODES = {
    "COMPUTED_COLORECTAL": "CRC",
    "COMPUTED_PAN_SARCOMA": "SARC",
}


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
_DATASET_STRING_ID_COLS = {
    "cancer-type-registry.csv": ("code",),
}

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


def _normalize_dataset_dtypes(cache_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes that are part of a dataset's public schema.

    Registry code columns are stable string identifiers. Keeping them as plain
    object dtype makes ``DataFrame.set_index`` rely on pandas' deprecated object
    dtype inference when future string inference is enabled.
    """
    key = cache_key.lower().removesuffix(".gz")
    for col in _DATASET_STRING_ID_COLS.get(key, ()):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def _concat_shard_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate CSV shards without letting all-NA shard columns set dtypes."""
    if not frames:
        return pd.DataFrame()

    columns: list[str] = []
    seen: set[str] = set()
    trimmed = []
    for frame in frames:
        for col in frame.columns:
            if col not in seen:
                seen.add(col)
                columns.append(col)
        trimmed.append(frame.dropna(axis=1, how="all"))

    df = pd.concat(trimmed, ignore_index=True)
    return df.reindex(columns=columns)


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
    df = _concat_shard_frames([pd.read_csv(str(p), low_memory=False)
                               for p in paths])
    df = _categorize_metadata(df)
    df = _normalize_dataset_dtypes(shard_dir.name + ".csv", df)
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
        csv_key = csv_path.name.removesuffix(".gz")
        df = pd.read_csv(str(csv_path), low_memory=False)
        df = _normalize_dataset_dtypes(csv_key, df)
        if csv_key == "cohort-registry.csv":
            df = _reconcile_computed_cohort_members(df)
        yield csv_key, df
    # Preserve the generic enumeration surface for datasets whose physical
    # copies moved to oncoref. They are not present in get_all_csv_paths(), but
    # callers of load_all_dataframes_dict() still see the historical keys.
    for dataset in sorted(_ONCOREF_DATASETS):
        yield f"{dataset}.csv", get_data(dataset)
    # Runtime ownership of the empirical summary rows moved to oncoref (#557).
    # Yield the delegated frame exactly once even in a wheel install, where the
    # legacy pirlygenes shard directory is intentionally absent.
    yield "cancer-reference-expression.csv", get_data(
        "cancer-reference-expression"
    )
    for shard_dir in _shard_directories():
        if shard_dir.name == "cancer-reference-expression":
            continue
        yield f"{shard_dir.name}.csv", _load_shard_directory(shard_dir)


def load_all_dataframes_dict():
    """
    Dictionary of csv_file -> df for all CSV files in the data directory
    """
    return {csv_file: df for csv_file, df in load_all_dataframes()}


def _invalidate_dataset_paths() -> None:
    global _DATASET_PATHS
    _DATASET_PATHS = None


def _reconcile_computed_cohort_members(df: pd.DataFrame) -> pd.DataFrame:
    """Derive computed-cohort membership from oncoref's live ontology.

    The packaged cohort registry is useful pirlygenes-specific source metadata,
    but its computed rows are snapshots.  Re-derive only those rows so a newly
    added cancer atom cannot leave ``n_codes`` and ``member_cohorts`` stale
    (oncoref#387) while preserving the Merkel cohort absent from oncoref's
    source-cohort registry.
    """
    if not {"cohort_id", "n_codes", "member_cohorts"} <= set(df.columns):
        return df

    import oncoref

    replacements: list[tuple[pd.Series, list[str]]] = []
    for cohort_id, aggregate_code in _COMPUTED_COHORT_AGGREGATE_CODES.items():
        mask = df["cohort_id"].astype(str).eq(cohort_id)
        if not mask.any():
            continue
        members = list(oncoref.cohort_aggregate_members(aggregate_code) or [])
        serialized = ";".join(members)
        current_members = df.loc[mask, "member_cohorts"].fillna("").astype(str)
        current_counts = pd.to_numeric(df.loc[mask, "n_codes"], errors="coerce")
        if not current_members.eq(serialized).all() or not current_counts.eq(
            len(members)
        ).all():
            replacements.append((mask, members))
    if not replacements:
        return df

    out = df.copy()
    for mask, members in replacements:
        out.loc[mask, "member_cohorts"] = ";".join(members)
        out.loc[mask, "n_codes"] = len(members)
    return out


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
    ``cancer-reference-expression`` table (multi-million-row summary frame)
    dominated test-suite time (#278).
    """
    normalized_name = name.lower()

    delegated_name = normalized_name.removesuffix(".csv")
    if _dataframes_dict is None and delegated_name in _ONCOREF_DATASETS:
        cache_key = f"{delegated_name}.csv"
        if cache_key not in _CACHED_DATAFRAMES:
            from oncoref.load_dataset import get_data as get_oncoref_data

            delegated = get_oncoref_data(
                delegated_name,
                copy=False,
            )
            if delegated_name == "cancer-cohort-aggregates":
                delegated = delegated.loc[
                    delegated["aggregate_code"].astype(str).isin(
                        _PIRLYGENES_AGGREGATE_CODES
                    )
                ].copy()
            _CACHED_DATAFRAMES[cache_key] = delegated
        cached = _CACHED_DATAFRAMES[cache_key]
        return cached.copy() if copy else cached

    # The empirical cancer-reference-expression rows are owned by oncoref. Keep
    # pirlygenes' generic get_data surface working, but never select the duplicate
    # in-repo/downloaded shard set at runtime. oncoref >=1.8.133 applies its
    # low-cardinality encoding at the owning cache boundary (oncoref#390), so do
    # not mutate or re-encode that shared frame here. The narrow source-label
    # compatibility view does not copy unless an old physical label is present.
    # Fixture injection deliberately bypasses this branch. See #557 / #528.
    if _dataframes_dict is None and normalized_name in (
        "cancer-reference-expression", "cancer-reference-expression.csv"
    ):
        cache_key = "cancer-reference-expression.csv"
        if cache_key not in _CACHED_DATAFRAMES:
            from oncoref.load_dataset import get_data as get_oncoref_data

            delegated = get_oncoref_data(
                "cancer-reference-expression", copy=False
            )
            delegated, _ = _normalize_reference_source_cohort_labels(delegated)
            _CACHED_DATAFRAMES[cache_key] = delegated
        cached = _CACHED_DATAFRAMES[cache_key]
        return cached.copy() if copy else cached

    # The cancer-type registry is owned by oncoref (the empirical base layer);
    # pirlygenes re-exports it rather than shipping a divergent copy. Routing the
    # load through oncoref keeps every consumer — cancer_type_registry(),
    # CANCER_TYPE_NAMES, resolve_cancer_type, the cancer_types sub-library — on a
    # single source of truth. Tests inject a fixture registry either by monkey-
    # patching get_data wholesale or by passing _dataframes_dict; both bypass this
    # branch, so the fixture path is unchanged. See pirlygenes#523 / oncoref#275.
    if _dataframes_dict is None and normalized_name in (
        "cancer-type-registry", "cancer-type-registry.csv"
    ):
        import oncoref

        registry = _normalize_dataset_dtypes(
            "cancer-type-registry", oncoref.cancer_type_registry()
        )
        return registry.copy() if copy else registry

    # cancer-subtype-groupings (cross-cutting MSI/MSS/POLE/HPV/MYCN/EBV axes) is
    # likewise owned by oncoref as of 1.8.95 — a lossless superset of pirlygenes'
    # former local CSV (adds the EBV_POS axis + STAD_MSI/CIN/GS and CRC_MSI
    # members). Re-export it the same way so cancer_subtype_groupings() /
    # cancer_subtype_group() delegate rather than ship a divergent copy. Fixture
    # injection bypasses this branch as above. See oncoref#325.
    if _dataframes_dict is None and normalized_name in (
        "cancer-subtype-groupings", "cancer-subtype-groupings.csv"
    ):
        import oncoref

        groupings = _normalize_dataset_dtypes(
            "cancer-subtype-groupings", oncoref.cancer_subtype_groupings()
        )
        return groupings.copy() if copy else groupings

    # cancer-apd1-response and cancer-tmb are owned by oncoref's structured,
    # provenance-bearing tables (value_basis / source_scope / estimate_type /
    # missing_reason / ...). pirlygenes re-exports them rather than shipping
    # divergent local CSVs, so oncoref's ongoing re-curation flows through. The
    # per-code accessors (cancer_apd1_response / cancer_tmb) delegate to oncoref's
    # resolvers, which own the source-scope taxonomy fallback (COAD_MSI/READ_MSI ->
    # CRC_MSI, CHOL/GBC -> BTC, NET_MIDGUT/NET_LUNG -> NET_NONPANCREATIC,
    # ACINIC -> SGC) AND the parent-inherit chain. A compat `trial` column is
    # synthesized from oncoref's split trial_name/trial_alias for callers that
    # expected the former single column. Fixture injection bypasses this branch as
    # for the registry above. See pirlygenes#541 / #507.
    if _dataframes_dict is None and normalized_name in (
        "cancer-apd1-response", "cancer-apd1-response.csv"
    ):
        import oncoref

        apd1 = oncoref.cancer_apd1_response_df()
        if "trial" not in apd1.columns and "trial_name" in apd1.columns:
            apd1 = apd1.copy()
            trial = apd1["trial_name"]
            if "trial_alias" in apd1.columns:
                trial = trial.fillna(apd1["trial_alias"])
            apd1["trial"] = trial
        apd1 = _normalize_dataset_dtypes("cancer-apd1-response", apd1)
        return apd1.copy() if copy else apd1

    if _dataframes_dict is None and normalized_name in (
        "cancer-tmb", "cancer-tmb.csv"
    ):
        import oncoref

        tmb = _normalize_dataset_dtypes("cancer-tmb", oncoref.cancer_tmb_df())
        return tmb.copy() if copy else tmb

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
        # Check downloadability against the *candidate* set (incl. the ".csv"
        # form), not the bare ``name``: downloadables registered with a ".csv"
        # suffix (pan-cancer-expression.csv, hpa-cell-type-expression.csv) are
        # requested by the bare stem, so ``is_downloadable(name)`` alone missed
        # them — leaving the pre-fetch _dataset_paths cache stale and raising a
        # spurious "not found" on a clean install right after the bundle fetched.
        miss = not any(c in paths for c in candidates)
        if miss and any(data_bundle.is_downloadable(c) for c in candidates):
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
                        loaded = _normalize_dataset_dtypes(
                            cache_key,
                            pd.read_csv(
                                str(resolved), low_memory=False,
                            ),
                        )
                        if cache_key == "cohort-registry.csv":
                            loaded = _reconcile_computed_cohort_members(loaded)
                        _CACHED_DATAFRAMES[cache_key] = loaded
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
            df = _dataframes_dict[candidate].copy() if copy else _dataframes_dict[candidate]
            return _normalize_dataset_dtypes(candidate, df)
    raise ValueError(f"Dataset {name} not found")
