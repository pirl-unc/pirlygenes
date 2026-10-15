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
  2. ``data_bundle.data_dir()`` — the cache populated by
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

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import datacache
import pandas as pd

from . import data_bundle

_BUNDLED_DATA_DIR = Path(__file__).parent / "data"
_DATASET_PATHS = {}
_CACHED_DATAFRAMES = {}

# Base-layer normalization/gene-family datasets owned by oncoref. Keep their
# historical pirlygenes get_data names as compatibility re-exports without
# shipping a second physical copy that can drift.
_ONCOREF_DATASETS = frozenset({
    "cancer-cohort-aggregates",
    "cancer-code-burden-map",
    "cancer-driver-genes",
    "cancer-driver-variants",
    "cancer-expression-source-candidates",
    "cancer-incidence-mortality",
    "cancer-lineage-groups",
    "cancer-reference-expression-availability",
    "cancer-reference-expression-samples",
    "clean-tpm-censored-genes",
    "cohort-registry",
    "legacy-dataset-dispositions",
    "ncbi-symbol-synonyms",
    "ribosomal-protein-pseudogenes",
    "therapy-benefit-toxicity-evidence",
})

# oncoref owns the complete computed-aggregate ontology.  Pirlygenes keeps its
# historical public expansion surface (CRC plus three sarcoma groupings) while
# sourcing those rows from that authority, so newly added children such as
# SARC_MPLPS cannot silently drift here again. Pan-expression rollups are a
# separate public contract in ``expression.PAN_CANCER_ROLLUP_MEMBERS``.
_PIRLYGENES_AGGREGATE_CODES = frozenset({
    "CRC",
    "SARC_ESS",
    "SARC_LPS",
    "SARC_RMS",
})


def data_roots(*, bundled_only: bool = False) -> tuple[Path, ...]:
    """Return active data directories in precedence order, without downloading.

    Bundled files take precedence. ``bundled_only=True`` never inspects the
    optional download cache. Otherwise, the second root follows DataCache's
    currently published immutable bundle, including refreshes by other processes.
    """
    if bundled_only:
        return (_BUNDLED_DATA_DIR,)
    return (_BUNDLED_DATA_DIR, data_bundle.data_dir())


def _shard_directories(*, roots=None) -> list[Path]:
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
    for root in data_roots() if roots is None else roots:
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


def get_all_csv_paths(*, roots=None) -> list:
    """Paths to every top-level CSV file across both data roots.

    Picks up both plain ``.csv`` and gzipped ``.csv.gz`` files. Sharded
    datasets (see :func:`dataset_paths`) are not enumerated here —
    they are loaded as a single logical CSV by :func:`get_data`.

    On name conflicts, the bundled root wins over the downloaded root.
    """
    seen: dict[str, Path] = {}
    for root in data_roots() if roots is None else roots:
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
_SHARD_CACHE_FORMAT = 4


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
    """Read CSV shards, reusing a transactional DataCache-derived Parquet.

    Every shard's name, size, mtime and ctime participates in the derivation
    identity, along with its immutable bundle directory and the schema version.
    DataCache copies the sources, serializes writers and publishes all output
    bytes atomically. Different installations/edits cannot share output files.
    Cache hits verify the derived artifact; any cache failure falls back to the
    authoritative CSVs without modifying caller-owned files.
    """
    paths = _shard_paths(shard_dir)
    if not paths:
        raise FileNotFoundError(f"no CSV shards found under {shard_dir}")
    frame = None

    def build(source_paths, output_paths):
        nonlocal frame
        frame = _concat_shard_frames([
            pd.read_csv(source_paths[p.name], low_memory=False) for p in paths
        ])
        frame = _categorize_metadata(frame)
        frame = _normalize_dataset_dtypes(shard_dir.name + ".csv", frame)
        frame.to_parquet(output_paths["data.parquet"], index=False)

    try:
        records = []
        for path in paths:
            info = path.stat()
            records.append((path.name, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        signature = repr((_SHARD_CACHE_FORMAT, str(shard_dir.resolve()), records))
        identity = sha256(signature.encode()).hexdigest()
        cache_dir = Path.home() / ".cache" / "pirlygenes" / "shard_cache"
        artifacts = datacache.materialize(
            cache_dir / f"{shard_dir.name}-{identity}",
            {p.name: {"path": p} for p in paths},
            transform={"version": str(_SHARD_CACHE_FORMAT), "options": signature},
            outputs={"data.parquet": {}},
            builder=build,
            download_options={"show_progress": False},
        )
        if frame is None:
            frame = _normalize_dataset_dtypes(
                shard_dir.name + ".csv", pd.read_parquet(artifacts["data.parquet"]),
            )
        return _categorize_metadata(frame)
    except Exception:
        # Caching is optional: inaccessible/corrupt artifacts or failed writes
        # must never prevent reading the caller-owned source. Discard any
        # partial build and run normal parsing/validation again on that source.
        frame = None
    frame = _concat_shard_frames([
        pd.read_csv(str(path), low_memory=False) for path in paths
    ])
    return _normalize_dataset_dtypes(shard_dir.name + ".csv", _categorize_metadata(frame))


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
    for name in data_bundle.DOWNLOADABLE_PATHS:
        if not any((root / name).exists() for root in data_roots()):
            data_bundle.ensure_local()
    # Also discover files added in-place to a development checkout.
    clear_dataset_cache()
    names = {path.name.removesuffix(".gz") for path in dataset_paths().values()}
    names.update(f"{name}.csv" for name in _ONCOREF_DATASETS)
    names.add("cancer-reference-expression.csv")
    for name in sorted(names):
        yield name if name.endswith(".csv") else f"{name}.csv", get_data(
            name.removesuffix(".csv")
        )


def load_all_dataframes_dict():
    """Return all logical datasets as a dictionary of CSV names to frames."""
    return dict(load_all_dataframes())


def clear_dataset_cache() -> None:
    """Clear discovered paths and loaded frames, without modifying disk data.

    Call this after editing bundled or legacy files in place. Managed bundle
    refreshes are detected automatically and do not require an explicit reset.
    """
    _DATASET_PATHS.clear()
    _CACHED_DATAFRAMES.clear()


def dataset_paths(*, bundled_only: bool = False) -> Mapping[str, Path]:
    """Map accepted dataset names to CSV files or shard directories.

    Includes stems, ``.csv`` names and lowercase aliases; bundled data wins
    name conflicts. The returned mapping is read-only. Discovery never fetches
    data and is cached separately for bundled-only and combined lookups, keyed
    by their active directories so a newly published bundle is seen immediately.
    See :func:`clear_dataset_cache` for files edited in place.
    """
    roots = data_roots(bundled_only=bundled_only)
    cached = _DATASET_PATHS.get(bundled_only)
    if cached is not None and cached[0] == roots:
        return cached[1]

    paths: dict[str, Path] = (
        {} if bundled_only else dict(dataset_paths(bundled_only=True))
    )
    for root in roots if bundled_only else roots[1:]:
        for csv_path in get_all_csv_paths(roots=[root]):
            csv_key = csv_path.name.removesuffix(".gz")
            stem_key = csv_key.removesuffix(".csv")
            for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
                paths.setdefault(key, csv_path)
        for shard_dir in _shard_directories(roots=[root]):
            stem_key = shard_dir.name
            csv_key = stem_key + ".csv"
            for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
                paths.setdefault(key, shard_dir)
    result = MappingProxyType(paths)
    _DATASET_PATHS[bundled_only] = (roots, result)
    if cached is not None:
        # Release frames from obsolete immutable installations. Delegated
        # oncoref frames use string keys and have their own ownership lifecycle.
        active_paths = set(paths.values())
        for key in list(_CACHED_DATAFRAMES):
            if isinstance(key, Path) and key not in active_paths:
                _CACHED_DATAFRAMES.pop(key, None)
    return result


def _reconcile_reference_expression_samples(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve pirlygenes' public sample-manifest schema over owner rows.

    Oncoref owns the physical manifest, including canonical lineage and
    pipeline provenance as of 1.8.162. Pirlygenes historically added a BeatAML
    ``subtype`` column, so derive only that compatibility field from owner rows.
    """
    out = samples.copy()

    # ``subtype`` was a BeatAML compatibility field. For included samples it is
    # identical to canonical lineage_label; excluded duplicate aliquots inherit
    # the unique subtype of the retained aliquot for the same case. This
    # reconstructs the historical 667 assignments entirely from owner metadata.
    subtype = pd.Series(pd.NA, index=out.index, dtype="object")
    if {"source_cohort", "case_id", "lineage_label"} <= set(out.columns):
        beataml = out["source_cohort"].astype(str).eq("BEATAML_OHSU_2022")
        subtype.loc[beataml] = out.loc[beataml, "lineage_label"]
        classified = out.loc[
            beataml
            & out["case_id"].notna()
            & out["lineage_label"].notna(),
            ["case_id", "lineage_label"],
        ]
        unambiguous = (
            classified.groupby("case_id")["lineage_label"]
            .agg(
                lambda values: (
                    values.iloc[0]
                    if values.astype(str).nunique() == 1
                    else pd.NA
                )
            )
        )
        missing = beataml & subtype.isna() & out["case_id"].notna()
        subtype.loc[missing] = out.loc[missing, "case_id"].map(unambiguous)
    out["subtype"] = subtype

    return out


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
        cached = _CACHED_DATAFRAMES.get(cache_key)
        if cached is None:
            if delegated_name == "cohort-registry":
                import oncoref

                # Use the owner's public resolver, which reconciles physical
                # source rows; its generic table is the unprocessed input.
                delegated = oncoref.cohort_registry_df()
            else:
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
            elif delegated_name == "cancer-reference-expression-samples":
                delegated = _reconcile_reference_expression_samples(delegated)
            elif delegated_name == "therapy-benefit-toxicity-evidence":
                # The owner adds a stable evidence_id. Keep the historical
                # pirlygenes generic-loader schema while the typed wrapper
                # delegates all filtering and evidence semantics upstream.
                delegated = delegated.drop(columns="evidence_id").copy()
            _CACHED_DATAFRAMES[cache_key] = delegated
            cached = delegated
        return cached.copy() if copy else cached

    # The empirical cancer-reference-expression rows are owned by oncoref. Keep
    # pirlygenes' generic get_data surface working, but never select the duplicate
    # in-repo/downloaded shard set at runtime. oncoref >=1.8.133 applies its
    # low-cardinality encoding at the owning cache boundary (oncoref#390), so do
    # not mutate or re-encode that shared frame here. oncoref also owns the
    # canonical physical source-cohort labels. Fixture injection deliberately
    # bypasses this branch. See #557 / #528.
    if _dataframes_dict is None and normalized_name in (
        "cancer-reference-expression", "cancer-reference-expression.csv"
    ):
        cache_key = "cancer-reference-expression.csv"
        cached = _CACHED_DATAFRAMES.get(cache_key)
        if cached is None:
            from oncoref.load_dataset import get_data as get_oncoref_data

            delegated = get_oncoref_data(
                "cancer-reference-expression", copy=False
            )
            _CACHED_DATAFRAMES[cache_key] = delegated
            cached = delegated
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
        paths = dataset_paths(bundled_only=True)
        if not any(c in paths for c in candidates):
            paths = dataset_paths()
        if (not any(c in paths for c in candidates)
                and any(data_bundle.is_downloadable(c) for c in candidates)):
            data_bundle.ensure_local()
            # Legacy installers may add files in-place without changing roots.
            clear_dataset_cache()
            paths = dataset_paths()

        for candidate in candidates:
            if candidate in paths:
                resolved = paths[candidate]
                cached = _CACHED_DATAFRAMES.get(resolved)
                if cached is None:
                    if resolved.is_dir():
                        loaded = _load_shard_directory(resolved)
                    else:
                        loaded = _normalize_dataset_dtypes(
                            resolved.name.removesuffix(".gz"),
                            pd.read_csv(str(resolved), low_memory=False),
                        )
                    _CACHED_DATAFRAMES[resolved] = loaded
                    cached = loaded
                return cached.copy() if copy else cached
        raise ValueError(f"Dataset {name} not found")

    for candidate in candidates:
        if candidate in _dataframes_dict:
            # Return a copy so callers that mutate in place (e.g. df["c"]=...,
            # df.fillna(0, inplace=True)) can't corrupt the shared cache.
            df = _dataframes_dict[candidate].copy() if copy else _dataframes_dict[candidate]
            return _normalize_dataset_dtypes(candidate, df)
    raise ValueError(f"Dataset {name} not found")
