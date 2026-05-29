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

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"
_DATASET_PATHS = None
_CACHED_DATAFRAMES = {}


def _shard_directories() -> list[Path]:
    """Subdirectories under ``data/`` that hold sharded CSV datasets.

    A shard directory ``data/<name>/`` containing one or more
    ``*.csv.gz`` files acts as a single logical dataset addressable
    as ``<name>`` via :func:`get_data` — its shards are loaded and
    concatenated transparently. Used to keep individual file sizes
    under GitHub's 100 MB push limit; cancer-reference-expression is
    sharded per ``source_cohort``.
    """
    out: list[Path] = []
    for child in sorted(_DATA_DIR.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.csv")) or any(child.glob("*.csv.gz")):
            out.append(child)
    return out


def _shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(list(shard_dir.glob("*.csv")) + list(shard_dir.glob("*.csv.gz")))


def get_all_csv_paths() -> list:
    """
    Get paths to all CSV files in the data directory.

    Picks up both plain ``.csv`` and gzipped ``.csv.gz`` files, keyed
    consistently by their underlying ``.csv`` name. Sharded datasets
    (see :func:`_shard_directories`) are not enumerated here — they
    are loaded as a single logical CSV by :func:`get_data`.

    Returns a list of Path objects for each CSV file.
    """
    return sorted(list(_DATA_DIR.glob("*.csv")) + list(_DATA_DIR.glob("*.csv.gz")))


def _load_shard_directory(shard_dir: Path) -> pd.DataFrame:
    """Concatenate every ``*.csv[.gz]`` shard in a sharded dataset directory."""
    frames = [
        pd.read_csv(str(p), low_memory=False) for p in _shard_paths(shard_dir)
    ]
    if not frames:
        raise FileNotFoundError(f"no CSV shards found under {shard_dir}")
    return pd.concat(frames, ignore_index=True)


def load_all_dataframes():
    """
    Generator that yields pairs of (csv_name, df) for all CSV files in the
    data directory. Gzipped files are transparently decompressed and keyed
    under their underlying ``.csv`` name so callers don't need to know
    the on-disk compression format. Sharded directories yield once as the
    full concatenated frame, keyed under ``<dirname>.csv``.
    """
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


def get_data(name, _dataframes_dict=None):
    candidates = [name, name.lower()]
    for candidate in list(candidates):
        candidates.append(candidate + ".csv")

    if _dataframes_dict is None:
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
                return _CACHED_DATAFRAMES[cache_key].copy()
        raise ValueError(f"Dataset {name} not found")

    for candidate in candidates:
        if candidate in _dataframes_dict:
            # Return a copy so callers that mutate in place (e.g. df["c"]=...,
            # df.fillna(0, inplace=True)) can't corrupt the shared cache.
            return _dataframes_dict[candidate].copy()
    raise ValueError(f"Dataset {name} not found")
