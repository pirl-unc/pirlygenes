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


def get_all_csv_paths() -> list:
    """
    Get paths to all CSV files in the data directory.

    Picks up both plain ``.csv`` and gzipped ``.csv.gz`` files, keyed
    consistently by their underlying ``.csv`` name.

    Returns a list of Path objects for each CSV file.
    """
    return sorted(list(_DATA_DIR.glob("*.csv")) + list(_DATA_DIR.glob("*.csv.gz")))


def load_all_dataframes():
    """
    Generator that yields pairs of (csv_name, df) for all CSV files in the
    data directory. Gzipped files are transparently decompressed and keyed
    under their underlying ``.csv`` name so callers don't need to know
    the on-disk compression format.
    """
    for csv_path in get_all_csv_paths():
        df = pd.read_csv(str(csv_path), low_memory=False)
        key = csv_path.name.removesuffix(".gz")
        yield key, df


def load_all_dataframes_dict():
    """
    Dictionary of csv_file -> df for all CSV files in the data directory
    """
    return {csv_file: df for csv_file, df in load_all_dataframes()}


def _dataset_paths():
    """Map accepted dataset names to their on-disk CSV path."""
    global _DATASET_PATHS
    if _DATASET_PATHS is not None:
        return _DATASET_PATHS

    paths = {}
    for csv_path in get_all_csv_paths():
        csv_key = csv_path.name.removesuffix(".gz")
        stem_key = csv_key.removesuffix(".csv")
        for key in {csv_key, csv_key.lower(), stem_key, stem_key.lower()}:
            paths[key] = csv_path
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
                csv_path = paths[candidate]
                cache_key = csv_path.name.removesuffix(".gz")
                if cache_key not in _CACHED_DATAFRAMES:
                    _CACHED_DATAFRAMES[cache_key] = pd.read_csv(
                        str(csv_path), low_memory=False
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
