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


def get_all_csv_paths() -> list:
    """
    Get paths to all CSV files in the data directory.

    Picks up both plain ``.csv`` and gzipped ``.csv.gz`` — the latter is used
    for references that would otherwise bloat the package (e.g. the #22
    tcga-deconvolved-expression reference, ~13 MB → ~4 MB gzipped).

    Returns a list of Path objects for each CSV file.
    """
    return sorted(
        list(_DATA_DIR.glob("*.csv")) + list(_DATA_DIR.glob("*.csv.gz"))
    )


def load_all_dataframes():
    """
    Generator that yields pairs of (csv_name, df) for all CSV files in the
    data directory. Gzipped files are transparently decompressed and keyed
    under their underlying ``.csv`` name so callers don't need to know
    the on-disk compression format.
    """
    for csv_path in get_all_csv_paths():
        df = pd.read_csv(str(csv_path))
        key = csv_path.name.removesuffix(".gz")
        yield key, df


def load_all_dataframes_dict():
    """
    Dictionary of csv_file -> df for all CSV files in the data directory
    """
    return {csv_file: df for csv_file, df in load_all_dataframes()}


_CACHED_DATAFRAMES = None


def get_data(name, _dataframes_dict=None):
    global _CACHED_DATAFRAMES
    if _dataframes_dict is None:
        if _CACHED_DATAFRAMES is None:
            _CACHED_DATAFRAMES = load_all_dataframes_dict()
        _dataframes_dict = _CACHED_DATAFRAMES
    candidates = [name, name.lower()]
    for candidate in list(candidates):
        candidates.append(candidate + ".csv")
    for candidate in candidates:
        if candidate in _dataframes_dict:
            # Return a copy so callers that mutate in place (e.g. df["c"]=...,
            # df.fillna(0, inplace=True)) can't corrupt the shared cache.
            return _dataframes_dict[candidate].copy()
    raise ValueError(f"Dataset {name} not found")
