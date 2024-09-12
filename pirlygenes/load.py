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

import importlib.resources as pkg_resources
from importlib.resources import contents
from pathlib import Path


import pandas as pd

def get_all_csv_paths() -> list[Path]:
    """
    Get paths to all CSV files in the data directory.
    
    Returns a list of Path objects for each CSV file.
    """
    csv_paths = []
    for resource in contents('pirlygenes.data'):
        if resource.endswith('.csv'):
            with pkg_resources.path('pirlygenes.data', resource) as path:
                csv_paths.append(path)
    return csv_paths

            
def load_all_dataframes():
    """
    Generator that yields pairs of (csv_file, df) for all CSV files in the data directory
    """
    for csv_path in get_all_csv_paths():
        df = pd.read_csv(str(csv_path))
        yield csv_path.name, df


def load_all_dataframes_dict():
    """
    Dictionary of csv_file -> df for all CSV files in the data directory
    """
    return {csv_file: df for csv_file, df in load_all_dataframes()}

def get_data(name, _dataframes_dict=None):
    if _dataframes_dict is None:
        _dataframes_dict = load_all_dataframes_dict()

    if name not in _dataframes_dict:
        raise ValueError(f"Dataset {name} not found")
    return _dataframes_dict[name]
