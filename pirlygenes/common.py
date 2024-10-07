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

import pandas as pd


def find_column(
    df: pd.DataFrame, candidates: list[str], column_name: str
) -> str | None:
    result = None
    for candidate in df.columns:
        if candidate.lower() in candidates:
            result = candidate
            break
    if result is None:
        raise ValueError(
            "Unable to find a column for %s in expression data, available columns: %s"
            % (
                column_name,
                list(
                    df.columns,
                ),
            )
        )
    return result
