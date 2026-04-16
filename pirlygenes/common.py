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

from contextlib import contextmanager
import pandas as pd
from typing import Iterator, Optional


def find_column(
    df: pd.DataFrame, candidates: list[str], column_name: str
) -> Optional[str]:
    result = None
    for col in df.columns:
        if col.lower() in candidates:
            result = col
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


@contextmanager
def without_dataframe_attrs(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Temporarily clear ``DataFrame.attrs`` during pandas-heavy helpers.

    Pandas deep-copies ``attrs`` in many column/subset/finalize paths.
    When callers attach a large object there (for example the retained
    transcript-level frame under ``attrs["transcript_expression"]``),
    otherwise-cheap helpers can become minute-scale. Clear attrs for the
    duration of the helper, then restore them.
    """
    saved_attrs = dict(getattr(df, "attrs", {}))
    if saved_attrs:
        df.attrs = {}
    try:
        yield df
    finally:
        if saved_attrs:
            df.attrs = saved_attrs
