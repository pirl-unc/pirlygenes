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

"""Regression fence for analyze-pipeline performance (issue #81).

v4.5.0 shipped with a pyensembl-lookup loop that turned a ~90s analyze
into a 10+ minute stall on real salmon quant.sf inputs. The test below
loads a small synthetic quant and asserts a proportional time bound.
The point is not to fence microbenchmarks but to catch a *catastrophic*
regression (10×+) the next time someone wires per-transcript work into
the load path.

We use 5k transcripts (not 120k) so the test finishes in a few seconds
while still exercising every branch in the load path. The bound is
proportional: 5k/120k of a 2-minute ceiling ≈ 5s, rounded up to 15s
for CI variance.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import pytest

N_TRANSCRIPTS = 5_000
TIME_CEILING_S = 15.0


def _make_fake_quant(path: Path, n: int = N_TRANSCRIPTS, seed: int = 42) -> None:
    rng = random.Random(seed)
    tpms = [rng.random() * 10 if rng.random() < 0.3 else 0.0 for _ in range(n)]
    pd.DataFrame(
        {
            "Name": [f"ENST{i:011d}.1" for i in range(n)],
            "Length": [1000] * n,
            "EffectiveLength": [900] * n,
            "TPM": tpms,
            "NumReads": [0] * n,
        }
    ).to_csv(path, sep="\t", index=False)


@pytest.mark.perf
def test_load_expression_data_perf_fence(tmp_path: Path) -> None:
    quant_path = tmp_path / "fake_quant.sf"
    _make_fake_quant(quant_path)

    from pirlygenes.load_expression import load_expression_data

    t0 = time.time()
    df = load_expression_data(
        str(quant_path),
        aggregate_gene_expression=True,
        save_aggregated_gene_expression=False,
        progress=False,
        verbose=False,
    )
    elapsed = time.time() - t0

    assert elapsed < TIME_CEILING_S, (
        f"load_expression_data took {elapsed:.1f}s on {N_TRANSCRIPTS} fake "
        f"transcripts — expected < {TIME_CEILING_S}s (issue #81)"
    )
    assert df is not None and len(df) > 0
