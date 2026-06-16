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

"""Representative-sample selection (cohort-level mechanical op).

Pick a bounded set of *typical* real samples that span a cohort's
within-cohort variation — the engine behind the packaged
``cancer-reference-expression-representatives`` artifact (#312) and reusable
for any gene×sample matrix.

The method is **k-means (in log1p-PCA space) + per-cluster medoid**: cluster
the samples, then for each cluster keep the real sample nearest the cluster
centroid. The medoid is an actual patient vector (never a synthetic average —
averaging correlated genes is non-physiological). Deterministic for a fixed
seed; dependency-free (pure NumPy, no scikit-learn).

Cluster on a **biology-only** matrix (drop the technical compartment first with
:func:`pirlygenes.expression.drop_technical_genes`) so the selection rides on
biological signal and is insensitive to the clean_tpm_16_9_75 fixed-fraction floor
(#304) — while the *stored* vectors stay in the clean_tpm_16_9_75 basis that matches
the aggregate references.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_representative_samples(matrix, k: int, *, seed: int = 0) -> list:
    """Pick up to ``k`` representative sample columns of ``matrix``.

    ``matrix`` is genes (rows) × samples (cols), values on a linear expression
    scale (e.g. clean_tpm_16_9_75); ideally already restricted to biological genes
    (see :func:`pirlygenes.expression.drop_technical_genes`). Returns the chosen
    column labels in cohort order. Fewer than ``k`` may be returned when
    clusters collapse (empty cluster, or two clusters share a nearest sample) or
    when ``n_samples <= k`` (all columns returned). Deterministic for a fixed
    ``seed``.

    Algorithm: log1p → mean-center → economy SVD to ≤50 PCs → k-means++ seeding
    → up to 25 Lloyd iterations → per-cluster medoid (real sample nearest the
    centroid, squared-Euclidean in PC space).
    """
    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)
    cols = list(matrix.columns)
    n = len(cols)
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if n <= k:
        return cols

    X = np.log1p(matrix.to_numpy(dtype=np.float64).T)  # samples × genes
    Xc = X - X.mean(axis=0)
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    d = int(min(50, U.shape[0] - 1, S.shape[0]))
    Y = U[:, :d] * S[:d]  # samples × d PCA scores

    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(n))]
    dist = ((Y - Y[centers[0]]) ** 2).sum(axis=1)
    while len(centers) < k:
        total = float(dist.sum())
        nxt = (int(rng.choice(n, p=dist / total)) if total > 0
               else int(rng.integers(n)))
        centers.append(nxt)
        dist = np.minimum(dist, ((Y - Y[nxt]) ** 2).sum(axis=1))

    cent = Y[np.array(centers)].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(25):  # Lloyd iterations
        assign = np.argmin(
            ((Y[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2), axis=1)
        new = np.array([Y[assign == j].mean(axis=0) if (assign == j).any()
                        else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            cent = new
            break
        cent = new

    medoids = []
    for j in range(k):
        members = np.where(assign == j)[0]
        if len(members) == 0:
            continue
        dd = ((Y[members] - cent[j]) ** 2).sum(axis=1)
        medoids.append(int(members[int(np.argmin(dd))]))
    medoids = sorted(set(medoids))
    return [cols[i] for i in medoids]
