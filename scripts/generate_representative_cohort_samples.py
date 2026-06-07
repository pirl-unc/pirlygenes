"""Package a bounded set of *representative* real per-sample expression vectors
per cohort (issue #312).

The shipped reference cohorts are per-cohort aggregates (median / quantiles), so
downstream can only validate classification / normalization against the cohort
*median* — which overstates accuracy and can't reconstruct a physiological
sample (genes are correlated; sampling each gene independently at its own
quantile is non-physiological). This script packages K real per-sample joint
vectors per cohort — medoids spanning the within-cohort variation — in the same
``clean_tpm_v4`` basis the aggregates use, so consumers (trufflepig) can run the
*honest* sample-level self-classification battery.

Source = the cached per-sample TPM parquets the builders write
(``<source>/derived/<stem>_per_sample_tpm.parquet``, linear raw TPM). For each
cohort we apply :func:`clean_tpm_matrix` (``fixed_fraction`` = clean_tpm_v4),
then pick K representatives by k-means (in log1p-PCA space) and taking each
cluster's medoid (the real sample nearest the cluster centroid). Representatives
are written with **anonymized** ids (``<CODE>_rep01`` …) — provenance is kept at
the cohort/source level only, never the upstream barcode.

Output (bundle artifact, keyed like ``cancer-reference-expression/``):
    pirlygenes/data/cancer-reference-expression-representatives/<CODE>.parquet
        wide: Ensembl_Gene_ID, Symbol, <CODE>_rep01 … (clean_tpm_v4, float32)
    pirlygenes/data/cancer-reference-expression-representatives/_provenance.csv
        one row per representative: cancer_code, representative_id,
        source_cohort, source_project, n_cohort_samples, cluster_rank

Run:  python scripts/generate_representative_cohort_samples.py [--k 12]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes import cohorts as _cohorts
from pirlygenes.expression.normalize import clean_tpm_matrix

OUT_DIR = Path(__file__).resolve().parent.parent / "pirlygenes" / "data" \
    / "cancer-reference-expression-representatives"
CACHE = Path.home() / ".cache" / "pirlygenes" / "expression"

# (source_id, source_cohort label, source_project) for every source that ships
# per-sample parquets. The label mirrors the reference-expression source_cohort.
_SOURCES = [
    ("treehouse-polya-25-01", "TREEHOUSE_POLYA_25_01", "Treehouse"),
    ("treehouse-ribod-25-01", "TREEHOUSE_RIBOD_25_01", "Treehouse"),
]


def _stem_to_code(source_id: str) -> dict[str, str]:
    """{parquet_stem: cancer_code} for a source — via the cohort registry where
    present, else identity (uppercased) for sources with no registry yet."""
    reg = _cohorts.cohorts_for_source(source_id)
    return {c.stem: c.code for c in reg.values()}


def select_representatives(clean: pd.DataFrame, k: int, *, seed: int = 0) -> list[str]:
    """Pick up to ``k`` representative sample columns of ``clean`` (genes ×
    samples, clean_tpm_v4) — k-means in log1p-PCA space, medoid per cluster.

    Deterministic for a fixed ``seed``. Returns the chosen column labels in
    cohort order. Fewer than ``k`` may be returned if clusters collapse."""
    cols = list(clean.columns)
    n = len(cols)
    if n <= k:
        return cols
    X = np.log1p(clean.T.to_numpy(dtype=np.float64))  # samples × genes
    Xc = X - X.mean(axis=0)
    # economy SVD -> top-d PCA scores for stable, fast clustering
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    d = int(min(50, U.shape[0] - 1, S.shape[0]))
    Y = U[:, :d] * S[:d]  # samples × d
    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centers = [int(rng.integers(n))]
    dist = ((Y - Y[centers[0]]) ** 2).sum(axis=1)
    while len(centers) < k:
        total = dist.sum()
        nxt = (int(rng.choice(n, p=dist / total)) if total > 0
               else int(rng.integers(n)))
        centers.append(nxt)
        dist = np.minimum(dist, ((Y - Y[nxt]) ** 2).sum(axis=1))
    cent = Y[np.array(centers)].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(25):  # Lloyd iterations
        assign = np.argmin(((Y[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2),
                           axis=1)
        new = np.array([Y[assign == j].mean(axis=0) if (assign == j).any()
                        else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            cent = new
            break
        cent = new
    # medoid per cluster = real sample nearest the cluster centroid
    medoids = []
    for j in range(k):
        members = np.where(assign == j)[0]
        if len(members) == 0:
            continue
        dd = ((Y[members] - cent[j]) ** 2).sum(axis=1)
        medoids.append(int(members[int(np.argmin(dd))]))
    medoids = sorted(set(medoids))
    return [cols[i] for i in medoids]


def build(k: int = 12) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = []
    n_cohorts = 0
    for source_id, source_cohort, source_project in _SOURCES:
        derived = CACHE / source_id / "derived"
        if not derived.exists():
            print(f"[skip] no cache for {source_id}", flush=True)
            continue
        stem_to_code = _stem_to_code(source_id)
        for parquet in sorted(derived.glob("*_per_sample_tpm.parquet")):
            stem = parquet.name[: -len("_per_sample_tpm.parquet")]
            code = stem_to_code.get(stem, stem.upper())
            df = pd.read_parquet(parquet)
            sample_cols = [c for c in df.columns
                           if c not in ("Ensembl_Gene_ID", "Symbol")]
            if not sample_cols:
                continue
            gene_table = df[["Symbol", "Ensembl_Gene_ID"]]
            clean = clean_tpm_matrix(df[sample_cols], gene_table=gene_table,
                                     censored_fill="fixed_fraction")
            chosen = select_representatives(clean, k)
            reps = clean[chosen]
            rep_ids = [f"{code}_rep{i:02d}" for i in range(1, len(chosen) + 1)]
            out = pd.DataFrame({
                "Ensembl_Gene_ID": df["Ensembl_Gene_ID"].astype(str).values,
                "Symbol": df["Symbol"].astype(str).values,
            })
            for rid, col in zip(rep_ids, chosen):
                out[rid] = reps[col].to_numpy(dtype=np.float32)
            out.to_parquet(OUT_DIR / f"{code}.parquet", index=False,
                           compression="zstd")
            for rank, rid in enumerate(rep_ids, start=1):
                provenance.append({
                    "cancer_code": code,
                    "representative_id": rid,
                    "source_cohort": source_cohort,
                    "source_project": source_project,
                    "n_cohort_samples": len(sample_cols),
                    "cluster_rank": rank,
                })
            n_cohorts += 1
            print(f"  {code}: {len(sample_cols)} samples -> {len(chosen)} reps",
                  flush=True)
    prov = pd.DataFrame(provenance).sort_values(["cancer_code", "cluster_rank"])
    prov.to_csv(OUT_DIR / "_provenance.csv", index=False)
    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.parquet")) / 1e6
    print(f"\ndone: {n_cohorts} cohorts, {len(prov)} representatives, "
          f"{total_mb:.0f} MB -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12,
                    help="representatives per cohort (medoids; default 12)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    build(k=args.k)
