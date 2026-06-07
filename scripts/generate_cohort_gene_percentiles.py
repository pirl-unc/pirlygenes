"""Per-gene × cohort tail-weighted percentile vectors (#298).

trufflepig wants to contextualize a sample's expression as a **percentile rank
within the relevant cohort** rather than an absolute TPM. The breakpoints are
reference-cohort statistics computed from per-patient expression — pirlygenes
owns both the per-patient data and the cohort definitions, so it computes/ships
them (compact) instead of shipping raw per-patient matrices to every consumer.

For every cohort with a cached per-sample matrix (the representatives sources),
this computes a **26-breakpoint tail-weighted percentile vector per gene** —
dense in the actionable upper tail where targeting candidacy lives:

    p0, p1, p5, p10, p15 … p90, p95, p96, p97, p98, p99, p100

Computed on the **biological clean_tpm_v4 view** (technical genes dropped), so
it sidesteps the fixed-fraction technical-inflation artifact (#304) and stays
compact. Quantiles are linear-interpolated empirical (numpy 'linear').

Stored as **log1p + float16** (the maintainer's size choice): ~3 sig figs on the
log scale — ample for percentile-rank lookup — at a fraction of float32. The
accessor (`cohort_gene_percentiles`) expm1's back to TPM by default. Artifact:
``cancer-reference-expression-percentiles/<CODE>.parquet``.

Run:  python scripts/generate_cohort_gene_percentiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_representative_cohort_samples import (  # noqa: E402
    _SOURCES,
    _stem_to_code,
)

from pirlygenes.expression.normalize import (  # noqa: E402
    clean_tpm_matrix,
    drop_technical_genes,
)

CACHE = Path.home() / ".cache" / "pirlygenes" / "expression"
OUT_DIR = Path(__file__).resolve().parent.parent / "pirlygenes" / "data" \
    / "cancer-reference-expression-percentiles"

# 26 tail-weighted breakpoints: every 5 through the middle, every 1 in the
# actionable upper tail; p0/p100 = min/max.
BREAKPOINTS = ([0, 1] + list(range(5, 91, 5)) + [95, 96, 97, 98, 99, 100])


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [f"p{b}" for b in BREAKPOINTS]
    n_cohorts = 0
    for source_id, _label, _project in _SOURCES:
        derived = CACHE / source_id / "derived"
        if not derived.exists():
            continue
        stem_to_code = _stem_to_code(source_id)
        for parquet in sorted(derived.glob("*_per_sample_tpm.parquet")):
            stem = parquet.name[: -len("_per_sample_tpm.parquet")]
            code = stem_to_code.get(stem, stem.upper())
            df = pd.read_parquet(parquet)
            sample_cols = [c for c in df.columns
                           if c not in ("Ensembl_Gene_ID", "Symbol")]
            if len(sample_cols) < 2:
                continue
            clean = clean_tpm_matrix(df[sample_cols],
                                     gene_table=df[["Symbol", "Ensembl_Gene_ID"]],
                                     censored_fill="fixed_fraction")
            clean.insert(0, "Ensembl_Gene_ID", df["Ensembl_Gene_ID"].astype(str).values)
            clean.insert(0, "Symbol", df["Symbol"].astype(str).values)
            bio = drop_technical_genes(clean)
            biovals = bio[sample_cols].to_numpy(dtype=np.float64)
            pct = np.percentile(biovals, BREAKPOINTS, axis=1).T  # genes × 26
            out = pd.DataFrame(np.log1p(pct).astype(np.float16), columns=cols)
            out.insert(0, "Ensembl_Gene_ID", bio["Ensembl_Gene_ID"].values)
            out.insert(0, "Symbol", bio["Symbol"].values)
            out.to_parquet(OUT_DIR / f"{code}.parquet", index=False,
                           compression="zstd")
            n_cohorts += 1
            print(f"  {code}: {len(out)} genes × {len(BREAKPOINTS)} breakpoints "
                  f"(n={len(sample_cols)})", flush=True)
    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*.parquet")) / 1e6
    print(f"\ndone: {n_cohorts} cohorts, {total_mb:.0f} MB -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    build()
