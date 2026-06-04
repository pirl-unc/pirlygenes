"""CTA population-addressability: the fraction of annual cancer incidence /
mortality (US and worldwide) addressable per CTA, = Σ over cancer types of
[that type's burden share × fraction of its patients expressing the CTA above a
TPM threshold]. One sorted horizontal bar per CTA, for each of
{incidence, mortality} × {US, worldwide} × {25, 50, 100 TPM} = 12 plots.

Each bar is labelled with the burden category that contributes most for that
CTA, and colored by the number of cancer types the CTA is expressed in (breadth).
Bars only count cohorts with per-sample CTA data, so the addressable % is a
lower bound — the covered-burden ceiling is annotated per plot. (issue #280)

Inputs: outputs/cta_patient_counts.csv (per cohort × CTA patient counts) +
the curated burden refs (pirlygenes.gene_sets_cancer.cancer_burden /
cancer_code_burden_map).

Run:  python analyses/cta_addressable_burden.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes import gene_sets_cancer as gsc

OUT = Path(__file__).resolve().parent / "outputs"
COUNTS = OUT / "cta_patient_counts.csv"
THRESHOLDS = [25, 50, 100]
TOPN = 40
# {plot-key: (burden metric column, axis label)}
METRICS = {
    "us_incidence": ("us_incidence_pct", "US incidence"),
    "us_mortality": ("us_mortality_pct", "US deaths"),
    "world_incidence": ("world_incidence_pct", "worldwide incidence"),
    "world_mortality": ("world_mortality_pct", "worldwide deaths"),
}
# Subtype cohorts whose patients are already counted by a parent cohort in the
# same burden category — drop them so a category's burden isn't double-weighted.
_REDUNDANT = {"BRCA_LumA", "BRCA_LumB", "BRCA_HER2", "BRCA_Basal", "BRCA_Normal",
              "HNSC_HPV_pos", "HNSC_HPV_neg"}


def _prepare():
    """Return (counts, cat_n, breadth): per-(category,Symbol) hit counts joined
    to category patient totals, plus per-CTA breadth (# cancer types expressed in)."""
    counts = pd.read_csv(COUNTS)
    counts = counts[~counts.cancer_code.isin(_REDUNDANT)].copy()
    # Robust resolution (alias -> explicit map -> parent chain -> tissue -> family);
    # warn loudly on anything unmapped rather than silently dropping it.
    cat = {c: gsc.burden_category(c) for c in counts.cancer_code.unique()}
    unmapped = sorted(c for c, v in cat.items() if v is None)
    if unmapped:
        print(f"  WARNING: {len(unmapped)} cohort(s) had NO burden category and "
              f"were dropped from addressability: {unmapped}", flush=True)
    counts["category"] = counts.cancer_code.map(cat)
    counts = counts[counts.category.notna()].copy()
    # category total patients = sum of its (deduped) cohorts' sample counts
    per_cohort = counts.groupby("cancer_code").agg(
        n=("n_samples", "first"), category=("category", "first"))
    cat_n = per_cohort.groupby("category").n.sum()
    # breadth: # cancer-type cohorts each CTA is detectable in (>25 TPM in >=1 patient)
    breadth = (counts[counts.n_gt25 > 0].groupby("Symbol").cancer_code.nunique())
    return counts, cat_n, breadth


def _addressable(counts, cat_n, threshold, burden_metric):
    """Per-CTA addressable % for one threshold + burden metric, with the
    top-contributing category. Returns a DataFrame sorted descending."""
    col = f"n_gt{threshold}"
    g = counts.groupby(["category", "Symbol"], as_index=False)[col].sum()
    g["pooled_frac"] = g[col] / g.category.map(cat_n)
    burden = gsc.cancer_burden(metric=burden_metric)  # {category: pct of all cancer}
    g["contrib"] = g.category.map(lambda c: burden.get(c, 0.0)) * g.pooled_frac
    per = g.groupby("Symbol").contrib.sum().rename("addressable").reset_index()
    top = g.loc[g.groupby("Symbol").contrib.idxmax(), ["Symbol", "category"]]
    per = per.merge(top, on="Symbol").sort_values("addressable", ascending=False)
    return per, burden


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    counts, cat_n, breadth = _prepare()
    n_cohorts = counts.cancer_code.nunique()
    n_cats = counts.category.nunique()
    bmax = int(breadth.max()) if len(breadth) else 1
    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=1, vmax=bmax)

    for mkey, (bcol, blabel) in METRICS.items():
        ceiling = sum(gsc.cancer_burden(metric=bcol).get(c, 0.0)
                      for c in counts.category.unique())
        for t in THRESHOLDS:
            per, _ = _addressable(counts, cat_n, t, bcol)
            top = per.head(TOPN).iloc[::-1]  # ascending for barh (max on top)
            colors = [cmap(norm(int(breadth.get(s, 1)))) for s in top.Symbol]
            fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.27)))
            ax.barh(range(len(top)), top.addressable, color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.Symbol, fontsize=6)
            for y, (val, cat) in enumerate(zip(top.addressable, top.category)):
                ax.text(val, y, f" {cat}", va="center", fontsize=5, color="0.3")
            ax.set_xlabel(f"% of annual {blabel} addressable "
                          f"(patient expresses this CTA > {t} TPM)")
            ax.set_title(f"CTA population addressability — {blabel} (> {t} TPM)\n"
                         f"top {len(top)} CTAs by addressable {blabel}", fontsize=10)
            ax.set_xlim(0, max(top.addressable) * 1.18)
            ax.grid(axis="x", alpha=0.3)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            cb.set_label("# cancer types expressed in", fontsize=7)
            ax.text(0.99, 0.01,
                    f"lower bound: {n_cohorts} per-sample cohorts → {n_cats} burden "
                    f"categories = {ceiling:.0f}% of {blabel} (uncovered types not scored)",
                    transform=ax.transAxes, ha="right", fontsize=6, color="gray")
            fig.tight_layout()
            fig.savefig(OUT / f"cta_addressable_{mkey}_t{t}.png", dpi=150)
            plt.close(fig)
            print(f"  {mkey} t{t}: top CTA {per.iloc[0].Symbol} "
                  f"({per.iloc[0].addressable:.1f}% of {blabel}); ceiling {ceiling:.0f}%",
                  flush=True)
    print(f"done -> 12 plots in {OUT}", flush=True)


if __name__ == "__main__":
    main()
