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

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes import gene_sets_cancer as gsc
from _run_layout import latest_run_dir

OUT = Path(__file__).resolve().parent / "outputs"
# Set in main() to the run_<ts>/ that holds cta_patient_counts' tables; this
# script both reads those tables and writes its plots into that same run dir.
COUNTS = OUT / "cta_patient_counts.csv"
UNION = OUT / "cta_union_counts.csv"
FIGDIR = OUT
TPM_THRESHOLDS = [25, 50, 100]
PERCENTILES = [80, 90, 95]
TOPN = 40


class Thr:
    """An addressability threshold: absolute TPM (``kind='tpm'``) or a
    within-sample percentile of the gene (``kind='pctile'``). Maps to the
    matching count columns in cta_patient_counts.csv / cta_union_counts.csv."""

    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    @property
    def count_col(self):
        return f"n_gt{self.value}" if self.kind == "tpm" else f"n_p{self.value}"

    def union_col(self, drop_mage):
        base = f"n_any_gt{self.value}" if self.kind == "tpm" else f"n_any_p{self.value}"
        return base + ("_nomage" if drop_mage else "")

    @property
    def slug(self):
        return f"t{self.value}" if self.kind == "tpm" else f"p{self.value}"

    @property
    def label(self):
        return (f"> {self.value} TPM" if self.kind == "tpm"
                else f"≥ {self.value}th within-sample percentile")


def _available_thresholds():
    """TPM thresholds always; percentile thresholds only if the counts CSV
    carries their columns (i.e. cta_patient_counts was built with percentiles)."""
    cols = set(pd.read_csv(COUNTS, nrows=1).columns)
    thrs = [Thr("tpm", t) for t in TPM_THRESHOLDS if f"n_gt{t}" in cols]
    thrs += [Thr("pctile", q) for q in PERCENTILES if f"n_p{q}" in cols]
    return thrs
# {plot-key: (burden metric column, axis label)}
METRICS = {
    "us_incidence": ("us_incidence_pct", "US incidence"),
    "us_mortality": ("us_mortality_pct", "US deaths"),
    "world_incidence": ("world_incidence_pct", "worldwide incidence"),
    "world_mortality": ("world_mortality_pct", "worldwide deaths"),
}
# Subtype / aggregate cohorts whose patients are already counted by another
# cohort in the same burden category — drop them so a category's burden isn't
# double-weighted (the finer atom cohorts are kept; the rollups are dropped).
_REDUNDANT = {"BRCA_LumA", "BRCA_LumB", "BRCA_HER2", "BRCA_Basal", "BRCA_Normal",
              "HNSC_HPVpos", "HNSC_HPVneg",
              "SARC", "SARC_RMS", "SARC_LPS"}


def _is_mage(sym):
    return str(sym).upper().startswith(("MAGEA", "MAGEB", "MAGEC"))


def _prepare(drop_mage=False):
    """Return (counts, cat_n, breadth): per-(category,Symbol) hit counts joined
    to category patient totals, plus per-CTA breadth (# cancer types expressed in).
    With drop_mage, MAGE-A/B/C proteins are excluded (the 'without MAGE' panel)."""
    counts = pd.read_csv(COUNTS)
    counts = counts[~counts.cancer_code.isin(_REDUNDANT)].copy()
    if drop_mage:
        counts = counts[~counts.Symbol.map(_is_mage)].copy()
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
    # breadth: # burden CATEGORIES each CTA is detectable in (>25 TPM in >=1
    # patient). Category-level (not cohort-level) so the color matches the
    # "+N more" label, which lists contributing categories — the whole plot is
    # then in one unit (e.g. PRAME's many lung cohorts collapse to one "lung").
    breadth = (counts[counts.n_gt25 > 0].groupby("Symbol").category.nunique())
    return counts, cat_n, breadth


def _label(cats):
    """Contributing cancer types in descending order: list up to 4; if more than
    4 contribute, list the top 3 and count the rest ('+N more')."""
    cats = list(cats)
    if len(cats) <= 4:
        return ", ".join(cats)
    return ", ".join(cats[:3]) + f", +{len(cats) - 3} more"


def _addressable(counts, cat_n, threshold, burden_metric):
    """Per-CTA addressable % for one threshold + burden metric, with a label of
    the top-contributing cancer types. Returns a DataFrame sorted descending."""
    col = threshold.count_col
    g = counts.groupby(["category", "Symbol"], as_index=False)[col].sum()
    g["pooled_frac"] = g[col] / g.category.map(cat_n)
    burden = gsc.cancer_burden(metric=burden_metric)  # {category: pct of all cancer}
    g["contrib"] = g.category.map(lambda c: burden.get(c, 0.0)) * g.pooled_frac
    per = g.groupby("Symbol").contrib.sum().rename("addressable").reset_index()
    gp = g[g.contrib > 0].sort_values(["Symbol", "contrib"], ascending=[True, False])
    lab = gp.groupby("Symbol", sort=False).category.apply(_label).rename("label")
    per = per.merge(lab, on="Symbol", how="left").sort_values("addressable", ascending=False)
    per["label"] = per["label"].fillna("")
    return per, burden


def _union_addressable(threshold, burden_metric, cat_n, drop_mage=False):
    """Burden-weighted fraction of patients expressing >=1 CTA protein over the
    threshold (the whole-panel union — 'All CTAs'). Each patient is counted once
    however many CTAs they express. With drop_mage, uses the MAGE-excluded union."""
    u = pd.read_csv(UNION)
    u = u[~u.cancer_code.isin(_REDUNDANT)].copy()
    u["category"] = [gsc.burden_category(c) for c in u.cancer_code]
    u = u[u.category.notna()]
    col = threshold.union_col(drop_mage)
    cat_hits = u.groupby("category")[col].sum()
    burden = gsc.cancer_burden(metric=burden_metric)
    return sum(burden.get(c, 0.0) * (h / cat_n[c])
               for c, h in cat_hits.items() if c in cat_n and cat_n[c] > 0)


def _render(drop_mage, suffix, title_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    counts, cat_n, breadth = _prepare(drop_mage=drop_mage)
    n_cohorts = counts.cancer_code.nunique()
    n_cats = counts.category.nunique()
    bmax = int(breadth.max()) if len(breadth) else 1
    cmap = matplotlib.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=1, vmax=bmax)

    for mkey, (bcol, blabel) in METRICS.items():
        ceiling = sum(gsc.cancer_burden(metric=bcol).get(c, 0.0)
                      for c in counts.category.unique())
        for t in _available_thresholds():
            per, _ = _addressable(counts, cat_n, t, bcol)
            union = _union_addressable(t, bcol, cat_n, drop_mage=drop_mage)
            top = per.head(TOPN).iloc[::-1]  # ascending for barh (max on top)
            colors = [cmap(norm(int(breadth.get(s, 1)))) for s in top.Symbol]
            fig, ax = plt.subplots(figsize=(10, max(6, (len(top) + 2) * 0.27)))
            ax.barh(range(len(top)), top.addressable, color=colors)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.Symbol, fontsize=6)
            for y, (val, lab) in enumerate(zip(top.addressable, top.label)):
                ax.text(val, y, f" {lab}", va="center", fontsize=5, color="0.3")
            # whole-panel union ("All CTAs"): patient expresses >=1 CTA protein
            # over threshold, counted once however many CTAs they express.
            uy = len(top) + 0.8
            ax.barh(uy, union, color="#d4a017", edgecolor="black", height=0.9)
            ax.text(union, uy, f"  All CTAs ({union:.0f}%)", va="center",
                    fontsize=6, fontweight="bold", color="#7a5c00")
            ax.set_yticks(list(range(len(top))) + [uy])
            ax.set_yticklabels(list(top.Symbol) + ["All CTAs"], fontsize=6)
            ax.set_xlabel(f"% of annual {blabel} addressable "
                          f"(patient expresses this CTA {t.label})")
            ax.set_title(f"CTA population addressability{title_tag} — {blabel} "
                         f"({t.label})", fontsize=11)
            ax.set_xlim(0, max(top.addressable.max(), union) * 1.22)
            ax.grid(axis="x", alpha=0.3)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
            cb.set_label("# cancer categories expressed in", fontsize=7)
            ax.text(0.99, 0.01,
                    f"lower bound: {n_cohorts} per-sample cohorts → {n_cats} burden "
                    f"categories = {ceiling:.0f}% of {blabel} (uncovered types not scored)",
                    transform=ax.transAxes, ha="right", fontsize=6, color="gray")
            fig.tight_layout()
            fig.savefig(FIGDIR / f"cta_addressable_{mkey}_{t.slug}{suffix}.png", dpi=300)
            plt.close(fig)
            print(f"  {mkey} {t.slug}{suffix}: top CTA {per.iloc[0].Symbol} "
                  f"({per.iloc[0].addressable:.1f}% of {blabel}); ceiling {ceiling:.0f}%",
                  flush=True)


def _burden_category_plot():
    """Separate reference plot: the burden categories that our cohorts map into,
    their US incidence + mortality share, and which cohort codes land in each —
    so the grouping driving addressability is transparent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counts = pd.read_csv(COUNTS)
    counts = counts[~counts.cancer_code.isin(_REDUNDANT)].copy()
    counts["category"] = counts.cancer_code.map(
        lambda c: gsc.burden_category(c))
    counts = counts[counts.category.notna()]
    # member cohort codes per category (deduped, registry display where short)
    members = (counts.groupby("category").cancer_code.unique()
               .apply(lambda xs: sorted(set(xs))))
    inc = gsc.cancer_burden(metric="us_incidence_pct")
    mort = gsc.cancer_burden(metric="us_mortality_pct")
    cats = sorted(members.index, key=lambda c: inc.get(c, 0.0))
    y = np.arange(len(cats))
    h = 0.4
    fig, ax = plt.subplots(figsize=(11, max(6, len(cats) * 0.42)))
    ax.barh(y + h / 2, [inc.get(c, 0.0) for c in cats], height=h,
            color="#4878a8", label="US incidence")
    ax.barh(y - h / 2, [mort.get(c, 0.0) for c in cats], height=h,
            color="#c44e52", label="US deaths")
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=7)
    for i, c in enumerate(cats):
        n_coh = len(members[c])
        codes = ", ".join(members[c][:6]) + (f", +{n_coh - 6}" if n_coh > 6 else "")
        x = max(inc.get(c, 0.0), mort.get(c, 0.0))
        ax.text(x + 0.15, i, f"{codes}", va="center", fontsize=5, color="0.35")
    ax.set_xlabel("% of annual US cancer cases / deaths")
    ax.set_title("Cancer burden by category (US incidence & deaths)", fontsize=11)
    ax.set_xlim(0, max(max(inc.values()), max(mort.values())) * 1.35)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "cta_burden_categories.png", dpi=300)
    plt.close(fig)
    print(f"  burden-category reference: {len(cats)} categories "
          f"({counts.cancer_code.nunique()} cohorts)", flush=True)


def main():
    global COUNTS, UNION, FIGDIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="run_<ts>/ holding cta_patient_counts.csv + "
                         "cta_union_counts.csv (default: latest run under "
                         "analyses/outputs). Plots are written into this dir.")
    args = ap.parse_args()
    run = args.run_dir or latest_run_dir(OUT)
    if run is None:
        raise SystemExit(
            "no run_<ts>/ with cta_patient_counts.csv found — run "
            "cta_patient_counts.py first (or pass --run-dir).")
    COUNTS = run / "cta_patient_counts.csv"
    UNION = run / "cta_union_counts.csv"
    FIGDIR = run
    print(f"reading tables + writing plots in {FIGDIR}", flush=True)
    _render(drop_mage=False, suffix="", title_tag="")
    _render(drop_mage=True, suffix="_noMAGE", title_tag=" (excl. MAGE-A/B/C)")
    _burden_category_plot()
    print(f"done -> 24 addressability plots + 1 burden-category plot in {FIGDIR}",
          flush=True)


if __name__ == "__main__":
    main()
