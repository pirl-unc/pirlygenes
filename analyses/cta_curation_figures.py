"""Regenerate the CTA *curation* documentation figures (the five embedded in
``docs/cta-curation.md``) from the tsarina CTA evidence table.

The original generator for these figures was never committed (they landed as
static PNGs in d086cf5), so this reconstructs them from the documented filter
logic against the single source of truth: ``tsarina.CTA_detailed_evidence()``.
Faithful to the docs, but styling may differ from the lost originals.

Five figures, written with the doc-referenced (hyphenated) filenames so the
``--promote-docs`` step in ``regenerate_plots.py`` can drop them straight into
``docs/``:

    cta-source-venn.png        overlap of the primary source databases
    cta-filter-funnel.png      kept vs dropped per source
    cta-filter-outcome.png     kept-confident / kept-weak / excluded per source
    cta-deflated-frac-dist.png distribution of the deflated reproductive fraction
    cta-protein-vs-rna.png     deflated RNA fraction by protein-reliability tier

Rides the shared analyses run layout (writes into ``<run>/cta_curation/`` when
invoked by ``regenerate_plots.py``)::

    python analyses/cta_curation_figures.py            # -> analyses/outputs/run_*/
    python analyses/cta_curation_figures.py --out-dir <run> --run-name cta_curation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _run_layout import add_layout_args, resolve_dirs

import tsarina

# Primary gene-contributing sources (the cross-reference tag ``daSilva2017`` —
# the full 1,103-gene set — and the tiny ``paralog:*`` tags are not primary
# sources, per docs/cta-curation.md).
PRIMARY_SOURCES = {
    "CTpedia": lambda tags: "CTpedia" in tags,
    "CTexploreR": lambda tags: "CTexploreR_CT" in tags or "CTexploreR_CTP" in tags,
    "daSilva2017_protein": lambda tags: "daSilva2017_protein" in tags,
    "placental_antigen": lambda tags: "placental_antigen" in tags,
}

# Deflated-RNA-fraction threshold each protein-reliability tier must clear
# (docs/cta-curation.md "Filter logic").
RELIABILITY_THRESHOLD = {
    "Enhanced": 0.80, "Supported": 0.90, "Approved": 0.95,
    "Uncertain": 0.98, "no data": 0.98,
}
RELIABILITY_ORDER = ["no data", "Uncertain", "Approved", "Supported", "Enhanced"]

KEPT = "#2a7f4f"
DROP = "#b0b0b0"
WEAK = "#f0c419"


def _tag_sets(df):
    """{source_label: set(Ensembl_Gene_ID)} for the primary sources."""
    out = {name: set() for name in PRIMARY_SOURCES}
    for ensg, raw in zip(df["Ensembl_Gene_ID"], df["source_databases"].fillna("")):
        tags = {t.strip() for t in str(raw).split(";") if t.strip()}
        for name, pred in PRIMARY_SOURCES.items():
            if pred(tags):
                out[name].add(ensg)
    return out


def _save(fig, figdir: Path, name: str) -> None:
    path = figdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}", flush=True)


def fig_source_venn(df, figdir):
    sets = _tag_sets(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    venn_sets = {k: sets[k] for k in ("CTpedia", "CTexploreR", "daSilva2017_protein")}
    try:
        from matplotlib_venn import venn3
        venn3([venn_sets["CTpedia"], venn_sets["CTexploreR"],
               venn_sets["daSilva2017_protein"]],
              set_labels=("CTpedia", "CTexploreR", "da Silva 2017\n(protein)"),
              ax=ax)
    except Exception:  # noqa: BLE001 — no venn lib: fall back to a size bar
        names = list(venn_sets)
        ax.barh(names, [len(venn_sets[n]) for n in names], color=KEPT)
        ax.set_xlabel("genes")
    extra = len(sets["placental_antigen"])
    ax.set_title(
        f"CTA source overlap (primary databases)\n"
        f"+{extra} placental-antigen genes folded in")
    _save(fig, figdir, "cta-source-venn.png")


def _per_source_counts(df):
    sets = _tag_sets(df)
    rows = []
    for name, members in sets.items():
        sub = df[df["Ensembl_Gene_ID"].isin(members)]
        passes = sub["passes_filters"].fillna(False).astype(bool)
        weak = sub["never_expressed"].fillna(False).astype(bool)
        rows.append({
            "source": name,
            "total": len(sub),
            "kept_confident": int((passes & ~weak).sum()),
            "kept_weak": int((passes & weak).sum()),
            "excluded": int((~passes).sum()),
        })
    # largest source first
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def fig_filter_funnel(df, figdir):
    rows = _per_source_counts(df)
    labels = [r["source"] for r in rows]
    kept = np.array([r["kept_confident"] + r["kept_weak"] for r in rows])
    dropped = np.array([r["excluded"] for r in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 2))
    ax.barh(y, kept, color=KEPT, label="passes filter")
    ax.barh(y, dropped, left=kept, color=DROP, label="excluded")
    for i, r in enumerate(rows):
        ax.text(r["total"] + 1, i, f"{kept[i]}/{r['total']}",
                va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("genes in source")
    ax.set_title("CTA filter funnel by source (kept vs excluded)")
    ax.legend(loc="lower right")
    _save(fig, figdir, "cta-filter-funnel.png")


def fig_filter_outcome(df, figdir):
    rows = _per_source_counts(df)
    labels = [r["source"] for r in rows]
    conf = np.array([r["kept_confident"] for r in rows])
    weak = np.array([r["kept_weak"] for r in rows])
    excl = np.array([r["excluded"] for r in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 2))
    ax.barh(y, conf, color=KEPT, label="kept (HPA-confident)")
    ax.barh(y, weak, left=conf, color=WEAK, label="kept (weak evidence)")
    ax.barh(y, excl, left=conf + weak, color=DROP, label="excluded")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("genes")
    ax.set_title("CTA filter outcome by source")
    ax.legend(loc="lower right")
    _save(fig, figdir, "cta-filter-outcome.png")


def fig_deflated_dist(df, figdir):
    frac = df["rna_deflated_reproductive_frac"].astype(float).to_numpy()
    passes = df["passes_filters"].fillna(False).astype(bool).to_numpy()
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([frac[passes], frac[~passes]], bins=bins, stacked=True,
            color=[KEPT, DROP], label=["passes filter", "excluded"])
    for thr in (0.80, 0.90, 0.95, 0.98):
        ax.axvline(thr, color="#555", ls="--", lw=0.8)
        ax.text(thr, ax.get_ylim()[1] * 0.97, f"{thr:.2f}", rotation=90,
                va="top", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("deflated reproductive fraction")
    ax.set_ylabel("CTA genes")
    ax.set_title("Deflated reproductive-fraction distribution")
    ax.legend()
    _save(fig, figdir, "cta-deflated-frac-dist.png")


def fig_protein_vs_rna(df, figdir):
    frac = df["rna_deflated_reproductive_frac"].astype(float)
    rel = df["protein_reliability"].fillna("no data").astype(str)
    passes = df["passes_filters"].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for i, tier in enumerate(RELIABILITY_ORDER):
        m = rel == tier
        if not m.any():
            continue
        x = i + (rng.random(int(m.sum())) - 0.5) * 0.6  # jitter
        ax.scatter(x, frac[m], s=14, alpha=0.6,
                   c=np.where(passes[m], KEPT, DROP))
        thr = RELIABILITY_THRESHOLD.get(tier)
        if thr is not None:
            ax.plot([i - 0.4, i + 0.4], [thr, thr], color="#c0392b", lw=2)
    ax.set_xticks(range(len(RELIABILITY_ORDER)), RELIABILITY_ORDER)
    ax.set_xlabel("protein reliability (HPA IHC)")
    ax.set_ylabel("deflated reproductive fraction")
    ax.set_title("Protein reliability vs RNA fraction\n"
                 "(red line = required RNA threshold for that tier)")
    _save(fig, figdir, "cta-protein-vs-rna.png")


def build(figdir: Path) -> None:
    df = tsarina.CTA_detailed_evidence().copy()
    print(f"CTA curation figures from {len(df)} evidence rows -> {figdir}")
    fig_source_venn(df, figdir)
    fig_filter_funnel(df, figdir)
    fig_filter_outcome(df, figdir)
    fig_deflated_dist(df, figdir)
    fig_protein_vs_rna(df, figdir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, Path(__file__).resolve().parent / "outputs")
    build(figdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
