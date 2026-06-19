"""CTA curation documentation figures, over the packaged CTA evidence table.

The five figures embedded in ``docs/cta-curation.md`` describe how the CTA panel
is built and filtered: source overlap, the per-source filter funnel/outcome, the
deflated reproductive-fraction distribution, and the protein-reliability-vs-RNA
tiered thresholds. They are derived purely from ``tsarina.CTA_detailed_evidence``
(the curation single source of truth), so they belong with the cohort-level plot
surface rather than the research ``analyses/`` scripts.

Exposed two ways over the *same* code:
  * ``pirlygenes plot cta-curation --out <dir>`` (CLI; ships in the wheel)
  * ``analyses/cta_curation_figures.py`` — a thin batch-driver wrapper that calls
    :func:`render` so the figures ride the shared analyses run layout and the
    ``regenerate_plots.py --promote-docs`` flow.

``matplotlib_venn`` (the ``pirlygenes[viz]`` extra) is used for the source-overlap
venn when available; without it that one figure degrades to a source-size bar, so
it stays an optional dependency rather than a hard one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Primary gene-contributing sources. The cross-reference tag ``daSilva2017`` (the
# full 1,103-gene set) and the tiny ``paralog:*`` tags are not primary sources
# (see docs/cta-curation.md).
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

# doc-referenced (hyphenated) filenames, so regenerate_plots.py --promote-docs
# can drop them straight into docs/.
FILENAMES = {
    "source_venn": "cta-source-venn.png",
    "filter_funnel": "cta-filter-funnel.png",
    "filter_outcome": "cta-filter-outcome.png",
    "deflated_dist": "cta-deflated-frac-dist.png",
    "protein_vs_rna": "cta-protein-vs-rna.png",
}


def _evidence():
    import tsarina

    return tsarina.CTA_detailed_evidence().copy()


def _tag_sets(df):
    """{source_label: set(Ensembl_Gene_ID)} for the primary sources."""
    out = {name: set() for name in PRIMARY_SOURCES}
    for ensg, raw in zip(df["Ensembl_Gene_ID"], df["source_databases"].fillna("")):
        tags = {t.strip() for t in str(raw).split(";") if t.strip()}
        for name, pred in PRIMARY_SOURCES.items():
            if pred(tags):
                out[name].add(ensg)
    return out


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
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def _save(fig, path, plt):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_source_venn(df, path, plt):
    sets = _tag_sets(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    keys = ("CTpedia", "CTexploreR", "daSilva2017_protein")
    placental = len(sets["placental_antigen"])
    # Only the missing-library case falls back; a real venn3 rendering error is
    # left to propagate rather than be silently masked as a bar.
    try:
        from matplotlib_venn import venn3
    except ImportError:
        ax.barh(list(keys), [len(sets[k]) for k in keys], color=KEPT)
        ax.set_xlabel("genes")
        ax.set_title(
            "CTA source sizes — install matplotlib_venn (or pirlygenes[viz])\n"
            f"for the overlap venn; +{placental} placental-antigen genes folded in")
    else:
        venn3([sets[k] for k in keys],
              set_labels=("CTpedia", "CTexploreR", "da Silva 2017\n(protein)"),
              ax=ax)
        ax.set_title("CTA source overlap (primary databases)\n"
                     f"+{placental} placental-antigen genes folded in")
    _save(fig, path, plt)


def _fig_filter_funnel(df, path, plt):
    rows = _per_source_counts(df)
    labels = [r["source"] for r in rows]
    kept = np.array([r["kept_confident"] + r["kept_weak"] for r in rows])
    dropped = np.array([r["excluded"] for r in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 2))
    ax.barh(y, kept, color=KEPT, label="passes filter")
    ax.barh(y, dropped, left=kept, color=DROP, label="excluded")
    for i, r in enumerate(rows):
        ax.text(r["total"] + 1, i, f"{kept[i]}/{r['total']}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("genes in source")
    ax.set_title("CTA filter funnel by source (kept vs excluded)")
    ax.legend(loc="lower right")
    _save(fig, path, plt)


def _fig_filter_outcome(df, path, plt):
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
    _save(fig, path, plt)


def _fig_deflated_dist(df, path, plt):
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
    _save(fig, path, plt)


def _fig_protein_vs_rna(df, path, plt):
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
        ax.scatter(x, frac[m], s=14, alpha=0.6, c=np.where(passes[m], KEPT, DROP))
        thr = RELIABILITY_THRESHOLD.get(tier)
        if thr is not None:
            ax.plot([i - 0.4, i + 0.4], [thr, thr], color="#c0392b", lw=2)
    ax.set_xticks(range(len(RELIABILITY_ORDER)), RELIABILITY_ORDER)
    ax.set_xlabel("protein reliability (HPA IHC)")
    ax.set_ylabel("deflated reproductive fraction")
    ax.set_title("Protein reliability vs RNA fraction\n"
                 "(red line = required RNA threshold for that tier)")
    _save(fig, path, plt)


_BUILDERS = {
    "source_venn": _fig_source_venn,
    "filter_funnel": _fig_filter_funnel,
    "filter_outcome": _fig_filter_outcome,
    "deflated_dist": _fig_deflated_dist,
    "protein_vs_rna": _fig_protein_vs_rna,
}


def render(out_dir="cta_curation_out") -> dict:
    """Write the five CTA curation figures into ``out_dir`` and return
    ``{"n_genes": int, "paths": {key: Path}}``. Reconstructed from
    ``tsarina.CTA_detailed_evidence`` per docs/cta-curation.md."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _evidence()
    paths = {}
    for key, builder in _BUILDERS.items():
        path = out / FILENAMES[key]
        builder(df, path, plt)
        paths[key] = path
    return {"n_genes": int(len(df)), "paths": paths}
