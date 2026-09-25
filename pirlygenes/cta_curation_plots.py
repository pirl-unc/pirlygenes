"""CTA curation documentation figures, over the packaged CTA evidence table.

The figures embedded in ``docs/cta-curation.md`` describe how the CTA panel
is built and filtered: source overlap, the per-source filter funnel/outcome, the
deflated reproductive-fraction distribution, and the protein-reliability-vs-RNA
tiered thresholds. They use oncoref's packaged candidate table and public CTA
sets (the gene-set authority), so they belong with the cohort-level plot
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

from pirlygenes.figure_export import save_figure

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
from oncoref.cta_tissues import HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS

# Primary gene-contributing sources. The cross-reference tag ``daSilva2017`` (the
# full 1,103-gene set) and the tiny ``paralog:*`` tags are not primary sources
# (see docs/cta-curation.md).
PRIMARY_SOURCES = {
    "CTpedia": lambda tags: "CTpedia" in tags,
    "CTexploreR": lambda tags: "CTexploreR_CT" in tags or "CTexploreR_CTP" in tags,
    "daSilva2017_protein": lambda tags: "daSilva2017_protein" in tags,
    "placental_antigen": lambda tags: "placental_antigen" in tags,
    "Gong 2021": lambda tags: bool(tags & {"Gong2021_placenta_PC", "Gong2021_placenta_ncRNA"}),
    "Bradley 2020": lambda tags: "Bradley2020_CPA" in tags,
}

SOURCE_LABELS = {
    "CTpedia": "CTpedia", "CTexploreR": "CTexploreR",
    "daSilva2017_protein": "da Silva 2017", "placental_antigen": "Prior placental list",
    "Gong 2021": "Gong 2021", "Bradley 2020": "Bradley 2020",
}

# Deflated-RNA-fraction threshold each protein-reliability tier must clear
# (docs/cta-curation.md "Filter logic").

RELIABILITY_THRESHOLD = {
    **{k: v for k, v in HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS.items() if k != "Missing"},
    "no data": HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS["Missing"],
}
RELIABILITY_ORDER = ["no data", "Uncertain", "Approved", "Supported", "Enhanced"]

KEPT = "#2a7f4f"
DROP = "#b0b0b0"
WEAK = "#f0c419"

# doc-referenced (hyphenated) filenames, so regenerate_plots.py --promote-docs
# can drop them straight into docs/.
FILENAMES = {
    "source_venn": "cta-source-venn.png",
    "stage_funnel": "cta-stage-funnel.png",
    "filter_funnel": "cta-filter-funnel.png",
    "filter_outcome": "cta-filter-outcome.png",
    "deflated_dist": "cta-deflated-frac-dist.png",
    "protein_vs_rna": "cta-protein-vs-rna.png",
}

PUBLICATION_FILENAMES = {
    "source_overlap": "cta-source-overlap.png",
    "placental_source_overlap": "cta-placental-source-overlap.png",
    "publication_funnel": "cta-publication-funnel.png",
    "placental_evidence_coverage": "cta-placental-evidence-coverage.png",
}


def publication_data_available():
    """Keep the plotting API usable with the released, pre-publication pin."""
    import importlib.util

    return importlib.util.find_spec("oncoref.cta_sources") is not None


def _evidence():
    from oncoref.cta import cta_evidence
    from oncoref.load_dataset import get_data

    # The public evidence view already removes non-CTA families. Join its
    # specificity annotations onto the raw nominations to preserve that stage.
    raw = get_data("cancer-testis-antigens").copy()
    reviewed = cta_evidence()
    columns = ["Ensembl_Gene_ID", *[c for c in reviewed if c.startswith("specificity_")]]
    return raw.merge(reviewed[columns], on="Ensembl_Gene_ID", how="left", validate="one_to_one")


def _bool_series(series):
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def stage_membership(df=None):
    """Auditable gene-level stages, ending at the exact shipped default set.

    Raw evidence must precede the public unfiltered set: that accessor already
    removes histones and alpha-tubulins. Specificity audits may override the HPA
    gate; fail explicitly if a future release makes the stages non-nested.
    """
    from oncoref import cta

    df = _evidence() if df is None else df.copy()
    ids = df.Ensembl_Gene_ID.astype(str)
    if ids.duplicated().any():
        raise ValueError("CTA candidate table must contain one row per gene")
    df["nominated"] = True
    df["non_cta_removed"] = ids.isin(cta.cta_unfiltered_gene_ids())
    df["hpa_restriction"] = df.non_cta_removed & _bool_series(df.passes_filters)
    df["default_panel"] = ids.isin(cta.cta_gene_ids())
    if not set(cta.cta_gene_ids()) <= set(ids):
        raise ValueError("Default CTA genes missing from candidate table")
    if (df.default_panel & ~df.hpa_restriction).any():
        raise ValueError("Specificity overrides require a branching CTA funnel")
    return df


STAGES = {
    "nominated": "Nominated source union",
    "non_cta_removed": "Non-CTA families removed",
    "hpa_restriction": "HPA reproductive restriction",
    "default_panel": "Default specificity + expression",
}


def stage_counts(df=None):
    membership = stage_membership(df)
    rows, previous = [], len(membership)
    for key, label in STAGES.items():
        n = int(membership[key].sum())
        rows.append({"stage": key, "label": label, "remaining": n,
                     "dropped": previous - n})
        previous = n
    return rows


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
    """Partition each source by reviewed membership, retaining HPA-only passes."""
    membership = stage_membership(df)
    sets = _tag_sets(membership)
    rows = []
    for name, members in sets.items():
        if not members:
            continue
        sub = membership[membership.Ensembl_Gene_ID.isin(members)]
        rows.append({
            "source": name,
            "total": len(sub),
            "default_panel": int(sub.default_panel.sum()),
            "hpa_pass_outside_default": int((sub.hpa_restriction & ~sub.default_panel).sum()),
            "family_or_hpa_excluded": int((~sub.hpa_restriction).sum()),
        })
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def _save(fig, path, plt):
    save_figure(fig, path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fig_stage_funnel(df, path, plt):
    stages = stage_counts(df)
    remaining = np.array([s["remaining"] for s in stages])
    dropped = np.array([s["dropped"] for s in stages])
    y = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.barh(y, remaining, color=KEPT, height=0.62, label="retained")
    ax.barh(y, dropped, left=remaining, color=DROP, height=0.62,
            label="removed at this step")
    for i, (n, d) in enumerate(zip(remaining, dropped)):
        ax.text(n / 2, i, f"{n:,}", ha="center", va="center", color="white",
                fontweight="bold")
        if d:
            ax.text(n + d + remaining[0] * .02, i, f"−{d:,}", va="center")
    ax.set_yticks(y, [s["label"] for s in stages])
    ax.invert_yaxis()
    ax.set_xlim(0, remaining[0] * 1.16)
    ax.set_xlabel("Unique candidate genes")
    ax.set_title("CTA nomination to the default antigen panel")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, path, plt)


def _fig_source_venn(df, path, plt):
    sets = _tag_sets(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    keys = ("CTpedia", "CTexploreR", "daSilva2017_protein")
    # Only the missing-library case falls back; a real venn3 rendering error is
    # left to propagate rather than be silently masked as a bar.
    try:
        from matplotlib_venn import venn3
    except ImportError:
        ax.barh(list(keys), [len(sets[k]) for k in keys], color=KEPT)
        ax.set_xlabel("genes")
        ax.set_title(
            "CTA source sizes — install matplotlib_venn (or pirlygenes[viz])\n"
            "for the overlap venn")
    else:
        venn3([sets[k] for k in keys],
              set_labels=("CTpedia", "CTexploreR", "da Silva 2017\n(protein)"),
              ax=ax)
        ax.set_title("CTA source overlap: three historical databases\n"
                     "Additional nomination sources shown in the source funnel")
    _save(fig, path, plt)


def _fig_filter_funnel(df, path, plt):
    membership = stage_membership(df)
    sets = _tag_sets(df)
    sets = {name: ids for name, ids in sets.items() if ids}
    sets["Paralog / cell-type additions"] = set(df.Ensembl_Gene_ID) - set.union(*sets.values())
    sets = dict(sorted(sets.items(), key=lambda pair: -len(pair[1])))
    labels = [SOURCE_LABELS.get(s, s) for s in sets]
    nominated = np.array([len(ids) for ids in sets.values()])
    hpa = np.array([int((membership.Ensembl_Gene_ID.isin(ids)
                        & membership.hpa_restriction).sum()) for ids in sets.values()])
    kept = np.array([int((membership.Ensembl_Gene_ID.isin(ids)
                         & membership.default_panel).sum()) for ids in sets.values()])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(labels) + 2.9))
    ax.barh(y, kept, color=KEPT, label="default panel", height=.65)
    ax.barh(y, hpa - kept, left=kept, color=WEAK,
            label="HPA pass, outside default", height=.65)
    ax.barh(y, nominated - hpa, left=hpa, color=DROP,
            label="family / HPA exclusion", height=.65)
    for i, total in enumerate(nominated):
        ax.text(total + nominated.max() * .02, i, f"{kept[i]}/{total}", va="center")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, nominated.max() * 1.28)
    ax.set_xlabel("Genes per nomination source (sources overlap; do not sum)")
    ax.set_title("CTA nomination outcomes by source")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, path, plt)


def _fig_filter_outcome(df, path, plt):
    rows = _per_source_counts(df)
    labels = [SOURCE_LABELS[r["source"]] for r in rows]
    kept = np.array([r["default_panel"] for r in rows])
    outside = np.array([r["hpa_pass_outside_default"] for r in rows])
    excl = np.array([r["family_or_hpa_excluded"] for r in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(labels) + 2.5))
    ax.barh(y, kept, color=KEPT, label="default panel")
    ax.barh(y, outside, left=kept, color=WEAK, label="HPA pass, outside default")
    ax.barh(y, excl, left=kept + outside, color=DROP, label="family / HPA exclusion")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Genes per nomination source (sources overlap; do not sum)")
    ax.set_title("CTA filter outcome by source")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=1)
    fig.tight_layout()
    _save(fig, path, plt)


def _fig_deflated_dist(df, path, plt):
    frac = df["rna_deflated_reproductive_frac"].astype(float).to_numpy()
    passes = _bool_series(df["passes_filters"]).to_numpy()
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([frac[passes], frac[~passes]], bins=bins, stacked=True,
            color=[KEPT, DROP], label=["passes HPA filter", "fails HPA filter"])
    for thr in sorted(set(RELIABILITY_THRESHOLD.values())):
        ax.axvline(thr, color="#555", ls="--", lw=0.8)
        ax.text(thr, ax.get_ylim()[1] * 0.97, f"{thr:.2f}", rotation=90,
                va="top", ha="right", fontsize="small", color="#555")
    ax.set_xlabel("deflated reproductive fraction")
    ax.set_ylabel("CTA genes")
    ax.set_title("Deflated reproductive-fraction distribution (HPA gate only)")
    ax.legend()
    _save(fig, path, plt)


def _fig_protein_vs_rna(df, path, plt):
    frac = df["rna_deflated_reproductive_frac"].astype(float)
    rel = df["protein_reliability"].fillna("no data").astype(str)
    passes = _bool_series(df["passes_filters"])
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
    ax.set_title("Protein reliability vs RNA fraction (HPA gate only)\n"
                 "(red line = required RNA threshold for that tier)")
    _save(fig, path, plt)


def source_overlap_counts(df=None):
    """Pairwise source intersections over candidate rows and the public default."""
    import pandas as pd

    membership = stage_membership(df)
    sets = {name: ids for name, ids in _tag_sets(membership).items() if ids}
    defaults = set(membership.loc[membership.default_panel, "Ensembl_Gene_ID"])
    return pd.DataFrame([
        {"source_a": a, "source_b": b, "candidate_overlap": len(x & y),
         "default_overlap": len(x & y & defaults)}
        for a, x in sets.items() for b, y in sets.items()
    ])


def _fig_source_overlap(df, path, plt):
    counts = source_overlap_counts(df)
    keys = list(counts.source_a.unique())
    labels = [SOURCE_LABELS[k] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.5))
    for ax, field, title in zip(axes, ("candidate_overlap", "default_overlap"),
                                 ("Nominated candidate table", "Retained in default panel")):
        values = counts.pivot(index="source_a", columns="source_b", values=field).loc[keys, keys].to_numpy()
        ax.imshow(values, cmap="Blues", vmin=0, vmax=values.max())
        for (i, j), n in np.ndenumerate(values):
            ax.text(j, i, str(n), ha="center", va="center",
                    color="white" if n > values.max() * .55 else "#182c3e")
        ax.set_xticks(range(len(keys)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(keys)), labels)
        ax.set_title(title)
        ax.spines[:].set_visible(False)
    fig.suptitle("CTA source overlap", fontsize=14)
    fig.text(.5, .018, "Cells count shared genes; diagonals show source totals. Sources overlap and must not be summed.\n"
             "Gong includes S5 and S6 tags present in the candidate table; complete published lists are shown in the intake funnel.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .96), w_pad=2.5)
    _save(fig, path, plt)


def placental_source_sets(df=None):
    """Current protein-coding identities from full published lists and prior tag."""
    from oncoref.cta_sources import BRADLEY, GONG_PC, GONG_NC, publication_membership

    df = _evidence() if df is None else df
    refs = publication_membership()
    coding = refs[refs.biotype.eq("protein_coding")]
    return {
        "Gong 2021\nS5 + S6": set(coding.loc[coding.source_tag.isin([GONG_PC, GONG_NC]), "Ensembl_Gene_ID"]),
        "Bradley 2020\nFigure 3": set(coding.loc[coding.source_tag.eq(BRADLEY), "Ensembl_Gene_ID"]),
        "Prior placental\nnominations": _tag_sets(df[df.biotype.eq("protein_coding")])["placental_antigen"],
    }


def _fig_placental_source_overlap(df, path, plt):
    sets = placental_source_sets(df)
    fig, ax = plt.subplots(figsize=(9, 6.8))
    try:
        from matplotlib_venn import venn3
    except ImportError:
        ax.barh(list(sets), [len(ids) for ids in sets.values()], color=KEPT)
        ax.set_xlabel("Source sizes (install matplotlib_venn to show intersections)")
    else:
        venn3(list(sets.values()), set_labels=tuple(sets), ax=ax,
              set_colors=("#4575a5", "#d09047", "#63977a"), alpha=.55)
    ax.set_title("Placental nomination overlap\nMapped, currently protein-coding genes")
    fig.text(.5, .05, "Gong: 70 coding genes (69 from S5; ERVH48-1 from S6). Bradley: 10 candidates.\n"
             "Gong supports 16 of 19 prior nominations, including 8 of the 9 retained genes.\n"
             "Source membership is nomination evidence, not proof of tumor antigenicity.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, 1))
    _save(fig, path, plt)


def _fig_publication_funnel(df, path, plt):
    from oncoref.cta_sources import BRADLEY, GONG_PC, GONG_NC, intake_counts

    counts = intake_counts()
    tags = (GONG_PC, GONG_NC, BRADLEY)
    titles = ("Gong 2021 · Supplementary Data 5\nPublished as protein-coding",
              "Gong 2021 · Supplementary Data 6\nPublished as noncoding",
              "Bradley 2020 · Figure 3\nCancer–placenta candidates")
    labels = ("Published nominations", "Canonical ID mapped", "Currently protein-coding",
              "Non-CTA families removed", "HPA reproductive restriction", "Default specificity + expression")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), sharey=True)
    for ax, tag, title in zip(axes, tags, titles):
        group = counts[counts.source_tag.eq(tag)]
        n, d = group.remaining.to_numpy(), group.dropped.to_numpy()
        y = np.arange(len(labels))
        ax.barh(y, n, height=.62, color=KEPT, label="retained")
        ax.barh(y, d, left=n, height=.62, color=DROP, label="removed at this step")
        for i, (kept, dropped) in enumerate(zip(n, d)):
            label = f"{kept}" + (f" (−{dropped})" if dropped else "")
            ax.text(kept + dropped + n[0] * .025, i, label, va="center", fontsize=9)
        ax.set_xlim(0, n[0] * 1.44)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Genes within this source")
        ax.set_title(title, fontsize=11, pad=15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Published nominations to the default CTA panel", fontsize=14)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.57, .08), ncol=2)
    fig.text(.57, .025, "S6 retains ERVH48-1, now protein-coding. Unmapped and noncoding entries remain in the source audit.\n"
             "The three lists overlap; their counts are not additive. Bradley validates VGLL1, which fails our HPA restriction gate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .18, 1, .93), w_pad=1.8)
    _save(fig, path, plt)


def _fig_placental_evidence_coverage(df, path, plt):
    from oncoref.cta_sources import placental_source_coverage
    from matplotlib.colors import ListedColormap

    data = placental_source_coverage().sort_values("Symbol")
    papers = {
        "Gong 2021": {"Gong2021_placenta_PC", "Gong2021_placenta_ncRNA"},
        "Bradley 2020": {"Bradley2020_CPA"},
        "Rull 2005": {"Rull2005_CGB_placenta"},
        "Rull 2008*": {"Rull2008_CGB_RNA"},
        "Kubiczak 2013*": {"Kubiczak2013_CGB_ovarian"},
        "Białas 2020": {"Bialas2020_CGB_cancer"},
        "McKellar 2025†": {"McKellar2025_CGB7_cancer"},
    }
    tags = data.publication_sources.str.split(";").map(set)
    values = np.array([[bool(t & p) for p in papers.values()] for t in tags], dtype=int)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.imshow(values, cmap=ListedColormap(["#f0f0f0", "#398c70"]), vmin=0, vmax=1, aspect="auto")
    for (i, j), present in np.ndenumerate(values):
        if present:
            ax.text(j, i, "●", color="white", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(len(papers)), papers, rotation=35, ha="left")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_yticks(range(len(data)), [r.Symbol + ("  ✓" if r.default_panel else "") for r in data.itertuples()])
    ax.spines[:].set_visible(False)
    ax.set_title("All 19 earlier placental nominations now have publication provenance", pad=88, fontsize=12)
    fig.text(.5, .025, "✓ Retained in default CTA panel (9/19). Blank cells mean no record in these curated source rows.\n"
             "* Combined CGB1/CGB2 assays, not separate positives. † Preprint.\n"
             "Expression / nomination evidence is distinct from gene-specific antigen validation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, 1))
    _save(fig, path, plt)


_BUILDERS = {
    "source_venn": _fig_source_venn,
    "stage_funnel": _fig_stage_funnel,
    "filter_funnel": _fig_filter_funnel,
    "filter_outcome": _fig_filter_outcome,
    "deflated_dist": _fig_deflated_dist,
    "protein_vs_rna": _fig_protein_vs_rna,
}

_PUBLICATION_BUILDERS = {
    "source_overlap": _fig_source_overlap,
    "placental_source_overlap": _fig_placental_source_overlap,
    "publication_funnel": _fig_publication_funnel,
    "placental_evidence_coverage": _fig_placental_evidence_coverage,
}


def _curation_provenance(membership, stages, *, kinds, font_scale):
    import oncoref
    from oncoref.version import DATA_VERSION as ONCOREF_DATA_VERSION
    from pirlygenes.version import __version__

    inputs = ["cancer-testis-antigens.csv", "cta-specificity-audit.csv"]
    if publication_data_available():
        inputs.extend(["cta-publication-sources.csv", "cta-publication-membership.csv",
                       "cta-gene-publication-evidence.csv"])
    gene_ids = sorted(membership.loc[membership.default_panel, "Ensembl_Gene_ID"])
    return {
        "pirlygenes_version": __version__,
        "oncoref_version": oncoref.__version__,
        "oncoref_data_version": ONCOREF_DATA_VERSION,
        "input_sha256": {
            name: hashlib.sha256(files("oncoref").joinpath("data").joinpath(name).read_bytes()).hexdigest()
            for name in inputs
        },
        "default_panel_definition": "oncoref.cta.cta_gene_ids() with default arguments",
        "default_gene_ids": gene_ids,
        "default_panel_sha256": hashlib.sha256(("\n".join(gene_ids) + "\n").encode()).hexdigest(),
        "stage_counts": stages,
        "figure_kinds": list(kinds),
        "font_scale": font_scale,
        "png_dpi": 300,
        "pdf_format": "vector",
    }


def render(out_dir="cta_curation_out", *, kinds=None, font_scale=1.0) -> dict:
    """Write figures plus auditable stage counts and gene membership tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _evidence()
    paths = {}
    builders = dict(_BUILDERS)
    filenames = dict(FILENAMES)
    if publication_data_available():
        builders.update(_PUBLICATION_BUILDERS)
        filenames.update(PUBLICATION_FILENAMES)
    with matplotlib.rc_context({"font.size": 10 * font_scale,
                                "axes.titlesize": 12 * font_scale,
                                "axes.labelsize": 10 * font_scale,
                                "xtick.labelsize": 10 * font_scale,
                                "ytick.labelsize": 10 * font_scale,
                                "legend.fontsize": 9 * font_scale}):
        for key in (kinds if kinds is not None else builders):
            path = out / filenames[key]
            builders[key](df, path, plt)
            paths[key] = path
    import pandas as pd

    stages = stage_counts(df)
    pd.DataFrame(stages).to_csv(out / "cta-stage-counts.csv", index=False)
    pd.DataFrame(_per_source_counts(df)).to_csv(out / "cta-source-outcome-counts.csv", index=False)
    membership = stage_membership(df)
    membership.to_csv(out / "cta-stage-membership.csv", index=False)
    provenance = _curation_provenance(membership, stages, kinds=paths, font_scale=font_scale)
    (out / "cta-curation-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    membership[membership.source_databases.fillna("").str.split(";").map(
        lambda tags: "placental_antigen" in tags
    )].to_csv(out / "placental-nomination-provenance.csv", index=False)
    source_overlap_counts(df).to_csv(out / "cta-source-overlap-counts.csv", index=False)
    if publication_data_available():
        from oncoref.cta_sources import (intake_counts, intake_membership, publication_sources,
                                        placental_source_coverage, gene_publication_evidence)

        placental_source_coverage().to_csv(out / "cta-placental-evidence-coverage.csv", index=False)
        gene_publication_evidence().to_csv(out / "cta-gene-publication-evidence.csv", index=False)
        intake_counts().to_csv(out / "cta-publication-intake-counts.csv", index=False)
        intake_membership().to_csv(out / "cta-publication-intake-membership.csv", index=False)
        publication_sources().to_csv(out / "cta-publication-sources.csv", index=False)
        pd.DataFrame([{"source": source.replace("\n", " "), "Ensembl_Gene_ID": gid}
                      for source, ids in placental_source_sets(df).items() for gid in sorted(ids)]
                     ).to_csv(out / "cta-placental-source-membership.csv", index=False)
    return {"n_genes": int(len(df)), "paths": paths, "stages": stages}
