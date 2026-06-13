"""Immune-checkpoint landscape composites — three figures from the curated
antigen / suppression factors:

  1. ici_causal_heatmap.png — cancers (rows) sorted by ICI response rate; columns
     are the causal factors: ICI ORR · TMB · CTA burden · viral status |
     TGF-beta · Wnt/beta-catenin (the two suppression pathways). Each cell is the
     factor's z-score across cancers; a vertical rule separates the antigen/
     response DRIVERS from the SUPPRESSORS. The per-column Spearman vs ORR is
     annotated.

  2. ici_antigen_load.png — a composite ANTIGEN LOAD index = mean percentile-rank
     of (CTA-p95 burden, median TMB, viral-antigen status). CTA = shared antigens,
     TMB = mutational neoantigens, viral = viral antigens. Ranked bar, lineage
     coloured.

  3. ici_cta_metrics_heatmap.png — ALL the CTA metrics together: the fraction of
     a cohort's patients with >=1 CTA above each absolute-TPM and within-sample
     percentile bar, rows sorted by p95 coverage.

Run:  python analyses/ici_landscape.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pirlygenes import gene_sets_cancer as gsc  # noqa: E402
from pirlygenes.gene_sets_cancer import cancer_type_registry  # noqa: E402
from _apd1_factors import (apd1_map, tmb_map, viral_score, cta_burden,  # noqa: E402
                           cohort_gene_matrix, curated_signatures)
from _panels import GENE_PANELS  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
FIGDIR = OUT

_TGFB = "aPD1_exclusion_TGFb_response"
_WNT = "aPD1_exclusion_Wnt"
# A secreted pair of immune-EXCLUSION genes (the suppression term, rank-inverted,
# in the net ICI-favourability composite):
#   TGFB1 — fibroblast/stromal TGF-beta traps T cells in peritumoral stroma (the
#           "excluded" phenotype; Mariathasan 2018, Nature, PMID 29443960).
#   WNT11 — secreted Wnt drives beta-catenin CD8 exclusion (down CXCL10/CCL4) and
#           anti-PD-1 resistance (Lu et al. 2025, Nat Commun, 10.1038/s41467-025-56714-z);
#           upregulated in ovarian-serous / gastric / papillary-RCC.
#   WNT5A — secreted Wnt drives tolerogenic DCs (IDO -> Treg) / immune evasion
#           (Holtzhausen 2015, Cancer Immunol Res, PMID 26041736).
#   IL10  — general secreted immunosuppressive cytokine (Treg/M2); included to
#           check whether it carries any per-cohort signal vs the exclusion genes.
# Shown as individual columns in the causal heatmap (each gene's Spearman vs ORR
# visible) and averaged into the favourability suppression term. Sourced from the
# central panel registry (proteoform-folded) so it can't drift from the figures.
_SECRETED_INHIBITORY = GENE_PANELS["secreted_inhibitory"]


def _z(s):
    return (s - s.mean()) / s.std(ddof=0)


def _lin(code: str) -> str:
    """Lineage group, with a base-code fallback for the synthetic pooled CRC
    tiers (CRC_MSI/CRC_MSS aren't registry codes -> CRC -> Epithelial)."""
    g = gsc.cancer_lineage_group(code)
    if g:
        return g
    return gsc.cancer_lineage_group(code.split("_")[0]) or "other"


def _lineage_colors(codes):
    groups = sorted({_lin(c) for c in codes})
    cmap = plt.get_cmap("tab10")
    return {g: cmap(i % 10) for i, g in enumerate(groups)}


def _factors():
    """Per-cohort factor table indexed by cancer code (only codes with an ICI
    ORR), with the raw factor values used by all three figures."""
    apd1 = apd1_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    reg = cancer_type_registry().set_index("code")
    tmb = tmb_map()
    cta = cta_burden(mat)
    sig = curated_signatures()

    def sigscore(key):
        genes = [g for g in sig.get(key, []) if g in mat.columns]
        if not genes:
            return pd.Series(np.nan, index=mat.index)
        return mat[genes].apply(_z).mean(axis=1)

    df = pd.DataFrame({
        "ICI ORR": pd.Series({c: apd1[c] for c in mat.index}),
        "TMB": np.log10(pd.Series({c: tmb.get(c, np.nan) for c in mat.index})),
        "CTA burden": pd.Series({c: cta.get(c, np.nan) for c in mat.index}),
        "viral": pd.Series({c: viral_score(c, reg) for c in mat.index}),
        "TGFβ": sigscore(_TGFB),
        "Wnt/β-catenin": sigscore(_WNT),
    })
    inhib = [g for g in _SECRETED_INHIBITORY if g in mat.columns]
    for g in inhib:                     # individual z-cols (shown in the heatmap)
        df[g] = _z(mat[g])
    df["secreted inhibitory"] = df[inhib].mean(axis=1) if inhib else np.nan
    return df


# --------------------------------------------------------------------------
def _causal_heatmap(df):
    drivers = ["ICI ORR", "TMB", "CTA burden", "viral"]
    genes = [g for g in _SECRETED_INHIBITORY if g in df.columns]
    suppr = ["TGFβ", "Wnt/β-catenin"] + genes      # pathway signatures + secreted genes
    cols = drivers + suppr
    Z = df[cols].apply(_z)
    order = df["ICI ORR"].sort_values(ascending=False).index
    Zo = Z.loc[order]

    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 4.5, max(8, 0.21 * len(order))))
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    im = ax.imshow(Zo.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    # Spearman vs ORR under each column header
    rho = {k: spearmanr(df[k], df["ICI ORR"], nan_policy="omit").statistic
           for k in cols}
    ax.set_xticklabels([f"{k}\n(ρ={rho[k]:+.2f})" for k in cols],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.axvline(len(drivers) - 0.5, color="black", lw=2)   # drivers | suppressors
    ax.text(len(drivers) / 2 - 0.5, -1.2, "antigen / response drivers",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(len(drivers) + len(suppr) / 2 - 0.5, -1.2, "suppression",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("ICI response causal factors (z-scored), cancers sorted by ORR",
                 fontsize=10, pad=28)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="z-score")
    fig.savefig(FIGDIR / "ici_causal_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(order)


def _antigen_load(df):
    sub = df[["CTA burden", "TMB", "viral"]]
    ranks = sub.rank(pct=True)                 # NaN stays NaN; per-column pctile
    load = ranks.mean(axis=1, skipna=True) * 100.0
    load = load.dropna().sort_values()
    colors = _lineage_colors(load.index)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.30 * len(load))))
    ax.barh(range(len(load)),
            load.values, edgecolor="white",
            color=[colors[_lin(c)] for c in load.index])
    ax.set_yticks(range(len(load)))
    ax.set_yticklabels(load.index, fontsize=7)
    ax.set_xlabel("antigen load  (mean percentile-rank of CTA burden, TMB, viral)")
    ax.set_xlim(0, 100)
    ax.set_title("Composite tumor antigen load by cancer type\n"
                 "shared antigens (CTA) + neoantigens (TMB) + viral antigens",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[g], label=g)
               for g in sorted(colors)]
    ax.legend(handles=handles, title="lineage", fontsize=6.5, title_fontsize=7,
              loc="lower right")
    fig.savefig(FIGDIR / "ici_antigen_load.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(load)


def _favorability(df):
    """Net ICI-favourability composite = mean of the DIRECTIONAL percentile-ranks
    of the antigen drivers (CTA burden, TMB, viral — high is favourable) and the
    secreted suppression term (TGFB1/WNT5A/WNT11/IL10 — high is UNfavourable, so its rank is
    inverted). A single 0-100 score combining 'how much antigen' with 'how much
    secreted brake'."""
    pos = df[["CTA burden", "TMB", "viral"]].rank(pct=True)
    neg = 1.0 - df[["secreted inhibitory"]].rank(pct=True)   # high suppression -> low
    score = (pd.concat([pos, neg], axis=1).mean(axis=1, skipna=True) * 100.0)
    score = score.dropna().sort_values()
    colors = _lineage_colors(score.index)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.30 * len(score))))
    ax.barh(range(len(score)), score.values, edgecolor="white",
            color=[colors[_lin(c)] for c in score.index])
    ax.axvline(50, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(range(len(score)))
    ax.set_yticklabels(score.index, fontsize=7)
    ax.set_xlabel("ICI-favourability  (mean directional rank: +CTA +TMB +viral "
                  "−secreted exclusion panel)")
    ax.set_xlim(0, 100)
    ax.set_title("Net ICI-favourability composite by cancer type\n"
                 "antigen load (CTA + TMB + viral) minus secreted exclusion "
                 "(TGFB1 + WNT5A + WNT11 + IL10)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[g], label=g)
               for g in sorted(colors)]
    ax.legend(handles=handles, title="lineage", fontsize=6.5, title_fontsize=7,
              loc="lower right")
    fig.savefig(FIGDIR / "ici_favorability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(score)


def _cta_metrics_heatmap():
    path = OUT / "_cta_union_counts.csv"
    if not path.exists():
        print("      (no _cta_union_counts.csv; skip CTA-metrics heatmap)", flush=True)
        return 0
    u = pd.read_csv(path)
    metrics = {">25 TPM": "n_any_gt25", ">50 TPM": "n_any_gt50",
               "≥p90": "n_any_p90", "≥p95": "n_any_p95"}
    frac = pd.DataFrame(
        {lab: 100.0 * u[col] / u["n_samples"] for lab, col in metrics.items()})
    frac.index = u["cancer_code"]
    frac = frac.loc[frac["≥p95"].sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(7, max(8, 0.20 * len(frac))))
    im = ax.imshow(frac.values, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(frac.columns)))
    ax.set_xticklabels(frac.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(frac)))
    ax.set_yticklabels(frac.index, fontsize=6)
    ax.axvline(1.5, color="white", lw=1.5)     # absolute-TPM | within-sample pctile
    ax.set_title("CTA burden — all metrics (% patients with ≥1 CTA on)\n"
                 "absolute TPM | within-sample percentile, sorted by ≥p95",
                 fontsize=10, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="% patients")
    fig.savefig(FIGDIR / "ici_cta_metrics_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(frac)


def main() -> int:
    global FIGDIR
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, FIGDIR = resolve_dirs(args, OUT)
    df = _factors()
    n1 = _causal_heatmap(df)
    n2 = _antigen_load(df)
    n4 = _favorability(df)
    n3 = _cta_metrics_heatmap()
    print(f"wrote ici_causal_heatmap.png ({n1}), ici_antigen_load.png ({n2}), "
          f"ici_favorability.png ({n4}), ici_cta_metrics_heatmap.png ({n3}) "
          f"-> {FIGDIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
