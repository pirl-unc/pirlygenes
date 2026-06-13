#!/usr/bin/env python
"""Holistic views of the causal factors of anti-PD-1 response across cancer types.

Builds two figures from the curated pathway signatures
(therapy-response-signatures.csv) + the antigen factors (TMB / indel / viral /
CTA), all on the aPD1-response cohorts:

1. apd1_landscape_heatmap.png - cancer types (rows, sorted by ORR) x pathway
   signatures (cols, grouped antigen | exclusion | circular). Each cell is the
   cohort's z-scored signature level; the per-column Spearman vs ORR is printed
   under each column. The eye should see: responders (top) high on antigen +
   circular, low on exclusion; non-responders (bottom) the reverse.

2. apd1_balance_sheet.png - for archetype cancers, a diverging "balance sheet"
   of standardized causal factors: antigen factors (green, push response UP) vs
   exclusion factors (red, push it DOWN). Reads why each cancer responds or not.

Circular signatures (IFN/checkpoints/Treg) are shown only as a labelled CONTRAST
band - they track response because they ARE the response, not because they cause it.

    python analyses/apd1_landscape.py
"""

from __future__ import annotations

import os

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from pirlygenes.gene_sets_cancer import cancer_type_registry  # noqa: E402

from _apd1_factors import (SIGNATURE_META, apd1_map,  # noqa: E402
                           cohort_gene_matrix, cta_burden, curated_signatures,
                           indel_map, tmb_map, viral_score, with_parent)

OUT = Path(os.environ.get("APD1_RUN_DIR",
          str(Path(__file__).resolve().parent / "outputs" / "apd1_causal_factors")))
OUT.mkdir(parents=True, exist_ok=True)
# column order within each axis (label -> therapy_class or special token)
_AXIS_ORDER = ["antigen", "exclusion", "circular"]


def _z(s):
    return (s - s.mean()) / s.std(ddof=0)


def _build():
    apd1 = apd1_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    orr = pd.Series({c: apd1[c] for c in mat.index})

    reg = cancer_type_registry().set_index("code")
    tmb, ind = tmb_map(), indel_map()

    cols = {}  # display label -> (axis, raw per-cohort series)
    # antigen factors that aren't expression signatures
    cols["TMB"] = ("antigen", np.log10(pd.Series(
        {c: with_parent(tmb, c, np.nan) for c in mat.index})))
    cols["indel"] = ("antigen", pd.Series(
        {c: with_parent(ind, c, 0.0) for c in mat.index}))
    cols["viral"] = ("antigen", pd.Series(
        {c: viral_score(c, reg) for c in mat.index}))
    cols["CTA"] = ("antigen", cta_burden(mat))
    # curated expression signatures
    for cls, genes in curated_signatures().items():
        present = [g for g in genes if g in mat.columns]
        if len(present) < 1:        # allow single-gene signatures (TGF-β = TGFB2)
            continue
        axis = SIGNATURE_META[cls][0]
        label = cls.replace("aPD1_", "").replace("exclusion_", "").replace(
            "circular_", "").replace("antigen_", "")
        cols[label] = (axis, mat[present].apply(_z).mean(axis=1))

    # order columns by axis
    ordered = sorted(cols, key=lambda k: (_AXIS_ORDER.index(cols[k][0]), k))
    Z = pd.DataFrame({k: _z(cols[k][1]) for k in ordered})
    axes_of = {k: cols[k][0] for k in ordered}
    rho = {k: spearmanr(cols[k][1], orr, nan_policy="omit").statistic
           for k in ordered}
    return Z, orr, axes_of, rho


def _heatmap(Z, orr, axes_of, rho):
    order = orr.sort_values(ascending=False).index
    Zo = Z.loc[order]
    fig, ax = plt.subplots(figsize=(13, 11))
    norm = TwoSlopeNorm(vmin=-2.2, vcenter=0, vmax=2.2)
    ax.imshow(Zo.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(Zo.columns)))
    ax.set_xticklabels([f"{c}\n(rho={rho[c]:+.2f})" for c in Zo.columns],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}  ({orr[c]:.0f}%)" for c in order], fontsize=7)
    # axis-group separators + headers
    prev, start = None, 0
    for i, c in enumerate(Zo.columns):
        a = axes_of[c]
        if prev is not None and a != prev:
            ax.axvline(i - 0.5, color="k", lw=1.5)
            ax.text((start + i - 1) / 2, -1.6, prev.upper(), ha="center",
                    fontsize=9, fontweight="bold")
            start = i
        prev = a
    ax.text((start + len(Zo.columns) - 1) / 2, -1.6, prev.upper(),
            ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Causal-factor landscape of anti-PD-1 response\n"
                 "rows = cancer types sorted by ORR; cells = z-scored factor "
                 "level (red high / blue low)", fontsize=11, pad=44)
    fig.colorbar(ax.images[0], ax=ax, shrink=0.4, label="z-score")
    fig.tight_layout()
    fig.savefig(OUT / "apd1_landscape_heatmap.png", dpi=300)
    plt.close(fig)


def _balance_sheet(Z, orr, axes_of):
    archetypes = ["SKCM", "CRC_MSI", "KIRC", "HL", "OV", "UCEC_CNH",
                  "LIHC", "PRAD"]
    archetypes = [c for c in archetypes if c in Z.index]
    # only causal antigen/exclusion factors (drop circular)
    feats = [c for c in Z.columns if axes_of[c] in ("antigen", "exclusion")]
    # signed contribution toward response: antigen as +z, exclusion as -z, so a
    # rightward bar is always FAVOURABLE regardless of factor type. Position
    # carries the meaning; colour (colourblind-safe blue/orange, NOT red/green)
    # just tags the factor type, and a dashed marker shows the NET balance.
    sign = {f: (1 if axes_of[f] == "antigen" else -1) for f in feats}
    ANTIGEN, EXCLUSION = "#2166ac", "#b35806"   # PuOr blue / orange (CB-safe)
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.4), sharex=True)
    for ax, code in zip(axs.flat, archetypes):
        contrib = (Z.loc[code, feats] * pd.Series(sign)).sort_values()
        colors = [ANTIGEN if axes_of[f] == "antigen" else EXCLUSION
                  for f in contrib.index]
        ax.barh(range(len(contrib)), contrib.values, color=colors)
        ax.set_yticks(range(len(contrib)))
        ax.set_yticklabels(contrib.index, fontsize=6)
        ax.axvline(0, color="k", lw=0.7)
        net = float(contrib.sum())
        ax.axvline(net, color="#222", lw=1.8, ls="--")   # NET balance marker
        ax.set_title(f"{code}   ORR {orr[code]:.0f}%   net {net:+.1f}",
                     fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axs.flat[len(archetypes):]:
        ax.axis("off")
    fig.suptitle("Causal balance sheet per cancer type — each factor's signed "
                 "contribution toward anti-PD-1 response\n"
                 "blue = antigen availability (TMB/indel/viral/CTA/MHC-I)    "
                 "orange = immune exclusion (TGF-β/Wnt/angio/adenosine)    "
                 "dashed = net", fontsize=12)
    fig.supxlabel("←  less favourable        contribution toward aPD-1 response "
                  "(z)        more favourable  →", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "apd1_balance_sheet.png", dpi=300)
    plt.close(fig)


def main() -> int:
    Z, orr, axes_of, rho = _build()
    _heatmap(Z, orr, axes_of, rho)
    _balance_sheet(Z, orr, axes_of)
    print(f"factors: {list(Z.columns)}")
    print(f"cohorts: {len(Z)}")
    print("per-factor Spearman vs ORR:")
    for k in sorted(rho, key=lambda x: rho[x]):
        print(f"  {k:16s} ({axes_of[k]:9s}) rho={rho[k]:+.2f}")
    print(f"\nwrote {OUT/'apd1_landscape_heatmap.png'} and "
          f"{OUT/'apd1_balance_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
