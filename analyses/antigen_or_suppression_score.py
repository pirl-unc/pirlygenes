#!/usr/bin/env python
"""Antigen-OR-suppression score: does a cancer type have an immune *handle* —
antigen from at least one source, OR an active suppressive mechanism?

Two composite axes, each an OR/max over cross-cohort percentiles so a cancer
scores high if ANY one component is high (the distinct-biology logic):

  ANTIGEN (≥1 source)  = max percentile over four antigen sources —
      CTA burden · TMB · viral etiology · indel/frameshift load.
  SUPPRESSION          = percentile of the curated TGF-β / Wnt immune-EXCLUSION
      signature (therapy-response-signatures.csv aPD1_exclusion_{TGFb_response,
      Wnt}). NOTE this is partly infiltrate-confounded (bulk TGFB1 tracks CD45+
      leukocyte content), so the suppression axis indexes the *curated exclusion
      program*, not a pure tumor-intrinsic brake.

  OR score = max(antigen, suppression)  — "has a handle" (target or brake).

The 2-D map sorts every cancer type into a therapeutic stratum:
  antigen-only          -> ICI-monotherapy candidates
  antigen + suppression -> ICI + suppression-release combination
  suppression-only      -> antigen induction (vaccine/CTA) + brake release
  NEITHER (silent)      -> no immune handle (hardest)

Points are coloured by anti-PD-1/ICI ORR (validation: antigen should push ORR
up, suppression down) and sized by √(cohort n).

    python analyses/antigen_or_suppression_score.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _apd1_factors import (apd1_map, cohort_gene_matrix,  # noqa: E402
                           cta_burden, curated_exclusion_genes, indel_map,
                           tmb_map, viral_score, with_parent)
from _panels import fold  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402
from suppressor_genes_vs_apd1 import _cohort_sizes  # noqa: E402
from pirlygenes.gene_sets_cancer import cancer_type_registry  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
THR = 0.5                      # quadrant cut = 50th cross-cohort percentile
ANTIGEN_SOURCES = ["CTA", "TMB", "viral", "indel"]


def _pct(s: pd.Series) -> pd.Series:
    """Cross-cohort percentile rank (0–1); NaNs stay NaN."""
    return s.rank(pct=True)


def _antigen_table(M: pd.DataFrame, codes, reg) -> pd.DataFrame:
    """Per-cohort percentile of each antigen source (missing source -> 0th pctile
    so it can't make a cohort look antigen-rich)."""
    tmb, ind = tmb_map(), indel_map()
    return pd.DataFrame({
        "CTA": _pct(cta_burden(M).reindex(codes)),
        "TMB": _pct(pd.Series({c: np.log10(with_parent(tmb, c, np.nan))
                               for c in codes})),
        "viral": _pct(pd.Series({c: viral_score(c, reg) for c in codes})),
        "indel": _pct(pd.Series({c: with_parent(ind, c, 0.0) for c in codes})),
    }).reindex(codes)


def _suppression(M: pd.DataFrame, codes) -> pd.Series:
    """Cross-cohort percentile of the curated TGF-β + Wnt exclusion signature
    (mean log-TPM over the present, proteoform-folded member genes)."""
    ex = curated_exclusion_genes()
    genes = ex.get("TGFb_response", []) + ex.get("Wnt", [])
    present = [g for g in fold(genes) if g in M.columns]
    return _pct(M.loc[codes, present].mean(axis=1))


def _quadrant(antigen: float, suppr: float) -> str:
    if antigen >= THR:
        return "antigen+suppression" if suppr >= THR else "antigen-only"
    return "suppression-only" if suppr >= THR else "NEITHER (silent)"


def _build() -> pd.DataFrame:
    orr_map = apd1_map()
    M = cohort_gene_matrix(list(orr_map))
    codes = [c for c in M.index if c in orr_map]
    M = M.loc[codes]
    reg = cancer_type_registry().set_index("code")
    ag = _antigen_table(M, codes, reg)
    df = ag.copy()
    df["antigen_any"] = ag[ANTIGEN_SOURCES].max(axis=1)
    df["suppression"] = _suppression(M, codes)
    df["OR_score"] = df[["antigen_any", "suppression"]].max(axis=1)
    df["ORR"] = pd.Series({c: orr_map[c] for c in codes})
    df["n_samples"] = _cohort_sizes(codes).reindex(codes)
    df["quadrant"] = [_quadrant(a, s)
                      for a, s in zip(df["antigen_any"], df["suppression"])]
    return df


def _plot(df: pd.DataFrame, figdir: Path) -> None:
    ok = df["antigen_any"].notna() & df["suppression"].notna()
    d = df[ok]
    sizes = 28.0 + 5.0 * np.sqrt(d["n_samples"].fillna(1.0).to_numpy())
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    # shade the immune-silent (no-handle) quadrant
    ax.add_patch(Rectangle((0, 0), THR, THR, color="#d62828", alpha=0.06, zorder=0))
    sc = ax.scatter(d["antigen_any"], d["suppression"], c=d["ORR"], cmap="viridis",
                    s=sizes, edgecolor="black", linewidth=0.4, zorder=3)
    for c, row in d.iterrows():
        ax.annotate(c, (row["antigen_any"], row["suppression"]), fontsize=5,
                    alpha=0.7, xytext=(2, 2), textcoords="offset points")
    ax.axvline(THR, color="0.4", lw=0.9, ls="--")
    ax.axhline(THR, color="0.4", lw=0.9, ls="--")
    corners = {(0.99, 0.98): ("antigen + suppression\n(ICI + brake-release)", "right", "top"),
               (0.01, 0.98): ("suppression-only\n(induce antigen + release)", "left", "top"),
               (0.99, 0.02): ("antigen-only\n(ICI monotherapy)", "right", "bottom"),
               (0.01, 0.02): ("NEITHER — immune-silent", "left", "bottom")}
    for (x, y), (txt, ha, va) in corners.items():
        ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=8,
                color="#7a1f1f" if "NEITHER" in txt else "#1d3557",
                fontweight="bold", alpha=0.85)
    cov = 100 * (d["OR_score"] >= THR).mean()
    ax.set_xlabel("ANTIGEN from ≥1 source  (max percentile: CTA · TMB · viral · indel)")
    ax.set_ylabel("SUPPRESSION  (curated TGF-β / Wnt exclusion percentile)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Does a cancer type have an immune handle? "
                 "antigen (≥1 source) OR suppression\n"
                 f"OR-score coverage = {cov:.0f}% of {len(d)} cancer types · "
                 "point size ∝ √n · colour = ORR", fontsize=11)
    fig.colorbar(sc, ax=ax, label="anti-PD-1 / ICI ORR (%)", shrink=0.7)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(figdir / "antigen_or_suppression_score.png", dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT)

    df = _build()
    _plot(df, figdir)
    cols = ANTIGEN_SOURCES + ["antigen_any", "suppression", "OR_score",
                              "quadrant", "ORR", "n_samples"]
    df[cols].sort_values("OR_score", ascending=False).to_csv(
        figdir / "antigen_or_suppression_score.csv", index_label="cancer_code")

    ok = df["antigen_any"].notna() & df["suppression"].notna()
    d = df[ok]
    cov = 100 * (d["OR_score"] >= THR).mean()
    print(f"antigen-OR-suppression score over {len(d)} cancer types")
    print(d["quadrant"].value_counts().to_string())
    print(f"OR-score coverage (>= {THR}): {cov:.0f}% have ≥1 antigen source OR "
          "suppression")
    print("immune-silent:", sorted(d.index[d["quadrant"] == "NEITHER (silent)"]))
    for k in ("antigen_any", "suppression", "OR_score"):
        print(f"  {k:12s} vs ORR Spearman = "
              f"{spearmanr(d[k], d['ORR']).statistic:+.2f}")
    print(f"wrote antigen_or_suppression_score.png + .csv -> {figdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
