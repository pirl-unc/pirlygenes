#!/usr/bin/env python
"""Multi-factor model of anti-PD-1 response from *causal* (non-inverted) factors.

We deliberately exclude the circular markers — an IFN / cytotoxic-infiltrate
signature (CXCL9, CD8A, GZMB, PD-L1, IDO1) "predicts" response only because it
*measures* the CD8 T cells that already responded. Instead we assemble the
upstream causal levers and ask how much of cross-cancer-type ORR they explain:

  ANTIGEN AVAILABILITY (is there anything for a T cell to see?)  [drives UP]
    * TMB           - SNV/indel mutational load (cancer-tmb.csv)
    * CTA burden    - mean cancer-testis-antigen expression (tsarina CTA panel)
    * viral         - virally-driven foreign antigen (registry viral_etiology:
                      defining=1.0, subset=0.5, none=0)
  TOLERANCE / EXCLUSION (will a CD8 T cell get in and engage?)   [drives DOWN]
    * exclusion     - curated TGF-beta-response + Wnt composite
                      (therapy-response-signatures.csv aPD1_exclusion_*)

Unmodeled causal factors we *can't* measure cleanly (flagged on the residuals):
  * frameshift/indel neoantigen load distinct from SNV TMB (Turajlic 2017,
    PMID 28694034) - highest pan-cancer in ccRCC.
  * endogenous retrovirus (ERV/HERV) expression (Smith 2018, PMID 29618660) -
    drives ICB response in ccRCC despite low TMB.
These are exactly why KIRC/KIRP respond (~25%) on near-zero TMB/CTA/viral, so
they should appear as large POSITIVE residuals (under-predicted by the model).

    python analyses/apd1_causal_factors.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pirlygenes.gene_sets_cancer import CTA_gene_names  # noqa: E402
from pirlygenes.gene_sets_cancer import cancer_type_registry  # noqa: E402

from _apd1_factors import (apd1_map, cohort_gene_matrix,  # noqa: E402
                           curated_exclusion_genes, indel_map, tmb_map,
                           viral_score, with_parent)

OUT = Path(__file__).resolve().parent / "outputs"


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def main() -> int:
    apd1 = apd1_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    orr = pd.Series({c: apd1[c] for c in mat.index})

    reg = cancer_type_registry().set_index("code")
    tmb = tmb_map()
    indel = indel_map()
    sig = curated_exclusion_genes()
    excl_genes = [g for g in sig["TGFb_response"] + sig["Wnt"]
                  if g in mat.columns]
    cta_genes = [g for g in CTA_gene_names() if g in mat.columns]

    F = pd.DataFrame(index=mat.index)
    F["ORR"] = orr
    F["TMB"] = [with_parent(tmb, c, np.nan) for c in mat.index]
    F["logTMB"] = np.log10(F["TMB"])
    # CTA burden = how many CTA antigens are actually ON in the cohort
    # (mean TPM >= 5), not a mean over 263 mostly-silent genes.
    F["CTA"] = (mat[cta_genes] >= np.log10(6)).sum(axis=1)
    F["viral"] = [viral_score(c, reg) for c in mat.index]
    F["indel"] = [with_parent(indel, c, 0.0) for c in mat.index]
    F["exclusion"] = mat[excl_genes].apply(_z).mean(axis=1)
    F = F.dropna(subset=["logTMB", "CTA", "exclusion"])
    # The two conceptual axes the user wants kept distinct:
    #   ANTIGEN AVAILABILITY = TMB + CTA + viral + indel (anything to see?)
    #   IMMUNE EXCLUSION     = TGF-beta-response + Wnt  (will a T cell get in?)
    F["antigen"] = F[["logTMB", "CTA", "viral", "indel"]].apply(_z).mean(axis=1)
    print(f"cohorts modeled: {len(F)}\n")

    # ---- univariate Spearman of each causal factor -----------------------
    print("=== Univariate Spearman vs aPD1 ORR (causal factors) ===")
    for col in ["logTMB", "CTA", "viral", "indel", "exclusion"]:
        rho, p = spearmanr(F[col], F["ORR"])
        print(f"  {col:10s} rho={rho:+.2f} (p={p:.3f})")

    # ---- two-axis model: ANTIGEN vs EXCLUSION ----------------------------
    Xa = np.column_stack([np.ones(len(F)), _z(F["antigen"]), _z(F["exclusion"])])
    ya = _z(F["ORR"]).to_numpy()
    ba, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    yha = Xa @ ba
    r2a = 1 - ((ya - yha) ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
    print("\n=== Two-axis model: ORR ~ antigen + exclusion ===")
    print(f"  R2 = {r2a:.2f}  beta[antigen]={ba[1]:+.2f}  "
          f"beta[exclusion]={ba[2]:+.2f}")

    # ---- standardized multiple regression (factor breakdown) -------------
    feats = ["logTMB", "CTA", "viral", "indel", "exclusion"]
    X = np.column_stack([_z(F[c]) for c in feats])
    X = np.column_stack([np.ones(len(F)), X])
    y = _z(F["ORR"]).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n=== Standardized OLS: ORR ~ {' + '.join(feats)} ===")
    print(f"  R2 = {r2:.2f} (n={len(F)})")
    for name, b in zip(feats, beta[1:]):
        print(f"  beta[{name:10s}] = {b:+.2f}")

    # residuals on the ORR scale (refit unstandardized for interpretable %)
    Xr = np.column_stack([np.ones(len(F))] + [F[c].pipe(_z).to_numpy()
                                              for c in feats])
    br, *_ = np.linalg.lstsq(Xr, F["ORR"].to_numpy(), rcond=None)
    F["pred"] = Xr @ br
    F["resid"] = F["ORR"] - F["pred"]
    print("\n=== Largest POSITIVE residuals (respond MORE than causal factors "
          "predict -> unmodeled antigen: indel/ERV) ===")
    for c, row in F.sort_values("resid", ascending=False).head(6).iterrows():
        print(f"  {c:12s} actual={row['ORR']:4.0f}%  pred={row['pred']:5.1f}%  "
              f"resid={row['resid']:+5.1f}  (TMB={row['TMB']:.1f}, "
              f"viral={row['viral']:.1f})")
    print("=== Largest NEGATIVE residuals (respond LESS than predicted -> "
          "tolerance/exclusion beyond the composite) ===")
    for c, row in F.sort_values("resid").head(6).iterrows():
        print(f"  {c:12s} actual={row['ORR']:4.0f}%  pred={row['pred']:5.1f}%  "
              f"resid={row['resid']:+5.1f}")

    _plot(F, feats, beta[1:], r2)
    F.to_csv(OUT / "_apd1_causal_factors.csv")
    print(f"\nwrote {OUT/'_apd1_causal_factors.csv'} and "
          f"{OUT/'apd1_causal_factors.png'}")
    return 0


def _plot(F, feats, betas, r2):
    fig = plt.figure(figsize=(16, 5.2))
    gyn = ["OV", "BRCA_Basal", "UCEC_CNL", "UCEC_CNH"]
    rcc = ["KIRC", "KIRP", "KICH"]

    # A. standardized coefficients
    axA = fig.add_subplot(1, 3, 1)
    colors = ["#2c7fb8" if b >= 0 else "#d95f0e" for b in betas]
    axA.barh(feats, betas, color=colors)
    axA.axvline(0, color="k", lw=0.8)
    axA.set_title(f"Standardized drivers of aPD1 ORR\n(OLS R2={r2:.2f}, "
                  f"n={len(F)})", fontsize=10)
    axA.set_xlabel("standardized beta (signed)")
    for i, b in enumerate(betas):
        axA.annotate(f"{b:+.2f}", (b, i), fontsize=8,
                     va="center", ha="left" if b >= 0 else "right")

    # B. predicted vs actual
    axB = fig.add_subplot(1, 3, 2)
    axB.scatter(F["pred"], F["ORR"], s=30, alpha=0.8)
    lim = [min(F["pred"].min(), F["ORR"].min()) - 3,
           max(F["pred"].max(), F["ORR"].max()) + 3]
    axB.plot(lim, lim, "k--", lw=0.8)
    for c, row in F.iterrows():
        hot = c in rcc or abs(row["resid"]) > 12
        axB.annotate(c, (row["pred"], row["ORR"]), fontsize=6,
                     color="red" if c in rcc else ("black" if hot else "gray"),
                     alpha=0.9 if hot else 0.4)
    rho, _ = spearmanr(F["pred"], F["ORR"])
    axB.set_title(f"Predicted vs actual ORR (Spearman={rho:+.2f})\n"
                  "red=RCC: respond despite low TMB (indel/ERV)", fontsize=10)
    axB.set_xlabel("predicted ORR (%)")
    axB.set_ylabel("actual ORR (%)")
    axB.grid(alpha=0.3)

    # C. 2-axis antigen x exclusion map
    axC = fig.add_subplot(1, 3, 3)
    antigen = _z(F["antigen"])
    sc = axC.scatter(antigen, _z(F["exclusion"]), c=F["ORR"], cmap="viridis",
                     s=70, edgecolor="k", linewidth=0.3)
    for c in F.index:
        col = "red" if c in rcc else ("darkgreen" if c in gyn else "black")
        w = "bold" if (c in rcc or c in gyn) else "normal"
        axC.annotate(c, (antigen[c], _z(F["exclusion"])[c]), fontsize=6,
                     color=col, fontweight=w, alpha=0.85)
    axC.axhline(0, color="gray", lw=0.6, ls=":")
    axC.axvline(0, color="gray", lw=0.6, ls=":")
    axC.set_xlabel("antigen availability  (z: TMB + CTA + viral + indel)  ->")
    axC.set_ylabel("immune exclusion  (TGF-beta-response + Wnt)  ->")
    axC.set_title("Antigen x exclusion map\ngreen=gyn (cold), red=RCC "
                  "(low-antigen responders)", fontsize=10)
    fig.colorbar(sc, ax=axC, label="aPD1 ORR (%)")
    fig.suptitle("Causal factors of anti-PD-1 response across cancer types "
                 "(IFN/infiltrate markers deliberately excluded)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "apd1_causal_factors.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
