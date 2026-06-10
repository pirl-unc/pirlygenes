#!/usr/bin/env python
"""Cross-cohort screen: which genes' bulk expression tracks anti-PD-1 response?

Motivation
----------
A cytotoxic/IFN signature (CXCL9, CD8A, GZMB, HLA-E, PD-L1, IDO1) "predicts"
aPD1 response only *circularly* — those genes are induced by the CD8 T cells
that already infiltrated and fired IFN-gamma, so they measure the outcome, not
a cause. This script instead screens for genes whose constitutive expression is
*negatively* associated with response across cancer types — candidate
**causal T-cell-exclusion / tolerance** genes that act upstream of any IFN tone.

Two analyses:
  1. BROAD SCREEN  — Spearman(cohort-mean log10 TPM, aPD1 ORR) for every gene
     detected in >= MIN_COHORTS cohorts. Ranked most-negative first.
  2. GYN-vs-SKCM CONTRAST — genes high in the cold gyn cancers
     (OV / BRCA_Basal / UCEC_*) and low in inflamed melanoma (SKCM).
The intersection (negative-screen AND gyn-high/SKCM-low) is the candidate
exclusion set. A curated literature panel (TGF-beta/CAF barrier, PGE2/COX-2,
Wnt, VEGF) is scored against the unbiased ranking for sanity.

Caveats (reported, not hidden):
  * ~31 cohorts -> per-gene rho is noisy; trust recurring *mechanism*, not any
    single gene. No multiple-testing survives at this n.
  * aPD1 ORR is per-cancer-type, confounded by TMB and lineage.
  * Stromal/CAF genes co-vary with tumor purity (low purity -> high stroma);
    that is mechanistically meaningful (barrier) but also a confound.
  * Microarray-proxy cohorts are dropped (cross-platform-incomparable TPM).

    python analyses/exclusion_vs_apd1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _apd1_factors import (apd1_map, cohort_gene_matrix,
                           curated_exclusion_genes)

OUT = Path(__file__).resolve().parent / "outputs"
MIN_COHORTS = 18  # gene must be present in >= this many aPD1 cohorts

# Curated, literature-grounded *causal exclusion* panel (NOT IFN-inducible).
CURATED = {
    "TGFB1": "TGF-beta/CAF barrier (Mariathasan 2018, PMID 29443960)",
    "TGFB2": "TGF-beta/CAF barrier",
    "TGFB3": "TGF-beta/CAF barrier",
    "CXCL12": "FAP+ CAF coat (Feig 2013, PMID 24277834)",
    "FAP": "immunosuppressive CAF",
    "LRRC15": "TGF-beta-driven myCAF",
    "COL11A1": "desmoplastic barrier",
    "POSTN": "desmoplastic barrier",
    "ACTA2": "myCAF / desmoplasia",
    "PMEPA1": "TGF-beta response",
    "PTGS2": "PGE2/COX-2 myeloid exclusion (Zelenay 2015, PMID 26343581)",
    "PTGES": "PGE2 synthase",
    "VEGFA": "angiogenic vascular barrier",
    "DKK1": "Wnt-axis intrinsic exclusion (Spranger 2015, PMID 25970248)",
    "CTNNB1": "Wnt (bulk TPM weak proxy)",
    "SERPINB9": "granzyme-B resistance",
    "CSF1": "TAM recruitment",
    "CXCL1": "CXCR2/MDSC recruitment",
    "CXCL5": "CXCR2/MDSC recruitment",
}
# Circular IFN/cytotoxic markers — shown as contrast, expected POSITIVE.
CIRCULAR = ["CD274", "IDO1", "CXCL9", "CD8A", "GZMB", "PRF1", "HLA-E", "HLA-A"]

# genuinely COLD gyn cohorts (UCEC_MSI is MSI-H/hot -> a low-exclusion control)
GYN_COLD = ["OV", "BRCA_Basal", "UCEC_CNL", "UCEC_CNH"]
HOT = ["SKCM"]


def main() -> int:
    apd1 = apd1_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    orr = pd.Series({c: apd1[c] for c in mat.index})
    print(f"cohorts with aPD1 + RNA-seq expression: {len(mat)}")
    print(f"cohorts: {sorted(mat.index)}\n")

    # ---- 1. broad screen --------------------------------------------------
    rows = []
    for g in mat.columns:
        col = mat[g]
        ok = col.notna()
        if ok.sum() < MIN_COHORTS:
            continue
        rho, p = spearmanr(col[ok], orr[ok])
        rows.append((g, rho, p, int(ok.sum()), float(col[ok].median())))
    screen = pd.DataFrame(rows, columns=["gene", "rho", "p", "n", "med_log10tpm"])
    screen = screen.sort_values("rho")

    print("=== Most NEGATIVELY correlated with aPD1 ORR (candidate exclusion) ===")
    print(screen.head(30).to_string(index=False,
          float_format=lambda x: f"{x:+.2f}"))
    print("\n=== Most POSITIVELY correlated (inflamed / hot) ===")
    print(screen.tail(15).iloc[::-1].to_string(index=False,
          float_format=lambda x: f"{x:+.2f}"))

    # ---- 2. gyn-vs-SKCM contrast -----------------------------------------
    gyn = [c for c in GYN_COLD if c in mat.index]
    hot = [c for c in HOT if c in mat.index]
    contrast = (mat.loc[gyn].mean(axis=0) - mat.loc[hot].mean(axis=0)).dropna()
    contrast = contrast.sort_values(ascending=False)
    print(f"\n=== High in gyn-cold {gyn} vs low in {hot} (top 30) ===")
    print(contrast.head(30).to_string(float_format=lambda x: f"{x:+.2f}"))

    # intersection: negative screen AND gyn-high/SKCM-low
    neg = set(screen[screen["rho"] < -0.15]["gene"])
    gyn_high = set(contrast[contrast > np.log10(3)].index)  # >3x higher
    inter = sorted(neg & gyn_high,
                   key=lambda g: screen.set_index("gene").loc[g, "rho"])
    print(f"\n=== Intersection (rho<-0.15 AND >3x gyn-high vs SKCM): {len(inter)} ===")
    sc = screen.set_index("gene")
    for g in inter[:40]:
        print(f"  {g:12s} rho={sc.loc[g,'rho']:+.2f}  "
              f"gyn-SKCM={contrast[g]:+.2f} log10")

    # ---- 3. curated panel scorecard --------------------------------------
    print("\n=== Curated causal-exclusion panel vs unbiased ranking ===")
    sc_rank = screen.reset_index(drop=True)
    pos = {g: i for i, g in enumerate(sc_rank["gene"])}
    n_tot = len(sc_rank)
    for g, why in CURATED.items():
        if g in pos:
            r = sc.loc[g, "rho"]
            pct = 100 * pos[g] / n_tot
            print(f"  {g:10s} rho={r:+.2f}  rank {pos[g]+1}/{n_tot} "
                  f"({pct:4.1f}%ile)  {why}")
        else:
            print(f"  {g:10s} (not detected in >= {MIN_COHORTS} cohorts)  {why}")

    print("\n=== Circular IFN/cytotoxic contrast (expected POSITIVE) ===")
    for g in CIRCULAR:
        if g in pos:
            print(f"  {g:10s} rho={sc.loc[g,'rho']:+.2f}  rank {pos[g]+1}/{n_tot}")
        else:
            print(f"  {g:10s} (not detected)")

    # ---- 4. mechanism-group composites: gyn-high/SKCM-low + aPD1 ----------
    # The two curated exclusion signatures live in
    # data/therapy-response-signatures.csv (therapy_class aPD1_exclusion_*) so
    # they are a single source of truth shared with the rest of the package.
    sigs = curated_exclusion_genes()
    tgfb_response = sigs["TGFb_response"]
    wnt = sigs["Wnt"]
    # Split TGF-beta into ligand/CAF vs response/contractile arms — the screen
    # shows they move oppositely across cancer types.
    GROUPS = {
        "TGFb_ligand_CAF": ["TGFB1", "TGFB3", "FAP", "CXCL12", "LRRC15",
                            "POSTN", "COL11A1"],
        "TGFb_response_myCAF": tgfb_response,
        "Wnt_axis": wnt,
        "PGE2_COX2": ["PTGS2", "PTGES"],
        "Angiogenic": ["VEGFA"],
        "Myeloid_CXCR2": ["CSF1", "CXCL1", "CXCL5"],
        "Gyn_tolerance": ["VTCN1", "MUC1", "FOLR1"],
    }

    def zcomposite(genes: list[str]) -> pd.Series:
        present = [g for g in genes if g in mat.columns]
        z = (mat[present] - mat[present].mean()) / mat[present].std(ddof=0)
        return z.mean(axis=1)

    def evaluate(comp: pd.Series, label: str) -> dict:
        ok = comp.notna()
        rho, p = spearmanr(comp[ok], orr[ok])
        gv = (comp.loc[[c for c in gyn if c in comp.index]].mean()
              - comp.loc[[c for c in hot if c in comp.index]].mean())
        # OLS ORR ~ comp
        x = comp[ok].to_numpy()
        y = orr[ok].to_numpy()
        slope, icpt = np.polyfit(x, y, 1)
        yhat = slope * x + icpt
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return {"label": label, "rho": rho, "p": p, "gyn_vs_skcm": gv, "r2": r2}

    print("\n=== Mechanism-group composites (z-scored, mean) ===")
    print(f"{'group':22s} {'rho_aPD1':>9s} {'p':>7s} "
          f"{'gyn-SKCM':>9s} {'OLS_R2':>7s}")
    grp_results = {}
    for name, genes in GROUPS.items():
        r = evaluate(zcomposite(genes), name)
        grp_results[name] = r
        print(f"{name:22s} {r['rho']:+9.2f} {r['p']:7.3f} "
              f"{r['gyn_vs_skcm']:+9.2f} {r['r2']:+7.2f}")

    # ---- 5. greedy forward selection (descriptive; LOO-checked) -----------
    POOL = sorted({g for gs in GROUPS.values() for g in gs}
                  | {"GAS6", "ISLR", "PMEPA1", "TGFB2", "ACTA2", "CTNNB1"})
    POOL = [g for g in POOL if g in mat.columns]
    chosen: list[str] = []
    best_rho = 0.0
    while True:
        cand = None
        for g in POOL:
            if g in chosen:
                continue
            rho = spearmanr(zcomposite(chosen + [g]),
                            orr, nan_policy="omit").statistic
            if rho < best_rho - 1e-9:
                best_rho, cand = rho, g
        if cand is None:
            break
        chosen.append(cand)
    print(f"\n=== Greedy composite minimizing aPD1 rho ({len(chosen)} genes) ===")
    print(f"  genes: {chosen}")
    comp = zcomposite(chosen)
    r = evaluate(comp, "greedy")
    print(f"  rho={r['rho']:+.2f} (p={r['p']:.3f})  "
          f"gyn-SKCM={r['gyn_vs_skcm']:+.2f}  OLS R2={r['r2']:+.2f}")
    # leave-one-cohort-out stability of rho
    loo = []
    idx = list(comp.dropna().index)
    for drop in idx:
        keep = [c for c in idx if c != drop]
        loo.append(spearmanr(comp[keep], orr[keep]).statistic)
    print(f"  LOO rho range: [{min(loo):+.2f}, {max(loo):+.2f}] "
          f"(median {float(np.median(loo)):+.2f})")

    # ---- 5b. COMBINED curated signature: SKCM vs gyn separation + aPD1 ----
    combined = zcomposite(tgfb_response + wnt)
    rc = evaluate(combined, "TGFb_response+Wnt (combined)")
    print(f"\n=== COMBINED curated signature ({len(tgfb_response+wnt)} genes:"
          f" TGFb-response + Wnt) ===")
    print(f"  aPD1 Spearman rho={rc['rho']:+.2f} (p={rc['p']:.3f}), "
          f"OLS R2={rc['r2']:+.2f}")
    ranked = combined.dropna().sort_values()
    pctile = {c: 100 * i / (len(ranked) - 1)
              for i, c in enumerate(ranked.index)}
    skcm_v = combined.get("SKCM", float("nan"))
    print(f"  SKCM composite z={skcm_v:+.2f}  "
          f"(rank {ranked.index.get_loc('SKCM')+1}/{len(ranked)}, "
          f"{pctile['SKCM']:.0f}%ile -> near the COLD-LOW end)")
    print("  gyn-cold cohorts:")
    for c in gyn:
        print(f"    {c:12s} z={combined[c]:+.2f}  ({pctile[c]:.0f}%ile)  "
              f"aPD1={orr[c]:.0f}%")
    gyn_mean = combined.loc[gyn].mean()
    print(f"  gyn mean z={gyn_mean:+.2f} vs SKCM z={skcm_v:+.2f}  "
          f"-> separation {gyn_mean - skcm_v:+.2f} z-units")
    # Mann-Whitney: gyn-cold vs all aPD1-responsive (ORR>=25%) cohorts
    resp = [c for c in combined.dropna().index if orr[c] >= 25]
    from scipy.stats import mannwhitneyu
    u, pu = mannwhitneyu(combined.loc[gyn], combined.loc[resp],
                         alternative="greater")
    print(f"  gyn-cold > responders (ORR>=25%, n={len(resp)}): "
          f"Mann-Whitney p={pu:.3f}")

    # ---- 5c. stepwise refinement: maximize gyn-high / hot-low separation --
    # Objective: gyn-cold cohorts should sit at HIGH composite percentile and
    # the LOW end should be immune-hot/responsive tumors. Score =
    # mean_pctile(gyn-cold) - mean_pctile(aPD1 responders, ORR>=25%).
    # Greedy add/remove over a *literature-supported* exclusion pool only.
    LIT_POOL = sorted({g for gs in GROUPS.values() for g in gs
                       if gs is not GROUPS["Gyn_tolerance"]}
                      | set(CURATED) | {"GAS6", "ISLR"})
    LIT_POOL = [g for g in LIT_POOL if g in mat.columns]
    gyn_set = [c for c in GYN_COLD if c in mat.index]
    resp_set = [c for c in mat.index if orr[c] >= 25]

    def pctile_score(genes: list[str]) -> float:
        if not genes:
            return -1e9
        comp = zcomposite(genes).dropna()
        rank = comp.rank(pct=True)
        return rank.loc[gyn_set].mean() - rank.loc[resp_set].mean()

    cur = list(tgfb_response + wnt)
    best = pctile_score(cur)
    while True:
        moves = []
        for g in LIT_POOL:  # try add and remove
            trial = cur + [g] if g not in cur else [x for x in cur if x != g]
            if trial:
                moves.append((pctile_score(trial), g, trial))
        moves.sort(reverse=True)
        if moves and moves[0][0] > best + 1e-9:
            best, _, cur = moves[0]
        else:
            break
    refined = cur
    rr = evaluate(zcomposite(refined), "refined")
    comp_ref = zcomposite(refined).dropna()
    rk = comp_ref.rank(pct=True)
    print(f"\n=== Stepwise-refined panel ({len(refined)} genes) ===")
    print(f"  genes: {sorted(refined)}")
    print(f"  gyn-pctile - responder-pctile = {best:+.2f}  "
          f"(aPD1 rho={rr['rho']:+.2f}, R2={rr['r2']:+.2f})")
    print("  gyn-cold percentiles:")
    for c in gyn_set:
        print(f"    {c:12s} {100*rk[c]:4.0f}%ile  aPD1={orr[c]:.0f}%")
    low5 = comp_ref.sort_values().head(6).index
    print(f"  lowest-exclusion cohorts (should be hot): "
          f"{[(c, int(orr[c])) for c in low5]}")

    # ---- 6. regression scatter plot --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    # left: combined curated signature (TGFb-response + Wnt, from CSV)
    for ax, comp_s, title in [
        (axes[0], combined,
         f"curated TGF-beta-response + Wnt ({len(tgfb_response+wnt)} genes)"),
        (axes[1], comp, f"greedy composite ({len(chosen)} genes)")]:
        ok = comp_s.notna()
        x, y = comp_s[ok], orr[ok]
        ax.scatter(x, y, s=28, alpha=0.8)
        for c in x.index:
            ax.annotate(c, (x[c], y[c]), fontsize=6, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")
        s, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, s * xs + b, "r--", lw=1)
        rho, p = spearmanr(x, y)
        ax.set_title(f"{title}\nSpearman rho={rho:+.2f} (p={p:.3f})", fontsize=9)
        ax.set_xlabel("composite expression (mean z log10 TPM)")
        ax.set_ylabel("anti-PD-1 ORR (%)")
        ax.grid(alpha=0.3)
    fig.suptitle("Causal T-cell-exclusion composites vs anti-PD-1 response "
                 "(RNA-seq cohorts; IFN/cytotoxic markers excluded)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "exclusion_composite_vs_apd1.png", dpi=130)
    print(f"\nwrote {OUT/'exclusion_composite_vs_apd1.png'}")

    OUT.mkdir(exist_ok=True)
    screen.to_csv(OUT / "_apd1_gene_screen.csv", index=False)
    contrast.rename("gyn_minus_skcm_log10").to_csv(
        OUT / "_apd1_gyn_vs_skcm.csv")
    print(f"\nwrote {OUT/'_apd1_gene_screen.csv'} and "
          f"{OUT/'_apd1_gyn_vs_skcm.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
