#!/usr/bin/env python
"""Enumerate candidate CAUSAL mechanisms of anti-PD-1 response and test each as a
measurable bulk-RNA signature.

Companion to apd1_causal_factors.py (which fits the 2-axis model + map). This
script does NOT touch that model; it screens a broader catalog of mechanism
signatures to see which ones (a) track ORR cross-type and (b) are genuinely
causal rather than circular readouts of an immune response already underway.

Circularity tag per mechanism:
  causal     - constitutive tumor/stroma property upstream of T-cell engagement
  borderline - partly inducible by IFN / confounded with total infiltrate
  circular   - a readout of the response itself (excluded from any predictor)

Each signature = mean z-scored log10 TPM of its genes across the aPD1 cohorts.
We report signed Spearman vs ORR and where the diagnostic outliers sit
(KIRC = low-antigen responder; LIHC = the residual we're chasing).

    python analyses/apd1_mechanism_screen.py
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

from _apd1_factors import (SIGNATURE_META, apd1_map,  # noqa: E402
                           cohort_gene_matrix, curated_signatures)
from _panels import fold, mechanism_controls  # noqa: E402

OUT = Path(os.environ.get("APD1_RUN_DIR",
          str(Path(__file__).resolve().parent / "outputs" / "apd1_causal_factors")))
OUT.mkdir(parents=True, exist_ok=True)

_TAG_COLOR = {"causal": "#2c7fb8", "borderline": "#d95f0e", "circular": "#999999"}


def _mechanisms() -> dict:
    """Build the test catalog: curated signatures (from the CSV) + the
    non-curated control panels (from _panels.mechanism_controls). Every panel is
    proteoform-folded so its symbols match the matrix's proteoform-ID columns."""
    out = {}
    for cls, genes in curated_signatures().items():
        axis, sign, tag = SIGNATURE_META[cls]
        short = cls.replace("aPD1_", "").replace("exclusion_", "").replace(
            "circular_", "").replace("antigen_", "")
        out[f"{axis}:{short}"] = (fold(genes), sign, tag, f"curated {cls}")
    out.update(mechanism_controls())
    return out


def main() -> int:
    apd1 = apd1_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    orr = pd.Series({c: apd1[c] for c in mat.index})

    rows = []
    for name, (genes, sign, tag, why) in _mechanisms().items():
        # genes are already proteoform-folded by _mechanisms (via _panels.fold),
        # so they match the matrix's proteoform-ID columns directly.
        present = [g for g in genes if g in mat.columns]
        if len(present) < 2:
            print(f"  SKIP {name}: <2 genes present ({present})")
            continue
        z = (mat[present] - mat[present].mean()) / mat[present].std(ddof=0)
        comp = z.mean(axis=1)
        rho, p = spearmanr(comp, orr)
        rk = comp.rank(pct=True)
        rows.append({
            "mechanism": name, "tag": tag, "n": len(present),
            "rho": rho, "p": p, "expect": "+" if sign > 0 else "-",
            "agrees": (np.sign(rho) == sign) or abs(rho) < 0.1,
            "KIRC_pctile": round(100 * rk["KIRC"]),
            "LIHC_pctile": round(100 * rk["LIHC"]),
            "why": why,
        })
    df = pd.DataFrame(rows).sort_values("rho")

    print("\n=== Causal-mechanism screen (signed Spearman vs aPD1 ORR) ===")
    print(f"{'mechanism':28s} {'tag':10s} {'n':>2s} {'rho':>6s} {'p':>6s} "
          f"{'exp':>3s} {'KIRC':>4s} {'LIHC':>4s}")
    for r in df.itertuples():
        flag = "" if r.agrees else "  <-- WRONG SIGN"
        print(f"{r.mechanism:28s} {r.tag:10s} {r.n:2d} {r.rho:+6.2f} "
              f"{r.p:6.3f} {r.expect:>3s} {r.KIRC_pctile:3d}% "
              f"{r.LIHC_pctile:3d}%{flag}")

    print("\nNotes:")
    for r in df.itertuples():
        print(f"  {r.mechanism:28s} {r.why}")

    _plot(df)
    df.to_csv(OUT / "_apd1_mechanism_screen.csv", index=False)
    print(f"\nwrote {OUT/'apd1_mechanism_screen.png'} and "
          f"{OUT/'_apd1_mechanism_screen.csv'}")
    return 0


def _plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = [_TAG_COLOR[t] for t in df["tag"]]
    ax.barh(df["mechanism"], df["rho"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    for i, r in enumerate(df.itertuples()):
        ax.annotate(f"{r.rho:+.2f}" + ("" if r.agrees else " !"),
                    (r.rho, i), fontsize=7, va="center",
                    ha="left" if r.rho >= 0 else "right")
    ax.set_xlabel("signed Spearman rho vs anti-PD-1 ORR")
    ax.set_title("Candidate causal mechanisms of anti-PD-1 response\n"
                 "blue=causal  orange=borderline/confounded  grey=circular "
                 "(outcome readout)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _TAG_COLOR.values()]
    ax.legend(handles, _TAG_COLOR.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "apd1_mechanism_screen.png", dpi=300)


if __name__ == "__main__":
    raise SystemExit(main())
