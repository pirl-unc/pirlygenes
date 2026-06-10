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

OUT = Path(os.environ.get("APD1_RUN_DIR",
          str(Path(__file__).resolve().parent / "outputs" / "apd1_causal_factors")))
OUT.mkdir(parents=True, exist_ok=True)

# Curated signatures (therapy-response-signatures.csv) are the source of truth;
# these are the NON-curated extras tested as controls / known failures:
# name -> (gene list, expected sign, circularity tag, one-line rationale)
_EXTRA_CONTROLS = {
    "antigen:CTA": (
        ["MAGEA1", "MAGEA3", "MAGEA4", "MAGEC2", "CTAG1B", "CTAG2", "PRAME",
         "SSX2", "GAGE1", "XAGE1B"], +1, "causal",
        "cancer-testis antigen expression (representative MAGEs; CTA panel owned "
        "by tsarina, not this CSV)"),
    "antigen:ERV_annotated": (
        ["ERV3-1", "ERVK-28", "ERVK3-1", "ERVMER34-1", "ERVW-1", "ERVFRD-1",
         "ERVH-1", "ERVV-1"], +1, "causal",
        "Ensembl-annotated ERVs (mostly placental syncytins - FAILS)"),
    "exclude:hypoxia": (
        ["CA9", "SLC2A1", "LDHA", "VEGFA", "PGK1"], -1, "borderline",
        "hypoxia (CA9 is also a ccRCC lineage marker - confounded, NOT curated)"),
    "exclude:myeloid_tolerance": (
        ["VSIG4", "MARCO", "CD163", "C1QA", "C1QB", "ARG1", "ALDH1A1", "IL10"],
        -1, "borderline",
        "resident tolerogenic myeloid (WRONG-SIGN: indexes infiltrate; NOT curated)"),
}
# axis -> expected response sign (antigen up, exclusion down, circular up)
_AXIS_SIGN = {"antigen": +1, "exclusion": -1, "circular": +1}
_TAG_COLOR = {"causal": "#2c7fb8", "borderline": "#d95f0e", "circular": "#999999"}


def _mechanisms() -> dict:
    """Build the test catalog: curated signatures (from CSV) + extra controls."""
    out = {}
    for cls, genes in curated_signatures().items():
        axis, sign, tag = SIGNATURE_META[cls]
        short = cls.replace("aPD1_", "").replace("exclusion_", "").replace(
            "circular_", "").replace("antigen_", "")
        out[f"{axis}:{short}"] = (genes, sign, tag, f"curated {cls}")
    out.update(_EXTRA_CONTROLS)
    return out


def main() -> int:
    apd1 = apd1_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    orr = pd.Series({c: apd1[c] for c in mat.index})

    rows = []
    for name, (genes, sign, tag, why) in _mechanisms().items():
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
    fig.savefig(OUT / "apd1_mechanism_screen.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
