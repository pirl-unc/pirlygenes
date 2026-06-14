#!/usr/bin/env python
"""Per-gene immunosuppressive expression vs immune-checkpoint response, across
cancer types.

One scatter panel per gene: x = a cohort's objective response rate, y = the
cohort's median expression (TPM, log axis); each point is one cancer type. The
panel title carries the cross-type Spearman rho. A genuine *constitutive*
suppressor should track DOWN with response (rho < 0); a positive rho flags a
gene that is really an IFN-induced *readout* of an active response (circular),
not a cause — which is why the canonical checkpoint ligand CD274/PD-L1 is shown
only as a labelled CONTRAST.

Gene panel (constitutive immunosuppression):
  * secreted immune-EXCLUSION — TGFB1, TGFB2, WNT11, WNT5A, IL10
    (the TGF-beta / non-canonical Wnt exclusion axis; secreted, tumor-intrinsic)
  * surface / secreted checkpoint ligands kept from the prior screen for their
    real negative signal — CD47 (don't-eat-me), NECTIN2 (CD112/TIGIT-PVR axis),
    LGALS9 (galectin-9 / TIM-3)
  * SUPPRESSOR composite = mean z(log TPM) over the eight genes
  * CD274 (PD-L1) — circular IFN-induced contrast (expected rho > 0)

Two response axes (same gene grid):
  1. anti-PD-1 MONOTHERAPY  — drops the PD-L1 and PD-1+CTLA-4 cohorts, the strict
     causal axis (mirrors the cta_*_vs_apd1 / factor-contribution plots).
  2. broad ICI              — pools PD-1 + PD-L1 + PD-1/CTLA-4-combo ORRs. The
     PD-L1 (atezolizumab, 1 cohort) and ipi+nivo combo (7 cohorts) arms are too
     sparse to stand alone, so they are folded into this broad axis rather than
     plotted separately; the combo/PD-L1 cohorts are ringed on the broad figure.

Sarcoma cohorts are coloured distinctly (14 subtypes now carry both an ORR and
reference expression).

    python analyses/suppressor_genes_vs_apd1.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _apd1_factors import cohort_gene_matrix, zscore, _pool_dict  # noqa: E402
from _panels import display_label, fold  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402
from pirlygenes.load_dataset import get_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"

# constitutive immunosuppressive genes, plotted one panel each (secreted
# immune-exclusion first, then the surface/secreted checkpoint ligands kept from
# the prior screen for their genuine negative signal).
SUPPRESSORS = ["TGFB1", "TGFB2", "WNT11", "WNT5A", "IL10",
               "CD47", "NECTIN2", "LGALS9"]
CONTRAST = "CD274"          # PD-L1 — circular IFN-induced readout (expect rho>0)

_SARC_COLOR = "#e85d04"     # sarcoma cohorts (the user's question — make visible)
_OTHER_COLOR = "#1b4965"
_RING = "#c1121f"           # PD-L1 / combo cohorts (broad-ICI only)


def _orr_axis(*, strict: bool) -> dict[str, float]:
    """``{cancer_code: ORR %}``. ``strict`` keeps only PD-1 MONOTHERAPY rows
    (drops PD-L1 + PD-1/CTLA-4 combo); broad keeps every checkpoint arm. The
    drug_target filter runs on the UNPOOLED rows, BEFORE colorectal pooling, so a
    pooled code (CRC_*) can never slip past an unpooled drop set."""
    df = get_data("cancer-apd1-response.csv")
    if strict:
        df = df[df["drug_target"] == "PD-1"]
    return _pool_dict(dict(zip(df["cancer_code"],
                               df["apd1_orr_pct"].astype(float))))


def _nonmono_codes() -> set[str]:
    """Pooled codes whose ORR comes from a PD-L1 or PD-1/CTLA-4-combo arm (ringed
    on the broad-ICI figure)."""
    df = get_data("cancer-apd1-response.csv")
    raw = set(df.loc[df["drug_target"].isin(["PD-L1", "PD-1+CTLA-4"]),
                     "cancer_code"].astype(str))
    pooled = _pool_dict({c: 1.0 for c in raw})   # map raw codes through CRC pool
    return set(pooled)


def _tpm_matrix(codes) -> pd.DataFrame:
    """cohort x gene LINEAR TPM (cohort_gene_matrix gives log10(TPM+1))."""
    logm = cohort_gene_matrix(list(codes))
    return 10.0 ** logm - 1.0


def _series_for(mat: pd.DataFrame, gene: str) -> pd.Series | None:
    """Linear-TPM cohort series for a gene, proteoform-folded onto the matrix
    columns (single-copy genes pass through). None if absent."""
    for g in fold([gene]):
        if g in mat.columns:
            return mat[g]
    return None


def _rho(y: pd.Series, orr: pd.Series):
    ok = y.notna() & orr.notna()
    if ok.sum() < 4:
        return np.nan
    return spearmanr(y[ok], orr[ok]).statistic


def _panel(ax, y: pd.Series, orr: pd.Series, title: str, *, logy: bool,
           ring: set[str]):
    # only cohorts with BOTH a finite ORR and a finite expression value (a gene
    # unmeasured in a cohort is missing, not zero — drop, don't plot a phantom);
    # on a log y-axis a non-positive TPM is unplottable, so require y>0 there.
    codes = [c for c in y.index if c in orr.index
             and np.isfinite(y[c]) and np.isfinite(orr[c])
             and (y[c] > 0 if logy else True)]
    xv = orr.reindex(codes).to_numpy()
    yv = y.reindex(codes).to_numpy()
    colors = [_SARC_COLOR if c.startswith("SARC") else _OTHER_COLOR for c in codes]
    edge = [_RING if c in ring else "white" for c in codes]
    lw = [1.4 if c in ring else 0.4 for c in codes]
    ax.scatter(xv, yv, c=colors, s=34, edgecolor=edge, linewidth=lw, zorder=3)
    for c, x, yy in zip(codes, xv, yv):
        ax.annotate(c, (x, yy), fontsize=4, alpha=0.6,
                    xytext=(2, 2), textcoords="offset points")
    if logy:
        ax.set_yscale("log")
    rho = _rho(y, orr)
    ax.set_title(f"{title}  (ρ={rho:+.2f})", fontsize=9)
    ax.set_xlabel("ORR (%)", fontsize=7)
    ax.set_ylabel("TPM" if logy else "mean z(log TPM)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, zorder=0)


def _figure(mat: pd.DataFrame, orr_map: dict, axis_label: str, fname: str,
            figdir: Path, *, ring: set[str]) -> None:
    orr = pd.Series(orr_map, dtype=float)
    series = {g: _series_for(mat, g) for g in SUPPRESSORS}
    present = [g for g, s in series.items() if s is not None]
    # composite = mean z(log TPM) over the present suppressors (built straight
    # from the resolved linear-TPM series, not by re-folding column names)
    logdf = pd.DataFrame({g: np.log10(series[g] + 1.0) for g in present})
    composite = logdf.apply(zscore).mean(axis=1)
    contrast = _series_for(mat, CONTRAST)

    panels = present + ["__composite__", "__contrast__"]
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, panels):
        if key == "__composite__":
            _panel(ax, composite, orr, "SUPPRESSOR composite (z)",
                   logy=False, ring=ring)
        elif key == "__contrast__":
            if contrast is not None:
                _panel(ax, contrast, orr, f"{display_label(CONTRAST)}/PD-L1 "
                       "[contrast]", logy=True, ring=ring)
            else:
                ax.axis("off")
        else:
            _panel(ax, series[key], orr, display_label(key),
                   logy=True, ring=ring)
    for ax in axes[len(panels):]:
        ax.axis("off")
    n = int((pd.Series(composite).notna() & orr.reindex(composite.index)
             .notna()).sum())
    sarc = sum(c.startswith("SARC") for c in composite.index if c in orr.index)
    fig.suptitle(
        f"Immunosuppressive gene expression vs {axis_label} "
        f"(RNA-seq cohorts, n={n}; {sarc} sarcoma subtypes in orange)\n"
        "constitutive suppressors expected ρ<0; CD274/PD-L1 shown as circular "
        "IFN-induced contrast (expected ρ>0)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figdir / fname, dpi=300)
    plt.close(fig)
    print(f"wrote {fname}  (n={n} cohorts, {sarc} sarcoma)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT)

    mono = _orr_axis(strict=True)
    broad = _orr_axis(strict=False)
    mat = _tpm_matrix(set(broad))           # superset of cohorts
    ring = _nonmono_codes()

    _figure(mat, mono, "anti-PD-1 monotherapy ORR",
            "suppressor_genes_vs_apd1_mono.png", figdir, ring=set())
    _figure(mat, broad, "broad ICI ORR (PD-1 + PD-L1 + ipi/nivo combo)",
            "suppressor_genes_vs_ici.png", figdir, ring=ring)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
