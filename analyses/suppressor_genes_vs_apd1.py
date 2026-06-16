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

Gene panel — SECRETED constitutive immunosuppression only (no membrane proteins):
  * TGF-beta / non-canonical Wnt exclusion axis — TGFB1, TGFB2, WNT11, WNT5A
  * IL10 — secreted immunosuppressive cytokine (Treg / M2)
  * LGALS9 — galectin-9, secreted TIM-3 ligand
  * SUPPRESSOR composite = MAX within-cohort percentile over the secreted genes
    (OR-logic: the cohort's single most-deployed secreted program, ranked within
    its own transcriptome — captures distinct biology a mean would dilute, and is
    scale-robust; mirrors the CTA-burden within-sample-percentile metric).
  * CXCL9 — secreted but IFN-INDUCED chemokine: a circular *readout* of an
    active T-cell response, shown as a contrast (expected rho > 0). The membrane
    checkpoint markers (CD47, NECTIN2, CD274/PD-L1) are excluded — secreted only.

Sample-size heterogeneity (n = 3 … 764 across cohorts; the distinct biology —
rare sarcoma subtypes, NPC, HL — lives in the SMALL cohorts, so they are kept,
not filtered): point AREA ∝ cohort n (radius ∝ √n) so noisy cohorts read small,
and each panel prints a robust ρ recomputed on n>=20 cohorts alongside the
all-cohort ρ. Points are coloured by coarse LINEAGE (epithelial / sarcoma /
neuroendocrine / heme / other) — the distinct-biology axis.

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

from _apd1_factors import cohort_gene_matrix, _pool_dict  # noqa: E402
from _panels import display_label, fold  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402
from pirlygenes.expression.accessors import cancer_reference_expression  # noqa: E402
from pirlygenes.gene_sets_cancer import cancer_type_registry  # noqa: E402
from pirlygenes.load_dataset import get_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"

# SECRETED constitutive immunosuppressive genes, one panel each (no membrane
# proteins — the TGF-beta / Wnt exclusion axis + IL10 + secreted galectin-9).
SUPPRESSORS = ["TGFB1", "TGFB2", "WNT11", "WNT5A", "IL10", "LGALS9"]
CONTRAST = "CXCL9"          # secreted but IFN-induced -> circular readout (rho>0)

# Coarse lineage classes (from the registry `family` column) — the distinct
# biology axis. "epithelial" = every carcinoma-* family + endocrine-epithelial +
# salivary (adenoid-cystic carcinoma); mesenchymal sarcomas, heme, and
# neuroendocrine are kept distinct; CNS/melanoma/germ-cell fall to "other".
_LINEAGE_COLOR = {
    "epithelial": "#1b4965",        # carcinomas
    "sarcoma": "#e85d04",           # mesenchymal
    "neuroendocrine": "#2a9d8f",
    "heme": "#9d4edd",
    "other": "#adb5bd",             # CNS, melanoma, germ-cell
}
_RING = "#c1121f"           # PD-L1 / combo cohorts (broad-ICI only)
_ROBUST_MIN_N = 20          # "robust" rho recomputed on cohorts with n >= this


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


def _lineage_class(code: str, reg: pd.DataFrame) -> str:
    """Coarse lineage class for colouring (epithelial / sarcoma / neuroendocrine
    / heme / other), from the registry ``family`` (base-code fallback)."""
    fam = "?"
    for k in (code, code.split("_")[0]):
        if k in reg.index:
            fam = str(reg.loc[k, "family"])
            break
    if fam.startswith("carcinoma") or fam in ("endocrine-epithelial", "salivary"):
        return "epithelial"
    if fam == "sarcoma":
        return "sarcoma"
    if fam.startswith("heme"):
        return "heme"
    if "neuroendocrine" in fam:
        return "neuroendocrine"
    return "other"


def _cohort_sizes(codes) -> pd.Series:
    """``{cancer_code: n_samples}`` for the richest RNA-seq source per code — the
    same richest-source-wins selection :func:`cohort_gene_matrix` uses — so point
    size can encode how reliable each cohort's median is."""
    fetch = list(dict.fromkeys(
        [c for c in codes if not c.startswith("CRC")] + ["UCEC", "UCS"]))
    long = cancer_reference_expression(cancer_types=fetch, normalize="tpm",
                                       include_provenance=True,
                                       collapse_cdna_identical=True)
    long = long[~long["processing_pipeline"].str.contains(
        "microarray_tpm_proxy", na=False)]
    n = (long.groupby(["cancer_code", "source_cohort"])["n_samples"]
         .max().reset_index())
    best = n.sort_values("n_samples").groupby("cancer_code").tail(1)
    return best.set_index("cancer_code")["n_samples"]


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


def _rho(y: pd.Series, orr: pd.Series, keep=None):
    """Spearman ρ of y vs ORR over their shared finite cohorts; ``keep`` (a set of
    codes) optionally restricts to a robust subset (e.g. n>=20). NaN if <4 pairs.
    Aligns y and orr on their shared index (they cover different cohort sets)."""
    df = pd.DataFrame({"y": y, "orr": orr}).dropna()
    if keep is not None:
        df = df[df.index.isin(keep)]
    if len(df) < 4:
        return np.nan
    return spearmanr(df["y"], df["orr"]).statistic


def _panel(ax, y: pd.Series, orr: pd.Series, title: str, *, logy: bool,
           ring: set[str], lineage: dict, n: pd.Series, robust: set[str]):
    # only cohorts with BOTH a finite ORR and a finite expression value (a gene
    # unmeasured in a cohort is missing, not zero — drop, don't plot a phantom);
    # on a log y-axis a non-positive value is unplottable, so require y>0 there.
    codes = [c for c in y.index if c in orr.index
             and np.isfinite(y[c]) and np.isfinite(orr[c])
             and (y[c] > 0 if logy else True)]
    xv = orr.reindex(codes).to_numpy()
    yv = y.reindex(codes).to_numpy()
    colors = [_LINEAGE_COLOR[lineage.get(c, "other")] for c in codes]
    edge = [_RING if c in ring else "white" for c in codes]
    lw = [1.4 if c in ring else 0.4 for c in codes]
    # point AREA ∝ n_samples (radius ∝ √n): tiny, noisy cohorts read small.
    sizes = [18.0 + 4.0 * np.sqrt(float(n.get(c, 1.0))) for c in codes]
    ax.scatter(xv, yv, c=colors, s=sizes, edgecolor=edge, linewidth=lw, zorder=3)
    for c, x, yy in zip(codes, xv, yv):
        ax.annotate(c, (x, yy), fontsize=4, alpha=0.55,
                    xytext=(2, 2), textcoords="offset points")
    if logy:
        ax.set_yscale("log")
    rho = _rho(y, orr)
    rho_r = _rho(y, orr, keep=robust)
    ax.set_title(f"{title}\nρ={rho:+.2f}  (n≥{_ROBUST_MIN_N}: {rho_r:+.2f})",
                 fontsize=8)
    ax.set_xlabel("ORR (%)", fontsize=7)
    ax.set_ylabel(_ylabel(logy), fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, zorder=0)


def _ylabel(logy: bool) -> str:
    # only the composite panel is non-log; everything else is a TPM axis
    return "TPM" if logy else "max secreted gene\nwithin-cohort %ile"


def _max_within_cohort_pctile(mat: pd.DataFrame, cols) -> pd.Series:
    """OR-logic composite that captures DISTINCT biology: for each cohort, the
    within-transcriptome percentile (rank of a gene against ALL genes in that same
    cohort, 0–100) of its STRONGEST secreted-exclusion gene. A mean dilutes a
    tumor that deploys only one program; the max asks "is at least one secreted
    suppressor highly expressed for this tumor's own baseline?" — scale/pipeline-
    robust (a per-cohort rank, not an absolute TPM), mirroring the CTA-burden
    within-sample-percentile metric."""
    pct = mat.rank(axis=1, pct=True) * 100.0     # within-cohort percentile per gene
    return pct[[c for c in cols if c in pct.columns]].max(axis=1)


def _figure(mat: pd.DataFrame, orr_map: dict, axis_label: str, fname: str,
            figdir: Path, *, ring: set[str], lineage: dict, n: pd.Series) -> None:
    orr = pd.Series(orr_map, dtype=float)
    series = {g: _series_for(mat, g) for g in SUPPRESSORS}
    present = [g for g, s in series.items() if s is not None]
    # composite = MAX within-cohort percentile over the secreted genes (OR-logic,
    # distinct biology) — see _max_within_cohort_pctile.
    composite = _max_within_cohort_pctile(mat, [series[g].name for g in present])
    contrast = _series_for(mat, CONTRAST)
    robust = {c for c in n.index if float(n.get(c, 0)) >= _ROBUST_MIN_N}
    kw = dict(ring=ring, lineage=lineage, n=n, robust=robust)

    panels = present + ["__composite__", "__contrast__"]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, panels):
        if key == "__composite__":
            _panel(ax, composite, orr, "SUPPRESSOR composite\n(max within-cohort "
                   "%ile)", logy=False, **kw)
        elif key == "__contrast__":
            if contrast is not None:
                _panel(ax, contrast, orr,
                       f"{display_label(CONTRAST)} [IFN-circular contrast]",
                       logy=True, **kw)
            else:
                ax.axis("off")
        else:
            _panel(ax, series[key], orr, display_label(key), logy=True, **kw)
    for ax in axes[len(panels):]:
        ax.axis("off")
    _lineage_legend(fig)
    ncoh = int((composite.notna() & orr.reindex(composite.index).notna()).sum())
    sarc = sum(c.startswith("SARC") for c in composite.index if c in orr.index)
    fig.suptitle(
        f"Secreted immunosuppressive gene expression vs {axis_label} "
        f"(RNA-seq cohorts, n={ncoh}; {sarc} sarcoma subtypes)\n"
        "colour = lineage · point size ∝ √(cohort n) · secreted suppressors "
        "expected ρ<0; CXCL9 (IFN-induced) is the circular contrast (ρ>0)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figdir / fname, dpi=300)
    plt.close(fig)
    print(f"wrote {fname}  (n={ncoh} cohorts, {sarc} sarcoma)", flush=True)


def _lineage_legend(fig) -> None:
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=col, markeredgecolor="white", label=name)
               for name, col in _LINEAGE_COLOR.items()]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor="none", markeredgecolor=_RING,
                          markeredgewidth=1.4, label="PD-L1 / ipi+nivo cohort"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.005))


def main() -> int:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT)

    mono = _orr_axis(strict=True)
    broad = _orr_axis(strict=False)
    mat = _tpm_matrix(set(broad))           # superset of cohorts
    ring = _nonmono_codes()
    reg = cancer_type_registry().set_index("code")
    lineage = {c: _lineage_class(c, reg) for c in mat.index}
    n = _cohort_sizes(list(mat.index))

    _figure(mat, mono, "anti-PD-1 monotherapy ORR",
            "suppressor_genes_vs_apd1_mono.png", figdir,
            ring=set(), lineage=lineage, n=n)
    _figure(mat, broad, "broad ICI ORR (PD-1 + PD-L1 + ipi/nivo combo)",
            "suppressor_genes_vs_ici.png", figdir,
            ring=ring, lineage=lineage, n=n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
