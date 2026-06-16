#!/usr/bin/env python
"""The placenta's immune-privilege secretome, and which of it tumors co-opt.

The placenta is the body's professional immune-privileged semi-allograft: it
suppresses maternal T cells while being half-foreign. Its trophoblast secretome
is therefore a curated catalogue of "how to evade T cells" that tumors can
re-deploy. Two figures:

1. placental_celltype_specificity.png — HPA cell-type expression of the placental
   immune-privilege panel: trophoblast (syncytio / cyto / extravillous) vs T-cell
   / B-cell / NK / macrophage vs epithelial-max. The panel lights up in
   trophoblast + epithelium and is ~dark in T cells (that's the point — these are
   things a tumor can secrete that T cells don't).

2. placental_factors_vs_response_antigen.png — for the subset tumors actually
   RE-EXPRESS (measurable in bulk cohorts), cross-cancer-type Spearman rho of each
   factor vs anti-PD-1 monotherapy ORR, broad-ICI ORR, TMB, and CTA burden — plus
   a CD45(PTPRC)-partial aPD-1 column, since (like TGFB1) a raw response
   correlation can be an immune-infiltrate artefact.

Honest scope: the canonical pregnancy proteins (PSG family, hCG = CGA/CGB,
syncytins ERVW-1/ERVFRD-1, placental galectins LGALS13/14) are placenta-RESTRICTED
and silent in these bulk tumor cohorts; only GDF15 / PGF / FLT1 / INHBA / EBI3 /
HLA-G (+ B7-H4/VTCN1) are re-expressed and testable.

    python analyses/placental_immune_privilege.py
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
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _apd1_factors import (cohort_gene_matrix, cta_burden,  # noqa: E402
                           tmb_map, with_parent)
from _panels import fold  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402
from suppressor_genes_vs_apd1 import _orr_axis, _rho  # noqa: E402
from pirlygenes.load_dataset import get_data  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"

# Placental immune-privilege panel, grouped by mechanism (display order).
PLACENTAL = [
    "HLA-G",      # non-classical MHC-I: inhibits NK + T (the classic one)
    "GDF15",      # MIC-1/PLAB: blocks T-cell LFA-1 adhesion (visugromab target)
    "VTCN1",      # B7-H4: co-inhibitory ligand
    "EBI3",       # IL-35 / IL-27 subunit: Treg cytokine
    "INHBA",      # activin A (TGF-beta family)
    "PGF",        # placental growth factor (VEGF family)
    "FLT1",       # sVEGFR1: VEGF sink / anti-angiogenic
    "PAEP",       # glycodelin: induces T-cell apoptosis
    "PSG1", "PSG4", "PSG5",      # pregnancy-specific glycoproteins (induce TGFb/IL10)
    "CGA", "CGB5", "CGB3",       # hCG: Treg induction
    "ERVW-1", "ERVFRD-1", "ERVH48-1",  # syncytin-1/-2 / suppressyn (ISU domain)
    "LGALS13", "LGALS14",        # placental galectins (T-cell apoptosis)
]


def _hpa_celltype_panel():
    """(DataFrame rows=panel x selected cell types in nTPM, ordered col groups)."""
    hpa = get_data("hpa-cell-type-expression.csv").set_index("Symbol")
    cells = [c for c in hpa.columns if c != "Ensembl_Gene_ID"]
    troph = [c for c in ["Syncytiotrophoblasts", "Cytotrophoblasts",
                         "Extravillous trophoblasts"] if c in cells]
    immune = [c for c in ["T-cells", "B-cells", "NK-cells", "Macrophages",
                          "Hofbauer cells"] if c in cells]
    epi = [c for c in cells if any(k in c for k in
           ("epithel", "glandular", "keratino", "Hepatocyt", "enterocyt", "duct",
            "Basal", "Squamous", "Secretory", "Club", "Alveolar", "prostatic",
            "Urothel", "Melanocyt", "Cholangio"))]
    rows = [g for g in PLACENTAL if g in hpa.index]
    sub = hpa.loc[rows]
    out = sub[troph].copy()
    out["Epithelial(max)"] = sub[epi].max(axis=1)
    for c in immune:
        out[c] = sub[c]
    groups = ([("trophoblast", t) for t in troph]
              + [("tumor", "Epithelial(max)")]
              + [("immune", c) for c in immune])
    return out, groups


def _celltype_figure(figdir: Path) -> None:
    mat, groups = _hpa_celltype_panel()
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    norm = LogNorm(vmin=1, vmax=max(10, float(np.nanmax(mat.to_numpy()))))
    im = ax.imshow(mat.clip(lower=0.1).to_numpy(), aspect="auto", cmap="magma",
                   norm=norm)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    # group separators between trophoblast | tumor | immune
    prev, start = None, 0
    for i, (grp, _c) in enumerate(groups):
        if prev is not None and grp != prev:
            ax.axvline(i - 0.5, color="white", lw=2)
            ax.text((start + i - 1) / 2, -0.9, prev.upper(), ha="center",
                    fontsize=9, fontweight="bold")
            start = i
        prev = grp
    ax.text((start + len(groups) - 1) / 2, -0.9, prev.upper(), ha="center",
            fontsize=9, fontweight="bold")
    ax.set_title("Placental immune-privilege secretome — HPA cell-type expression\n"
                 "trophoblast + epithelium HIGH, T cells ~0 (what a tumor can "
                 "secrete that T cells don't)", fontsize=11, pad=26)
    fig.colorbar(im, ax=ax, shrink=0.5, label="nTPM (log)")
    fig.tight_layout()
    fig.savefig(figdir / "placental_celltype_specificity.png", dpi=300)
    plt.close(fig)


def _partial_rho(y: pd.Series, target: pd.Series, ctrl: pd.Series) -> float:
    """Spearman of y vs target controlling for ctrl (rank-residualised)."""
    df = pd.concat([y.rename("y"), target.rename("t"), ctrl.rename("z")],
                   axis=1).dropna()
    if len(df) < 5:
        return np.nan
    Z = np.column_stack([np.ones(len(df)), df["z"].rank()])
    ry = df["y"].rank() - Z @ np.linalg.lstsq(Z, df["y"].rank(), rcond=None)[0]
    rt = df["t"].rank() - Z @ np.linalg.lstsq(Z, df["t"].rank(), rcond=None)[0]
    return float(spearmanr(ry, rt).statistic)


def _tumor_correlation_figure(figdir: Path) -> None:
    mono, broad = _orr_axis(strict=True), _orr_axis(strict=False)
    M = cohort_gene_matrix(list(broad))
    codes = [c for c in M.index if c in broad]
    M = M.loc[codes]
    tmb = tmb_map()
    axes_data = {
        "aPD-1 mono": pd.Series({c: mono[c] for c in codes if c in mono}),
        "broad ICI": pd.Series({c: broad[c] for c in codes}),
        "TMB": pd.Series({c: np.log10(with_parent(tmb, c, np.nan)) for c in codes}),
        "CTA burden": cta_burden(M).reindex(codes),
    }
    cd45 = M["PTPRC"] if "PTPRC" in M.columns else None
    # only factors tumors actually re-express (measurable in bulk)
    measurable = []
    for g in PLACENTAL:
        cols = [c for c in fold([g]) if c in M.columns]
        if cols and int(((10 ** M[cols[0]] - 1) > 1).sum()) >= 15:
            measurable.append((g, cols[0]))
    rows, labels = [], []
    for g, col in measurable:
        y = M[col]
        r = {k: _rho(y, s) for k, s in axes_data.items()}
        r = {"aPD-1 mono": r["aPD-1 mono"],
             "aPD-1 | CD45": (_partial_rho(y, axes_data["aPD-1 mono"], cd45)
                              if cd45 is not None else np.nan),
             "broad ICI": r["broad ICI"], "TMB": r["TMB"],
             "CTA burden": r["CTA burden"]}
        rows.append(r)
        labels.append(g)
    R = pd.DataFrame(rows, index=labels)
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(R) + 2.2))
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
    im = ax.imshow(R.to_numpy(), aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(R.columns)))
    ax.set_xticklabels(R.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(R.index)))
    ax.set_yticklabels(R.index, fontsize=9)
    for i in range(len(R.index)):
        for j in range(len(R.columns)):
            v = R.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.33 else "black")
    ax.axvline(1.5, color="0.2", lw=1.5)   # response axes | antigen axes-ish divider
    ax.set_title("Tumor-co-opted placental factors vs response & antigen\n"
                 "(cross-cancer-type Spearman ρ; aPD-1|CD45 = controlled for "
                 "leukocyte infiltrate)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(figdir / "placental_factors_vs_response_antigen.png", dpi=300)
    plt.close(fig)
    print("measurable placental factors:", [g for g, _ in measurable])
    print(R.round(2).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT)
    _celltype_figure(figdir)
    _tumor_correlation_figure(figdir)
    print(f"wrote placental_celltype_specificity.png + "
          f"placental_factors_vs_response_antigen.png -> {figdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
