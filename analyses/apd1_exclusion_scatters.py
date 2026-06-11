#!/usr/bin/env python
"""Two-axis scatters relating the causal aPD1 factors to response.

1. ``tgfb_vs_tmb_by_response.png`` — TGF-beta (TGFB2) expression vs tumor
   mutational burden, each cohort colored by its anti-PD-1 ORR. Visualises the
   two orthogonal axes of the model at once: antigen availability (TMB, x) and
   immune exclusion (TGF-beta, y); responders cluster high-TMB / low-exclusion.

2. ``cta_vs_apd1_by_exclusion/<set>.png`` — cancer-testis-antigen burden vs
   anti-PD-1 ORR, one panel per immune-exclusion gene set used as the color
   (TGF-beta, Wnt, Wnt-target, angiogenesis, adenosine, and the combined
   exclusion composite). Shows whether, at a given antigen load (CTA), high
   exclusion-program expression tracks with the non-responding cohorts.

All expression is the collapse_protein_identical + coverage-aware cohort matrix
(protein-identical CTA paralogs summed; see _apd1_factors). Honors APD1_RUN_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _apd1_factors import (SIGNATURE_META, apd1_map,  # noqa: E402
                           cohort_gene_matrix, cta_burden, curated_signatures,
                           tmb_map, with_parent)
from pirlygenes.expression.protein_groups import (  # noqa: E402
    fold_symbols_to_canonical)

OUT = Path(os.environ.get(
    "APD1_RUN_DIR",
    str(Path(__file__).resolve().parent / "outputs" / "apd1_causal_factors")))


def _out_path(name: str, *, subdir: str | None = None) -> Path:
    d = OUT / subdir if subdir else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _set_expression(mat: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean log10(TPM+1) over a gene set's members present in the (already
    protein-identical-collapsed) matrix; genes folded onto canonical symbols."""
    present = [g for g in fold_symbols_to_canonical(genes) if g in mat.columns]
    if not present:
        return pd.Series(np.nan, index=mat.index)
    return mat[present].mean(axis=1)


def _scatter(x, y, color, labels, *, xlabel, ylabel, clabel, title, path, cmap):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(x, y, c=color, cmap=cmap, s=90, edgecolor="black",
                    linewidth=0.5, zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(lab, (xi, yi), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    apd1 = apd1_map()
    tmb = tmb_map()
    mat = cohort_gene_matrix(list(apd1))
    mat = mat.loc[[c for c in mat.index if c in apd1]]
    codes = list(mat.index)

    orr = pd.Series({c: apd1[c] for c in codes})
    logtmb = pd.Series({c: np.log10(with_parent(tmb, c, np.nan)) for c in codes})
    cta = cta_burden(mat)

    # immune-exclusion gene sets (+ TGF-beta alone) for coloring
    sigs = curated_signatures()
    excl = {k.replace("aPD1_exclusion_", ""): v for k, v in sigs.items()
            if SIGNATURE_META.get(k, (None,))[0] == "exclusion"}
    set_expr = {name: _set_expression(mat, genes) for name, genes in excl.items()}
    # combined exclusion composite = mean over all exclusion genes
    all_excl = sorted({g for genes in excl.values() for g in genes})
    set_expr["exclusion_all"] = _set_expression(mat, all_excl)

    tgfb = set_expr.get("TGFb_response")

    # --- 1. TGF-beta vs TMB, colored by ORR ------------------------------
    _scatter(
        logtmb.values, tgfb.values, orr.values, codes,
        xlabel="log10(median TMB mut/Mb)",
        ylabel="TGF-β (TGFB2) log10(TPM+1)",
        clabel="anti-PD-1 ORR (%)",
        title="TGF-β vs TMB across cohorts (color = anti-PD-1 ORR)",
        path=_out_path("tgfb_vs_tmb_by_response.png"), cmap="viridis",
    )

    # --- 2. CTA burden vs ORR, one panel per exclusion set as color ------
    pretty = {
        "TGFb_response": "TGF-β (TGFB2)", "Wnt": "Wnt ligands",
        "Wnt_target": "Wnt target program", "angiogenesis": "angiogenesis",
        "adenosine": "adenosine", "exclusion_all": "exclusion composite",
    }
    for name, expr in set_expr.items():
        if not np.isfinite(expr.values).any():
            continue
        _scatter(
            cta.values, orr.values, expr.reindex(codes).values, codes,
            xlabel="CTA burden (# antigens ON, protein-identical-collapsed)",
            ylabel="anti-PD-1 ORR (%)",
            clabel=f"{pretty.get(name, name)} log10(TPM+1)",
            title=f"CTA burden vs anti-PD-1 ORR (color = {pretty.get(name, name)})",
            path=_out_path(f"{name}.png", subdir="cta_vs_apd1_by_exclusion"),
            cmap="magma_r",
        )
    print(f"wrote tgfb_vs_tmb_by_response.png + cta_vs_apd1_by_exclusion/ to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
