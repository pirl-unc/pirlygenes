"""Candidate immunosuppressive / inhibitor genes vs ICI response.

This is a compact companion to ``suppressor_genes_vs_apd1.py``. Instead of one
scatter per gene, it summarizes the WNT11-like negative candidates and the
checkpoint/readout contrast genes in a single Spearman-rho bar plot. Bar hue
distinguishes secreted/soluble factors, surface proteins, and
cytosolic/intracellular enzymes or pathway readouts; within each hue, the darker
shade is anti-PD-1 and the lighter shade is broad ICI.

Negative rho means the gene is higher in low-response cancers; positive rho
marks genes that behave like inflamed/readout markers across cancer types.

Run:  python analyses/inhibitor_candidates_vs_ici.py
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
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _apd1_factors import apd1_map, cohort_gene_matrix  # noqa: E402
from _panels import display_label, fold  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402
from pirlygenes import gene_sets_cancer as gsc  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"


# Genes from the quick scorecard:
#   * negative candidates: candidates with WNT11-like negative correlation.
#   * secreted controls: genes often named as suppressive, but positive here.
#   * inflamed readouts: checkpoint/T-cell genes expected to go positive.
# Tuple fields: gene, response-behavior group, compartment, short note.
GENES = [
    ("ARG1", "negative candidate", "cytosolic/intracellular",
     "arginase / myeloid tolerance"),
    ("NKD1", "negative candidate", "cytosolic/intracellular", "Wnt target"),
    ("WNT11", "negative candidate", "secreted/soluble",
     "non-canonical Wnt ligand"),
    ("AXIN2", "negative candidate", "cytosolic/intracellular", "Wnt target"),
    ("ARG2", "negative candidate", "cytosolic/intracellular",
     "arginase family"),
    ("HHLA2", "negative candidate", "surface", "checkpoint-like"),
    ("FOLR1", "negative candidate", "surface", "gyn-tolerance marker"),
    ("TGFB1", "secreted suppressor contrast", "secreted/soluble",
     "TGF-beta ligand"),
    ("WNT5A", "secreted suppressor contrast", "secreted/soluble",
     "non-canonical Wnt ligand"),
    ("IL10", "secreted suppressor contrast", "secreted/soluble",
     "secreted cytokine"),
    ("LGALS9", "secreted suppressor contrast", "secreted/soluble",
     "galectin-9"),
    ("TIGIT", "inflamed checkpoint/readout", "surface", "T-cell checkpoint"),
    ("ICOS", "inflamed checkpoint/readout", "surface", "T-cell activation"),
    ("CTLA4", "inflamed checkpoint/readout", "surface", "T-cell checkpoint"),
    ("IDO1", "inflamed checkpoint/readout", "cytosolic/intracellular",
     "IFN-induced enzyme"),
    ("HAVCR2", "inflamed checkpoint/readout", "surface", "TIM-3"),
    ("LAG3", "inflamed checkpoint/readout", "surface", "T-cell checkpoint"),
    ("PDCD1", "inflamed checkpoint/readout", "surface", "PD-1"),
    ("CD274", "inflamed checkpoint/readout", "surface", "PD-L1"),
    ("PDCD1LG2", "inflamed checkpoint/readout", "surface", "PD-L2"),
]

COMPARTMENT_COLORS = {
    "secreted/soluble": {
        "apd1": "#13796f",
        "ici": "#74cfc3",
    },
    "surface": {
        "apd1": "#5f3495",
        "ici": "#b895df",
    },
    "cytosolic/intracellular": {
        "apd1": "#7d421f",
        "ici": "#d1905a",
    },
}


def _orr_maps() -> tuple[dict[str, float], dict[str, float]]:
    """Strict anti-PD-1 and broad ICI response maps."""
    ici = apd1_map()
    rdf = gsc.cancer_apd1_response_df()
    drop = set(rdf.loc[rdf["drug_target"].isin(["PD-L1", "PD-1+CTLA-4"]),
                       "cancer_code"].astype(str))
    strict = {c: v for c, v in ici.items() if c not in drop}
    return strict, ici


def _gene_series(mat: pd.DataFrame, gene: str) -> tuple[pd.Series | None, str | None]:
    """Log10(TPM+1) cohort series, with proteoform folding."""
    for folded in fold([gene]):
        if folded in mat.columns:
            return mat[folded], folded
    return None, None


def _rho(series: pd.Series, orr_map: dict[str, float]) -> tuple[float, float, int]:
    orr = pd.Series(orr_map, dtype=float)
    df = pd.DataFrame({"expression": series, "orr": orr}).dropna()
    df = df[np.isfinite(df["expression"]) & np.isfinite(df["orr"])]
    if len(df) < 4 or df["expression"].nunique() < 2:
        return np.nan, np.nan, len(df)
    stat = spearmanr(df["expression"], df["orr"])
    return float(stat.statistic), float(stat.pvalue), len(df)


def _score_table() -> pd.DataFrame:
    strict, ici = _orr_maps()
    mat = cohort_gene_matrix(list(ici))
    mat = mat.loc[[c for c in mat.index if c in ici]]

    rows = []
    for gene, group, compartment, note in GENES:
        series, matrix_col = _gene_series(mat, gene)
        if series is None:
            continue
        rho_strict, p_strict, n_strict = _rho(series, strict)
        rho_broad, p_broad, n_broad = _rho(series, ici)
        rows.append({
            "gene": gene,
            "display": display_label(gene),
            "matrix_col": matrix_col,
            "group": group,
            "compartment": compartment,
            "note": note,
            "rho_apd1": rho_strict,
            "p_apd1": p_strict,
            "n_apd1": n_strict,
            "rho_ici": rho_broad,
            "p_ici": p_broad,
            "n_ici": n_broad,
            "median_log10_tpm": float(series.dropna().median()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["rho_apd1", "rho_ici"], ascending=[True, True])


def _plot(df: pd.DataFrame, figdir: Path) -> None:
    order = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(order))
    h = 0.36
    fig, ax = plt.subplots(figsize=(10.2, max(6.2, 0.44 * len(order))))

    apd1_colors = [COMPARTMENT_COLORS[c]["apd1"] for c in order["compartment"]]
    ici_colors = [COMPARTMENT_COLORS[c]["ici"] for c in order["compartment"]]
    ax.barh(y + h / 2, order["rho_apd1"], height=h, color=apd1_colors)
    ax.barh(y - h / 2, order["rho_ici"], height=h, color=ici_colors)
    ax.axvline(0, color="0.25", lw=0.9)
    ax.set_yticks(y)
    labels = [f"{row.display}  ({row.note})" for row in order.itertuples()]
    ax.set_yticklabels(labels, fontsize=8.3)
    ax.set_xlabel("Spearman rho vs objective response rate")
    ax.set_title(
        "Candidate inhibitor genes vs anti-PD-1 / broad ICI response\n"
        "bar hue = protein compartment; darker shade = anti-PD-1, "
        "lighter shade = broad ICI",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.28)

    x_min = min(-0.34, float(np.nanmin(order[["rho_apd1", "rho_ici"]].to_numpy())) - 0.08)
    x_max = max(0.42, float(np.nanmax(order[["rho_apd1", "rho_ici"]].to_numpy())) + 0.08)
    ax.set_xlim(x_min, x_max)

    legend_handles = [
        Patch(facecolor=shades[axis], label=f"{compartment} · {label}")
        for compartment, shades in COMPARTMENT_COLORS.items()
        for axis, label in (
            ("apd1", "anti-PD-1"),
            ("ici", "broad ICI"),
        )
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=2,
              fontsize=7.5, frameon=True, columnspacing=1.0,
              handlelength=1.2)
    fig.tight_layout()
    fig.savefig(figdir / "inhibitor_candidates_vs_ici.png", dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT)

    df = _score_table()
    df.to_csv(figdir / "inhibitor_candidates_vs_ici.csv", index=False)
    _plot(df, figdir)
    print(f"wrote inhibitor_candidates_vs_ici.png + .csv -> {figdir}", flush=True)
    print(df[["gene", "rho_apd1", "rho_ici", "group"]].round(3).to_string(index=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
