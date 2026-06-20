"""Candidate immunosuppressive / inhibitor genes vs ICI response.

This is a compact companion to ``suppressor_genes_vs_apd1.py``. Instead of one
scatter per gene, it summarizes the WNT11-like negative candidates and the
checkpoint/readout contrast genes in a single Spearman-rho bar plot.

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
from matplotlib.lines import Line2D  # noqa: E402

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
GENES = [
    ("ARG1", "negative candidate", "arginase / myeloid tolerance"),
    ("NKD1", "negative candidate", "Wnt target"),
    ("WNT11", "negative candidate", "non-canonical Wnt ligand"),
    ("AXIN2", "negative candidate", "Wnt target"),
    ("ARG2", "negative candidate", "arginase family"),
    ("HHLA2", "negative candidate", "checkpoint-like"),
    ("FOLR1", "negative candidate", "gyn-tolerance marker"),
    ("TGFB1", "secreted suppressor contrast", "TGF-beta ligand"),
    ("WNT5A", "secreted suppressor contrast", "non-canonical Wnt ligand"),
    ("IL10", "secreted suppressor contrast", "secreted cytokine"),
    ("LGALS9", "secreted suppressor contrast", "galectin-9"),
    ("TIGIT", "inflamed checkpoint/readout", "T-cell checkpoint"),
    ("ICOS", "inflamed checkpoint/readout", "T-cell activation"),
    ("CTLA4", "inflamed checkpoint/readout", "T-cell checkpoint"),
    ("IDO1", "inflamed checkpoint/readout", "IFN-induced enzyme"),
    ("HAVCR2", "inflamed checkpoint/readout", "TIM-3"),
    ("LAG3", "inflamed checkpoint/readout", "T-cell checkpoint"),
    ("PDCD1", "inflamed checkpoint/readout", "PD-1"),
    ("CD274", "inflamed checkpoint/readout", "PD-L1"),
    ("PDCD1LG2", "inflamed checkpoint/readout", "PD-L2"),
]

GROUP_COLOR = {
    "negative candidate": "#a83232",
    "secreted suppressor contrast": "#8c6d31",
    "inflamed checkpoint/readout": "#3f7f5f",
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
    for gene, group, note in GENES:
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

    ax.barh(y + h / 2, order["rho_apd1"], height=h, color="#1b6ca8",
            label="anti-PD-1 monotherapy ORR")
    ax.barh(y - h / 2, order["rho_ici"], height=h, color="#f6a21e",
            label="broad ICI ORR")
    ax.axvline(0, color="0.25", lw=0.9)
    ax.set_yticks(y)
    labels = [f"{row.display}  ({row.note})" for row in order.itertuples()]
    ax.set_yticklabels(labels, fontsize=8.3)
    ax.set_xlabel("Spearman rho vs objective response rate")
    ax.set_title(
        "Candidate inhibitor genes vs anti-PD-1 / broad ICI response\n"
        "negative rho = higher in low-response cancers; positive rho = "
        "inflamed checkpoint/readout behavior",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.28)

    x_min = min(-0.34, float(np.nanmin(order[["rho_apd1", "rho_ici"]].to_numpy())) - 0.08)
    x_max = max(0.42, float(np.nanmax(order[["rho_apd1", "rho_ici"]].to_numpy())) + 0.08)
    ax.set_xlim(x_min, x_max)
    x_marker = x_min + 0.015
    ax.scatter([x_marker] * len(order), y,
               c=[GROUP_COLOR[g] for g in order["group"]],
               marker="s", s=30, zorder=4, clip_on=False)

    axis_handles, axis_labels = ax.get_legend_handles_labels()
    group_handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=7,
               markerfacecolor=color, markeredgecolor=color, label=group)
        for group, color in GROUP_COLOR.items()
    ]
    ax.legend(axis_handles + group_handles, axis_labels + list(GROUP_COLOR),
              loc="upper right", fontsize=8, frameon=True)
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
