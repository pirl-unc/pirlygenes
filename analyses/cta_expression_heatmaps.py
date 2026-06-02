"""Cancer-testis antigen (CTA) expression across the top cancer cohorts.

Produces, for each of three summary statistics (Q1 / median / Q3):

  * a heatmap PNG  (rows = cohorts, cols = CTAs, log-scaled TPM colour bar)
  * a wide CSV     (the exact ordered matrix behind the PNG)

plus two machine-readable rollups across all three:

  * cta_expression_long.csv   — tidy (cohort, CTA, q1, median, q3) for every
                                cell shown in any panel
  * cta_expression_summary.md — markdown table of the standout CTAs

Rows are the top 30 cancer cohorts with n>=10 samples (parent umbrella
codes naturally stand in for subtypes too small to survive the cut).
Columns are the top 30 CTAs by that statistic. Rows are sorted by row
mean, columns by column max.

Run from anywhere (outputs land next to this file):

    python analyses/cta_expression_heatmaps.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

import pirlygenes.expression.accessors as accessors
import pirlygenes.gene_sets_cancer as gsc

N_CANCER_TYPES = 30
N_CTAS = 30
MIN_SAMPLES = 10
TPM_FLOOR = 0.01  # LogNorm needs strictly positive values
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# (value column in cancer_reference_expression, file slug, axis label)
STATS = [
    ("q1", "q1", "Q1 (25th percentile) TPM"),
    ("expression", "median", "Median TPM"),
    ("q3", "q3", "Q3 (75th percentile) TPM"),
]


def representative_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """One row-set per cancer_code: the source_cohort with the most samples,
    keeping only codes with n>=MIN_SAMPLES, then the top N by sample count."""
    cohort_sizes = (
        df[["cancer_code", "source_cohort", "n_samples"]]
        .drop_duplicates()
        .sort_values("n_samples", ascending=False)
    )
    rep = cohort_sizes.drop_duplicates("cancer_code", keep="first")
    rep = rep[rep["n_samples"] >= MIN_SAMPLES]
    return rep.head(N_CANCER_TYPES)


def _parent_name_by_code() -> dict:
    """code → parent display name, for subtype rows whose own name is a bare
    molecular label (e.g. BRCA_LumB's name is just 'Luminal B')."""
    reg = gsc.cancer_type_registry()
    name_by_code = dict(zip(reg["code"], reg["name"]))
    out = {}
    for code, parent in zip(reg["code"], reg["parent_code"]):
        if isinstance(parent, str) and parent in name_by_code:
            out[code] = name_by_code[parent]
    return out


def _row_label(code: str, n_by_code: dict, parent_name: dict) -> str:
    name = gsc.CANCER_TYPE_NAMES.get(code) or code
    parent = parent_name.get(code)
    # Prefix the parent lineage for subtypes whose name doesn't already
    # convey it (so "Luminal B" reads as "Breast … — Luminal B").
    if parent and parent.lower() not in name.lower() and name.lower() not in parent.lower():
        name = f"{parent} — {name}"
    return f"{name} [{code}] (n={int(n_by_code[code])})"


def cta_rows(df: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    """Long CTA rows restricted to the representative cohorts, labelled."""
    cta_ids = set(gsc.CTA_gene_ids())
    keep = set(zip(rep["cancer_code"], rep["source_cohort"]))
    sub = df[df["Ensembl_Gene_ID"].isin(cta_ids)]
    sub = sub[
        [(c, s) in keep for c, s in zip(sub["cancer_code"], sub["source_cohort"])]
    ]
    n_by_code = dict(zip(rep["cancer_code"], rep["n_samples"]))
    parent_name = _parent_name_by_code()
    return sub.assign(
        row=sub["cancer_code"].map(lambda c: _row_label(c, n_by_code, parent_name))
    )


def order_and_trim(mat: pd.DataFrame) -> pd.DataFrame:
    """Top CTAs by column-max; rows sorted by row-mean, cols by col-max."""
    top_cols = mat.max(axis=0).sort_values(ascending=False).head(N_CTAS).index
    mat = mat[top_cols]
    row_order = mat.mean(axis=1).sort_values(ascending=False).index
    col_order = mat.max(axis=0).sort_values(ascending=False).index
    return mat.loc[row_order, col_order]


def plot(mat: pd.DataFrame, label: str, fname: str, *, log_scale: bool = True) -> None:
    if log_scale:
        data = mat.clip(lower=TPM_FLOOR)
        norm = LogNorm(vmin=TPM_FLOOR, vmax=float(np.nanmax(data.to_numpy())))
        scale_txt = "log scale"
    else:
        # Linear: dominated by the few high-expressers (PRAME, XAGE1, …);
        # most cells read near-floor. Complements the log view.
        data = mat
        norm = Normalize(vmin=0.0, vmax=float(np.nanmax(data.to_numpy())))
        scale_txt = "linear scale"
    fig, ax = plt.subplots(figsize=(15, 11))
    sns.heatmap(
        data, ax=ax, cmap="magma", norm=norm,
        cbar_kws={"label": f"{label}  ({scale_txt}, TPM)", "shrink": 0.6},
        linewidths=0.3, linecolor="0.9",
    )
    ax.set_title(
        f"Cancer-testis antigen expression — {label} ({scale_txt})\n"
        f"top {N_CANCER_TYPES} cohorts (n≥{MIN_SAMPLES}) × top {N_CTAS} CTAs",
        fontsize=13,
    )
    ax.set_xlabel("Cancer-testis antigen")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_long_csv(rows: pd.DataFrame, shown_symbols: set, path: Path) -> None:
    """Tidy (cohort, CTA, q1, median, q3) for every displayed cell."""
    long = rows[rows["Symbol"].isin(shown_symbols)].copy()
    long = long.rename(columns={"expression": "median"})
    out = (
        long[[
            "cancer_code", "row", "n_samples", "Symbol", "q1", "median", "q3",
        ]]
        .rename(columns={"row": "cohort", "Symbol": "cta"})
        .sort_values(["cancer_code", "cta"])
    )
    out.to_csv(path, index=False)


def write_summary_md(matrices: dict, path: Path) -> None:
    """Markdown: top CTAs by median column-max, with their peak cohort."""
    med = matrices["median"]
    colmax = med.max(axis=0).sort_values(ascending=False)
    lines = [
        "# CTA expression — top cohorts × top CTAs",
        "",
        f"Top {N_CANCER_TYPES} cancer cohorts (n≥{MIN_SAMPLES}) × top {N_CTAS} "
        "CTAs. Values are median TPM unless noted; see the per-stat CSVs for "
        "Q1/Q3. White cells in the heatmaps are genes **not measured** in that "
        "cohort (NaN), not zero.",
        "",
        "## Standout CTAs (by peak median TPM across these cohorts)",
        "",
        "| CTA | peak median TPM | peak cohort | cohorts measured |",
        "| --- | ---: | --- | ---: |",
    ]
    for sym in colmax.head(15).index:
        col = med[sym]
        peak_label = col.idxmax()
        peak_code = peak_label.split("[")[-1].split("]")[0]
        measured = int(col.notna().sum())
        lines.append(
            f"| {sym} | {col.max():.1f} | {peak_code} | {measured}/{N_CANCER_TYPES} |"
        )
    lines += [
        "",
        "Notes:",
        "- PRAME is the single highest-expressed CTA here (peak in SKCM), "
        "measured in every cohort — the clearest broad target signal.",
        "- The MAGE-A cluster (MAGEA3/A6/A4/A12/…) is the next tier, strongest "
        "in melanoma / lung / HNSC / bladder.",
        "- A column like H1-6 (a testis histone) ranks high by *max* but is "
        "NaN in most cohorts — read it as sparsely-measured, not broadly on.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading cancer reference expression...")
    df = accessors.cancer_reference_expression()
    rep = representative_cohorts(df)
    rows = cta_rows(df, rep)
    print(f"selected {len(rep)} cohorts (n>={MIN_SAMPLES}, top {N_CANCER_TYPES})")

    matrices: dict = {}
    shown_symbols: set = set()
    for value_col, slug, label in STATS:
        full = rows.pivot_table(
            index="row", columns="Symbol", values=value_col, aggfunc="max",
        )
        mat = order_and_trim(full)
        matrices[slug] = mat
        shown_symbols.update(mat.columns)
        plot(mat, label, f"cta_{slug}.png", log_scale=True)
        plot(mat, label, f"cta_{slug}_linear.png", log_scale=False)
        mat.to_csv(OUT_DIR / f"cta_{slug}.csv", index_label="cohort")
        print(f"  wrote cta_{slug}.png + cta_{slug}_linear.png + cta_{slug}.csv "
              f"({mat.shape[0]}×{mat.shape[1]})")

    write_long_csv(rows, shown_symbols, OUT_DIR / "cta_expression_long.csv")
    write_summary_md(matrices, OUT_DIR / "cta_expression_summary.md")
    print("  wrote cta_expression_long.csv + cta_expression_summary.md")
    print(f"outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
