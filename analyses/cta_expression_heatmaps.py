"""Cancer-testis antigen (CTA) expression across the top cancer cohorts.

For each of three cohort orderings × three statistics × two colour scales,
writes a heatmap PNG; plus a wide CSV of each ordered matrix and a markdown
summary.

Cohort orderings (rows = the top 30 cohorts, n>=10, selected + sorted by):
  * size    — sample count
  * maxcta  — max over CTAs of the cohort's median TPM
  * breadth — number of CTAs the cohort expresses above HIGH_TPM

Columns = top 30 CTAs by peak (max across cohorts), ordered by breadth of
high expression (# cohorts > HIGH_TPM), tie-broken by peak.

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
HIGH_TPM = 30.0   # "highly expressed" threshold for breadth metrics
TPM_FLOOR = 0.01  # LogNorm needs strictly positive values
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# (value column, file slug, axis label)
STATS = [
    ("q1", "q1", "Q1 (25th pct) TPM"),
    ("expression", "median", "Median TPM"),
    ("q3", "q3", "Q3 (75th pct) TPM"),
]

# (metric key, human label)
COHORT_METRICS = [
    ("size", "cohorts by sample size"),
    ("maxcta", "cohorts by max CTA TPM"),
    ("breadth", f"cohorts by # CTAs >{HIGH_TPM:g} TPM"),
]


def representative_cohorts(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One representative cohort per code (largest, n>=MIN_SAMPLES), then the
    top N by ``metric`` — sample size, max CTA TPM, or # CTAs > HIGH_TPM —
    with a ``metric_score`` column (rows are sorted by it)."""
    sizes = (
        df[["cancer_code", "source_cohort", "n_samples"]]
        .drop_duplicates()
        .sort_values("n_samples", ascending=False)
    )
    rep = sizes.drop_duplicates("cancer_code", keep="first")
    rep = rep[rep["n_samples"] >= MIN_SAMPLES]

    if metric == "size":
        rep = rep.assign(metric_score=rep["n_samples"].astype(float))
    else:
        cta_ids = set(gsc.CTA_gene_ids())
        keep = set(zip(rep["cancer_code"], rep["source_cohort"]))
        cta = df[df["Ensembl_Gene_ID"].isin(cta_ids)]
        cta = cta[
            [(c, s) in keep for c, s in zip(cta["cancer_code"], cta["source_cohort"])]
        ]
        if metric == "maxcta":
            score = cta.groupby("cancer_code")["expression"].max()
        elif metric == "breadth":
            score = (
                cta.assign(hi=cta["expression"] > HIGH_TPM)
                .groupby("cancer_code")["hi"].sum()
            )
        else:
            raise ValueError(metric)
        rep = rep.assign(metric_score=rep["cancer_code"].map(score).fillna(0.0))

    return rep.sort_values("metric_score", ascending=False).head(N_CANCER_TYPES)


def _parent_name_by_code() -> dict:
    reg = gsc.cancer_type_registry()
    name_by_code = dict(zip(reg["code"], reg["name"]))
    return {
        code: name_by_code[parent]
        for code, parent in zip(reg["code"], reg["parent_code"])
        if isinstance(parent, str) and parent in name_by_code
    }


def _row_label(code: str, n_by_code: dict, parent_name: dict) -> str:
    name = gsc.CANCER_TYPE_NAMES.get(code) or code
    parent = parent_name.get(code)
    if parent and parent.lower() not in name.lower() and name.lower() not in parent.lower():
        name = f"{parent} — {name}"
    return f"{name} [{code}] (n={int(n_by_code[code])})"


def cta_rows(df: pd.DataFrame, rep: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Long CTA rows for the selected cohorts (labelled), plus the row order
    (cohort labels in metric_score order)."""
    cta_ids = set(gsc.CTA_gene_ids())
    keep = set(zip(rep["cancer_code"], rep["source_cohort"]))
    sub = df[df["Ensembl_Gene_ID"].isin(cta_ids)]
    sub = sub[
        [(c, s) in keep for c, s in zip(sub["cancer_code"], sub["source_cohort"])]
    ]
    n_by_code = dict(zip(rep["cancer_code"], rep["n_samples"]))
    parent_name = _parent_name_by_code()
    sub = sub.assign(
        row=sub["cancer_code"].map(lambda c: _row_label(c, n_by_code, parent_name))
    )
    row_order = [_row_label(c, n_by_code, parent_name) for c in rep["cancer_code"]]
    return sub, row_order


def order_and_trim(mat: pd.DataFrame, row_order: list) -> pd.DataFrame:
    """Top CTAs by peak; columns ordered by breadth (# cohorts > HIGH_TPM),
    tie-broken by peak. Rows ordered by the caller's cohort metric."""
    top_cols = mat.max(axis=0).sort_values(ascending=False).head(N_CTAS).index
    mat = mat[top_cols]
    breadth = (mat > HIGH_TPM).sum(axis=0)
    col_order = (
        pd.DataFrame({"breadth": breadth, "peak": mat.max(axis=0)})
        .sort_values(["breadth", "peak"], ascending=False)
        .index
    )
    rows = [r for r in row_order if r in mat.index]
    return mat.loc[rows, col_order]


def plot(mat, stat_label, metric_label, fname, *, log_scale=True) -> None:
    if log_scale:
        data = mat.clip(lower=TPM_FLOOR)
        norm = LogNorm(vmin=TPM_FLOOR, vmax=float(np.nanmax(data.to_numpy())))
        scale_txt = "log"
    else:
        data = mat
        norm = Normalize(vmin=0.0, vmax=float(np.nanmax(data.to_numpy())))
        scale_txt = "linear"
    fig, ax = plt.subplots(figsize=(15, 11))
    sns.heatmap(
        data, ax=ax, cmap="magma", norm=norm,
        cbar_kws={"label": f"{stat_label}  ({scale_txt} scale)", "shrink": 0.6},
        linewidths=0.3, linecolor="0.9",
    )
    ax.set_title(
        f"CTA expression — {stat_label} ({scale_txt} scale)\n"
        f"top {N_CANCER_TYPES} {metric_label} (n≥{MIN_SAMPLES}) × top {N_CTAS} "
        f"CTAs (by breadth >{HIGH_TPM:g})",
        fontsize=12,
    )
    ax.set_xlabel("Cancer-testis antigen")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(matrices: dict, path: Path) -> None:
    """Markdown: top CTAs by breadth (# cohorts > HIGH_TPM) for the
    max-CTA / median panel."""
    med = matrices[("maxcta", "median")]
    breadth = (med > HIGH_TPM).sum(axis=0).sort_values(ascending=False)
    lines = [
        "# CTA expression — top cohorts × top CTAs",
        "",
        f"Cohorts: top {N_CANCER_TYPES} by max CTA TPM (n≥{MIN_SAMPLES}); "
        f"CTAs ordered by breadth (# cohorts with median >{HIGH_TPM:g} TPM). "
        "White heatmap cells = gene not measured in that cohort (NaN), not zero.",
        "",
        f"## Broadest CTAs (# cohorts with median >{HIGH_TPM:g} TPM)",
        "",
        f"| CTA | # cohorts >{HIGH_TPM:g} | peak median TPM | peak cohort |",
        "| --- | ---: | ---: | --- |",
    ]
    for sym in breadth.head(15).index:
        col = med[sym]
        peak_code = col.idxmax().split("[")[-1].split("]")[0]
        lines.append(
            f"| {sym} | {int(breadth[sym])} | {col.max():.1f} | {peak_code} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading cancer reference expression...")
    df = accessors.cancer_reference_expression()

    matrices: dict = {}
    for metric, metric_label in COHORT_METRICS:
        rep = representative_cohorts(df, metric)
        rows, row_order = cta_rows(df, rep)
        print(f"[{metric}] {len(rep)} cohorts ({metric_label})")
        for value_col, slug, stat_label in STATS:
            full = rows.pivot_table(
                index="row", columns="Symbol", values=value_col, aggfunc="max",
            )
            mat = order_and_trim(full, row_order)
            matrices[(metric, slug)] = mat
            plot(mat, stat_label, metric_label, f"cta_{metric}_{slug}.png", log_scale=True)
            plot(mat, stat_label, metric_label, f"cta_{metric}_{slug}_linear.png", log_scale=False)
            mat.to_csv(OUT_DIR / f"cta_{metric}_{slug}.csv", index_label="cohort")
        print(f"  wrote 6 PNGs + 3 CSVs for [{metric}]")

    write_summary_md(matrices, OUT_DIR / "cta_expression_summary.md")
    print(f"wrote {len(COHORT_METRICS)*len(STATS)*2} PNGs total to {OUT_DIR}")


if __name__ == "__main__":
    main()
