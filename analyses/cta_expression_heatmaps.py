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
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

import pirlygenes.expression.accessors as accessors
import pirlygenes.gene_sets_cancer as gsc

N_CANCER_TYPES = 30
N_CTAS = 30
MIN_SAMPLES = 10        # a cohort is eligible if SOME source has at least this many
MIN_DISPLAY_SAMPLES = 5  # but we only display a source with at least this many
HIGH_TPM = 30.0   # "highly expressed" threshold for breadth metrics
TPM_FLOOR = 0.01  # LogNorm needs strictly positive values
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# (value column, file slug, axis label)
STATS = [
    ("q1", "q1", "Q1 (25th pct) TPM"),
    ("expression", "median", "Median TPM"),
    ("q3", "q3", "Q3 (75th pct) TPM"),
]

# "Coarse" scale: a continuous dark→hot colormap (low = dark, high = hot)
# whose colour nodes are anchored on a log axis at the thresholds that match
# the clinical intuition — <10 TPM effectively useless, ~30 actionable, >100
# very interesting. Anchoring (rather than discrete bands) keeps the gradient
# smooth while still putting the perceptual ramp where it carries meaning.
COARSE_VMIN, COARSE_VMAX = 0.01, 1000.0
COARSE_ANCHORS = [0.01, 1, 10, 30, 100, 300, 1000]
COARSE_NODE_COLORS = [
    "#000004",  # 0.01  black (silent)
    "#1b0c41",  # 1     deep indigo
    "#4a0c6b",  # 10    purple (useless below here)
    "#a52c60",  # 30    magenta (actionable)
    "#ed6925",  # 100   orange (very interesting above here)
    "#fbb61a",  # 300   amber
    "#fcffa4",  # 1000  pale yellow (exceptional)
]

# (metric key, human label)
COHORT_METRICS = [
    ("size", "cohorts by sample size"),
    ("maxcta", "cohorts by max CTA TPM"),
    ("breadth", f"cohorts by # CTAs >{HIGH_TPM:g} TPM"),
]


def _representative_source(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one ``source_cohort`` per ``cancer_code``: the most gene-rich
    source the cohort has.

    Why most-genes rather than most-samples: several cohorts carry both a
    large legacy microarray (e.g. SARC_WDLPS GSE30929, n=52, ~13.6k genes)
    and a smaller RNA-seq set (Treehouse, n=5, ~34.5k genes). The microarray
    leaves white cells for any gene it never probed (SPATA4, DPPA5, …) *and*
    its TPM-proxy isn't on the same scale as RNA-seq TPM. Preferring the
    most-gene-rich source therefore fills those gaps and keeps every
    displayed cell on one comparable scale.

    Eligibility vs. display are separate gates: a cohort qualifies if *some*
    source has >= MIN_SAMPLES, but we only ever display a source with
    >= MIN_DISPLAY_SAMPLES (so we don't show an n=2 median). For a cohort
    whose gene-rich source is too small, this falls back to its next
    most-gene-rich source that clears the display floor.
    """
    meta = (
        df.groupby(["cancer_code", "source_cohort"], as_index=False)
        .agg(
            n_samples=("n_samples", "first"),
            n_genes=("Ensembl_Gene_ID", "nunique"),
        )
    )
    eligible = set(meta.loc[meta["n_samples"] >= MIN_SAMPLES, "cancer_code"])
    displayable = meta[
        meta["cancer_code"].isin(eligible)
        & (meta["n_samples"] >= MIN_DISPLAY_SAMPLES)
    ]
    return (
        displayable.sort_values(["n_genes", "n_samples"], ascending=False)
        .drop_duplicates("cancer_code", keep="first")
        .reset_index(drop=True)
    )


def representative_cohorts(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One representative source per code, then the top N by ``metric`` —
    sample size, max CTA TPM, or # CTAs > HIGH_TPM — with a ``metric_score``
    column (rows are sorted by it). See :func:`_representative_source` for how
    the per-code source is chosen."""
    rep = _representative_source(df)

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


def _select_max_coverage_ctas(mat: pd.DataFrame, k: int) -> list:
    """Choose <=k CTA columns so that (1) every cohort contributes its single
    best CTA (guaranteed representation — no cancer is left without a column),
    and (2) the chosen set maximizes the number of cohorts that *highly*
    express at least one chosen CTA (greedy max-coverage over the HIGH_TPM
    threshold). Leftover budget is filled by peak expression.

    This replaces the old "top-k by global peak", which let a handful of
    pan-cancer high-expressers (MAGEA cluster, PRAME, CTAG1B) eat the column
    budget and hid the signature antigen of any cancer whose best CTA wasn't
    globally loud.
    """
    chosen: list = []
    seen: set = set()
    # 1. representation: each cohort's argmax CTA (even if below HIGH_TPM)
    for cohort in mat.index:
        row = mat.loc[cohort]
        if not row.notna().any():
            continue
        best = row.idxmax()
        if best not in seen:
            seen.add(best)
            chosen.append(best)
        if len(chosen) >= k:
            return chosen[:k]
    # 2. greedy max-coverage of cohorts with >=1 highly-expressed chosen CTA
    hi = mat > HIGH_TPM
    cohort_sets = {cta: set(hi.index[hi[cta]]) for cta in mat.columns}
    covered: set = set()
    for cta in chosen:
        covered |= cohort_sets.get(cta, set())
    while len(chosen) < k:
        best_cta, best_gain = None, 0
        for cta in mat.columns:
            if cta in seen:
                continue
            gain = len(cohort_sets[cta] - covered)
            if gain > best_gain:
                best_gain, best_cta = gain, cta
        if best_cta is None or best_gain == 0:
            break
        seen.add(best_cta)
        chosen.append(best_cta)
        covered |= cohort_sets[best_cta]
    # 3. fill any remaining budget by peak expression (use the full column set)
    if len(chosen) < k:
        for cta in mat.max(axis=0).sort_values(ascending=False).index:
            if cta not in seen:
                seen.add(cta)
                chosen.append(cta)
            if len(chosen) >= k:
                break
    return chosen[:k]


def order_and_trim(mat: pd.DataFrame, row_order: list) -> pd.DataFrame:
    """Coverage-maximizing CTA columns (every cohort contributes >=1; the set
    maximizes # cohorts highly expressing a chosen CTA — see
    :func:`_select_max_coverage_ctas`); columns then ordered by breadth
    (# cohorts > HIGH_TPM), tie-broken by peak. Rows in the caller's order."""
    cols = _select_max_coverage_ctas(mat, N_CTAS)
    mat = mat[cols]
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
    ax.set_facecolor("0.7")  # unmeasured (NaN) cells show as gray, not white
    sns.heatmap(
        data, ax=ax, cmap="magma", norm=norm,
        cbar_kws={"label": f"{stat_label}  ({scale_txt} scale)", "shrink": 0.6},
        linewidths=0.3, linecolor="0.9",
    )
    ax.set_title(
        f"CTA expression — {stat_label} ({scale_txt} scale)\n"
        f"top {N_CANCER_TYPES} {metric_label} × {N_CTAS} coverage-maximizing "
        f"CTAs (every cohort contributes its best; set maximizes # cohorts "
        f">{HIGH_TPM:g} TPM); most-gene-rich source per cohort",
        fontsize=12,
    )
    ax.set_xlabel("Cancer-testis antigen")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coarse_cmap() -> LinearSegmentedColormap:
    """Continuous dark→hot colormap with nodes anchored on a log axis at
    COARSE_ANCHORS (so the perceptual ramp lands on the meaningful TPM
    thresholds while staying smooth)."""
    lo, hi = np.log10(COARSE_VMIN), np.log10(COARSE_VMAX)
    stops = [(np.log10(a) - lo) / (hi - lo) for a in COARSE_ANCHORS]
    return LinearSegmentedColormap.from_list(
        "coarse_hot", list(zip(stops, COARSE_NODE_COLORS))
    )


def plot_coarse(mat, stat_label, metric_label, fname) -> None:
    """Heatmap on the continuous 'coarse' scale: low = dark, high = hot, with
    the gradient anchored on the useless/actionable/interesting TPM
    thresholds (see COARSE_ANCHORS)."""
    data = mat.clip(lower=COARSE_VMIN, upper=COARSE_VMAX)
    norm = LogNorm(vmin=COARSE_VMIN, vmax=COARSE_VMAX)
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_facecolor("0.7")  # unmeasured (NaN) cells show as gray, not white
    sns.heatmap(
        data, ax=ax, cmap=_coarse_cmap(), norm=norm,
        cbar_kws={
            "label": f"{stat_label}  (coarse scale)",
            "shrink": 0.6,
            "ticks": COARSE_ANCHORS,
        },
        linewidths=0.3, linecolor="0.9",
    )
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(COARSE_ANCHORS)
    cbar.set_ticklabels([f"{a:g}" for a in COARSE_ANCHORS])
    ax.set_title(
        f"CTA expression — {stat_label} (coarse scale)\n"
        f"top {N_CANCER_TYPES} {metric_label} × top {N_CTAS} CTAs   "
        "(<10 useless · 30 actionable · >100 very interesting)",
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
            if value_col == "expression":  # the median — also the coarse scale
                plot_coarse(
                    mat, stat_label, metric_label, f"cta_{metric}_{slug}_coarse.png"
                )
            mat.to_csv(OUT_DIR / f"cta_{metric}_{slug}.csv", index_label="cohort")
        print(f"  wrote 7 PNGs + 3 CSVs for [{metric}]")

    write_summary_md(matrices, OUT_DIR / "cta_expression_summary.md")
    n_png = len(COHORT_METRICS) * (len(STATS) * 2 + 1)  # +1 actionable per metric
    print(f"wrote {n_png} PNGs total to {OUT_DIR}")


if __name__ == "__main__":
    main()
