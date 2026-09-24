"""Reproducible HPA cancer profiles for the five publication-added CTA genes.

RNA and IHC are unpaired cohorts; IHC fractions count scored patients, not
stained cells. Native HPA pTPM is kept separate from clean TPM references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from oncoref import hpa_cancer
from oncoref.version import __version__ as oncoref_version
from pirlygenes.figure_export import save_figure

GENES = {
    "INSL4": "ENSG00000120211",
    "GCM1": "ENSG00000137270",
    "CYP19A1": "ENSG00000137869",
    "HTRA4": "ENSG00000169495",
    "KISS1": "ENSG00000170498",
}
SYMBOLS = {value: key for key, value in GENES.items()}


def profile_data():
    """Fetch only through the reference owner; preserve the full observed scope."""
    ids = list(GENES.values())
    rna = hpa_cancer.hpa_cancer_rna_prevalence(ids)
    ihc = hpa_cancer.hpa_cancer_ihc_prevalence(ids)
    comparison = hpa_cancer.hpa_cancer_rna_ihc_comparison(
        ids, cohort="TCGA", threshold=1.0, include_missing_ihc=True
    )
    for frame in (rna, ihc, comparison):
        frame["symbol"] = frame.gene_id.map(SYMBOLS)
    return rna, ihc, comparison


def _matrix(frame, row_key, value, rows):
    """Strict reshape: unknown/missing observations remain NaN, never zero."""
    return frame.pivot(index=row_key, columns="symbol", values=value).reindex(
        index=rows, columns=list(GENES)
    ).astype(float)


def _heatmap(ax, values, annotations, labels, *, title, norm=None, cmap="YlGnBu"):
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#e7e7e7")
    image = ax.imshow(values, aspect="auto", cmap=colors, norm=norm,
                      **({} if norm else {"vmin": 0, "vmax": 1}))
    ax.set_xticks(range(len(GENES)), GENES)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=7)
    ax.set_title(title, pad=30, fontsize=12)
    for (i, j), value in np.ndenumerate(values):
        shade = float(image.norm(value)) if np.isfinite(value) else 0
        ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=8,
                color="white" if shade > .62 else "#222222")
    ax.spines[:].set_visible(False)
    return image


def _fraction_label(value):
    if not np.isfinite(value):
        return "NA"
    if 0 < value < .01:
        return "<1%"
    if .99 < value < 1:
        return ">99%"
    return f"{value:.0%}"


def plot_rna(rna, cohort, out):
    selected = rna[rna.cohort.eq(cohort)]
    metadata = selected[["cancer_code", "cancer_type"]].drop_duplicates().sort_values("cancer_type")
    rows = list(metadata.cancer_code)
    labels = [f"{r.cancer_type} ({r.cancer_code})" for r in metadata.itertuples()]
    mean = _matrix(selected, "cancer_code", "mean_ptpm", rows).to_numpy()
    prevalence = _matrix(selected, "cancer_code", "prevalence_ptpm_ge_1", rows).to_numpy()
    samples = _matrix(selected, "cancer_code", "samples", rows).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, len(rows) * .37 + 2)),
                             gridspec_kw={"width_ratios": [1, 1]}, layout="constrained")
    means = np.array([["NA" if np.isnan(v) else f"{v:.2g}" for v in row] for row in mean])
    # +0.01 permits genuine zeros on the log colour scale; annotations are raw pTPM.
    image = _heatmap(axes[0], mean + .01, means, labels,
                    title="Mean RNA (native pTPM)", norm=LogNorm(.01, 100.01))
    colorbar = fig.colorbar(image, ax=axes[0], shrink=.65, pad=.02)
    colorbar.set_ticks([.01, .11, 1.01, 10.01, 100.01], labels=["0", "0.1", "1", "10", "100"])
    colorbar.set_label("Mean pTPM (log colour scale)")
    annotations = np.empty(prevalence.shape, dtype=object)
    for idx, value in np.ndenumerate(prevalence):
        annotations[idx] = "NA" if np.isnan(value) else f"{_fraction_label(value)}\nn={samples[idx]:.0f}"
    image = _heatmap(axes[1], prevalence, annotations, [""] * len(rows),
                    title="RNA ≥1 pTPM: fraction of measured samples")
    fig.colorbar(image, ax=axes[1], shrink=.65, pad=.02, label="RNA-positive fraction")
    fig.suptitle(f"Five newly retained placental CTA genes · HPA 25.1 {cohort}\n"
                 "Cohorts stay separate; n counts measured RNA samples", fontsize=14)
    path = out / f"placental-rna-{cohort.lower()}.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def plot_ihc(comparison, out):
    rows = sorted(comparison.cancer.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9.5), layout="constrained")
    for ax, field, positive, title in zip(
        axes, ("prevalence_detected", "prevalence_medium_high"),
        ("detected", "medium_high"), ("Any staining (low / medium / high)", "Medium or high staining")
    ):
        values = _matrix(comparison, "cancer", field, rows).to_numpy()
        counts = _matrix(comparison, "cancer", positive, rows).to_numpy()
        totals = _matrix(comparison, "cancer", "total", rows).to_numpy()
        annotations = np.empty(values.shape, dtype=object)
        for idx, value in np.ndenumerate(values):
            annotations[idx] = "NA" if np.isnan(value) else f"{value:.0%}\n{counts[idx]:.0f}/{totals[idx]:.0f}"
        image = _heatmap(ax, values, annotations, rows if ax is axes[0] else [""] * len(rows), title=title)
        fig.colorbar(image, ax=ax, shrink=.65, pad=.02, label="Fraction of scored patients")
    fig.suptitle("Available cancer IHC · HPA 25.1\n"
                 "Cells show positive/scored patients; grey NA means no usable measurement", fontsize=14)
    path = out / "placental-cancer-ihc.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def plot_comparison(comparison, out):
    # Keep RNA-only genes in compatible anatomical groups; missing IHC is never
    # plotted at zero. Separate rows avoid overlapping zero-heavy scatter labels.
    data = comparison[comparison.mapping_status.isin(["single", "pooled"])]
    rows = sorted(data.cancer.unique())
    fig, axes = plt.subplots(1, 5, figsize=(14, 8), sharey=True, layout="constrained")
    y = np.arange(len(rows))
    for ax, gene in zip(axes, GENES):
        subset = data[data.symbol.eq(gene)].set_index("cancer").reindex(rows)
        ax.scatter(subset.rna_prevalence * 100, y - .12, color="#237da1", s=35,
                   marker="o", label="RNA samples ≥1 pTPM")
        ax.scatter(subset.prevalence_detected * 100, y + .12, color="#b86a31", s=35,
                   marker="s", label="IHC-positive patients")
        missing = subset.prevalence_detected.isna().all()
        ax.set_title(gene + ("\nIHC unavailable" if missing else ""))
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Positive fraction (%)")
        ax.set_yticks(y, rows)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Cancer RNA prevalence and IHC staining fraction · HPA 25.1\n"
                 "Unpaired cohorts; 16 compatible groups · glioma and three unmapped groups omitted", fontsize=13)
    path = out / "placental-rna-vs-ihc.png"
    save_figure(fig, path)
    plt.close(fig)
    return path


def render(out_dir):
    """Write four figures plus raw measurements, counts and owner provenance."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rna, ihc, comparison = profile_data()
    for name, table in (("rna", rna), ("ihc", ihc), ("comparison", comparison)):
        table.to_csv(out / f"placental-{name}.csv", index=False)
    hpa_cancer.hpa_cancer_crosswalk().to_csv(out / "hpa-cancer-crosswalk.csv", index=False)
    hpa_cancer.hpa_cancer_rna_cohorts().to_csv(out / "hpa-cancer-rna-cohorts.csv", index=False)
    hpa_cancer.hpa_cancer_assay_limitations().to_csv(out / "hpa-cancer-assay-limitations.csv", index=False)
    provenance = {
        "oncoref_version": oncoref_version, "gene_ids": GENES,
        "sources": hpa_cancer.hpa_cancer_sources(),
        "comparison_status_counts": comparison.comparison_status.value_counts().to_dict(),
        "limitations": ["Unpaired RNA and IHC cohorts.", "IHC fraction counts scored patients, not cells.",
                        "Missing IHC is not zero staining.", "Aggregate IHC lacks antibody-specific reliability.",
                        "RNA uses native HPA pTPM, not clean TPM."],
    }
    (out / "profile-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10}):
        paths = [plot_rna(rna, cohort, out) for cohort in ("TCGA", "validation")]
        paths.extend([plot_ihc(comparison, out), plot_comparison(comparison, out)])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.out):
        print(path)


if __name__ == "__main__":
    main()
