# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import seaborn as sns
import matplotlib.pyplot as plt

from adjustText import adjust_text

from .common import _guess_gene_cols
from .gene_sets_cancer import pan_cancer_expression, housekeeping_gene_ids
from .plot_data_helpers import _strip_ensembl_version
from .plot_strip import default_gene_sets


# -------------------- cancer type aliases --------------------

CANCER_TYPE_NAMES = {
    "ACC": "Adrenocortical Carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "BRCA": "Breast Invasive Carcinoma",
    "CESC": "Cervical Squamous Cell Carcinoma",
    "CHOL": "Cholangiocarcinoma",
    "COAD": "Colon Adenocarcinoma",
    "DLBC": "Diffuse Large B-Cell Lymphoma",
    "ESCA": "Esophageal Carcinoma",
    "GBM": "Glioblastoma Multiforme",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney Renal Clear Cell Carcinoma",
    "KIRP": "Kidney Renal Papillary Cell Carcinoma",
    "LAML": "Acute Myeloid Leukemia",
    "LGG": "Brain Lower Grade Glioma",
    "LIHC": "Liver Hepatocellular Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "LUSC": "Lung Squamous Cell Carcinoma",
    "MESO": "Mesothelioma",
    "OV": "Ovarian Serous Cystadenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate Adenocarcinoma",
    "READ": "Rectum Adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach Adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumor",
    "THCA": "Thyroid Carcinoma",
    "THYM": "Thymoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UCS": "Uterine Carcinosarcoma",
    "UVM": "Uveal Melanoma",
}

CANCER_TYPE_ALIASES = {
    "prostate": "PRAD",
    "breast": "BRCA",
    "lung_adeno": "LUAD",
    "lung_squamous": "LUSC",
    "melanoma": "SKCM",
    "skin": "SKCM",
    "colon": "COAD",
    "colorectal": "COAD",
    "rectal": "READ",
    "pancreatic": "PAAD",
    "pancreas": "PAAD",
    "liver": "LIHC",
    "kidney_clear": "KIRC",
    "kidney_papillary": "KIRP",
    "kidney_chromophobe": "KICH",
    "kidney": "KIRC",
    "ovarian": "OV",
    "ovary": "OV",
    "cervical": "CESC",
    "cervix": "CESC",
    "bladder": "BLCA",
    "stomach": "STAD",
    "gastric": "STAD",
    "glioblastoma": "GBM",
    "gbm": "GBM",
    "head_neck": "HNSC",
    "hnscc": "HNSC",
    "thyroid": "THCA",
    "endometrial": "UCEC",
    "uterine": "UCEC",
    "testicular": "TGCT",
    "testis": "TGCT",
    "sarcoma": "SARC",
    "adrenocortical": "ACC",
    "adrenal": "ACC",
    "cholangiocarcinoma": "CHOL",
    "bile_duct": "CHOL",
    "dlbcl": "DLBC",
    "lymphoma": "DLBC",
    "esophageal": "ESCA",
    "esophagus": "ESCA",
    "aml": "LAML",
    "leukemia": "LAML",
    "low_grade_glioma": "LGG",
    "lgg": "LGG",
    "glioma": "LGG",
    "mesothelioma": "MESO",
    "pheochromocytoma": "PCPG",
    "paraganglioma": "PCPG",
    "thymoma": "THYM",
    "uterine_carcinosarcoma": "UCS",
    "uveal_melanoma": "UVM",
}


def resolve_cancer_type(cancer_type):
    """Resolve a cancer type name or alias to a TCGA code.

    Accepts TCGA codes (e.g. ``"PRAD"``), common names (e.g. ``"prostate"``),
    or case-insensitive variants. Returns the TCGA code or raises ValueError.
    """
    if cancer_type is None:
        return None
    key = cancer_type.strip().lower().replace(" ", "_").replace("-", "_")
    # Direct TCGA code match (case-insensitive)
    upper = cancer_type.strip().upper()
    # Check alias map first, then try as TCGA code
    if key in CANCER_TYPE_ALIASES:
        return CANCER_TYPE_ALIASES[key]
    # Check if it's already a valid TCGA code
    ref = pan_cancer_expression()
    fpkm_cols = {c.replace("FPKM_", "") for c in ref.columns if c.startswith("FPKM_")}
    if upper in fpkm_cols:
        return upper
    raise ValueError(
        f"Unknown cancer type '{cancer_type}'. "
        f"Valid codes: {sorted(fpkm_cols)}. "
        f"Aliases: {sorted(CANCER_TYPE_ALIASES.keys())}"
    )


# -------------------- sample vs pan-cancer scatter --------------------


def _prepare_sample_vs_cancer_data(
    df_gene_expr,
    gene_sets,
    cancer_type,
):
    """Shared data prep for sample-vs-cancer scatter plots.

    Returns (plot_df, named_cats, cat_to_color, x_label).
    """
    import pandas as pd
    from .plot_data_helpers import (
        normalize_gene_sets,
        _remap_retired_gene_ids,
        _create_gene_to_category_list_mapping,
    )
    from .gene_names import aliases

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)

    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    cat_to_ids, id_to_name = normalize_gene_sets(gene_sets)
    cat_to_ids, id_to_name = _remap_retired_gene_ids(
        cat_to_ids, id_to_name, df,
        gene_id_col=gene_id_col, gene_name_col=gene_name_col,
        verbose=False,  # already logged during strip plot
    )

    # Internally we still HK-normalize the reference so the cross-dataset
    # scale mismatch (sample TPM totals ~1M; TCGA `FPKM_*` cohort medians
    # total ~250K over the ~20K genes we track) cancels out. For display
    # we then multiply back by the sample's own HK median TPM — so both
    # axes end up on a familiar TPM scale that's correctly rescaled to
    # this sample's sequencing depth.
    ref = pan_cancer_expression(normalize="housekeeping")
    if cancer_type is not None:
        cancer_type = resolve_cancer_type(cancer_type)
        ref_col = f"FPKM_{cancer_type}"
        ref["_ref_hk"] = ref[ref_col].astype(float)
        cancer_label = CANCER_TYPE_NAMES.get(cancer_type, cancer_type)
        cohort_label = f"{cancer_label} cohort ({cancer_type})"
    else:
        fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
        ref["_ref_hk"] = ref[fpkm_cols].astype(float).mean(axis=1)
        cohort_label = "Mean across 33 TCGA cancer cohorts"

    ref_lookup = dict(zip(
        ref["Ensembl_Gene_ID"].map(_strip_ensembl_version),
        ref["_ref_hk"],
    ))

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    # Sample's own HK-median TPM. Used as the rescaling factor that
    # brings the HK-normalized cohort reference back onto this sample's
    # TPM scale.
    hk_ids = housekeeping_gene_ids()
    hk_mask = df[gene_id_col].isin(hk_ids)
    hk_median_tpm = df.loc[hk_mask, tpm_col].astype(float).median()
    if not (hk_median_tpm > 0):
        hk_median_tpm = 1.0

    gene_to_category = _create_gene_to_category_list_mapping(cat_to_ids)
    name_from_df = dict(zip(df[gene_id_col].astype(str), df[gene_name_col].astype(str)))

    rows = []
    for _, row in df.iterrows():
        gid = str(row[gene_id_col])
        tpm = float(row[tpm_col])
        ref_hk = ref_lookup.get(gid)
        if ref_hk is None:
            continue
        # Both axes on TPM scale. Reference rescaled via sample's HK median.
        sample_val = tpm
        cohort_val = float(ref_hk) * hk_median_tpm
        cats = gene_to_category.get(gid, ["other"])
        name = name_from_df.get(gid) or id_to_name.get(gid) or gid
        display_name = aliases.get(name, name)
        for cat in cats:
            rows.append((gid, display_name, cat, sample_val, cohort_val))

    plot_df = pd.DataFrame(rows, columns=[
        "gene_id", "gene_name", "category", "sample_hk", "cohort_hk",
    ])
    # Column names kept (`sample_hk` / `cohort_hk`) for wire-compatibility
    # with downstream panel code; the stored values are TPM, not % of HK.
    plot_df["sample_log"] = plot_df["sample_hk"] + 0.001
    plot_df["cohort_log"] = plot_df["cohort_hk"] + 0.001
    plot_df["enrichment"] = (plot_df["sample_hk"] + 0.001) / (plot_df["cohort_hk"] + 0.001)

    named_cats = list(cat_to_ids.keys())
    palette = sns.color_palette("tab10", len(named_cats))
    cat_to_color = dict(zip(named_cats, palette))

    sample_label = "Sample TPM"
    cohort_axis_label = f"{cohort_label} (TPM)"

    return plot_df, named_cats, cat_to_color, sample_label, cohort_axis_label


def _draw_scatter_panel(
    ax, plot_df, highlight_cat, color,
    num_labels=10, adjust_args=None,
):
    """Draw a single scatter panel: sample (x) vs cohort (y), one category highlighted."""
    # Background: all genes faded
    bg = plot_df[plot_df.category == "other"]
    if len(bg):
        ax.scatter(
            bg.sample_log, bg.cohort_log,
            c=[(0.88, 0.88, 0.88)], alpha=0.12, s=6, zorder=1,
        )

    # Also fade other named categories
    other_named = plot_df[
        (plot_df.category != "other") & (plot_df.category != highlight_cat)
    ]
    if len(other_named):
        ax.scatter(
            other_named.sample_log, other_named.cohort_log,
            c=[(0.78, 0.78, 0.78)], alpha=0.18, s=8, zorder=1,
        )

    # Highlight category — mark sample-enriched genes (enrichment > 5x) with edge ring
    hi = plot_df[plot_df.category == highlight_cat]
    if len(hi):
        enriched = hi[hi.enrichment > 5]
        normal = hi[hi.enrichment <= 5]
        if len(normal):
            ax.scatter(
                normal.sample_log, normal.cohort_log,
                color=color, alpha=0.8, s=30, zorder=3, edgecolors="none",
            )
        if len(enriched):
            ax.scatter(
                enriched.sample_log, enriched.cohort_log,
                color=color, alpha=0.9, s=50, zorder=4,
                edgecolors="black", linewidths=0.8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(highlight_cat.replace("_", " "), fontsize=11, fontweight="bold")

    # Diagonal: y = x (genes on this line have same expression in sample and cohort)
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    if lims[1] > lims[0]:
        ax.plot(lims, lims, ":", color="#bbbbbb", linewidth=0.8, alpha=0.5, zorder=0)

    # Labels: top N by enrichment (sample-enriched), with fallback to top by sample expression
    if len(hi) and num_labels > 0:
        top_enriched = hi.nlargest(num_labels, "enrichment")
        texts = []
        for _, row in top_enriched.iterrows():
            texts.append(ax.text(
                row.sample_log, row.cohort_log, row.gene_name,
                fontsize=8, alpha=0.9, ha="left", va="bottom",
            ))
        if adjust_args is not None:
            adjust_text(texts, ax=ax, **adjust_args)


def plot_sample_vs_cancer(
    df_gene_expr,
    gene_sets=default_gene_sets,
    cancer_type=None,
    save_to_filename=None,
    save_dpi=300,
    num_labels_per_category=10,
    always_label_genes=None,
    figsize=(10, 8),
    adjust_args=dict(
        expand=(1.3, 1.8),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.3),
        min_arrow_len=7,
        expand_axes=True,
        ensure_inside_axes=False,
    ),
):
    """Scatter plots: sample TPM vs pan-cancer reference expression.

    Generates one figure per gene-set category — the category's genes are
    highlighted and labeled while everything else is faint gray.

    Output is controlled by ``save_to_filename``:

    * ``"dir/prefix.pdf"`` — multi-page PDF (one page per category) **plus**
      individual PNGs in ``dir/prefix/`` (one per category).
    * ``"dir/prefix.png"`` — individual PNGs in ``dir/prefix/`` only.
    * ``None`` — returns figures without saving.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data with gene ID, gene name, and TPM columns.
    gene_sets : dict
        Category name -> list of gene names/IDs.
    cancer_type : str or None
        TCGA cancer type code (e.g. ``"LUAD"``, ``"SKCM"``). If None,
        uses mean across all cancer types.
    save_to_filename : str or None
        Output path. Extension determines format (see above).
    save_dpi : int
        DPI for saved figures.
    num_labels_per_category : int
        Number of genes to label per category (top by sample TPM).
    always_label_genes : list of str or None
        Gene names/IDs to always label regardless of expression.
    figsize : tuple
        Figure size per category (width, height).
    """
    from pathlib import Path

    plot_df, named_cats, cat_to_color, sample_label, cohort_label = \
        _prepare_sample_vs_cancer_data(df_gene_expr, gene_sets, cancer_type)

    # Generate one figure per category
    figures = {}
    for cat in named_cats:
        fig, ax = plt.subplots(figsize=figsize)
        _draw_scatter_panel(
            ax, plot_df, cat, cat_to_color[cat],
            num_labels=num_labels_per_category,
            adjust_args=adjust_args,
        )
        ax.set_xlabel(sample_label, fontsize=10)
        ax.set_ylabel(cohort_label, fontsize=10)
        fig.tight_layout()
        figures[cat] = fig

    # Save
    if save_to_filename:
        out = Path(save_to_filename)
        stem = out.stem
        png_dir = out.parent / stem
        png_dir.mkdir(parents=True, exist_ok=True)

        # Individual PNGs
        saved_pngs = []
        for cat, fig in figures.items():
            png_path = png_dir / f"{cat}.png"
            fig.savefig(png_path, dpi=save_dpi, bbox_inches="tight")
            saved_pngs.append(png_path)
        print(f"Saved {len(saved_pngs)} PNGs to {png_dir}/")

        # Multi-page PDF if requested
        if out.suffix.lower() == ".pdf":
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(out) as pdf:
                for cat in named_cats:
                    pdf.savefig(figures[cat], bbox_inches="tight")
            print(f"Saved multi-page PDF to {out}")

        for fig in figures.values():
            plt.close(fig)

    return figures
