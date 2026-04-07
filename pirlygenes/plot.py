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

try:
    from adjustText import adjust_text
except ImportError:
    def adjust_text(*args, **kwargs):
        return args[0] if args else []

from .plot_data_helpers import prepare_gene_expr_df
from .gene_ids import find_canonical_gene_ids_and_names
from .gene_sets_cancer import (
    CTA_gene_id_to_name,
    housekeeping_gene_ids,
    cancer_surfaceome_gene_id_to_name,
    pan_cancer_expression,
)
from .load_dataset import get_data


def _load_immune_gene_sets():
    """Load immune gene sets as {category: {ensembl_id: symbol}} from CSV."""
    df = get_data("immune-gene-sets")
    result = {}
    for cat, group in df.groupby("Category"):
        result[cat] = dict(zip(group["Ensembl_Gene_ID"], group["Symbol"]))
    return result


# ------------------------ helpers ------------------------


def _guess_gene_cols(df):
    """Best-effort guess for gene ID and name columns in df_gene_expr."""
    id_candidates = ["gene_id", "ensembl_gene_id", "canonical_gene_id", "GeneID"]
    name_candidates = [
        "gene_display_name",
        "gene_name",
        "canonical_gene_name",
        "gene_symbol",
        "symbol",
        "GeneName",
    ]
    gene_id_col = next((c for c in id_candidates if c in df.columns), None)
    gene_name_col = next((c for c in name_candidates if c in df.columns), None)
    if gene_id_col is None:
        raise KeyError(
            "Could not find a gene ID column in df_gene_expr. "
            "Tried: %s" % (id_candidates,)
        )
    if gene_name_col is None:
        raise KeyError(
            "Could not find a gene name column in df_gene_expr. "
            "Tried: %s" % (name_candidates,)
        )
    return gene_id_col, gene_name_col


def pick_genes_to_annotate(
    df,
    num_per_category=10,
    verbose=False,
    category_col="category",
    tpm_col="TPM",
    gene_id_col="gene_id",
    gene_name_col="gene_display_name",
):
    """
    Returns a set of GENE IDs to annotate (top N by TPM per category).
    Note: we store IDs for stable matching; labels come from df['gene_display_name'] later.
    """
    genes_to_annotate = set()

    for c in [gene_id_col, gene_name_col, tpm_col, category_col]:
        assert c in df.columns, "%s not in %s" % (c, df.columns)

    for cat, df_cat in df.groupby(category_col, observed=True):
        df_cat_sorted = df_cat.sort_values(tpm_col, ascending=False)
        top = df_cat_sorted.head(num_per_category)
        if verbose:
            cols = [gene_id_col, gene_name_col, tpm_col, category_col]
            print(cat)
            print(top[cols])
        genes_to_annotate.update(top[gene_id_col].tolist())
    return genes_to_annotate


def _normalize_label_token(token):
    return str(token).strip().upper()


def resolve_always_label_gene_ids(
    df,
    always_label_genes,
    gene_id_col="gene_id",
    gene_name_col="gene_display_name",
):
    if not always_label_genes:
        return set()
    wanted = {_normalize_label_token(tok) for tok in always_label_genes}
    forced = set()
    for _, row in df[[gene_id_col, gene_name_col]].drop_duplicates().iterrows():
        gene_id = str(row[gene_id_col])
        name = str(row[gene_name_col])
        if _normalize_label_token(gene_id) in wanted:
            forced.add(gene_id)
        elif _normalize_label_token(name) in wanted:
            forced.add(gene_id)

    # Also resolve user-provided gene names/aliases to canonical Ensembl IDs.
    resolved_ids, _ = find_canonical_gene_ids_and_names(list(always_label_genes))
    expr_gene_ids = set(df[gene_id_col].astype(str))
    for gid in resolved_ids:
        if gid:
            gid_str = str(gid).split(".", 1)[0]
            if gid_str in expr_gene_ids:
                forced.add(gid_str)
    return forced


# ------------------------ defaults ------------------------

def _build_default_gene_sets():
    """Build default gene sets using pre-resolved Ensembl IDs (no pyensembl lookup)."""
    sets = _load_immune_gene_sets()  # APM, MHC1, TLR, Growth_receptors, Oncogenes, Immune_checkpoints
    sets["CTAs"] = CTA_gene_id_to_name()
    sets["Cancer_surfaceome"] = cancer_surfaceome_gene_id_to_name()
    return sets


default_gene_sets = _build_default_gene_sets()

# ------------------------ main plot ------------------------


def plot_gene_expression(
    df_gene_expr,
    gene_sets=default_gene_sets,
    save_to_filename=None,
    save_dpi=300,
    plot_height=12.0,
    plot_aspect=1.4,
    num_labels_per_category=10,
    always_label_genes=None,
    adjust_args=dict(
        expand=(1.05, 1.3),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.3),
        min_arrow_len=7,
        expand_axes=True,
        ensure_inside_axes=False,
    ),
):
    # Pick the correct ID/name columns from the incoming DF
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)

    # - join by IDs, label with names, include 'other' as a category and place it first.
    df_gene_expr_annot = prepare_gene_expr_df(
        df_gene_expr,
        gene_sets=gene_sets,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,  # ok if None; helper will resolve names
        other_category_name="other",  # <- ensure lowercase to match your filter
        place_other_first=True,  # <- 'other' left-most
    )

    # Just in case the "prepared" DF changed any column names, get them again
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr_annot)

    # Choose which IDs to annotate (top N per category)
    genes_to_annotate = pick_genes_to_annotate(
        df_gene_expr_annot,
        num_per_category=num_labels_per_category,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,
    )
    forced_gene_ids = resolve_always_label_gene_ids(
        df_gene_expr_annot,
        always_label_genes=always_label_genes,
        gene_id_col=gene_id_col,
        gene_name_col=gene_name_col,
    )
    genes_to_annotate.update(forced_gene_ids)

    # Sanity checks
    for col in ("category", "TPM", gene_id_col, gene_name_col):
        assert col in df_gene_expr_annot.columns, df_gene_expr_annot.columns

    # Plot
    other_name = "other"
    cat = sns.catplot(
        data=df_gene_expr_annot,
        x="category",
        y="TPM",
        jitter=0.01,
        height=plot_height,
        aspect=plot_aspect,
        alpha=0.5,
        hue="category",
    )
    plt.yscale("log")

    # Fade "other" category points so named categories stand out
    ax = cat.ax
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) == 0:
            continue
        # Check if this collection corresponds to "other" by x-position
        cats = list(df_gene_expr_annot["category"].cat.categories)
        if other_name in cats:
            other_idx = cats.index(other_name)
            # PathCollection x-coords are jittered around integer category index
            x_coords = offsets[:, 0]
            is_other = (x_coords > other_idx - 0.5) & (x_coords < other_idx + 0.5)
            if is_other.any() and is_other.all():
                coll.set_alpha(0.12)

    # Highlight housekeeping genes in "other" with a distinct color and labels
    _hk_ids = housekeeping_gene_ids()
    hk_in_other = df_gene_expr_annot[
        (df_gene_expr_annot["category"] == other_name)
        & (df_gene_expr_annot[gene_id_col].isin(_hk_ids))
    ]
    if len(hk_in_other) > 0:
        cats = list(df_gene_expr_annot["category"].cat.categories)
        other_idx = cats.index(other_name) if other_name in cats else 0
        import numpy as _np
        hk_x = other_idx + _np.random.default_rng(42).uniform(-0.01, 0.01, len(hk_in_other))
        ax.scatter(
            hk_x, hk_in_other["TPM"].values,
            color="#e74c3c", alpha=0.7, s=18, zorder=5, label="Housekeeping",
            edgecolors="none",
        )
        # Label top 5, bottom 5, and always ACTB — fed through adjust_text
        top5 = hk_in_other.nlargest(5, "TPM")
        bot5 = hk_in_other.nsmallest(5, "TPM")
        label_ids = set(top5[gene_id_col]) | set(bot5[gene_id_col])
        # Always include ACTB
        actb_rows = hk_in_other[hk_in_other[gene_name_col].str.upper() == "ACTB"]
        label_ids.update(actb_rows[gene_id_col])

        hk_texts = []
        for (_, row), xi in zip(hk_in_other.iterrows(), hk_x):
            if row[gene_id_col] in label_ids:
                hk_texts.append(
                    ax.text(
                        xi, row["TPM"], row[gene_name_col],
                        fontsize=9, color="#c0392b", alpha=0.85,
                        ha="right", va="top",
                    )
                )
        adjust_text(
            hk_texts,
            **{**adjust_args, "expand": (1.2, 1.6)},
        )

    # Reference lines at key TPM thresholds
    for tpm_thresh in (100, 1000):
        ax.axhline(
            y=tpm_thresh,
            color="#cccccc",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
            zorder=1,
        )

    # Annotate with display names, never raw ENSG; skip the 'other' column
    # Map categories to integer x-positions for adjust_text compatibility
    cat_col = df_gene_expr_annot["category"]
    if hasattr(cat_col, "cat"):
        cat_order = list(cat_col.cat.categories)
    else:
        cat_order = list(dict.fromkeys(cat_col))
    cat_to_x = {c: i for i, c in enumerate(cat_order)}

    texts = []
    forced_mask = df_gene_expr_annot[gene_id_col].isin(forced_gene_ids)
    mask = (
        ((df_gene_expr_annot.TPM > 0.1) | forced_mask)
        & (df_gene_expr_annot.category != "other")
        & (df_gene_expr_annot[gene_id_col].isin(genes_to_annotate))
    )
    for _, row in df_gene_expr_annot[mask].iterrows():
        # Use the friendly display name
        label = row[gene_name_col]
        texts.append(
            cat.ax.text(
                cat_to_x[row.category],
                row.TPM,
                label,
                color="black",
                alpha=0.8,
                ha="right",
                va="top",
            )
        )

    adjust_text(texts, **adjust_args)

    if save_to_filename:
        cat.figure.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")

    return cat


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
        _strip_ensembl_version,
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
    )

    ref = pan_cancer_expression(normalize="housekeeping")
    if cancer_type is not None:
        cancer_type = resolve_cancer_type(cancer_type)
        ref_col = f"FPKM_{cancer_type}"
        ref["_ref_value"] = ref[ref_col].astype(float)
        cancer_label = CANCER_TYPE_NAMES.get(cancer_type, cancer_type)
        cohort_label = f"{cancer_label} cohort ({cancer_type})"
    else:
        fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
        ref["_ref_value"] = ref[fpkm_cols].astype(float).mean(axis=1)
        cohort_label = "Mean across 33 TCGA cancer cohorts"

    ref_lookup = dict(zip(
        ref["Ensembl_Gene_ID"].map(_strip_ensembl_version),
        ref["_ref_value"],
    ))

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    # Normalize sample TPM to housekeeping (same scale as cohort reference)
    hk_ids = housekeeping_gene_ids()
    hk_mask = df[gene_id_col].isin(hk_ids)
    hk_median_tpm = df.loc[hk_mask, tpm_col].astype(float).median()
    if hk_median_tpm <= 0:
        hk_median_tpm = 1.0  # fallback

    gene_to_category = _create_gene_to_category_list_mapping(cat_to_ids)
    name_from_df = dict(zip(df[gene_id_col].astype(str), df[gene_name_col].astype(str)))

    rows = []
    for _, row in df.iterrows():
        gid = str(row[gene_id_col])
        tpm = float(row[tpm_col])
        sample_hk = tpm / hk_median_tpm  # housekeeping-normalized
        ref_val = ref_lookup.get(gid)
        if ref_val is None:
            continue
        cats = gene_to_category.get(gid, ["other"])
        name = name_from_df.get(gid) or id_to_name.get(gid) or gid
        display_name = aliases.get(name, name)
        for cat in cats:
            rows.append((gid, display_name, cat, sample_hk, ref_val))

    plot_df = pd.DataFrame(rows, columns=[
        "gene_id", "gene_name", "category", "sample_hk", "cohort_hk",
    ])
    # Offsets for log scale
    plot_df["sample_log"] = plot_df["sample_hk"] + 0.001
    plot_df["cohort_log"] = plot_df["cohort_hk"] + 0.001
    # Enrichment ratio: sample / cohort (high = sample-enriched)
    plot_df["enrichment"] = (plot_df["sample_hk"] + 0.001) / (plot_df["cohort_hk"] + 0.001)

    named_cats = list(cat_to_ids.keys())
    palette = sns.color_palette("tab10", len(named_cats))
    cat_to_color = dict(zip(named_cats, palette))

    sample_label = "Sample (housekeeping-normalized TPM)"
    cohort_axis_label = f"{cohort_label} (housekeeping-normalized)"

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

    # Highlight category — mark sample-enriched genes (enrichment > 5) with edge ring
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
        expand=(1.1, 1.4),
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

    return figures
