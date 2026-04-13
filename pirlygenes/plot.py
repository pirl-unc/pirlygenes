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

from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from adjustText import adjust_text

from .plot_data_helpers import prepare_gene_expr_df
from .gene_ids import find_canonical_gene_ids_and_names
from .gene_sets_cancer import (
    CTA_gene_id_to_name,
    housekeeping_gene_ids,
    cancer_surfaceome_gene_id_to_name,
    pan_cancer_expression,
    therapy_target_gene_id_to_name,
)
from .load_dataset import get_data


def _load_gene_sets():
    """Load immune gene sets as {category: {ensembl_id: symbol}} from CSV."""
    df = get_data("gene-sets")
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
        df_cat_sorted = df_cat[df_cat[tpm_col] > 0.1].sort_values(
            tpm_col, ascending=False
        )
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
    sets = _load_gene_sets()  # APM, MHC1, TLR, Growth_receptors, Oncogenes, Immune_checkpoints
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
    plot_height=14.0,
    plot_aspect=1.6,
    num_labels_per_category=5,
    always_label_genes=None,
    adjust_args=dict(
        expand=(1.4, 2.0),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.3),
        min_arrow_len=7,
        expand_axes=True,
        ensure_inside_axes=False,
    ),
    verbose=True,
    source_file=None,
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
        verbose=verbose,
        source_file=source_file,
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

    # Plot — build palette with gray for "other", tab10 for named categories
    other_name = "other"
    cat_col = df_gene_expr_annot["category"]
    cat_order = list(cat_col.cat.categories) if hasattr(cat_col, "cat") and hasattr(cat_col.cat, "categories") else list(dict.fromkeys(cat_col))
    named_cats_strip = [c for c in cat_order if c != other_name]
    named_palette = sns.color_palette("tab10", len(named_cats_strip))
    palette_dict = {c: color for c, color in zip(named_cats_strip, named_palette)}
    palette_dict[other_name] = (0.8, 0.8, 0.8)  # gray for "other"

    cat = sns.catplot(
        data=df_gene_expr_annot,
        x="category",
        y="TPM",
        jitter=0.01,
        height=plot_height,
        aspect=plot_aspect,
        alpha=0.5,
        hue="category",
        palette=palette_dict,
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
                        fontsize=7, color="#c0392b", alpha=0.85,
                        ha="right", va="top",
                    )
                )
        adjust_text(
            hk_texts,
            force_text=(0.5, 1.0),
            force_points=(0.5, 1.0),
            **{**adjust_args, "expand": (1.4, 2.0)},
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
    point_x = []
    point_y = []
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
                fontsize=8,
                color="black",
                alpha=0.8,
                ha="right",
                va="top",
            )
        )
        point_x.append(cat_to_x[row.category])
        point_y.append(row.TPM)

    adjust_text(
        texts,
        x=point_x,
        y=point_y,
        ax=cat.ax,
        force_text=(0.5, 1.0),
        force_points=(0.5, 1.0),
        **adjust_args,
    )

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


# -------------------- healthy tissue expression for therapy targets ----------


ESSENTIAL_TISSUES = [
    "brain", "heart_muscle", "liver", "lung", "kidney",
    "bone_marrow", "spleen", "pancreas", "colon", "stomach",
]

_THERAPY_PLOT_ORDER = [
    "ADC",
    "CAR-T",
    "TCR-T",
    "bispecific-antibodies",
    "radioligand",
]
_THERAPY_PLOT_LABELS = {
    "ADC": "ADC",
    "CAR-T": "CAR-T",
    "TCR-T": "TCR-T",
    "bispecific-antibodies": "Bispecific",
    "radioligand": "Radio",
}

# Map essential tissue labels to nTPM column names
_ESSENTIAL_TISSUE_COLS = {
    "brain": ["nTPM_cerebral_cortex", "nTPM_cerebellum", "nTPM_basal_ganglia",
              "nTPM_hippocampal_formation", "nTPM_amygdala", "nTPM_midbrain",
              "nTPM_hypothalamus", "nTPM_spinal_cord", "nTPM_choroid_plexus"],
    "heart": ["nTPM_heart_muscle"],
    "liver": ["nTPM_liver"],
    "lung": ["nTPM_lung"],
    "kidney": ["nTPM_kidney"],
    "bone_marrow": ["nTPM_bone_marrow"],
    "spleen": ["nTPM_spleen"],
    "pancreas": ["nTPM_pancreas"],
    "colon": ["nTPM_colon"],
    "stomach": ["nTPM_stomach"],
}


def _ordered_therapy_tuple(therapies):
    therapies = set(therapies)
    return tuple(sorted(therapies, key=_THERAPY_PLOT_ORDER.index))


def _therapy_combo_label(therapies):
    if not therapies:
        return "Other"
    return " + ".join(_THERAPY_PLOT_LABELS[t] for t in therapies)


def _therapy_combo_sort_key(therapies):
    return (len(therapies), [_THERAPY_PLOT_ORDER.index(t) for t in therapies])


def _therapy_base_colors():
    return dict(
        zip(_THERAPY_PLOT_ORDER, sns.color_palette("Set2", len(_THERAPY_PLOT_ORDER)))
    )


def _therapy_combo_colors(therapy_combos):
    import numpy as np

    base_palette = _therapy_base_colors()
    combo_to_color = {}
    for combo in sorted(set(therapy_combos), key=_therapy_combo_sort_key):
        if len(combo) == 1:
            combo_to_color[combo] = base_palette[combo[0]]
        else:
            rgb = np.array([base_palette[t] for t in combo]).mean(axis=0)
            combo_to_color[combo] = tuple(rgb.clip(0, 1))
    return combo_to_color


def _draw_therapy_marker(
    ax,
    x,
    y,
    therapies,
    marker="o",
    size=80,
    alpha=0.85,
    zorder=3,
):
    base_palette = _therapy_base_colors()
    ordered = _ordered_therapy_tuple(therapies)
    fill_color = base_palette.get(ordered[0], "gray") if ordered else "gray"

    ax.scatter(
        x,
        y,
        s=size,
        color=fill_color,
        marker=marker,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.6,
        zorder=zorder,
    )

    ring_size = size + 34
    for therapy in ordered[1:]:
        ax.scatter(
            x,
            y,
            s=ring_size,
            facecolors="none",
            edgecolors=base_palette.get(therapy, "gray"),
            marker=marker,
            alpha=0.95,
            linewidths=1.8,
            zorder=zorder - 0.1,
        )
        ring_size += 34

    return fill_color


def _approved_radioligand_gene_ids():
    df = get_data("radioligand-targets")
    if "Status_Bucket" not in df.columns:
        return set()
    mask = df["Status_Bucket"].astype(str).str.startswith("FDA_approved", na=False)
    return set(df.loc[mask, "Ensembl_Gene_ID"].astype(str))


def _collect_ranked_therapy_targets(df_gene_expr, top_k=10, tpm_threshold=30):
    from .plot_data_helpers import _strip_ensembl_version

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)

    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    sample_tpm = dict(zip(df[gene_id_col].astype(str), df[tpm_col].astype(float)))
    sample_name = dict(zip(df[gene_id_col].astype(str), df[gene_name_col].astype(str)))

    therapy_to_targets = {}
    gene_to_therapies = defaultdict(set)
    gene_name_from_sets = {}
    for therapy in _THERAPY_PLOT_ORDER:
        targets = {}
        for gid, sym in therapy_target_gene_id_to_name(therapy).items():
            gid_clean = _strip_ensembl_version(str(gid))
            targets[gid_clean] = sym
            gene_to_therapies[gid_clean].add(therapy)
            gene_name_from_sets.setdefault(gid_clean, sym)
        therapy_to_targets[therapy] = targets

    approved_sources = {
        "ADC": "ADC-approved",
        "CAR-T": "CAR-T",
        "TCR-T": "TCR-T-approved",
        "bispecific-antibodies": "bispecific-antibodies-approved",
    }
    gene_to_approved_therapies = defaultdict(set)
    for therapy, dataset in approved_sources.items():
        for gid in therapy_target_gene_id_to_name(dataset):
            gid_clean = _strip_ensembl_version(str(gid))
            gene_to_approved_therapies[gid_clean].add(therapy)
    for gid in _approved_radioligand_gene_ids():
        gene_to_approved_therapies[_strip_ensembl_version(str(gid))].add("radioligand")

    selected_gene_ids = set()
    for therapy, targets in therapy_to_targets.items():
        scored = []
        for gid in targets:
            tpm = sample_tpm.get(gid, 0.0)
            if tpm >= tpm_threshold:
                scored.append((gid, tpm))
        scored.sort(
            key=lambda item: (
                -item[1],
                sample_name.get(item[0], gene_name_from_sets.get(item[0], item[0])),
            )
        )
        selected_gene_ids.update(gid for gid, _ in scored[:top_k])

    records = []
    for gid in sorted(
        selected_gene_ids,
        key=lambda gene_id: (
            -sample_tpm.get(gene_id, 0.0),
            sample_name.get(gene_id, gene_name_from_sets.get(gene_id, gene_id)),
        ),
    ):
        therapies = _ordered_therapy_tuple(gene_to_therapies.get(gid, ()))
        approved_therapies = _ordered_therapy_tuple(
            gene_to_approved_therapies.get(gid, ())
        )
        display_name = sample_name.get(gid) or gene_name_from_sets.get(gid) or gid
        records.append(
            {
                "gene_id": gid,
                "symbol": display_name,
                "sample_tpm": sample_tpm.get(gid, 0.0),
                "therapies": therapies,
                "therapy_label": _therapy_combo_label(therapies),
                "approved_therapies": approved_therapies,
                "approved_label": _therapy_combo_label(approved_therapies)
                if approved_therapies
                else "",
                "has_approved": bool(approved_therapies),
            }
        )
    return records


def plot_therapy_target_tissues(
    df_gene_expr,
    top_k=10,
    tpm_threshold=30,
    save_to_filename=None,
    save_dpi=300,
):
    """For top expressed therapy targets, show healthy tissue expression vs sample.

    For each therapy category, takes the top K genes above a TPM threshold
    and creates a sorted bar plot of HPA normal tissue expression alongside
    the sample TPM value.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    top_k : int
        Number of top genes per therapy category.
    tpm_threshold : float
        Minimum sample TPM to include a gene.
    save_to_filename : str or None
        Output path (PDF with one page per gene, or PNG directory).
    """
    import numpy as np
    from pathlib import Path
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.backends.backend_pdf import PdfPages
    from .gene_sets_cancer import pan_cancer_expression

    # Load normal tissue expression
    ref = pan_cancer_expression()
    ntpm_cols = sorted([c for c in ref.columns if c.startswith("nTPM_")])
    tissue_labels = [c.replace("nTPM_", "").replace("_", " ") for c in ntpm_cols]
    ref_by_id = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")

    records = _collect_ranked_therapy_targets(
        df_gene_expr, top_k=top_k, tpm_threshold=tpm_threshold
    )
    if not records:
        print(f"No therapy targets above {tpm_threshold} TPM")
        return None
    combo_to_color = _therapy_combo_colors([record["therapies"] for record in records])

    figures = {}
    out = Path(save_to_filename) if save_to_filename else None
    pdf = PdfPages(out) if out is not None and out.suffix.lower() == ".pdf" else None
    if out is not None and out.suffix.lower() != ".pdf":
        out_dir = out.parent / out.stem
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Generate one page per gene
    for record in records:
        gid = record["gene_id"]
        sym = record["symbol"]
        s_tpm = record["sample_tpm"]
        combo_color = combo_to_color.get(record["therapies"], "#d62728")

        # Get tissue expression
        if gid in ref_by_id.index:
            tissue_vals = [float(ref_by_id.loc[gid, c]) if c in ref_by_id.columns else 0 for c in ntpm_cols]
        else:
            tissue_vals = [0] * len(ntpm_cols)

        # Sort tissues by expression
        sorted_pairs = sorted(zip(tissue_labels, tissue_vals), key=lambda x: -x[1])
        t_labels = [p[0] for p in sorted_pairs]
        t_vals = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(t_labels))
        ax.barh(y, t_vals, color="#aec7e8", edgecolor="none", height=0.7)

        # Sample TPM as a vertical line
        ax.axvline(x=s_tpm, color=combo_color, linewidth=2, linestyle="-", alpha=0.85)

        ax.set_yticks(y)
        ax.set_yticklabels(t_labels, fontsize=7)
        ax.set_xlabel("Expression (nTPM / TPM)", fontsize=10)
        subtitle = (
            f"Sample TPM: {s_tpm:.1f} | therapy categories: {record['therapy_label']}"
        )
        if record["has_approved"]:
            subtitle += f" | approved: {record['approved_label']}"
            ax.scatter(
                [0.97],
                [0.96],
                transform=ax.transAxes,
                marker="^",
                s=70,
                color=combo_color,
                edgecolors="black",
                linewidths=0.5,
                clip_on=False,
                zorder=6,
            )
            ax.text(
                0.93,
                0.96,
                "approved",
                transform=ax.transAxes,
                fontsize=8,
                ha="right",
                va="center",
                alpha=0.85,
            )
        ax.set_title(
            f"{sym} — healthy tissue vs sample\n{subtitle}",
            fontsize=11,
        )
        legend_handles = [
            Patch(facecolor="#aec7e8", edgecolor="none", label="Normal tissue (nTPM)"),
            Line2D(
                [],
                [],
                color=combo_color,
                linewidth=2,
                label=f"Sample TPM = {s_tpm:.1f}",
            ),
        ]
        if record["has_approved"]:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="^",
                    linestyle="None",
                    color=combo_color,
                    markeredgecolor="black",
                    label=f"Approved target ({record['approved_label']})",
                )
            )
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        elif out_dir is not None:
            fig.savefig(out_dir / f"{sym}.png", dpi=save_dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            figures[sym] = fig

    if pdf is not None:
        pdf.close()
        print(f"Saved {out} ({len(records)} pages)")
    elif out_dir is not None:
        print(f"Saved {len(records)} PNGs to {out_dir}/")

    return figures


def plot_therapy_target_safety(
    df_gene_expr,
    top_k=10,
    tpm_threshold=30,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """Scatter: sample TPM vs max essential-tissue expression for therapy targets.

    X-axis is sample expression, Y-axis is max expression in essential
    tissues (brain, heart, liver, lung, kidney). Genes in the lower-right
    (high sample, low essential tissue) are the safest therapy targets.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    top_k : int
        Number of top genes per therapy category to include.
    tpm_threshold : float
        Minimum sample TPM to include a gene.
    save_to_filename : str or None
        Output path.
    """
    from matplotlib.lines import Line2D
    from .gene_sets_cancer import pan_cancer_expression

    ref = pan_cancer_expression()
    ref_by_id = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")

    # Essential tissue columns
    essential_cols = []
    for tissue, cols in _ESSENTIAL_TISSUE_COLS.items():
        essential_cols.extend([c for c in cols if c in ref_by_id.columns])

    records = _collect_ranked_therapy_targets(
        df_gene_expr, top_k=top_k, tpm_threshold=tpm_threshold
    )
    if not records:
        print(f"No therapy targets above {tpm_threshold} TPM")
        return None, None
    base_palette = _therapy_base_colors()

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    point_x = []
    point_y = []

    for record in records:
        gid = record["gene_id"]
        sym = record["symbol"]
        tpm = record["sample_tpm"]
        max_ess = 0
        if gid in ref_by_id.index:
            vals = [
                float(ref_by_id.loc[gid, c])
                for c in essential_cols
                if c in ref_by_id.columns
            ]
            max_ess = max(vals) if vals else 0
        y_value = max_ess + 0.1
        marker = "^" if record["has_approved"] else "o"
        _draw_therapy_marker(
            ax,
            tpm,
            y_value,
            record["therapies"],
            marker=marker,
            size=80,
            alpha=0.85,
            zorder=3,
        )
        point_x.append(tpm)
        point_y.append(y_value)
        texts.append(
            ax.text(
                tpm * 1.03,
                y_value * 1.03,
                sym,
                fontsize=8,
                va="bottom",
                ha="left",
                alpha=0.85,
                zorder=4,
            )
        )

    therapy_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=base_palette[therapy],
            markeredgecolor="white",
            color=base_palette[therapy],
            label=_THERAPY_PLOT_LABELS[therapy],
            markersize=8,
        )
        for therapy in _THERAPY_PLOT_ORDER
    ]
    marker_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="#666666",
            label="Trial-only target",
            markersize=8,
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="None",
            color="#666666",
            markeredgecolor="black",
            label="Approved target",
            markersize=8,
        ),
    ]
    legend = ax.legend(
        handles=therapy_handles,
        loc="upper left",
        fontsize=8,
        frameon=False,
        title="Therapy categories\n(extra categories = outer rings)",
        title_fontsize=9,
    )
    ax.add_artist(legend)
    ax.legend(
        handles=marker_handles,
        loc="lower right",
        fontsize=8,
        frameon=False,
        title="Marker",
        title_fontsize=9,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample TPM", fontsize=11)
    ax.set_ylabel(
        "Max expression in selected essential tissues (nTPM)",
        fontsize=10,
    )
    ax.set_title(
        "Therapy target safety: sample expression vs essential tissue expression\n"
        "(brain, heart, liver, lung, kidney, bone marrow, spleen, pancreas, colon, stomach)",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    adjust_text(
        texts,
        x=point_x,
        y=point_y,
        ax=ax,
        expand=(1.15, 1.45),
        force_text=(0.15, 0.3),
        force_static=(0.12, 0.25),
        force_pull=(0.01, 0.02),
        arrowprops=dict(arrowstyle="-", color="#666666", alpha=0.35, lw=0.6),
        min_arrow_len=6,
        ensure_inside_axes=False,
    )

    # Diagonal reference
    lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])]
    if lims[1] > lims[0]:
        ax.plot(lims, lims, ":", color="#cccccc", linewidth=0.8, alpha=0.5, zorder=0)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


# -------------------- gene-set vs essential-tissue toxicity view -----------


def _resolve_gene_set_symbols(gene_set):
    """Return a list of gene symbols from a flexible gene-set specifier.

    Accepts:
        - an iterable of symbols: ("IFI6", "ISG15", ...) used directly
        - a str referring to a Category in `data/gene-sets.csv`
          (case-insensitive; the set `"Interferon response"` or
          `"interferon_response"` would match). Loaded via get_data.

    Raises ValueError if a string reference doesn't resolve.
    """
    if isinstance(gene_set, str):
        from .load_dataset import get_data
        df = get_data("gene-sets")
        target = gene_set.strip().lower().replace(" ", "_")
        categories_available = {
            str(c).strip().lower().replace(" ", "_"): c
            for c in df["Category"].dropna().unique()
        }
        matched = categories_available.get(target)
        if matched is None:
            raise ValueError(
                f"Gene set {gene_set!r} not found in data/gene-sets.csv. "
                f"Available: {sorted(categories_available.values())}"
            )
        symbols = df.loc[df["Category"] == matched, "Symbol"].astype(str).tolist()
        return [s for s in symbols if s]
    return [str(s) for s in gene_set]


def plot_geneset_vs_vital_tissues(
    df_gene_expr,
    gene_set,
    title=None,
    toxicity_tpm_threshold=10.0,
    vital_tissues=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Plot a sample's expression of a gene set against vital-tissue baselines.

    For each gene in the set, renders a horizontal strip on a log TPM
    axis with:
        - the sample's TPM (blue filled marker)
        - each vital tissue's nTPM (small tissue-colored dots, flipped
          to red when the tissue value exceeds `toxicity_tpm_threshold`
          — a visual flag that targeting this gene therapeutically
          would affect a tissue that can't tolerate damage / regenerate)

    Per-gene reading: "is the sample high AND are vital tissues also
    high on this gene?" Genes where only the sample is high are safer
    therapeutic targets; genes where one or more vital tissues are
    above the threshold are toxicity risks.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Sample expression (same schema expected by `_guess_gene_cols`).
    gene_set : str or iterable of str
        Either a gene-set Category name in `data/gene-sets.csv`
        (e.g. "Interferon response", "MHC1_presentation") or an explicit
        list of gene symbols.
    title : str or None
        Figure title.
    toxicity_tpm_threshold : float
        nTPM above which a vital-tissue dot is colored red. Default 10.0.
    vital_tissues : list of str or None
        Keys into `_ESSENTIAL_TISSUE_COLS` to display. Default: all.
    save_to_filename, save_dpi : plot save options.
    figsize : tuple or None
        Explicit size. Default scales with gene count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .gene_sets_cancer import pan_cancer_expression
    from .plot_data_helpers import _strip_ensembl_version

    symbols = _resolve_gene_set_symbols(gene_set)
    if not symbols:
        print(f"Gene set {gene_set!r} resolved to zero genes.")
        return None

    if vital_tissues is None:
        vital_tissues = list(_ESSENTIAL_TISSUE_COLS.keys())
    else:
        missing = [t for t in vital_tissues if t not in _ESSENTIAL_TISSUE_COLS]
        if missing:
            raise ValueError(
                f"Unknown vital_tissues: {missing}. "
                f"Available: {list(_ESSENTIAL_TISSUE_COLS.keys())}"
            )

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)
    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column in sample. Columns: {list(df.columns)}")

    ref = pan_cancer_expression()
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    sample_by_symbol = {}
    for _, row in df.iterrows():
        gid = str(row[gene_id_col])
        sym = id_to_sym.get(gid)
        if sym is None:
            sym = str(row.get(gene_name_col, ""))
        if not sym:
            continue
        tpm = float(row[tpm_col])
        if sym not in sample_by_symbol or tpm > sample_by_symbol[sym]:
            sample_by_symbol[sym] = tpm

    # For tissue groups like "brain" that span several nTPM_* columns,
    # use the MAX — any single CNS region with high expression is a
    # toxicity concern, no need to average it out.
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    tissue_columns_resolved = {
        tissue: [c for c in _ESSENTIAL_TISSUE_COLS[tissue] if c in ref_by_sym.columns]
        for tissue in vital_tissues
    }

    def _tissue_value(sym, tissue):
        cols = tissue_columns_resolved.get(tissue, [])
        if sym not in ref_by_sym.index or not cols:
            return None
        return float(ref_by_sym.loc[sym, cols].astype(float).max())

    rows = []
    for sym in symbols:
        sample_tpm = sample_by_symbol.get(sym)
        tissue_vals = {t: _tissue_value(sym, t) for t in vital_tissues}
        has_any = (sample_tpm is not None and sample_tpm > 0) or any(
            v is not None and v > 0 for v in tissue_vals.values()
        )
        if has_any:
            rows.append((sym, sample_tpm or 0.0, tissue_vals))

    if not rows:
        print(f"Gene set {gene_set!r} had no expression in sample or reference.")
        return None

    n = len(rows)
    if figsize is None:
        figsize = (11, 0.32 * n + 2.2)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)

    tissue_colors = {
        "brain": "#6a3d9a",
        "heart": "#e31a1c",
        "liver": "#ff7f00",
        "lung": "#1f78b4",
        "kidney": "#33a02c",
        "bone_marrow": "#b15928",
        "spleen": "#a6cee3",
        "pancreas": "#fb9a99",
        "colon": "#b2df8a",
        "stomach": "#fdbf6f",
    }

    # Vertical jitter spread for tissue dots — overlapping tissues at
    # similar TPM become distinguishable. Spread is symmetric around the
    # gene's row index, inside ±0.35 so rows don't visually collide.
    n_tissues = max(1, len(vital_tissues))
    jitter_map = {
        tissue: (idx / max(1, n_tissues - 1) - 0.5) * 0.7
        for idx, tissue in enumerate(vital_tissues)
    }

    for i, (sym, sample_tpm, tissue_vals) in enumerate(rows):
        x_sample = max(sample_tpm, 0.05)
        ax.scatter(
            [x_sample], [i], s=110, color="#1f77b4",
            edgecolor="black", linewidth=0.8, zorder=5, label=None,
        )
        ax.text(
            x_sample * 1.2, i, f"{sample_tpm:.0f}", fontsize=7.5,
            va="center", color="#1f77b4", fontweight="bold", zorder=6,
        )
        for tissue, val in tissue_vals.items():
            if val is None:
                continue
            base = tissue_colors.get(tissue, "#888888")
            if val >= toxicity_tpm_threshold:
                color = "#d62728"
                edge = "#7a0000"
                lw = 0.9
                size = 55
            else:
                color = base
                edge = "white"
                lw = 0.4
                size = 32
            x = max(val, 0.05)
            y_pos = i + jitter_map.get(tissue, 0.0)
            ax.scatter([x], [y_pos], s=size, color=color, edgecolor=edge,
                       linewidth=lw,
                       zorder=4 if color == "#d62728" else 3,
                       alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlim(0.05, 10_000)
    ax.axvline(toxicity_tpm_threshold, linestyle="--", color="#d62728",
               linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "TPM (log scale) — sample in blue, vital tissues colored by tissue; red = above threshold"
    )
    if title is None:
        title = (
            f"{gene_set} vs vital tissues"
            if isinstance(gene_set, str)
            else "Gene set vs vital tissues"
        )
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_entries = [
        Line2D([], [], marker="o", color="#1f77b4", markeredgecolor="black",
               markersize=10, linestyle="", label="sample"),
    ]
    for tissue in vital_tissues:
        legend_entries.append(
            Line2D([], [], marker="o",
                   color=tissue_colors.get(tissue, "#888888"),
                   markersize=6, linestyle="",
                   label=tissue.replace("_", " "))
        )
    legend_entries.append(
        Line2D([], [], marker="o", color="#d62728", markeredgecolor="#7a0000",
               markersize=8, linestyle="",
               label=f"tissue > {toxicity_tpm_threshold:g} TPM")
    )
    ax.legend(
        handles=legend_entries, loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend_entries), 6), fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


# -------------------- per-cancer-type CTA detail view ----------------------


def plot_ctas_vs_cancer_type_detail(
    df_gene_expr,
    cancer_type,
    top_k=30,
    min_sample_tpm=1.0,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Per-sample CTA expression zoomed to one cancer type.

    Each row is one CTA gene. Each row plots, on a single log TPM axis:

        - sample TPM ................ large blue filled circle + label
        - TCGA cohort median ........ orange diamond (cancer_type's FPKM_*
                                      column, treated as TPM-equivalent)
        - tissue-of-origin normal ... green square (nTPM for the normal
                                      tissue mapped from cancer_type via
                                      CANCER_TO_TISSUE)
        - testis .................... purple triangle (expected baseline
                                      for CTAs — the tissue they're
                                      *defined* against)
        - max non-testis vital ...... red X when its value crosses the
                                      10-TPM toxicity threshold, else a
                                      small gray X. The offending tissue
                                      name is annotated to the right.

    Reads per gene:

        - sample ≫ cohort → outlier up-regulation in this patient
        - sample ≈ cohort → this CTA is typical for the cancer type
        - sample ≪ cohort → cohort expresses it but this patient doesn't
        - high red X → toxicity risk (a non-testis vital tissue also
          expresses this CTA, regardless of tumor/cohort signal)

    The data is cohort-level (one median per cancer type), so "per-
    patient variation within PRAD" is not available. This plot shows
    what IS in the dataset: how a single sample compares to the per-
    cohort median + normal-tissue baselines.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Sample expression with gene ID, name, TPM columns.
    cancer_type : str
        TCGA cancer code or alias (e.g. "PRAD", "prostate").
    top_k : int
        Render at most this many CTAs, ranked by sample TPM.
    min_sample_tpm : float
        Skip CTAs where sample TPM is below this; keeps the plot focused
        on genes that are actually expressed in this patient.
    save_to_filename, save_dpi : plot save options.
    figsize : tuple or None
        Explicit size. Default scales with the number of rendered rows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.lines import Line2D

    from .gene_sets_cancer import CTA_gene_id_to_name, pan_cancer_expression
    from .plot_data_helpers import _strip_ensembl_version
    from .tumor_purity import CANCER_TO_TISSUE

    cancer_code = resolve_cancer_type(cancer_type)
    ref = pan_cancer_expression()
    cancer_col = f"FPKM_{cancer_code}"
    if cancer_col not in ref.columns:
        raise ValueError(
            f"Unknown cancer_type {cancer_type!r} → code {cancer_code!r} "
            f"has no FPKM column in the reference."
        )

    # CTAs as Ensembl IDs + display symbols.
    cta_id_to_name = CTA_gene_id_to_name()
    if not cta_id_to_name:
        print("No CTAs available in reference.")
        return None

    # Build a row per CTA: sample TPM, cancer-cohort TPM-equivalent,
    # tissue-of-origin nTPM, testis nTPM, max non-testis vital nTPM.
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    sample = df_gene_expr.copy()
    sample[gene_id_col] = sample[gene_id_col].astype(str).map(_strip_ensembl_version)
    tpm_col = "TPM" if "TPM" in sample.columns else next(
        (c for c in sample.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column in sample. Columns: {list(sample.columns)}")
    sample_tpm = dict(zip(
        sample[gene_id_col].astype(str),
        sample[tpm_col].astype(float),
    ))

    ref_by_id = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index(
        "Ensembl_Gene_ID"
    )
    origin_tissue = CANCER_TO_TISSUE.get(cancer_code)
    origin_col = f"nTPM_{origin_tissue}" if origin_tissue else None
    testis_col = "nTPM_testis" if "nTPM_testis" in ref_by_id.columns else None

    # Vital-tissue columns minus testis (testis is treated separately as
    # the "expected" CTA baseline, not a toxicity concern).
    vital_tissue_cols = []
    for tissue in ESSENTIAL_TISSUES:
        for col in _ESSENTIAL_TISSUE_COLS.get(tissue, []):
            if col in ref_by_id.columns and col != testis_col:
                vital_tissue_cols.append((tissue, col))

    rows = []
    for gid, sym in cta_id_to_name.items():
        gid_clean = _strip_ensembl_version(str(gid))
        s_tpm = sample_tpm.get(gid_clean, 0.0)
        if s_tpm < min_sample_tpm:
            continue
        if gid_clean not in ref_by_id.index:
            continue
        ref_row = ref_by_id.loc[gid_clean]
        cohort_val = float(ref_row[cancer_col])
        origin_val = float(ref_row[origin_col]) if origin_col and origin_col in ref_row else None
        testis_val = float(ref_row[testis_col]) if testis_col else None

        worst_vital = None
        for tissue, col in vital_tissue_cols:
            val = float(ref_row[col]) if col in ref_row else 0.0
            if worst_vital is None or val > worst_vital[1]:
                worst_vital = (tissue, val, col)

        rows.append({
            "symbol": sym,
            "sample": s_tpm,
            "cohort": cohort_val,
            "origin": origin_val,
            "testis": testis_val,
            "worst_vital": worst_vital,
        })

    if not rows:
        print(
            f"No CTAs with sample TPM ≥ {min_sample_tpm} found for {cancer_code}."
        )
        return None

    # Rank by sample TPM descending; truncate to top_k.
    rows.sort(key=lambda r: -r["sample"])
    rows = rows[:top_k]
    n = len(rows)
    if figsize is None:
        figsize = (13, 0.34 * n + 2.2)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(n)

    toxicity_threshold = 10.0

    for i, r in enumerate(rows):
        # Sample — large blue circle, TPM labeled inline to the right of
        # the marker.
        x_sample = max(r["sample"], 0.05)
        ax.scatter([x_sample], [i], s=120, color="#1f77b4",
                   edgecolor="black", linewidth=0.8, zorder=6)
        ax.text(x_sample * 1.2, i, f"{r['sample']:.0f}",
                fontsize=7.5, va="center", color="#1f77b4",
                fontweight="bold", zorder=7)

        # Cohort median — orange diamond.
        x_cohort = max(r["cohort"], 0.05)
        ax.scatter([x_cohort], [i], s=70, marker="D", color="#ff7f0e",
                   edgecolor="#704000", linewidth=0.6, zorder=5, alpha=0.95)

        # Tissue-of-origin normal — green square.
        if r["origin"] is not None:
            x_origin = max(r["origin"], 0.05)
            ax.scatter([x_origin], [i], s=65, marker="s", color="#2ca02c",
                       edgecolor="#175417", linewidth=0.6, zorder=5, alpha=0.95)

        # Testis — purple triangle (CTAs' expected reference).
        if r["testis"] is not None:
            x_testis = max(r["testis"], 0.05)
            ax.scatter([x_testis], [i], s=65, marker="^", color="#9467bd",
                       edgecolor="#4a285e", linewidth=0.6, zorder=5, alpha=0.95)

        # Worst non-testis vital tissue — only prominent if it crosses
        # the toxicity threshold; otherwise faint gray X.
        if r["worst_vital"] is not None:
            tissue, val, _col = r["worst_vital"]
            x_vital = max(val, 0.05)
            if val >= toxicity_threshold:
                ax.scatter([x_vital], [i], s=95, marker="X",
                           color="#d62728", edgecolor="#7a0000",
                           linewidth=0.9, zorder=5.5)
                ax.text(x_vital * 1.15, i,
                        f"  {tissue} {val:.0f}",
                        fontsize=7, va="center", color="#d62728",
                        zorder=6)
            else:
                ax.scatter([x_vital], [i], s=45, marker="X",
                           color="#999999", edgecolor="none", zorder=4,
                           alpha=0.7)

    ax.axvline(toxicity_threshold, linestyle="--", color="#d62728",
               linewidth=0.8, alpha=0.45, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.05, max(1000.0, max(r["sample"] for r in rows) * 3))
    ax.set_yticks(y)
    ax.set_yticklabels([r["symbol"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("TPM (log scale)")
    origin_display = origin_tissue.replace("_", " ") if origin_tissue else "origin tissue"
    ax.set_title(
        f"CTA detail — {cancer_code} (sample vs cohort / {origin_display} / testis / worst vital tissue)",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = [
        Line2D([], [], marker="o", color="#1f77b4", markeredgecolor="black",
               markersize=10, linestyle="", label="sample"),
        Line2D([], [], marker="D", color="#ff7f0e", markeredgecolor="#704000",
               markersize=8, linestyle="", label=f"{cancer_code} cohort"),
    ]
    if origin_tissue:
        legend.append(Line2D(
            [], [], marker="s", color="#2ca02c", markeredgecolor="#175417",
            markersize=8, linestyle="",
            label=f"normal {origin_display}",
        ))
    legend.extend([
        Line2D([], [], marker="^", color="#9467bd", markeredgecolor="#4a285e",
               markersize=8, linestyle="", label="testis"),
        Line2D([], [], marker="X", color="#d62728", markeredgecolor="#7a0000",
               markersize=9, linestyle="",
               label=f"vital tissue > {toxicity_threshold:g} TPM"),
        Line2D([], [], marker="X", color="#999999", markersize=6,
               linestyle="", label="worst vital tissue (below threshold)"),
    ])
    ax.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend), 3), fontsize=8, frameon=False,
    )
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


# -------------------- cancer-type gene signature plots --------------------


def _sample_expression_by_symbol(df_gene_expr):
    import pandas as pd
    from .plot_data_helpers import _strip_ensembl_version

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    raw_values = df[tpm_col].astype(float)
    hk_mask = df[gene_id_col].isin(housekeeping_gene_ids())
    hk_median = df.loc[hk_mask, tpm_col].astype(float).median()
    if not (hk_median > 0):  # catches NaN and <= 0
        hk_median = 1.0
    hk_values = raw_values / hk_median

    # Resolve symbols from Ensembl IDs via pan-cancer reference
    ref_lookup = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    id_to_symbol = dict(zip(ref_lookup["Ensembl_Gene_ID"], ref_lookup["Symbol"]))
    if "canonical_gene_name" in df.columns:
        fallback = df["canonical_gene_name"].fillna("").astype(str)
    else:
        fallback = df[gene_name_col].fillna("").astype(str)
    symbols = df[gene_id_col].map(id_to_symbol).fillna(fallback)

    expr_df = pd.DataFrame(
        {
            "gene_id": df[gene_id_col],
            "Symbol": symbols,
            "sample_raw": raw_values,
            "sample_hk": hk_values,
        }
    )
    expr_df = expr_df[expr_df["Symbol"].astype(str).str.strip().ne("")]
    # Aggregate by Ensembl ID (unique), then map to symbol.
    # Sum across rows with same ID (alt-haplotype reads are split by aligner).
    grouped = expr_df.groupby("gene_id", as_index=False, sort=False).agg(
        {"Symbol": "first", "sample_raw": "sum", "sample_hk": "sum"}
    )
    return (
        dict(zip(grouped["Symbol"], grouped["sample_raw"])),
        dict(zip(grouped["Symbol"], grouped["sample_hk"])),
    )


def estimate_tumor_expression(
    df_gene_expr,
    cancer_type,
    purity,
):
    """Estimate true tumor cell expression by deconvolving TME contribution.

    For each gene: ``tumor_expr = (observed - (1-purity) * tme_ref) / purity``

    Genes are categorized into:
    - **CTA**: cancer-testis antigens (vaccination targets)
    - **therapy_target**: genes with active therapy trials
    - **surface**: known surface proteins (ADC/CAR-T/bispecific targets)
    - **other**: remaining genes with meaningful tumor signal

    Returns a DataFrame with columns: gene_id, symbol, category,
    observed_tpm, tme_expected, tumor_adjusted, tcga_median,
    tcga_percentile, is_surface, therapies.
    """
    import numpy as np
    import pandas as pd
    from .gene_sets_cancer import (
        CTA_gene_id_to_name,
        therapy_target_gene_id_to_name,
        surface_protein_gene_ids,
        cancer_surfaceome_gene_id_to_name,
    )

    cancer_code = resolve_cancer_type(cancer_type)

    # Sample expression
    sample_raw, _ = _sample_expression_by_symbol(df_gene_expr)

    # Reference data
    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    # TME tissues
    ptprc_row = ref_dedup.loc["PTPRC"] if "PTPRC" in ref_dedup.index else None
    if ptprc_row is not None:
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float)
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro
        if c.replace("nTPM_", "") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))

    # Cancer type origin tissue: map cancer type to closest normal tissue
    cancer_col = f"FPKM_{cancer_code}"
    tcga_expr = ref_dedup[cancer_col].astype(float) if cancer_col in ref_dedup.columns else None

    # Build gene lookup sets
    cta_map = CTA_gene_id_to_name()  # {ensembl_id: name}
    cta_symbols = set(cta_map.values())

    # Therapy targets across all therapy types
    _all_therapy_keys = [
        "ADC", "ADC-approved", "CAR-T", "CAR-T-approved",
        "TCR-T", "TCR-T-approved", "bispecific-antibodies",
        "bispecific-antibodies-approved", "radioligand",
    ]
    gene_therapies = {}  # symbol -> set of base therapy types
    for tt in _all_therapy_keys:
        try:
            tmap = therapy_target_gene_id_to_name(tt)
            base = tt.replace("-approved", "").replace("-trials", "")
            for gid, gname in tmap.items():
                gene_therapies.setdefault(gname, set()).add(base)
        except Exception:
            pass

    # Surface proteins
    try:
        surf_ids = surface_protein_gene_ids()
        cancer_surf = cancer_surfaceome_gene_id_to_name()
        ref_flat = ref.drop_duplicates(subset="Ensembl_Gene_ID")
        eid_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
        surf_symbols = {eid_to_sym.get(eid, "") for eid in surf_ids}
        surf_symbols |= set(cancer_surf.values())
        surf_symbols.discard("")
    except Exception:
        surf_symbols = set()

    # TME reference: mean across TME tissues for each gene
    if tme_cols:
        tme_mean = ref_dedup[tme_cols].astype(float).mean(axis=1)
    else:
        tme_mean = pd.Series(0, index=ref_dedup.index)

    # TCGA distribution for percentile calculation
    cancer_expr_all = ref_dedup[fpkm_cols].astype(float)

    # Build result rows — only process genes that the sample expresses
    # or that are in a known target category
    interesting_symbols = set(cta_symbols) | set(gene_therapies.keys())
    interesting_symbols |= {s for s, v in sample_raw.items() if v > 0.1}

    rows = []
    purity_clamp = max(purity, 0.01)  # avoid division by zero

    for symbol in interesting_symbols:
        if symbol not in ref_dedup.index:
            continue
        observed = sample_raw.get(symbol, 0.0)
        tme_ref = float(tme_mean.get(symbol, 0))
        tcga_med = float(tcga_expr[symbol]) if tcga_expr is not None else 0.0

        # Purity adjustment
        tumor_adj = max(0, (observed - (1 - purity_clamp) * tme_ref) / purity_clamp)

        # TCGA percentile
        ref_vals = cancer_expr_all.loc[symbol].values
        n = len(ref_vals)
        below = np.sum(ref_vals < tumor_adj)
        equal = np.sum(np.isclose(ref_vals, tumor_adj, atol=0.01))
        pctile = float((below + 0.5 * equal) / n)

        # Categorize
        is_cta = symbol in cta_symbols
        is_surface = symbol in surf_symbols
        therapies = gene_therapies.get(symbol, set())
        is_therapy = bool(therapies)

        # Filter: only include genes with meaningful tumor signal
        # or that are in a known category
        if tumor_adj < 0.5 and not is_cta and not is_therapy:
            continue

        if is_cta:
            category = "CTA"
        elif is_therapy:
            category = "therapy_target"
        elif is_surface and tumor_adj > 1:
            category = "surface"
        else:
            category = "other"

        eid = ref_dedup.loc[symbol, "Ensembl_Gene_ID"] if "Ensembl_Gene_ID" in ref_dedup.columns else ""

        rows.append({
            "gene_id": eid,
            "symbol": symbol,
            "category": category,
            "observed_tpm": round(observed, 2),
            "tme_expected": round(tme_ref, 2),
            "tumor_adjusted": round(tumor_adj, 2),
            "tcga_median": round(tcga_med, 2),
            "tcga_percentile": round(pctile, 3),
            "is_surface": is_surface,
            "is_cta": is_cta,
            "therapies": ", ".join(sorted(therapies)) if therapies else "",
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("tumor_adjusted", ascending=False).reset_index(drop=True)
    return result


def estimate_tumor_expression_ranges(
    df_gene_expr,
    cancer_type,
    purity_result,
    decomposition_results=None,
):
    """Estimate tumor-specific expression with uncertainty bounds.

    For each gene, computes a 3×3 grid of estimates by crossing
    (low, med, high) TME background with (low, med, high) purity:

        tumor_expr = max(0, (observed - (1-purity) * tme_bg) / purity)

    TME bounds come from either:

    - the 25th / 50th / 75th percentile across TME reference tissues, or
    - the 25th / 50th / 75th percentile across candidate decomposition
      hypotheses when ``decomposition_results`` is provided.

    Purity bounds come from the ``overall_lower`` / ``overall_estimate`` /
    ``overall_upper`` fields of ``estimate_tumor_purity()``.

    Parameters
    ----------
    df_gene_expr : DataFrame
        Sample gene expression with TPM column.
    cancer_type : str
        TCGA cancer type code or name.
    purity_result : dict
        Return value of ``estimate_tumor_purity()``.
    decomposition_results : list, optional
        Candidate ``DecompositionResult`` objects from
        ``pirlygenes.decomposition.decompose_sample()``.

    Returns
    -------
    DataFrame with columns: symbol, category, observed_tpm,
        tme_lo, tme_med, tme_hi,
        est_1 … est_9 (the 3×3 grid, ascending order),
        median_est, therapies, is_surface, is_cta.
    """
    import numpy as np
    import pandas as pd
    from .gene_sets_cancer import (
        CTA_gene_id_to_name,
        therapy_target_gene_id_to_name,
        surface_protein_gene_ids,
        cancer_surfaceome_gene_id_to_name,
    )

    from .gene_sets_cancer import housekeeping_gene_ids
    from .tumor_purity import TCGA_MEDIAN_PURITY

    cancer_code = resolve_cancer_type(cancer_type)

    # --- Sample expression (raw TPM and HK-normalized) ---
    sample_raw, sample_hk = _sample_expression_by_symbol(df_gene_expr)

    # Sample HK median (for converting back from fold-HK to TPM)
    hk_ids = housekeeping_gene_ids()
    ref_full = pan_cancer_expression()
    ref_flat = ref_full.drop_duplicates(subset="Ensembl_Gene_ID")
    id_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
    hk_syms = {id_to_sym[gid] for gid in hk_ids if gid in id_to_sym}
    sample_hk_vals = [sample_raw[s] for s in hk_syms if sample_raw.get(s, 0) > 0]
    sample_hk_median = float(np.median(sample_hk_vals)) if sample_hk_vals else 1.0

    # --- Reference data ---
    ref_dedup = ref_full.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref_full.columns if c.startswith("nTPM_")]
    fpkm_cols = [c for c in ref_full.columns if c.startswith("FPKM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    # TME tissues (curated immune + stromal)
    tme_cols = [c for c in ntpm_nonrepro if c.replace("nTPM_", "") in _TME_TISSUES]

    # --- HK-normalize reference columns ---
    # Each column (nTPM tissue or FPKM cancer type) gets its own HK median
    hk_in_ref = sorted(hk_syms & set(ref_dedup.index))
    ref_hk_medians = {}
    for col in tme_cols + fpkm_cols:
        ref_hk_medians[col] = ref_dedup.loc[hk_in_ref, col].astype(float).median()

    # TME reference in HK-fold space: per gene, per tissue or decomposition.
    # When decomposition hypotheses are available, use their inferred
    # non-tumor background profile instead of a generic TME tissue panel.
    decomp_backgrounds = []
    if decomposition_results:
        for result in decomposition_results:
            bg = getattr(result, "tme_background_hk", None)
            if bg:
                decomp_backgrounds.append(bg)

    # Per-gene max across healthy tissues (precomputed vector; used to
    # flag rows where sample signal could be entirely TME-explained).
    # Vectorized once here to avoid a per-gene `.loc[].max()` inside the
    # main loop — that was a measured ~30x slowdown on full panels.
    if ntpm_nonrepro:
        _max_healthy = ref_dedup[ntpm_nonrepro].astype(float).max(axis=1)
        max_healthy_tpm_by_symbol = _max_healthy.to_dict()
    else:
        max_healthy_tpm_by_symbol = {}

    # --- Full-coverage TPM-space TME background (fixes issue #45) --------
    #
    # The decomposition's `tme_background_hk` dict only covers genes that
    # were in the decomposition's signature panel. For target-list genes
    # like FN1 / COL1A1 / IGKC that AREN'T signature genes but ARE clearly
    # stromal/immune expressed, `tme_background_hk.get(sym, 0.0)` returns 0
    # → no TME subtraction → `tumor_tpm ≈ sample_tpm / purity` which
    # inflates the "Tumor TPM" reported in the target table.
    #
    # Fix: when we have a decomposition result with a `fractions` dict,
    # build a full-gene signature matrix using the same cell-type /
    # bulk-tissue references the decomposition engine uses, and compute:
    #
    #   tme_bg_tpm[g] = Σ_{c ≠ tumor} fractions[c] × ref_tpm[g, c]
    #
    # This gives a per-gene TPM-scale TME background for every gene in
    # the reference, not just signature genes. The formula in the loop
    # prefers this TPM-space path when available and falls back to the
    # HK-fold path when it isn't.
    #
    # Matched-normal split (issue #50): when the decomposition included a
    # `matched_normal_<tissue>` component (epithelial primaries with
    # `use_matched_normal=True`), its contribution is tracked separately
    # as `matched_normal_tpm_by_symbol`. This lets the target report
    # distinguish "subtracted stromal/immune background" from "subtracted
    # benign parent-tissue contribution" per gene. The formula sums
    # them — total non-tumor background is still `tme_only + matched_normal`.
    tme_bg_tpm_by_symbol = None
    tme_only_tpm_by_symbol = None
    matched_normal_tpm_by_symbol = None
    matched_normal_component_name = None
    matched_normal_tissue = None
    matched_normal_fraction_global = 0.0
    if decomposition_results:
        top_result = decomposition_results[0]
        top_fractions = getattr(top_result, "fractions", None) or {}
        non_tumor_components = [
            c for c, f in top_fractions.items() if c != "tumor" and f > 0
        ]
        matched_normal_component_name = getattr(
            top_result, "matched_normal_tissue", None,
        )
        if matched_normal_component_name is not None:
            matched_normal_tissue = matched_normal_component_name
            matched_normal_component_name = f"matched_normal_{matched_normal_tissue}"
            matched_normal_fraction_global = float(
                getattr(top_result, "matched_normal_fraction", 0.0) or 0.0
            )
        if non_tumor_components:
            from .decomposition.signature import build_signature_matrix
            try:
                _genes, sym_list, matrix, _cols = build_signature_matrix(
                    non_tumor_components,
                    gene_subset=None,
                    sample_by_eid=None,
                )
                non_tumor_fracs = np.array(
                    [float(top_fractions[c]) for c in non_tumor_components],
                    dtype=float,
                )
                # Per-gene expected non-tumor contribution to sample TPM:
                # Σ_c fractions[c] × ref_tpm[g, c]. `fractions[c]` is
                # already scaled by (1 − tumor_fraction), so summing
                # directly gives absolute non-tumor TPM — no extra
                # (1 − p) multiplier needed at the formula site.
                tme_tpm_vec = matrix @ non_tumor_fracs
                tme_bg_tpm_by_symbol = {
                    str(sym): float(val)
                    for sym, val in zip(sym_list, tme_tpm_vec)
                }
                # Split out the matched-normal contribution so the report
                # can distinguish TME-only subtraction from parent-tissue
                # subtraction (issue #50). The matched-normal column is
                # present only when the decomposition was run with
                # `use_matched_normal=True` on an epithelial primary.
                if matched_normal_component_name in non_tumor_components:
                    mn_idx = non_tumor_components.index(matched_normal_component_name)
                    mn_frac = float(non_tumor_fracs[mn_idx])
                    mn_vec = matrix[:, mn_idx] * mn_frac
                    matched_normal_tpm_by_symbol = {
                        str(sym): float(val)
                        for sym, val in zip(sym_list, mn_vec)
                    }
                    tme_only_vec = tme_tpm_vec - mn_vec
                    tme_only_tpm_by_symbol = {
                        str(sym): float(max(0.0, val))
                        for sym, val in zip(sym_list, tme_only_vec)
                    }
                else:
                    tme_only_tpm_by_symbol = dict(tme_bg_tpm_by_symbol)
            except Exception:
                tme_bg_tpm_by_symbol = None
                tme_only_tpm_by_symbol = None
                matched_normal_tpm_by_symbol = None

    # --- Purity-adjusted TCGA (HK-normalized, then deconvolved) ---
    # For each FPKM cancer-type column, compute:
    #   tcga_hk = FPKM / FPKM_HK_median
    #   tme_hk  = median TME tissue fold (same as sample TME reference)
    #   tcga_tumor_hk = (tcga_hk - (1-tcga_purity) * tme_hk) / tcga_purity
    # We'll compute this per-gene in the loop.

    # --- Purity bounds ---
    p_lo = max(purity_result.get("overall_lower") or 0.01, 0.01)
    p_med = max(purity_result.get("overall_estimate") or 0.05, 0.01)
    p_hi = max(purity_result.get("overall_upper") or p_med, 0.01)
    p_lo, p_med, p_hi = sorted([p_lo, p_med, p_hi])

    # --- Gene category lookups ---
    cta_symbols = set(CTA_gene_id_to_name().values())

    _all_therapy_keys = [
        "ADC", "ADC-approved", "CAR-T", "CAR-T-approved",
        "TCR-T", "TCR-T-approved", "bispecific-antibodies",
        "bispecific-antibodies-approved", "radioligand",
    ]
    gene_therapies = {}
    for tt in _all_therapy_keys:
        try:
            tmap = therapy_target_gene_id_to_name(tt)
            base = tt.replace("-approved", "").replace("-trials", "")
            for gid, gname in tmap.items():
                gene_therapies.setdefault(gname, set()).add(base)
        except Exception:
            pass

    try:
        surf_ids = surface_protein_gene_ids()
        cancer_surf = cancer_surfaceome_gene_id_to_name()
        eid_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
        surf_symbols = {eid_to_sym.get(eid, "") for eid in surf_ids}
        surf_symbols |= set(cancer_surf.values())
        surf_symbols.discard("")
    except Exception:
        surf_symbols = set()

    # --- Compute 9-point estimates for every expressed gene ---
    cancer_expr_all = ref_dedup[fpkm_cols].astype(float)
    rows = []
    for symbol in sample_raw:
        if symbol not in ref_dedup.index:
            continue
        observed = sample_raw[symbol]
        if observed < 0.01:
            continue

        # HK-normalize sample
        sample_fold = observed / sample_hk_median

        # TME background in HK-fold space.
        if decomp_backgrounds:
            tme_folds = [float(bg.get(symbol, 0.0)) for bg in decomp_backgrounds]
        else:
            tme_folds = []
            for col in tme_cols:
                hk_m = ref_hk_medians.get(col, 0)
                if hk_m > 0:
                    tme_folds.append(float(ref_dedup.loc[symbol, col]) / hk_m)
        if not tme_folds:
            tme_folds = [0.0]
        tme_fold_lo = float(np.percentile(tme_folds, 25))
        tme_fold_med = float(np.median(tme_folds))
        tme_fold_hi = float(np.percentile(tme_folds, 75))

        # TME-explainability flag: the max TPM this gene reaches in ANY
        # single healthy reference tissue. Used below to decide how
        # aggressively to clamp toward the cohort prior.
        max_healthy_tpm = float(max_healthy_tpm_by_symbol.get(symbol, 0.0))
        tme_explainable = max_healthy_tpm >= observed * 0.5

        # Cohort prior for tumor expression in this cancer type. Computed
        # by deconvolving the TCGA cancer-cohort median against the cohort's
        # assumed median purity, then rescaling to the sample's TPM scale.
        # This prior implicitly includes the "reactive stroma" signal
        # typical for this cancer type (TCGA medians are bulk tumor
        # samples, so tissue-specific cancer-associated fibroblast and
        # infiltrate contributions are baked in). Used for empirical-
        # Bayes shrinkage of the sample-based estimate at low purity.
        cancer_col = f"FPKM_{cancer_code}"
        cohort_prior_tpm = 0.0
        tcga_tumor_fold = 0.0
        if (
            cancer_col in ref_dedup.columns
            and cancer_col in ref_hk_medians
            and ref_hk_medians[cancer_col] > 0
        ):
            cancer_hk_m = ref_hk_medians[cancer_col]
            tcga_fold = float(ref_dedup.loc[symbol, cancer_col]) / cancer_hk_m
            tcga_p = TCGA_MEDIAN_PURITY.get(cancer_code, 0.7)
            tcga_tumor_fold = max(
                0.0, (tcga_fold - (1 - tcga_p) * tme_fold_med) / tcga_p
            )
            cohort_prior_tpm = tcga_tumor_fold * sample_hk_median

        # Empirical-Bayes shrinkage weight. At high purity the sample-
        # based estimate is reliable (w_sample → 1). At low purity the
        # 1/p division in the deconvolution inflates noise; shrinkage
        # pulls estimates back toward the cohort prior (w_sample → 0
        # as purity → 0). `k_shrinkage` is the purity at which weights
        # are 50/50 — tuned so that CTAs and real tumor markers at
        # moderate purity (~0.4) are mostly sample-driven, while
        # low-purity stromal-like genes get anchored to cohort.
        #
        # `k_shrinkage` doubles for tme_explainable genes (to ~2×
        # stronger shrinkage), reflecting the extra uncertainty about
        # whether the signal is genuinely tumor-cell-derived.
        k_shrinkage = 0.20
        if tme_explainable:
            k_shrinkage = 0.40

        # Shrinkage floor: when the cohort prior is near-zero (e.g.
        # CTAs — activated in a minority of samples, so median = 0),
        # the prior mean is uninformative. Empirical-Bayes shrinkage
        # toward 0 would wrongly pull real sample-specific signal
        # down. Skip shrinkage in that regime and trust the sample.
        skip_shrinkage = cohort_prior_tpm < 1.0

        # 9-point estimate grid. TPM-space path (preferred) uses the
        # decomposition's per-gene expected TME contribution directly.
        # HK-fold path is the fallback when no decomposition is
        # available or the gene is absent from the reference. Every
        # grid point is shrunk toward the cohort prior and clamped at
        # `observed_tpm` when tme_explainable=True (tumor cells can't
        # contribute more than the observed signal if a single healthy
        # tissue could explain it alone).
        def _apply_priors(raw_tumor_tpm, purity_used):
            if skip_shrinkage:
                shrunk = raw_tumor_tpm
            else:
                w_sample = float(purity_used) / (float(purity_used) + k_shrinkage)
                shrunk = (
                    w_sample * raw_tumor_tpm
                    + (1.0 - w_sample) * cohort_prior_tpm
                )
            if tme_explainable:
                shrunk = min(shrunk, observed)
            return max(0.0, shrunk)

        estimates = []
        if tme_bg_tpm_by_symbol is not None and symbol in tme_bg_tpm_by_symbol:
            bg_tpm_mid = tme_bg_tpm_by_symbol[symbol]
            # Uncertainty on the TME estimate itself: ±50%. Reference
            # cell-type profiles may under- or over-estimate the actual
            # infiltrate composition in the specific sample.
            for bg_tpm in [bg_tpm_mid * 0.5, bg_tpm_mid, bg_tpm_mid * 1.5]:
                for p in [p_lo, p_med, p_hi]:
                    raw = max(0.0, (observed - bg_tpm)) / p
                    estimates.append(_apply_priors(raw, p))
        else:
            for bg in [tme_fold_lo, tme_fold_med, tme_fold_hi]:
                for p in [p_lo, p_med, p_hi]:
                    tumor_fold = max(0.0, (sample_fold - (1 - p) * bg) / p)
                    raw = tumor_fold * sample_hk_median
                    estimates.append(_apply_priors(raw, p))
        estimates.sort()
        median_est = float(np.median(estimates))

        ref_vals = cancer_expr_all.loc[symbol].values
        n = len(ref_vals)
        below = np.sum(ref_vals < median_est)
        equal = np.sum(np.isclose(ref_vals, median_est, atol=0.01))
        tcga_percentile = float((below + 0.5 * equal) / n)

        # Ratio vs purity-adjusted TCGA median for matched cancer type.
        # Uses the already-computed cohort tumor fold above.
        #
        # Three outcomes, each consumed by a distinct plot label:
        #   - finite positive → fold-change vs TCGA ("0.3x", "1.5x", ...)
        #   - inf             → sample expresses the gene but the TCGA
        #                       cohort tumor-component is essentially zero.
        #                       Rendered as a red "absent in TCGA" alert —
        #                       flags atypical expression for this cancer.
        #   - None            → both the sample tumor-component and TCGA
        #                       tumor-component are essentially zero.
        #                       Rendered as a gray "0 in TCGA" — nothing to
        #                       compare.
        #
        # `tcga_tumor_fold` clips to exactly 0.0 whenever the TCGA cohort
        # median is explainable by TME alone; previously the ≤ 0 branch
        # collapsed to None unconditionally, so a CTA with FPKM_<cancer>
        # median ≈ 0 and strong sample expression rendered as a quiet gray
        # label instead of the intended red "absent in TCGA" alert. The
        # sample-side check below restores the intended semantics.
        our_tumor_fold = median_est / sample_hk_median
        if tcga_tumor_fold > 0.001:
            vs_tcga = float(our_tumor_fold / tcga_tumor_fold)
        elif our_tumor_fold > 0.001:
            vs_tcga = float("inf")
        else:
            vs_tcga = None

        # Categorize
        is_cta = symbol in cta_symbols
        is_surface = symbol in surf_symbols
        therapies = gene_therapies.get(symbol, set())

        if is_cta:
            category = "CTA"
        elif therapies:
            category = "therapy_target"
        elif is_surface:
            category = "surface"
        else:
            category = "other"

        eid = ref_dedup.loc[symbol, "Ensembl_Gene_ID"] if "Ensembl_Gene_ID" in ref_dedup.columns else ""

        # Per-gene matched-normal split reporting (issue #50). Zero when
        # no matched-normal component is active or the gene isn't in the
        # signature matrix. `estimation_path` records which branch
        # produced the estimate, so the target report can annotate each
        # gene with its provenance.
        mn_tpm = 0.0
        if matched_normal_tpm_by_symbol is not None:
            mn_tpm = float(matched_normal_tpm_by_symbol.get(symbol, 0.0))
        if tme_only_tpm_by_symbol is not None:
            tme_only_tpm = float(tme_only_tpm_by_symbol.get(symbol, 0.0))
        elif tme_bg_tpm_by_symbol is not None:
            tme_only_tpm = float(tme_bg_tpm_by_symbol.get(symbol, 0.0))
        else:
            tme_only_tpm = 0.0

        if tme_explainable:
            estimation_path = "clamped"
        elif mn_tpm > 0.0:
            estimation_path = "matched_normal_split"
        elif tme_bg_tpm_by_symbol is not None and symbol in tme_bg_tpm_by_symbol:
            estimation_path = "tme_only"
        else:
            estimation_path = "tme_fold_fallback"

        rows.append({
            "gene_id": eid,
            "symbol": symbol,
            "category": category,
            "observed_tpm": round(observed, 2),
            "tme_fold_lo": round(tme_fold_lo, 4),
            "tme_fold_med": round(tme_fold_med, 4),
            "tme_fold_hi": round(tme_fold_hi, 4),
            "max_healthy_tpm": round(max_healthy_tpm, 2),
            "tme_explainable": bool(tme_explainable),
            "cohort_prior_tpm": round(cohort_prior_tpm, 2),
            "tme_only_tpm": round(tme_only_tpm, 2),
            "matched_normal_tpm": round(mn_tpm, 2),
            "matched_normal_tissue": matched_normal_tissue or "",
            "matched_normal_fraction": round(matched_normal_fraction_global, 4),
            "estimation_path": estimation_path,
            **{f"est_{i+1}": round(estimates[i], 2) for i in range(9)},
            "median_est": round(median_est, 2),
            "pct_cancer_median": round(vs_tcga, 2) if vs_tcga is not None else None,
            "tcga_percentile": round(tcga_percentile, 3),
            "is_surface": is_surface,
            "is_cta": is_cta,
            "therapies": ", ".join(sorted(therapies)) if therapies else "",
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("median_est", ascending=False).reset_index(drop=True)
    return result


def plot_matched_normal_attribution(
    df_ranges,
    cancer_type,
    category,
    top_n=15,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Stacked horizontal-bar plot of per-gene tumor / matched-normal / TME
    attribution for a single target category (issue #55).

    For each of the top ``top_n`` genes in ``category`` (ranked by
    ``median_est``), draw a horizontal bar broken into:

    - ``tumor``: ``observed_tpm − matched_normal_tpm − tme_only_tpm``
    - ``matched_normal_tpm``: benign parent-tissue contribution
    - ``tme_only_tpm``: stromal / immune / host-tissue contribution

    Only useful when ``df_ranges`` carries non-zero ``matched_normal_tpm``
    for at least one gene in the category (i.e. the decomposition ran
    with ``use_matched_normal=True`` for an epithelial primary). Returns
    ``None`` otherwise — the CLI uses that to skip emitting an empty
    figure.

    Emitted as a standalone PNG (one per category) rather than as a
    panel in a composite figure, following the project's plot-crowding
    preference.
    """
    import numpy as np

    cancer_code = resolve_cancer_type(cancer_type)

    sub = df_ranges[df_ranges["category"] == category].head(top_n).copy()
    if sub.empty:
        return None
    if "matched_normal_tpm" not in sub.columns:
        return None
    if (sub["matched_normal_tpm"].astype(float) <= 0).all():
        return None

    sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)
    n = len(sub)
    if figsize is None:
        figsize = (10, max(3.0, 0.4 * n + 1.5))

    observed = sub["observed_tpm"].astype(float).values
    mn = sub["matched_normal_tpm"].astype(float).values
    tme = sub["tme_only_tpm"].astype(float).values
    # Tumor-cell attribution is whatever observed signal remains after
    # TME and matched-normal are subtracted. Floor at 0 to guard against
    # tiny over-subtractions from solver jitter.
    tumor_attr = np.maximum(0.0, observed - mn - tme)

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, tumor_attr, color="#e74c3c", label="tumor cells")
    ax.barh(y, mn, left=tumor_attr, color="#3498db", label="matched-normal tissue")
    ax.barh(y, tme, left=tumor_attr + mn, color="#95a5a6", label="other TME (stromal/immune)")

    # Symbols on the y-axis. Therapy-target annotation appended when present.
    labels = []
    for _, row in sub.iterrows():
        sym = str(row["symbol"])
        if row.get("therapies"):
            sym = f"{sym}  [{row['therapies']}]"
        flags = []
        if row.get("tme_explainable"):
            flags.append("⚠")
        path = str(row.get("estimation_path", ""))
        if path == "clamped":
            flags.append("clamp")
        if flags:
            sym = f"{sym}  {' '.join(flags)}"
        labels.append(sym)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    # Cohort-prior marker: small tick on each bar so reviewers can see
    # where the TCGA cohort would have placed tumor-cell expression.
    if "cohort_prior_tpm" in sub.columns:
        priors = sub["cohort_prior_tpm"].astype(float).values
        for i, prior in enumerate(priors):
            if prior > 0:
                ax.plot([prior], [i], marker="|", color="black", markersize=14, markeredgewidth=2)

    ax.set_xlabel("TPM (stacked: tumor + matched-normal + other TME = observed)", fontsize=10)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    mn_tissue = ""
    if "matched_normal_tissue" in sub.columns:
        nonempty = sub["matched_normal_tissue"].astype(str).replace("nan", "")
        nonempty = [v for v in nonempty.unique() if v]
        if nonempty:
            mn_tissue = nonempty[0]
    title = f"Matched-normal attribution — {cancer_code} {category}"
    if mn_tissue:
        title += f"\n(benign {mn_tissue} subtracted before purity division; black tick = TCGA cohort prior)"
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_tumor_expression_ranges(
    df_ranges,
    purity_result,
    cancer_type,
    top_n=15,
    categories=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Strip plot of 9-point tumor expression estimates per gene.

    Parameters
    ----------
    df_ranges : DataFrame
        Output of ``estimate_tumor_expression_ranges()``.
    purity_result : dict
        Output of ``estimate_tumor_purity()``.
    cancer_type : str
        Cancer type code for title.
    top_n : int
        Max genes per category panel.
    categories : list of str, optional
        Which categories to plot. Default: CTA, therapy_target, surface.
    save_to_filename : str, optional
        Path to save the figure.
    """
    import numpy as np

    if categories is None:
        categories = ["therapy_target", "CTA", "surface"]

    cancer_code = resolve_cancer_type(cancer_type)
    p_lo = max(purity_result.get("overall_lower") or 0.01, 0.01)
    p_med = max(purity_result.get("overall_estimate") or 0.05, 0.01)
    p_hi = max(purity_result.get("overall_upper") or p_med, 0.01)

    cat_titles = {
        "CTA": "Cancer-Testis Antigens",
        "therapy_target": "Therapeutic Targets",
        "surface": "Surface Proteins",
        "other": "Other Tumor Genes",
    }
    cat_colors = {
        "CTA": "#e74c3c",
        "therapy_target": "#3498db",
        "surface": "#2ecc71",
        "other": "#95a5a6",
    }
    # Count genes per panel to size adaptively
    panel_counts = []
    for cat in categories:
        n = min(top_n, len(df_ranges[df_ranges["category"] == cat]))
        panel_counts.append(max(n, 1))
    total_genes = sum(panel_counts)

    n_panels = len(categories)
    if figsize is None:
        # ~0.4 inches per gene row, minimum 2 inches per panel
        panel_heights = [max(2.0, 0.4 * n) for n in panel_counts]
        figsize = (14, sum(panel_heights) + 1.5)

    fig, axes = plt.subplots(
        n_panels, 2, figsize=figsize, squeeze=False,
        gridspec_kw={
            "width_ratios": [3, 1],
            "height_ratios": panel_counts,
        },
    )
    est_cols = [f"est_{i+1}" for i in range(9)]

    # Adaptive font/marker sizes: larger when fewer genes
    base_font = min(12, max(8, int(200 / max(total_genes, 1))))
    marker_s = min(80, max(30, int(600 / max(total_genes, 1))))
    diamond_s = min(120, max(50, int(900 / max(total_genes, 1))))

    for ax_idx, cat in enumerate(categories):
        ax_strip = axes[ax_idx, 0]
        ax_pct = axes[ax_idx, 1]
        sub = df_ranges[df_ranges["category"] == cat].head(top_n).copy()
        sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)

        if sub.empty:
            ax_strip.set_title(cat_titles.get(cat, cat))
            ax_strip.text(0.5, 0.5, "No genes", ha="center", va="center",
                          transform=ax_strip.transAxes, fontsize=base_font, color="gray")
            ax_pct.set_visible(False)
            continue

        color = cat_colors.get(cat, "#95a5a6")
        y_positions = np.arange(len(sub))

        # --- Left panel: 9-point strip plot ---
        for i, (_, row) in enumerate(sub.iterrows()):
            vals = [row[c] for c in est_cols]
            vals_plot = [max(v, 0.01) for v in vals]
            median_v = max(row["median_est"], 0.01)

            ax_strip.scatter(vals_plot, [i] * 9, color=color, alpha=0.4,
                             s=marker_s, zorder=3)
            ax_strip.scatter([median_v], [i], color=color, marker="D",
                             s=diamond_s, edgecolors="black", linewidths=0.7,
                             zorder=5)
            ax_strip.plot([min(vals_plot), max(vals_plot)], [i, i],
                          color=color, alpha=0.3, linewidth=2.5, zorder=2)

        labels = []
        for _, row in sub.iterrows():
            label = row["symbol"]
            if row["therapies"]:
                label += f"  [{row['therapies']}]"
            labels.append(label)

        ax_strip.set_yticks(y_positions)
        ax_strip.set_yticklabels(labels, fontsize=base_font)
        ax_strip.set_xscale("log")
        ax_strip.set_xlabel("Tumor-specific expression (TPM)", fontsize=base_font)
        ax_strip.set_title(cat_titles.get(cat, cat), fontsize=base_font + 2,
                           fontweight="bold", color=color)
        ax_strip.set_ylim(-0.5, len(sub) - 0.5)
        ax_strip.grid(axis="x", alpha=0.2)

        # --- Right panel: % of cancer type median (log scale) ---
        for i, (_, row) in enumerate(sub.iterrows()):
            pct = row.get("pct_cancer_median")
            if pct is None or (isinstance(pct, float) and np.isnan(pct)):
                # Gene not in TCGA reference for this cancer type
                ax_pct.text(0.5, i, "0 in TCGA", fontsize=base_font - 2,
                            color="gray", ha="center", va="center",
                            transform=ax_pct.get_yaxis_transform())
                continue
            if isinstance(pct, float) and np.isinf(pct):
                # Sample expresses the gene but the TCGA cohort reference is
                # zero — so this gene is absent from TCGA {cancer_code} but
                # present in THIS sample. Draw a solid dark-red band across
                # the row with white text inside so the reader can't mistake
                # it for a tissue-decomposition claim or a general property
                # of the cancer type.
                ax_pct.axhspan(i - 0.35, i + 0.35, color="#6b0000",
                               alpha=1.0, zorder=3, linewidth=0)
                ax_pct.text(0.5, i, f"absent in TCGA {cancer_code}",
                            fontsize=base_font - 2, color="white",
                            fontweight="bold", ha="center", va="center",
                            zorder=4,
                            transform=ax_pct.get_yaxis_transform())
                continue
            bar_color = color if pct >= 0.5 else "#d4a017"
            ax_pct.barh(i, max(pct, 0.001), color=bar_color, alpha=0.7, height=0.6)
            lbl = f"{pct:.1f}x" if pct < 10 else f"{pct:.0f}x"
            ax_pct.text(max(pct, 0.001) * 1.2, i, lbl, fontsize=base_font - 1,
                        va="center", color="black")

        ax_pct.set_xscale("log")
        ax_pct.axvline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax_pct.set_yticks([])
        ax_pct.set_xlabel(f"vs {cancer_code} median", fontsize=base_font)
        ax_pct.set_title(f"vs {cancer_code}", fontsize=base_font, color="gray")
        ax_pct.set_ylim(-0.5, len(sub) - 0.5)
        ax_pct.grid(axis="x", alpha=0.15)

    # Suptitle with purity info and caveat
    fig.suptitle(
        f"Purity-adjusted tumor expression — {cancer_code}\n"
        f"Purity: {p_lo:.0%} / {p_med:.0%} / {p_hi:.0%} (low / est / high)\n"
        f"Values are deconvolved estimates — may overstate expression at low purity",
        fontsize=10, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")

    return fig


def plot_purity_adjusted_targets(
    df_gene_expr,
    cancer_type,
    purity,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 10),
    top_n=40,
):
    """Plot purity-adjusted tumor expression for key gene categories.

    Shows observed vs purity-adjusted expression for CTAs, therapy
    targets, and surface proteins, with TCGA percentile context.
    """
    import numpy as np
    import pandas as pd

    adj = estimate_tumor_expression(df_gene_expr, cancer_type, purity)
    cancer_code = resolve_cancer_type(cancer_type)

    # Select top genes per category
    categories = ["CTA", "therapy_target", "surface"]
    selected = []
    for cat in categories:
        sub = adj[adj["category"] == cat].head(top_n // len(categories))
        selected.append(sub)
    selected = pd.concat(selected, ignore_index=True) if selected else adj.head(0)
    # Add high-expression "other" if space remains
    remaining = top_n - len(selected)
    if remaining > 0:
        other = adj[(adj["category"] == "other") & (adj["tumor_adjusted"] > 10)]
        selected = pd.concat([selected, other.head(remaining)], ignore_index=True)

    if selected.empty:
        return None

    selected = selected.sort_values(
        ["category", "tumor_adjusted"], ascending=[True, False]
    ).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Left: horizontal bar chart of purity-adjusted expression
    y = np.arange(len(selected))
    cat_colors = {
        "CTA": "#e74c3c",
        "therapy_target": "#3498db",
        "surface": "#2ecc71",
        "other": "#95a5a6",
    }
    colors = [cat_colors.get(c, "#95a5a6") for c in selected["category"]]

    ax1.barh(y, selected["tumor_adjusted"], color=colors, alpha=0.8, height=0.7)
    # Overlay observed as dots
    ax1.scatter(
        selected["observed_tpm"], y,
        color="black", s=20, zorder=5, label="observed TPM"
    )
    ax1.set_yticks(y)
    labels = []
    for _, row in selected.iterrows():
        suffix = ""
        if row["is_surface"]:
            suffix += " [S]"
        if row["therapies"]:
            suffix += f" ({row['therapies']})"
        labels.append(f"{row['symbol']}{suffix}")
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Expression (TPM)")
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_title(f"Purity-adjusted tumor expression — {cancer_code} (purity={purity:.0%})")
    ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc="lower right")

    # Right: TCGA percentile heatmap
    pctiles = selected["tcga_percentile"].values
    ax2.barh(y, pctiles, color=colors, alpha=0.8, height=0.7)
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("TCGA percentile")
    ax2.set_title("vs TCGA cancer types")
    ax2.invert_yaxis()

    # Category legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="CTA (vaccination target)"),
        Patch(facecolor="#3498db", label="Therapy target (in trials)"),
        Patch(facecolor="#2ecc71", label="Surface protein"),
        Patch(facecolor="#95a5a6", label="Other tumor gene"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")

    return fig


_signature_panel_cache = {}


def _get_cancer_type_signature_panels(n_signature_genes=20):
    """Return robust per-cancer signature panels used for ranking and plotting.

    Reuses `_select_tumor_specific_genes_for_panel` from `tumor_purity` so
    the plotting / ranking paths see the same tier-filtered, family-excluded
    panels as the purity estimator. When the strict tiers leave a cancer
    type short of `n_signature_genes`, a z-score fallback is appended — but
    that fallback still respects the configured family-exclusion regex and
    the `immune_origin_cancer_types` bypass, so the two paths never silently
    diverge on borderline cases.

    Cache is keyed on a fingerprint of TUMOR_PURITY_PARAMETERS so tuning
    parameters in-process invalidates stale panels.
    """
    from .tumor_purity import (
        TUMOR_PURITY_PARAMETERS,
        _cached_reference_matrices,
        _compile_excluded_gene_matcher,
        _params_fingerprint,
        _select_tumor_specific_genes_for_panel,
    )

    params_fp = _params_fingerprint(["tumor_specific_markers"])
    cache_key = (int(n_signature_genes), params_fp)
    cached = _signature_panel_cache.get(cache_key)
    if cached is not None:
        return {code: list(genes) for code, genes in cached.items()}

    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    fpkm_cols = ref_matrices["fpkm_cols"]
    expr_matrix = ref_matrices["expr_matrix"]
    z_matrix = ref_matrices["z_matrix"]

    is_excluded_default = _compile_excluded_gene_matcher()
    immune_origin = set(
        TUMOR_PURITY_PARAMETERS["tumor_specific_markers"].get(
            "immune_origin_cancer_types", []
        )
        or []
    )

    panels = {}
    for col in fpkm_cols:
        code = col.replace("FPKM_", "")
        genes = _select_tumor_specific_genes_for_panel(
            code,
            n=n_signature_genes,
            exclude_lineage=False,
        )
        if len(genes) >= n_signature_genes:
            panels[code] = genes[:n_signature_genes]
            continue

        # Fallback top-up: raw top-N-by-z-score, but still respecting the
        # configured family exclusion so the plotting panel can't silently
        # pick up MT-* / rearranged-receptor genes that the purity path
        # just filtered out. Immune-origin codes skip the filter for the
        # same reason the purity path does.
        is_excluded = (
            (lambda _sym: False) if code in immune_origin else is_excluded_default
        )
        z_col = z_matrix[col]
        expr_col = expr_matrix[col]
        fallback_candidates = z_col[expr_col > 0.01].nlargest(n_signature_genes * 3)
        for gene in fallback_candidates.index:
            if gene in genes:
                continue
            if is_excluded(gene):
                continue
            genes.append(gene)
            if len(genes) >= n_signature_genes:
                break
        panels[code] = genes[:n_signature_genes]

    _signature_panel_cache[cache_key] = {
        code: tuple(genes) for code, genes in panels.items()
    }
    return panels


def _compute_cancer_type_signature_stats(
    df_gene_expr,
    n_signature_genes=20,
    min_fold=2.0,
):
    """Score each cancer type by how well the sample matches its signature genes.

    Uses z-score–based gene selection (most specifically expressed genes per
    cancer type) and midrank percentile scoring — the sample's expression of
    each signature gene is ranked against the cross-cancer distribution.
    This is robust to TPM-vs-FPKM scale differences.
    """
    import numpy as np

    from .tumor_purity import _cached_reference_matrices

    sample_raw_by_symbol, sample_hk_by_symbol = _sample_expression_by_symbol(df_gene_expr)
    # HK-normalize both sides so percentile comparison is on the same
    # scale (sample TPM/hk vs reference FPKM/hk). This is consistent
    # normalization, not mixed — both are divided by their own HK median.
    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    ref_by_sym = ref_matrices["ref_by_sym"]
    expr_matrix = ref_matrices["expr_matrix"]
    sig = _get_cancer_type_signature_panels(n_signature_genes=n_signature_genes)

    stats = []
    for code in sorted(sig.keys()):
        genes = sig[code]
        cohort_col = f"FPKM_{code}"
        gene_details = []
        percentiles = []
        for gene in genes:
            sample_raw = float(sample_raw_by_symbol.get(gene, 0.0))
            sample_hk = float(sample_hk_by_symbol.get(gene, 0.0))
            cohort_hk = 0.0
            percentile = 0.5
            if gene in ref_by_sym.index and cohort_col in ref_by_sym.columns:
                cohort_hk = float(ref_by_sym.loc[gene, cohort_col])
                # Midrank percentile: robust to ties at zero
                ref_vals = expr_matrix.loc[gene].values
                n = len(ref_vals)
                below = np.sum(ref_vals < sample_hk)
                equal = np.sum(np.isclose(ref_vals, sample_hk, atol=1e-6))
                percentile = float((below + 0.5 * equal) / n)
            percentiles.append(percentile)
            log_diff = abs(np.log2(sample_hk + 1) - np.log2(cohort_hk + 1))
            gene_details.append(
                {
                    "gene": gene,
                    "sample_raw": sample_raw,
                    "sample_hk": sample_hk,
                    "cohort_hk": cohort_hk,
                    "log_diff": log_diff,
                    "percentile": percentile,
                }
            )

        if gene_details:
            score = float(np.mean(percentiles))
            mean_sample_raw = float(np.mean([g["sample_raw"] for g in gene_details]))
        else:
            score = 0.0
            mean_sample_raw = 0.0

        stats.append(
            {
                "code": code,
                "genes": genes,
                "n_genes": len(genes),
                "score": score,
                "mean_sample_raw": mean_sample_raw,
                "gene_details": gene_details,
            }
        )

    stats.sort(key=lambda row: (-row["score"], row["code"]))
    for rank, row in enumerate(stats, start=1):
        row["rank"] = rank
    return stats


# ---------------------------------------------------------------------------
# Unified embedding gene selection
# ---------------------------------------------------------------------------

_REPRODUCTIVE_TISSUES = {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}
_IG_TR_PREFIXES = ("IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD")
_CURATED_CTAS = [
    "PRAME",    # UCS/SKCM/UCEC — high-expression melanoma/uterine marker
    "MAGEA3",   # SKCM/LUSC — melanoma and squamous cancers
    "CTCFL",    # UCS/OV — gynecologic cancers (BORIS, CTCF paralog)
    "SMC1B",    # CESC/LAML — meiotic cohesin, cervical cancer marker
    "LIN28B",   # TGCT/UCS — stem cell / embryonal marker
    "SSX1",     # THCA/UVM/SKCM — thyroid + melanomas
    "C1orf94",  # LGG/GBM — brain tumor CTA marker
    "SYCP3",    # LAML/TGCT — synaptonemal complex, meiosis marker
    "FATE1",    # ACC — adrenocortical CTA marker
]

_STROMAL_TISSUES = {"smooth_muscle", "skeletal_muscle", "heart_muscle", "adipose_tissue"}

# Immune/lymphoid tissues that represent TME infiltrate.  Curated to
# exclude epithelial organs that merely contain resident immune cells
# (which would inflate the TME background estimate).
_IMMUNE_TISSUES = {
    "bone_marrow", "lymph_node", "spleen", "thymus", "tonsil", "appendix",
}

_TME_TISSUES = _STROMAL_TISSUES | _IMMUNE_TISSUES

# Clinically important genes with low TME background.  The data-driven
# algorithm may miss these because their best z-score peaks in a sibling
# type or they sit just below per-type selection cutoffs.
_CURATED_TME_BOOST = [
    # Cancer-testis antigens
    "PRAME",      # S/N_tme≈179  UCS/UCEC/SKCM — immunotherapy target
    "MAGEA3",     # S/N_tme≈27   SKCM — vaccine target
    "LIN28B",     # S/N_tme≈19   TGCT — embryonal/stem marker
    "SYCP3",      # S/N_tme≈12   LAML — meiosis marker
    # Glioma / neuroendocrine (brain-restricted, TME-silent)
    "DLL3",       # S/N_tme≈94   LGG/GBM — Rova-T & BiTE target
    "PTPRZ1",     # S/N_tme≈29   LGG/GBM — glioma phosphatase
    # Melanocyte lineage (melanocyte-restricted)
    "MLANA",      # S/N_tme≈1472 UVM/SKCM — MART-1, TIL/TCR target
    "TYR",        # S/N_tme≈726  UVM/SKCM — tyrosinase
    # Therapy targets with low TME expression
    "MSLN",       # S/N_tme≈10   MESO/OV — mesothelin, CAR-T target
    "CDKN2A",     # S/N_tme≈13   UCS/OV — p16, broad tumor marker
    "COL11A1",    # S/N_tme≈11   MESO/BRCA/PAAD — desmoplastic, ADC target
    # Lineage transcription factors
    "FOXA1",      # S/N_tme≈5    PRAD/BRCA — luminal breast/prostate
    "ASCL2",      # S/N_tme≈10   COAD/READ — intestinal stem cell TF
    "DLX5",       # S/N_tme≈9    UCEC/UCS — homeobox, endometrial
    "SOX2",       # S/N_tme≈22   LUSC/LGG — squamous & neural stem cell TF
    "TH",         # S/N_tme≈792  PCPG — tyrosine hydroxylase, catecholamine
    # Therapy targets & lineage markers for additional types
    "FLT3",       # S/N_tme≈47   LAML — gilteritinib/midostaurin target
    "LIN28A",     # S/N_tme≈319  TGCT — embryonal pluripotency marker
    "NANOG",      # S/N_tme≈111  TGCT — pluripotency TF, germ cell tumors
    "PMEL",       # S/N_tme≈104  UVM/SKCM — gp100, melanoma vaccine target
    "OLIG2",      # S/N_tme≈10   LGG/GBM — oligodendrocyte lineage TF
    "NKX3-1",     # S/N_tme≈6    PRAD — prostate lineage TF, diagnostic
    "STEAP2",     # S/N_tme≈4    PRAD — prostate surface antigen
    "MITF",       # S/N_tme≈4    UVM/SKCM — master melanocyte TF
    "FOXN1",      # S/N_tme≈8    THYM — thymic epithelial TF
]

# Lineage markers for cancer types whose defining genes are also expressed
# in normal tissue (high TME background).  NOT TME-low — these won't help
# at very low purity, but they prevent these types from collapsing into a
# featureless cluster in embedding space.
_CURATED_LINEAGE_BOOST = [
    "UPK2",       # BLCA — uroplakin, 120x vs other cancers
    "APOA2",      # LIHC — liver secretory protein, 70000x vs others
    "SFTPB",      # LUAD — surfactant protein B, 12000x vs others
    "NAPSA",      # LUAD — napsin A, lung adeno IHC marker
    "KRT6A",      # ESCA — squamous keratin, 214x vs others
    "PGC",        # STAD — pepsinogen C, best available (3x vs others)
]

_embedding_gene_cache = {}
_tme_gene_cache = {}
_bottleneck_gene_cache = {}
_hierarchy_feature_cache = {}
_hierarchy_site_cache = {}


def _select_embedding_genes_bottleneck(n_genes_per_type=5):
    """Select genes for cancer-type embedding using bottleneck scoring.

    For each gene × cancer type, two z-scores are computed:

    * **z_tme** — how far the gene's cancer-type expression is above
      the distribution of TME (immune + stromal) tissue expression.
      High z_tme ⇒ gene visible above microenvironment background.

    * **z_other** — how far the gene's cancer-type expression is above
      the distribution of *all* cancer types.
      High z_other ⇒ gene is specific to this cancer type.

    The combined score is ``min(z_tme, z_other)`` — the bottleneck.
    A gene ranks high only if it scores well on *both* axes.  This
    naturally balances purity robustness (TME silence) against
    cancer-type discrimination without any hard threshold on either.

    Evaluation on 160 individual TCGA samples diluted 1:1 with GTEx
    immune expression (simulating 5% tumor purity) showed this method
    achieves 56% top-1 / 76% top-5 nearest-neighbor accuracy with only
    158 genes — the best purity-robust performance tested.  See
    ``eval/`` for the full comparison across 21 gene sets, 9
    normalizations, and 5 purity levels.

    Parameters
    ----------
    n_genes_per_type : int
        Number of top-scoring genes to select per cancer type (default 5).

    Returns
    -------
    ref_filtered : DataFrame
        Subset of pan-cancer reference for the selected genes.
    metadata : dict
        Per-type gene lists, total gene count, etc.
    """
    import numpy as np

    cache_key = n_genes_per_type
    if cache_key in _bottleneck_gene_cache:
        return _bottleneck_gene_cache[cache_key]

    ref = pan_cancer_expression()
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    ref_dedup = ref.drop_duplicates(subset="Symbol")
    cancer_expr = ref_dedup[fpkm_cols].astype(float)

    # Identify TME tissues (immune via PTPRC, plus stromal)
    ptprc_row = ref_dedup[ref_dedup["Symbol"] == "PTPRC"]
    if len(ptprc_row):
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float).iloc[0]
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro
        if c.replace("nTPM_", "") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))
    tme_expr = ref_dedup[tme_cols].astype(float)

    # IG/TR exclusion
    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(s.startswith(p) for p in _IG_TR_PREFIXES)
    )

    # Log-transform
    log_cancer = np.log2(cancer_expr + 1)
    log_tme = np.log2(tme_expr + 1)

    # z_tme: z-score of cancer expr against TME tissue distribution
    tme_mean = log_tme.mean(axis=1)
    tme_std = log_tme.std(axis=1).replace(0, 0.1)
    z_tme = log_cancer.sub(tme_mean.values, axis=0).div(tme_std.values, axis=0)

    # z_other: z-score of cancer expr against all cancer types
    cancer_mean = log_cancer.mean(axis=1)
    cancer_std = log_cancer.std(axis=1).replace(0, 0.1)
    z_other = log_cancer.sub(cancer_mean.values, axis=0).div(cancer_std.values, axis=0)

    # Bottleneck score: min of the two positive z-scores
    z_tme_pos = z_tme.clip(lower=0)
    z_other_pos = z_other.clip(lower=0)
    import pandas as pd
    bottleneck = pd.DataFrame(
        np.minimum(z_tme_pos.values, z_other_pos.values),
        index=cancer_expr.index,
        columns=cancer_expr.columns,
    )

    # Select top genes per type
    selected_idx = []
    per_type = {}
    for col in fpkm_cols:
        code = col.replace("FPKM_", "")
        mask = (cancer_expr[col] > 0.5) & (~is_rearranged.values)
        valid = bottleneck[col][mask]
        top = valid.nlargest(n_genes_per_type)
        syms = list(ref_dedup.loc[top.index, "Symbol"].values)
        per_type[code] = syms
        selected_idx.extend(top.index)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "per_type": per_type,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, g in per_type.items() if g]),
        "method": "bottleneck",
        "tme_tissues": sorted(c.replace("nTPM_", "") for c in tme_cols),
    }

    result = (ref_filtered, metadata)
    _bottleneck_gene_cache[cache_key] = result
    return result


def _select_tme_low_genes(n_genes_per_type=3, sn_tme_threshold=10):
    """Select genes with low tumor microenvironment (TME) background.

    These genes are silent in immune and stromal cells, so their signal
    is detectable even at very low tumor purity (down to ~5%).

    Parameters
    ----------
    sn_tme_threshold : float
        Minimum ratio of cancer expression to TME tissue expression.
        Default 10 means genes are visible at ~10% purity.
    """
    import numpy as np

    cache_key = (n_genes_per_type, sn_tme_threshold)
    if cache_key in _tme_gene_cache:
        return _tme_gene_cache[cache_key]

    ref = pan_cancer_expression()
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    ref_dedup = ref.drop_duplicates(subset="Symbol")
    cancer_expr = ref_dedup[fpkm_cols].astype(float)
    normal_expr = ref_dedup[ntpm_nonrepro].astype(float)

    # Immune tissues (PTPRC-defined) + stromal tissues = TME background
    ptprc_row = ref_dedup[ref_dedup["Symbol"] == "PTPRC"]
    if len(ptprc_row):
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float).iloc[0]
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro
        if c.replace("nTPM_", "") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))
    tme_max = normal_expr[tme_cols].max(axis=1) if tme_cols else normal_expr.max(axis=1)

    sn_tme = cancer_expr.max(axis=1) / (tme_max + 0.01)

    # IG/TR exclusion
    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(s.startswith(p) for p in _IG_TR_PREFIXES)
    )

    # Z-scores for cancer-type specificity
    g_mean = cancer_expr.mean(axis=1)
    g_std_raw = cancer_expr.std(axis=1).replace(0, np.nan)
    z_mat = cancer_expr.sub(g_mean, axis=0).div(g_std_raw, axis=0).fillna(0)
    best_z = z_mat.max(axis=1)

    base_mask = (~is_rearranged.values) & (cancer_expr.max(axis=1) > 1) & (best_z > 1)

    # Tiered selection: strict S/N first, then relax for underrepresented types
    selected_idx = []
    per_type = {}
    covered = set()

    for tier_thresh in [sn_tme_threshold, 3, 1.5]:
        tier_mask = base_mask & (sn_tme > tier_thresh)
        tier_best_cancer = z_mat[tier_mask].idxmax(axis=1)
        for code_col in fpkm_cols:
            code = code_col.replace("FPKM_", "")
            if code in covered:
                continue
            genes = tier_best_cancer[tier_best_cancer == code_col].index
            top = best_z.loc[genes].nlargest(n_genes_per_type).index
            if len(top):
                syms = list(ref_dedup.loc[top, "Symbol"].values)
                per_type[code] = syms
                selected_idx.extend(top)
                covered.add(code)

    # Final fallback: composite score for any still-missing types
    fallback_mask = base_mask & (sn_tme > 0.5)
    if fallback_mask.any():
        fallback_score = best_z[fallback_mask] * np.log2(
            cancer_expr.max(axis=1)[fallback_mask] + 1
        ) * np.minimum(sn_tme[fallback_mask], 3) / 3
        fallback_best = z_mat[fallback_mask].idxmax(axis=1)
        for code_col in fpkm_cols:
            code = code_col.replace("FPKM_", "")
            if code in covered:
                continue
            genes = fallback_best[fallback_best == code_col].index
            top = fallback_score.loc[genes].nlargest(n_genes_per_type).index
            syms = list(ref_dedup.loc[top, "Symbol"].values) if len(top) else []
            per_type[code] = syms
            selected_idx.extend(top)
            covered.add(code)

    # Fill any remaining types with empty
    for code_col in fpkm_cols:
        code = code_col.replace("FPKM_", "")
        if code not in per_type:
            per_type[code] = []

    selected_idx = list(dict.fromkeys(selected_idx))

    # --- Curated boost: clinically important TME-low markers ---
    selected_syms = set(ref_dedup.loc[selected_idx, "Symbol"].values)
    boost_added = []
    for sym in _CURATED_TME_BOOST:
        if sym in selected_syms:
            continue
        hits = ref_dedup[ref_dedup["Symbol"] == sym]
        if hits.empty:
            continue
        idx = hits.index[0]
        gene_sn = sn_tme.loc[idx]
        gene_expr = cancer_expr.loc[idx].max()
        if gene_sn > 3 and gene_expr > 1:
            selected_idx.append(idx)
            selected_syms.add(sym)
            boost_added.append(sym)

    # --- Lineage boost: high-discrimination markers for types without TME-low genes ---
    lineage_added = []
    for sym in _CURATED_LINEAGE_BOOST:
        if sym in selected_syms:
            continue
        hits = ref_dedup[ref_dedup["Symbol"] == sym]
        if hits.empty:
            continue
        idx = hits.index[0]
        gene_expr = cancer_expr.loc[idx].max()
        if gene_expr > 5:
            selected_idx.append(idx)
            selected_syms.add(sym)
            lineage_added.append(sym)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "per_type": per_type,
        "boost_added": boost_added,
        "lineage_added": lineage_added,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, g in per_type.items() if g]),
        "sn_tme_threshold": sn_tme_threshold,
        "tme_tissues": sorted(c.replace("nTPM_", "") for c in tme_cols),
    }

    result = (ref_filtered, metadata)
    _tme_gene_cache[cache_key] = result
    return result


def _select_embedding_genes(n_genes_per_type=3):
    """Select a unified gene set for cancer-type embeddings.

    Applies biologically-informed filters to select genes that discriminate
    cancer types without being confounded by immune infiltrate or normal
    tissue contamination.

    Returns
    -------
    ref_filtered : DataFrame
        Subset of pan-cancer reference data for the selected genes.
    metadata : dict
        Per-cancer-type gene lists, excluded genes, CTA additions, etc.
    """
    import numpy as np

    cache_key = n_genes_per_type
    if cache_key in _embedding_gene_cache:
        return _embedding_gene_cache[cache_key]

    ref = pan_cancer_expression()
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]

    # Exclude reproductive tissues from the normal-tissue denominator
    # so that cancer-testis antigens can pass the S/N filter.
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    ref_dedup = ref.drop_duplicates(subset="Symbol")
    cancer_expr = ref_dedup[fpkm_cols].astype(float)
    normal_expr = ref_dedup[ntpm_nonrepro].astype(float)
    normal_max = normal_expr.max(axis=1)

    # --- Data-driven immune tissue identification via PTPRC (CD45) ---
    ptprc_row = ref_dedup[ref_dedup["Symbol"] == "PTPRC"]
    if len(ptprc_row):
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float).iloc[0]
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    immune_sum = normal_expr[immune_cols].sum(axis=1) if immune_cols else 0
    total_sum = normal_expr.sum(axis=1)
    immune_frac = np.where(total_sum > 0.01, immune_sum / total_sum, 0.0)

    # --- Exclusion masks ---
    is_immune = (immune_frac > 0.5) & (total_sum > 10)
    is_rearranged = ref_dedup["Symbol"].apply(
        lambda s: any(s.startswith(p) for p in _IG_TR_PREFIXES)
    )
    excluded = is_immune | is_rearranged.values

    # --- Z-scores and S/N ---
    g_mean = cancer_expr.mean(axis=1)
    g_std_raw = cancer_expr.std(axis=1).replace(0, np.nan)
    z_mat = cancer_expr.sub(g_mean, axis=0).div(g_std_raw, axis=0).fillna(0)
    best_z = z_mat.max(axis=1)
    sn = cancer_expr.max(axis=1) / (normal_max + 0.01)

    # --- Primary selection: S/N pathway ---
    primary_mask = (
        (best_z > 1)
        & (sn > 3)
        & (cancer_expr.max(axis=1) > 0.1)
        & (normal_max >= 0.5)
        & ~excluded
    )
    primary_best = z_mat[primary_mask].idxmax(axis=1)

    selected_idx = []
    per_type = {}
    for code_col in fpkm_cols:
        code = code_col.replace("FPKM_", "")
        code_genes = primary_best[primary_best == code_col].index
        top = best_z.loc[code_genes].nlargest(n_genes_per_type).index
        syms = list(ref_dedup.loc[top, "Symbol"].values)
        per_type[code] = syms
        selected_idx.extend(top)

    # --- Fallback for underrepresented cancer types ---
    # Use a composite score: z-score × log2(expr+1) × min(S/N, 3)/3
    # This favors genes that are type-specific (z), reasonably expressed
    # (log-expr), and have at least partial cancer vs. normal enrichment (S/N).
    fallback_types = []
    fallback_mask = (best_z > 1) & (cancer_expr.max(axis=1) > 0.1) & ~excluded
    for code_col in fpkm_cols:
        code = code_col.replace("FPKM_", "")
        if len(per_type.get(code, [])) >= 2:
            continue
        fallback_types.append(code)
        avail = fallback_mask & ~ref_dedup.index.isin(selected_idx)
        z_col = z_mat.loc[avail, code_col]
        expr_col = cancer_expr.loc[avail, code_col]
        sn_col = sn.loc[avail]
        composite = z_col * np.log2(expr_col + 1) * np.clip(sn_col, 0, 3) / 3
        top = composite.nlargest(n_genes_per_type).index
        syms = list(ref_dedup.loc[top, "Symbol"].values)
        per_type[code] = per_type.get(code, []) + syms
        selected_idx.extend(top)

    # --- CTA boost ---
    selected_syms = set(ref_dedup.loc[list(dict.fromkeys(selected_idx)), "Symbol"].values)
    cta_added = []
    for cta in _CURATED_CTAS:
        if cta in selected_syms:
            continue
        cta_rows = ref_dedup[ref_dedup["Symbol"] == cta]
        if len(cta_rows) == 0:
            continue
        cta_row = cta_rows.iloc[0]
        cta_expr = cancer_expr.loc[cta_row.name]
        if cta_expr.max() < 1.0:
            continue  # median < 1 FPKM in best type
        selected_idx.append(cta_row.name)
        cta_added.append(cta)

    selected_idx = list(dict.fromkeys(selected_idx))
    ref_filtered = ref_dedup.loc[selected_idx]

    metadata = {
        "per_type": per_type,
        "fallback_types": fallback_types,
        "cta_added": cta_added,
        "n_genes": len(selected_idx),
        "n_types": len([t for t, g in per_type.items() if g]),
    }

    result = (ref_filtered, metadata)
    _embedding_gene_cache[cache_key] = result
    return result


def _cancer_type_score_matrix(df_gene_expr, n_signature_genes=20):
    """Build feature matrix using cancer-type signature scores.

    Each cancer type (and the sample) is represented as a vector of scores:
    "how well does this expression profile match each cancer type's signature?"

    For reference cancer types, the score is the midrank percentile of that
    type's median expression among all cancer types, for the target type's
    signature genes.  For the sample, same scoring via
    ``_compute_cancer_type_signature_stats``.

    Returns (matrix, labels) where matrix is (34, 33) — 33 cancer types + sample.
    """
    import numpy as np

    from .tumor_purity import _cached_reference_matrices

    ref_matrices = _cached_reference_matrices(normalize="housekeeping")
    fpkm_cols = ref_matrices["fpkm_cols"]
    labels = [c.replace("FPKM_", "") for c in fpkm_cols]

    expr_matrix = ref_matrices["expr_matrix"]
    sig = _get_cancer_type_signature_panels(n_signature_genes=n_signature_genes)

    # Score each reference cancer type against all signatures
    ref_scores = np.zeros((len(labels), len(labels)))
    for j, target_code in enumerate(labels):
        genes = sig[target_code]
        for i, source_code in enumerate(labels):
            source_col = f"FPKM_{source_code}"
            pcts = []
            for gene in genes:
                if gene not in expr_matrix.index:
                    continue
                val = float(expr_matrix.loc[gene, source_col])
                ref_vals = expr_matrix.loc[gene].values
                n = len(ref_vals)
                below = np.sum(ref_vals < val)
                equal = np.sum(np.isclose(ref_vals, val, atol=1e-6))
                pcts.append((below + 0.5 * equal) / n)
            ref_scores[i, j] = float(np.mean(pcts)) if pcts else 0.5

    # Score the sample
    sample_stats = _compute_cancer_type_signature_stats(
        df_gene_expr, n_signature_genes=n_signature_genes,
    )
    sample_scores = np.zeros(len(labels))
    for stat in sample_stats:
        j = labels.index(stat["code"])
        sample_scores[j] = stat["score"]

    matrix = np.vstack([ref_scores, sample_scores[None, :]])
    labels.append("SAMPLE")
    return matrix, labels


def _reference_cancer_expression_df(cancer_code):
    """Return a TCGA-median expression frame for one cancer type."""
    import pandas as pd

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"FPKM_{cancer_code}"].astype(float),
        }
    )


def _hierarchy_feature_labels():
    from .tumor_purity import _CANCER_FAMILY_PANELS, CANCER_TO_TISSUE

    ref = pan_cancer_expression()
    codes = [c.replace("FPKM_", "") for c in ref.columns if c.startswith("FPKM_")]
    families = list(_CANCER_FAMILY_PANELS)
    site_labels = sorted(
        set(CANCER_TO_TISSUE.values())
        | {"appendix", "bone_marrow", "lymph_node", "smooth_muscle", "spleen", "adipose_tissue"}
    )
    feature_labels = (
        [f"support::{code}" for code in codes]
        + [f"family::{family}" for family in families]
        + [f"site::{site}" for site in site_labels]
        + ["purity::best_estimate"]
    )
    return codes, families, site_labels, feature_labels


def _hierarchy_feature_vector(df_gene_expr, candidate_codes, family_labels, site_labels):
    """Build one hierarchy-aware embedding vector for a sample/profile."""
    from .tumor_purity import (
        rank_cancer_type_candidates,
        _score_cancer_family_panels,
        _score_host_tissues,
    )

    candidate_trace = rank_cancer_type_candidates(
        df_gene_expr,
        candidate_codes=candidate_codes,
        top_k=len(candidate_codes),
    )
    trace_by_code = {row["code"]: row for row in candidate_trace}
    sample_raw_by_symbol, _ = _sample_expression_by_symbol(df_gene_expr)
    family_scores = _score_cancer_family_panels(sample_raw_by_symbol)
    max_family_score = max(family_scores.values(), default=0.0)
    if max_family_score > 0:
        family_features = [
            float(family_scores.get(family, 0.0) / max_family_score)
            for family in family_labels
        ]
    else:
        family_features = [0.0 for _ in family_labels]

    site_scores = {
        tissue: score
        for tissue, score, _ in _score_host_tissues(
            sample_raw_by_symbol,
            tissues=site_labels,
            top_n=None,
        )
    }
    max_site_score = max(site_scores.values(), default=0.0)
    if max_site_score > 0:
        site_features = [
            float(site_scores.get(site, 0.0) / max_site_score)
            for site in site_labels
        ]
    else:
        site_features = [0.0 for _ in site_labels]

    support_features = [
        float(trace_by_code.get(code, {}).get("support_norm", 0.0))
        for code in candidate_codes
    ]
    best_purity = float(candidate_trace[0]["purity_estimate"]) if candidate_trace else 0.0
    return np.array(support_features + family_features + site_features + [best_purity], dtype=float)


def _reference_family_feature_matrix(candidate_codes, family_labels):
    """Build normalized family-panel features for TCGA reference centroids."""
    from .tumor_purity import _CANCER_FAMILY_PANELS

    ref_hk = pan_cancer_expression(normalize="housekeeping").drop_duplicates(subset="Symbol")
    ref_hk = ref_hk.set_index("Symbol")

    rows = []
    for code in candidate_codes:
        col = f"FPKM_{code}"
        family_values = []
        for family in family_labels:
            genes = [gene for gene in _CANCER_FAMILY_PANELS[family] if gene in ref_hk.index]
            if genes:
                values = sorted(ref_hk.loc[genes, col].astype(float).tolist())
                upper_half = values[len(values) // 2:] if len(values) >= 3 else values
                score = float(np.median(upper_half)) if upper_half else 0.0
            else:
                score = 0.0
            family_values.append(score)
        max_family = max(family_values, default=0.0)
        if max_family > 0:
            family_values = [float(value / max_family) for value in family_values]
        rows.append(family_values)
    return np.asarray(rows, dtype=float)


def _reference_site_feature_matrix(candidate_codes, site_labels):
    """Build normalized host/background context features for TCGA centroids."""
    from .tumor_purity import _score_host_tissues

    cache_key = (tuple(candidate_codes), tuple(site_labels))
    cached = _hierarchy_site_cache.get(cache_key)
    if cached is not None:
        return cached

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    rows = []
    for code in candidate_codes:
        sample_raw_by_symbol = ref[f"FPKM_{code}"].astype(float).to_dict()
        site_scores = {
            tissue: score
            for tissue, score, _ in _score_host_tissues(
                sample_raw_by_symbol,
                tissues=site_labels,
                top_n=None,
            )
        }
        max_site_score = max(site_scores.values(), default=0.0)
        if max_site_score > 0:
            site_values = [
                float(site_scores.get(site, 0.0) / max_site_score)
                for site in site_labels
            ]
        else:
            site_values = [0.0 for _ in site_labels]
        rows.append(site_values)
    cached = np.asarray(rows, dtype=float)
    _hierarchy_site_cache[cache_key] = cached
    return cached


def _hierarchy_embedding_metadata():
    """Describe the hierarchy-aware embedding feature space."""
    candidate_codes, family_labels, site_labels, feature_labels = _hierarchy_feature_labels()
    return {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": len(feature_labels),
        "n_types": len(candidate_codes),
        "n_genes": 0,
        "per_type": {},
        "families": family_labels,
        "sites": site_labels,
        "codes": candidate_codes,
        "feature_labels": feature_labels,
    }


def _cancer_type_hierarchy_matrix(df_gene_expr):
    """Build a family-aware support-space matrix for TCGA centroids + sample."""
    import numpy as np
    from .tumor_purity import TCGA_MEDIAN_PURITY

    candidate_codes, family_labels, site_labels, _feature_labels = _hierarchy_feature_labels()
    cache_key = tuple(candidate_codes)
    cached = _hierarchy_feature_cache.get(cache_key)
    if cached is None:
        ref_scores, labels = _cancer_type_score_matrix(_reference_cancer_expression_df(candidate_codes[0]))
        ref_labels = labels[:-1]
        ref_matrix = ref_scores[:-1]
        ref_families = _reference_family_feature_matrix(ref_labels, family_labels)
        ref_sites = _reference_site_feature_matrix(ref_labels, site_labels)
        ref_purity = np.array(
            [[float(TCGA_MEDIAN_PURITY.get(code, 0.5))] for code in ref_labels],
            dtype=float,
        )
        cached = (np.hstack([ref_matrix, ref_families, ref_sites, ref_purity]), ref_labels)
        _hierarchy_feature_cache[cache_key] = cached

    ref_matrix, labels = cached
    sample_vector = _hierarchy_feature_vector(df_gene_expr, candidate_codes, family_labels, site_labels)
    matrix = np.vstack([ref_matrix, sample_vector[None, :]])
    out_labels = list(labels) + ["SAMPLE"]
    return matrix, out_labels


def _cancer_type_feature_matrix(df_gene_expr, n_genes=10, method="zscore"):
    """Build feature matrix for PCA/MDS of cancer types + sample.

    Gene selection is unified: a single biologically-informed gene set is
    used regardless of normalization method.  The *method* parameter only
    controls how expression values are transformed before embedding.

    Parameters
    ----------
    method : str
        ``"zscore"`` — z-score of log2(1+raw) across cancer types (default).
        ``"hk"`` — log2(HK-normalized + 1).
        ``"hk_zscore"`` — z-score of log2(HK-normalized + 1).
        ``"rank"`` — percentile rank within each gene across cancer types.
        ``"score"`` — cancer-type signature scores (33-d vector).
        ``"hierarchy"`` — family-aware support-score space with purity anchor.
    """
    import warnings

    import numpy as np
    from scipy.stats import rankdata
    from .plot_data_helpers import _strip_ensembl_version

    if method == "robust":
        warnings.warn(
            "method='robust' is deprecated; gene selection is now unified. "
            "Using 'zscore'.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "zscore"

    if method == "score":
        return _cancer_type_score_matrix(df_gene_expr)
    if method == "hierarchy":
        return _cancer_type_hierarchy_matrix(df_gene_expr)

    gene_id_col, _ = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )

    if method in ("hk", "hk_zscore"):
        hk_mask = df[gene_id_col].isin(housekeeping_gene_ids())
        hk_median = df.loc[hk_mask, tpm_col].astype(float).median()
        if not (hk_median > 0):  # catches NaN and <= 0
            hk_median = 1.0
        sample_by_id = dict(zip(
            df[gene_id_col].astype(str), df[tpm_col].astype(float) / hk_median,
        ))
        ref_full = pan_cancer_expression(normalize="housekeeping")
    else:
        sample_by_id = dict(zip(
            df[gene_id_col].astype(str), df[tpm_col].astype(float),
        ))
        ref_full = pan_cancer_expression()

    fpkm_cols = [c for c in ref_full.columns if c.startswith("FPKM_")]
    labels = [c.replace("FPKM_", "") for c in fpkm_cols]

    # Gene selection
    if method == "tme":
        ref_filtered, _meta = _select_tme_low_genes(n_genes_per_type=n_genes)
    elif method == "bottleneck":
        ref_filtered, _meta = _select_embedding_genes_bottleneck(n_genes_per_type=n_genes)
    else:
        ref_filtered, _meta = _select_embedding_genes(n_genes_per_type=n_genes)

    # Map gene set to potentially HK-normalized reference
    gene_ids = list(ref_filtered["Ensembl_Gene_ID"])
    ref_norm = ref_full[ref_full["Ensembl_Gene_ID"].isin(gene_ids)].drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    # Preserve the gene order from _select_embedding_genes
    ref_norm = ref_norm.set_index("Ensembl_Gene_ID").loc[
        [gid for gid in gene_ids if gid in ref_norm["Ensembl_Gene_ID"].values]
    ].reset_index()

    sample_vals = np.array([
        sample_by_id.get(row["Ensembl_Gene_ID"], 0.0)
        for _, row in ref_norm.iterrows()
    ])
    ref_vals = ref_norm[fpkm_cols].astype(float).values  # (genes, cancers)

    if method in ("zscore", "hk_zscore", "tme", "bottleneck"):
        log_ref = np.log2(ref_vals + 1)
        log_sample = np.log2(sample_vals + 1)
        g_std = log_ref.std(axis=1)
        var_mask = g_std >= 0.1
        log_ref = log_ref[var_mask]
        log_sample = log_sample[var_mask]
        g_std = g_std[var_mask]
        g_mean = log_ref.mean(axis=1)
        z_ref = np.clip((log_ref - g_mean[:, None]) / g_std[:, None], -3, 3)
        z_sample = np.clip((log_sample - g_mean) / g_std, -3, 3)
        matrix = np.vstack([z_ref.T, z_sample[None, :]])
    elif method == "hk":
        combined = np.vstack([ref_vals.T, sample_vals[None, :]])
        matrix = np.log2(combined + 1)
    elif method == "rank":
        combined = np.vstack([ref_vals.T, sample_vals[None, :]])  # (34, genes)
        ranked = np.apply_along_axis(
            lambda col: rankdata(col, method="average") / len(col),
            axis=0, arr=combined,
        )
        matrix = ranked
    else:
        raise ValueError(f"Unknown method: {method}")

    labels.append("SAMPLE")
    return matrix, labels


def get_embedding_feature_metadata(method="hierarchy", n_genes=10):
    """Return metadata describing the active embedding feature space."""
    if method == "hierarchy":
        return _hierarchy_embedding_metadata()
    if method == "tme":
        return _select_tme_low_genes(n_genes_per_type=n_genes)[1]
    if method == "bottleneck":
        return _select_embedding_genes_bottleneck(n_genes_per_type=n_genes)[1]
    return _select_embedding_genes(n_genes_per_type=n_genes)[1]


def _plot_embedding_with_labels(
    coords,
    labels,
    *,
    title,
    xlabel,
    ylabel,
    method=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    family_palette = {}
    label_to_family = {}
    if method == "hierarchy":
        from .tumor_purity import _CANCER_FAMILY_BY_CODE

        family_order = []
        for label in labels:
            if label == "SAMPLE":
                continue
            family = _CANCER_FAMILY_BY_CODE.get(label, label)
            label_to_family[label] = family
            if family not in family_order:
                family_order.append(family)
        palette = sns.color_palette("tab20", max(len(family_order), 1))
        family_palette = {family: palette[idx] for idx, family in enumerate(family_order)}

    nearest_neighbors = []
    if "SAMPLE" in labels:
        sample_idx = labels.index("SAMPLE")
        sample_coords = coords[sample_idx]
        for i, label in enumerate(labels):
            if label == "SAMPLE":
                continue
            dist = float(np.linalg.norm(coords[i] - sample_coords))
            nearest_neighbors.append((dist, label))
        nearest_neighbors.sort(key=lambda item: (item[0], item[1]))

    for i, label in enumerate(labels):
        if label == "SAMPLE":
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                s=220,
                color="red",
                edgecolors="black",
                linewidths=1.5,
                zorder=5,
                marker="*",
            )
            texts.append(
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    label,
                    fontsize=10,
                    fontweight="bold",
                    color="red",
                    va="center",
                )
            )
        else:
            point_color = family_palette.get(label_to_family.get(label), "steelblue")
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                s=60,
                alpha=0.7,
                color=point_color,
                edgecolors="white",
                linewidths=0.5,
                zorder=2,
            )
            texts.append(
                ax.text(
                    coords[i, 0],
                    coords[i, 1],
                    label,
                    fontsize=7,
                    alpha=0.8,
                    va="center",
                )
            )

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", alpha=0.35),
        expand=(1.05, 1.2),
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.2)

    if nearest_neighbors:
        nearest_text = "\n".join(
            f"{label} ({dist:.2f})" for dist, label in nearest_neighbors[:5]
        )
        ax.text(
            0.98,
            0.02,
            "Nearest TCGA centroids\n" + nearest_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
        )

    if family_palette:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor="white", markersize=7, label=family)
            for family, color in list(family_palette.items())[:10]
        ]
        ax.legend(handles=handles, title="Family", loc="upper left", fontsize=8, title_fontsize=9, framealpha=0.9)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cancer_type_genes(
    df_gene_expr,
    n_per_tail=5,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 22),
):
    """Show all signature genes for the closest and most distant cancer types.

    Cancer types are ranked by signature similarity to the sample. The plot
    then shows the top ``n_per_tail`` closest and bottom ``n_per_tail`` most
    distant cancer types, while still plotting all signature genes for each
    selected row. The gray bar marks the mean sample TPM for that row.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    n_per_tail : int
        Number of cancer types to show in each tail (closest / most distant).
    save_to_filename : str or None
        Output path.
    """
    import numpy as np
    stats = _compute_cancer_type_signature_stats(df_gene_expr, n_signature_genes=20)
    if not stats:
        return None, None

    top_stats = stats[:n_per_tail]
    bottom_stats = sorted(stats[-n_per_tail:], key=lambda row: (row["score"], row["code"]))
    selected_stats = top_stats + bottom_stats

    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)

    y_pos = 0
    y_ticks = []
    y_labels = []
    texts = []

    for idx, row in enumerate(selected_stats):
        code = row["code"]
        gene_details = sorted(
            row["gene_details"],
            key=lambda item: (-item["sample_raw"], item["gene"]),
        )
        label = f"{row['rank']:>2}. {code} — score={row['score']:.2f}"
        y_ticks.append(y_pos)
        y_labels.append(label)

        x_values = [detail["sample_raw"] + 0.01 for detail in gene_details]
        ax.scatter(
            x_values,
            y_pos + rng.uniform(-0.14, 0.14, len(x_values)),
            s=18,
            alpha=0.45,
            color="#2166ac",
            edgecolors="none",
            zorder=2,
        )

        mean_x = row["mean_sample_raw"] + 0.01
        ax.plot(
            [mean_x, mean_x],
            [y_pos - 0.28, y_pos + 0.28],
            color="#999999",
            linewidth=1.6,
            alpha=0.7,
            zorder=1,
        )

        for detail in gene_details[:5]:
            x = detail["sample_raw"] + 0.01
            jitter = rng.uniform(-0.12, 0.12)
            ax.scatter(
                x,
                y_pos + jitter,
                s=30,
                alpha=0.85,
                color="#2166ac",
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            texts.append(
                ax.text(
                    x,
                    y_pos + jitter,
                    detail["gene"],
                    fontsize=8,
                    va="center",
                    ha="left",
                    alpha=0.85,
                    color="#2166ac",
                )
            )

        if idx == len(top_stats) - 1 and bottom_stats:
            ax.axhline(y=y_pos + 0.5, color="#cccccc", linewidth=0.8, alpha=0.8, zorder=0)

        y_pos += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Sample TPM", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(
        "Cancer-type signature genes: sample expression\n"
        "(top 5 closest and bottom 5 most distant cancer types; gray bar = mean sample TPM)",
        fontsize=11,
    )
    ax.invert_yaxis()

    # Reference lines
    for tpm_thresh in (10, 100):
        ax.axvline(x=tpm_thresh, color="#cccccc", linestyle="--",
                   linewidth=0.7, alpha=0.5, zorder=1)

    # Fix x-axis limits before adjustText to prevent blowout
    ax.set_xlim(left=0.005)
    ax.autoscale_view(scalex=True, scaley=False)

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", alpha=0.25),
        expand=(1.03, 1.2),
        expand_axes=False,
        ensure_inside_axes=True,
    )

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cancer_type_disjoint_genes(
    df_gene_expr,
    n_genes=20,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 12),
):
    """Bar chart of cancer-type signature similarity scores.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Patient expression data.
    n_genes : int
        Max disjoint genes per cancer type to consider.
    save_to_filename : str or None
        Output path.
    """
    stats = _compute_cancer_type_signature_stats(
        df_gene_expr,
        n_signature_genes=n_genes,
        min_fold=2.0,
    )

    fig, ax = plt.subplots(figsize=figsize)

    scores = [row["score"] for row in stats]
    labels = [
        f"{row['code']} ({CANCER_TYPE_NAMES.get(row['code'], row['code'])})"
        for row in stats
    ]
    y = np.arange(len(stats))

    # Color bars by score intensity
    colors = [plt.cm.Blues(0.25 + 0.65 * s) for s in scores]
    ax.barh(y, scores, color=colors, edgecolor="none", height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Signature similarity score", fontsize=10)
    # Definition belongs on the axis label, not in a subtitle.
    ax.set_title("Cancer type similarity score", fontsize=11)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 1)

    # Annotate bars with top contributing genes
    for i, row in enumerate(stats):
        top3 = sorted(row["gene_details"], key=lambda detail: -detail["percentile"])[:3]
        top3_str = ", ".join(detail["gene"] for detail in top3)
        if top3_str:
            ax.text(min(row["score"] + 0.01, 0.99), i, top3_str,
                    fontsize=5.5, va="center", alpha=0.7)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


# -------------------- cohort-only plots (no sample needed) --------------------


def plot_cohort_heatmap(
    save_to_filename=None,
    save_dpi=300,
    figsize=(18, 14),
    zscore=True,
):
    """Heatmap of curated cancer-type genes × cancer types."""
    import numpy as np
    from .gene_sets_cancer import pan_cancer_expression, cancer_types

    # Load curated cancer-type genes
    from .load_dataset import get_data
    ct_df = get_data("cancer-type-genes")
    gene_symbols = sorted(ct_df["Symbol"].unique())

    # Get expression
    ref = pan_cancer_expression(genes=gene_symbols, normalize="housekeeping")
    codes = cancer_types()
    fpkm_cols = [f"FPKM_{c}" for c in codes if f"FPKM_{c}" in ref.columns]
    codes = [c.replace("FPKM_", "") for c in fpkm_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in gene_symbols if s in ref_dedup.index]
    matrix = ref_dedup.loc[present, fpkm_cols].astype(float)
    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 expression across cancers"
        cbar_label = "z-score"
    else:
        cmap, vmin, vmax = "RdBu_r", -5, 5
        subtitle = "log2 housekeeping-normalized"
        cbar_label = "log2(HK-normalized)"
    matrix.columns = codes

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=4)
    ax.set_title(f"Curated cancer-type genes × TCGA cancer types — {subtitle}", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_disjoint_counts(
    n_genes=30,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """Bar chart of disjoint signature gene counts per cancer type (no sample)."""
    import numpy as np
    from .gene_sets_cancer import top_enriched_per_cancer_type

    sig = top_enriched_per_cancer_type(n=n_genes, disjoint=True, min_fold=2.0)
    stats = [(code, len(genes)) for code, genes in sig.items()]
    stats.sort(key=lambda x: -x[1])

    codes = [s[0] for s in stats]
    counts = [s[1] for s in stats]
    labels = [f"{c} ({CANCER_TYPE_NAMES.get(c, c)})" for c in codes]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(codes))
    ax.barh(y, counts, color="#2166ac", edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"Disjoint signature genes (of {n_genes} max)", fontsize=10)
    ax.set_title("Cancer-type-specific disjoint gene counts", fontsize=11)
    ax.invert_yaxis()

    for i, (code, count) in enumerate(stats):
        top3 = sig[code][:3]
        ax.text(count + 0.3, i, ", ".join(top3), fontsize=5.5, va="center", alpha=0.7)

    fig.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_pca(
    n_genes=20,
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """PCA of 33 TCGA cancer type centroids (no sample)."""
    import numpy as np
    from sklearn.decomposition import PCA
    from .gene_sets_cancer import top_enriched_per_cancer_type, pan_cancer_expression

    sig = top_enriched_per_cancer_type(n=n_genes, disjoint=True)
    all_symbols = set()
    for genes in sig.values():
        all_symbols.update(genes)

    ref = pan_cancer_expression(normalize="housekeeping")
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    codes = [c.replace("FPKM_", "") for c in fpkm_cols]

    ref_filtered = ref[ref["Symbol"].isin(all_symbols)].copy()
    gene_order = sorted(ref_filtered["Symbol"].unique())

    feature_matrix = []
    for col in fpkm_cols:
        vals = []
        for sym in gene_order:
            row_mask = ref_filtered["Symbol"] == sym
            v = ref_filtered.loc[row_mask, col].astype(float).values
            vals.append(v[0] if len(v) > 0 else 0)
        feature_matrix.append(vals)

    X = np.array(feature_matrix)
    X = np.log2(X + 1)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords[:, 0], coords[:, 1], s=80, alpha=0.7,
               color="steelblue", edgecolors="white", linewidths=0.5, zorder=2)
    for i, code in enumerate(codes):
        ax.text(coords[i, 0], coords[i, 1], f" {code}",
                fontsize=8, alpha=0.8, va="center")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)", fontsize=11)
    ax.set_title("TCGA cancer type centroids in gene-signature PCA space", fontsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_therapy_targets(
    save_to_filename=None,
    save_dpi=300,
    figsize=(16, 10),
    zscore=True,
):
    """Heatmap of therapy targets × cancer types showing expression.

    Rows are therapy target genes (from ADC, CAR-T, bispecific, radioligand,
    TCR-T registries), columns are cancer types.
    """
    import numpy as np
    from .gene_sets_cancer import (
        therapy_target_gene_id_to_name,
        pan_cancer_expression,
        cancer_types,
    )

    # Collect all therapy targets
    all_targets = {}
    for therapy in ["ADC", "CAR-T", "TCR-T", "bispecific-antibodies", "radioligand"]:
        d = therapy_target_gene_id_to_name(therapy)
        all_targets.update(d)
    target_symbols = sorted(set(all_targets.values()))

    ref = pan_cancer_expression(genes=target_symbols, normalize="housekeeping")
    codes = cancer_types()
    fpkm_cols = [f"FPKM_{c}" for c in codes if f"FPKM_{c}" in ref.columns]
    codes = [c.replace("FPKM_", "") for c in fpkm_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in target_symbols if s in ref_dedup.index]
    matrix = ref_dedup.loc[present, fpkm_cols].astype(float)
    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 expression across cancers"
        cbar_label = "z-score"
    else:
        cmap, vmin, vmax = "YlOrRd", -5, 3
        subtitle = "log2 housekeeping-normalized"
        cbar_label = "log2(HK-normalized)"
    matrix.columns = codes

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=6)
    ax.set_title(f"Therapy targets × TCGA cancer types — {subtitle}", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def _plot_geneset_by_cancer_heatmap(
    gene_symbols,
    title,
    save_to_filename=None,
    save_dpi=300,
    figsize=(16, 12),
    cmap="YlOrRd",
    top_n_per_cancer=None,
    zscore=True,
):
    """Shared helper: heatmap of gene set × cancer types."""
    import numpy as np
    from .gene_sets_cancer import pan_cancer_expression, cancer_types

    ref = pan_cancer_expression(genes=gene_symbols, normalize="housekeeping")
    codes = cancer_types()
    fpkm_cols = [f"FPKM_{c}" for c in codes if f"FPKM_{c}" in ref.columns]
    codes = [c.replace("FPKM_", "") for c in fpkm_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in gene_symbols if s in ref_dedup.index]

    if not present:
        return None, None

    matrix = ref_dedup.loc[present, fpkm_cols].astype(float)

    # Filter to top N genes by max expression across any cancer type
    if top_n_per_cancer and len(present) > top_n_per_cancer:
        max_expr = matrix.max(axis=1)
        top_idx = max_expr.nlargest(top_n_per_cancer).index
        matrix = matrix.loc[top_idx]
        present = list(top_idx)

    # Sort rows by mean expression (highest at top)
    row_means = matrix.mean(axis=1)
    sort_order = row_means.sort_values(ascending=False).index
    matrix = matrix.loc[sort_order]
    present = list(sort_order)

    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        use_cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 expression across cancers"
        cbar_label = "z-score"
    else:
        use_cmap, vmin, vmax = cmap, -5, 3
        subtitle = "log2 housekeeping-normalized"
        cbar_label = "log2(HK-normalized)"
    matrix.columns = codes

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.values, aspect="auto", cmap=use_cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=5 if len(present) > 50 else 6)
    ax.set_title(f"{title} — {subtitle}", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


def plot_cohort_surface_proteins(
    save_to_filename=None, save_dpi=300, figsize=(16, 14), zscore=True,
):
    """Heatmap of cancer surfaceome targets × cancer types."""
    from .gene_sets_cancer import cancer_surfaceome_gene_names
    genes = sorted(cancer_surfaceome_gene_names())
    return _plot_geneset_by_cancer_heatmap(
        genes, "Tumor-specific surface proteins (TCSA L3) × cancer types",
        save_to_filename=save_to_filename, save_dpi=save_dpi,
        figsize=figsize, cmap="YlOrRd", zscore=zscore,
    )


def plot_cohort_ctas(
    save_to_filename=None, save_dpi=300, figsize=(16, 14), zscore=True,
):
    """Heatmap of CTA genes × cancer types."""
    import numpy as np
    from .gene_sets_cancer import CTA_gene_names, pan_cancer_expression, cancer_types

    genes = sorted(CTA_gene_names())
    ref = pan_cancer_expression(genes=genes)  # raw values, not HK-normalized
    codes = cancer_types()
    fpkm_cols = [f"FPKM_{c}" for c in codes if f"FPKM_{c}" in ref.columns]
    codes_clean = [c.replace("FPKM_", "") for c in fpkm_cols]

    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    present = [s for s in genes if s in ref_dedup.index]
    matrix = ref_dedup.loc[present, fpkm_cols].astype(float)

    # Filter to top 50 by max expression, sort by mean descending
    max_expr = matrix.max(axis=1)
    top50 = max_expr.nlargest(50).index
    matrix = matrix.loc[top50]
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    present = list(matrix.index)

    matrix = np.log2(matrix + 1)

    if zscore:
        row_mean = matrix.mean(axis=1)
        row_std = matrix.std(axis=1).clip(lower=0.1)
        matrix = matrix.sub(row_mean, axis=0).div(row_std, axis=0)
        cmap, vmin, vmax = "RdBu_r", -3, 3
        subtitle = "z-score of log2 FPKM across cancers"
        cbar_label = "z-score"
    else:
        cmap, vmin, vmax = "magma_r", -3, 8
        subtitle = "log2 FPKM"
        cbar_label = "log2(FPKM + 1)"
    matrix.columns = codes_clean

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(codes_clean)))
    ax.set_xticklabels(codes_clean, fontsize=7, rotation=90)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=6)
    ax.set_title(f"Cancer-testis antigens × cancer types (top 50) — {subtitle}", fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, ax


_METHOD_LABELS = {
    "zscore": "z-score",
    "hk": "HK-normalized",
    "hk_zscore": "HK z-score",
    "rank": "percentile rank",
    "hierarchy": "hierarchical support space",
    "tme": "TME-low genes",
    "score": "signature scores",
    "bottleneck": "bottleneck genes",
}


def plot_cancer_type_pca(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """PCA scatter showing where the sample falls among cancer-type centroids."""
    from sklearn.decomposition import PCA
    X, labels = _cancer_type_feature_matrix(df_gene_expr, n_genes=n_genes, method=method)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    mlabel = _METHOD_LABELS.get(method)
    title = "Sample among TCGA cancer types — PCA"
    if mlabel:
        title += f" ({mlabel})"
    return _plot_embedding_with_labels(
        coords,
        labels,
        title=title,
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)",
        method=method,
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
    )


def plot_cancer_type_mds(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """MDS embedding of the sample with TCGA cancer type centroids."""
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    X, labels = _cancer_type_feature_matrix(df_gene_expr, n_genes=n_genes, method=method)
    distances = pairwise_distances(X, metric="euclidean")
    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
    ).fit_transform(distances)
    mlabel = _METHOD_LABELS.get(method)
    title = "Sample among TCGA cancer types — MDS"
    if mlabel:
        title += f" ({mlabel})"
    return _plot_embedding_with_labels(
        coords,
        labels,
        title=title,
        xlabel="MDS1",
        ylabel="MDS2",
        method=method,
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
    )


def plot_cancer_type_umap(
    df_gene_expr,
    n_genes=10,
    method="zscore",
    save_to_filename=None,
    save_dpi=300,
    figsize=(12, 10),
):
    """UMAP embedding of the sample with TCGA cancer type centroids."""
    from umap import UMAP

    X, labels = _cancer_type_feature_matrix(df_gene_expr, n_genes=n_genes, method=method)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value")
        coords = UMAP(
            n_components=2,
            n_neighbors=min(15, len(labels) - 1),
            random_state=42,
        ).fit_transform(X)
    mlabel = _METHOD_LABELS.get(method)
    title = "Sample among TCGA cancer types — UMAP"
    if mlabel:
        title += f" ({mlabel})"
    return _plot_embedding_with_labels(
        coords,
        labels,
        title=title,
        xlabel="UMAP1",
        ylabel="UMAP2",
        method=method,
        save_to_filename=save_to_filename,
        save_dpi=save_dpi,
        figsize=figsize,
    )
