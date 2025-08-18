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

from .plot_data_helpers import prepare_gene_expr_df
from .gene_sets_old import (
    APM_genes,
    MHC1_genes,
    TLR_signaling,
    growth_receptor_genes,
    oncogenes,
    checkpoints,
)
from .gene_sets_cancer import CTA_gene_names

# ------------------------ helpers ------------------------


def _guess_gene_cols(df):
    """Best-effort guess for gene ID and name columns in df_gene_expr."""
    id_candidates = ["gene_id", "ensembl_gene_id", "canonical_gene_id", "GeneID"]
    name_candidates = [
        "gene_display_name",
        "gene_name",
        "canonical_gene_name",
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


# ------------------------ defaults ------------------------

default_gene_sets = dict(
    APM=APM_genes,
    MHC1=MHC1_genes,
    TLR=TLR_signaling,
    Growth_receptors=growth_receptor_genes,
    Oncogenes=oncogenes,
    Immune_checkpoints=checkpoints,
    CTAs=CTA_gene_names(),
)

# ------------------------ main plot ------------------------


def plot_gene_expression(
    df_gene_expr,
    gene_sets=default_gene_sets,
    save_to_filename=None,
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
        df_gene_expr_annot, gene_id_col=gene_id_col, gene_name_col=gene_name_col
    )

    # Sanity checks
    for col in ("category", "TPM", gene_id_col, gene_name_col):
        assert col in df_gene_expr_annot.columns, df_gene_expr_annot.columns

    # Plot
    cat = sns.catplot(
        data=df_gene_expr_annot,
        x="category",
        y="TPM",
        jitter=0.01,
        height=10,
        alpha=0.5,
        hue="category",
    )
    plt.yscale("log")

    # Annotate with display names, never raw ENSG; skip the 'other' column
    texts = []
    mask = (
        (df_gene_expr_annot.TPM > 0.1)
        & (df_gene_expr_annot.category != "other")
        & (df_gene_expr_annot[gene_id_col].isin(genes_to_annotate))
    )
    for _, row in df_gene_expr_annot[mask].iterrows():
        # Use the friendly display name
        label = row[gene_name_col]
        texts.append(
            cat.ax.text(
                row.category,
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
        cat.figure.savefig(save_to_filename)

    return cat
