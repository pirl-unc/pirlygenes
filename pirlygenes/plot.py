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
)
from .gene_sets_cancer import get_CTAs


def pick_genes_to_annotate(df, num_per_category=10, verbose=False):
    genes_to_annotate = set()
    assert "category" in df, df.columns
    assert "TPM" in df, df.columns
    for cat, df_cat in df.groupby("category"):
        df_cat_sorted = df_cat.sort_values("TPM", ascending=False)
        top = df_cat_sorted.head(num_per_category)
        if verbose:
            print(cat)
            print(top[["gene", "TPM"]])
        genes_to_annotate.update(top.gene)
    return genes_to_annotate


default_gene_sets = dict(
    APM=APM_genes,
    MHC1=MHC1_genes,
    TLR=TLR_signaling,
    Growth_receptors=growth_receptor_genes,
    Oncogenes=oncogenes,
    CTAs=get_CTAs(),
)


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

    df_gene_expr_annot = prepare_gene_expr_df(
        df_gene_expr,
        gene_sets=gene_sets,
    )
    genes_to_annotate = pick_genes_to_annotate(df_gene_expr_annot)
    assert "category" in df_gene_expr_annot.columns, df_gene_expr_annot.columns
    assert "TPM" in df_gene_expr_annot.columns, df_gene_expr_annot.columns
    cat = sns.catplot(
        data=df_gene_expr_annot,
        x="category",
        y="TPM",
        jitter=0.01,
        height=10,
        alpha=0.6,
        hue="category",
    )
    plt.yscale("log")

    texts = []
    for _, row in df_gene_expr_annot[
        (df_gene_expr_annot.TPM > 0.1)
        & (df_gene_expr_annot.category != "other")
        & (df_gene_expr_annot.gene.isin(genes_to_annotate))
    ].iterrows():
        texts.append(
            cat.ax.text(
                row.category,
                row.TPM,
                row.gene,
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
