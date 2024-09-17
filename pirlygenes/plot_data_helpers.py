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

import numpy as np
import pandas as pd

from .gene_aliases import aliases


def check_gene_names_in_gene_sets(df_gene_expr, gene_sets):
    gene_names_in_expression_data = set(df_gene_expr.gene)
    all_genes_in_sets = sum([list(genes) for genes in gene_sets.values()], start=[])

    for g in all_genes_in_sets:
        if g not in gene_names_in_expression_data:
            print(f"Gene {g} not found in expression data")


def normalize_categories(gene_sets, priority_category=None):
    for cat, genes in list(gene_sets.items()):
        if priority_category and cat != priority_category:
            gene_sets[cat] = sorted(set(genes).difference(gene_sets[priority_category]))
        else:
            gene_sets[cat] = sorted(genes)

    print("Categories:", list(gene_sets.keys()))
    return gene_sets


def _create_gene_to_category_mapping_keep_one(gene_sets):

    gene_to_category = defaultdict(lambda: "other")
    for category, genes in gene_sets.items():
        for g in genes:
            gene_to_category[g] = category
    return gene_to_category


def _create_gene_to_category_list_mapping(gene_sets):

    gene_to_categories = defaultdict(list)
    for category, genes in gene_sets.items():
        for g in genes:
            gene_to_categories[g].append(category)
    return gene_to_categories


def prepare_gene_expr_df(
    df_gene_expr, gene_sets, priority_category=None, TPM_offset=10.0**-4
):

    check_gene_names_in_gene_sets(df_gene_expr, gene_sets)
    gene_sets = normalize_categories(gene_sets, priority_category=priority_category)
    original_genes = df_gene_expr.gene

    gene_to_alias = {g: aliases.get(g, g) for g in original_genes}
    tpm_values = df_gene_expr["TPM"].copy()
    if TPM_offset:
        tpm_values += TPM_offset
    gene_to_tpm_values = {g: tpm for (g, tpm) in zip(original_genes, tpm_values)}

    log_TPM = np.log10(10.0**-4 + tpm_values)
    gene_to_log_tpm_values = {g: tpm for (g, tpm) in zip(original_genes, log_TPM)}

    new_cats = []
    new_genes = []
    gene_to_categories = _create_gene_to_category_list_mapping(gene_sets)
    for g, cats in gene_to_categories.items():
        for cat in cats:
            new_genes.append(g)
            new_cats.append(cat)
    return pd.DataFrame(
        dict(
            gene=new_genes,
            category=new_cats,
            TPM=[gene_to_tpm_values[g] for g in new_genes],
            log_TPM=[gene_to_log_tpm_values[g] for g in new_genes],
            gene_alias=[gene_to_alias[g] for g in new_genes],
        )
    )
