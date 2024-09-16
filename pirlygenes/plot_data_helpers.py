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
    
    print("Categories:" , list(gene_sets.keys()))
    return gene_sets

def _create_gene_to_category_mapping(gene_sets):
    
    gene_to_category = defaultdict(lambda: "other")
    for category, genes in gene_sets.items():
        for g in genes:
            gene_to_category[g] = category
    return gene_to_category

    
def prepare_gene_expr_df(
        df_gene_expr,
        gene_sets, 
        priority_category=None, 
        TPM_offset=10.0**-4):
    
    df_gene_expr_annot = df_gene_expr.copy()
    
    df_gene_expr_annot["gene_alias"] = [aliases.get(g, g) for g in df_gene_expr_annot.gene]

    check_gene_names_in_gene_sets(df_gene_expr, gene_sets)
    gene_sets = normalize_categories(gene_sets, priority_category=priority_category)
    gene_to_category = _create_gene_to_category_mapping(gene_sets)
    
    df_gene_expr_annot["category"] = [gene_to_category[g] for g in df_gene_expr_annot.gene]
    categories = list(gene_sets.keys())
    if "other" not in categories:
        categories = ["other"] + categories
    df_gene_expr_annot["category"] = pd.Categorical(df_gene_expr_annot["category"], categories)
    
    if TPM_offset:
        df_gene_expr_annot["TPM"] += TPM_offset
        
    df_gene_expr_annot["log_TPM"] = np.log10(10.0**-4 + df_gene_expr["TPM"])
    
    return df_gene_expr_annot
          