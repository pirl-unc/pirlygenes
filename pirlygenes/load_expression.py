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

import pandas as pd

from .aggregate_gene_expression import aggregate_gene_expression as tx2gene
from .gene_ids import find_canonical_gene_ids_and_names
from .gene_aliases import display_name


def load_expression_data(input_path, aggregate_gene_expression=False):

    if ".csv" in input_path:
        df = pd.read_csv(input_path)
    elif ".xlsx" in input_path:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unrecognized file format for {input_path}")

    if aggregate_gene_expression:
        df = tx2gene(df)

    df = df.rename(columns={"Gene Symbol": "gene", "Gene": "gene"})

    df = df.rename(
        columns={
            "Gene ID": "ensembl_gene_id",
            "Gene_ID": "ensembl_gene_id",
            "Ensembl Gene ID": "ensembl_gene_id",
            "Ensembl_Gene_ID": "ensembl_gene_id",
        }
    )

    columns = sorted(set(df.columns))

    if "gene" not in columns:
        raise ValueError(
            f"Gene column not found in {input_path}, available columns: {columns}"
        )

    if "ensembl_gene_id" not in columns:
        gene_ids, canonical_gene_names = find_canonical_gene_ids_and_names(df.gene)

        df["ensembl_gene_id"] = gene_ids
        df["canonical_gene_name"] = canonical_gene_names
        df["gene_display_name"] = [
            display_name(gene_name) for gene_name in canonical_gene_names
        ]
    return df
