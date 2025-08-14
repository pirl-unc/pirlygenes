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
from tqdm import tqdm

from .aggregate_gene_expression import aggregate_gene_expression as tx2gene
from .gene_ids import (
    find_canonical_gene_ids_and_names,
    find_gene_name_from_ensembl_gene_id,
)
from .gene_names import display_name, short_gene_name


def get_canonical_gene_name_from_gene_ids_string(gene_ids_string):
    gene_ids = gene_ids_string.split(";")
    gene_names = [find_gene_name_from_ensembl_gene_id(gene_id) for gene_id in gene_ids]
    not_none_gene_names = [name for name in gene_names if name is not None]
    return ";".join(not_none_gene_names)


def load_expression_data(input_path, aggregate_gene_expression=False):

    if ".csv" in input_path:
        df = pd.read_csv(input_path)
    elif ".tsv" in input_path or ".sf" in input_path or ".txt" in input_path:
        df = pd.read_csv(input_path, sep="\t")
    elif ".xlsx" in input_path:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unrecognized file format for {input_path}")

    if aggregate_gene_expression:
        df = tx2gene(df)

    df = df.rename(
        columns={
            "Gene Symbol": "gene",
            "Gene": "gene",
            "Gene Name": "gene",
            "Gene ID": "ensembl_gene_id",
            "Gene_ID": "ensembl_gene_id",
            "Ensembl Gene ID": "ensembl_gene_id",
            "Ensembl_Gene_ID": "ensembl_gene_id",
        }
    )
    if "gene" not in set(df.columns):
        raise ValueError(
            f"Gene column not found in {input_path}, available columns: {sorted(set(df.columns))}"
        )
    df["gene"] = df["gene"].apply(short_gene_name)

    if "ensembl_gene_id" not in set(df.columns):
        gene_ids, canonical_gene_names = find_canonical_gene_ids_and_names(df.gene)
        if not gene_ids:
            raise ValueError(
                f"Unable to find Ensembl gene IDs for any of the genes in {input_path}"
            )
        if len(gene_ids) != len(df):
            raise ValueError(
                f"Number of Ensembl gene IDs ({len(gene_ids)}) does not match number of rows in {input_path} ({len(df)})"
            )

        df["ensembl_gene_id"] = gene_ids

        if not canonical_gene_names:
            raise ValueError(
                f"Unable to find canonical gene names for any of the genes in {input_path}"
            )
        if len(canonical_gene_names) != len(df):
            raise ValueError(
                f"Number of canonical gene names ({len(canonical_gene_names)}) does not match number of rows in {input_path} ({len(df)})"
            )
        if "canonical_gene_name" in set(df.columns):
            raise ValueError(
                f"Column 'canonical_gene_name' already exists in {input_path}, please rename it before loading."
            )
        else:
            df["canonical_gene_name"] = [
                (
                    gs
                    if type(gs) is str
                    else (
                        ""
                        if gs is None
                        else ";".join(gs) if type(gs) in (list, tuple) else "?"
                    )
                )
                for gs in canonical_gene_names
            ]

    if "canonical_gene_name" not in set(df.columns):
        df["canonical_gene_name"] = df["ensembl_gene_id"].apply(
            get_canonical_gene_name_from_gene_ids_string
        )

    if "gene_display_name" not in set(df.columns):
        df["gene_display_name"] = [
            ";".join([display_name(gene_name) for gene_name in gene_names.split(";")])
            for gene_names in df.canonical_gene_name
        ]
    df.to_csv("debug-expression_data.csv", index=False)
    return df
