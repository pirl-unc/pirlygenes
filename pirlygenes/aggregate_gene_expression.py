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


# -----------------------------------------------------------
# Aggregate gene expression data from transcript level expression
# to gene level expression regardless of which exact transcript
# reference was used

from collections import Counter

import pandas as pd

from .gene_ids import find_gene_and_ensembl_release_by_name, find_name_from_ensembl
from .transcript_to_gene import extra_tx_mappings
from .common import find_column


def aggregate_gene_expression(
    df: pd.DataFrame,
    tx_to_gene_name: dict[str, str] = extra_tx_mappings,
    transcript_id_column_candidates: list[str] = [
        "transcript",
        "transcript_id",
        "transcriptid",
        "target",
        "target_id",
        "targetid",
        "name",
    ],
    tpm_column_candidates: list[str] = [
        "tpm",
    ],
) -> pd.DataFrame:
    transcript_id_column = find_column(
        df, transcript_id_column_candidates, "transcript ID"
    )
    tpm_column = find_column(df, tpm_column_candidates, "TPM")

    c = Counter()
    unknown_genes_tpm = 0
    n_unknown = 0
    for t, tpm in zip(df[transcript_id_column], df[tpm_column]):
        gene_name = None
        if t in tx_to_gene_name:
            gene_name = tx_to_gene_name[t]

        else:
            t = t.split(".")[0]
            if t in tx_to_gene_name:
                gene_name = tx_to_gene_name[t]
            else:
                gene_name = find_name_from_ensembl(t, verbose=False)

            if not gene_name:
                gene_name = extra_tx_mappings.get(t)

        if gene_name:
            c[gene_name] += tpm
        else:
            if tpm > 1:
                n_unknown += 1

                print("? ", n_unknown, t, tpm)

            unknown_genes_tpm += tpm

    known_genes_tpm = sum(c.values())

    print(
        f"Assigned {known_genes_tpm:.2f} TPM to known genes, {unknown_genes_tpm:.2f} to unknown gene names; {known_genes_tpm * 100 / (known_genes_tpm + unknown_genes_tpm):.4f}% known"
    )

    df_gene_expr = pd.DataFrame({"gene": c.keys(), "TPM": c.values()})

    df_gene_expr = df_gene_expr.sort_values("TPM")

    gene_ids = []
    ensembl_versions = []
    for gene_name in df_gene_expr.gene:
        pair = find_gene_and_ensembl_release_by_name(gene_name)
        if pair is None:
            ensembl_release = gene_id = None
        else:
            ensembl_genome, gene = pair
            gene_id = gene.id
            ensembl_release = ensembl_genome.release
        gene_ids.append(gene_id)
        ensembl_versions.append(ensembl_release)
    df_gene_expr["gene_id"] = gene_ids
    df_gene_expr["ensembl_release"] = ensembl_versions
    df_gene_expr["ensembl_release"] = (
        df_gene_expr["ensembl_release"].fillna(-1).astype(int)
    )

    return df_gene_expr
