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

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .aggregate_gene_expression import aggregate_gene_expression as tx2gene
from .gene_ids import (
    find_canonical_gene_ids_and_names,
    find_gene_name_from_ensembl_gene_id,
)
from .gene_names import display_name, short_gene_name

import re

# Patterns for fuzzy column guessing (tried in order, first match wins).
# Each is matched case-insensitively against the full column name.
_GENE_NAME_PATTERNS = [
    r"^gene[_\s]?sym",        # gene_symbol, gene symbol, genesym...
    r"^gene[_\s]?name",       # gene_name, gene name
    r"^symbol$",
    r"^gene$",
    r"^name$",
    r"^hgnc[_\s]?symbol",
    r"^hugo[_\s]?symbol",
]
_GENE_ID_PATTERNS = [
    r"ensembl.*gene.*id",     # ensembl_gene_id, Ensembl Gene ID, ...
    r"^gene[_\s]?id$",
    r"^ensg$",
]


def _guess_col(columns, patterns):
    """Return the first column that matches any pattern, or None."""
    for pat in patterns:
        for col in columns:
            if re.search(pat, col, re.IGNORECASE):
                return col
    return None


def get_canonical_gene_name_from_gene_ids_string(gene_ids_string):
    if pd.isna(gene_ids_string):
        return ""
    gene_ids = str(gene_ids_string).split(";")
    gene_names = [find_gene_name_from_ensembl_gene_id(gene_id) for gene_id in gene_ids]
    not_none_gene_names = [name for name in gene_names if name is not None]
    return ";".join(not_none_gene_names)


def load_expression_data(
    input_path,
    aggregate_gene_expression=False,
    save_aggregated_gene_expression=True,
    aggregated_output_path=None,
    gene_name_col=None,
    gene_id_col=None,
    verbose=True,
    progress=True,
):
    if verbose:
        print(f"[load] Loading expression data from: {input_path}")

    if ".csv" in input_path:
        df = pd.read_csv(input_path)
    elif ".tsv" in input_path or ".sf" in input_path or ".txt" in input_path:
        df = pd.read_csv(input_path, sep="\t")
    elif ".xlsx" in input_path:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unrecognized file format for {input_path}")
    if verbose:
        print(f"[load] Loaded {len(df)} rows and {len(df.columns)} columns")

    if aggregate_gene_expression:
        if verbose:
            print("[load] Aggregating transcript-level TPM values to gene-level TPM")
        df = tx2gene(df, verbose=verbose, progress=progress)

        if save_aggregated_gene_expression:
            if aggregated_output_path:
                output_path = Path(aggregated_output_path)
            else:
                input_file = Path(input_path)
                output_path = input_file.with_name(f"{input_file.stem}.gene_tpm.csv")
            df.to_csv(output_path, index=False)
            if verbose:
                print(f"[load] Saved aggregated gene-level TPM CSV to: {output_path}")

    # Apply explicit overrides first, then fall back to auto-rename
    if gene_name_col and gene_name_col in df.columns and gene_name_col != "gene":
        df = df.rename(columns={gene_name_col: "gene"})
    if gene_id_col and gene_id_col in df.columns and gene_id_col != "ensembl_gene_id":
        df = df.rename(columns={gene_id_col: "ensembl_gene_id"})

    df = df.rename(
        columns={
            "Gene Symbol": "gene",
            "gene_symbol": "gene",
            "Gene": "gene",
            "Gene Name": "gene",
            "Gene_Name": "gene",
            "symbol": "gene",
            "Symbol": "gene",
            "Gene ID": "ensembl_gene_id",
            "Gene_ID": "ensembl_gene_id",
            "Ensembl Gene ID": "ensembl_gene_id",
            "Ensembl_Gene_ID": "ensembl_gene_id",
            "gene_id": "ensembl_gene_id",
            "canonical_gene_id": "ensembl_gene_id",
            "GeneID": "ensembl_gene_id",
        }
    )

    # Fuzzy fallback: case-insensitive guess from column names
    if "gene" not in set(df.columns):
        col = _guess_col(df.columns, _GENE_NAME_PATTERNS)
        if col:
            if verbose:
                print(f"[load] Auto-detected gene name column: '{col}'")
            df = df.rename(columns={col: "gene"})
    if "ensembl_gene_id" not in set(df.columns):
        col = _guess_col(df.columns, _GENE_ID_PATTERNS)
        if col:
            if verbose:
                print(f"[load] Auto-detected gene ID column: '{col}'")
            df = df.rename(columns={col: "ensembl_gene_id"})

    if "gene" not in set(df.columns):
        raise ValueError(
            f"Gene column not found in {input_path}, available columns: {sorted(set(df.columns))}"
        )
    df["gene"] = df["gene"].fillna("").astype(str).apply(short_gene_name)

    if "ensembl_gene_id" not in set(df.columns):
        if verbose:
            print("[load] Resolving Ensembl gene IDs from gene symbols")
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
        if verbose:
            print("[load] Finished resolving Ensembl gene IDs")

    if "canonical_gene_name" not in set(df.columns):
        if verbose:
            print("[load] Resolving canonical gene names from Ensembl gene IDs")
        iterator = df["ensembl_gene_id"]
        if progress:
            iterator = tqdm(
                iterator, total=len(df), desc="Resolving canonical gene names"
            )
        df["canonical_gene_name"] = [
            get_canonical_gene_name_from_gene_ids_string(gene_ids_string)
            for gene_ids_string in iterator
        ]

    if "gene_display_name" not in set(df.columns):
        if verbose:
            print("[load] Computing display labels for genes")
        iterator = df.canonical_gene_name
        if progress:
            iterator = tqdm(iterator, total=len(df), desc="Formatting display names")
        df["gene_display_name"] = [
            ";".join([display_name(gene_name) for gene_name in gene_names.split(";")])
            for gene_names in iterator
        ]
    if verbose:
        print(
            f"[load] Expression data ready: {len(df)} rows, columns={list(df.columns)}"
        )
    return df
