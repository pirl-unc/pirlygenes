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
from tqdm import tqdm

from .gene_ids import (
    find_gene_and_ensembl_release_by_name,
    find_gene_name_from_ensembl_transcript_id,
)
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
    for t, tpm in tqdm(
        zip(df[transcript_id_column], df[tpm_column]), "Aggregating gene expression"
    ):
        gene_name = None
        if t in tx_to_gene_name:
            gene_name = tx_to_gene_name[t]

        else:
            t = t.split(".")[0]
            if t in tx_to_gene_name:
                gene_name = tx_to_gene_name[t]
            else:
                gene_name = find_gene_name_from_ensembl_transcript_id(t, verbose=False)

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


####
# FASTER VERSION
###

from functools import lru_cache
from collections import defaultdict
import pandas as pd

from .gene_ids import (
    find_gene_and_ensembl_release_by_name,
    find_gene_name_from_ensembl_transcript_id,
)
from .transcript_to_gene import extra_tx_mappings
from .common import find_column


def _expanded_tx_map(tx_to_gene_name: dict[str, str]) -> dict[str, str]:
    """
    Expand a transcript->gene dict to include versionless keys.
    If both versioned and versionless exist, keep the first-seen value.
    """
    out = {}
    for k, v in tx_to_gene_name.items():
        if k not in out:
            out[k] = v
        k0 = k.split(".", 1)[0]
        out.setdefault(k0, v)
    return out


def aggregate_gene_expression(
    df: pd.DataFrame,
    tx_to_gene_name: dict[str, str] = extra_tx_mappings,
    transcript_id_column_candidates: list[str] = (
        "transcript transcript_id transcriptid target target_id targetid name".split()
    ),
    tpm_column_candidates: list[str] = ("tpm",),
) -> pd.DataFrame:
    # Find columns
    transcript_id_column = find_column(
        df, transcript_id_column_candidates, "transcript ID"
    )
    tpm_column = find_column(df, tpm_column_candidates, "TPM")

    # Normalize inputs
    tx_raw = df[transcript_id_column].astype(str)
    tpm = pd.to_numeric(df[tpm_column], errors="coerce").fillna(0.0)

    # Versionless transcript ids
    tx0 = tx_raw.str.split(".", n=1).str[0]

    # Fast path: map via dict (include versionless keys)
    tx_map = _expanded_tx_map(tx_to_gene_name or {})
    gene_series = tx0.map(tx_map)

    # Resolve unknowns once per unique transcript id
    unknown_mask = gene_series.isna()
    if unknown_mask.any():
        unknown_unique = pd.Index(tx0[unknown_mask].unique())

        @lru_cache(maxsize=None)
        def _resolve_tx(t: str):
            g = find_gene_name_from_ensembl_transcript_id(t, verbose=False)
            if g:
                return g
            # fall back to any late-loaded mapping
            return extra_tx_mappings.get(t)

        resolved = {t: _resolve_tx(t) for t in unknown_unique}
        gene_series.loc[unknown_mask] = tx0[unknown_mask].map(resolved)

    # Compute known/unknown TPM totals
    unknown_mask = gene_series.isna()
    unknown_genes_tpm = float(tpm[unknown_mask].sum())

    # (Optional but preserved) emit the same-style unknown prints (TPM>1)
    # NOTE: This I/O can still be a bottleneck; comment out for an extra speedup.
    debug_df = pd.DataFrame({"tx": tx0[unknown_mask], "TPM": tpm[unknown_mask]})
    debug_df = debug_df[debug_df["TPM"] > 1]
    n_unknown = 0
    for tx, tpm_val in zip(debug_df["tx"], debug_df["TPM"]):
        n_unknown += 1
        print("? ", n_unknown, tx, tpm_val)

    # Aggregate by gene (vectorized)
    known = pd.DataFrame(
        {"gene": gene_series[~unknown_mask], "TPM": tpm[~unknown_mask]}
    )
    df_gene_expr = known.groupby("gene", as_index=False, sort=False)["TPM"].sum()
    df_gene_expr = df_gene_expr.sort_values("TPM")

    known_genes_tpm = float(df_gene_expr["TPM"].sum())
    denom = known_genes_tpm + unknown_genes_tpm
    pct_known = (known_genes_tpm * 100.0 / denom) if denom > 0 else 0.0
    print(
        f"Assigned {known_genes_tpm:.2f} TPM to known genes, "
        f"{unknown_genes_tpm:.2f} to unknown gene names; {pct_known:.4f}% known"
    )

    # Gene IDs + Ensembl releases with caching
    @lru_cache(maxsize=None)
    def _lookup_gene_meta(gene_name: str):
        pair = find_gene_and_ensembl_release_by_name(gene_name)
        if pair is None:
            return (None, -1)
        ensembl_genome, gene = pair
        return (gene.id, ensembl_genome.release)

    metas = [_lookup_gene_meta(g) for g in df_gene_expr["gene"]]
    if metas:
        gene_ids, releases = zip(*metas)
    else:
        gene_ids, releases = (), ()
    df_gene_expr["gene_id"] = list(gene_ids)
    df_gene_expr["ensembl_release"] = (
        pd.Series(releases, index=df_gene_expr.index).fillna(-1).astype(int)
    )

    return df_gene_expr
