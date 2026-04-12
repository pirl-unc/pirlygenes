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

import warnings
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
_TPM_PATTERNS = [
    r"^TPM$",
    r"^gene[_\s]?tpm",
    r"(^|[_\s])tpm($|[_\s])",
]

# Strict FPKM column patterns — match raw FPKM columns only, not derived
# columns like log2_fpkm, FPKM_zscore, FPKM_adjusted, or per-cohort
# reference columns like FPKM_COAD.  A full column-name match is required.
_RAW_FPKM_PATTERNS = [
    r"^FPKM$",
    r"^fpkm$",
    r"^gene[_\s]?fpkm$",
    r"^rna[_\s]?fpkm$",
    r"^mrna[_\s]?fpkm$",
]


def _guess_col(columns, patterns):
    """Return the first column that matches any pattern, or None."""
    for pat in patterns:
        for col in columns:
            if re.search(pat, col, re.IGNORECASE):
                return col
    return None


def _select_sample_rows(df, sample_id_col=None, sample_id_value=None, verbose=True):
    if sample_id_col is None and sample_id_value is None:
        return df
    if not sample_id_col or sample_id_value is None:
        raise ValueError(
            "Both sample_id_col and sample_id_value must be provided to select a sample"
        )
    if sample_id_col not in df.columns:
        raise ValueError(
            f"Sample column '{sample_id_col}' not found, available columns: {sorted(df.columns)}"
        )

    mask = df[sample_id_col].astype(str) == str(sample_id_value)
    if not mask.any():
        available = sorted(df[sample_id_col].dropna().astype(str).unique().tolist())
        if len(available) > 8:
            available_preview = available[:8] + ["..."]
        else:
            available_preview = available
        raise ValueError(
            f"No rows matched {sample_id_col}={sample_id_value!r}; "
            f"available values include: {available_preview}"
        )

    out = df.loc[mask].copy()
    if verbose:
        print(
            f"[load] Selected {len(out)} rows where "
            f"{sample_id_col}={sample_id_value!r}"
        )
    return out


def get_canonical_gene_name_from_gene_ids_string(gene_ids_string):
    if pd.isna(gene_ids_string):
        return ""
    gene_ids = str(gene_ids_string).split(";")
    gene_names = [find_gene_name_from_ensembl_gene_id(gene_id) for gene_id in gene_ids]
    not_none_gene_names = [name for name in gene_names if name is not None]
    return ";".join(not_none_gene_names)


_MIN_RECOMMENDED_ENSEMBL_RELEASE = 110
_ensembl_release_check_done = False


def _check_installed_ensembl_releases():
    """Warn once if no Ensembl release >= 110 is installed.

    The alt-haplotype alias mapping canonicalises to primary-contig IDs
    that exist on the primary assembly.  Many of these IDs were added or
    updated in Ensembl 110+, so older installed releases may fail to
    resolve them to gene symbols downstream, leading to missing names in
    reports and plots.
    """
    global _ensembl_release_check_done
    if _ensembl_release_check_done:
        return
    _ensembl_release_check_done = True
    try:
        from pyensembl.shell import collect_all_installed_ensembl_releases
    except ImportError:
        return
    try:
        installed = [
            g for g in collect_all_installed_ensembl_releases()
            if g.species.latin_name == "homo_sapiens"
        ]
    except Exception:
        return
    if not installed:
        warnings.warn(
            "No human Ensembl releases are installed via pyensembl. "
            "Gene symbol resolution will be incomplete. Install with: "
            f"pyensembl install --release {_MIN_RECOMMENDED_ENSEMBL_RELEASE} "
            "--species homo_sapiens",
            UserWarning,
            stacklevel=3,
        )
        return
    latest = max(g.release for g in installed)
    if latest < _MIN_RECOMMENDED_ENSEMBL_RELEASE:
        warnings.warn(
            f"Installed Ensembl release {latest} is older than the "
            f"recommended minimum ({_MIN_RECOMMENDED_ENSEMBL_RELEASE}). "
            "Recently-added alt-haplotype MHC/KIR genes may fail to "
            f"resolve to symbols. Install with: pyensembl install "
            f"--release {_MIN_RECOMMENDED_ENSEMBL_RELEASE} --species homo_sapiens",
            UserWarning,
            stacklevel=3,
        )


def _load_ensembl_id_aliases():
    """Return {alt_haplotype_id: primary_contig_id} mapping from bundled data.

    The aliases cover alt-haplotype genes (MHC, KIR, olfactory receptors on
    HSCHR6_MHC_*/HSCHR19KIR_* contigs) that have a primary-chromosome
    equivalent, plus a few retired IDs with documented successors.  Alt-
    haplotype IDs represent alleles of the same gene as the primary-contig
    ID, so their expression should be summed when both are present.

    Chains (A→B→C) are resolved transitively: if the bundled data contains
    both entries, callers get A→C directly.  Cycles and over-long chains
    are detected and raise (indicates bad input data).
    """
    from .plot_data_helpers import _strip_ensembl_version
    try:
        from .load_dataset import get_data
        df = get_data("ensembl-id-aliases")
    except Exception:
        return {}
    raw = dict(zip(
        df["alt_haplotype_id"].astype(str).map(_strip_ensembl_version),
        df["primary_contig_id"].astype(str).map(_strip_ensembl_version),
    ))
    # Transitively resolve chains so the returned dict has no A→B where B
    # is itself a key.  Cap iterations to catch cycles.
    resolved = {}
    for src in raw:
        dst = raw[src]
        seen = {src}
        for _ in range(16):
            if dst not in raw:
                break
            if dst in seen:
                raise ValueError(
                    f"Cycle in ensembl-id-aliases starting from {src} (loop at {dst})"
                )
            seen.add(dst)
            dst = raw[dst]
        else:
            raise ValueError(
                f"Chain from {src} exceeded 16 hops in ensembl-id-aliases"
            )
        resolved[src] = dst
    return resolved


def _detect_and_convert_to_tpm(df, verbose=True):
    """Convert FPKM column to TPM if the input looks like FPKM.

    TPM_i = 1e6 * FPKM_i / sum_g(FPKM_g).  This is a within-sample
    rescaling that makes values sum to 1M; it does not correct gene-length
    bias or protocol differences, but it harmonizes the units so
    downstream comparisons (e.g. sample vs. TCGA cohort) are on the same
    scale.

    Only raw FPKM columns are converted (full-name match against
    _RAW_FPKM_PATTERNS).  Derived columns like ``log2_fpkm``,
    ``FPKM_zscore``, or per-cohort reference columns like ``FPKM_COAD``
    are deliberately NOT treated as raw FPKM — those contain values that
    have already been transformed or belong to a different use case.

    Emits a UserWarning when conversion happens, since this modifies the
    values in the input DataFrame.  The warning surfaces even when
    ``verbose=False`` so callers always see that a conversion occurred.
    """
    # Detect raw FPKM column by strict pattern; derived columns are ignored
    fpkm_col = _guess_col(df.columns, _RAW_FPKM_PATTERNS)
    tpm_col = next(
        (c for c in df.columns if c.upper() == "TPM" or c.lower().endswith("_tpm")),
        None,
    )
    if fpkm_col and not tpm_col:
        total = df[fpkm_col].astype(float).sum()
        if total > 0:
            df = df.copy()
            df[fpkm_col] = 1e6 * df[fpkm_col].astype(float) / total
            df = df.rename(columns={fpkm_col: "TPM"})
            warnings.warn(
                f"FPKM column '{fpkm_col}' was converted to TPM in-place "
                f"(original sum={total:.0f}; new sum=1e6). This is a "
                "within-sample rescaling; gene-length bias is not corrected.",
                UserWarning,
                stacklevel=2,
            )
            if verbose:
                print(f"[load] Converted FPKM column '{fpkm_col}' to TPM (sum was {total:.0f})")
    return df


def _first_non_empty(series):
    """Return the first non-empty, non-null value in a series; fall back to
    first value.  Used to merge string columns (symbol, display name)
    across rows that share a canonical ID: prefer a populated value over
    an empty one regardless of row order."""
    for val in series:
        if pd.notna(val) and str(val).strip() != "":
            return val
    return series.iloc[0] if len(series) else None


def _apply_id_aliases_and_sum(df, verbose=True):
    """Map alt-haplotype Ensembl IDs to their primary-contig equivalents
    and sum TPM values (alt-haplotype IDs represent alleles of the same gene).

    Operates by ID only.  Runs AFTER symbol-based consolidation in the
    load pipeline: symbol consolidation handles inputs where alt-haplotype
    rows share a gene symbol with the primary, and this step catches
    alt-haplotype rows that lack a symbol (e.g. a sample exported with
    only Ensembl IDs).

    Sums TPM across rows that collapse onto the same canonical ID
    (alt-haplotype alleles contribute to the same gene's total expression).
    Non-TPM string columns are merged by preferring the first non-empty
    value, so a remapped alt-haplotype row with an empty symbol inherits
    the primary row's symbol.
    """
    if "ensembl_gene_id" not in df.columns:
        return df
    aliases = _load_ensembl_id_aliases()
    if not aliases:
        return df

    from .plot_data_helpers import _strip_ensembl_version

    df = df.copy()
    original_ids = df["ensembl_gene_id"].astype(str).map(_strip_ensembl_version)
    canonical_ids = original_ids.map(lambda gid: aliases.get(gid, gid))
    n_remapped = int((canonical_ids != original_ids).sum())
    if n_remapped == 0:
        return df
    df["ensembl_gene_id"] = canonical_ids

    # Sum TPM across rows that now share the same canonical ID; for other
    # columns, prefer the first non-empty value (avoids inheriting an
    # empty symbol from an alt-haplotype row just because it came first).
    tpm_col = next((c for c in df.columns if c.upper() == "TPM"), None)
    if tpm_col and df["ensembl_gene_id"].duplicated().any():
        agg_map = {tpm_col: "sum"}
        for col in df.columns:
            if col != "ensembl_gene_id" and col != tpm_col:
                agg_map[col] = _first_non_empty
        df = df.groupby("ensembl_gene_id", as_index=False).agg(agg_map)

    if verbose:
        print(f"[load] Remapped {n_remapped} alt-haplotype Ensembl IDs to primary contig and summed TPM")
    return df


def _consolidate_gene_ids(df, verbose=True):
    """Collapse alt-haplotype / retired Ensembl IDs to one ID per gene symbol.

    GENCODE transcriptomes include genes on alternate haplotype contigs
    (e.g., B2M has ENSG00000166710 on primary + ENSG00000273686 on alt).
    Salmon distributes reads across both, splitting the expression.

    For each gene symbol with multiple IDs, pick one canonical ID
    (prefer the one in our pan-cancer reference) and SUM TPM across
    all alt-haplotype copies (reads are split by the aligner, so
    summing recovers the true total).
    """
    if "ensembl_gene_id" not in df.columns:
        return df

    from .plot_data_helpers import _strip_ensembl_version
    from .gene_sets_cancer import pan_cancer_expression

    df = df.copy()
    df["_id_stripped"] = df["ensembl_gene_id"].astype(str).map(_strip_ensembl_version)

    # Find the symbol column (canonical_gene_name or gene)
    sym_col = "canonical_gene_name" if "canonical_gene_name" in df.columns else "gene"
    if sym_col not in df.columns:
        return df

    symbols = df[sym_col].fillna("").astype(str)
    # Only consolidate genes with actual symbols (skip empty/nan)
    has_symbol = symbols.str.strip().ne("") & symbols.str.upper().ne("NAN")

    # Count IDs per symbol — only process multi-ID symbols
    id_counts = df.loc[has_symbol].groupby(sym_col)["_id_stripped"].nunique()
    multi_id_symbols = set(id_counts[id_counts > 1].index)

    if not multi_id_symbols:
        df.drop(columns=["_id_stripped"], inplace=True)
        return df

    # Build set of IDs in our pan-cancer reference for canonical preference
    ref_ids = set(pan_cancer_expression()["Ensembl_Gene_ID"])

    tpm_col = next((c for c in df.columns if c.upper() == "TPM"), None)

    n_consolidated = 0
    rows_to_drop = []
    id_remap = {}  # old_id → canonical_id

    for sym in multi_id_symbols:
        mask = has_symbol & (symbols == sym)
        sub = df[mask]
        ids = sub["_id_stripped"].unique()

        # Pick canonical: prefer ID in reference, then highest TPM
        in_ref = [i for i in ids if i in ref_ids]
        if in_ref:
            canonical = in_ref[0]
        elif tpm_col:
            canonical = sub.loc[sub[tpm_col].astype(float).idxmax(), "_id_stripped"]
        else:
            canonical = ids[0]

        # Map all alt IDs to canonical
        for alt_id in ids:
            if alt_id != canonical:
                id_remap[alt_id] = canonical

        # Sum TPM across alt-haplotype copies (reads are split by aligner)
        # and keep one row with the summed value
        if tpm_col and len(sub) > 1:
            total_tpm = sub[tpm_col].astype(float).sum()
            best_idx = sub[tpm_col].astype(float).idxmax()
            drop_idx = sub.index.difference([best_idx])
            rows_to_drop.extend(drop_idx)
            df.loc[best_idx, tpm_col] = total_tpm
            df.loc[best_idx, "_id_stripped"] = canonical
            n_consolidated += 1

    if rows_to_drop:
        df.drop(index=rows_to_drop, inplace=True)

    # Apply ID remapping
    df["ensembl_gene_id"] = df["_id_stripped"]
    df.drop(columns=["_id_stripped"], inplace=True)

    if verbose and n_consolidated:
        print(
            f"[load] Consolidated {n_consolidated} genes with multiple "
            f"Ensembl IDs (alt haplotypes / retired IDs)"
        )
    return df


def _attach_gene_sidecar_if_present(input_path, df, verbose=True):
    """Hydrate one-column TPM vectors using a sibling Gene.csv sidecar.

    Some sample bundles store gene names in ``Gene.csv`` and one or more
    parallel TPM vectors in separate one-column CSVs. When such a vector is
    loaded directly, attach the gene-name sidecar so the rest of the loader can
    treat it as ordinary gene-level expression data.
    """
    if list(df.columns) != ["TPM"]:
        return df, False

    sidecar = Path(input_path).with_name("Gene.csv")
    if not sidecar.exists():
        return df, False

    gene_df = pd.read_csv(sidecar)
    if len(gene_df.columns) != 1 or len(gene_df) != len(df):
        return df, False

    out = pd.DataFrame(
        {
            "gene": gene_df.iloc[:, 0].astype(str),
            "TPM": pd.to_numeric(df["TPM"], errors="coerce").fillna(0.0),
        }
    )
    if verbose:
        print(f"[load] Attached gene names from sidecar: {sidecar}")
    return out, True


def load_expression_data(
    input_path,
    aggregate_gene_expression=False,
    save_aggregated_gene_expression=True,
    aggregated_output_path=None,
    gene_name_col=None,
    gene_id_col=None,
    sample_id_col=None,
    sample_id_value=None,
    verbose=True,
    progress=True,
):
    _check_installed_ensembl_releases()
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

    df, used_sidecar = _attach_gene_sidecar_if_present(input_path, df, verbose=verbose)
    df = _select_sample_rows(
        df,
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        verbose=verbose,
    )
    # FPKM→TPM: convert BEFORE any aggregation or ID remapping so downstream
    # consolidation (which sums values) operates on consistent units.
    df = _detect_and_convert_to_tpm(df, verbose=verbose)
    if used_sidecar and aggregate_gene_expression:
        if verbose:
            print("[load] Input is already gene-level via Gene.csv sidecar; skipping transcript aggregation")
        aggregate_gene_expression = False

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
            "ensembl_gene": "ensembl_gene_id",
            "gene_id": "ensembl_gene_id",
            "canonical_gene_id": "ensembl_gene_id",
            "GeneID": "ensembl_gene_id",
            "tpm": "TPM",
            "gene_tpm": "TPM",
            "gene_tpm_cognizant_corrector": "TPM",
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
    if "TPM" not in set(df.columns):
        col = _guess_col(df.columns, _TPM_PATTERNS)
        if col:
            if verbose:
                print(f"[load] Auto-detected TPM column: '{col}'")
            df = df.rename(columns={col: "TPM"})

    if "gene" not in set(df.columns) and "ensembl_gene_id" not in set(df.columns):
        raise ValueError(
            f"Gene column not found in {input_path}, available columns: {sorted(set(df.columns))}"
        )
    if "TPM" not in set(df.columns):
        raise ValueError(
            f"TPM column not found in {input_path}, available columns: {sorted(set(df.columns))}"
        )
    if "gene" in set(df.columns):
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
        if (
            "ensembl_gene_id" in set(df.columns)
            and df["ensembl_gene_id"].astype(str).str.strip().ne("").any()
        ):
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
        elif "gene" in set(df.columns) and df["gene"].astype(str).str.strip().ne("").any():
            if verbose:
                print("[load] Using gene symbols as canonical gene names")
            df["canonical_gene_name"] = df["gene"].fillna("").astype(str)

    if "gene" not in set(df.columns):
        if verbose:
            print("[load] Using canonical gene names as gene symbols")
        df["gene"] = df["canonical_gene_name"].fillna("").astype(str).apply(short_gene_name)
    elif "canonical_gene_name" in set(df.columns):
        missing_gene = df["gene"].astype(str).str.strip().eq("")
        if missing_gene.any():
            df.loc[missing_gene, "gene"] = (
                df.loc[missing_gene, "canonical_gene_name"]
                .fillna("")
                .astype(str)
                .apply(short_gene_name)
            )

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
    # Consolidate alternate-locus / retired Ensembl IDs that map to the
    # same gene symbol.  Pick the canonical ID (the one in our pan-cancer
    # reference, or highest-expressed) and aggregate TPM via max.
    df = _consolidate_gene_ids(df, verbose=verbose)

    # Apply ID-based aliases (alt-haplotype MHC/KIR → primary contig).
    # Runs after symbol consolidation so either approach can consolidate
    # whatever the other missed.  Sums TPM across alt-haplotype alleles.
    df = _apply_id_aliases_and_sum(df, verbose=verbose)

    if verbose:
        print(
            f"[load] Expression data ready: {len(df)} rows, columns={list(df.columns)}"
        )
    return df
