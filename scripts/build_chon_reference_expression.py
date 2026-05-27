#!/usr/bin/env python
"""Per-sample chondrosarcoma builder from GSE299759 (Meijer 2026).

Replaces the summary-only import for CHON with per-sample raw-counts
→ length-normalized TPM rollup. Source file is a single Ensembl-keyed
raw-counts TSV (54 samples) with no header for the gene_id column —
first column is the gene ID, remaining columns are sample IDs.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)


SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE299nnn/GSE299759/suppl/"
    "GSE299759_raw_counts.tsv.gz"
)
CANCER_CODE = "CHON"
SOURCE_COHORT = "GSE299759_MEIJER_2026"
SOURCE_PROJECT = "GEO"
PIPELINE = "gse299759_chondrosarcoma_raw_counts_to_tpm_ensembl{ensembl}_clean_tpm_v1"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _read_counts(path: Path) -> pd.DataFrame:
    # First column is the gene_id (no header); remaining columns are samples.
    with gzip.open(path, "rt") as h:
        df = pd.read_csv(h, sep="\t", index_col=0)
    df.index.name = "source_gene_id"
    return df


def _gene_length(gene) -> float:
    """Sum of unique exon lengths (gene span as a fallback)."""
    try:
        return float(gene.length)
    except Exception:
        return float("nan")


def _harmonize_and_lengths(gene_ids: pd.Index, ensembl_release: int) -> pd.DataFrame:
    genome = EnsemblRelease(ensembl_release)
    rows = []
    counts = {"resolved": 0, "dropped": 0}
    for raw_id in gene_ids:
        sid = str(raw_id).split(".", 1)[0]
        try:
            gene = genome.gene_by_id(sid)
        except Exception:
            gene = None
        if gene is None:
            counts["dropped"] += 1
            continue
        rows.append({
            "source_gene_id": str(raw_id),
            "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
            "Symbol": gene.gene_name or sid,
            "gene_length_kb": _gene_length(gene) / 1000.0,
        })
        counts["resolved"] += 1
    return pd.DataFrame(rows), counts


def _counts_to_tpm(counts_df: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = mapping.merge(
        counts_df.reset_index(),
        on="source_gene_id", how="inner",
    )
    sample_cols = [c for c in merged.columns if c not in {"source_gene_id", "Ensembl_Gene_ID", "Symbol", "gene_length_kb"}]
    # RPK per row (counts / length_kb)
    lengths_kb = merged["gene_length_kb"].astype(float).replace(0, np.nan)
    rpk = merged[sample_cols].div(lengths_kb, axis=0).fillna(0.0)
    rpk["Ensembl_Gene_ID"] = merged["Ensembl_Gene_ID"]
    rpk["Symbol"] = merged["Symbol"]
    # Sum by ENSG, then per-sample TPM (sum to 1e6)
    by_gene = rpk.groupby("Ensembl_Gene_ID", as_index=False).agg(
        {"Symbol": "first", **{c: "sum" for c in sample_cols}}
    )
    values_only = by_gene[sample_cols]
    sample_sums = values_only.sum(axis=0)
    tpm = values_only.div(sample_sums.where(sample_sums > 0), axis=1).fillna(0.0) * 1_000_000.0
    gene_table = by_gene[["Ensembl_Gene_ID", "Symbol"]].copy()
    tpm.index = gene_table["Ensembl_Gene_ID"]
    return gene_table, tpm


def _technical_mask(gene_table: pd.DataFrame) -> pd.Series:
    remove = {str(g) for g in _TECHNICAL_RNA_GROUPS}
    qc = [classify_gene_qc(sym, ensembl_id=ensg)
          for sym, ensg in zip(gene_table["Symbol"], gene_table["Ensembl_Gene_ID"])]
    return pd.Series([k.group in remove for k in qc], index=gene_table.index)


def _clean_tpm(values: pd.DataFrame, removable: pd.Series) -> pd.DataFrame:
    clean = values.copy()
    clean.loc[removable.to_numpy(), :] = 0.0
    remaining = clean.sum(axis=0)
    scale = pd.Series(np.nan, index=remaining.index, dtype=float)
    pos = remaining > 0
    scale.loc[pos] = 1_000_000.0 / remaining.loc[pos]
    return clean.mul(scale, axis=1).fillna(0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression" / "gse299759-chon",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = args.cache_dir / "GSE299759_raw_counts.tsv.gz"
    print(f"downloading {file_path.name}...")
    _download(SOURCE_URL, file_path)

    print("reading raw counts...")
    counts_df = _read_counts(file_path)
    print(f"  shape: {counts_df.shape} (genes × samples)")

    print(f"harmonizing Ensembl IDs (release {args.ensembl_release}) + computing gene lengths...")
    mapping, harm_counts = _harmonize_and_lengths(counts_df.index, args.ensembl_release)
    print(f"  resolved={harm_counts['resolved']}, dropped={harm_counts['dropped']}")

    print("counts → TPM (length-normalized + per-sample renormalize to 1e6)...")
    gene_table, tpm = _counts_to_tpm(counts_df, mapping)
    print(f"  canonical genes: {len(gene_table)}")

    print("computing stats...")
    clean = _clean_tpm(tpm, _technical_mask(gene_table))
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = CANCER_CODE
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = (
        f"GSE299759 (Meijer 2026, chondrosarcoma) raw counts; "
        f"length-normalized to TPM with Ensembl release "
        f"{args.ensembl_release} gene lengths; downloaded from GEO supplementary"
    )
    assign_stats(out, tpm, clean)
    out["processing_pipeline"] = PIPELINE.format(ensembl=args.ensembl_release)
    out["notes"] = (
        f"Per-sample TPMs from GSE299759 (Meijer 2026, "
        f"n={tpm.shape[1]} chondrosarcoma). Source data is "
        "raw HTSeq counts keyed by Ensembl ID; length-normalized "
        f"to TPM with Ensembl release {args.ensembl_release} gene "
        "lengths. TPM_clean computed per-sample by technical-RNA "
        "zeroing + denominator rescaling."
    )
    out = round_stat_columns(out)[list(REFERENCE_COLUMNS)]
    print(f"  built {len(out)} gene rows for {CANCER_CODE}")

    upsert_to_shard(
        args.summary_output, out,
        source_cohort=SOURCE_COHORT, cancer_codes=[CANCER_CODE],
    )
    print(f"upserted {len(out)} rows into shard {SOURCE_COHORT}.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
