#!/usr/bin/env python
"""Per-sample PanNET builder from GSE118014 (Alvarez 2018).

Replaces the summary-only import for PANNET with per-sample
log2(TPM+1) → TPM rollup. Source file is a single HUGO-keyed
log2(TPM+1) TSV (33 samples), same shape as the Treehouse matrix
just smaller, so we reuse the Treehouse-builder primitives
(inverse log2, HUGO → Ensembl harmonization, two-compartment fixed-fraction
clean-TPM, full v5.3 stat suite).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes.builders.treehouse import (
    _aggregate_by_ensembl,
    _build_or_load_symbol_mapping,
    _clean_tpm,
    _inverse_log2,
    _technical_mask,
)
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)


SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/GSE118014/suppl/"
    "GSE118014_PanNETs_log2TPM_33_RSEM_STAR_Process_Samples.txt.gz"
)
CANCER_CODE = "PANNET"
SOURCE_COHORT = "GSE118014_ALVAREZ_2018"
SOURCE_PROJECT = "GEO"
PIPELINE = "gse118014_pannet_log2tpm_to_tpm_ensembl{ensembl}_clean_tpm_v4"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _read_log2tpm(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as h:
        df = pd.read_csv(h, sep="\t")
    df = df.set_index(df.columns[0])
    df.index.name = "source_symbol"
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression" / "gse118014-pannet",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = args.cache_dir / "GSE118014_PanNETs_log2TPM.txt.gz"
    print(f"downloading {file_path.name}...")
    _download(SOURCE_URL, file_path)

    print("reading log2(TPM+1) matrix...")
    log2_values = _read_log2tpm(file_path)
    log2_values.index.name = "source_symbol"
    n_samples = log2_values.shape[1]
    print(f"  shape: {log2_values.shape} (genes × samples)")

    print(f"inverse-transforming + harmonizing HUGO symbols → Ensembl {args.ensembl_release}...")
    tpm = _inverse_log2(log2_values)
    mapping_cache = args.cache_dir / f"symbol_to_ensembl_{args.ensembl_release}.parquet"
    mapping = _build_or_load_symbol_mapping(
        log2_values.index,
        ensembl_release=args.ensembl_release,
        cache_path=mapping_cache,
        refresh=False,
    )
    gene_table, values = _aggregate_by_ensembl(tpm, mapping)
    print(f"  canonical genes: {len(gene_table)}")

    print("computing stats...")
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = CANCER_CODE
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = (
        "GSE118014 (Alvarez 2018, PanNETs) RSEM/STAR log2(TPM+1); "
        f"HUGO symbols harmonized to Ensembl release {args.ensembl_release}; "
        "downloaded from GEO supplementary"
    )
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE.format(ensembl=args.ensembl_release)
    out["notes"] = (
        f"Per-sample TPMs from GSE118014 (Alvarez 2018, n={n_samples}). "
        "Source data is RSEM/STAR log2(TPM+1) keyed by HUGO symbol. "
        f"HUGO symbols mapped to Ensembl release {args.ensembl_release}; "
        "duplicate symbol mappings dropped. TPM_clean computed per-sample "
        "by two-compartment fixed-fraction clean-TPM (technical 25% / biological 75%, each renormalized within its group)."
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
