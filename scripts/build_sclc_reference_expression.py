#!/usr/bin/env python
"""Per-sample SCLC builder from cBioPortal sclc_ucologne_2015 datahub.

Source: SCLC UCologne 2015 (George 2015, PMID 26168399). Per-sample
RNA-seq FPKM matrix downloaded via cBioPortal datahub (Git LFS), then
converted to TPM by per-column sum-to-1e6, then three-compartment fixed-fraction
clean-TPM, then v5.3 stat suite.

Replaces the old summary-only import for SCLC.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes.builders.treehouse import (
    _aggregate_by_ensembl,
    _build_or_load_symbol_mapping,
    _clean_tpm,
    _technical_mask,
)
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)


SOURCE_URL = (
    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/"
    "public/sclc_ucologne_2015/data_mrna_seq_fpkm.txt"
)
CANCER_CODE = "SCLC"
SOURCE_COHORT = "SCLC_UCOLOGNE_2015"
SOURCE_PROJECT = "University of Cologne"
PIPELINE = "sclc_ucologne_2015_fpkm_to_tpm_ensembl{ensembl}_clean_tpm_16_9_75"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _read_fpkm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Hugo_Symbol + Entrez_Gene_Id columns, rest are samples
    df = df.dropna(subset=["Hugo_Symbol"])
    df = df.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"], errors="ignore")
    df.index.name = "source_symbol"
    # Drop rows with empty symbol
    df = df[df.index.astype(str).str.strip() != ""]
    return df


def _fpkm_to_tpm(fpkm: pd.DataFrame) -> pd.DataFrame:
    """Per-sample sum-to-1e6 conversion."""
    sums = fpkm.sum(axis=0)
    return fpkm.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression" / "sclc-ucologne-2015",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = args.cache_dir / "data_mrna_seq_fpkm.txt"
    print(f"downloading {file_path.name} (cBioPortal datahub via LFS)...")
    _download(SOURCE_URL, file_path)

    print("reading FPKM matrix...")
    fpkm = _read_fpkm(file_path)
    n_samples = fpkm.shape[1]
    print(f"  shape: {fpkm.shape} (genes × samples)")

    print(f"FPKM → TPM (per-sample sum-to-1e6)...")
    tpm_by_symbol = _fpkm_to_tpm(fpkm)

    print(f"harmonizing HUGO symbols → Ensembl {args.ensembl_release}...")
    mapping_cache = args.cache_dir / f"symbol_to_ensembl_{args.ensembl_release}.parquet"
    mapping = _build_or_load_symbol_mapping(
        fpkm.index,
        ensembl_release=args.ensembl_release,
        cache_path=mapping_cache,
        refresh=False,
    )
    gene_table, values = _aggregate_by_ensembl(tpm_by_symbol, mapping)
    print(f"  canonical genes: {len(gene_table)}")

    print("computing stats...")
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = CANCER_CODE
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = (
        "SCLC UCologne 2015 (George 2015, PMID 26168399) RNA-seq FPKM "
        f"from cBioPortal datahub sclc_ucologne_2015; FPKM → TPM by "
        f"per-sample sum-to-1e6; HUGO symbols harmonized to Ensembl "
        f"release {args.ensembl_release}"
    )
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE.format(ensembl=args.ensembl_release)
    out["notes"] = (
        f"Per-sample TPMs from SCLC UCologne 2015 (n={n_samples} "
        "RNA-seq samples; some patients also have microarray which "
        "is not included here). Source FPKMs from cBioPortal datahub; "
        "FPKM → TPM by per-sample sum-to-1e6 (the per-sample TPM "
        f"identity). HUGO symbols harmonized to Ensembl release "
        f"{args.ensembl_release}; duplicate symbol mappings dropped. "
        "TPM_clean computed per-sample by three-compartment fixed-fraction clean-TPM (ribosomal-protein 16% / other-technical 9% / biological 75%, each renormalized within its compartment) + "
        "denominator rescaling."
    )
    out["tumor_origin"] = "primary"  # George 2015 is a primary-tumor cohort
    # reindex (not strict select): tumor_origin / metastasis_site aren't set
    # by assign_stats; reindex backfills metastasis_site as NaN.
    out = round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))
    print(f"  built {len(out)} gene rows for {CANCER_CODE}")

    upsert_to_shard(
        args.summary_output, out,
        source_cohort=SOURCE_COHORT, cancer_codes=[CANCER_CODE],
    )
    print(f"upserted {len(out)} rows into shard {SOURCE_COHORT}.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
