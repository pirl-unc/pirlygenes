#!/usr/bin/env python
"""SCLC molecular subtype split via TF-dominance (Rudin 2019, Gay 2021).

Splits the 81 SCLC UCologne 2015 samples into 4 subtypes by which
transcription factor is dominantly expressed (highest TPM among
ASCL1 / NEUROD1 / POU2F3 / YAP1). This is the simplified version of
the Gay 2021 (PMID 33442380) classifier — they use NMF / consensus
clustering, but the resulting subtype labels are well-approximated
by simple TF dominance. Suitable when the rigorous Gay supplements
aren't easily joinable.

Outputs 4 cohorts under
``source_cohort=SCLC_UCOLOGNE_2015_TF_DOMINANCE``.

For more rigorous calls, replace this with the Gay 2021 NMF
classifier or with their published per-sample labels (PMID 33442380
supplements; sample IDs may need cross-referencing).
"""

from __future__ import annotations

import argparse
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


SOURCE_CACHE = (
    Path.home() / ".cache" / "pirlygenes" / "expression" / "sclc-ucologne-2015"
)
FPKM_FILE = SOURCE_CACHE / "data_mrna_seq_fpkm.txt"
CANCER_CODES = ["SCLC_ASCL1", "SCLC_NEUROD1", "SCLC_POU2F3", "SCLC_YAP1"]
TF_SYMBOLS = ["ASCL1", "NEUROD1", "POU2F3", "YAP1"]
SOURCE_COHORT = "SCLC_UCOLOGNE_2015_TF_DOMINANCE"


def _read_fpkm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["Hugo_Symbol"])
    df = df.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"], errors="ignore")
    df.index.name = "source_symbol"
    df = df[df.index.astype(str).str.strip() != ""]
    return df


def _fpkm_to_tpm(fpkm: pd.DataFrame) -> pd.DataFrame:
    sums = fpkm.sum(axis=0)
    return fpkm.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0


def _assign_subtype(tpm: pd.DataFrame) -> dict[str, str]:
    """Map sample → dominant-TF subtype label."""
    # Deduplicate the index first (some HUGO symbols may appear twice).
    deduped = tpm.groupby(level=0).max()
    tf_tpm = deduped.reindex(TF_SYMBOLS).fillna(0.0)
    # Argmax per column → TF index → SCLC_<TF> label
    out: dict[str, str] = {}
    for sample in tf_tpm.columns:
        values = tf_tpm[sample]
        if values.max() == 0:
            # No TF expressed at all — skip
            continue
        winner = TF_SYMBOLS[int(values.values.argmax())]
        out[sample] = f"SCLC_{winner.upper()}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    if not FPKM_FILE.exists():
        raise SystemExit(
            f"missing {FPKM_FILE}. Run "
            "`python scripts/build_sclc_reference_expression.py` first "
            "to download + cache the FPKM matrix."
        )

    print("reading SCLC FPKM + converting to TPM...")
    fpkm = _read_fpkm(FPKM_FILE)
    tpm = _fpkm_to_tpm(fpkm)
    print(f"  shape: {tpm.shape}")

    print("assigning TF-dominance subtypes (by max TPM of ASCL1/NEUROD1/POU2F3/YAP1)...")
    assignments = _assign_subtype(tpm)
    from collections import Counter
    print(f"  {Counter(assignments.values())}")

    print(f"harmonizing HUGO → Ensembl {args.ensembl_release} (cache)...")
    mapping = _build_or_load_symbol_mapping(
        tpm.index, ensembl_release=args.ensembl_release,
        cache_path=SOURCE_CACHE / f"symbol_to_ensembl_{args.ensembl_release}.parquet",
        refresh=False,
    )
    gene_table, values = _aggregate_by_ensembl(tpm, mapping)

    summaries = []
    for code in CANCER_CODES:
        cols = [s for s, c in assignments.items() if c == code]
        if not cols:
            print(f"  skipping {code}: no samples (TF not dominant in any sample)")
            continue
        sub_values = values[cols]
        clean = _clean_tpm(sub_values, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = SOURCE_COHORT
        out["source_project"] = "University of Cologne (TF-dominance)"
        out["source_version"] = (
            "SCLC UCologne 2015 FPKM (George 2015 PMID 26168399) × "
            "TF-dominance subtype assignment (max TPM of ASCL1 / "
            "NEUROD1 / POU2F3 / YAP1 per sample)"
        )
        assign_stats(out, sub_values, clean)
        out["processing_pipeline"] = (
            f"sclc_ucologne_2015_tf_dominance_ensembl"
            f"{args.ensembl_release}_clean_tpm_v1"
        )
        out["notes"] = (
            f"SCLC {code.removeprefix('SCLC_')}-dominant subtype "
            f"(n={len(cols)}). Subtype assigned per sample by the "
            "TF with highest TPM among ASCL1/NEUROD1/POU2F3/YAP1. "
            "Approximation of the Gay 2021 (PMID 33442380) "
            "NMF-derived classifier; rigorous version would replace "
            "this with their published per-sample labels."
        )
        out = round_stat_columns(out)[list(REFERENCE_COLUMNS)]
        summaries.append(out)
        print(f"  {code}: n={len(cols)} → {len(out)} gene rows")

    combined = pd.concat(summaries, ignore_index=True)
    upsert_to_shard(
        args.summary_output, combined,
        source_cohort=SOURCE_COHORT, cancer_codes=CANCER_CODES,
    )
    print(f"upserted {len(combined)} rows into shard {SOURCE_COHORT}.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
