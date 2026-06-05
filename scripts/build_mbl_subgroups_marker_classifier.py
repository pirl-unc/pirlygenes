#!/usr/bin/env python
"""MBL subgroup split via simplified marker-gene classifier.

Cavalli 2017 / Northcott 2012 / Taylor 2012 define 4 medulloblastoma
molecular subgroups (WNT / SHH / Group 3 / Group 4). Their
rigorous classifier uses a 22-gene PAM (predict-active-margin)
signature; this script approximates it by per-sample max-TPM among
4 marker genes:

  WNT      → WIF1   (canonical WNT-pathway secreted antagonist;
                     paradoxically a defining marker in MB-WNT)
  SHH      → GLI2   (SHH pathway TF; alternative: PTCH1)
  Group 3  → MYC    (MYC amplification/overexpression hallmark)
  Group 4  → KCNA1  (Group-4-specific marker per Northcott 2012)

Applied to the 125 cached Treehouse MBL samples.

Distribution expectation (published): WNT ~10%, SHH ~30%, G3 ~25%,
G4 ~35%. If observed deviates substantially, swap in the rigorous
Cavalli 2017 supplementary-table-based classifier (PMID 28617753;
sample IDs need EGAZ → ICGC_MB cross-reference).
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
    _inverse_log2,
    _read_tpm_columns,
    _technical_mask,
)
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)


RELEASE_DIR = (
    Path.home() / ".cache" / "pirlygenes" / "expression" / "treehouse-polya-25-01"
)
TPM_PATH = RELEASE_DIR / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
CLINICAL_PATH = (
    RELEASE_DIR / "clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv"
)

MARKER_TO_SUBGROUP = {
    "WIF1": "MBL_WNT",
    "GLI2": "MBL_SHH",
    "MYC": "MBL_G3",
    "KCNA1": "MBL_G4",
}
CANCER_CODES = ["MBL_WNT", "MBL_SHH", "MBL_G3", "MBL_G4"]
SOURCE_COHORT = "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    # Sample selection
    print("reading clinical for MBL samples...")
    clin = pd.read_csv(CLINICAL_PATH, sep="\t")
    mbl_samples = clin.loc[
        clin["disease"] == "medulloblastoma", "th_dataset_id"
    ].astype(str).tolist()
    print(f"  {len(mbl_samples)} MBL samples")

    print(f"reading log2 TPM for {len(mbl_samples)} columns (one disk scan)...")
    log2 = _read_tpm_columns(TPM_PATH, mbl_samples)
    tpm = _inverse_log2(log2)
    print(f"  shape: {tpm.shape}")

    # Classify per sample by max-TPM among the 4 markers
    print("classifying by marker-gene dominance...")
    marker_tpm = tpm.reindex(list(MARKER_TO_SUBGROUP.keys())).fillna(0.0)
    marker_tpm = marker_tpm.groupby(level=0).max()  # dedup any duplicate symbol rows
    marker_tpm = marker_tpm.reindex(list(MARKER_TO_SUBGROUP.keys())).fillna(0.0)
    assignments: dict[str, str] = {}
    for sample in marker_tpm.columns:
        values = marker_tpm[sample]
        if values.max() == 0:
            continue
        marker = list(MARKER_TO_SUBGROUP.keys())[int(values.values.argmax())]
        assignments[sample] = MARKER_TO_SUBGROUP[marker]
    from collections import Counter
    print(f"  {Counter(assignments.values())}")

    print(f"harmonizing HUGO → Ensembl {args.ensembl_release} (cache)...")
    mapping = _build_or_load_symbol_mapping(
        tpm.index, ensembl_release=args.ensembl_release,
        cache_path=RELEASE_DIR / "derived" / f"symbol_to_ensembl_{args.ensembl_release}.parquet",
        refresh=False,
    )
    gene_table, values = _aggregate_by_ensembl(tpm, mapping)

    summaries = []
    for code in CANCER_CODES:
        cols = [s for s, c in assignments.items() if c == code]
        if not cols:
            print(f"  skipping {code}: 0 samples")
            continue
        sub_values = values[cols]
        clean = _clean_tpm(sub_values, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = SOURCE_COHORT
        out["source_project"] = "Treehouse (MBL subgroup markers)"
        out["source_version"] = (
            "Treehouse Tumor Compendium 25.01 PolyA MBL samples × "
            "simplified marker-gene subgroup classifier "
            "(WIF1 / GLI2 / MYC / KCNA1 max-TPM)"
        )
        assign_stats(out, sub_values, clean)
        out["processing_pipeline"] = (
            f"treehouse_polya_25_01_mbl_subgroup_markers_ensembl"
            f"{args.ensembl_release}_clean_tpm_v1"
        )
        out["notes"] = (
            f"MBL subgroup = '{code.removeprefix('MBL_')}' (n={len(cols)}) "
            "via per-sample max-TPM among 4 marker genes "
            "(WIF1=WNT, GLI2=SHH, MYC=G3, KCNA1=G4). "
            "Approximation of the Cavalli 2017 (PMID 28617753) PAM "
            "22-gene classifier; rigorous version would replace this "
            "with their published per-sample labels."
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
