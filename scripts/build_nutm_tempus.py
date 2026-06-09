#!/usr/bin/env python
"""Per-sample TPM for the NUT carcinoma (NUTM) cohort from the UNC NUTM1 case
series (whole-transcriptome RNA-seq; institutional cohort).

NUT carcinoma has no usable *public* primary-tumor bulk expression (the largest
cohort is DUA-walled; EGA has 1-3 controlled-access BAMs), so the Treehouse
compendium left NUTM at n=1 — a degenerate reference. This packages the UNC
case series' gene-level TPM (``gene_tpm_cognizant_corrector`` keyed by Ensembl
id) into the standard per-sample parquet, taking NUTM to n=3.

Privacy: only the **anonymized medoid** vectors (``NUTM_rep01`` …) are shipped
in the bundle (the representatives generator anonymizes ids); the per-sample
parquet — keyed by the source sample id — lives in the local cache only and is
never published. No patient/sample ids enter the shipped artifacts.

Run:  python scripts/build_nutm_tempus.py [--data-dir ~/data/unc-nutm1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes import cohorts as _cohorts
from pirlygenes.gene_ids import gene_for_ensembl_id, strip_version
from pirlygenes.expression.stats import build_reference_rows, upsert_to_shard

SOURCE_ID = "unc-nutm1"
SOURCE_COHORT = "UNC_NUTM1"
CANCER_CODE = "NUTM"
ENSEMBL = 112


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / "data" / "unc-nutm1")
    ap.add_argument("--summary-output", type=Path,
                    default=Path("pirlygenes/data/cancer-reference-expression"))
    ap.add_argument("--ensembl-release", default=ENSEMBL, type=int)
    args = ap.parse_args()

    rna = (args.data_dir / "data_backfill" / "Data" / "Group_Level_Molecular"
           / "normalized_rna.csv")
    print(f"reading {rna} ...", flush=True)
    df = pd.read_csv(rna, usecols=["partner_sample_id", "ensembl_gene",
                                   "gene_tpm_cognizant_corrector"])
    df["ensembl_gene"] = df["ensembl_gene"].map(strip_version)
    # genes × samples TPM matrix (source sample id as the column)
    wide = df.pivot_table(index="ensembl_gene", columns="partner_sample_id",
                          values="gene_tpm_cognizant_corrector", aggfunc="sum")
    sample_cols = list(wide.columns)
    print(f"  {wide.shape[0]} genes × {len(sample_cols)} samples", flush=True)

    # Renormalize each sample to 1e6 (the corrector TPM is a per-gene subset and
    # may not sum to exactly 1e6) so the per-sample basis matches every other
    # cohort before the generators apply clean_tpm_v4.
    sums = wide[sample_cols].sum(axis=0)
    wide[sample_cols] = wide[sample_cols].div(sums.where(sums > 0), axis=1) * 1_000_000.0

    genome = EnsemblRelease(args.ensembl_release)
    symbols = []
    for ensg in wide.index:
        g = gene_for_ensembl_id(genome, ensg)
        symbols.append(g.gene_name if g is not None else ensg)
    gene_table = pd.DataFrame({
        "Ensembl_Gene_ID": list(wide.index),
        "Symbol": symbols,
    })
    values = wide.reset_index(drop=True)

    path = _cohorts.write_per_sample(gene_table, values, SOURCE_ID, CANCER_CODE)
    print(f"wrote {CANCER_CODE}: {len(gene_table)} genes × {len(sample_cols)} "
          f"samples -> {path}", flush=True)

    # Summary reference (so the packaged NUTM reference matches the n=3 medoid
    # provenance, not the legacy Treehouse n=1 — issue #344).
    out = build_reference_rows(
        gene_table, values,
        cancer_code=CANCER_CODE,
        source_cohort=SOURCE_COHORT,
        source_project="UNC NUTM1 case series",
        source_version=("UNC NUTM1 case series; whole-transcriptome RNA-seq "
                        "gene-level TPM; Ensembl release "
                        f"{args.ensembl_release}"),
        processing_pipeline="unc_nutm1_gene_tpm_renorm1e6_clean_tpm_v4",
        notes=(f"NUT carcinoma, UNC case series (n={len(sample_cols)}; "
               "whole-transcriptome RNA-seq). The only per-sample NUTM "
               "expression; public NUT data is cell-line / controlled-access "
               "only."),
        tumor_origin="primary",
    )
    upsert_to_shard(args.summary_output, out, source_cohort=SOURCE_COHORT,
                    cancer_codes=[CANCER_CODE])
    print(f"upserted {len(out)} NUTM summary rows ({SOURCE_COHORT})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
