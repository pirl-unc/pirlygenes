#!/usr/bin/env python
"""Build packaged CLL-map expression reference summaries.

Input is the public CLL-map raw TPM matrix:

    cllmap_rnaseq_tpms_full.tsv.gz

The raw sample matrix is intentionally not packaged. This script writes
compact cohort-summary and sample-provenance CSVs that fit the shared
pirlygenes tumor-reference expression contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.builders import oncoref_source as _osrc
from pirlygenes.gene_ids import strip_version as _strip_version
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
from pirlygenes.expression.normalize import clean_tpm_matrix as _clean_tpm


PORTAL_NONREDUNDANT_EXCLUSIONS = {
    "CRC-0007",
    "CRC-0011",
    "CRC-0028",
    "CRC-0033",
    "DFCI-5053",
    "JB-0010",
}
SUSPECTED_NON_CLL_EXCLUSIONS = {
    "GCLL-0136",  # CLL-map downloads page flags this sample as suspected MCL.
}
SOURCE_URL = (
    "https://data.broadinstitute.org/cllmap/data/downloads/"
    "cllmap_rnaseq_tpms_full.tsv.gz"
)
PIPELINE = "cllmap_raw_tpm_gencode19_oncoref_canonical_clean_tpm_16_9_75"
SOURCE_VERSION = (
    "CLL-map RNA-SeQC v2.3.6 GENCODE19; source ids canonicalized to "
    "unversioned Ensembl via oncoref; downloaded 2026-05-18"
)


def _read_tpm_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={"Name": "Ensembl_Gene_ID", "Description": "Symbol"})
    df["source_gene_id"] = df["Ensembl_Gene_ID"].astype(str)
    df["Ensembl_Gene_ID"] = df["Ensembl_Gene_ID"].map(_strip_version)
    df["Symbol"] = df["Symbol"].fillna("").astype(str)
    return df


def _sample_manifest(sample_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sample_id in sample_cols:
        reason = ""
        if sample_id in PORTAL_NONREDUNDANT_EXCLUSIONS:
            reason = "portal_nonredundant_exclusion"
        elif sample_id in SUSPECTED_NON_CLL_EXCLUSIONS:
            reason = "suspected_mcl"
        rows.append({
            "cancer_code": "CLL",
            "source_cohort": "CLLMAP_2022",
            "source_project": "CLL-map",
            "case_id": sample_id,
            "sample_id": sample_id,
            "source_file_id": "",
            "source_file_name": "cllmap_rnaseq_tpms_full.tsv.gz",
            "sample_type": "CLL RNA-seq",
            "included": reason == "",
            "exclusion_reason": reason,
            "lineage_label": "CLL" if reason == "" else "",
            "lineage_evidence_source": (
                "CLL-map cohort" if reason == "" else "CLL-map downloads note"
            ),
            "raw_unit": "TPM",
            "processing_pipeline": PIPELINE,
            "source_url": SOURCE_URL,
        })
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, included_cols: list[str],
               *, source_id: str | None = None) -> pd.DataFrame:
    values = df[included_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if source_id:
        # Persist the per-sample raw-TPM matrix (included samples) for medoids +
        # percentiles, from the same values the summary uses.
        from pirlygenes import cohorts as _cohorts
        _cohorts.write_per_sample(
            df[["Ensembl_Gene_ID", "Symbol"]], values, source_id, "CLL")
    clean = _clean_tpm(values, gene_table=df)
    out = df[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = "CLL"
    out["source_cohort"] = "CLLMAP_2022"
    out["source_project"] = "CLL-map"
    out["source_version"] = SOURCE_VERSION
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE
    out["tumor_origin"] = "primary"
    out["notes"] = (
        "Raw CLL-map TPMs; portal duplicate exclusions applied; "
        "GCLL-0136 excluded as suspected MCL; GENCODE19 source ids "
        "canonicalized to unversioned Ensembl via oncoref (id → symbol "
        "rescue), duplicate loci summed in linear TPM space."
    )
    return round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument(
        "--ensembl-release", default=112, type=int,
        help="(Accepted for CLI compatibility; canonicalization now uses "
             "oncoref's bundled authority, not a pyensembl release.)",
    )
    args = parser.parse_args()

    df = _read_tpm_matrix(args.input)
    sample_cols = [c for c in df.columns if c not in {
        "Ensembl_Gene_ID",
        "Symbol",
        "source_gene_id",
    }]
    manifest = _sample_manifest(sample_cols)
    included = manifest.loc[manifest["included"], "sample_id"].tolist()
    # Canonicalize GENCODE19 source ids → unversioned Ensembl via oncoref,
    # summing duplicate loci in linear TPM space. The versioned source id is the
    # row id; the source Symbol rescues retired ids with a still-current symbol.
    matrix = df.set_index("source_gene_id")[sample_cols]
    canon = _osrc.canonicalize_source(
        matrix, row_id_name="source_gene_id", symbols=df["Symbol"].tolist(),
    )
    # No per-sample sample_qc() gate here (unlike the generic geo-matrix build):
    # every mapped sample enters the stats. See oncoref_source's module docstring —
    # QC-gating the source-specific cohorts is a deliberate data-affecting follow-up.
    summary = _summarize(canon.matrix, included, source_id="cllmap")
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    upsert_to_shard(
        args.summary_output,
        summary,
        source_cohort="CLLMAP_2022",
        cancer_codes=["CLL"],
    )
    # Upsert (not bare overwrite): a plain to_csv here wiped every cohort built
    # before CLLMAP from the shared samples manifest (the v5.20.0 truncation).
    from pirlygenes.expression.stats import upsert_samples_manifest
    upsert_samples_manifest(args.samples_output, manifest)

    print(
        f"Wrote {len(summary)} genes from {len(included)} included CLL samples "
        f"({len(sample_cols)} total source columns)."
    )
    stats = canon.mapping_stats
    print(
        "Gene ID canonicalization (oncoref): "
        f"{stats['n_resolved_rows']}/{stats['n_source_rows']} source rows "
        f"resolved, {stats['n_unresolved_rows']} unresolved dropped "
        f"({stats['n_high_expression_unresolved_rows']} high-expression)."
    )


if __name__ == "__main__":
    main()
