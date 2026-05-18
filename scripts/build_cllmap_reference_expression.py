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

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc


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
PIPELINE = "cllmap_raw_tpm_gencode19_ensembl112_clean_tpm_v1"
SOURCE_VERSION = (
    "CLL-map RNA-SeQC v2.3.6 GENCODE19; Ensembl IDs harmonized to "
    "Ensembl release 112; downloaded 2026-05-18"
)


def _strip_version(value: object) -> str:
    return str(value).split(".", 1)[0]


def _read_tpm_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={"Name": "Ensembl_Gene_ID", "Description": "Symbol"})
    df["source_gene_id"] = df["Ensembl_Gene_ID"].astype(str)
    df["Ensembl_Gene_ID"] = df["Ensembl_Gene_ID"].map(_strip_version)
    df["Symbol"] = df["Symbol"].fillna("").astype(str)
    return df


def _gene_by_id(genome: EnsemblRelease, gene_id: str):
    try:
        return genome.gene_by_id(gene_id)
    except Exception:
        return None


def _unique_gene_by_symbol(genome: EnsemblRelease, symbol: str):
    if not symbol:
        return None
    try:
        genes = genome.genes_by_name(symbol)
    except Exception:
        return None
    gene_ids = {gene.gene_id.split(".", 1)[0] for gene in genes}
    if len(gene_ids) != 1:
        return None
    return genes[0]


def _harmonize_gene_ids(
    df: pd.DataFrame,
    *,
    ensembl_release: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map source GENCODE19 IDs to an Ensembl release used by CI/tests.

    CLL-map's raw matrix is GENCODE19-keyed. Most stable IDs still resolve in
    modern Ensembl; for retired IDs with an unambiguous current symbol, remap
    to the current Ensembl ID. Retired anonymous loci that cannot be resolved
    by ID or unique symbol are dropped from the packaged summary.
    """
    genome = EnsemblRelease(ensembl_release)
    by_id_cache = {}
    by_symbol_cache = {}
    canonical_ids: list[str | None] = []
    canonical_symbols: list[str | None] = []
    counts = {"source_id": 0, "symbol": 0, "dropped": 0}

    for source_id, symbol in zip(df["Ensembl_Gene_ID"], df["Symbol"]):
        gene = by_id_cache.get(source_id)
        if source_id not in by_id_cache:
            gene = _gene_by_id(genome, str(source_id))
            by_id_cache[source_id] = gene
        if gene is not None:
            canonical_ids.append(gene.gene_id.split(".", 1)[0])
            canonical_symbols.append(gene.gene_name or str(symbol))
            counts["source_id"] += 1
            continue

        symbol_key = str(symbol)
        gene = by_symbol_cache.get(symbol_key)
        if symbol_key not in by_symbol_cache:
            gene = _unique_gene_by_symbol(genome, symbol_key)
            by_symbol_cache[symbol_key] = gene
        if gene is not None:
            canonical_ids.append(gene.gene_id.split(".", 1)[0])
            canonical_symbols.append(gene.gene_name or symbol_key)
            counts["symbol"] += 1
            continue

        canonical_ids.append(None)
        canonical_symbols.append(None)
        counts["dropped"] += 1

    out = df.copy()
    out["Ensembl_Gene_ID"] = canonical_ids
    out["Symbol"] = canonical_symbols
    out = out[out["Ensembl_Gene_ID"].notna()].reset_index(drop=True)
    return out, counts


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


def _collapse_duplicate_genes(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    value_block = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta = df[["Ensembl_Gene_ID", "Symbol", "source_gene_id"]].copy()
    work = pd.concat([meta, value_block], axis=1)
    symbol = (
        work.groupby("Ensembl_Gene_ID", sort=False)["Symbol"]
        .agg(lambda s: next((v for v in s if v), ""))
        .rename("Symbol")
    )
    source_gene_id = (
        work.groupby("Ensembl_Gene_ID", sort=False)["source_gene_id"]
        .agg(lambda s: ";".join(sorted(set(map(str, s)))))
        .rename("source_gene_id")
    )
    values = work.groupby("Ensembl_Gene_ID", sort=False)[sample_cols].sum()
    return pd.concat([symbol, source_gene_id, values], axis=1).reset_index()


def _technical_mask(df: pd.DataFrame) -> pd.Series:
    remove_groups = {str(group) for group in _TECHNICAL_RNA_GROUPS}
    qc = [
        classify_gene_qc(symbol, ensembl_id=ensg)
        for symbol, ensg in zip(df["Symbol"], df["Ensembl_Gene_ID"])
    ]
    return pd.Series([klass.group in remove_groups for klass in qc], index=df.index)


def _clean_tpm(values: pd.DataFrame, removable: pd.Series) -> pd.DataFrame:
    clean = values.copy()
    clean.loc[removable, :] = 0.0
    remaining = clean.sum(axis=0)
    scale = pd.Series(np.nan, index=remaining.index, dtype=float)
    positive = remaining > 0
    scale.loc[positive] = 1_000_000.0 / remaining.loc[positive]
    clean = clean.mul(scale, axis=1)
    return clean.fillna(0.0)


def _summarize(df: pd.DataFrame, included_cols: list[str]) -> pd.DataFrame:
    values = df[included_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    clean = _clean_tpm(values, _technical_mask(df))
    out = df[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = "CLL"
    out["source_cohort"] = "CLLMAP_2022"
    out["source_project"] = "CLL-map"
    out["source_version"] = SOURCE_VERSION
    out["TPM_median"] = values.median(axis=1)
    out["TPM_q1"] = values.quantile(0.25, axis=1)
    out["TPM_q3"] = values.quantile(0.75, axis=1)
    out["TPM_mean"] = values.mean(axis=1)
    out["TPM_clean_median"] = clean.median(axis=1)
    out["TPM_clean_q1"] = clean.quantile(0.25, axis=1)
    out["TPM_clean_q3"] = clean.quantile(0.75, axis=1)
    out["n_samples"] = len(included_cols)
    out["n_detected"] = (values > 0).sum(axis=1)
    out["processing_pipeline"] = PIPELINE
    out["notes"] = (
        "Raw CLL-map TPMs; portal duplicate exclusions applied; "
        "GCLL-0136 excluded as suspected MCL; GENCODE19 IDs harmonized "
        "to Ensembl release 112 by source ID or unique symbol."
    )
    numeric_cols = [
        "TPM_median",
        "TPM_q1",
        "TPM_q3",
        "TPM_mean",
        "TPM_clean_median",
        "TPM_clean_q1",
        "TPM_clean_q3",
    ]
    out[numeric_cols] = out[numeric_cols].round(6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    df = _read_tpm_matrix(args.input)
    sample_cols = [c for c in df.columns if c not in {
        "Ensembl_Gene_ID",
        "Symbol",
        "source_gene_id",
    }]
    manifest = _sample_manifest(sample_cols)
    included = manifest.loc[manifest["included"], "sample_id"].tolist()
    df, harmonized = _harmonize_gene_ids(
        df,
        ensembl_release=args.ensembl_release,
    )
    collapsed = _collapse_duplicate_genes(df, sample_cols)
    summary = _summarize(collapsed, included)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    manifest.to_csv(args.samples_output, index=False)

    print(
        f"Wrote {len(summary)} genes from {len(included)} included CLL samples "
        f"({len(sample_cols)} total source columns)."
    )
    print(
        "Gene ID harmonization: "
        f"{harmonized['source_id']} source IDs retained, "
        f"{harmonized['symbol']} remapped by unique symbol, "
        f"{harmonized['dropped']} unresolved rows dropped."
    )


if __name__ == "__main__":
    main()
