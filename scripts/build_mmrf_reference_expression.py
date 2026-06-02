#!/usr/bin/env python
"""Build the packaged MMRF CoMMpass multiple-myeloma expression reference.

Inputs are open GDC STAR-counts gene expression files for project
``MMRF-COMMPASS``. The raw per-sample matrices are not packaged; this script
writes cohort summaries and a sample/file provenance manifest into the shared
``cancer-reference-expression`` tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
from pirlygenes.expression.normalize import clean_tpm_matrix as _clean_tpm, technical_rna_mask as _technical_mask


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
PROJECT_ID = "MMRF-COMMPASS"
CANCER_CODE = "MM"
SOURCE_COHORT = "MMRF_COMMPASS"
SOURCE_PROJECT = "MMRF CoMMpass"
SOURCE_URL = "https://portal.gdc.cancer.gov/projects/MMRF-COMMPASS"
PIPELINE = "gdc_star_counts_tpm_ensembl112_clean_tpm_v1"
SOURCE_VERSION = (
    "GDC STAR - Counts, GENCODE v36; Ensembl IDs harmonized to "
    "Ensembl release 112; queried 2026-05-18"
)
STAR_TPM_COL = "tpm_unstranded"


def _strip_version(value: object) -> str:
    return str(value).split(".", 1)[0]


def _gdc_filters() -> dict:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [PROJECT_ID],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": ["RNA-Seq"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {"field": "access", "value": ["open"]},
            },
        ],
    }


def _query_manifest(size: int = 2000) -> list[dict]:
    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.samples.submitter_id",
        "cases.samples.sample_type",
        "cases.diagnoses.primary_diagnosis",
        "analysis.workflow_type",
    ]
    params = {
        "filters": _gdc_filters(),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(size),
    }
    request = urllib.request.Request(
        GDC_FILES_ENDPOINT,
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)
    hits = payload["data"]["hits"]
    total = int(payload["data"]["pagination"]["total"])
    if len(hits) != total:
        raise RuntimeError(f"GDC query returned {len(hits)} of {total} files")
    return hits


def _only_or_empty(items: list[dict], key: str) -> str:
    if not items:
        return ""
    value = items[0].get(key, "")
    return "" if value is None else str(value)


def _flatten_hit(hit: dict) -> dict:
    case = hit.get("cases", [{}])[0]
    sample = case.get("samples", [{}])[0]
    diagnosis = case.get("diagnoses", [{}])[0]
    project = case.get("project", {})
    return {
        "cancer_code": CANCER_CODE,
        "source_cohort": SOURCE_COHORT,
        "source_project": SOURCE_PROJECT,
        "case_id": case.get("submitter_id", ""),
        "sample_id": sample.get("submitter_id", ""),
        "source_file_id": hit.get("file_id", ""),
        "source_file_name": hit.get("file_name", ""),
        "source_project_id": project.get("project_id", ""),
        "sample_type": sample.get("sample_type", ""),
        "primary_diagnosis": _only_or_empty([diagnosis], "primary_diagnosis"),
        "md5sum": hit.get("md5sum", ""),
        "file_size": hit.get("file_size", ""),
        "workflow_type": hit.get("analysis", {}).get("workflow_type", ""),
        "raw_unit": "TPM",
        "processing_pipeline": PIPELINE,
        "source_url": SOURCE_URL,
    }


def _sample_order(sample_id: str) -> int:
    match = re.search(r"_(\d+)_BM_CD138pos$", sample_id)
    if match:
        return int(match.group(1))
    return 9999


def _build_sample_manifest(hits: list[dict]) -> pd.DataFrame:
    manifest = pd.DataFrame([_flatten_hit(hit) for hit in hits])
    manifest["included"] = False
    manifest["exclusion_reason"] = ""
    manifest["lineage_label"] = ""
    manifest["lineage_evidence_source"] = ""

    eligible = (
        manifest["primary_diagnosis"].eq("Multiple myeloma")
        & manifest["sample_type"].eq("Primary Blood Derived Cancer - Bone Marrow")
        & manifest["sample_id"].str.endswith("BM_CD138pos")
    )
    manifest.loc[~eligible, "exclusion_reason"] = "not_primary_bm_cd138pos"

    for _case_id, group in manifest[eligible].groupby("case_id", sort=False):
        ordered = group.assign(
            _sample_order=group["sample_id"].map(_sample_order),
        ).sort_values(["_sample_order", "sample_id", "source_file_id"])
        keep_idx = ordered.index[0]
        manifest.loc[keep_idx, "included"] = True
        duplicate_idx = ordered.index[1:]
        manifest.loc[duplicate_idx, "exclusion_reason"] = (
            "duplicate_primary_bm_cd138pos"
        )

    included = manifest["included"]
    manifest.loc[included, "exclusion_reason"] = ""
    manifest.loc[included, "lineage_label"] = CANCER_CODE
    manifest.loc[included, "lineage_evidence_source"] = (
        "GDC MMRF-COMMPASS diagnosis; primary BM CD138+ sample ID"
    )
    return manifest.sort_values(["case_id", "sample_id"]).reset_index(drop=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - MD5 is used only for GDC file checksums.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(row: pd.Series, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / str(row["source_file_name"])
    expected_md5 = str(row["md5sum"])
    if path.exists() and _md5(path) == expected_md5:
        return path
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    url = f"{GDC_DATA_ENDPOINT}/{row['source_file_id']}"
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    observed_md5 = _md5(tmp_path)
    if observed_md5 != expected_md5:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {row['source_file_id']}: "
            f"{observed_md5} != {expected_md5}"
        )
    tmp_path.replace(path)
    return path


def _read_star_tpm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        usecols=["gene_id", "gene_name", "gene_type", STAR_TPM_COL],
        low_memory=False,
    )
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    df["source_gene_id"] = df["gene_id"].map(_strip_version)
    df["Symbol"] = df["gene_name"].fillna("").astype(str)
    df["gene_type"] = df["gene_type"].fillna("").astype(str)
    df["TPM"] = pd.to_numeric(df[STAR_TPM_COL], errors="coerce").fillna(0.0)
    return df[["source_gene_id", "Symbol", "gene_type", "TPM"]]


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


def _harmonize_gene_table(
    template: pd.DataFrame,
    *,
    ensembl_release: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    genome = EnsemblRelease(ensembl_release)
    rows = []
    counts = {"source_id": 0, "symbol": 0, "dropped": 0}
    for row in template[["source_gene_id", "Symbol", "gene_type"]].itertuples(
        index=False,
    ):
        source_id = str(row.source_gene_id)
        symbol = str(row.Symbol)
        gene = _gene_by_id(genome, source_id)
        if gene is not None:
            rows.append({
                "source_gene_id": source_id,
                "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
                "Symbol": gene.gene_name or symbol,
                "gene_type": row.gene_type,
            })
            counts["source_id"] += 1
            continue
        gene = _unique_gene_by_symbol(genome, symbol)
        if gene is not None:
            rows.append({
                "source_gene_id": source_id,
                "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
                "Symbol": gene.gene_name or symbol,
                "gene_type": row.gene_type,
            })
            counts["symbol"] += 1
            continue
        counts["dropped"] += 1
    mapping = pd.DataFrame(rows)
    symbol = (
        mapping.groupby("Ensembl_Gene_ID", sort=False)["Symbol"]
        .agg(lambda s: next((v for v in s if v), ""))
        .rename("Symbol")
    )
    gene_type = (
        mapping.groupby("Ensembl_Gene_ID", sort=False)["gene_type"]
        .agg(lambda s: next((v for v in s if v), ""))
        .rename("gene_type")
    )
    collapsed = pd.concat([symbol, gene_type], axis=1).reset_index()
    return mapping, counts | {"canonical_genes": len(collapsed)}


def _sample_vector(
    path: Path,
    mapping: pd.DataFrame,
    gene_ids: pd.Index,
) -> pd.Series:
    tpm = _read_star_tpm(path)
    merged = tpm.merge(
        mapping[["source_gene_id", "Ensembl_Gene_ID"]],
        on="source_gene_id",
        how="inner",
    )
    values = merged.groupby("Ensembl_Gene_ID", sort=False)["TPM"].sum()
    return values.reindex(gene_ids, fill_value=0.0)


def _build_values(
    included: pd.DataFrame,
    *,
    cache_dir: Path,
    ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    first_path = _download_file(included.iloc[0], cache_dir)
    template = _read_star_tpm(first_path)
    mapping, harmonized = _harmonize_gene_table(
        template,
        ensembl_release=ensembl_release,
    )
    gene_table = (
        mapping[["Ensembl_Gene_ID", "Symbol"]]
        .drop_duplicates("Ensembl_Gene_ID")
        .reset_index(drop=True)
    )
    gene_ids = pd.Index(gene_table["Ensembl_Gene_ID"])
    arrays = []
    sample_ids = []
    for idx, row in included.reset_index(drop=True).iterrows():
        path = _download_file(row, cache_dir)
        arrays.append(_sample_vector(path, mapping, gene_ids).to_numpy(dtype=float))
        sample_ids.append(str(row["sample_id"]))
        if (idx + 1) % 25 == 0 or idx + 1 == len(included):
            print(f"Processed {idx + 1}/{len(included)} included files.")
    values = pd.DataFrame(
        np.column_stack(arrays),
        index=gene_ids,
        columns=sample_ids,
    )
    return gene_table, values, harmonized


def _summarize(gene_table: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    clean = _clean_tpm(values, _technical_mask(gene_table))
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = CANCER_CODE
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = SOURCE_VERSION
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE
    out["notes"] = (
        "Open GDC MMRF-COMMPASS STAR-counts TPMs; one deterministic "
        "primary bone-marrow CD138+ sample per case; GENCODE v36 IDs "
        "harmonized to Ensembl release 112."
    )
    return round_stat_columns(out)[list(REFERENCE_COLUMNS)]


def _upsert_summary(
    path: Path,
    new_rows: pd.DataFrame,
    *,
    cancer_code: str,
    source_cohort: str,
) -> pd.DataFrame:
    """Per-gene reference write — uses the sharded layout."""
    return upsert_to_shard(
        path,
        new_rows,
        source_cohort=source_cohort,
        cancer_codes=[cancer_code],
    )


def _upsert_samples(
    path: Path,
    new_rows: pd.DataFrame,
    *,
    cancer_code: str,
    source_cohort: str,
) -> pd.DataFrame:
    """Samples manifest write — single CSV, not sharded."""
    if path.exists():
        existing = pd.read_csv(path)
        keep = ~(
            existing["cancer_code"].astype(str).eq(cancer_code)
            & existing["source_cohort"].astype(str).eq(source_cohort)
        )
        out = pd.concat(
            [existing[keep].reindex(columns=new_rows.columns), new_rows],
            ignore_index=True,
        )
    else:
        out = new_rows.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    hits = _query_manifest()
    manifest = _build_sample_manifest(hits)
    included = manifest[manifest["included"]].reset_index(drop=True)
    if included.empty:
        raise RuntimeError("No MMRF samples passed inclusion filters")
    gene_table, values, harmonized = _build_values(
        included,
        cache_dir=args.cache_dir,
        ensembl_release=args.ensembl_release,
    )
    summary = _summarize(gene_table, values)
    _upsert_summary(
        args.summary_output,
        summary,
        cancer_code=CANCER_CODE,
        source_cohort=SOURCE_COHORT,
    )
    _upsert_samples(
        args.samples_output,
        manifest,
        cancer_code=CANCER_CODE,
        source_cohort=SOURCE_COHORT,
    )
    excluded = len(manifest) - len(included)
    print(
        f"Wrote {len(summary)} MM genes from {len(included)} included samples "
        f"({len(manifest)} total GDC files; {excluded} excluded)."
    )
    print(
        "Gene ID harmonization: "
        f"{harmonized['source_id']} source IDs retained, "
        f"{harmonized['symbol']} remapped by unique symbol, "
        f"{harmonized['dropped']} unresolved rows dropped, "
        f"{harmonized['canonical_genes']} canonical genes."
    )


if __name__ == "__main__":
    main()
