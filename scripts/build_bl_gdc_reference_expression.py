#!/usr/bin/env python
"""Build the packaged CGCI Burkitt lymphoma expression reference.

Inputs are open GDC STAR-counts gene expression files for project
``CGCI-BLGSP``. The raw per-sample matrices are not packaged; this script
writes cohort summaries and a sample/file provenance manifest into the shared
``cancer-reference-expression`` tables.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import urllib.request
from urllib.error import URLError
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.builders.gene_mapping import resolve_symbol
from pirlygenes.gene_ids import strip_version as _strip_version
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    write_reference_rows,
)
from pirlygenes.expression.normalize import clean_tpm_matrix as _clean_tpm, technical_rna_mask as _technical_mask


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
PROJECT_ID = "CGCI-BLGSP"
CANCER_CODE = "BL"
SOURCE_COHORT = "CGCI_BLGSP"
SOURCE_PROJECT = "CGCI Burkitt Lymphoma Genome Sequencing Project"
SOURCE_URL = "https://portal.gdc.cancer.gov/projects/CGCI-BLGSP"
PIPELINE = "gdc_star_counts_tpm_ensembl112_clean_tpm_16_9_75"
SOURCE_VERSION = (
    "GDC STAR - Counts, GENCODE v36; Ensembl IDs harmonized to "
    "Ensembl release 112; queried 2026-05-19"
)
STAR_TPM_COL = "tpm_unstranded"
SAMPLE_COLUMNS = [
    "cancer_code",
    "source_cohort",
    "source_project",
    "case_id",
    "sample_id",
    "source_file_id",
    "source_file_name",
    "source_project_id",
    "sample_type",
    "primary_diagnosis",
    "md5sum",
    "file_size",
    "workflow_type",
    "raw_unit",
    "processing_pipeline",
    "source_url",
    "lineage_evidence_source",
    "included",
    "exclusion_reason",
    "lineage_label",
]
BL_DIAGNOSES = {
    "Burkitt lymphoma, NOS (Includes all variants)",
    "Burkitt-like lymphoma",
}


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
    with urllib.request.urlopen(request, timeout=120) as response:
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
        "lineage_evidence_source": "",
        "included": False,
        "exclusion_reason": "",
        "lineage_label": "",
    }


def _build_sample_manifest(hits: list[dict]) -> pd.DataFrame:
    manifest = pd.DataFrame([_flatten_hit(hit) for hit in hits])
    eligible = (
        manifest["sample_type"].eq("Primary Tumor")
        & manifest["primary_diagnosis"].isin(BL_DIAGNOSES)
    )
    manifest.loc[~eligible, "exclusion_reason"] = "not_primary_burkitt_tumor"
    for _case_id, group in manifest[eligible].groupby("case_id", sort=False):
        ordered = group.sort_values(["sample_id", "source_file_id"])
        keep_idx = ordered.index[0]
        manifest.loc[keep_idx, "included"] = True
        duplicate_idx = ordered.index[1:]
        manifest.loc[duplicate_idx, "exclusion_reason"] = (
            "duplicate_primary_burkitt_tumor"
        )
    included = manifest["included"]
    manifest.loc[included, "exclusion_reason"] = ""
    manifest.loc[included, "lineage_label"] = CANCER_CODE
    manifest.loc[included, "lineage_evidence_source"] = (
        "GDC CGCI-BLGSP primary diagnosis and primary-tumor sample type"
    )
    return manifest[SAMPLE_COLUMNS].sort_values(
        ["case_id", "sample_id", "source_file_id"],
    ).reset_index(drop=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - MD5 is used only for GDC checksums.
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
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, tmp_path.open(
                "wb",
            ) as handle:
                shutil.copyfileobj(response, handle)
            break
        except (TimeoutError, URLError, http.client.RemoteDisconnected) as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(
                f"Retrying {row['source_file_id']} after download error "
                f"({attempt}/3): {exc}"
            )
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
    """Symbol → Ensembl gene object via the shared resolver (direct → Entrez
    → NCBI-synonym/alias rescue) — the GENCODE-v36→Ensembl-112 fallback now
    recovers renamed symbols the same way as every other builder."""
    if not symbol:
        return None
    resolved = resolve_symbol(genome, symbol)
    return _gene_by_id(genome, resolved[0]) if resolved else None


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
    return mapping, counts | {"canonical_genes": mapping["Ensembl_Gene_ID"].nunique()}


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
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = CANCER_CODE
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = SOURCE_VERSION
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE
    out["tumor_origin"] = "primary"
    out["metastasis_site"] = pd.NA
    out["notes"] = (
        "Open GDC CGCI-BLGSP STAR-counts TPMs; one deterministic primary "
        "Burkitt/Burkitt-like tumor sample per case; GENCODE v36 IDs "
        "harmonized to Ensembl release 112 (shared resolver: direct → Entrez "
        "→ NCBI-synonym/alias rescue)."
    )
    return round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))


def _upsert_reference(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    return write_reference_rows(
        path,
        new_rows,
        source_cohort=SOURCE_COHORT,
        cancer_codes=[CANCER_CODE],
    )


def _upsert_samples(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Delegates to the shared, column-union-preserving samples-manifest upsert
    (replaces every source_cohort present in new_rows, preserving others' rows
    AND columns)."""
    from pirlygenes.expression.stats import upsert_samples_manifest
    return upsert_samples_manifest(path, new_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    manifest = _build_sample_manifest(_query_manifest())
    included = manifest[manifest["included"]].reset_index(drop=True)
    if included.empty:
        raise RuntimeError("No included BL samples")
    gene_table, values, harmonized = _build_values(
        included,
        cache_dir=args.cache_dir,
        ensembl_release=args.ensembl_release,
    )
    # Persist the per-sample raw-TPM matrix so this cohort gets medoid
    # representatives + percentiles like every other per-sample cohort
    # (registry source id == cache-dir name). The generators apply clean_tpm_16_9_75.
    from pirlygenes import cohorts as _cohorts
    _cohorts.write_per_sample(gene_table, values, args.cache_dir.name, CANCER_CODE)
    summary = _summarize(gene_table, values)
    combined_summary = _upsert_reference(args.summary_output, summary)
    combined_samples = _upsert_samples(args.samples_output, manifest)
    print(
        f"BL: {len(summary)} genes, {len(included)} included samples, "
        f"{harmonized}"
    )
    excluded = int((~manifest["included"]).sum())
    print(
        f"Wrote {len(combined_summary)} reference rows and "
        f"{len(combined_samples)} sample provenance rows "
        f"({len(manifest)} total GDC files; {excluded} excluded)."
    )


if __name__ == "__main__":
    main()
