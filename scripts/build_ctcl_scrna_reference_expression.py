#!/usr/bin/env python
"""Build a CTCL scRNA-derived tumor reference from GSE171811.

The GSE171811 GEO supplementary archive exposes per-sample GEX and TCR beta
matrices over the same single-cell barcodes. This builder chooses each
disease patient's dominant blood TCR beta clonotype, pseudobulks GEX counts
from cells carrying that clone across available disease blood/skin samples,
and converts the resulting case-level vectors to nTPM-like counts per million.

The raw single-cell matrices are intentionally not packaged; this script
writes compact cohort-summary and sample-provenance rows into the shared
``cancer-reference-expression`` tables.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.builders.gene_mapping import resolve_symbol
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    write_reference_rows,
)
from pirlygenes.expression.normalize import clean_tpm_matrix as _clean_tpm, technical_rna_mask as _technical_mask
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

ACCESSION = "GSE171811"
SOURCE_COHORT = "GSE171811_ECCITE_CTCL"
SOURCE_PROJECT = "GEO"
SOURCE_URL = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ACCESSION}"
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171811"
RAW_TAR = f"{ACCESSION}_RAW.tar"
SOFT_GZ = f"{ACCESSION}_family.soft.gz"
PIPELINE = "gse171811_ctcl_scrna_tcrb_pseudobulk_ntpm_ensembl112_clean_tpm_16_9_75"
SOURCE_VERSION = (
    "GSE171811 ECCITE-seq GEX/TCR beta matrices; dominant disease TCR beta "
    "clonotype pseudobulked per case; symbols harmonized to Ensembl release "
    "112; downloaded 2026-05-20"
)
NOTES = (
    "GEO GSE171811 CTCL ECCITE-seq single-cell GEX/TCR beta matrices; "
    "healthy-control samples excluded; disease cells selected by each "
    "patient's dominant blood TCR beta clonotype; clone-positive cells "
    "pseudobulked to case-level nTPM-like counts per million over retained, "
    "uniquely Ensembl-harmonized genes."
)
SAMPLE_RE = re.compile(r"^(GSM\d+)_(.+)_(Blood|Skin)_GEX\.tsv\.gz$")


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, tmp_path.open(
        "wb",
    ) as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(path)
    return path


def _parse_soft_samples(path: Path) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                current = {"geo_accession": line.split(" = ", 1)[1]}
                continue
            if current is None:
                continue
            if line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
                samples[current["geo_accession"]] = current
                continue
            if line.startswith("!Sample_geo_accession = "):
                current["geo_accession"] = line.split(" = ", 1)[1]
                continue
            if line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1]
                if ":" in value:
                    key, val = value.split(":", 1)
                    current[key.strip().lower()] = val.strip()
    return samples


def _sample_parts(name: str) -> tuple[str, str, str] | None:
    match = SAMPLE_RE.match(name)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _sample_records(tar_names: list[str]) -> list[dict[str, str]]:
    records = []
    for gex_name in sorted(name for name in tar_names if name.endswith("_GEX.tsv.gz")):
        parts = _sample_parts(gex_name)
        if parts is None:
            continue
        gsm_id, case_id, compartment = parts
        records.append({
            "gsm_id": gsm_id,
            "case_id": case_id,
            "compartment": compartment,
            "gex_name": gex_name,
            "tcrb_name": gex_name.replace("_GEX.tsv.gz", "_TCRb.tsv.gz"),
        })
    return records


def _top_tcrb_clone(tf: tarfile.TarFile, tcrb_name: str) -> tuple[str, int, int]:
    with gzip.GzipFile(fileobj=tf.extractfile(tcrb_name)) as gz:
        reader = csv.reader(io.TextIOWrapper(gz), delimiter="\t")
        header = next(reader)
        top_clone = ""
        top_count = -1
        for row in reader:
            count = sum(int(value) for value in row[1:] if value)
            if count > top_count:
                top_clone = row[0]
                top_count = count
    return top_clone, top_count, len(header) - 1


def _case_clones(
    tf: tarfile.TarFile,
    records: list[dict[str, str]],
) -> dict[str, str]:
    clones = {}
    for record in records:
        if record["case_id"] == "HC1" or record["compartment"] != "Blood":
            continue
        clone, _count, _n_cells = _top_tcrb_clone(tf, record["tcrb_name"])
        clones[record["case_id"]] = clone
    return clones


def _selected_cell_indices(
    tf: tarfile.TarFile,
    tcrb_name: str,
    clone: str,
) -> tuple[list[int], int]:
    with gzip.GzipFile(fileobj=tf.extractfile(tcrb_name)) as gz:
        reader = csv.reader(io.TextIOWrapper(gz), delimiter="\t")
        header = next(reader)
        for row in reader:
            if row[0] == clone:
                indices = [
                    idx
                    for idx, value in enumerate(row[1:], start=1)
                    if value and value != "0"
                ]
                return indices, len(header) - 1
    return [], len(header) - 1


def _add_gex_counts(
    tf: tarfile.TarFile,
    gex_name: str,
    selected_indices: list[int],
    case_counts: dict[str, defaultdict[str, float]],
    case_id: str,
) -> None:
    if not selected_indices:
        return
    with gzip.GzipFile(fileobj=tf.extractfile(gex_name)) as gz:
        reader = csv.reader(io.TextIOWrapper(gz), delimiter="\t")
        next(reader)
        for row in reader:
            count = sum(float(row[idx]) for idx in selected_indices)
            if count > 0:
                case_counts[row[0]][case_id] += count


def _build_case_counts(
    tar_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with tarfile.open(tar_path) as tf:
        records = _sample_records(tf.getnames())
        case_clone = _case_clones(tf, records)
        case_counts: dict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float),
        )
        manifest_rows = []
        for record in records:
            case_id = record["case_id"]
            is_healthy = case_id == "HC1"
            clone = "" if is_healthy else case_clone[case_id]
            selected_indices: list[int] = []
            n_cells = 0
            if not is_healthy:
                selected_indices, n_cells = _selected_cell_indices(
                    tf,
                    record["tcrb_name"],
                    clone,
                )
                _add_gex_counts(
                    tf,
                    record["gex_name"],
                    selected_indices,
                    case_counts,
                    case_id,
                )
            else:
                _clone, _count, n_cells = _top_tcrb_clone(tf, record["tcrb_name"])
            manifest_rows.append({
                "cancer_code": "CTCL",
                "source_cohort": SOURCE_COHORT,
                "source_project": SOURCE_PROJECT,
                "case_id": case_id,
                "sample_id": (
                    f"{record['gsm_id']}_{record['case_id']}_{record['compartment']}"
                ),
                "source_file_id": record["gsm_id"],
                "source_file_name": f"{RAW_TAR}:{record['gex_name']}",
                "source_project_id": ACCESSION,
                "sample_type": record["compartment"],
                "primary_diagnosis": "",
                "md5sum": "",
                "file_size": "",
                "workflow_type": "GEO scRNA GEX/TCR beta matrices",
                "raw_unit": "scRNA UMI counts",
                "processing_pipeline": PIPELINE,
                "source_url": SOURCE_URL,
                "lineage_evidence_source": (
                    f"dominant TCR beta clone {clone}; "
                    f"{len(selected_indices)} of {n_cells} cells selected"
                    if not is_healthy
                    else "healthy-control sample excluded"
                ),
                "included": not is_healthy and len(selected_indices) > 0,
                "exclusion_reason": "healthy_control" if is_healthy else "",
                "lineage_label": "CTCL" if not is_healthy else "",
            })
    cases = sorted(case_clone)
    rows = []
    for symbol, counts_by_case in case_counts.items():
        row = {"source_symbol": symbol}
        for case_id in cases:
            row[case_id] = counts_by_case.get(case_id, 0.0)
        rows.append(row)
    counts = pd.DataFrame(rows)
    return counts, pd.DataFrame(manifest_rows)


def _gene_by_symbol(genome: EnsemblRelease, symbol: str):
    """Symbol → Ensembl gene object via the shared resolver (direct → Entrez
    → NCBI-synonym/alias rescue), so renamed symbols are recovered the same
    way as in every other builder. None if unresolved/ambiguous."""
    if not symbol:
        return None
    resolved = resolve_symbol(genome, symbol)
    if resolved is None:
        return None
    try:
        return genome.gene_by_id(resolved[0])
    except Exception:
        return None


def _harmonize_symbols(
    counts: pd.DataFrame,
    *,
    ensembl_release: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    genome = EnsemblRelease(ensembl_release)
    cache = {}
    rows = []
    stats = {"symbol": 0, "dropped": 0}
    for _, row in counts.iterrows():
        symbol = str(row["source_symbol"])
        if symbol not in cache:
            cache[symbol] = _gene_by_symbol(genome, symbol)
        gene = cache[symbol]
        if gene is None:
            stats["dropped"] += 1
            continue
        out = row.drop(labels=["source_symbol"]).to_dict()
        out["Ensembl_Gene_ID"] = gene.gene_id.split(".", 1)[0]
        out["Symbol"] = gene.gene_name or symbol
        rows.append(out)
        stats["symbol"] += 1
    mapped = pd.DataFrame(rows)
    stats["canonical_genes"] = mapped["Ensembl_Gene_ID"].nunique()
    return mapped, stats


def _collapse_to_ntpm(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_cols = [
        col for col in mapped.columns if col not in {"Ensembl_Gene_ID", "Symbol"}
    ]
    values = mapped[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    work = pd.concat([mapped[["Ensembl_Gene_ID", "Symbol"]], values], axis=1)
    symbol = (
        work.groupby("Ensembl_Gene_ID", sort=False)["Symbol"]
        .agg(lambda s: next((value for value in s if value), ""))
        .rename("Symbol")
    )
    collapsed = work.groupby("Ensembl_Gene_ID", sort=False)[sample_cols].sum()
    sums = collapsed.sum(axis=0)
    ntpm = collapsed.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0
    gene_table = symbol.reindex(ntpm.index).reset_index()
    return gene_table, ntpm


def _summarize(gene_table: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = "CTCL"
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = SOURCE_VERSION
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE
    out["tumor_origin"] = "primary"   # CTCL skin/blood diagnostic samples
    out["metastasis_site"] = pd.NA
    out["notes"] = NOTES
    return round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))


def _upsert_reference(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    codes = sorted(new_rows["cancer_code"].astype(str).unique())
    return write_reference_rows(
        path, new_rows, source_cohort=SOURCE_COHORT, cancer_codes=codes,
    )


def _upsert_samples(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Delegates to the shared, column-union-preserving samples-manifest upsert
    (replaces every source_cohort present in new_rows, preserving others' rows
    AND columns)."""
    from pirlygenes.expression.stats import upsert_samples_manifest
    return upsert_samples_manifest(path, new_rows)


def _apply_soft_metadata(
    manifest: pd.DataFrame,
    samples: dict[str, dict[str, str]],
) -> pd.DataFrame:
    out = manifest.copy()
    for idx, row in out.iterrows():
        sample = samples.get(str(row["source_file_id"]), {})
        if sample:
            out.at[idx, "sample_type"] = (
                sample.get("tissue", "") or str(row["sample_type"])
            )
            out.at[idx, "primary_diagnosis"] = sample.get("disease state", "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    source_dir = args.cache_dir / ACCESSION
    tar_path = _download(f"{GEO_FTP}/suppl/{RAW_TAR}", source_dir / RAW_TAR)
    soft_path = _download(f"{GEO_FTP}/soft/{SOFT_GZ}", source_dir / SOFT_GZ)
    samples = _parse_soft_samples(soft_path)

    counts, manifest = _build_case_counts(tar_path)
    manifest = _apply_soft_metadata(manifest, samples)
    mapped, stats = _harmonize_symbols(
        counts,
        ensembl_release=args.ensembl_release,
    )
    gene_table, values = _collapse_to_ntpm(mapped)
    from pirlygenes import cohorts as _cohorts
    _cohorts.write_per_sample(gene_table, values, args.cache_dir.name, "CTCL")
    summary = _summarize(gene_table, values)
    combined_summary = _upsert_reference(args.summary_output, summary)
    combined_samples = _upsert_samples(args.samples_output, manifest)
    included = manifest["included"].astype(bool)
    print(
        f"CTCL: {len(summary)} genes, {values.shape[1]} case pseudobulks, "
        f"{included.sum()} included source samples, {stats}"
    )
    print(
        f"Wrote {len(combined_summary)} reference rows and "
        f"{len(combined_samples)} sample provenance rows."
    )


if __name__ == "__main__":
    main()
