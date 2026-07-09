#!/usr/bin/env python
"""Build packaged TARGET ALL B-ALL/T-ALL expression references.

Inputs are open GDC STAR-counts gene expression files for
``TARGET-ALL-P1``/``TARGET-ALL-P2``/``TARGET-ALL-P3`` plus the public
TARGET ALL Phase 1/2 sample matrices. GDC clinical metadata only labels
these cases as acute lymphocytic leukemia, so B/T lineage comes from the
sample-matrix ``Cell of origin`` comments. Cases without an explicit
B/T lineage are retained in the provenance manifest as excluded rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
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
PROJECT_IDS = ["TARGET-ALL-P1", "TARGET-ALL-P2", "TARGET-ALL-P3"]
SOURCE_COHORT = "TARGET_ALL_2018"
SOURCE_PROJECT = "TARGET ALL"
SOURCE_URL = "https://portal.gdc.cancer.gov/projects/TARGET-ALL-P2"
PIPELINE = "gdc_star_counts_tpm_ensembl112_clean_tpm_16_9_75"
SOURCE_VERSION = (
    "GDC STAR - Counts, GENCODE v36; TARGET ALL Phase 1/2 sample "
    "matrices 2019-06-06; Ensembl IDs harmonized to Ensembl release "
    "112; queried 2026-05-19"
)
STAR_TPM_COL = "tpm_unstranded"
MATRIX_FILES = {
    "TARGET_ALL_P1": {
        "file_id": "c5b8499d-8777-4ad5-bb6b-48cfeca68aed",
        "members": ["TARGET_ALL_SampleMatrix_Phase1_20190606.xlsx"],
    },
    "TARGET_ALL_P2": {
        "file_id": "313fd46b-8111-45dd-88df-4e637a4d2249",
        "members": [
            "TARGET_ALL_SampleMatrix_Phase2_Discovery_20190606.xlsx",
            "TARGET_ALL_SampleMatrix_Phase2_Validation_20190606.xlsx",
        ],
    },
}
PRIMARY_SAMPLE_TYPES = {
    "Primary Blood Derived Cancer - Bone Marrow": 0,
    "Primary Blood Derived Cancer - Peripheral Blood": 1,
}


def _gdc_filters() -> dict:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": PROJECT_IDS,
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
        "cancer_code": "",
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


def _download_url(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(path)
    return path


def _ensure_sample_matrices(cache_dir: Path) -> list[tuple[str, Path]]:
    matrix_dir = cache_dir / "sample_matrices"
    out = []
    for source, record in MATRIX_FILES.items():
        archive = matrix_dir / f"{source}.tar.gz"
        _download_url(f"{GDC_DATA_ENDPOINT}/{record['file_id']}", archive)
        extract_dir = matrix_dir / source
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            present = {member.name for member in tar.getmembers()}
            missing = set(record["members"]) - present
            if missing:
                raise RuntimeError(
                    f"{source} sample matrix archive is missing {sorted(missing)}"
                )
            for member in record["members"]:
                path = extract_dir / member
                if not path.exists():
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Could not read {member} from {source}")
                    with extracted, path.open("wb") as handle:
                        shutil.copyfileobj(extracted, handle)
                out.append((source, path))
    return out


def _parse_lineage_label(comment: object) -> tuple[str, str]:
    text = "" if comment is None else str(comment)
    match = re.search(r"Cell of origin:\s*([^;\n\r]+)", text)
    if not match:
        return "", ""
    raw = match.group(1).strip()
    if raw.lower().startswith("indeterminate"):
        return raw, ""
    if re.match(r"^B(\b|-|\s+precursor|\s+cell)", raw, flags=re.IGNORECASE):
        return raw, "B_ALL"
    if re.match(r"^T(\b|-|\s+cell)", raw, flags=re.IGNORECASE):
        return raw, "T_ALL"
    return raw, ""


def _lineage_table(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for source, path in _ensure_sample_matrices(cache_dir):
        df = pd.read_excel(path, sheet_name="Sample Names")
        df = df[df["Case USI"].notna()].copy()
        matrix_name = path.stem.replace("TARGET_ALL_SampleMatrix_", "")
        evidence = f"{source} {matrix_name} sample matrix"
        for _idx, row in df.iterrows():
            case_id = str(row["Case USI"])
            comment = row.get("Comments", "")
            raw, code = _parse_lineage_label(comment)
            rows.append({
                "case_id": case_id,
                "raw_lineage": raw,
                "lineage_code": code,
                "lineage_evidence_source": evidence,
            })
    lineage = pd.DataFrame(rows)
    resolved = []
    for case_id, group in lineage.groupby("case_id", sort=False):
        codes = sorted({str(code) for code in group["lineage_code"] if code})
        if len(codes) != 1:
            continue
        chosen = group[group["lineage_code"].eq(codes[0])].iloc[-1]
        resolved.append({
            "case_id": case_id,
            "raw_lineage": chosen["raw_lineage"],
            "lineage_code": chosen["lineage_code"],
            "lineage_evidence_source": (
                f"{chosen['lineage_evidence_source']}; "
                f"Cell of origin: {chosen['raw_lineage']}"
            ),
        })
    return pd.DataFrame(resolved)


def _sample_type_rank(sample_type: str) -> int:
    return PRIMARY_SAMPLE_TYPES.get(sample_type, 99)


def _build_sample_manifest(hits: list[dict], cache_dir: Path) -> pd.DataFrame:
    manifest = pd.DataFrame([_flatten_hit(hit) for hit in hits])
    lineage = _lineage_table(cache_dir)
    manifest = manifest.merge(lineage, on="case_id", how="left")
    manifest["lineage_code"] = manifest["lineage_code"].fillna("")
    manifest["raw_lineage"] = manifest["raw_lineage"].fillna("")
    manifest["lineage_evidence_source"] = (
        manifest["lineage_evidence_source"].fillna("")
    )
    manifest["cancer_code"] = manifest["lineage_code"]
    manifest["included"] = False
    manifest["exclusion_reason"] = ""
    manifest["lineage_label"] = manifest["lineage_code"]

    has_lineage = manifest["lineage_code"].isin(["B_ALL", "T_ALL"])
    primary = manifest["sample_type"].isin(PRIMARY_SAMPLE_TYPES)
    diagnosis = manifest["primary_diagnosis"].astype(str).str.strip()
    all_diagnosis = diagnosis.eq("Acute lymphocytic leukemia")

    manifest.loc[~has_lineage, "exclusion_reason"] = "no_b_or_t_lineage"
    manifest.loc[
        has_lineage & ~primary,
        "exclusion_reason",
    ] = "not_primary_diagnostic_blood_or_marrow"
    manifest.loc[
        has_lineage & primary & ~all_diagnosis,
        "exclusion_reason",
    ] = "not_acute_lymphocytic_leukemia"

    eligible = has_lineage & primary & all_diagnosis
    for _case_id, group in manifest[eligible].groupby("case_id", sort=False):
        ordered = group.assign(
            _sample_type_rank=group["sample_type"].map(_sample_type_rank),
        ).sort_values(["_sample_type_rank", "sample_id", "source_file_id"])
        keep_idx = ordered.index[0]
        manifest.loc[keep_idx, "included"] = True
        duplicate_idx = ordered.index[1:]
        manifest.loc[duplicate_idx, "exclusion_reason"] = "duplicate_primary_sample"

    included = manifest["included"]
    manifest.loc[included, "exclusion_reason"] = ""
    manifest = manifest.drop(columns=["raw_lineage", "lineage_code"])
    return manifest.sort_values(["case_id", "sample_id"]).reset_index(drop=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - MD5 is used only for GDC checksums.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(row: pd.Series, cache_dir: Path) -> Path:
    files_dir = cache_dir / "star_counts"
    files_dir.mkdir(parents=True, exist_ok=True)
    path = files_dir / str(row["source_file_name"])
    expected_md5 = str(row["md5sum"])
    if path.exists() and _md5(path) == expected_md5:
        return path
    url = f"{GDC_DATA_ENDPOINT}/{row['source_file_id']}"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    observed_md5 = ""
    for attempt in range(1, 6):
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                tmp_path.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
        except (TimeoutError, URLError) as exc:
            tmp_path.unlink(missing_ok=True)
            print(
                f"Retrying {row['source_file_id']} after download error "
                f"({attempt}/5): {exc}."
            )
            continue
        observed_md5 = _md5(tmp_path)
        if observed_md5 == expected_md5:
            break
        tmp_path.unlink(missing_ok=True)
        print(
            f"Retrying {row['source_file_id']} after MD5 mismatch "
            f"({attempt}/5)."
        )
    if observed_md5 != expected_md5:
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
    → NCBI-synonym/alias rescue), so the GENCODE-v36→Ensembl-112 fallback
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


def _summarize_one(
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
    *,
    cancer_code: str,
    notes: str,
) -> pd.DataFrame:
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cancer_code
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = SOURCE_VERSION
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE
    out["tumor_origin"] = "primary"
    out["metastasis_site"] = pd.NA
    out["notes"] = notes
    return round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))


def _summarize(
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
    included: pd.DataFrame,
    *,
    source_id: str | None = None,
) -> pd.DataFrame:
    frames = []
    notes_by_code = {
        "B_ALL": (
            "Open GDC TARGET ALL STAR-counts TPMs; one deterministic "
            "primary diagnostic BM/PB sample per case, preferring BM; "
            "B-lineage assigned from TARGET Phase 1/2 sample-matrix "
            "Cell of origin comments; GENCODE v36 IDs harmonized to "
            "Ensembl release 112."
        ),
        "T_ALL": (
            "Open GDC TARGET ALL STAR-counts TPMs; one deterministic "
            "primary diagnostic BM/PB sample per case, preferring BM; "
            "T-lineage assigned from TARGET Phase 2 validation "
            "sample-matrix Cell of origin comments; GENCODE v36 IDs "
            "harmonized to Ensembl release 112."
        ),
    }
    for code in ["B_ALL", "T_ALL"]:
        sample_ids = included.loc[included["cancer_code"].eq(code), "sample_id"]
        if sample_ids.empty:
            continue
        code_values = values.loc[:, list(sample_ids)]
        if source_id:
            # Persist the per-code per-sample matrix for medoids + percentiles
            # (same selection that drives the summary — one source of truth).
            from pirlygenes import cohorts as _cohorts
            _cohorts.write_per_sample(gene_table, code_values, source_id, code)
        frames.append(
            _summarize_one(
                gene_table,
                code_values,
                cancer_code=code,
                notes=notes_by_code[code],
            )
        )
    if not frames:
        raise RuntimeError("No TARGET ALL lineage summaries were produced")
    return pd.concat(frames, ignore_index=True)


def _upsert_source_cohort(
    path: Path,
    new_rows: pd.DataFrame,
    *,
    source_cohort: str,
) -> pd.DataFrame:
    """Sharded write for per-gene reference frames.

    Used only for the summary (per-gene-per-cohort) frame which has
    the REFERENCE_COLUMNS schema. The samples manifest write must
    NOT use this — it's a single-file write, see _upsert_samples.
    """
    codes = sorted(new_rows["cancer_code"].astype(str).unique())
    return write_reference_rows(
        path, new_rows, source_cohort=source_cohort, cancer_codes=codes,
    )


def _upsert_samples(path: Path, new_rows: pd.DataFrame, *, source_cohort: str) -> pd.DataFrame:
    """Delegates to the shared, column-union-preserving samples-manifest upsert
    (cohorts to replace are derived from new_rows' own source_cohort values)."""
    from pirlygenes.expression.stats import upsert_samples_manifest
    return upsert_samples_manifest(path, new_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    hits = _query_manifest()
    manifest = _build_sample_manifest(hits, args.cache_dir)
    included = manifest[manifest["included"]].reset_index(drop=True)
    if included.empty:
        raise RuntimeError("No TARGET ALL samples passed inclusion filters")
    gene_table, values, harmonized = _build_values(
        included,
        cache_dir=args.cache_dir,
        ensembl_release=args.ensembl_release,
    )
    summary = _summarize(gene_table, values, included,
                         source_id=args.cache_dir.name)
    _upsert_source_cohort(
        args.summary_output,
        summary,
        source_cohort=SOURCE_COHORT,
    )
    _upsert_samples(
        args.samples_output,
        manifest,
        source_cohort=SOURCE_COHORT,
    )
    excluded = len(manifest) - len(included)
    counts = included["cancer_code"].value_counts().to_dict()
    print(
        f"Wrote {len(summary)} TARGET ALL genes from {len(included)} "
        f"included samples ({len(manifest)} total GDC files; "
        f"{excluded} excluded; B_ALL={counts.get('B_ALL', 0)}, "
        f"T_ALL={counts.get('T_ALL', 0)})."
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
