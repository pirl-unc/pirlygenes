#!/usr/bin/env python
"""Per-sample GDC STAR-counts builders for TARGET-NBL / TARGET-RT / TARGET-WT.

Replaces the summary-only imports for NBL_MYCN_amp / nonamp, RT,
and WILMS with per-sample TPM rollups. Pattern mirrors
``scripts/build_mmrf_reference_expression.py`` — open GDC STAR-counts
download, deterministic one-sample-per-case filter, harmonize to
Ensembl 112, two-compartment fixed-fraction clean-TPM (technical 25% / biological 75%, each renormalized within its group), compute
the v5.3 stat suite, upsert to the shard directory.

NBL split: MYCN amplification status from cBioPortal
nbl_target_2018_pub (n=303 amplified / 770 non-amplified / 16
unknown). 16 unknowns go to NBL_MYCN_nonamp by default (the
larger / less-aggressive cohort) but are flagged in notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.builders.gene_mapping import resolve_symbol
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
from pirlygenes.expression.normalize import clean_tpm_matrix as _clean_tpm, technical_rna_mask as _technical_mask


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
STAR_TPM_COL = "tpm_unstranded"


@dataclass(frozen=True)
class TargetProject:
    project_id: str          # e.g. "TARGET-NBL"
    source_cohort: str       # cancer-reference-expression source_cohort tag
    source_project: str      # human-readable
    pipeline_id: str
    cancer_code: str         # default cancer_code; subtype splits override per sample
    primary_diagnosis_keywords: tuple[str, ...]  # for inclusion filter
    cache_subdir: str        # cache subdir name


PROJECTS = [
    TargetProject(
        project_id="TARGET-NBL",
        source_cohort="TARGET_NBL_2018",
        source_project="TARGET Neuroblastoma",
        pipeline_id="gdc_star_counts_tpm_ensembl112_clean_tpm_v4",
        cancer_code="NBL",  # subtype-split into NBL_MYCN_amp / NBL_MYCN_nonamp
        primary_diagnosis_keywords=("neuroblastoma", "ganglioneuroblastoma"),
        cache_subdir="target-nbl",
    ),
    TargetProject(
        project_id="TARGET-RT",
        source_cohort="TARGET_RT_2017",
        source_project="TARGET Rhabdoid Tumor",
        pipeline_id="gdc_star_counts_tpm_ensembl112_clean_tpm_v4",
        cancer_code="RT",
        primary_diagnosis_keywords=("rhabdoid",),
        cache_subdir="target-rt",
    ),
    TargetProject(
        project_id="TARGET-WT",
        source_cohort="TARGET_WT_2015",
        source_project="TARGET Wilms Tumor",
        pipeline_id="gdc_star_counts_tpm_ensembl112_clean_tpm_v4",
        cancer_code="WILMS",
        primary_diagnosis_keywords=("nephroblastoma", "wilms"),
        cache_subdir="target-wt",
    ),
]


def _strip_version(v: object) -> str:
    return str(v).split(".", 1)[0]


def _gdc_filters(project_id: str) -> dict:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": [project_id]}},
            {"op": "in", "content": {"field": "data_type",
                                     "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "experimental_strategy",
                                     "value": ["RNA-Seq"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type",
                                     "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }


def _query_manifest(project: TargetProject) -> list[dict]:
    fields = [
        "file_id", "file_name", "md5sum", "file_size",
        "cases.submitter_id", "cases.project.project_id",
        "cases.samples.submitter_id", "cases.samples.sample_type",
        "cases.diagnoses.primary_diagnosis",
        "analysis.workflow_type",
    ]
    params = {
        "filters": json.dumps(_gdc_filters(project.project_id)),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "5000",
    }
    req = urllib.request.Request(
        GDC_FILES_ENDPOINT,
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        payload = json.load(r)
    hits = payload["data"]["hits"]
    total = int(payload["data"]["pagination"]["total"])
    if len(hits) != total:
        raise RuntimeError(
            f"GDC for {project.project_id} returned {len(hits)} of {total}"
        )
    return hits


def _flatten_hit(hit: dict, project: TargetProject) -> dict:
    case = hit.get("cases", [{}])[0]
    sample = case.get("samples", [{}])[0]
    diagnosis = (case.get("diagnoses") or [{}])[0]
    return {
        "cancer_code": project.cancer_code,
        "source_cohort": project.source_cohort,
        "source_project": project.source_project,
        "case_id": case.get("submitter_id", ""),
        "sample_id": sample.get("submitter_id", ""),
        "source_file_id": hit.get("file_id", ""),
        "source_file_name": hit.get("file_name", ""),
        "source_project_id": case.get("project", {}).get("project_id", ""),
        "sample_type": sample.get("sample_type", ""),
        "primary_diagnosis": diagnosis.get("primary_diagnosis", ""),
        "md5sum": hit.get("md5sum", ""),
        "file_size": hit.get("file_size", ""),
        "workflow_type": hit.get("analysis", {}).get("workflow_type", ""),
        "raw_unit": "TPM",
        "processing_pipeline": project.pipeline_id,
        "source_url": f"https://portal.gdc.cancer.gov/projects/{project.project_id}",
    }


# Deterministic sample-per-case rank: prefer Primary Tumor, then Recurrent,
# then Metastatic, by lexicographic sample_id within that.
_SAMPLE_TYPE_RANK = {
    "Primary Tumor": 0,
    "Primary Blood Derived Cancer - Bone Marrow": 0,
    "Recurrent Tumor": 1,
    "Metastatic": 2,
    "Additional - New Primary": 3,
}


def _sample_type_rank(s: str) -> int:
    return _SAMPLE_TYPE_RANK.get(s, 9)


def _build_sample_manifest(hits: list[dict], project: TargetProject) -> pd.DataFrame:
    manifest = pd.DataFrame([_flatten_hit(h, project) for h in hits])
    manifest["included"] = False
    manifest["exclusion_reason"] = ""
    manifest["lineage_label"] = ""
    manifest["lineage_evidence_source"] = (
        f"GDC {project.project_id} primary_diagnosis match"
    )

    # Inclusion: primary diagnosis matches a project keyword
    diag = manifest["primary_diagnosis"].astype(str).str.lower()
    keyword_match = pd.Series(False, index=manifest.index)
    for kw in project.primary_diagnosis_keywords:
        keyword_match |= diag.str.contains(kw, na=False)
    manifest.loc[~keyword_match, "exclusion_reason"] = "primary_diagnosis_mismatch"

    # One deterministic sample per case among eligible rows
    eligible_idx = manifest.index[keyword_match]
    eligible = manifest.loc[eligible_idx].copy()
    eligible["_rank"] = eligible["sample_type"].map(_sample_type_rank)
    for _case_id, group in eligible.groupby("case_id", sort=False):
        ordered = group.sort_values(["_rank", "sample_id", "source_file_id"])
        keep = ordered.index[0]
        manifest.loc[keep, "included"] = True
        dupes = ordered.index[1:]
        manifest.loc[dupes, "exclusion_reason"] = "duplicate_for_case"

    included = manifest["included"]
    manifest.loc[included, "exclusion_reason"] = ""
    manifest.loc[included, "lineage_label"] = project.cancer_code

    return manifest.sort_values(["case_id", "sample_id"]).reset_index(drop=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - GDC file checksum, not crypto.
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_file(row: pd.Series, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / str(row["source_file_name"])
    expected = str(row["md5sum"])
    if path.exists() and _md5(path) == expected:
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    url = f"{GDC_DATA_ENDPOINT}/{row['source_file_id']}"
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    observed = _md5(tmp)
    if observed != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {row['source_file_id']}: {observed} != {expected}"
        )
    tmp.replace(path)
    return path


def _read_star_tpm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", comment="#",
        usecols=["gene_id", "gene_name", "gene_type", STAR_TPM_COL],
        low_memory=False,
    )
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    df["source_gene_id"] = df["gene_id"].map(_strip_version)
    df["Symbol"] = df["gene_name"].fillna("").astype(str)
    df["gene_type"] = df["gene_type"].fillna("").astype(str)
    df["TPM"] = pd.to_numeric(df[STAR_TPM_COL], errors="coerce").fillna(0.0)
    return df[["source_gene_id", "Symbol", "gene_type", "TPM"]]


def _resolved_gene(genome, resolved):
    """The Ensembl gene object for a (ENSG, symbol, method) resolve_symbol
    result, or None."""
    if resolved is None:
        return None
    try:
        return genome.gene_by_id(resolved[0])
    except Exception:
        return None


def _harmonize_gene_table(template: pd.DataFrame, ensembl_release: int):
    genome = EnsemblRelease(ensembl_release)
    rows: list[dict[str, str]] = []
    counts = {"source_id": 0, "symbol": 0, "dropped": 0}
    for r in template[["source_gene_id", "Symbol", "gene_type"]].itertuples(index=False):
        sid = str(r.source_gene_id)
        sym = str(r.Symbol)
        try:
            gene = genome.gene_by_id(sid)
        except Exception:
            gene = None
        if gene is not None:
            rows.append({"source_gene_id": sid,
                         "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
                         "Symbol": gene.gene_name or sym,
                         "gene_type": r.gene_type})
            counts["source_id"] += 1
            continue
        # shared resolver: direct → Entrez → NCBI-synonym/alias rescue
        resolved = resolve_symbol(genome, sym)
        g = _resolved_gene(genome, resolved)
        if g is not None:
            rows.append({"source_gene_id": sid,
                         "Ensembl_Gene_ID": g.gene_id.split(".", 1)[0],
                         "Symbol": g.gene_name or sym,
                         "gene_type": r.gene_type})
            counts["symbol"] += 1
            continue
        counts["dropped"] += 1
    mapping = pd.DataFrame(rows)
    canonical = (
        mapping.drop_duplicates("Ensembl_Gene_ID")[["Ensembl_Gene_ID", "Symbol", "gene_type"]]
        .reset_index(drop=True)
    )
    counts["canonical_genes"] = len(canonical)
    return mapping, counts, canonical


def _sample_vector(path: Path, mapping: pd.DataFrame, gene_ids: pd.Index) -> pd.Series:
    tpm = _read_star_tpm(path)
    merged = tpm.merge(
        mapping[["source_gene_id", "Ensembl_Gene_ID"]],
        on="source_gene_id", how="inner",
    )
    values = merged.groupby("Ensembl_Gene_ID", sort=False)["TPM"].sum()
    return values.reindex(gene_ids, fill_value=0.0)


def _build_values(included: pd.DataFrame, cache_dir: Path, ensembl_release: int):
    first = _download_file(included.iloc[0], cache_dir)
    template = _read_star_tpm(first)
    mapping, harm, canonical = _harmonize_gene_table(template, ensembl_release)
    gene_ids = pd.Index(canonical["Ensembl_Gene_ID"])
    arrays, sample_ids = [], []
    for idx, row in included.reset_index(drop=True).iterrows():
        p = _download_file(row, cache_dir)
        arrays.append(_sample_vector(p, mapping, gene_ids).to_numpy(dtype=float))
        sample_ids.append(str(row["sample_id"]))
        if (idx + 1) % 25 == 0 or idx + 1 == len(included):
            print(f"  processed {idx + 1}/{len(included)} files")
    values = pd.DataFrame(
        np.column_stack(arrays), index=gene_ids, columns=sample_ids,
    )
    return canonical, values, harm


def _summarize_one(
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
    *,
    cancer_code: str,
    project: TargetProject,
    extra_notes: str = "",
) -> pd.DataFrame:
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cancer_code
    out["source_cohort"] = project.source_cohort
    out["source_project"] = project.source_project
    out["source_version"] = (
        f"GDC STAR-Counts, GENCODE v36; Ensembl IDs harmonized to release 112"
    )
    assign_stats(out, values, clean)
    out["processing_pipeline"] = project.pipeline_id
    out["tumor_origin"] = "primary"
    out["metastasis_site"] = pd.NA
    out["notes"] = (
        f"Open GDC {project.project_id} STAR-counts TPMs; one "
        f"deterministic primary-tumor sample per case; GENCODE v36 IDs "
        f"harmonized to Ensembl release 112 (shared resolver: direct → "
        f"Entrez → NCBI-synonym/alias rescue)."
    )
    if extra_notes:
        out["notes"] = out["notes"] + " " + extra_notes
    return round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))


def _upsert_samples_manifest(path: Path, manifest: pd.DataFrame, source_cohort: str):
    # Delegates to the shared, column-union-preserving upsert (cohorts to
    # replace are derived from the manifest's own source_cohort values).
    from pirlygenes.expression.stats import upsert_samples_manifest
    return upsert_samples_manifest(path, manifest)


def _fetch_nbl_mycn(cache_path: Path) -> dict[str, str]:
    """Return TARGET-NBL case_submitter_id → 'amp' | 'nonamp' | 'unknown'.

    MYCN is a sample-level attribute in cBioPortal study
    ``nbl_target_2018_pub`` (patientAttribute: False). Sample IDs
    look like ``TARGET-30-PAUDVA-09``; the case submitter_id is the
    first three fields. Each patient has at most one MYCN call so
    deduping is safe.
    """
    if cache_path.exists() and cache_path.stat().st_size > 100:
        df = pd.read_csv(cache_path)
    else:
        url = (
            "https://www.cbioportal.org/api/studies/nbl_target_2018_pub"
            "/clinical-data?clinicalDataType=SAMPLE&attributeId=MYCN"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        rows = [
            {
                "patientId": d.get("patientId")
                or "-".join(str(d["sampleId"]).split("-")[:3]),
                "mycn": d["value"],
            }
            for d in data
        ]
        df = pd.DataFrame(rows).drop_duplicates("patientId")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    def _norm(v: str) -> str:
        s = str(v).strip().lower()
        if s == "amplified":
            return "amp"
        if s == "not amplified":
            return "nonamp"
        return "unknown"

    return dict(zip(df["patientId"].astype(str), df["mycn"].map(_norm)))


def _build_project(project: TargetProject, args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_root) / project.cache_subdir
    print(f"\n===== {project.project_id} =====")
    hits = _query_manifest(project)
    manifest = _build_sample_manifest(hits, project)
    included = manifest[manifest["included"]].reset_index(drop=True)
    if included.empty:
        raise RuntimeError(f"no eligible samples for {project.project_id}")
    print(f"  {len(included)} included / {len(manifest)} total files")
    gene_table, values, harm = _build_values(
        included, cache_dir=cache_dir, ensembl_release=args.ensembl_release,
    )

    # NBL: split by MYCN status; else single cohort
    if project.project_id == "TARGET-NBL":
        mycn_cache = cache_dir / "cbioportal_nbl_mycn.csv"
        mycn = _fetch_nbl_mycn(mycn_cache)
        # Map sample_id -> case_id -> mycn
        # TARGET sample IDs look like TARGET-30-PAXXXX-09; case_id is first 3 fields
        sample_case = {sid: "-".join(sid.split("-")[:3]) for sid in values.columns}
        sample_mycn = {sid: mycn.get(sample_case[sid], "unknown") for sid in values.columns}
        # Unknowns default to nonamp (the larger / less-aggressive cohort)
        amp_cols = [s for s, m in sample_mycn.items() if m == "amp"]
        nonamp_cols = [s for s, m in sample_mycn.items() if m != "amp"]
        unk_n = sum(1 for s in values.columns if sample_mycn[s] == "unknown")
        print(f"  MYCN split: amp={len(amp_cols)}, nonamp={len(nonamp_cols)} "
              f"(includes {unk_n} unknown defaulted to nonamp)")
        summaries = []
        # Umbrella NBL cohort: all samples, no MYCN split
        summaries.append(_summarize_one(
            gene_table, values,
            cancer_code="NBL", project=project,
            extra_notes=(
                "Umbrella aggregate over all TARGET-NBL samples; "
                f"children NBL_MYCN_amp (n={len(amp_cols)}) and "
                f"NBL_MYCN_nonamp (n={len(nonamp_cols)}) carry the "
                "subtype-specific rows."
            ),
        ))
        if amp_cols:
            summaries.append(_summarize_one(
                gene_table, values[amp_cols],
                cancer_code="NBL_MYCN_amp", project=project,
                extra_notes=("MYCN status = Amplified per cBioPortal "
                             "nbl_target_2018_pub MYCN attribute."),
            ))
        if nonamp_cols:
            summaries.append(_summarize_one(
                gene_table, values[nonamp_cols],
                cancer_code="NBL_MYCN_nonamp", project=project,
                extra_notes=(
                    f"MYCN status = Not Amplified per cBioPortal "
                    f"nbl_target_2018_pub MYCN attribute "
                    f"({unk_n} samples with unknown MYCN status defaulted "
                    "to nonamp bucket)."
                ),
            ))
        combined = pd.concat(summaries, ignore_index=True)
        upsert_to_shard(
            args.summary_output,
            combined,
            source_cohort=project.source_cohort,
            cancer_codes=["NBL", "NBL_MYCN_amp", "NBL_MYCN_nonamp"],
        )
    else:
        summary = _summarize_one(
            gene_table, values,
            cancer_code=project.cancer_code, project=project,
        )
        upsert_to_shard(
            args.summary_output,
            summary,
            source_cohort=project.source_cohort,
            cancer_codes=[project.cancer_code],
        )

    _upsert_samples_manifest(
        Path(args.samples_output), manifest, project.source_cohort,
    )
    print(
        f"  done: {project.cancer_code} from {len(included)} samples; "
        f"gene harmonization: source_id={harm['source_id']}, "
        f"symbol={harm['symbol']}, dropped={harm['dropped']}, "
        f"canonical={harm['canonical_genes']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument(
        "--samples-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression-samples.csv.gz"),
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression",
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated project_ids (TARGET-NBL,TARGET-RT,TARGET-WT)",
    )
    args = parser.parse_args()

    projects = PROJECTS
    if args.only:
        wanted = {p.strip() for p in args.only.split(",") if p.strip()}
        projects = [p for p in PROJECTS if p.project_id in wanted]
        if not projects:
            raise SystemExit(f"--only={args.only!r} matched no projects")
    for project in projects:
        _build_project(project, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
