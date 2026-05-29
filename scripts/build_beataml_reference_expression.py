#!/usr/bin/env python
"""Per-sample BeatAML 1.0 builder.

Replaces the summary-only imports for LAML_APL / LAML_ELN_Adv /
LAML_ELN_Fav / LAML_ELN_Int with per-sample TPM rollups.

Source: open GDC STAR-counts for project BEATAML1.0-COHORT (882
cases / 735 RNA-Seq files).

ELN2017 risk approximation from GDC primary_diagnosis:
- APL: "Acute promyelocytic leukaemia, PML-RAR-alpha"
- Fav (favorable):
    Acute myeloid leukemia with mutated CEBPA
    Acute myeloid leukemia, CBF-beta/MYH11
    Acute myeloid leukemia with t(8;21)(q22;q22); RUNX1-RUNX1T1
    Acute myeloid leukemia with mutated NPM1 (defaults Fav — ELN2017
        is Fav for NPM1+/FLT3-ITD- or NPM1+/FLT3-ITDlow, Adv for
        NPM1-/FLT3-ITDhigh; we don't have per-sample FLT3-ITD VAF
        from GDC clinical so default to Fav)
- Adv (adverse):
    Acute myeloid leukemia with myelodysplasia-related changes
    Therapy related myeloid neoplasm
    Acute myeloid leukemia with inv(3) or t(3;3); RPN1-EVI1
    Acute myeloid leukemia with t(6;9); DEK-NUP214
    Acute erythroid leukaemia (rare, generally adverse)
- Int (intermediate, default for remaining AML diagnoses):
    Acute myeloid leukemia, NOS
    Acute myelomonocytic leukemia
    Acute monoblastic and monocytic leukemia
    Acute myeloid leukemia, minimal differentiation
    Acute myeloid leukemia without/with maturation
    Acute myeloid leukemia with t(9;11); MLLT3-MLL  (KMT2A-rearr;
        ELN2017 sometimes Adv, but default Int here)
    Acute myeloid leukemia with maturation
    Acute megakaryoblastic leukaemia

Excluded (non-AML): mixed-phenotype leukemia, refractory cytopenia,
myelodysplastic syndrome subtypes, primary myelofibrosis, etc. Their
samples land in the manifest with `excluded=non_aml_diagnosis`.

This is an approximation of clinical ELN2017 — for rigorous use, the
BeatAML2 Vizome portal supplies investigator-curated ELN calls
(would need a separate fetcher).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reuse all the heavy infrastructure from build_target_subprojects.
import importlib.util
import sys

_HERE = Path(__file__).resolve().parent
_TARGET = _HERE / "build_target_subprojects.py"
_spec = importlib.util.spec_from_file_location("build_target_subprojects", _TARGET)
_target_module = importlib.util.module_from_spec(_spec)
sys.modules["build_target_subprojects"] = _target_module
_spec.loader.exec_module(_target_module)

# Pull what we need from the shared module
_query_manifest = _target_module._query_manifest
_flatten_hit = _target_module._flatten_hit
_md5 = _target_module._md5
_download_file = _target_module._download_file
_read_star_tpm = _target_module._read_star_tpm
_harmonize_gene_table = _target_module._harmonize_gene_table
_technical_mask = _target_module._technical_mask
_clean_tpm = _target_module._clean_tpm
_sample_vector = _target_module._sample_vector
_summarize_one = _target_module._summarize_one
_upsert_samples_manifest = _target_module._upsert_samples_manifest
TargetProject = _target_module.TargetProject

import numpy as np
import pandas as pd
import urllib.request
import urllib.parse
import json

from pirlygenes.expression.stats import upsert_to_shard


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

SOURCE_COHORT = "BEATAML_OHSU_2022"
SOURCE_PROJECT = "BeatAML 1.0"
PIPELINE = "gdc_beataml_star_counts_tpm_ensembl112_clean_tpm_v1"

# A TargetProject-shaped record reused for _summarize_one's signature.
BEATAML_PROJECT = TargetProject(
    project_id="BEATAML1.0-COHORT",
    source_cohort=SOURCE_COHORT,
    source_project=SOURCE_PROJECT,
    pipeline_id=PIPELINE,
    cancer_code="LAML",  # placeholder; subtype-split per sample
    primary_diagnosis_keywords=("leukemia", "leukaemia", "myeloid", "promyelocytic"),
    cache_subdir="beataml-ohsu",
)


APL_DIAGS = {"Acute promyelocytic leukaemia, PML-RAR-alpha"}

FAV_DIAGS = {
    "Acute myeloid leukemia with mutated CEBPA",
    "Acute myeloid leukemia, CBF-beta/MYH11",
    "Acute myeloid leukemia with t(8;21)(q22;q22); RUNX1-RUNX1T1",
    "Acute myeloid leukemia with mutated NPM1",
}

ADV_DIAGS = {
    "Acute myeloid leukemia with myelodysplasia-related changes",
    "Therapy related myeloid neoplasm",
    "Acute myeloid leukemia with inv(3)(q21q26.2) or t(3;3)(q21;q26.2); RPN1-EVI1",
    "Acute myeloid leukemia with t(6;9)(p23;q34); DEK-NUP214",
    "Acute erythroid leukaemia",
}

# Everything else AML-like → Int. Anything not in any bucket is excluded.
INT_DIAGS = {
    "Acute myeloid leukemia, NOS",
    "Acute myelomonocytic leukemia",
    "Acute monoblastic and monocytic leukemia",
    "Acute myeloid leukemia, minimal differentiation",
    "Acute myeloid leukemia without maturation",
    "Acute myeloid leukemia with maturation",
    "Acute myeloid leukemia with t(9;11)(p22;q23); MLLT3-MLL",
    "Acute megakaryoblastic leukaemia",
    "Myeloid sarcoma",
}


def _classify(diag: str) -> str | None:
    """Map GDC primary_diagnosis → LAML_APL / LAML_ELN_Fav/Int/Adv or None."""
    if diag in APL_DIAGS:
        return "LAML_APL"
    if diag in FAV_DIAGS:
        return "LAML_ELN_Fav"
    if diag in ADV_DIAGS:
        return "LAML_ELN_Adv"
    if diag in INT_DIAGS:
        return "LAML_ELN_Int"
    return None


def _gdc_filters() -> dict:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.program.name",
                                     "value": ["BEATAML1.0"]}},
            {"op": "in", "content": {"field": "data_type",
                                     "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "experimental_strategy",
                                     "value": ["RNA-Seq"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type",
                                     "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }


def _query_beataml_manifest() -> list[dict]:
    fields = [
        "file_id", "file_name", "md5sum", "file_size",
        "cases.submitter_id", "cases.project.project_id",
        "cases.samples.submitter_id", "cases.samples.sample_type",
        "cases.diagnoses.primary_diagnosis",
        "analysis.workflow_type",
    ]
    params = {
        "filters": json.dumps(_gdc_filters()),
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
    return payload["data"]["hits"]


def _build_manifest(hits: list[dict]) -> pd.DataFrame:
    rows = [_flatten_hit(h, BEATAML_PROJECT) for h in hits]
    manifest = pd.DataFrame(rows)
    manifest["included"] = False
    manifest["exclusion_reason"] = ""
    manifest["lineage_label"] = ""
    manifest["lineage_evidence_source"] = (
        "GDC BEATAML1.0-COHORT primary_diagnosis → ELN2017-approx mapping"
    )

    manifest["subtype"] = manifest["primary_diagnosis"].astype(str).map(_classify)
    eligible = manifest["subtype"].notna()
    manifest.loc[~eligible, "exclusion_reason"] = "non_aml_diagnosis_or_unknown"

    # One sample per case (Primary Blood/BM types preferred)
    elig_idx = manifest.index[eligible]
    for _case, group in manifest.loc[elig_idx].groupby("case_id", sort=False):
        ordered = group.sort_values(["sample_id", "source_file_id"])
        keep = ordered.index[0]
        manifest.loc[keep, "included"] = True
        dupes = ordered.index[1:]
        manifest.loc[dupes, "exclusion_reason"] = "duplicate_for_case"

    included = manifest["included"]
    manifest.loc[included, "exclusion_reason"] = ""
    manifest.loc[included, "lineage_label"] = manifest.loc[included, "subtype"]
    return manifest.sort_values(["case_id", "sample_id"]).reset_index(drop=True)


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
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression" / "beataml-ohsu",
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    print("BeatAML: querying GDC manifest...")
    hits = _query_beataml_manifest()
    manifest = _build_manifest(hits)
    included = manifest[manifest["included"]].reset_index(drop=True)
    print(f"  manifest: {len(manifest)} files; {len(included)} included")
    subtype_counts = included["subtype"].value_counts().to_dict()
    print(f"  subtype distribution: {subtype_counts}")

    # Download + read TPMs (mirrors TARGET pattern)
    first = _download_file(included.iloc[0], args.cache_dir)
    template = _read_star_tpm(first)
    mapping, harm, canonical = _harmonize_gene_table(template, args.ensembl_release)
    gene_ids = pd.Index(canonical["Ensembl_Gene_ID"])
    arrays, sample_ids, sample_subtypes = [], [], []
    for idx, row in included.iterrows():
        p = _download_file(row, args.cache_dir)
        arrays.append(_sample_vector(p, mapping, gene_ids).to_numpy(dtype=float))
        sample_ids.append(str(row["sample_id"]))
        sample_subtypes.append(str(row["subtype"]))
        if (idx + 1) % 25 == 0 or idx + 1 == len(included):
            print(f"  processed {idx + 1}/{len(included)}")
    values = pd.DataFrame(
        np.column_stack(arrays), index=gene_ids, columns=sample_ids,
    )
    sub_series = pd.Series(sample_subtypes, index=sample_ids)

    # Per-subtype summarize + upsert
    summaries = []
    cancer_codes = []
    for subtype in ("LAML_APL", "LAML_ELN_Fav", "LAML_ELN_Int", "LAML_ELN_Adv"):
        cols = sub_series[sub_series == subtype].index.tolist()
        if not cols:
            print(f"  skipping {subtype}: no samples")
            continue
        sub_values = values[cols]
        summary = _summarize_one(
            canonical, sub_values,
            cancer_code=subtype, project=BEATAML_PROJECT,
            extra_notes=(
                f"ELN2017-approx subtype = '{subtype}' assigned by GDC "
                "primary_diagnosis → ELN bucket. See script docstring."
            ),
        )
        summaries.append(summary)
        cancer_codes.append(subtype)
        print(f"  {subtype}: {len(cols)} samples; {len(summary)} gene rows")

    combined = pd.concat(summaries, ignore_index=True)
    upsert_to_shard(
        args.summary_output, combined,
        source_cohort=SOURCE_COHORT, cancer_codes=cancer_codes,
    )
    _upsert_samples_manifest(Path(args.samples_output), manifest, SOURCE_COHORT)
    print(
        f"Wrote {len(combined)} BeatAML reference rows across "
        f"{len(cancer_codes)} subtypes. Gene harmonization: "
        f"{harm['source_id']} source_id retained, {harm['symbol']} by "
        f"symbol, {harm['dropped']} dropped, {harm['canonical_genes']} "
        "canonical genes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
