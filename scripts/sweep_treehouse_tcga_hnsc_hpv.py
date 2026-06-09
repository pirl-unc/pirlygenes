#!/usr/bin/env python
"""Split TCGA-HNSC by HPV status (HNSC_HPV_pos / HNSC_HPV_neg).

Uses cBioPortal study ``hnsc_tcga_pan_can_atlas_2018`` SUBTYPE
calls (HPV+ / HPV-, n=72 / 415 patients) as the authoritative HPV
assignment — derived from Cao 2016 (PMID 27568064)'s genome-wide
HPV-integration analysis. Joins against TCGA-HNSC samples already
in the Treehouse 25.01 PolyA cache by case submitter_id prefix.

Two cohorts under
``source_cohort=TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV`` (separate
shard from the parent HNSC cohort).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from pirlygenes.builders.treehouse import (
    TreehouseCohort,
    TreehouseRelease,
    run_sweep,
    tcga_case_predicate,
)
from pirlygenes.cohorts import cohorts_for_group


CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"
RELEASE_DIR = CACHE_ROOT / "treehouse-polya-25-01"

RELEASE = TreehouseRelease(
    source_id="treehouse-polya-25-01",
    source_cohort="TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV",
    source_project="Treehouse (TCGA-HNSC) × cBioPortal HPV",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA TCGA-HNSC samples "
        "× cBioPortal hnsc_tcga_pan_can_atlas_2018 HPV calls "
        "(Cao 2016 PMID 27568064)"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_tcga_hnsc_hpv_log2tpm_to_tpm",
)


# Cohort definitions (code, stem, selection="hpv:<cBioPortal label>") come from
# the single registry in pirlygenes.cohorts (group "tcga_hnsc_hpv").


def _fetch_cbioportal_hpv(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    url = (
        "https://www.cbioportal.org/api/studies/hnsc_tcga_pan_can_atlas_2018"
        "/clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    df = pd.DataFrame(
        [{"patientId": d["patientId"], "hpv_subtype": d["value"]} for d in data]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    cache = RELEASE_DIR / "derived" / "cbioportal_hnsc_hpv.csv"
    hpv = _fetch_cbioportal_hpv(cache)
    print(f"loaded HPV calls: {hpv['hpv_subtype'].value_counts().to_dict()}")

    cohorts = []
    for c in cohorts_for_group("tcga_hnsc_hpv"):
        hpv_label = c.selection.split(":", 1)[1]
        cases = set(
            hpv.loc[hpv["hpv_subtype"] == hpv_label, "patientId"].astype(str)
        )
        cohorts.append(
            TreehouseCohort(
                cancer_code=c.code,
                disease_label=c.disease_label,
                sample_predicate=tcga_case_predicate(cases),
                extra_notes=(
                    f"HPV subtype = '{hpv_label}' per cBioPortal "
                    "hnsc_tcga_pan_can_atlas_2018 (Cao 2016 "
                    "PMID 27568064). Sample selection: TCGA-HNSC "
                    "samples in Treehouse 25.01 PolyA whose case "
                    "submitter_id is classified as this HPV subtype "
                    "in the cBioPortal study."
                ),
                cache_stem=c.stem,
            )
        )

    run_sweep(
        RELEASE,
        cohorts,
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        refresh_cache=args.refresh_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
