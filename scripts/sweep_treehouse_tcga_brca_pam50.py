#!/usr/bin/env python
"""Split TCGA-BRCA into PAM50 molecular subtypes.

Uses the PAM50 calls from cBioPortal study
``brca_tcga_pan_can_atlas_2018`` (Hoadley 2018, PMID 29625050) as the
authoritative subtype assignments — same calls used by every other
TCGA pan-cancer downstream paper. Matches them against TCGA-BRCA
samples already in the Treehouse 25.01 PolyA cache via the case
submitter_id prefix (first three fields of the aliquot barcode).

cBioPortal API call cached at
``~/.cache/pirlygenes/expression/treehouse-polya-25-01/derived/cbioportal_brca_pam50.csv``.

Five subtype cohorts under
``source_cohort=TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50`` (distinct
shard from the parent BRCA cohort so the per-source file stays
manageable).

PAM50 → registry code mapping:
  BRCA_Basal  → BRCA_Basal
  BRCA_Her2   → BRCA_HER2
  BRCA_LumA   → BRCA_LumA
  BRCA_LumB   → BRCA_LumB
  BRCA_Normal → BRCA_Normal

Per the Parker 2009 (PMID 19204204) classifier, expect ~50% LumA,
~20% LumB, ~17% Basal, ~10% Her2, ~3% Normal-like — matches the
Hoadley 2018 distribution.
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
    source_cohort="TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
    source_project="Treehouse (TCGA-BRCA) × cBioPortal PAM50",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA TCGA-BRCA samples "
        "× cBioPortal brca_tcga_pan_can_atlas_2018 PAM50 calls "
        "(Hoadley 2018 PMID 29625050)"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_tcga_brca_pam50_log2tpm_to_tpm",
)


# Cohort definitions (code, stem, selection="pam50:<cBioPortal label>") come
# from the single registry in pirlygenes.cohorts (group "tcga_brca_pam50").


def _fetch_cbioportal_pam50(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    url = (
        "https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018"
        "/clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    df = pd.DataFrame(
        [{"patientId": d["patientId"], "pam50": d["value"]} for d in data]
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

    cache = RELEASE_DIR / "derived" / "cbioportal_brca_pam50.csv"
    pam50 = _fetch_cbioportal_pam50(cache)
    counts = pam50["pam50"].value_counts().to_dict()
    print(f"loaded cBioPortal PAM50 calls: {counts}")

    cohorts = []
    for c in cohorts_for_group("tcga_brca_pam50"):
        pam50_label = c.selection.split(":", 1)[1]
        cases = set(
            pam50.loc[pam50["pam50"] == pam50_label, "patientId"].astype(str)
        )
        if not cases:
            print(f"  skipping {c.code}: no cases")
            continue
        cohorts.append(
            TreehouseCohort(
                cancer_code=c.code,
                disease_label=c.disease_label,
                sample_predicate=tcga_case_predicate(cases),
                extra_notes=(
                    f"PAM50 subtype = '{pam50_label}' per cBioPortal "
                    "brca_tcga_pan_can_atlas_2018 (Hoadley 2018 "
                    "PMID 29625050). Sample selection: TCGA-BRCA "
                    "samples in Treehouse 25.01 PolyA whose case "
                    "submitter_id is classified as this PAM50 subtype "
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
