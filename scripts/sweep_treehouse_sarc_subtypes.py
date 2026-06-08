#!/usr/bin/env python
"""Add the two sourceable TCGA-SARC histology splits.

Three registry codes (SARC_GIST / SARC_MYXLPS / SARC_WDLPS) aren't
covered by Treehouse's compendium-wide disease labels. Of those:

- **SARC_GIST**: Treehouse has 19 "gastrointestinal stromal tumor"
  samples (all non-TCGA, from Treehouse's own clinical partners +
  publicly-available repositories). Build directly from those under
  source_cohort=TREEHOUSE_POLYA_25_01.

- **SARC_WDLPS**: TCGA-SARC has 5 "Liposarcoma, well differentiated"
  cases; all 5 are also in Treehouse (under disease="liposarcoma").
  Route them out of the parent SARC_LPS_UNSPEC cohort by joining
  the GDC TCGA-SARC histology table on the case submitter_id.
  Tag with source_cohort=TREEHOUSE_POLYA_25_01_TCGA_SUBSET.

- **SARC_MYXLPS** (myxoid liposarcoma): NOT in TCGA-SARC. Treehouse
  has 1 "pleomorphic myxoid liposarcoma" sample, too rare to build.
  Left empty in the registry; needs an external source
  (e.g. GSE128064 or other GEO accession). Tracked in audit.

GDC API call cached at
``~/.cache/pirlygenes/expression/treehouse-polya-25-01/derived/tcga_sarc_histology.csv``.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
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

# Cohort definitions (code, stem, disease_label, selection) come from the
# single registry in pirlygenes.cohorts (group "sarc_subtypes").
_SUBTYPES = {c.code: c for c in cohorts_for_group("sarc_subtypes")}

# Two distinct source_cohort tags = two distinct shards.
RELEASE_TREEHOUSE = TreehouseRelease(
    source_id="treehouse-polya-25-01",
    source_cohort="TREEHOUSE_POLYA_25_01",
    source_project="Treehouse",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA "
        "(hugo_log2tpm_58581genes_2025-02-27); downloaded from "
        "public.gi.ucsc.edu/~ekephart/public-data/"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_log2tpm_to_tpm",
)

RELEASE_TCGA = TreehouseRelease(
    source_id="treehouse-polya-25-01",
    source_cohort="TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
    source_project="Treehouse (TCGA samples)",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA, TCGA-SARC "
        "histology split via GDC primary_diagnosis lookup; "
        "downloaded from public.gi.ucsc.edu/~ekephart/public-data/"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_tcga_sarc_histology_log2tpm_to_tpm",
    # Per-code to match the TCGA_SUBSET source_cohort layout (else collides
    # with the per-code __SARC_WDLPS shard).
    per_cancer_code_shards=True,
)


def _fetch_sarc_histology(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    filters = {"op": "in", "content": {"field": "project.project_id",
                                       "value": ["TCGA-SARC"]}}
    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,diagnoses.primary_diagnosis",
        "format": "JSON",
        "size": "500",
    }
    url = "https://api.gdc.cancer.gov/cases?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as r:
        hits = json.load(r)["data"]["hits"]
    rows = []
    for h in hits:
        diag = (h.get("diagnoses") or [{}])[0].get("primary_diagnosis", "")
        rows.append({"submitter_id": h["submitter_id"], "primary_diagnosis": diag})
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _build_histology_predicate(histology: pd.DataFrame, primary_diagnosis: str):
    cases = histology.loc[
        histology["primary_diagnosis"].eq(primary_diagnosis),
        "submitter_id",
    ].astype(str)
    return tcga_case_predicate(cases)


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

    # --- GIST: non-TCGA Treehouse samples under disease label ---
    gist = _SUBTYPES["SARC_GIST"]
    print("=== SARC_GIST: Treehouse direct (n=19) ===")
    run_sweep(
        RELEASE_TREEHOUSE,
        [
            TreehouseCohort(
                cancer_code=gist.code,
                disease_label=gist.disease_label,
                extra_notes=(
                    "All 19 GIST samples in Treehouse 25.01 PolyA. "
                    "Not from TCGA-SARC (TCGA-SARC excluded GIST); "
                    "Treehouse-internal + publicly-available repositories."
                ),
                cache_stem=gist.stem,
            ),
        ],
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        refresh_cache=args.refresh_cache,
    )

    # --- WDLPS: TCGA-SARC histology overlay ---
    wdlps = _SUBTYPES["SARC_WDLPS"]
    primary_diagnosis = wdlps.selection.split(":", 1)[1]
    print()
    print("=== SARC_WDLPS: TCGA-SARC histology overlay (n=5) ===")
    histology_cache = RELEASE_DIR / "derived" / "tcga_sarc_histology.csv"
    histology = _fetch_sarc_histology(histology_cache)
    pred = _build_histology_predicate(histology, primary_diagnosis)
    run_sweep(
        RELEASE_TCGA,
        [
            TreehouseCohort(
                cancer_code=wdlps.code,
                disease_label=wdlps.disease_label,
                sample_predicate=pred,
                extra_notes=(
                    f"TCGA-SARC histology = '{primary_diagnosis}' (n=5). "
                    "Routed out of the TCGA-SARC project via GDC "
                    "primary_diagnosis lookup on the case submitter_id."
                ),
                cache_stem=wdlps.stem,
            ),
        ],
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        refresh_cache=args.refresh_cache,
    )

    print()
    print("SARC_MYXLPS: deferred. TCGA-SARC has no myxoid liposarcoma; "
          "needs external GEO source (see docs/archive/expression-data-audit-2026-05.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
