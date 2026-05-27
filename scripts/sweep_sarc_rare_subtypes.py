#!/usr/bin/env python
"""Sarcoma rare-subtype sweep — fill in registry codes from cached data.

10 registry codes covered with zero new downloads. Two paths:

(a) **Treehouse direct** — registered as own disease labels in
    Treehouse 25.01 PolyA clinical:

      angiosarcoma                              n=20  → SARC_ANGIO
      alveolar soft part sarcoma                n=3   → SARC_ASPS (low n)
      desmoplastic small round cell tumor       n=9   → SARC_DSRCT
      epithelioid sarcoma                       n=5   → SARC_EPITH
      inflammatory myofibroblastic tumor        n=4   → SARC_IMT
      infantile fibrosarcoma                    n=2   → SARC_IFS (low n)
      malignant peripheral nerve sheath tumor   n=13  → SARC_MPNST
      epithelioid hemangioendothelioma          n=1   → SARC_EHE (low n)

    Tag with source_cohort=TREEHOUSE_POLYA_25_01.

(b) **TCGA-SARC histology overlay** — TCGA-SARC has these
    histologies that aren't represented in Treehouse's labels:

      Dedifferentiated liposarcoma              n=50  → SARC_DDLPS
                                                       (upgrades the
                                                        GSE75885 n=19
                                                        summary import)
      Pleomorphic liposarcoma                   n=2   → SARC_PLEOLPS
                                                       (upgrades the
                                                        GSE75885 n=4
                                                        summary import)

    Tag with source_cohort=TREEHOUSE_POLYA_25_01_TCGA_SUBSET.

The histology mapping is cached at
``~/.cache/pirlygenes/expression/treehouse-polya-25-01/derived/
tcga_sarc_histology.csv`` from the earlier SARC sub-histology sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.builders.treehouse import (
    TreehouseCohort,
    TreehouseRelease,
    run_sweep,
)


CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"
RELEASE_DIR = CACHE_ROOT / "treehouse-polya-25-01"

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
        "histology split via GDC primary_diagnosis lookup"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_tcga_sarc_histology_log2tpm_to_tpm",
)


TREEHOUSE_DIRECT = [
    ("SARC_ANGIO", "angiosarcoma"),
    ("SARC_ASPS", "alveolar soft part sarcoma"),
    ("SARC_DSRCT", "desmoplastic small round cell tumor"),
    ("SARC_EPITH", "epithelioid sarcoma"),
    ("SARC_IMT", "inflammatory myofibroblastic tumor"),
    ("SARC_IFS", "infantile fibrosarcoma"),
    ("SARC_MPNST", "malignant peripheral nerve sheath tumor"),
    ("SARC_EHE", "epithelioid hemangioendothelioma"),
]

# TCGA-SARC primary_diagnosis → registry code
TCGA_SARC_OVERLAYS = [
    ("SARC_DDLPS", "Dedifferentiated liposarcoma", "liposarcoma"),
    ("SARC_PLEOLPS", "Pleomorphic liposarcoma", "liposarcoma"),
]


def _build_histology_predicate(histology: pd.DataFrame, primary_diagnosis: str):
    cases = set(
        histology.loc[
            histology["primary_diagnosis"].eq(primary_diagnosis),
            "submitter_id",
        ].astype(str)
    )

    def _pred(row: dict) -> bool:
        dsid = str(row.get("th_dataset_id", ""))
        if not dsid.startswith("TCGA"):
            return False
        case_id = "-".join(dsid.split("-")[:3])
        return case_id in cases
    return _pred


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

    # (a) Treehouse direct
    print("=== Treehouse direct (8 codes) ===")
    cohorts_direct = [
        TreehouseCohort(
            cancer_code=code,
            disease_label=label,
            extra_notes=(
                f"All Treehouse samples labelled disease == '{label}'. "
                "Mostly Treehouse-internal + publicly-available "
                "non-TCGA repositories."
            ),
        )
        for code, label in TREEHOUSE_DIRECT
    ]
    run_sweep(
        RELEASE_TREEHOUSE,
        cohorts_direct,
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        refresh_cache=args.refresh_cache,
    )

    # (b) TCGA-SARC histology overlays
    print()
    print("=== TCGA-SARC histology overlays (2 codes) ===")
    histology_cache = RELEASE_DIR / "derived" / "tcga_sarc_histology.csv"
    if not histology_cache.exists():
        # Re-fetch via the SARC subtypes script's helper
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location(
            "_sarc_subtypes",
            Path(__file__).parent / "sweep_treehouse_sarc_subtypes.py",
        )
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._fetch_sarc_histology(histology_cache)
    histology = pd.read_csv(histology_cache)

    cohorts_tcga = []
    for code, primary_diagnosis, disease_label in TCGA_SARC_OVERLAYS:
        cohorts_tcga.append(
            TreehouseCohort(
                cancer_code=code,
                disease_label=disease_label,
                sample_predicate=_build_histology_predicate(
                    histology, primary_diagnosis,
                ),
                extra_notes=(
                    f"TCGA-SARC primary_diagnosis = "
                    f"'{primary_diagnosis}'. Routed out of the "
                    "TCGA-SARC project via GDC histology lookup."
                ),
                cache_stem=f"tcga_sarc_{code.removeprefix('SARC_').lower()}",
            )
        )
    run_sweep(
        RELEASE_TCGA,
        cohorts_tcga,
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        refresh_cache=args.refresh_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
