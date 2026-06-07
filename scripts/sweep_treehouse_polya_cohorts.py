#!/usr/bin/env python
"""Thin wrapper around pirlygenes.builders.treehouse for the 25.01 PolyA sweep.

Builds the 15 pediatric / sarcoma cancer codes the Treehouse PolyA
compendium covers. See docs/archive/expression-data-audit-2026-05.md Status B
Treehouse-PolyA section for the disease-label mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pirlygenes.builders.treehouse import (
    TreehouseCohort,
    TreehouseRelease,
    run_sweep,
)


CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"

RELEASE = TreehouseRelease(
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
    cache_dir=CACHE_ROOT / "treehouse-polya-25-01",
    pipeline_prefix="treehouse_polya_25_01_log2tpm_to_tpm",
)

COHORTS = [
    TreehouseCohort("ATRT", "atypical teratoid/rhabdoid tumor"),
    TreehouseCohort("SARC_EWS", "Ewing sarcoma"),
    TreehouseCohort("HEPB", "hepatoblastoma"),
    TreehouseCohort("MBL", "medulloblastoma"),
    TreehouseCohort("NUTM", "NUT midline carcinoma"),
    TreehouseCohort("SARC_OS", "osteosarcoma"),
    TreehouseCohort("SARC_RMS_ARMS", "alveolar rhabdomyosarcoma"),
    TreehouseCohort("SARC_RMS_ERMS", "embryonal rhabdomyosarcoma"),
    TreehouseCohort("SARC_RMS_PRMS", "pleomorphic rhabdomyosarcoma"),
    TreehouseCohort("SARC_RMS_SSRMS", "spindle cell/sclerosing rhabdomyosarcoma"),
    TreehouseCohort("SARC_LMS", "leiomyosarcoma"),
    TreehouseCohort("SARC_LPS_UNSPEC", "liposarcoma"),
    TreehouseCohort("SARC_MYXFIB", "myxofibrosarcoma"),
    TreehouseCohort("SARC_SYN", "synovial sarcoma"),
    TreehouseCohort("SARC_UPS", "undifferentiated pleomorphic sarcoma"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated cancer_codes; restrict the sweep to these.",
    )
    args = parser.parse_args()

    cohorts = COHORTS
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        cohorts = [c for c in COHORTS if c.cancer_code in wanted]
        if not cohorts:
            raise SystemExit(f"--only={args.only!r} matched no cohorts")

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
