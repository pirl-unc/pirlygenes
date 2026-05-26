#!/usr/bin/env python
"""Thin wrapper around pirlygenes.builders.treehouse for the 25.01 RiboD sweep.

Builds the two cancer codes the Treehouse RiboD compendium covers:
CHOR (chordoma) and RB (retinoblastoma). The RiboD compendium uses
ribo-depleted library prep so values for genes in the technical-RNA
panel (rRNA / mtRNA) are not directly comparable to PolyA cohorts —
keep these under TREEHOUSE_RIBOD_25_01 source_cohort so consumers
can hold the assay difference.
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
    source_id="treehouse-ribod-25-01",
    source_cohort="TREEHOUSE_RIBOD_25_01",
    source_project="Treehouse",
    release_label=(
        "Treehouse Tumor Compendium 25.01 RiboDeplete "
        "(hugo_log2tpm_58581genes_2025-03-19); downloaded from "
        "public.gi.ucsc.edu/~ekephart/public-data/"
    ),
    tpm_filename="Tumor-25.01-Ribodeplete_hugo_log2tpm_58581genes_2025-03-19.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-RiboD_20250306v1.tsv",
    cache_dir=CACHE_ROOT / "treehouse-ribod-25-01",
    pipeline_prefix="treehouse_ribod_25_01_log2tpm_to_tpm",
)

COHORTS = [
    TreehouseCohort("CHOR", "chordoma"),
    TreehouseCohort("RB", "retinoblastoma"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression.csv.gz"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--only", default=None)
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
