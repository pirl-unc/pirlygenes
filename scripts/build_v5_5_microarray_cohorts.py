#!/usr/bin/env python
"""Build 3 microarray TPM-proxy shards for v5.5.0.

ESS_HG + ESS_LG via GSE85383 (Yoshida 2017, Agilent SurePrint G3
GPL22303): 4 HG-ESS, 9 LG-ESS, 8 UUS, 4 LMS in one platform table.
We split via Sample_characteristics_ch1 regex into two cohorts.

MTC via GSE32662 (Costanzo 2011, Agilent GPL6480): 49 primary
medullary thyroid carcinoma cases on a 4x44K array.

All three use the generalized microarray builder
(``pirlygenes.builders.affy_gpl570.build_microarray_source``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pirlygenes.builders.affy_gpl570 import build_microarray_source

CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"

SUMMARY_OUTPUT_DEFAULT = Path("pirlygenes/data/cancer-reference-expression")

# GSE85383 metadata: probe table + per-cohort routing regex (matched
# against Sample_characteristics_ch1 fields exposed by series_matrix
# parsing).
GSE85383 = dict(
    series_matrix_url=(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE85nnn/GSE85383/"
        "matrix/GSE85383_series_matrix.txt.gz"
    ),
    series_matrix_filename="GSE85383_series_matrix.txt.gz",
    platform_id="GPL22303",
    platform_name=(
        "Agilent SurePrint G3 Human Gene Expression v3 8x60K (GPL22303)"
    ),
    source_project="GEO",
    citation="PMID 29066508 (Yoshida 2017 ESS / UUS)",
)

GSE32662 = dict(
    series_matrix_url=(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE32nnn/GSE32662/"
        "matrix/GSE32662_series_matrix.txt.gz"
    ),
    series_matrix_filename="GSE32662_series_matrix.txt.gz",
    platform_id="GPL6480",
    platform_name="Agilent Human Genome 4x44K v1 (GPL6480)",
    source_project="GEO",
    citation="GSE32662 (Pringle 2012 medullary thyroid carcinoma)",
)


def build_ess_lg(summary_output: Path, ensembl_release: int) -> int:
    return build_microarray_source(
        **GSE85383,
        cache_dir=CACHE_ROOT / "gse85383-ess",
        cancer_code="ESS_LG",
        source_cohort="GSE85383_YOSHIDA_2017_ESS",
        summary_output=summary_output,
        ensembl_release=ensembl_release,
        # Match low-grade ESS in any char_* field. Yoshida 2017
        # tagged samples as "low-grade endometrial stromal sarcoma".
        sample_include_regex=r"(?i)low.?grade",
        tumor_origin="primary",
        extra_notes=(
            "LG-ESS subset of GSE85383 (n≈9 of 25 total cohort). "
            "Other histologies (UUS, HG-ESS, LMS) excluded via the "
            "low-grade include regex."
        ),
    )


def build_ess_hg(summary_output: Path, ensembl_release: int) -> int:
    return build_microarray_source(
        **GSE85383,
        cache_dir=CACHE_ROOT / "gse85383-ess",
        cancer_code="ESS_HG",
        source_cohort="GSE85383_YOSHIDA_2017_ESS",
        summary_output=summary_output,
        ensembl_release=ensembl_release,
        sample_include_regex=r"(?i)high.?grade",
        tumor_origin="primary",
        extra_notes=(
            "HG-ESS subset of GSE85383 (n≈4 of 25 total cohort). "
            "Low-n — interpret with caution."
        ),
    )


def build_mtc(summary_output: Path, ensembl_release: int) -> int:
    return build_microarray_source(
        **GSE32662,
        cache_dir=CACHE_ROOT / "gse32662-mtc",
        cancer_code="MTC",
        source_cohort="GSE32662_PRINGLE_2012_MTC",
        summary_output=summary_output,
        ensembl_release=ensembl_release,
        # No filtering — entire series is primary MTC (49 cases, 52
        # hybridized samples with 3 replicates).
        tumor_origin="primary",
        extra_notes=(
            "49 primary medullary thyroid carcinoma cases "
            "(52 hybridized samples include 3 replicates)."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-output", type=Path, default=SUMMARY_OUTPUT_DEFAULT,
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--only", choices=("ess_lg", "ess_hg", "mtc"), action="append",
        help="Build only the specified cohort(s). Default: all 3.",
    )
    args = parser.parse_args()
    wanted = set(args.only) if args.only else {"ess_lg", "ess_hg", "mtc"}
    print(f"building cohorts: {sorted(wanted)}")

    counts = {}
    if "ess_lg" in wanted:
        print("\n=== ESS_LG via GSE85383 ===")
        counts["ESS_LG"] = build_ess_lg(args.summary_output, args.ensembl_release)
    if "ess_hg" in wanted:
        print("\n=== ESS_HG via GSE85383 ===")
        counts["ESS_HG"] = build_ess_hg(args.summary_output, args.ensembl_release)
    if "mtc" in wanted:
        print("\n=== MTC via GSE32662 ===")
        counts["MTC"] = build_mtc(args.summary_output, args.ensembl_release)
    print(f"\ndone. samples per cohort: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
