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

# GSE30929 (Singer 2007 MSKCC) — 140 primary liposarcoma cases on
# Affymetrix HG-U133A (GPL96), explicitly histology-tagged via
# char_subtype: 52 WD-LPS / 40 DD-LPS / 20 pleomorphic / 17 myxoid /
# 11 myxoid-round-cell. The round-cell variant IS high-grade
# myxoid LPS (WHO grade III), so both collapse to SARC_MYXLPS (n=28).
GSE30929 = dict(
    series_matrix_url=(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE30nnn/GSE30929/"
        "matrix/GSE30929_series_matrix.txt.gz"
    ),
    series_matrix_filename="GSE30929_series_matrix.txt.gz",
    platform_id="GPL96",
    platform_name="Affymetrix HG-U133A (GPL96)",
    source_project="GEO",
    citation=(
        "PMID 22241786 (Singer 2007 MSKCC liposarcoma 140-sample "
        "expression-classifier cohort)"
    ),
    source_cohort="GSE30929_SINGER_2007_LPS",
)

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
        cancer_code="SARC_ESS_LG",
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
        cancer_code="SARC_ESS_HG",
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


def _build_gse30929_subtype(
    cancer_code: str, include_regex: str, n_expected: int, notes_extra: str,
    summary_output: Path, ensembl_release: int,
) -> int:
    return build_microarray_source(
        series_matrix_url=GSE30929["series_matrix_url"],
        series_matrix_filename=GSE30929["series_matrix_filename"],
        platform_id=GSE30929["platform_id"],
        platform_name=GSE30929["platform_name"],
        source_project=GSE30929["source_project"],
        citation=GSE30929["citation"],
        source_cohort=GSE30929["source_cohort"],
        cache_dir=CACHE_ROOT / "gse30929-lps",
        cancer_code=cancer_code,
        summary_output=summary_output,
        ensembl_release=ensembl_release,
        sample_include_regex=include_regex,
        tumor_origin="primary",
        extra_notes=(
            f"{cancer_code} subset of GSE30929 (Singer 2007 MSKCC, n≈"
            f"{n_expected} of 140 total LPS). Routed via "
            f"char_subtype={include_regex!r}. {notes_extra}"
        ),
    )


def build_lps_subtypes(summary_output: Path, ensembl_release: int) -> dict[str, int]:
    """GSE30929 → 4 LPS shards (WDLPS, DDLPS, PLEOLPS, MYXLPS)."""
    out = {}
    out["SARC_WDLPS"] = _build_gse30929_subtype(
        "SARC_WDLPS", r"(?i)well-differentiated", 52,
        "Includes atypical lipomatous tumor (ALT) — same WHO entity "
        "as WDLPS.", summary_output, ensembl_release,
    )
    out["SARC_DDLPS"] = _build_gse30929_subtype(
        "SARC_DDLPS", r"(?i)dedifferentiated", 40,
        "Complements existing GSE75885 (n=19) + TCGA (n=48) primary "
        "DDLPS cohorts.", summary_output, ensembl_release,
    )
    out["SARC_PLEOLPS"] = _build_gse30929_subtype(
        "SARC_PLEOLPS", r"(?i)pleomorphic", 20,
        "Complements existing GSE75885 (n=4) + TCGA (n=2) primary "
        "pleomorphic LPS.", summary_output, ensembl_release,
    )
    out["SARC_MYXLPS"] = _build_gse30929_subtype(
        "SARC_MYXLPS", r"(?i)myxoid", 28,
        "Includes both classic myxoid (n=17) and myxoid/round-cell "
        "(n=11) — the round-cell variant is the high-grade form of "
        "the same FUS::DDIT3-fusion entity, not a distinct disease.",
        summary_output, ensembl_release,
    )
    return out


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
        "--only", choices=("ess_lg", "ess_hg", "mtc", "lps"), action="append",
        help="Build only the specified cohort group(s). Default: all.",
    )
    args = parser.parse_args()
    wanted = set(args.only) if args.only else {"ess_lg", "ess_hg", "mtc", "lps"}
    print(f"building cohorts: {sorted(wanted)}")

    counts = {}
    if "ess_lg" in wanted:
        print("\n=== ESS_LG via GSE85383 ===")
        counts["SARC_ESS_LG"] = build_ess_lg(args.summary_output, args.ensembl_release)
    if "ess_hg" in wanted:
        print("\n=== ESS_HG via GSE85383 ===")
        counts["SARC_ESS_HG"] = build_ess_hg(args.summary_output, args.ensembl_release)
    if "mtc" in wanted:
        print("\n=== MTC via GSE32662 ===")
        counts["MTC"] = build_mtc(args.summary_output, args.ensembl_release)
    if "lps" in wanted:
        print("\n=== 4 LPS subtypes via GSE30929 (Singer 2007 MSKCC) ===")
        counts.update(build_lps_subtypes(args.summary_output, args.ensembl_release))
    print(f"\ndone. samples per cohort: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
