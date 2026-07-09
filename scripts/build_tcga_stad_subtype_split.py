#!/usr/bin/env python
"""Split the TCGA-STAD per-sample cohort into the four molecular subtypes.

EBV-positive / MSI (hypermutated, MMRd) / GS (genomically stable, diffuse) /
CIN (chromosomal-instability, intestinal) — Cancer Genome Atlas Research Network
2014 (PMID 25079317). Joins the cBioPortal per-patient SUBTYPE (study
``stad_tcga_pan_can_atlas_2018``) onto the TCGA-STAD samples already in the
Treehouse PolyA per-sample parquet, by case submitter-id — the same per-case-join
pattern as ``build_tcga_ucec_subtype_split.py`` /
``build_tcga_coadread_msi_split.py``. The label map is produced by
``scripts/sweep_treehouse_tcga_stad_subtype.py`` (cached
``cbioportal_stad_subtype.csv``).

Extends trufflepig's MSI-vs-MSS classifier with a gastric MMR axis beyond CRC and
endometrium (pirlygenes#529): STAD_MSI = MSI-positive, STAD_GS/STAD_CIN =
MSS negatives, STAD_EBV = immune-hot holdout confounder (MSS but checkpoint-hot).

Run:  python scripts/sweep_treehouse_tcga_stad_subtype.py   # if label cache absent
      python scripts/build_tcga_stad_subtype_split.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.builders.subtype_split import build_subtype_split
from pirlygenes.downloads import source_cache_dir
from pirlygenes.expression.stats import build_reference_rows

SOURCE_ID = "treehouse-polya-25-01"
SUMMARY_COHORT = "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE"

# cBioPortal SUBTYPE value -> pirlygenes cancer code. The pan-can-atlas STAD study
# labels the four TCGA classes with the ``STAD_`` prefix, mirroring how the UCEC
# study labels ``UCEC_POLE`` etc.; the sweep prints the actual value counts and
# build_subtype_split warns on any code that matches zero samples, so a string
# drift surfaces at build time rather than corrupting a cohort.
SUBTYPE_TO_CODE = {
    "STAD_EBV": "STAD_EBV",
    "STAD_MSI": "STAD_MSI",
    "STAD_GS": "STAD_GS",
    "STAD_CIN": "STAD_CIN",
}
_CODE_DESC = {
    "STAD_EBV": "EBV-positive (PIK3CA-mutant, PD-L1/2-amplified, immune-hot holdout)",
    "STAD_MSI": "MSI-hypermutated / MMRd (MSI-like positive)",
    "STAD_GS": "genomically stable / diffuse (CDH1-RHOA, MSS, often immune-cold)",
    "STAD_CIN": "chromosomal instability / intestinal (TP53-mutant, RTK-RAS-amplified, MSS)",
}


def _label_by_case(cache_csv: Path) -> dict[str, str]:
    if not cache_csv.exists():
        raise SystemExit(
            f"missing STAD subtype label cache {cache_csv} — run "
            "scripts/sweep_treehouse_tcga_stad_subtype.py first")
    df = pd.read_csv(cache_csv)
    return dict(zip(df["patientId"].astype(str), df["stad_subtype"].astype(str)))


def _summary_row(gene_table: pd.DataFrame, values: pd.DataFrame,
                 code: str) -> pd.DataFrame:
    return build_reference_rows(
        gene_table, values,
        cancer_code=code,
        source_cohort=SUMMARY_COHORT,
        source_project="Treehouse (TCGA-STAD) × cBioPortal molecular subtype",
        source_version=(
            "Treehouse 25.01 PolyA TCGA-STAD × cBioPortal "
            "stad_tcga_pan_can_atlas_2018 SUBTYPE (TCGA 2014)"),
        processing_pipeline=(
            "treehouse_polya_25_01_tcga_stad_subtype_log2tpm_to_tpm_clean_tpm_16_9_75"),
        notes=(
            f"{code}: TCGA-STAD samples in Treehouse 25.01 PolyA split by "
            f"cBioPortal molecular subtype — {_CODE_DESC.get(code, code)}. "
            "PMID 25079317."),
        tumor_origin="primary",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-output", type=Path,
                    default=Path("pirlygenes/data/cancer-reference-expression"))
    args = ap.parse_args()
    cache_csv = (source_cache_dir(SOURCE_ID) / "derived"
                 / "cbioportal_stad_subtype.csv")
    label_by_case = _label_by_case(cache_csv)
    n = {}
    for v in label_by_case.values():
        n[v] = n.get(v, 0) + 1
    print(f"cBioPortal STAD subtype labels: {n}", flush=True)

    written = build_subtype_split(
        source_id=SOURCE_ID, parent_code="STAD",
        label_by_case=label_by_case, code_by_label=SUBTYPE_TO_CODE,
        summary_cohort=SUMMARY_COHORT, summary_output=args.summary_output,
        make_summary_row=_summary_row,
    )
    print(f"STAD subtype split: built {sorted(written)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
