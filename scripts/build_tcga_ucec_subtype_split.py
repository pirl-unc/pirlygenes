#!/usr/bin/env python
"""Split the TCGA-UCEC per-sample cohort into the four molecular subtypes.

POLE-ultramutated / MSI-hypermutated (MMRd) / copy-number-low (NSMP, MSS) /
copy-number-high (p53-abnormal serous-like, MSS) — Kandoth 2013 (PMID 23636398).
Joins the cBioPortal per-patient SUBTYPE (study ``ucec_tcga_pan_can_atlas_2018``)
onto the TCGA-UCEC samples already in the Treehouse PolyA per-sample parquet, by
case submitter-id — the same per-case-join pattern as
``build_tcga_coadread_msi_split.py``. The label map is produced by
``scripts/sweep_treehouse_tcga_ucec_subtype.py`` (cached
``cbioportal_ucec_subtype.csv``).

Gives trufflepig's MSI-vs-MSS classifier the MMR/POLE axis beyond CRC
(pirlygenes#529): UCEC_MSI = MMRd-positive, UCEC_CNL/UCEC_CNH = pMMR/MSS
negatives, UCEC_POLE = ultramutated holdout confounder.

Run:  python scripts/sweep_treehouse_tcga_ucec_subtype.py   # if label cache absent
      python scripts/build_tcga_ucec_subtype_split.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.builders.subtype_split import build_subtype_split
from pirlygenes.downloads import source_cache_dir
from pirlygenes.expression.stats import build_reference_rows

SOURCE_ID = "treehouse-polya-25-01"
SUMMARY_COHORT = "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE"

# cBioPortal SUBTYPE value -> pirlygenes cancer code.
SUBTYPE_TO_CODE = {
    "UCEC_POLE": "UCEC_POLE",
    "UCEC_MSI": "UCEC_MSI",
    "UCEC_CN_LOW": "UCEC_CNL",
    "UCEC_CN_HIGH": "UCEC_CNH",
}
_CODE_DESC = {
    "UCEC_POLE": "POLE-ultramutated (checkpoint-hyper-responsive holdout)",
    "UCEC_MSI": "MSI-hypermutated / MMRd (MSI-like positive)",
    "UCEC_CNL": "copy-number-low / NSMP (p53-wild-type endometrioid, MSS)",
    "UCEC_CNH": "copy-number-high / p53-abnormal serous-like (MSS, immune-cold)",
}


def _label_by_case(cache_csv: Path) -> dict[str, str]:
    if not cache_csv.exists():
        raise SystemExit(
            f"missing UCEC subtype label cache {cache_csv} — run "
            "scripts/sweep_treehouse_tcga_ucec_subtype.py first")
    df = pd.read_csv(cache_csv)
    return dict(zip(df["patientId"].astype(str), df["ucec_subtype"].astype(str)))


def _summary_row(gene_table: pd.DataFrame, values: pd.DataFrame,
                 code: str) -> pd.DataFrame:
    return build_reference_rows(
        gene_table, values,
        cancer_code=code,
        source_cohort=SUMMARY_COHORT,
        source_project="Treehouse (TCGA-UCEC) × cBioPortal molecular subtype",
        source_version=(
            "Treehouse 25.01 PolyA TCGA-UCEC × cBioPortal "
            "ucec_tcga_pan_can_atlas_2018 SUBTYPE (Kandoth 2013)"),
        processing_pipeline=(
            "treehouse_polya_25_01_tcga_ucec_subtype_log2tpm_to_tpm_clean_tpm_16_9_75"),
        notes=(
            f"{code}: TCGA-UCEC samples in Treehouse 25.01 PolyA split by "
            f"cBioPortal molecular subtype — {_CODE_DESC.get(code, code)}. "
            "PMID 23636398."),
        tumor_origin="primary",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-output", type=Path,
                    default=Path("pirlygenes/data/cancer-reference-expression"))
    args = ap.parse_args()
    cache_csv = (source_cache_dir(SOURCE_ID) / "derived"
                 / "cbioportal_ucec_subtype.csv")
    label_by_case = _label_by_case(cache_csv)
    n = {}
    for v in label_by_case.values():
        n[v] = n.get(v, 0) + 1
    print(f"cBioPortal UCEC subtype labels: {n}", flush=True)

    written = build_subtype_split(
        source_id=SOURCE_ID, parent_code="UCEC",
        label_by_case=label_by_case, code_by_label=SUBTYPE_TO_CODE,
        summary_cohort=SUMMARY_COHORT, summary_output=args.summary_output,
        make_summary_row=_summary_row,
    )
    print(f"UCEC subtype split: built {sorted(written)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
