#!/usr/bin/env python
"""Generate the cached cBioPortal UCEC molecular-class map used by the CTA-plot
UCEC subtype split (analyses/cta_patient_counts.py).

Fetches the TCGA molecular class per TCGA-UCEC patient (POLE / MSI / CN_LOW /
CN_HIGH) from cBioPortal study ``ucec_tcga_pan_can_atlas_2018`` (SUBTYPE
attribute; the four TCGA endometrial classes, Kandoth 2013 PMID 23636398) and
writes ``<treehouse cache>/derived/cbioportal_ucec_subtype.csv``
(``patientId,ucec_subtype``).

The plot's ``_cbioportal_derived_cohorts`` reads that map and splits the
Treehouse TCGA-UCEC samples by case id — the same per-sample-reconstructable
pattern as the BRCA PAM50 / HNSC HPV splits. (Building full reference-expression
shards for the four subtypes is a separate, heavier step; this map is all the
CTA plots need.)

    python scripts/sweep_treehouse_tcga_ucec_subtype.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = (Path.home() / ".cache" / "pirlygenes" / "expression"
         / "treehouse-polya-25-01" / "derived")
CACHE_CSV = CACHE / "cbioportal_ucec_subtype.csv"
URL = (
    "https://www.cbioportal.org/api/studies/ucec_tcga_pan_can_atlas_2018"
    "/clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE"
)


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = json.load(r)
    df = pd.DataFrame(
        [{"patientId": d["patientId"], "ucec_subtype": d["value"]} for d in data]
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    print(f"wrote {CACHE_CSV} ({len(df)} patients): "
          f"{df['ucec_subtype'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
