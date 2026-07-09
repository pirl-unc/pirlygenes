#!/usr/bin/env python
"""Generate the cached cBioPortal STAD molecular-class map used by the TCGA-STAD
subtype split (scripts/build_tcga_stad_subtype_split.py).

Fetches the TCGA molecular class per TCGA-STAD patient (EBV / MSI / GS / CIN)
from cBioPortal study ``stad_tcga_pan_can_atlas_2018`` (SUBTYPE attribute; the
four TCGA gastric classes, Cancer Genome Atlas Research Network 2014,
PMID 25079317) and writes ``<treehouse cache>/derived/cbioportal_stad_subtype.csv``
(``patientId,stad_subtype``).

The build reads that map and splits the Treehouse TCGA-STAD samples by case id —
the same per-case-join pattern as the UCEC / COAD-READ-MSI splits. (The exact
SUBTYPE strings cBioPortal returns are printed here as a value-count summary; the
builder's ``SUBTYPE_TO_CODE`` must match them, and warns loudly on any that
don't.)

    python scripts/sweep_treehouse_tcga_stad_subtype.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = (Path.home() / ".cache" / "pirlygenes" / "expression"
         / "treehouse-polya-25-01" / "derived")
CACHE_CSV = CACHE / "cbioportal_stad_subtype.csv"
URL = (
    "https://www.cbioportal.org/api/studies/stad_tcga_pan_can_atlas_2018"
    "/clinical-data?clinicalDataType=PATIENT&attributeId=SUBTYPE"
)


def main() -> int:
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = json.load(r)
    df = pd.DataFrame(
        [{"patientId": d["patientId"], "stad_subtype": d["value"]} for d in data]
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    print(f"wrote {CACHE_CSV} ({len(df)} patients): "
          f"{df['stad_subtype'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
