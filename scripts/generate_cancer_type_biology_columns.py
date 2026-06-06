#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License").
"""Regenerate the biology-attribute columns of ``cancer-type-registry.csv``:
``viral_etiology``, ``viral_agent``, ``fusion_driven``, ``fusion_driver``.

These are the centralized per-cancer-type facts surfaced by
``pirlygenes.cancer_types`` so they aren't curated piecemeal elsewhere.

- **fusion** is *derived* from the cited ``cancer-fusions.csv``: ``defining``
  when any row is ``is_defining``; ``subtype`` when recurrent named fusion(s)
  exist but none is defining; ``rare`` for the curated minority-subset entities
  below; ``none`` otherwise (mutation/CNV/viral-driven or complex-karyotype
  types with no recurrent fusion). ``fusion_driver`` lists the canonical
  fusion(s) for defining/subtype types.
- **viral** is a small curated controlled vocabulary of well-established
  textbook associations; every other type is ``none``. No fabrication — cells
  without a clear public source stay ``none``/blank.

Re-run after editing ``cancer-fusions.csv`` or the curation dicts here:

    python scripts/generate_cancer_type_biology_columns.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pirlygenes.load_dataset import get_data

REGISTRY = Path("pirlygenes/data/cancer-type-registry.csv")

# Recurrent fusions only in a minority subset (not defining, no single recurrent
# driver to name) — curated, cited in cancer-fusions.csv notes.
FUSION_RARE = {"SARC_GIST"}  # NTRK/FGFR-rearranged in the KIT/PDGFRA-WT subset

# Well-established viral etiologies: code -> (etiology, agent). etiology is
# 'defining' (virus defines the entity/subtype) or 'subset' (drives a
# meaningful minority). Everything not listed is ('none', '').
VIRAL = {
    "HNSC_HPVpos": ("defining", "HPV"),
    "CESC": ("defining", "HPV"),
    "ANSC": ("defining", "HPV"),
    "VAGC": ("defining", "HPV"),
    "PENSCC": ("subset", "HPV"),
    "VSCC": ("subset", "HPV"),
    "NPC": ("defining", "EBV"),
    "NEC_MERKEL": ("defining", "MCPyV"),
    "SARC_KS": ("defining", "HHV8"),
    "LIHC": ("subset", "HBV;HCV"),
    "STAD": ("subset", "EBV"),
    "DLBC": ("subset", "EBV"),
    "BL": ("subset", "EBV"),
    "HL": ("subset", "EBV"),
}


def _fusion_columns(fus: pd.DataFrame) -> tuple[dict, dict]:
    driven, driver = {}, {}
    for code, d in fus.groupby("cancer_code"):
        code = str(code)
        is_def = d["is_defining"].astype(str).str.lower().isin(["true", "1", "yes"])
        src = d[is_def] if is_def.any() else d
        drivers = []
        for r in src.itertuples():
            g5, g3 = str(getattr(r, "gene_5prime", "")), str(getattr(r, "gene_3prime", ""))
            if g5 not in ("", "nan") and g3 not in ("", "nan"):
                drivers.append(f"{g5}-{g3}")
        drivers = list(dict.fromkeys(drivers))
        driven[code] = "defining" if is_def.any() else ("subtype" if drivers else "none")
        driver[code] = "; ".join(drivers)
    for c in FUSION_RARE:
        driven[c] = "rare"
    return driven, driver


def main() -> int:
    reg = get_data("cancer-type-registry.csv").copy()
    fus = get_data("cancer-fusions")
    fdriven, fdriver = _fusion_columns(fus)
    codes = reg["code"].astype(str)
    reg["viral_etiology"] = [VIRAL.get(c, ("none", ""))[0] for c in codes]
    reg["viral_agent"] = [VIRAL.get(c, ("none", ""))[1] for c in codes]
    reg["fusion_driven"] = [fdriven.get(c, "none") for c in codes]
    reg["fusion_driver"] = [fdriver.get(c, "") for c in codes]
    reg.to_csv(REGISTRY, index=False)
    print(
        f"wrote {len(reg)} rows | viral non-none: "
        f"{int((reg.viral_etiology != 'none').sum())} | fusion: "
        f"{reg.fusion_driven.value_counts().to_dict()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
