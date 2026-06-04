#!/usr/bin/env python
"""Phase C registry rename migration (reproducible, column-targeted).

Renames sarcoma cohort codes to the consistent ``SARC_`` prefix and normalises
the HNSC HPV subtype codes, across the cancer-type registry, the bundled
per-source reference-expression CSVs, the curated panels, and the TMB table.

This is deliberately **column-aware** — it only rewrites values in known
cohort-code columns of known files — because some old codes (notably ``OS``,
``EWS``) are common substrings / gene aliases elsewhere (e.g.
``ncbi-symbol-synonyms.csv.gz`` ``alias`` column) that must NOT be touched.

Re-runnable / idempotent: values already renamed are left alone.

    python scripts/migrate_phase_c_rename.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

DATA = Path("pirlygenes/data")

RENAME = {
    "OS": "SARC_OS",
    "EWS": "SARC_EWS",
    "CHON": "SARC_CHON",
    "CHOR": "SARC_CHOR",
    "GCTB": "SARC_GCTB",
    "ESS_LG": "SARC_ESS_LG",
    "ESS_HG": "SARC_ESS_HG",
    "RMS_ERMS": "SARC_RMS_ERMS",
    "RMS_ARMS": "SARC_RMS_ARMS",
    "RMS_PRMS": "SARC_RMS_PRMS",
    "RMS_SSRMS": "SARC_RMS_SSRMS",
    "HNSC_HPV_pos": "HNSC_HPVpos",
    "HNSC_HPV_neg": "HNSC_HPVneg",
}

# {file (relative to DATA): [code-bearing columns to remap]}. Bundled
# reference-expression shards are discovered dynamically below.
FILE_COLUMNS = {
    "cancer-type-registry.csv": ["code", "parent_code", "mixture_cohort"],
    "cancer-tmb.csv": ["cancer_code"],
    "cancer-fusions.csv": ["cancer_code"],
    "cancer-key-genes.csv": ["cancer_code"],
    "cancer-lineage-panels.csv": ["Child_Code"],
    "cancer-expression-source-candidates.csv": ["cancer_code"],
    "fusion-surrogate-expression.csv": ["cancer_code"],
    "rare-cancer-fusion-rules.csv": ["cancer_code"],
    "rare-cancer-rna-surrogates.csv": ["cancer_code"],
    "tumor-up-vs-matched-normal.csv": ["cancer_code"],
}


def _remap_column(df, col):
    if col not in df.columns:
        return 0
    n = df[col].isin(RENAME).sum()
    if n:
        df[col] = df[col].map(lambda v: RENAME.get(v, v))
    return int(n)


def main():
    total = 0
    # 1. curated CSVs (plain text)
    for rel, cols in FILE_COLUMNS.items():
        p = DATA / rel
        df = pd.read_csv(p, dtype=str)
        changed = sum(_remap_column(df, c) for c in cols)
        if changed:
            df.to_csv(p, index=False)
            print(f"  {rel}: {changed} cell(s) remapped in {cols}")
            total += changed
    # 2. bundled per-source reference-expression shards (gzip) — cancer_code only
    for f in glob.glob(str(DATA / "cancer-reference-expression" / "*.csv.gz")):
        df = pd.read_csv(f, dtype=str)
        if "cancer_code" not in df.columns:
            continue
        changed = _remap_column(df, "cancer_code")
        if changed:
            df.to_csv(f, index=False, compression="gzip")
            print(f"  {Path(f).name}: {changed} cancer_code cell(s) remapped")
            total += changed
    print(f"done: {total} cells remapped across data files")


if __name__ == "__main__":
    main()
