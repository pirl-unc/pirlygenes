#!/usr/bin/env python
"""Neuroendocrine code-rename follow-up (#288): scheme-consistency pass.

After the first NE wave (scripts/migrate_ne_rename.py) two codes were flagged
as confusing/inconsistent and renamed for full NET_<site> consistency + to
remove the ambiguous "LC" token (lung-carcinoid vs large-cell):

  - PANNET        -> NET_PANCREAS   (parallel to NET_LUNG/NET_MIDGUT/NET_RECTAL)
  - NEC_LUNG_LC   -> NEC_LUNG_LARGECELL  (LC was ambiguous; spell out large-cell)

Same column-aware + exact-token (semicolon-split) machinery as the first wave.
Old codes (PANNET, NEC_LUNG_LC, and the pre-5.16 LUNG_NET_LCNEC) resolve to the
new names via _RENAMED_CODE_ALIASES.

    python scripts/migrate_ne_rename2.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

DATA = Path("pirlygenes/data")

RENAME = {
    "PANNET": "NET_PANCREAS",
    "NEC_LUNG_LC": "NEC_LUNG_LARGECELL",
}

FILE_COLUMNS = {
    "cancer-type-registry.csv": ["code", "parent_code", "mixture_cohort"],
    "cancer-tmb.csv": ["cancer_code"],
    "cancer-fusions.csv": ["cancer_code"],
    "cancer-key-genes.csv": ["cancer_code"],
    "cancer-lineage-panels.csv": ["Child_Code"],
    "cancer-expression-source-candidates.csv": ["cancer_code", "reference_code"],
    "lineage-genes.csv": ["Cancer_Type"],
    "cancer-code-burden-map.csv": ["cancer_code"],
    "cancer-cohort-aggregates.csv": ["aggregate_code", "member_code"],
    "degenerate-subtype-pairs.csv": ["members"],
    "fusion-surrogate-expression.csv": ["cancer_code"],
    "rare-cancer-fusion-rules.csv": ["cancer_code"],
    "rare-cancer-rna-surrogates.csv": [
        "cancer_code", "context_codes", "excluded_context_codes",
    ],
    "tumor-up-vs-matched-normal.csv": ["cancer_code"],
}


def _remap_cell(value):
    if not isinstance(value, str) or not value:
        return value, 0
    parts = value.split(";")
    new = [RENAME.get(p, p) for p in parts]
    n = sum(1 for a, b in zip(parts, new) if a != b)
    return (";".join(new), n) if n else (value, 0)


def _remap_column(df, col):
    if col not in df.columns:
        return 0
    total, out = 0, []
    for v in df[col]:
        nv, n = _remap_cell(v)
        out.append(nv)
        total += n
    if total:
        df[col] = out
    return total


def main():
    total = 0
    for rel, cols in FILE_COLUMNS.items():
        p = DATA / rel
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype=str)
        changed = sum(_remap_column(df, c) for c in cols)
        if changed:
            df.to_csv(p, index=False)
            print(f"  {rel}: {changed} cell(s) remapped in {cols}")
            total += changed
    for f in glob.glob(str(DATA / "cancer-reference-expression" / "*.csv.gz")):
        df = pd.read_csv(f, dtype=str)
        changed = _remap_column(df, "cancer_code")
        if changed:
            df.to_csv(f, index=False, compression="gzip")
            print(f"  {Path(f).name}: {changed} cancer_code cell(s) remapped")
            total += changed
    print(f"done: {total} cells remapped across data files")


if __name__ == "__main__":
    main()
