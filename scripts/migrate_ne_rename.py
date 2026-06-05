#!/usr/bin/env python
"""Neuroendocrine code-rename migration (#288), reproducible + column-targeted.

Completes the Phase-C rename wave for the neuroendocrine codes and applies the
one-separator normalization the audit doc called for, across the cancer-type
registry, the bundled per-source reference-expression shards, the curated
panels, and the rule/aggregate tables.

Scheme (mirrors the SARC_/NET_/NEC_ convention; PANNET stays — it already
encodes "pancreatic NET"):
  - well-differentiated NET   -> NET_<site>
  - poorly-differentiated NEC -> NEC_<entity>
  - subtype codes carry a single separator (no internal underscore in the
    subtype token): NBL_MYCN_amp -> NBL_MYCNamp, LAML_ELN_Fav -> LAML_ELNfav.

Deliberately **column-aware** and **exact-token** (splitting semicolon-joined
cells), because some old codes (notably ``MEC``) are substrings of gene symbols
(``MECOM``/``MECP2``) elsewhere that must NOT be touched.

Re-runnable / idempotent. Pair with ``CANCER_TYPE_ALIASES`` /
``_RENAMED_CODE_ALIASES`` so old codes keep resolving.

    python scripts/migrate_ne_rename.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

DATA = Path("pirlygenes/data")

RENAME = {
    # neuroendocrine NET_ (well-diff) / NEC_ (poorly-diff)
    "MID_NET": "NET_MIDGUT",
    "REC_NET": "NET_RECTAL",
    "LUNG_NET_LC": "NET_LUNG",
    "LUNG_NET_LCNEC": "NEC_LUNG_LC",
    "MEC": "NEC_MERKEL",
    # one-separator normalization (subtype token has no internal underscore)
    "NBL_MYCN_amp": "NBL_MYCNamp",
    "NBL_MYCN_nonamp": "NBL_MYCNnonamp",
    "LAML_ELN_Fav": "LAML_ELNfav",
    "LAML_ELN_Int": "LAML_ELNint",
    "LAML_ELN_Adv": "LAML_ELNadv",
}

# {file (relative to DATA): [code-bearing columns]}. Values may be
# semicolon-joined lists of codes (e.g. degenerate-subtype-pairs.members,
# rna-surrogate context columns) — handled token-wise. Reference-expression
# shards are discovered dynamically (cancer_code only). cancer-driver-genes /
# cancer-type-genes are intentionally absent: their "MEC" is the MECOM gene.
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
    """Exact-token remap of a single cell, splitting a ``;``-joined list so each
    code is matched whole (MECOM is never touched). Returns (new_value, n)."""
    if not isinstance(value, str) or not value:
        return value, 0
    parts = value.split(";")
    new = [RENAME.get(p, p) for p in parts]
    n = sum(1 for a, b in zip(parts, new) if a != b)
    return (";".join(new), n) if n else (value, 0)


def _remap_column(df, col):
    if col not in df.columns:
        return 0
    total = 0
    out = []
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
