"""Generate tumor-up-vs-matched-normal panel rows for a cancer code.

Reproducible recipe used to seed the curated ``tumor-up-vs-matched-
normal.csv`` for cancer types whose TCGA cohort data is already in
``pan-cancer-expression.csv`` (column ``tcga_<code>``) and whose
primary_tissue has an HPA normal reference (column ``nTPM_<tissue>``).

Filter logic:

- fold = tumor_tpm / (matched_normal_ntpm + 1.0)   # +1.0 smoothing
- keep genes with fold >= 10 AND tumor_tpm >= 5
- drop genes where max(immune, muscle, fat) nTPM exceeds the tumor TPM
  (stromal-dominated genes)
- drop housekeeping / MT / ribosomal / Ig-TR rearranged / histone /
  olfactory+taste receptors / hyphenated read-through names / a
  curated TCGA-median-outlier blocklist
- take top-10 per cancer by fold

Usage::

    python scripts/generate_matched_normal.py LUAD lung
    python scripts/generate_matched_normal.py --all-tcga-missing

The script prints CSV-ready rows to stdout; append to the catalog
and shrink ``_MISSING_MATCHED_NORMAL`` in the completeness test.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd


_EXCLUDED_SYMBOLS = {
    # TCGA-cohort-median outlier artifacts (low-abundance pseudogenes
    # or single-sample-driven medians). Add to this set when manual
    # review rejects a row.
    "INS", "GCG", "A1BG", "ZACN", "RAMACL", "ZFTRAF1", "ITGAE",
    "WASH6P", "MKKS", "SLC39A4", "ECE2", "HERC3", "H3Y1", "H4C11",
    "H2BC12L", "FUNDC2", "TBL3", "SRRM3", "SMN2", "KCTD2", "MTCP1",
    "CCDC163", "ZDHHC19", "TMEM265", "TMC4", "C4A", "LGALS13",
    "DCAF8L1", "CCNI2", "C19orf81", "TMEM151A", "LYZL6", "ZNF716",
}


def _build_exclusion_set(df):
    from pirlygenes.gene_sets_cancer import (
        housekeeping_gene_names, mitochondrial_gene_names,
    )
    exclude = set(housekeeping_gene_names()) | set(mitochondrial_gene_names())
    exclude |= _EXCLUDED_SYMBOLS
    for sym in df.index:
        if not isinstance(sym, str):
            continue
        if sym.startswith((
            "RPS", "RPL", "RPLP", "MRPS", "MRPL",   # ribosomal
            "IGH", "IGK", "IGL",                    # Ig chains
            "TRBV", "TRAV", "TRGV", "TRDV",         # TCR chains
            "HIST1", "HIST2", "HIST3", "HIST4",     # histones
            "OR", "TAS2R", "TAS1R",                 # chemosensory receptors
        )):
            exclude.add(sym)
        if "-" in sym:  # read-through / fusion names (APOC4-APOC2)
            exclude.add(sym)
    return exclude


def candidates_for(cancer_code, hpa_tissue, top_n=10):
    from pirlygenes.gene_sets_cancer import pan_cancer_expression

    df = pan_cancer_expression().set_index("Symbol")
    tumor_col = f"tcga_{cancer_code}"
    hpa_col = f"nTPM_{hpa_tissue}"
    if tumor_col not in df.columns:
        raise SystemExit(f"No TCGA column for {cancer_code}: {tumor_col!r} not found")
    if hpa_col not in df.columns:
        raise SystemExit(f"No HPA column for tissue {hpa_tissue!r}")

    immune_cols = [f"nTPM_{t}" for t in (
        "bone_marrow", "spleen", "lymph_node", "tonsil", "appendix",
    )]
    muscle_cols = [f"nTPM_{t}" for t in (
        "smooth_muscle", "skeletal_muscle", "heart_muscle",
    )]
    fat_col = "nTPM_adipose_tissue"

    sub = df[[tumor_col, hpa_col] + immune_cols + muscle_cols + [fat_col]].copy()
    sub = sub.dropna(subset=[tumor_col])
    sub["tumor_tpm"] = sub[tumor_col].astype(float)
    sub["mn_ntpm"] = sub[hpa_col].fillna(0).astype(float)
    sub["max_immune"] = sub[immune_cols].max(axis=1).fillna(0)
    sub["max_muscle"] = sub[muscle_cols].max(axis=1).fillna(0)
    sub["max_fat"] = sub[fat_col].fillna(0)
    sub["fold"] = sub["tumor_tpm"] / (sub["mn_ntpm"] + 1.0)

    max_other = sub[["max_immune", "max_muscle", "max_fat"]].max(axis=1)
    exclude = _build_exclusion_set(df)
    picks = sub[
        (sub["fold"] >= 10)
        & (sub["tumor_tpm"] >= 5)
        & (max_other < sub["tumor_tpm"])
        & (~sub.index.isin(exclude))
    ].sort_values("fold", ascending=False).head(top_n)

    id_map = pan_cancer_expression().set_index("Symbol")["Ensembl_Gene_ID"].to_dict()
    rows = []
    for sym, r in picks.iterrows():
        rows.append({
            "cancer_code": cancer_code,
            "matched_normal_tissue": hpa_tissue,
            "symbol": sym,
            "ensembl_gene_id": id_map.get(sym, ""),
            "fold_change_vs_matched_normal": round(r["fold"], 1),
            "tumor_tpm": round(r["tumor_tpm"], 2),
            "matched_normal_ntpm": round(r["mn_ntpm"], 2),
            "max_immune_ntpm": round(r["max_immune"], 2),
            "max_muscle_ntpm": round(r["max_muscle"], 2),
            "max_fat_ntpm": round(r["max_fat"], 2),
        })
    return rows


# Cancer codes → HPA tissue used to seed the v4.49.3 sweep.
V4_49_3_SWEEP = {
    "LUAD": "lung", "MESO": "lung",  # pleura proxy via lung
    "GBM": "cerebral_cortex", "LGG": "cerebral_cortex",
    "SKCM": "skin", "UVM": "retina",  # eye proxy via retina
    "UCEC": "endometrium", "UCS": "endometrium",
    "TGCT": "testis", "THYM": "thymus",
    "PCPG": "adrenal_gland", "ACC": "adrenal_gland",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cancer_code", nargs="?", help="TCGA cohort code (e.g. LUAD)")
    p.add_argument("hpa_tissue", nargs="?", help="HPA tissue column key")
    p.add_argument("--all-tcga-missing", action="store_true",
                   help="Run the v4.49.3 sweep: regenerate rows for all 12 codes")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    if args.all_tcga_missing:
        all_rows = []
        for code, tissue in V4_49_3_SWEEP.items():
            all_rows.extend(candidates_for(code, tissue, top_n=args.top))
        out = pd.DataFrame(all_rows)
    else:
        if not args.cancer_code or not args.hpa_tissue:
            p.error("cancer_code and hpa_tissue required (or use --all-tcga-missing)")
        rows = candidates_for(args.cancer_code, args.hpa_tissue, top_n=args.top)
        out = pd.DataFrame(rows)

    out.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
