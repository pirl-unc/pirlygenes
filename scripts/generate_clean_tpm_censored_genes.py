"""Generate the canonical clean-TPM censored-gene list.

One explicit, auditable list of the genes the clean-TPM transform censors —
materialized from :func:`pirlygenes.expression.qc.classify_gene_qc` (the
complete definition the bundled references were built with), so every consumer
(``clean_tpm_removal_mask``, ``technical_rna_mask``, the builders) reads the
*same* explicit list instead of re-running the per-gene regex, with no second,
drifting definition (the standalone gene-family CSVs were missing ~1779
ribosomal-protein genes the classifier catches).

Two guarantees baked in at generation:
  * **categorized** — each row is ``technical`` (mtDNA / rRNA-like / mt-like
    pseudogene / polyA-bias lncRNA) or ``ribosomal_protein`` (RP mRNA +
    pseudogenes), so the runtime ``exclude_ribosomal_proteins`` boolean is a
    simple category filter.
  * **CTA-safe** — curated cancer targets (the CTA / surfaceome / key-gene /
    lineage / fusion panels via ``_default_protected_symbols``) are excluded, so
    a CTA ribosomal-protein paralog (RPL10L) or histone CTA (H1-6) is never
    censored — matching how the references were built (runtime protect-subtract).

Universe = the bundled ``cancer-reference-expression`` gene set (the union
across every packaged cohort/source/release — comprehensive for our data), so
the list restricted to any reference gene_table reproduces the old QC mask
exactly (membership-preserving → no reference regeneration).

The checked-in censored list is also used as a stability seed: re-running the
generator should not shrink the censor contract just because the packaged
reference-expression universe changes. The explicit rRNA/rRNA-pseudogene family
is unioned in as well. A few rRNA features are valid Ensembl genes but absent
from the current reference expression universe, and the clean-TPM censor list
should still capture them when a downstream source exposes those rows.

Run:  python scripts/generate_clean_tpm_censored_genes.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pirlygenes.expression.normalize import (
    _DEFAULT_NORMALIZE_REMOVE_GROUPS,
    _RIBOSOMAL_PROTEIN_GROUPS,
    _default_protected_symbols,
)
from pirlygenes.expression.qc import classify_gene_qc
from pirlygenes.load_dataset import get_data

OUT = Path(__file__).resolve().parent.parent / "pirlygenes" / "data" \
    / "clean-tpm-censored-genes.csv"

_TECHNICAL = {str(g) for g in _DEFAULT_NORMALIZE_REMOVE_GROUPS}
_RIBOSOMAL = {str(g) for g in _RIBOSOMAL_PROTEIN_GROUPS}


def build() -> None:
    ref = get_data("cancer-reference-expression", copy=False)
    gt = (ref[["Symbol", "Ensembl_Gene_ID"]]
          .dropna(subset=["Ensembl_Gene_ID"])
          .drop_duplicates("Ensembl_Gene_ID")
          .reset_index(drop=True))
    existing = get_data("clean-tpm-censored-genes", copy=False)[
        ["Symbol", "Ensembl_Gene_ID"]
    ]
    rrna = get_data("rrna-and-pseudogenes", copy=False)[["Symbol", "Ensembl_Gene_ID"]]
    gt = (pd.concat([existing, gt, rrna], ignore_index=True)
          .dropna(subset=["Ensembl_Gene_ID"])
          .drop_duplicates("Ensembl_Gene_ID")
          .reset_index(drop=True))
    protected = _default_protected_symbols()
    rows = []
    for sym, ensg in zip(gt["Symbol"].astype(str), gt["Ensembl_Gene_ID"].astype(str)):
        bare = ensg.split(".")[0]
        if sym in protected:
            continue  # CTA / curated target — never censored
        qc = classify_gene_qc(sym, ensembl_id=bare)
        if qc.group in _TECHNICAL:
            category = "technical"
        elif qc.group in _RIBOSOMAL:
            category = "ribosomal_protein"
        else:
            continue
        rows.append({"Ensembl_Gene_ID": bare, "Symbol": sym, "category": category})
    df = (pd.DataFrame(rows, columns=["Ensembl_Gene_ID", "Symbol", "category"])
          .sort_values(["category", "Symbol"])
          .reset_index(drop=True))
    df.to_csv(OUT, index=False)
    n_tech = int((df["category"] == "technical").sum())
    n_ribo = int((df["category"] == "ribosomal_protein").sum())
    print(f"wrote {len(df)} censored genes ({n_tech} technical + {n_ribo} "
          f"ribosomal_protein) -> {OUT}", flush=True)


if __name__ == "__main__":
    build()
