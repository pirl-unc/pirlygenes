"""Enrich the lineage / compartment marker panels with marker ROLE, SOURCE and a
literature REFERENCE — so each gene's distinct *use* in classification is explicit.

Grounded in a literature audit (parallel molecular-pathology reviews; PMIDs below).
Adds three columns to ``cancer-family-panels.csv`` and ``cancer-compartment-panels.csv``:

  role    — how the marker is used:
            ``anchor``       lineage-specific positive: confirms AND discriminates
                             this type (TG=thyroid, ALB=liver, KLK3=prostate, ...).
            ``confirmatory`` positive but PROMISCUOUS: confirms a broader class,
                             does NOT discriminate organ (PAX8 spans renal/Müllerian/
                             thyroid; NKX2-1 lung+thyroid; GATA3 breast+urothelial;
                             pan-keratins; CHGA/SYP; PTPRC). Good for "is it in this
                             class?", useless for "which member?".
            ``negative``     SURPRISING if high: argues AGAINST this type, a rule-out
                             (KRT7 high -> not colorectal; WT1 high -> not endometrioid).
  source  — where the transcript comes from in a bulk tumor sample:
            ``tumor``        the malignant cells.
            ``immune``       immune infiltrate (TIL/TAM) — in a non-hematolymphoid
                             tumor this signal is infiltrate, not the cancer cells.
            ``stroma``       cancer-associated fibroblast / ECM.
  reference — primary PMID supporting the panel / marker.

Pairwise "if high -> X not Y" sets live in cancer-type-discriminators.csv (a
fourth role). Re-run after editing the curated layer below:

    python scripts/enrich_cancer_marker_panels.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"

# ── PROMISCUOUS markers: confirm a class but can't discriminate organ ──────────
# (empirically: genes shared across >=2 family panels, plus audit-flagged SHARED).
CONFIRMATORY = {
    "PAX8", "PAX2", "NKX2-1", "GATA3", "FOXA1", "ESR1", "AR", "PGR",
    "KRT5", "KRT6A", "KRT7", "KRT8", "KRT14", "KRT17", "KRT18", "KRT19", "KRT20",
    "TP63", "SOX2", "SOX10", "S100B", "GFAP", "AQP1", "AQP4", "MITF",
    "CEACAM5", "CEACAM6", "CDX2", "MUC1", "MUC5AC", "MUC6", "TFF1", "TFF2", "TFF3",
    "VIL1", "HEPH", "EPHB2", "REG4", "SALL4", "LIN28A", "LIN28B", "KIT", "NCAM1",
    "FOLH1", "STEAP1", "STEAP2", "TMPRSS2", "AMACR", "IGF2BP1", "IGF2BP3", "TDGF1",
    "DPPA4", "XBP1", "PRDM1", "CD38", "SDC1", "SPI1", "ENO2", "SCG2", "SCG3",
    "CTNNB1", "S100P", "CLDN18", "KRT13", "DSG3", "PPL", "KRT6B",
}

# ── INFILTRATE markers: expression in a solid tumor reflects immune/stromal TME ─
IMMUNE = {"PTPRC", "CD3D", "CD3E", "CD3G", "CD2", "CD5", "CD7", "CD19", "MS4A1",
          "CD79A", "CD79B", "PAX5", "CSF1R", "LYZ", "ITGAM", "MPO", "ELANE",
          "MZB1", "TNFRSF17"}
STROMA = {"VIM", "COL1A1", "COL1A2", "COL3A1", "PDGFRB", "POSTN", "ACTA2",
          "TAGLN", "THBS4", "DCN", "LUM", "FAP"}
HEME_FAMILIES = {"HEME_BCELL", "HEME_TCELL", "HEME_MYELOID", "HEME_PLASMA"}

# Primary PMID per family. EVERY id here was verified against NCBI PubMed (the
# authoritative TCGA molecular-characterization paper for the type, or the Human
# Protein Atlas tissue map 25613900 / blood atlas 31857451 where no single-type
# TCGA paper applies). The first-draft per-marker PMIDs were agent-supplied and
# ~70% were hallucinated; they were discarded after a PubMed esummary check.
FAMILY_REF = {
    "PROSTATE": "26544944", "CRC": "22810696", "GASTRIC": "25079317",
    "ESCA_SQ": "25631445", "SQUAMOUS": "25631445", "RENAL": "23792563",
    "GLIAL": "26061751", "MELANOCYTIC": "26091043", "NEUROENDOCRINE": "26168399",
    "HEME_BCELL": "31857451", "HEME_TCELL": "31857451", "HEME_MYELOID": "31857451",
    "HEME_PLASMA": "31857451", "EMBRYONAL": "19211792", "GERM_CELL": "25613900",
    "CNS_EMBRYONAL": "22832581", "EPENDYMAL": "25613900",
    "SELLAR_EPITHELIAL": "25613900", "MENINGIOMA": "25613900",
    "NERVE_SHEATH": "25613900", "CHOROID_PLEXUS": "25613900",
    "LUAD": "25079552", "BRCA": "23000897", "PAAD": "26343385", "LIHC": "28622513",
    "OV": "21720365", "UCEC": "23636398", "BLCA": "28988769", "THCA": "25417114",
}

# High-confidence membership corrections from the audit (kept minimal/defensible).
ADD = {  # {family: [(symbol)]} — strong markers the panel lacked
    "CRC": ["SATB2", "KRT20"], "MELANOCYTIC": ["SOX10", "TYRP1"],
    "EPENDYMAL": ["FOXJ1"], "CHOROID_PLEXUS": ["OTX2"], "GERM_CELL": ["SOX17"],
    "NEUROENDOCRINE": ["NEUROD1"], "GASTRIC": ["PGC"],
}
DROP = {  # {family: [symbol]} — low-specificity / spatially-dependent / weak-RNA
    "CRC": ["SLC12A2", "HEPH"], "NEUROENDOCRINE": ["ENO2", "RESP18"],
    "EPENDYMAL": ["MUC1", "NHERF1"],
}

# Curated NEGATIVE / exclusion markers (HIGH -> argues against the type). Each is
# referenced to its family's verified PMID (FAMILY_REF) — the molecular
# characterization / HPA tissue-specificity that establishes the marker belongs
# to a DIFFERENT lineage. (NKX3-1 is prostate-restricted -> high in BLCA argues
# prostate; TG is thyroid-restricted -> high in RENAL argues thyroid; etc.)
NEGATIVES = {
    "CRC": ["KRT7", "NKX2-1", "PAX8"],
    "GASTRIC": ["SATB2", "NKX2-1"],
    "PROSTATE": ["GATA3", "KRT7", "CDX2"],
    "LIHC": ["KRT7", "KRT19", "EPCAM"],
    "THCA": ["CALCA", "CHGA"],
    "RENAL": ["TG", "NKX2-1", "WT1"],
    "SQUAMOUS": ["NAPSA", "NKX2-1", "CDX2"],
    "MELANOCYTIC": ["KRT8", "EPCAM", "PTPRC"],
    "GERM_CELL": ["KRT8", "GATA3"],
    "MENINGIOMA": ["SOX10", "GFAP"],
    "NERVE_SHEATH": ["MLANA", "EPCAM"],
    "GLIAL": ["TTR", "EPCAM", "SYP"],
    "BLCA": ["NKX3-1", "TTF1"],
    "OV": ["KRT20", "CDX2"],
    "HEME_PLASMA": ["MS4A1", "CD19"],
}

# Compartment source/role: compartment markers CONFIRM the broad class (by design),
# and two compartments are infiltrate/stroma rather than tumor.
COMPARTMENT_SOURCE = {"HEMATOLYMPHOID": "immune", "MESENCHYMAL": "stroma"}
COMPARTMENT_REF = {  # all PubMed-verified: TCGA pan-cancer cell-of-origin +
    "EPITHELIAL": "29625048", "MESENCHYMAL": "29625048",   # per-compartment landmark
    "HEMATOLYMPHOID": "31857451", "MELANOCYTIC": "26091043",
    "NEURAL_GLIAL": "18171944", "GERM_CELL": "25613900",
    "NEUROENDOCRINE": "26168399",
}


def _genome():
    for r in range(115, 90, -1):
        try:
            g = EnsemblRelease(r); g.gene_ids(); return g
        except Exception:
            continue
    raise SystemExit("no installed GRCh38 Ensembl release")


def _ensg(sym, g):
    ids = {x.gene_id.split(".", 1)[0] for x in g.genes_by_name(sym)}
    if len(ids) != 1:
        raise SystemExit(f"ambiguous/unknown ENSG for {sym}: {ids}")
    return ids.pop()


def _role(sym):
    return "confirmatory" if sym in CONFIRMATORY else "anchor"


def _source(sym, family):
    # In a HEMATOLYMPHOID neoplasm the immune-lineage genes ARE the tumor cells
    # (CD19/PAX5 in a B-cell lymphoma = the malignant clone), so they are
    # source=tumor here. The "this signal is infiltrate" caveat for immune-lineage
    # genes lives on the HEMATOLYMPHOID *compartment* panel (where, in a solid
    # tumor, the same genes = TILs) — see COMPARTMENT_SOURCE — and on immune-gene
    # NEGATIVE markers in non-heme families (PTPRC high in a melanoma = infiltrate,
    # argues against melanoma).
    if family in HEME_FAMILIES:
        return "tumor"
    if sym in IMMUNE:
        return "immune"
    if sym in STROMA:
        return "stroma"
    return "tumor"


def main() -> int:
    g = _genome()
    fam = pd.read_csv(DATA / "cancer-family-panels.csv")

    # idempotent: if the file was already enriched, strip the added negative rows
    # and the role/source/reference columns back to the base before re-processing,
    # so a re-run doesn't compound (overwrite negatives' roles + re-append them).
    if "role" in fam.columns:
        fam = (fam[fam["role"] != "negative"]
               .drop(columns=["role", "source", "reference"], errors="ignore"))

    # apply membership corrections
    fam = fam[~fam.apply(lambda r: r["Symbol"] in DROP.get(r["Family"], []), axis=1)]
    add_rows = []
    disp = dict(fam[["Family", "display_name"]].drop_duplicates().values)
    grp = dict(fam[["Family", "family_group"]].drop_duplicates().values)
    for family, syms in ADD.items():
        for s in syms:
            if not ((fam.Family == family) & (fam.Symbol == s)).any():
                add_rows.append({"Family": family, "family_group": grp[family],
                                 "display_name": disp[family], "Symbol": s,
                                 "Ensembl_Gene_ID": _ensg(s, g)})
    fam = pd.concat([fam, pd.DataFrame(add_rows)], ignore_index=True)

    # positive-marker annotations
    fam["role"] = fam["Symbol"].map(_role)
    fam["source"] = [_source(s, f) for s, f in zip(fam.Symbol, fam.Family)]
    fam["reference"] = "PMID:" + fam["Family"].map(FAMILY_REF).fillna("")

    # negative / exclusion markers as role=negative rows
    neg_rows = []
    for family, markers in NEGATIVES.items():
        for s in markers:
            neg_rows.append({"Family": family, "family_group": grp.get(family, family),
                             "display_name": disp.get(family, family), "Symbol": s,
                             "Ensembl_Gene_ID": _ensg(s, g), "role": "negative",
                             "source": _source(s, family),
                             "reference": "PMID:" + FAMILY_REF.get(family, "25613900")})
    fam = pd.concat([fam, pd.DataFrame(neg_rows)], ignore_index=True)
    fam.to_csv(DATA / "cancer-family-panels.csv", index=False)

    # compartment panels: role=confirmatory (confirm class), source, reference
    comp = pd.read_csv(DATA / "cancer-compartment-panels.csv")
    comp["role"] = "confirmatory"
    comp["source"] = comp["Compartment"].map(COMPARTMENT_SOURCE).fillna("tumor")
    comp["reference"] = "PMID:" + comp["Compartment"].map(COMPARTMENT_REF).fillna("")
    comp.to_csv(DATA / "cancer-compartment-panels.csv", index=False)

    npos = (fam.role != "negative").sum()
    print(f"family panels: {npos} positive + {len(neg_rows)} negative rows; "
          f"roles={dict(fam.role.value_counts())}")
    print(f"  membership: +{len(add_rows)} added, dropped per {DROP}")
    print(f"compartment panels: {len(comp)} rows annotated (role/source/reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
