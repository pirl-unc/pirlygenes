"""Generate two multi-granularity cancer-classification marker resources.

The lineage marker panels span three granularities:
  - COMPARTMENT (coarsest, this file): cell-of-origin super-class — epithelial,
    mesenchymal, hematolymphoid, melanocytic, neural/glial, germ-cell,
    neuroendocrine. "What broad kind of tumor is this?"
  - FAMILY (cancer-family-panels.csv): organ lineage (LUAD, BRCA, RENAL, ...).
  - SUBTYPE (cancer-lineage-panels.csv): child discriminators within a family.

This script writes the coarse-tier ``cancer-compartment-panels.csv`` and a
``cancer-type-discriminators.csv`` of CONTRASTIVE marker sets — the markers that
separate two confusable cancer types (the diagnostic-pathology "differential"
question), each row stating which side it favors and the contrast's honest
separability. Markers are RNA-detectable (DNA/CNV-only events — RB1/CDKN2A loss,
1p19q codeletion, TP53/PTEN status — are deliberately excluded; they are not
expression markers).

Curated against TCGA marker papers, diagnostic-IHC differential references, and
Human Protein Atlas tissue-enrichment (PMIDs in ``source``). Re-run after editing
the COMPARTMENT_PANELS / DISCRIMINATORS specs:

    python scripts/generate_cancer_taxonomy_panels.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"

# ── Coarse tier: compartment (cell-of-origin super-class) panels ──────────────
# {compartment: (display_name, [symbols])}. Pan-compartment, RNA-expressed,
# tumor-intrinsic markers (MESENCHYMAL is a diagnosis-of-exclusion compartment —
# its markers overlap CAF/stroma; HEMATOLYMPHOID markers double as TIL infiltrate
# so call only when the panel dominates the transcriptome).
COMPARTMENT_PANELS: dict[str, tuple[str, list[str]]] = {
    "EPITHELIAL": ("Epithelial / carcinoma",
                   ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT7", "ELF3"]),
    "MESENCHYMAL": ("Mesenchymal / sarcoma",
                    ["VIM", "COL1A1", "COL1A2", "COL3A1", "PDGFRB"]),
    "HEMATOLYMPHOID": ("Hematolymphoid",
                       ["PTPRC", "CD3D", "CD3E", "MS4A1", "CD79A", "CD79B",
                        "MZB1", "LYZ"]),
    "MELANOCYTIC": ("Melanocytic / melanoma",
                    ["MLANA", "PMEL", "TYR", "DCT", "TYRP1", "MITF", "SOX10"]),
    "NEURAL_GLIAL": ("Neural / glial (CNS)",
                     ["GFAP", "OLIG2", "S100B", "SOX2", "OLIG1", "GLI3"]),
    "GERM_CELL": ("Germ cell",
                  ["POU5F1", "NANOG", "SALL4", "LIN28A", "KIT", "DPPA4"]),
    "NEUROENDOCRINE": ("Neuroendocrine",
                       ["CHGA", "CHGB", "SYP", "INSM1", "NCAM1", "ASCL1"]),
}

# ── Discriminators: contrastive sets separating confusable type pairs ─────────
# {contrast: dict(type_a, type_b, separability, favors={code: [(symbol, dir,
# tier)]}, source=pmid)}. ``dir`` high = enriched in the favored type, low =
# depleted (the negative call, e.g. WT1-low favours endometrioid-UCEC).
DISCRIMINATORS: dict[str, dict] = {
    # — thoracic —
    "LUAD_vs_LUSC": dict(type_a="LUAD", type_b="LUSC", separability="strong",
        source="22301491", favors={
            "LUAD": [("NKX2-1", "high", "primary"), ("NAPSA", "high", "primary"),
                     ("SFTPC", "high", "primary"), ("SFTPB", "high", "primary")],
            "LUSC": [("TP63", "high", "primary"), ("KRT5", "high", "primary"),
                     ("KRT6A", "high", "primary"), ("DSG3", "high", "supporting"),
                     ("SOX2", "high", "supporting")]}),
    "LUAD_vs_MESO": dict(type_a="LUAD", type_b="MESO", separability="good",
        source="29073361", favors={
            "LUAD": [("NKX2-1", "high", "primary"), ("NAPSA", "high", "primary"),
                     ("CEACAM5", "high", "supporting"), ("EPCAM", "high", "supporting"),
                     ("CLDN4", "high", "supporting")],
            "MESO": [("WT1", "high", "primary"), ("CALB2", "high", "primary"),
                     ("MSLN", "high", "supporting"), ("PDPN", "high", "supporting"),
                     ("KRT5", "high", "supporting")]}),
    "SCLC_vs_LUAD": dict(type_a="SCLC", type_b="LUAD", separability="good",
        source="28614209", favors={  # NKX2-1 excluded — positive in BOTH
            "SCLC": [("ASCL1", "high", "primary"), ("INSM1", "high", "primary"),
                     ("CHGA", "high", "supporting"), ("SYP", "high", "supporting"),
                     ("NCAM1", "high", "supporting"), ("NEUROD1", "high", "supporting")],
            "LUAD": [("NAPSA", "high", "primary"), ("SFTPC", "high", "primary"),
                     ("SFTPB", "high", "supporting"), ("CLDN4", "high", "supporting")]}),
    # — GI / hepatobiliary —
    "LIHC_vs_CHOL": dict(type_a="LIHC", type_b="CHOL", separability="good",
        source="25723115", favors={
            "LIHC": [("ALB", "high", "primary"), ("ARG1", "high", "primary"),
                     ("GPC3", "high", "primary"), ("HNF4A", "high", "supporting"),
                     ("CPS1", "high", "supporting"), ("FGA", "high", "supporting")],
            "CHOL": [("KRT7", "high", "primary"), ("KRT19", "high", "primary"),
                     ("MUC1", "high", "supporting"), ("EPCAM", "high", "supporting"),
                     ("SPP1", "high", "supporting"), ("S100P", "high", "supporting")]}),
    "CRC_vs_STAD": dict(type_a="CRC", type_b="STAD", separability="strong",
        source="21577201", favors={
            "CRC": [("SATB2", "high", "primary"), ("CDX2", "high", "primary"),
                    ("CDH17", "high", "supporting"), ("GUCY2C", "high", "supporting"),
                    ("KRT20", "high", "supporting")],
            "STAD": [("MUC5AC", "high", "primary"), ("TFF1", "high", "supporting"),
                     ("GKN1", "high", "primary"), ("GKN2", "high", "primary"),
                     ("PGC", "high", "supporting"), ("MUC6", "high", "supporting")]}),
    "PAAD_vs_STAD": dict(type_a="PAAD", type_b="STAD", separability="moderate",
        source="31658955", favors={
            "PAAD": [("GATA6", "high", "primary"), ("MSLN", "high", "supporting"),
                     ("CLDN18", "high", "supporting"), ("KRT17", "high", "supporting")],
            "STAD": [("GKN1", "high", "primary"), ("GKN2", "high", "primary"),
                     ("PGC", "high", "supporting"), ("MUC6", "high", "supporting"),
                     ("MUC5AC", "high", "supporting")]}),
    # — GU / gyn / endocrine —
    "BLCA_vs_PRAD": dict(type_a="BLCA", type_b="PRAD", separability="strong",
        source="20182344", favors={
            "BLCA": [("GATA3", "high", "primary"), ("UPK1B", "high", "primary"),
                     ("UPK2", "high", "primary"), ("UPK3A", "high", "primary"),
                     ("KRT20", "high", "supporting"), ("KRT5", "high", "supporting")],
            "PRAD": [("NKX3-1", "high", "primary"), ("KLK3", "high", "primary"),
                     ("KLK2", "high", "supporting"), ("FOLH1", "high", "supporting"),
                     ("AMACR", "high", "supporting"), ("HOXB13", "high", "supporting")]}),
    "BLCA_vs_BRCA": dict(type_a="BLCA", type_b="BRCA", separability="good",
        source="33756509", favors={  # GATA3 excluded — positive in BOTH
            "BLCA": [("UPK1B", "high", "primary"), ("UPK2", "high", "primary"),
                     ("UPK3A", "high", "primary"), ("KRT20", "high", "supporting")],
            "BRCA": [("TRPS1", "high", "primary"), ("SCGB2A2", "high", "primary"),
                     ("PIP", "high", "supporting"), ("ESR1", "high", "supporting"),
                     ("SOX10", "high", "supporting")]}),
    "OV_vs_UCEC": dict(type_a="OV", type_b="UCEC", separability="good",
        source="15084838", favors={
            "OV": [("WT1", "high", "primary"), ("SOX17", "high", "supporting"),
                   ("MUC16", "high", "supporting")],
            "UCEC": [("WT1", "low", "primary"), ("ESR1", "high", "supporting"),
                     ("PGR", "high", "supporting"), ("VIM", "high", "supporting"),
                     ("HOXA10", "high", "supporting"), ("HOXA11", "high", "supporting")]}),
    "KIRC_vs_KIRP": dict(type_a="KIRC", type_b="KIRP", separability="strong",
        source="23792563", favors={
            "KIRC": [("CA9", "high", "primary"), ("CA12", "high", "supporting"),
                     ("NDUFA4L2", "high", "primary"), ("VEGFA", "high", "supporting"),
                     ("MME", "high", "supporting")],
            "KIRP": [("AMACR", "high", "primary"), ("MET", "high", "primary"),
                     ("KRT7", "high", "supporting"), ("VIM", "high", "supporting")]}),
    "KIRC_vs_KICH": dict(type_a="KIRC", type_b="KICH", separability="strong",
        source="30872830", favors={
            "KIRC": [("CA9", "high", "primary"), ("NDUFA4L2", "high", "primary")],
            "KICH": [("FOXI1", "high", "primary"), ("KIT", "high", "primary"),
                     ("PVALB", "high", "supporting"), ("GATA3", "high", "supporting")]}),
    "THCA_vs_MTC": dict(type_a="THCA", type_b="MTC", separability="strong",
        source="25417114", favors={
            "THCA": [("TG", "high", "primary"), ("TPO", "high", "primary"),
                     ("SLC5A5", "high", "supporting"), ("TSHR", "high", "supporting")],
            "MTC": [("CALCA", "high", "primary"), ("CALCB", "high", "primary"),
                    ("CEACAM5", "high", "supporting"), ("ASCL1", "high", "supporting"),
                    ("CHGA", "high", "supporting"), ("TG", "low", "primary")]}),
    # — neuroendocrine / skin —
    "NEC_MERKEL_vs_SCLC": dict(type_a="NEC_MERKEL", type_b="SCLC",
        separability="good", source="9888708", favors={
            "NEC_MERKEL": [("KRT20", "high", "primary"), ("NEFL", "high", "supporting"),
                           ("NEFM", "high", "supporting"), ("ATOH1", "high", "primary")],
            "SCLC": [("NKX2-1", "high", "primary"), ("ASCL1", "high", "supporting"),
                     ("NEUROD1", "high", "supporting")]}),
}

# pyensembl returns >1 gene id for a few symbols (pseudoautosomal / patch);
# these are well-established single biological loci.
ENSG_OVERRIDES = {"PMEL": "ENSG00000185664"}


def _resolve(symbol: str, genome: EnsemblRelease) -> str | None:
    if symbol in ENSG_OVERRIDES:
        return ENSG_OVERRIDES[symbol]
    try:
        ids = {g.gene_id.split(".", 1)[0] for g in genome.genes_by_name(symbol)}
    except Exception:
        ids = set()
    return ids.pop() if len(ids) == 1 else None


def _newest_release() -> EnsemblRelease:
    for r in range(115, 90, -1):
        try:
            g = EnsemblRelease(r); g.gene_ids(); return g
        except Exception:
            continue
    raise SystemExit("no installed GRCh38 Ensembl release")


def main() -> int:
    g = _newest_release()
    missing: list[str] = []

    comp_rows = []
    for comp, (disp, syms) in COMPARTMENT_PANELS.items():
        for s in syms:
            ensg = _resolve(s, g)
            if ensg is None:
                missing.append(f"compartment {comp}:{s}"); continue
            comp_rows.append({"Compartment": comp, "display_name": disp,
                              "Symbol": s, "Ensembl_Gene_ID": ensg})
    pd.DataFrame(comp_rows).to_csv(DATA / "cancer-compartment-panels.csv", index=False)

    disc_rows = []
    for contrast, spec in DISCRIMINATORS.items():
        for code, markers in spec["favors"].items():
            for sym, direction, tier in markers:
                ensg = _resolve(sym, g)
                if ensg is None:
                    missing.append(f"discriminator {contrast}:{sym}"); continue
                disc_rows.append({
                    "contrast": contrast, "type_a": spec["type_a"],
                    "type_b": spec["type_b"], "favors": code, "Symbol": sym,
                    "Ensembl_Gene_ID": ensg, "direction": direction, "tier": tier,
                    "separability": spec["separability"], "source": "PMID:" + spec["source"]})
    pd.DataFrame(disc_rows).to_csv(DATA / "cancer-type-discriminators.csv", index=False)

    print(f"compartment panels: {len(comp_rows)} rows / {len(COMPARTMENT_PANELS)} compartments")
    print(f"discriminators: {len(disc_rows)} rows / {len(DISCRIMINATORS)} contrasts")
    if missing:
        print("UNRESOLVED (skipped):", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
