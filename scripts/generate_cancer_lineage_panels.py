#!/usr/bin/env python
"""Generate ``data/cancer-lineage-panels.csv`` from a curated panel spec.

Per GitHub issue #266 (link). The existing
``cancer-family-panels.csv`` covers parent families (SQUAMOUS,
RENAL, etc.) and distinguishes a family from non-family codes —
but once the parent family wins, picking the right CHILD requires
lineage-discriminating markers (e.g. SCGB2A2 / mammaglobin
discriminates BRCA_BASAL from non-mammary squamous cancers, even
though both express the basal-keratin set).

This script:
  1. Encodes the curated panels as a declarative ``LINEAGE_PANELS`` dict.
  2. Resolves each symbol to its Ensembl gene ID via pyensembl.
  3. Writes ``pirlygenes/data/cancer-lineage-panels.csv`` with one
     row per (family, child_code, symbol).

Schema mirrors ``cancer-family-panels.csv`` plus a Child_Code and
Direction column:

    Family, Child_Code, Symbol, Ensembl_Gene_ID, Direction

``Direction`` is ``"high"`` (gene is elevated in the child relative
to its family siblings — the typical discriminator) or ``"low"``
(gene is depressed in the child — also useful, e.g. NKX2-1 low in
LUSC discriminates from LUAD).

Re-run after editing ``LINEAGE_PANELS``:

    python scripts/generate_cancer_lineage_panels.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pyensembl import EnsemblRelease


# Each panel: ``Family -> {child_code: [(symbol, direction), ...]}``.
# Curated from issue #266; expand here when new discrimination
# panels are validated against actual cohort expression data.
#
# Direction conventions:
#   "high"  — gene is elevated in this child relative to family siblings
#   "low"   — gene is depressed; used for negative-discrimination
#             (e.g. LUSC has LOW NKX2-1, distinguishing from LUAD)
LINEAGE_PANELS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "SQUAMOUS": {
        # SCGB2A2 / mammaglobin is the standout — near-zero in
        # non-mammary squamous cancers, detectable even in basal-like
        # BRCA. Combined with the GATA3/FOXA1 axis it cleanly
        # discriminates basal BRCA from the rest of the squamous
        # family.
        "BRCA_BASAL": [
            ("SCGB2A2", "high"), ("SCGB2A1", "high"), ("SCGB1D2", "high"),
            ("FOXA1", "high"), ("GATA3", "high"),
            ("MUCL1", "high"), ("MLPH", "high"), ("ANKRD30A", "high"),
        ],
        # Urothelial: uroplakins are urothelium-specific; GATA3+FOXA1
        # combo distinguishes from squamous epithelia.
        "BLCA_BASAL": [
            ("UPK1A", "high"), ("UPK1B", "high"), ("UPK2", "high"),
            ("UPK3A", "high"), ("UPK3B", "high"), ("S100P", "high"),
            ("GATA3", "high"), ("FOXA1", "high"),
        ],
        # Esophageal: anterior-foregut markers (AGR2/3, TFF1/3) +
        # esophageal-specific VSIG1, MAL, MUC22, EVPL.
        "ESCA": [
            ("AGR2", "high"), ("AGR3", "high"), ("TFF1", "high"),
            ("TFF3", "high"), ("VSIG1", "high"), ("MAL", "high"),
            ("MUC22", "high"), ("EVPL", "high"),
        ],
        # Head & neck (oral cavity / oropharynx): kallikreins KLK10/11,
        # MUC21, BPIFB1, SPRR2A/B late-cornification markers.
        "HNSC": [
            ("KLK10", "high"), ("KLK11", "high"), ("MUC21", "high"),
            ("BPIFB1", "high"), ("SPRR2A", "high"), ("SPRR2B", "high"),
        ],
        # Lung squamous: strong SOX2/TP63, low NKX2-1
        # (NKX2-1 high = LUAD; low = LUSC). SPRR2A is broadly squamous.
        "LUSC": [
            ("SOX2", "high"), ("TP63", "high"), ("SPRR2A", "high"),
            ("NKX2-1", "low"),
        ],
        # Cervical: HPV-associated; KRT4/13 are mucosal-squamous markers
        # not strongly expressed in skin/lung squamous. PAX8 + MMP10
        # round out the discriminators.
        "CESC": [
            ("KRT4", "high"), ("KRT13", "high"), ("PAX8", "high"),
            ("MMP10", "high"),
        ],
    },
    "GI_ADENO": {
        # Colorectal: CDX2 + CDH17 + GUCY2C + CEA family — classic
        # intestinal-lineage markers.
        "COAD": [
            ("CDX2", "high"), ("CDH17", "high"), ("GUCY2C", "high"),
            ("CEACAM5", "high"), ("CEACAM6", "high"), ("VIL1", "high"),
        ],
        "READ": [
            ("CDX2", "high"), ("CDH17", "high"), ("GUCY2C", "high"),
            ("CEACAM5", "high"), ("CEACAM6", "high"), ("VIL1", "high"),
        ],
        # Gastric: gastric-mucin (MUC5AC/MUC6) + trefoil-factor (TFF1/2)
        # + gastrokine (GKN1/2) + CLDN18 (gastric-tight-junction).
        "STAD": [
            ("MUC5AC", "high"), ("MUC6", "high"), ("CLDN18", "high"),
            ("TFF1", "high"), ("TFF2", "high"),
            ("GKN1", "high"), ("GKN2", "high"),
        ],
    },
    "HEPATOBILIARY": {
        # Hepatocellular: hepatocyte serum-protein synthesis program
        # (AFP, ALB, F2, HP, APOB, GC, AHSG) + HNF4A.
        "LIHC": [
            ("AFP", "high"), ("ALB", "high"), ("F2", "high"),
            ("HNF4A", "high"), ("HP", "high"),
            ("APOB", "high"), ("GC", "high"), ("AHSG", "high"),
        ],
        # Cholangiocarcinoma: biliary epithelial — KRT19 + MUC1/5AC +
        # EPCAM-high (vs LIHC which is EPCAM-low).
        "CHOL": [
            ("KRT19", "high"), ("MUC1", "high"),
            ("MUC5AC", "high"), ("EPCAM", "high"),
        ],
        # Pancreatic ductal: KRT19 + MUC1 + CLDN18 + AGR2 + S100P
        # (ductal program) + PRSS1 + PNLIP (pancreatic enzyme leak).
        "PAAD": [
            ("KRT19", "high"), ("MUC1", "high"), ("CLDN18", "high"),
            ("AGR2", "high"), ("S100P", "high"),
            ("PRSS1", "high"), ("PNLIP", "high"),
        ],
        # Pediatric hepatoblastoma: AFP extremely high + fetal markers
        # DLK1 and GPC3.
        "HEPB": [
            ("AFP", "high"), ("DLK1", "high"), ("GPC3", "high"),
        ],
    },
    "LUNG": {
        # Alveolar adenocarcinoma: surfactant-family + NKX2-1 (TTF-1)
        # + NAPSA + SLC34A2.
        "LUAD": [
            ("SFTPC", "high"), ("SFTPB", "high"), ("SFTPA1", "high"),
            ("NKX2-1", "high"), ("NAPSA", "high"), ("SLC34A2", "high"),
        ],
        # Mesothelial: WT1 + MSLN + CALB2 + KRT5/KRT6A + PDPN.
        "MESO": [
            ("WT1", "high"), ("MSLN", "high"), ("CALB2", "high"),
            ("KRT5", "high"), ("KRT6A", "high"), ("PDPN", "high"),
        ],
        # Lung neuroendocrine carcinoid: classic NE program. ASCL1 is
        # specifically high in classic SCLC + carcinoid.
        "LUNG_NET_LC": [
            ("CHGA", "high"), ("SYP", "high"), ("INSM1", "high"),
            ("ASCL1", "high"),
        ],
        "LUNG_NET_LCNEC": [
            ("CHGA", "high"), ("SYP", "high"), ("INSM1", "high"),
            ("ASCL1", "high"),
        ],
        "SCLC": [
            ("CHGA", "high"), ("SYP", "high"), ("INSM1", "high"),
            ("ASCL1", "high"),
        ],
    },
    "ENDOCRINE": {
        # Thyroid follicular: TG + TPO + TSHR. PAX8 also high in OV →
        # need TG as the unique marker.
        "THCA": [
            ("TG", "high"), ("TPO", "high"), ("TSHR", "high"),
            ("PAX8", "high"),
        ],
        # Medullary thyroid: calcitonin (CALCA) + CEACAM5 + NE markers.
        "MTC": [
            ("CALCA", "high"), ("CEACAM5", "high"),
            ("CHGA", "high"), ("ASCL1", "high"),
        ],
        # Adrenocortical: steroidogenesis program.
        "ACC": [
            ("CYP11A1", "high"), ("STAR", "high"), ("MC2R", "high"),
            ("NR5A1", "high"), ("INHA", "high"),
        ],
        # Pheochromocytoma / paraganglioma: catecholamine biosynthesis.
        "PCPG": [
            ("TH", "high"), ("DBH", "high"), ("PNMT", "high"),
            ("CHGA", "high"), ("SYP", "high"),
        ],
    },
    "GU_RENAL": {
        # Prostate: kallikrein family (KLK3 / PSA, KLK2) + TMPRSS2
        # + FOLH1 (PSMA) + NKX3-1 + HOXB13 + AR.
        "PRAD": [
            ("KLK3", "high"), ("KLK2", "high"), ("TMPRSS2", "high"),
            ("FOLH1", "high"), ("NKX3-1", "high"),
            ("HOXB13", "high"), ("AR", "high"),
        ],
        # Bladder luminal: uroplakins + KRT20 + GATA3 + FOXA1 high +
        # PPARG-driven luminal program (ELF3 as proxy).
        "BLCA_LUMINAL": [
            ("UPK1A", "high"), ("UPK1B", "high"), ("UPK2", "high"),
            ("UPK3A", "high"), ("KRT20", "high"),
            ("GATA3", "high"), ("FOXA1", "high"), ("PPARG", "high"),
        ],
        # Clear-cell renal: CA9 (VHL-loss surrogate) + NDUFA4L2 + AQP1.
        "KIRC": [
            ("CA9", "high"), ("NDUFA4L2", "high"), ("AQP1", "high"),
        ],
        # Papillary renal: MET pathway, AQP1, S100A4 lower.
        "KIRP": [
            ("MET", "high"), ("AQP1", "high"), ("S100A4", "low"),
        ],
        # Chromophobe: distal-tubule program.
        "KICH": [
            ("RHCG", "high"), ("KIT", "high"), ("TFCP2L1", "high"),
            ("FOXI1", "high"), ("PARM1", "high"),
        ],
    },
    "GYNECOLOGIC_GLANDULAR": {
        # Ovarian serous: PAX8 + WT1 + MSLN + MUC16/CA125 + FOLR1.
        # (PAX8 also in thyroid; combine with WT1 to discriminate from
        # endometrial, where WT1 is weak.)
        "OV": [
            ("PAX8", "high"), ("WT1", "high"), ("MSLN", "high"),
            ("MUC16", "high"), ("FOLR1", "high"),
        ],
        # Endometrial: PAX8 + ESR1 + PGR (different pattern from BRCA
        # luminal — endometrial ESR1/PGR coexist with PAX8) + MUC16.
        "UCEC": [
            ("PAX8", "high"), ("ESR1", "high"), ("PGR", "high"),
            ("MUC16", "high"),
        ],
    },
    "NET": {
        # Pancreatic NE: islet markers (insulin, glucagon, somatostatin)
        # + PDX1 + NEUROD1.
        "PANNET": [
            ("PDX1", "high"), ("NEUROD1", "high"),
            ("INS", "high"), ("GCG", "high"), ("SST", "high"),
        ],
        # Midgut NE: CDX2 (intestinal lineage) + TPH1 (serotonin
        # synthesis — classic carcinoid signature) + GCG.
        "MID_NET": [
            ("CDX2", "high"), ("TPH1", "high"), ("GCG", "high"),
        ],
        # SCLC also in LUNG above; duplicated here for NET classifier.
        "SCLC": [
            ("ASCL1", "high"), ("INSM1", "high"),
            ("CHGA", "high"), ("SYP", "high"),
        ],
        # Merkel cell: KRT20 + neurofilaments + SOX2.
        "MEC": [
            ("KRT20", "high"), ("NEFM", "high"), ("SOX2", "high"),
        ],
    },
    "BONE_EWS": {
        # Osteosarcoma: osteoblast lineage (RUNX2 / COL1A1 / ALPL),
        # bone-matrix sialoproteins (SPP1, IBSP), DLX5 + DMP1 + MEPE
        # round out the panel that trufflepig already uses.
        "SARC_OS": [
            ("RUNX2", "high"), ("COL1A1", "high"), ("ALPL", "high"),
            ("SPP1", "high"), ("IBSP", "high"),
            ("DLX5", "high"), ("DMP1", "high"), ("MEPE", "high"),
        ],
        # Ewing: EWSR1-FLI1 target genes (NKX2-2, CAV1) + CD99 (MIC2).
        "SARC_EWS": [
            ("NKX2-2", "high"), ("CD99", "high"), ("CAV1", "high"),
        ],
        # Chondrosarcoma: cartilage matrix (COL2A1, ACAN, COMP) + SOX9
        # + COL11A1.
        "SARC_CHON": [
            ("COL2A1", "high"), ("SOX9", "high"), ("ACAN", "high"),
            ("COMP", "high"), ("COL11A1", "high"),
        ],
    },
    "MESENCHYMAL": {
        # Rhabdomyosarcoma: skeletal-muscle differentiation TFs +
        # desmin. Trufflepig already curates these.
        "SARC_RMS_ERMS": [
            ("MYOD1", "high"), ("MYOG", "high"), ("DES", "high"),
            ("MYF5", "high"), ("MYF6", "high"),
        ],
        "SARC_RMS_ARMS": [
            ("MYOD1", "high"), ("MYOG", "high"), ("DES", "high"),
            ("MYF5", "high"), ("MYF6", "high"),
        ],
    },
}


# Canonical ENSG for symbols where pyensembl returns multiple gene
# ids (typically pseudoautosomal-region paralogs). These are well-
# established single biological targets.
CANONICAL_ENSG_OVERRIDES: dict[str, str] = {
    # CD99 / MIC2 — Ewing's classic CD99 marker. Ensembl has two
    # entries (ENSG00000002586 and ENSG00000275879) due to the
    # pseudoautosomal region; canonical CDS is the X-linked one.
    "CD99": "ENSG00000002586",
}


def _resolve_ensembl_id(symbol: str, genome: EnsemblRelease) -> str | None:
    """Look up the canonical ENSG for a HUGO symbol.

    Returns None when the symbol is ambiguous (multiple distinct gene
    ids and not in :data:`CANONICAL_ENSG_OVERRIDES`) or unknown —
    the caller prints a warning and skips.
    """
    if symbol in CANONICAL_ENSG_OVERRIDES:
        return CANONICAL_ENSG_OVERRIDES[symbol]
    try:
        genes = genome.genes_by_name(symbol)
    except Exception:
        return None
    ids = {g.gene_id.split(".", 1)[0] for g in genes}
    if len(ids) == 1:
        return next(iter(ids))
    return None


def build_rows(panels: dict, genome: EnsemblRelease) -> list[dict]:
    rows: list[dict] = []
    missing: list[tuple[str, str, str]] = []
    for family, by_child in panels.items():
        for child_code, items in by_child.items():
            for symbol, direction in items:
                ensembl_id = _resolve_ensembl_id(symbol, genome)
                if ensembl_id is None:
                    missing.append((family, child_code, symbol))
                    continue
                rows.append({
                    "Family": family,
                    "Child_Code": child_code,
                    "Symbol": symbol,
                    "Ensembl_Gene_ID": ensembl_id,
                    "Direction": direction,
                })
    if missing:
        print("\nWARNING: could not resolve these symbols (skipped):")
        for fam, child, sym in missing:
            print(f"  {fam} / {child} / {sym}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("pirlygenes/data/cancer-lineage-panels.csv"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    print(f"using Ensembl release {args.ensembl_release}")
    genome = EnsemblRelease(args.ensembl_release)
    rows = build_rows(LINEAGE_PANELS, genome)
    print(f"\nresolved {len(rows)} (family, child_code, symbol) rows "
          f"across {len(LINEAGE_PANELS)} families and "
          f"{sum(len(v) for v in LINEAGE_PANELS.values())} child codes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Family", "Child_Code", "Symbol",
                        "Ensembl_Gene_ID", "Direction"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
