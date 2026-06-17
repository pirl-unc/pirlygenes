"""Generate the cancer-classification ONTOLOGY: the node hierarchy that ties the
compartment / supertype / family / subtype tiers together, plus the SUPERTYPE
marker panels (where the promiscuous "confirmatory" markers become anchors).

See docs/cancer-classification-ontology.md for the reasoning. Two outputs:

  cancer-classification-ontology.csv  — one row per node:
      node, tier (compartment|supertype|family), parent (';'-separated for the
      DAG nodes), display_name, module ('' or 'cross_cutting'/'oncofetal').
  cancer-supertype-panels.csv  — anchor markers for each supertype (the
      promiscuous genes promoted to their proper level): Supertype, display_name,
      Symbol, Ensembl_Gene_ID, role (anchor), source, reference (PubMed-verified).

Core principle: a marker's role is LEVEL-RELATIVE — PAX8 is `confirmatory` for the
four PAX8+ families but the `anchor` of the PAX8_LINEAGE supertype. Re-run:

    python scripts/generate_cancer_ontology.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"

# ── Supertypes: marker-defined lineage programs that group >=2 families ───────
# {supertype: (display, parent_compartment, [anchor_symbols], verified_pmid)}.
SUPERTYPES = {
    "SQUAMOUS_PROGRAM": ("Squamous program", "EPITHELIAL",
                         ["TP63", "KRT5", "KRT6A", "DSG3", "SOX2"], "25631445"),
    "PAX8_LINEAGE": ("PAX8 lineage (renal/Müllerian/thyroid)", "EPITHELIAL",
                     ["PAX8", "PAX2"], "25613900"),
    "TTF1_LINEAGE": ("TTF-1 lineage (lung/thyroid)", "EPITHELIAL",
                     ["NKX2-1"], "25079552"),
    "LUMINAL_GATA3": ("GATA3/FOXA1 luminal (breast/urothelial/prostate)",
                      "EPITHELIAL", ["GATA3", "FOXA1"], "23000897"),
    "FOREGUT": ("Foregut glandular (gastric/pancreatic)", "EPITHELIAL",
                ["CLDN18"], "25079317"),
}

# ── Compartments (the coarsest tier) ──────────────────────────────────────────
COMPARTMENTS = {
    "EPITHELIAL": "Epithelial / carcinoma", "MESENCHYMAL": "Mesenchymal / sarcoma",
    "HEMATOLYMPHOID": "Hematolymphoid", "MELANOCYTIC": "Melanocytic / melanoma",
    "NEURAL_GLIAL": "Neural / glial (CNS)", "GERM_CELL": "Germ cell / primitive",
    "NEUROENDOCRINE": "Neuroendocrine",
}

# ── Family -> parent(s). DAG: ';'-separated (thyroid inherits two programs). ───
FAMILY_PARENT = {
    "SQUAMOUS": "SQUAMOUS_PROGRAM", "ESCA_SQ": "SQUAMOUS_PROGRAM",
    "RENAL": "PAX8_LINEAGE", "OV": "PAX8_LINEAGE", "UCEC": "PAX8_LINEAGE",
    "THCA": "PAX8_LINEAGE;TTF1_LINEAGE",            # multiple inheritance
    "LUAD": "TTF1_LINEAGE",
    "BRCA": "LUMINAL_GATA3", "BLCA": "LUMINAL_GATA3", "PROSTATE": "LUMINAL_GATA3",
    "GASTRIC": "FOREGUT", "PAAD": "FOREGUT",
    "CRC": "EPITHELIAL", "LIHC": "EPITHELIAL", "SELLAR_EPITHELIAL": "EPITHELIAL",
    "NEUROENDOCRINE": "NEUROENDOCRINE",
    "MELANOCYTIC": "MELANOCYTIC",
    "GLIAL": "NEURAL_GLIAL", "EPENDYMAL": "NEURAL_GLIAL",
    "CHOROID_PLEXUS": "NEURAL_GLIAL", "NERVE_SHEATH": "NEURAL_GLIAL",
    "MENINGIOMA": "NEURAL_GLIAL",
    "GERM_CELL": "GERM_CELL", "EMBRYONAL": "GERM_CELL", "CNS_EMBRYONAL": "GERM_CELL",
    "HEME_BCELL": "HEMATOLYMPHOID", "HEME_TCELL": "HEMATOLYMPHOID",
    "HEME_MYELOID": "HEMATOLYMPHOID", "HEME_PLASMA": "HEMATOLYMPHOID",
}
# nodes carrying a cross-cutting program flag (override organ / recur across branches)
MODULE = {"NEUROENDOCRINE": "cross_cutting", "EMBRYONAL": "oncofetal"}


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


def main() -> int:
    g = _genome()
    fam = pd.read_csv(DATA / "cancer-family-panels.csv")
    fam_display = dict(fam[["Family", "display_name"]].drop_duplicates().values)

    # 1) node hierarchy
    nodes = []
    for c, disp in COMPARTMENTS.items():
        nodes.append({"node": c, "tier": "compartment", "parent": "",
                      "display_name": disp, "module": MODULE.get(c, "")})
    for s, (disp, parent, _a, _p) in SUPERTYPES.items():
        nodes.append({"node": s, "tier": "supertype", "parent": parent,
                      "display_name": disp, "module": ""})
    for famname in sorted(set(fam["Family"])):
        parent = FAMILY_PARENT.get(famname, "EPITHELIAL")
        nodes.append({"node": famname, "tier": "family", "parent": parent,
                      "display_name": fam_display.get(famname, famname),
                      "module": MODULE.get(famname, "")})
    pd.DataFrame(nodes).to_csv(DATA / "cancer-classification-ontology.csv", index=False)

    # 2) supertype anchor panels
    rows = []
    for s, (disp, _parent, anchors, pmid) in SUPERTYPES.items():
        for sym in anchors:
            rows.append({"Supertype": s, "display_name": disp, "Symbol": sym,
                         "Ensembl_Gene_ID": _ensg(sym, g), "role": "anchor",
                         "source": "tumor", "reference": "PMID:" + pmid})
    pd.DataFrame(rows).to_csv(DATA / "cancer-supertype-panels.csv", index=False)

    nfam_grouped = sum(1 for f, p in FAMILY_PARENT.items() if p in SUPERTYPES)
    print(f"ontology: {len(nodes)} nodes "
          f"({len(COMPARTMENTS)} compartment / {len(SUPERTYPES)} supertype / "
          f"{len(set(fam['Family']))} family); {nfam_grouped} families under a supertype")
    print(f"supertype panels: {len(rows)} anchor markers / {len(SUPERTYPES)} supertypes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
