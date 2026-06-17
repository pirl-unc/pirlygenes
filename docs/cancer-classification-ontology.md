# Cancer-type classification ontology (expression markers)

A coherent ontology for telling cancer types apart from **bulk tumor RNA-seq**.
It unifies the curated lineage/compartment/discriminator panels under one model so
the marker *roles* and the *granularity* tiers stop being two ad-hoc lists and
become two orthogonal axes of one structure.

Grounded in: Hoadley et al., *Cell-of-Origin Patterns Dominate the Molecular
Classification of 10,000 Tumors from 33 Types of Cancer*, Cell 2018
(PMID 29625048 — pan-cancer expression clusters by cell-of-origin, with
cross-tissue groups: squamous, pan-GI, pan-kidney, neuroendocrine); the Human
Protein Atlas tissue map (PMID 25613900) and blood atlas (PMID 31857451); and the
per-type TCGA molecular-characterization papers cited on each panel.

## 1. Two orthogonal axes

Classification is not a flat label problem. Every marker sits at the intersection
of two independent axes:

**Axis A — GRANULARITY (the is-a taxonomy).** Four tiers, coarse → fine:

| tier | question | example nodes | where it lives today |
|---|---|---|---|
| **compartment** | what broad cell-of-origin class? | EPITHELIAL, HEMATOLYMPHOID, MELANOCYTIC, NEURAL_GLIAL, GERM_CELL, NEUROENDOCRINE, MESENCHYMAL | `cancer-compartment-panels.csv` |
| **supertype** | which lineage *program* within the compartment? | SQUAMOUS_PROGRAM, PAX8_LINEAGE, TTF1_LINEAGE, LUMINAL_GATA3, FOREGUT | `cancer-supertype-panels.csv` + `cancer-classification-ontology.csv` (node tree) |
| **family** | which organ lineage? | LUAD, BRCA, CRC, RENAL, THCA, HEME_BCELL, GLIAL | `cancer-family-panels.csv` |
| **subtype** | which molecular/morphologic child? | BRCA_Basal, COAD_MSI, LUAD_EGFR, SARC_RMS_ERMS | `cancer-lineage-panels.csv`, `cancer-subtype-groupings.csv` |

**Axis B — MARKER ROLE (relative to a node).** How a gene is *used* at a given node:

- `anchor` — positive and **specific to this node**: confirms AND discriminates it.
- `confirmatory` — positive but **specific to an ancestor**, not this node: routes
  you down the right branch, can't pick the leaf.
- `negative` — **positive in a different branch**: high here is surprising → rule-out.
- `discriminator` — separates **two siblings**: "if high → X not Y".

Plus a **SOURCE** flag (orthogonal to both): `tumor` | `immune` | `stroma` —
whether the transcript is the malignant clone, infiltrating TIL/TAM, or CAF stroma.

## 2. The unifying principle

> **A marker's role is relative to the level. A `confirmatory` (promiscuous)
> marker at tier N is an `anchor` at its parent tier (N−1).**

Promiscuity is not a defect — it is the *signal that the marker belongs to a
higher node*. This dissolves the "confirmatory-but-homeless" problem: every gene
is placed at the node where it is specific, and inherited downward.

- **PAX8** is `confirmatory` for THCA / RENAL / OV / UCEC (can't tell them apart)
  → it is the **`anchor` of the PAX8_LINEAGE supertype** (renal + Müllerian +
  thyroid developmental program). PMID 25613900 (HPA: PAX8 tissue set).
- **NKX2-1 / TTF-1** is `confirmatory` for LUAD and THCA → **`anchor` of TTF1_LINEAGE**
  (lung + thyroid). PMID 25079552 (TCGA-LUAD), 25417114 (TCGA-THCA).
- **GATA3 / FOXA1** are `confirmatory` for BRCA and BLCA → **`anchor` of LUMINAL_GATA3**.
  PMID 23000897 (TCGA-BRCA), 28988769 (TCGA-BLCA).
- **pan-keratins (KRT8/18/19), EPCAM** are `confirmatory` for every carcinoma family
  → **`anchor` of the EPITHELIAL compartment**. PMID 29625048.
- **CHGA / SYP / INSM1** are `confirmatory` for every NE family → **`anchor` of the
  NEUROENDOCRINE compartment**. PMID 26168399 (SCLC).
- **PTPRC (CD45)** is `confirmatory` across all heme families → **`anchor` of the
  HEMATOLYMPHOID compartment**. PMID 31857451.

## 3. The tree (compartment → supertype → family)

Markers shown at the node where they are an `anchor`; descendants inherit them as
`confirmatory`. Family `anchor`s in **bold**; the supertype `anchor` (the
promiscuous gene) leads each group.

```
EPITHELIAL            anchors: EPCAM, KRT8, KRT18, KRT19, CDH1
├── SQUAMOUS          anchor: TP63, KRT5, KRT6A, DSG3, SOX2
│     ├── HNSC, LUSC, CESC, ESCA_SQ   (organ-of-origin barely separable by RNA;
│     │                                use HPV surrogate CDKN2A for CESC/oroph.)
├── PAX8_LINEAGE      anchor: PAX8, PAX2
│     ├── RENAL       **CA9, NDUFA4L2** (clear-cell), **AMACR/MET** (papillary), **FOXI1/KIT** (chromophobe)
│     ├── THCA        **TG, TPO, SLC5A5, TSHR**        (also TTF1_LINEAGE — see §4)
│     ├── OV          **WT1, SOX17, FOLR1, WFDC2**     (serous)
│     └── UCEC        **HOXA10, HOXA11, VIM** + ESR1/PGR   (WT1-negative)
├── TTF1_LINEAGE      anchor: NKX2-1
│     ├── LUAD        **NAPSA, SFTPC, SFTPB, SLC34A2**
│     └── THCA        (shared node with PAX8_LINEAGE)
├── LUMINAL_GATA3     anchor: GATA3, FOXA1
│     ├── BRCA        **TRPS1, SCGB2A2, ESR1, PIP, SOX10(basal)**
│     ├── BLCA        **UPK1A/1B/2/3A (uroplakins), KRT20**   (UROTHELIAL)
│     └── PRAD        **NKX3-1, KLK3, KLK2, FOLH1, HOXB13** (FOXA1+, GATA3−)
├── FOREGUT           anchor: CLDN18, foregut program
│     ├── GASTRIC     **GKN1, GKN2, PGC, MUC6, MUC5AC**
│     └── PAAD        **GATA6, MSLN** (+ S100P, KRT17)
├── INTESTINAL        anchor: CDX2, CDH17
│     └── CRC         **SATB2, GUCY2C, KRT20**
└── HEPATOCYTE        anchor: ALB, HNF4A
      └── LIHC        **ARG1, GPC3, APOB, FGA, SERPINA1**

NEUROENDOCRINE   anchor: CHGA, CHGB, SYP, INSM1, NCAM1   (CROSS-CUTS organ — see §4)
   └── SCLC, NET_*, NEC_MERKEL (KRT20/ATOH1), MTC (calcitonin), NEPC
       resolved by an organ-anchor module + grade (MKI67)

HEMATOLYMPHOID   anchor: PTPRC
├── LYMPHOID
│     ├── HEME_BCELL  **PAX5, MS4A1, CD19, CD79A/B**
│     ├── HEME_TCELL  **CD3D, CD3E, CD3G**
│     └── (NK)        **NCAM1, KLRD1, NCR1**   [gap]
├── MYELOID    HEME_MYELOID  **MPO, ELANE** (CSF1R/LYZ/ITGAM = TAM/immune SOURCE)
└── PLASMA     HEME_PLASMA   **TNFRSF17, MZB1** ; defining NEGATIVES: PTPRC-low, MS4A1−, CD19−

MELANOCYTIC      anchor: MLANA, PMEL, TYR, DCT, TYRP1, MITF, SOX10
NEURAL_GLIAL     anchor: GFAP, OLIG2, S100B, SOX2
├── GLIAL       **OLIG2, ALDH1L1**            EPENDYMAL **FOXJ1**   CHOROID_PLEXUS **TTR, CLIC6, KCNJ13, OTX2**
└── NERVE_SHEATH **MPZ, PLP1, PMP22** (SOX10/S100B shared w/ melanoma)
GERM_CELL/PRIMITIVE  anchor: POU5F1, NANOG, SALL4, LIN28A
├── GERM_CELL   **SOX17(seminoma), SOX2(EC), AFP(yolk-sac), CGB(chorio)**
└── EMBRYONAL   **DLK1, IGF2** (cross-compartment oncofetal program)
MESENCHYMAL      anchor: VIM (diagnosis-of-EXCLUSION: CAF/stroma SOURCE; low all-other)
   └── sarcoma families = fusion/CNV-defined (SARC subtype key-genes)
```

## 4. Honest non-tree structure (a DAG, not a strict tree)

Real lineage biology has two unavoidable exceptions; the ontology models them
explicitly rather than forcing a tree:

1. **Multiple inheritance — thyroid.** THCA follicular cells express *both* the
   PAX8 and the TTF1/NKX2-1 program, so THCA is a child of **both** PAX8_LINEAGE
   and TTF1_LINEAGE. Its own anchors (TG/TPO/SLC5A5/TSHR) resolve it from the
   other members of either parent (lung via surfactants; renal via CA9; Müllerian
   via WT1).
2. **Cross-cutting program — neuroendocrine.** NE is a differentiation *program*
   that overrides organ lineage: a lung small-cell, a pancreatic NET, a Merkel and
   an MTC share CHGA/SYP/INSM1 and cluster together (Hoadley 2018), not with their
   organ's carcinomas. NE is therefore a **module** applied on top of the organ
   tree, resolved by an organ-anchor + grade, not a normal compartment child.

The developmental-TF modules (germ-cell pluripotency, embryonal oncofetal) behave
the same way — programs that recur across branches.

## 5. The classification algorithm (a tree/DAG walk)

```
1. COMPARTMENT  — which compartment anchor panel dominates the transcriptome?
                  (guard: HEMATOLYMPHOID / MESENCHYMAL signal in a solid tumor may
                   be SOURCE=immune/stroma infiltrate, not the tumor — require the
                   panel to dominate, not merely be present.)
   + apply the NEUROENDOCRINE module test (CHGA/SYP/INSM1 high → NE overrides organ).
2. SUPERTYPE    — within the compartment, which lineage-program anchor is on?
                  (squamous? PAX8+? TTF1+? GATA3+? foregut? intestinal? hepatocyte?)
3. FAMILY       — within the supertype, the family ANCHORS pick the lineage;
                  NEGATIVES prune sibling branches; DISCRIMINATORS resolve the hard
                  pairs (cancer-type-discriminators.csv).
4. SUBTYPE      — molecular (MSI/HPV/MYCN/PAM50) or morphologic child.
```

At every step: `anchor`s of the *current* node decide; `confirmatory` markers
(= anchors of an ancestor) only confirm the branch is right; `negative` markers
rule a branch out; `discriminator`s break sibling ties. Promotion of a marker up
a tier (PAX8: family-confirmatory → supertype-anchor) is the single mechanical
change that makes the whole thing consistent.

## 6. Do we need supertypes and subtypes?

**Supertypes — YES.** They are *where the promiscuous markers anchor*. Without
them, PAX8/NKX2-1/GATA3/keratins/CHGA are "confirmatory" at 2–4 families with no
home, the disambiguation logic (PAX8+ → then TG vs CA9 vs WT1) is implicit, and
the curation double-lists shared genes. The supertype tier is the missing scaffold
that turns "shared genes" from noise into structure. It already exists in skeletal
form as `family_group` (SQUAMOUS, CNS, HEMATOLYMPHOID, PRIMITIVE, NEUROENDOCRINE);
this ontology completes it with the marker-defined glandular supertypes
(PAX8_LINEAGE, TTF1_LINEAGE, LUMINAL_GATA3, FOREGUT, INTESTINAL, HEPATOCYTE).

**Subtypes — YES, and partially present.** The subtype tier already lives in
`cancer-lineage-panels.csv` (child discriminators) and `cancer-subtype-groupings.csv`
(MSI/MSS/HPV/MYCN/PAM50). The ontology formalizes it as the leaf tier with its own
anchors (e.g. BRCA_Basal: KRT5/KRT14/SOX10; LUAD_EGFR; COAD_MSI).
```
```
