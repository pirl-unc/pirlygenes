# Expression data audit — all cancer codes (2026-05-26)

Cross-reference of the cancer-type registry against actual bundled
expression data and the YAML source registry. Companion to
`docs/expression-data-refresh-plan.md` — the audit defines the work
queue for plan milestones 1–8.

## Universe + headline counts

- **125 codes** in `pirlygenes/data/cancer-type-registry.csv`
  (116 leaf + 9 umbrella/parent codes).
- **41 codes** have any row in `cancer-reference-expression.csv.gz` today
  (33% coverage).
- **1 code** (EWS) has the v5.3 extended stat suite populated
  (`TPM_std/min/max/p5/p10/p90/p95` + clean companions).
- **40 codes** have legacy 17-column rows only — need rebuild against
  their existing source to populate v5.3 columns.
- **84 codes** have *no* expression rows. Of those:
  - **32** are TCGA bulk-adult cohorts → cheap backfill from the
    already-cached Treehouse 25.01 PolyA file (9,806 TCGA samples
    bundled there).
  - **10** are TCGA-derived subtypes (BRCA PAM50, HNSC HPV-split,
    LUAD mutation-split) — need a classifier on top of TCGA data.
  - **37** are literature-curated entries with no expression source
    declared in the registry. Research per code.
  - **5** are umbrella/parent codes that aggregate over child subtypes
    and don't need their own rows.

## Status A — v5.3 extended stats populated (1 code)

| code | name | source | n_samples |
|---|---|---|---|
| EWS | Ewing Sarcoma | TREEHOUSE_POLYA_25_01 | 101 |

## Status B — bundled with legacy stats, needs v5.3 rebuild (40 codes)

Cheap. The builder pattern is wired to the shared
`pirlygenes.expression.stats.assign_stats` helper; running each
`build_*` script end-to-end will populate the 15 new columns. Codes
grouped by builder:

### `build_treehouse_reference_expression.py` (Treehouse POLYA, 14 codes)

Single 6.19 GB file is already on disk at
`~/.cache/pirlygenes/expression/treehouse-polya-25-01/`; re-invoke
with `--disease-label / --output-cancer-code` per row:

| code | Treehouse `disease` label | n |
|---|---|---|
| ATRT | atypical teratoid/rhabdoid tumor | 4 |
| HEPB | hepatoblastoma | 20 |
| MBL | medulloblastoma | 125 |
| NUTM | NUT midline carcinoma | 1 |
| OS | osteosarcoma | 262 |
| RMS_ARMS | alveolar rhabdomyosarcoma | 73 |
| RMS_ERMS | embryonal rhabdomyosarcoma | 95 |
| RMS_PRMS | pleomorphic rhabdomyosarcoma | 6 |
| RMS_SSRMS | spindle cell/sclerosing rhabdomyosarcoma | 8 |
| SARC_LMS | leiomyosarcoma | 151 |
| SARC_LPS_UNSPEC | liposarcoma | 92 |
| SARC_MYXFIB | myxofibrosarcoma | 41 |
| SARC_SYN | synovial sarcoma | 50 |
| SARC_UPS | undifferentiated pleomorphic sarcoma | 110 |

Effort: ~3-5 min compute per code (no download). ~1 hour total.

### `build_treehouse_reference_expression.py` (Treehouse RiboD, 2 codes)

Different compendium file (~1 GB), not yet downloaded:

| code | Treehouse `disease` label | n |
|---|---|---|
| CHOR | chordoma (RiboD release) | 3 |
| RB | retinoblastoma | 15 |

Effort: ~5 min download + ~5 min compute per code.

### Existing per-cohort GDC / GEO / Broad builders (24 codes)

| code | builder | source | n |
|---|---|---|---|
| BL | `build_bl_gdc_reference_expression.py` | CGCI-BLGSP via GDC | 184 |
| MM | `build_mmrf_reference_expression.py` | MMRF-COMMPASS via GDC | 764 |
| B_ALL | `build_target_all_reference_expression.py` | TARGET-ALL via GDC | 154 |
| T_ALL | `build_target_all_reference_expression.py` | TARGET-ALL via GDC | 264 |
| CTCL | `build_ctcl_scrna_reference_expression.py` | GSE171811 | 7 |
| CML | `build_geo_heme_reference_expression.py` | GSE100026 | 5 |
| MDS | `build_geo_heme_reference_expression.py` | GSE114922 | 82 |
| MCL | `build_geo_heme_reference_expression.py` | GSE271664 | 51 |
| MPN | `build_geo_heme_reference_expression.py` | GSE283710 | 45 |
| CLL | `build_cllmap_reference_expression.py` | CLLMAP 2022 | 708 |
| LAML_APL | `import_cancer_specific_expression.py` (BeatAML summary import) | BEATAML_OHSU_2022 | 18 |
| LAML_ELN_Adv | "" | "" | 175 |
| LAML_ELN_Fav | "" | "" | 140 |
| LAML_ELN_Int | "" | "" | 100 |
| NBL_MYCN_amp | `import_cancer_specific_expression.py` (TARGET-NBL summary) | TARGET_NBL_2018 | 29 |
| NBL_MYCN_nonamp | "" | "" | 113 |
| RT | `import_cancer_specific_expression.py` (TARGET-RT summary) | TARGET_RT_2017 | 43 |
| WILMS | `import_cancer_specific_expression.py` (TARGET-WT summary) | TARGET_WT_2015 | 130 |
| CHON | `import_cancer_specific_expression.py` (GSE299759 summary) | GSE299759 | 54 |
| PANNET | `import_cancer_specific_expression.py` (GSE118014 summary) | GSE118014 | 33 |
| SARC_DDLPS | `import_cancer_specific_expression.py` (GSE75885 summary) | GSE75885 | 19 |
| SARC_LGFMS | "" | "" | 2 |
| SARC_PLEOLPS | "" | "" | 4 |
| SCLC | `import_cancer_specific_expression.py` (UCologne summary) | SCLC_UCOLOGNE_2015 | 81 |

Note: 9 of these (BeatAML, TARGET-NBL/RT/WT, GSE299759, GSE118014,
GSE75885, SCLC UCologne) currently come in via summary-only imports —
their per-sample data exists in the source but the *importer* only
captured median/Q1/Q3. To get v5.3 extended stats for these, either:
- write a per-sample builder per source (closer to "proper"), or
- accept NaN for the new columns until then (current behavior).

The 15 with real per-sample builders (BL/MM/B_ALL/T_ALL/CTCL/CML/
MDS/MCL/MPN/CLL) just need the build script re-run.

## Status C — TCGA bulk-adult, no rows yet (32 codes)

**Critical finding: every one of these is already in the cached
Treehouse 25.01 PolyA file** (9,806 TCGA samples bundled). Backfill
is a parameterized run of the Treehouse builder — no new download
required.

Treehouse disease label → registry code mapping:

| code | Treehouse `disease` | TCGA n in Treehouse | declared YAML id |
|---|---|---|---|
| ACC | adrenocortical carcinoma | 77 | tcga-acc |
| BLCA | bladder urothelial carcinoma | 407 | tcga-blca |
| BRCA | breast invasive carcinoma | 1099 | tcga-brca |
| CESC | cervical & endocervical cancer | 306 | tcga-cesc |
| CHOL | cholangiocarcinoma | 36 | tcga-chol |
| COAD | colon adenocarcinoma | 290 | tcga-coad |
| DLBC | diffuse large B-cell lymphoma | 47 | tcga-dlbc |
| ESCA | esophageal carcinoma | 182 | tcga-esca |
| GBM | glioma (subset by GDC project) | ~150 of 689 | tcga-gbm |
| HNSC | head & neck squamous cell carcinoma | 520 | tcga-hnsc |
| KICH | kidney chromophobe | 66 | tcga-kich |
| KIRC | kidney clear cell carcinoma | 531 | tcga-kirc |
| KIRP | kidney papillary cell carcinoma | 289 | tcga-kirp |
| LAML | acute myeloid leukemia (TCGA subset) | 173 | tcga-laml |
| LGG | glioma (subset by GDC project) | ~530 of 689 | tcga-lgg |
| LIHC | hepatocellular carcinoma | 368 | tcga-lihc |
| LUAD | lung adenocarcinoma | 515 | tcga-luad |
| LUSC | lung squamous cell carcinoma | 498 | tcga-lusc |
| MESO | mesothelioma | 87 | tcga-meso |
| OV | ovarian serous cystadenocarcinoma | 426 | tcga-ov |
| PAAD | pancreatic adenocarcinoma | 179 | tcga-paad |
| PCPG | pheochromocytoma & paraganglioma | 182 | tcga-pcpg |
| PRAD | prostate adenocarcinoma | 496 | tcga-prad |
| READ | rectum adenocarcinoma | 93 | tcga-read |
| SARC | mixed sarcoma subtypes | sum of TCGA-SARC ≈ 250 | tcga-sarc |
| SARC_GIST | gastrointestinal stromal tumor | 19 | (add to YAML) |
| SARC_MYXLPS | liposarcoma (myxoid subset) | needs subtype split of 60 | (add to YAML) |
| SARC_WDLPS | liposarcoma (well-diff subset) | needs subtype split of 60 | (add to YAML) |
| SKCM | skin cutaneous melanoma | 469 | tcga-skcm |
| STAD | stomach adenocarcinoma | 414 | tcga-stad |
| TGCT | testicular germ cell tumor | 154 | tcga-tgct |
| THCA | thyroid carcinoma | 512 | tcga-thca |
| THYM | thymoma | 119 | tcga-thym |
| UCEC | uterine corpus endometrioid carcinoma | 181 | tcga-ucec |
| UCS | uterine carcinosarcoma | 57 | tcga-ucs |
| UVM | uveal melanoma | 79 | tcga-uvm |

Notes:
- **GBM vs LGG**: Treehouse uses one "glioma" bucket; split by GDC
  project (TCGA-GBM vs TCGA-LGG) using the `study_accession` /
  `th_dataset_id` prefix or the icd_disease field.
- **LAML**: Treehouse's 173 is the TCGA-LAML subset of its total
  617 AML samples (which also include BeatAML + Treehouse-pediatric).
  Filter by `th_dataset_id.startswith("TCGA")` AND
  `disease == "acute myeloid leukemia"`.
- **SARC subtypes (GIST/MYXLPS/WDLPS)**: Treehouse's liposarcoma label
  doesn't split by sub-histology; would need the TCGA-SARC
  histological annotation (available via TCGA-SARC biospecimen
  manifest) overlaid.

Effort estimate for the straightforward 30 (excluding GBM/LGG split
and SARC sub-histology): ~3-5 min compute per code, ~2 hours total.

## Status D — TCGA-derived subtypes, needs classifier (10 codes)

| code | parent | classifier needed | data path |
|---|---|---|---|
| BRCA_Basal | BRCA | PAM50 (centroid-based) | TCGA-BRCA TPM + PAM50 50-gene panel |
| BRCA_HER2 | BRCA | PAM50 | "" |
| BRCA_LumA | BRCA | PAM50 | "" |
| BRCA_LumB | BRCA | PAM50 | "" |
| BRCA_Normal | BRCA | PAM50 | "" |
| HNSC_HPV_pos | HNSC | HPV status | TCGA pan-cancer HPV paper (Cao 2016 PMID: 27568064) supplies per-sample HPV calls |
| HNSC_HPV_neg | HNSC | HPV status | "" |
| LUAD_EGFR | LUAD | EGFR mutation call | TCGA MAF (open) + EGFR hotspot panel |
| LUAD_KRAS | LUAD | KRAS mutation call | "" |
| LUAD_STK11 | LUAD | STK11 or KEAP1 mutation | "" |

Approach: build the parent (BRCA / HNSC / LUAD) from Treehouse-bundled
TCGA samples first; then add a downstream step that runs the
classifier and writes per-subtype rows with
`source_cohort=TCGA_BRCA_PAM50` etc. (already declared in the registry).

PAM50 reference centroids are public (Parker 2009, PMID: 19204204);
MAF for TCGA-LUAD is open via the GDC API.

## Status E — literature-curated, needs research per code (37 codes)

These have no expression source declared in the registry. Triaged by
the realistic effort to add per-sample data:

### Tractable: identified candidate public sources

| code | name | best public source | rough n |
|---|---|---|---|
| HL | Hodgkin Lymphoma | GSE12453 (Steidl 2010, classical HL microdissected HRS cells) | ~30 |
| FL | Follicular Lymphoma | GSE65135 (Dave 2004 reanalysis) or GSE16131 (Brodtkorb 2014) | ~50 |
| HCL | Hairy Cell Leukemia | GSE16729 (Tiacci 2011) | 20+ |
| NPC | Nasopharyngeal Carcinoma | GSE12452 (Sengupta 2006), GSE34573 (Bose 2013) | 30-50 |
| MEC | Merkel Cell Carcinoma | GSE161517 (Verhaegen 2022) | 30+ |
| MTC | Medullary Thyroid Carcinoma | GSE32662 (Pringle 2012) | 15-20 |
| LUNG_NET_LC | Lung Typical/Atypical Carcinoid | GSE118336 (Alcala 2019) — full LCNEC+carcinoid panel | 35+ |
| LUNG_NET_LCNEC | Large-Cell Neuroendocrine Carcinoma Lung | GSE118336 (Alcala 2019) | 75+ |
| MID_NET | Midgut / Small Bowel Carcinoid | GSE65286 (Andersson 2016) | 30+ |
| GCTB | Giant Cell Tumor of Bone | GSE150404 (Lutfi 2022) or treehouse partial | 10-15 |
| MBL_G3 | Medulloblastoma Group 3 | Treehouse MBL 125 + Cavalli 2017 subgroup labels (PMID: 28617753) | partial of 125 |
| MBL_G4 | Medulloblastoma Group 4 | "" | partial of 125 |
| MBL_SHH | Medulloblastoma SHH | "" | partial of 125 |
| MBL_WNT | Medulloblastoma WNT | "" | partial of 125 |
| SCLC_ASCL1 | SCLC ASCL1-dominant | SCLC UCologne 81 + Gay 2021 subtype labels (PMID: 33442380) | partial of 81 |
| SCLC_NEUROD1 | SCLC NEUROD1-dominant | "" | "" |
| SCLC_POU2F3 | SCLC POU2F3-dominant | "" | "" |
| SCLC_YAP1 | SCLC YAP1-dominant | "" (controversial — may be absent in modern reanalyses) | "" |
| SARC_ANGIO | Angiosarcoma | Treehouse "angiosarcoma" 20 | 20 |
| SARC_ASPS | Alveolar Soft Part Sarcoma | Treehouse "alveolar soft part sarcoma" 3 + GSE39058 | 3 + ~15 |
| SARC_CCS | Clear Cell Sarcoma | Treehouse + GSE160741 (Bartenstein 2021) | 5-15 |
| SARC_DSRCT | Desmoplastic Small Round Cell Tumor | Treehouse "desmoplastic small round cell tumor" 9 | 9 |
| SARC_EHE | Epithelioid Hemangioendothelioma | Treehouse 1 + GSE161365 | 1 + small |
| SARC_EPITH | Epithelioid Sarcoma | Treehouse "epithelioid sarcoma" 5 + GSE127165 | 5 + ~10 |
| SARC_IFS | Infantile Fibrosarcoma | Treehouse 2 + GSE39057 | 2 + ~10 |
| SARC_IMT | Inflammatory Myofibroblastic Tumor | Treehouse 4 + GSE186575 | 4 + small |
| SARC_MPNST | Malignant Peripheral Nerve Sheath Tumor | Treehouse "malignant peripheral nerve sheath tumor" 13 + GSE141438 | 13+ |
| SARC_GIST | GIST | Treehouse "gastrointestinal stromal tumor" 19 + TCGA-SARC subset | 20+ |

### Harder: low-n disease, limited public per-sample data

| code | name | notes |
|---|---|---|
| ACINIC | Acinic Cell Carcinoma | Mostly salivary; very small n in any single GEO study; consider TCGA-HNSC subtype split |
| ADCC | Adenoid Cystic Carcinoma | GSE59702 (Stenman 2014) ~20; multiple small GEO sets |
| ESS_HG | High-grade Endometrial Stromal Sarcoma | Rare; GSE36110 has a few; ICGC has some |
| ESS_LG | Low-grade Endometrial Stromal Sarcoma | Same — rare; aggregate small GEO + ICGC |
| PCN | Plasmacytoma | Solitary; closest is MM (CoMMpass) plus selected GEO; n likely <20 |
| SARC_DFSP | Dermatofibrosarcoma Protuberans | GSE51445 (~10); add to GEO heme-style builder |
| SARC_EMC | Extraskeletal Myxoid Chondrosarcoma | Very rare; GSE21050 has a few |
| SARC_KS | Kaposi Sarcoma | KS-associated tumor samples scattered across GSE16162 etc. |
| SARC_PEC | PEComa | Rare; Treehouse has 1; aggregate small case-series |
| SARC_SFT | Solitary Fibrous Tumor | GSE51425 (Vivero 2014) |

For these, the realistic plan is per-code mini-builders that pull a
single GEO accession and add it as a low-n cohort with explicit
`n_samples < 30` flagging in `notes`.

## Status F — umbrella codes (5 codes)

These aggregate over child subtypes; do not need standalone rows:

| code | name | children |
|---|---|---|
| BRCA | Breast Invasive Carcinoma | BRCA_Basal / HER2 / LumA / LumB / Normal |
| HNSC | Head & Neck Squamous Cell Carcinoma | HNSC_HPV_pos / HPV_neg |
| LUAD | Lung Adenocarcinoma | LUAD_EGFR / KRAS / STK11 |
| MBL | Medulloblastoma | MBL_G3 / G4 / SHH / WNT |
| SARC | Sarcoma | SARC_*  (many) |
| SCLC | Small Cell Lung Cancer | SCLC_ASCL1 / NEUROD1 / POU2F3 / YAP1 |
| NBL | Neuroblastoma | NBL_MYCN_amp / nonamp |

(Note: BRCA/HNSC/LUAD/SARC do also get bulk rows in Status B/C; the
umbrella designation here means "aggregate is also meaningful but the
clinically-actionable rows live on the subtypes.")

## Recommended next-action order

1. **Treehouse PolyA sweep (Status B, 14 codes)** — single cached
   file, parameterized re-runs. ~1 hour. Brings 14 cohorts to v5.3.
2. **TCGA backfill from Treehouse (Status C, 30 of 32 codes)** —
   same cached file. ~2 hours. Brings 30 net-new TCGA cohorts in.
3. **GBM/LGG split + SARC sub-histology (Status C, 2 + 3 codes)** —
   needs additional metadata. ~1-2 hours.
4. **Existing per-sample builders sweep (Status B, 10 codes via
   GDC/GEO/Broad)** — each builder re-run. ~3-4 hours total
   (downloads cached already in most cases).
5. **TCGA subtypes (Status D, 10 codes)** — PAM50 classifier + MAF
   joins. ~half a day each subtype family.
6. **Treehouse RiboD compendium download + 2 RiboD codes (Status B)**
   — ~30 min.
7. **GEO heme-style per-summary builders for the 9 currently-summary
   codes (Status B)** — write a per-summary fetcher; 1-2 hours each.
8. **Status E "tractable" codes** — one-by-one, depending on which
   clinical questions come first. ~1-2 hours per code.
9. **Status E "harder" codes** — aggregate small case-series; expect
   low n; flag explicitly in `notes`.

## Open questions

- For Status D, should mutation-subtype rows (LUAD_EGFR, etc.)
  carry the parent's full n_samples in the count column (with the
  subtype mask telling consumers how many were used for the stats),
  or the per-subtype n_samples only? Current convention is the
  latter; revisit if the parent comparison loses signal.
- For Status C SARC subtype splits (GIST / MYXLPS / WDLPS), is the
  TCGA-SARC biospecimen sub-histology mapping authoritative? Consider
  cross-referencing against the TCGA-SARC paper (Cancer Cell 2017,
  PMID: 29023870).
- Several Treehouse counts in Status C are lower than the original
  TCGA cohort size (e.g. LAML 173 vs TCGA-LAML's ~200). Treehouse
  applies its own sample-QC; if a "proper TCGA backfill" is needed,
  the GDC builder (plan milestone 3) is authoritative.
