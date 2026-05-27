# Expression-source manifest — per cancer code

One row per cancer code in `pirlygenes/data/cancer-type-registry.csv`
giving the best-identified public source of per-sample expression
data, the implementation status, and a pointer to the
fetcher/builder script.

This is the "find one or more sources for each one" deliverable for
the expression-data refresh project. See also:
- `docs/expression-data-audit.md` for the bucket-level summary.
- `docs/expression-data-refresh-plan.md` for the rolling milestone
  plan.

## Legend

- **status v5.3** = full per-sample stat suite populated.
- **status legacy** = summary-imported median/Q1/Q3 only; v5.3
  columns NaN until per-sample builder lands.
- **status missing** = no rows bundled yet.
- **status: low-n** = source exists but n<10; flagged in notes.

## Codes (sorted by status, then code)

### Status: v5.3 populated (74 codes after current session — list condensed)

Bundled via per-sample builders. Re-runnable; see the named script.

| code(s) | source | script |
|---|---|---|
| EWS, ATRT, HEPB, MBL, NUTM, OS, RMS_ARMS/ERMS/PRMS/SSRMS, SARC_LMS/LPS_UNSPEC/MYXFIB/SYN/UPS, SARC_GIST | Treehouse 25.01 PolyA | `scripts/sweep_treehouse_polya_cohorts.py`, `scripts/sweep_treehouse_sarc_subtypes.py` |
| CHOR, RB | Treehouse 25.01 RiboD | `scripts/sweep_treehouse_ribod_cohorts.py` |
| 31 TCGA: ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SARC_WDLPS, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM | Treehouse-bundled TCGA samples; for GBM/LGG split via GDC project lookup, SARC_WDLPS via GDC histology | `scripts/sweep_treehouse_tcga_cohorts.py`, `_glioma_split.py`, `_sarc_subtypes.py` |
| BRCA_Basal/HER2/LumA/LumB/Normal | TCGA-BRCA × cBioPortal PAM50 (Hoadley 2018) | `scripts/sweep_treehouse_tcga_brca_pam50.py` |
| HNSC_HPV_pos/HNSC_HPV_neg | TCGA-HNSC × cBioPortal HPV (Cao 2016) | `scripts/sweep_treehouse_tcga_hnsc_hpv.py` |
| LUAD_EGFR/KRAS/STK11 | TCGA-LUAD × cBioPortal MAF | `scripts/sweep_treehouse_tcga_luad_mutations.py` |
| BL | CGCI-BLGSP via GDC | `scripts/build_bl_gdc_reference_expression.py` |
| MM | MMRF CoMMpass via GDC | `scripts/build_mmrf_reference_expression.py` |
| B_ALL, T_ALL | TARGET-ALL (P1/P2/P3) via GDC | `scripts/build_target_all_reference_expression.py` |
| RT | TARGET-RT via GDC | `scripts/build_target_subprojects.py --only TARGET-RT` |
| CLL | CLL-map TPM matrix (Broad) | `scripts/build_cllmap_reference_expression.py` |
| CTCL | GSE171811 (ECCITE-seq pseudobulk) | `scripts/build_ctcl_scrna_reference_expression.py` |
| CML, MDS, MCL, MPN | GSE100026 / GSE114922 / GSE271664 / GSE283710 | `scripts/build_geo_heme_reference_expression.py` |

### Status: in progress (queued background runs)

| code(s) | source | script | ETA |
|---|---|---|---|
| NBL_MYCN_amp, NBL_MYCN_nonamp | TARGET-NBL via GDC + cBioPortal MYCN attribute | `scripts/build_target_subprojects.py --only TARGET-NBL` | retry in background after HTTP 500 |
| WILMS | TARGET-WT via GDC | `scripts/build_target_subprojects.py --only TARGET-WT` | queued after NBL |
| LAML_APL, LAML_ELN_Fav, LAML_ELN_Int, LAML_ELN_Adv | BEATAML1.0-COHORT via GDC + ELN2017-approximated by GDC primary_diagnosis | `scripts/build_beataml_reference_expression.py` | running |

### Status: legacy (medians only, per-sample builder needed)

These came in via `scripts/import_cancer_specific_expression.py`
from upstream summary tables. Per-sample data exists at the source
but needs a custom builder per accession.

| code | source | notes for builder |
|---|---|---|
| CHON | GEO **GSE299759** (Meijer 2026, chondrosarcoma) | Supplementary file is a per-sample TPM TSV. Extend `build_geo_heme_reference_expression.py` GEO_SOURCES with a new GeoSource. |
| PANNET | GEO **GSE118014** (Alvarez 2018, pancreatic NET) | Per-sample TPM in supplementary. Same pattern as CHON. |
| SARC_DDLPS, SARC_LGFMS, SARC_PLEOLPS | GEO **GSE75885** (Delespaul 2017, soft-tissue sarcoma) | Single GSE with multiple histologies. Need per-sample TPM + sample → histology metadata to route. n=19/2/4 respectively (low for LGFMS and PLEOLPS). |
| SCLC | **SCLC UCologne 2015** (George 2015, PMID 26168399) | Nature paper supplements include per-sample RNA-seq counts. Hybrid microarray + RNA-seq; need to use the RNA-seq subset. |

### Status: missing (no rows bundled), candidate source identified

#### TCGA-derived, classifier needed

| code | source | next step |
|---|---|---|
| SARC_MYXLPS (Myxoid Liposarcoma) | TCGA-SARC has no MYXLPS; **GSE128064** (Hofvander 2018) or aggregated case-series | Need a non-TCGA GEO builder; n likely <30. |

#### NET axis

| code | source | next step |
|---|---|---|
| LUNG_NET_LC (Lung Typical/Atypical Carcinoid) | **GSE118336** (Alcala 2019, ~35) | Per-sample TPM in supplementary; needs new GEO builder. |
| LUNG_NET_LCNEC (Large-Cell NE Carcinoma) | **GSE118336** (Alcala 2019, ~75) | Same accession as LC; split by histology field. |
| MID_NET (Midgut/Small Bowel Carcinoid) | **GSE65286** (Andersson 2016, ~30) | Per-sample expression; small-bowel NETs. |
| MTC (Medullary Thyroid Carcinoma) | **GSE32662** (Pringle 2012, ~15-20) | Microarray (Affymetrix); need to use RMA-normalized + Treehouse-comparable scaling caveat. |
| MEC (Merkel Cell Carcinoma) | **GSE161517** (Verhaegen 2022, ~30) | Per-sample RNA-seq. |
| NPC (Nasopharyngeal Carcinoma) | **GSE12452** (Sengupta 2006, ~30) and/or **GSE34573** (Bose 2013, ~30) | Both microarray; aggregate if room. |

#### Lymphoma / heme

| code | source | next step |
|---|---|---|
| HL (Hodgkin Lymphoma) | **GSE12453** (Steidl 2010, microdissected HRS cells, n~30) | Microarray; HRS cell-specific, not whole-tumor — flag in notes. |
| FL (Follicular Lymphoma) | **GSE65135** (Dave 2004 reanalysis) or **GSE16131** (Brodtkorb 2014) | ~50 samples; per-sample microarray or RNA-seq depending on accession. |
| HCL (Hairy Cell Leukemia) | **GSE16729** (Tiacci 2011) | ~20 samples; microarray. |

#### Sarcoma subtypes

| code | source | next step |
|---|---|---|
| SARC_ANGIO (Angiosarcoma) | Treehouse 25.01 PolyA `disease=angiosarcoma` (n=20) | Same builder pattern as SARC_GIST; one disease-filter sweep. |
| SARC_ASPS (Alveolar Soft Part Sarcoma) | Treehouse "alveolar soft part sarcoma" (n=3) + **GSE39058** (n=~15) | Combine for ~18 samples. |
| SARC_CCS (Clear Cell Sarcoma) | Treehouse n~1 + **GSE160741** (Bartenstein 2021, ~10) | Low n. |
| SARC_DFSP (Dermatofibrosarcoma Protuberans) | **GSE51445** (~10) | New GEO builder. |
| SARC_DSRCT (Desmoplastic Small Round Cell Tumor) | Treehouse "desmoplastic small round cell tumor" (n=9) | Treehouse sweep, similar to SARC_GIST. |
| SARC_EHE (Epithelioid Hemangioendothelioma) | Treehouse n~1 + **GSE161365** | Very low n. |
| SARC_EPITH (Epithelioid Sarcoma) | Treehouse "epithelioid sarcoma" (n=5) + **GSE127165** (~10) | Combine. |
| SARC_IFS (Infantile Fibrosarcoma) | Treehouse "infantile fibrosarcoma" (n=2) + **GSE39057** (~10) | Combine. |
| SARC_IMT (Inflammatory Myofibroblastic Tumor) | Treehouse "inflammatory myofibroblastic tumor" (n=4) + **GSE186575** | Low n. |
| SARC_MPNST (Malignant Peripheral Nerve Sheath Tumor) | Treehouse "malignant peripheral nerve sheath tumor" (n=13) + **GSE141438** | Treehouse-direct sweep (similar to SARC_GIST). |
| SARC_KS (Kaposi Sarcoma) | **GSE16162** + scattered KS-associated tumor samples | KS-AIDS related; flag context in notes. |
| SARC_PEC (PEComa) | Treehouse n~1 + small case-series; **GSE189557** | Very low n. |
| SARC_SFT (Solitary Fibrous Tumor) | **GSE51425** (Vivero 2014) | Microarray; flag tech. |
| SARC_EMC (Extraskeletal Myxoid Chondrosarcoma) | **GSE21050** (very rare) | Few samples available; low n. |

#### Other rare entities

| code | source | next step |
|---|---|---|
| ACINIC (Acinic Cell Carcinoma) | Mostly salivary; small n in **GSE40631** (~15) | Microarray subset of larger salivary gland panel. |
| ADCC (Adenoid Cystic Carcinoma) | **GSE59702** (Stenman 2014, ~20) + several small GEO sets | Aggregate. |
| ESS_HG (High-grade Endometrial Stromal Sarcoma) | **GSE36110** (~few) + ICGC | Rare; aggregate. |
| ESS_LG (Low-grade Endometrial Stromal Sarcoma) | Same as ESS_HG sources | Rare. |
| GCTB (Giant Cell Tumor of Bone) | **GSE150404** (Lutfi 2022, ~10-15) | New GEO builder. |
| PCN (Plasmacytoma) | Closest is MM (CoMMpass) plus selected GEO; n<20 likely | Document as low-n; consider deferring. |

#### Subtype splits requiring per-cohort metadata join

| code(s) | source | next step |
|---|---|---|
| MBL_G3, MBL_G4, MBL_SHH, MBL_WNT (Medulloblastoma subgroups) | Treehouse MBL n=125 + **Cavalli 2017** subgroup labels (PMID 28617753) | Cavalli supplements provide sample → subgroup mapping. Build classifier overlay analogous to `sweep_treehouse_tcga_brca_pam50.py`. |
| SCLC_ASCL1, SCLC_NEUROD1, SCLC_POU2F3, SCLC_YAP1 (SCLC molecular subtypes) | SCLC UCologne n=81 + **Gay 2021** (PMID 33442380) subtype labels | Gay supplements provide sample → subtype mapping. YAP1 controversial — may have ≤2 samples. |

### Status: umbrella codes (aggregate, don't need own rows)

| code | children |
|---|---|
| BRCA | BRCA_Basal/HER2/LumA/LumB/Normal |
| HNSC | HNSC_HPV_pos/HPV_neg |
| LUAD | LUAD_EGFR/KRAS/STK11 |
| MBL | MBL_G3/G4/SHH/WNT |
| NBL | NBL_MYCN_amp/nonamp |
| SARC | SARC_* (many) |
| SCLC | SCLC_ASCL1/NEUROD1/POU2F3/YAP1 |

## Per-source priority for follow-up sessions

1. **Treehouse-direct sweeps for additional disease labels** —
   1-2 hr each, no new download: SARC_ANGIO (n=20), SARC_DSRCT (n=9),
   SARC_MPNST (n=13). Reuse `sweep_treehouse_sarc_subtypes.py`
   pattern.
2. **MBL / SCLC subgroup classifiers** — overlay published subgroup
   labels on existing MBL n=125 / SCLC n=81 cohorts. Pattern matches
   `sweep_treehouse_tcga_brca_pam50.py`.
3. **GEO per-sample builders for legacy-import cohorts** —
   CHON / PANNET / SCLC / SARC_DDLPS / LGFMS / PLEOLPS. Each ~1-2 hr
   including format-specific extraction.
4. **Lit-curated tractable** — HL, FL, HCL, NPC, MEC, MTC,
   LUNG_NET_LC, LUNG_NET_LCNEC, MID_NET, GCTB. Each ~1-2 hr;
   small GSEs, mostly microarray; flag tech in notes.
5. **Sarcoma-rare** — SARC_DFSP / EMC / KS / PEC / SFT etc.
   Low n (<30); aggregate per-code.
6. **Truly rare** — ACINIC / ADCC / ESS_HG / ESS_LG / PCN. May
   yield n<10; document low-quality flag.

## Notes for builder authors

- All new builders should write per-source shards via
  `pirlygenes.expression.stats.upsert_to_shard` — never the legacy
  single-file path.
- Microarray-sourced cohorts (HL, FL, HCL, MTC, NPC, several
  sarcoma subtypes) need a note in the `notes` cell flagging the
  tech difference so consumers can hold "TPM" comparisons with care.
- Subtype overlays (BRCA PAM50, HNSC HPV, LUAD MUT, MBL subgroup,
  SCLC subtype) should use distinct `source_cohort` tags so each
  ends up in its own shard and doesn't push the parent shard past
  the GitHub 100 MB hard limit. Pattern:
  `TREEHOUSE_POLYA_25_01_TCGA_<PARENT>_<CLASSIFIER>`.
