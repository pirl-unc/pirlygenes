---
name: project_apd1_and_cohort_hunt
description: aPD1 ORR curation shipped (5.22.5); data-hunt results for NUT/TGCT/ATRT/CRC-MSI + CTA-therapy gaps
metadata:
  type: project
---

Shipped 5.22.5: `cancer-apd1-response.csv` (wheel-bundled like cancer-tmb.csv) — anti-PD-1
monotherapy ORR per cancer code (cancer_code, apd1_orr_pct, drug, trial, setting, pmid_doi,
confidence, notes), 23 cancers. Accessors `cancer_apd1_response_df` / `cancer_apd1_response`
(resolve_cancer_type + parent-inherit, mirroring cancer_tmb) in gene_sets_cancer.py. Enables
TMB-vs-aPD1 and CTA-burden-vs-aPD1 plotting axes. Also filled the analyses 9mer-payload-vs-TMB
plot variant matrix (base / _noHEPB / _noHEPB_subtypesonly) for both load metrics.

Data-hunt results (2026-06 web research, cited):
- **NUT carcinoma (NUTM)**: STILL no usable public primary-tumor bulk expression. Largest is
  Caris 54-tumor WTS but DUA-walled (not in GEO/SRA/dbGaP). EGA has 1-3 controlled-access BAMs
  (Seoul 2017 EGAD00001003117 = 2 primary + Ty-82). Everything in GEO is cell lines/mouse.
  Keep NUTM at 1 sample; `reference_nut_carcinoma_no_public_expression` confirmed.
- **TGCT extra cohorts**: GSE10615 (Affy U133A, n=27 ped GCT, seminoma+YST, public matrix —
  easiest onboard); TCGA-TGCT via recount3 (~150, full subtypes — but check Treehouse overlap);
  GSE1818 (all 5 histologies but 2-color log-ratio, signature-only).
- **ATRT extra cohorts**: GSE70678 (Affy U133Plus2, n=49, full TYR/SHH/MYC subgroup labels,
  public matrix — best onboard); OpenPedCan ATRT (RNA-seq TPM, subtypes, open — check Treehouse
  overlap).
- **CRC MSS/MSI split**: VERIFIED feasible via cBioPortal `coadread_tcga_pan_can_atlas_2018`,
  SAMPLE attribute `MSI_SENSOR_SCORE` (>=10 = MSI-H; 78 MSI-H / 506 MSS of 584), same per-case
  join as the HNSC-HPV / BRCA-PAM50 splits. Derive COAD vs READ from the TCGA project code (not
  CANCER_TYPE_DETAILED — mucinous is ambiguous). Build COAD_MSI/MSS + READ_MSI/MSS (READ_MSI
  ~6, maybe fold to CRC_MSI/MSS). Then add aPD1 rows COAD_MSI=45/COAD_MSS=0 etc.

CTA-directed therapy → cancer-type mapping (for curation-gap check): FDA-APPROVED — afami-cel
(MAGE-A4 TCR-T) synovial sarcoma [SARC_SYN]; tebentafusp (gp100 ImmTAC, a lineage antigen not a
classic CTA) uveal melanoma [UVM]. ADVANCED — lete-cel (NY-ESO-1) synovial + myxoid LPS
[SARC_MYXLPS]; IMA203/brenetafusp (PRAME) cutaneous [SKCM] + uveal melanoma; KK-LC-1 (CT83)
gastric/cervical/lung/breast (phase 1); MAGE-A3 = cautionary (fatal cross-reactivity, halted).
CTA panel curation lives in **tsarina** (don't edit pirlygenes cancer-testis-antigens.csv).

Next batch (proposed, one DATA_VERSION regen): CRC MSI split + TGCT GSE10615 + ATRT GSE70678,
then aPD1 subtype rows for the MSI/dMMR pairs.