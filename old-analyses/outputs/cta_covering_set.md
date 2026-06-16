# Covering set of CTAs — actionable target for most cancers

For each cancer code: its most-gene-rich source; a CTA covers the code if clean-TPM > 30 at the given statistic. Greedy set cover. CTAs are subset antigens, so the **q3** view (target in ≥25% of patients) is the clinically relevant one — median understates them.

96 cancer codes with CTA data.

# Statistic: median (≥50% of patients)

## Covering set by cancer TYPE (each code = 1)

Coverable: 23/96 codes (24% of weight). Full cover needs 6 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | PRAME | 16% (15 codes) | ADCC, NBL, NBL_MYCNamp, NBL_MYCNnonamp, OV, RT… |
| 2 | XAGE1A | 20% (19 codes) | LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11 |
| 3 | CT47A11 | 21% (20 codes) | MTC |
| 4 | DPPA5 | 22% (21 codes) | TGCT |
| 5 | MAGEA3 | 23% (22 codes) | NEC_LUNG_LARGECELL |
| 6 | RGPD4 | 24% (23 codes) | SARC_PLEOLPS |

**Milestones:** target not reached

## Covering set by PATIENTS (≈US incidence)

Coverable: 23/96 codes (20% of weight). Full cover needs 6 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | PRAME | 10% (15 codes) | ADCC, NBL, NBL_MYCNamp, NBL_MYCNnonamp, OV, RT… |
| 2 | XAGE1A | 19% (19 codes) | LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11 |
| 3 | DPPA5 | 19% (20 codes) | TGCT |
| 4 | CT47A11 | 20% (21 codes) | MTC |
| 5 | MAGEA3 | 20% (22 codes) | NEC_LUNG_LARGECELL |
| 6 | RGPD4 | 20% (23 codes) | SARC_PLEOLPS |

**Milestones:** target not reached

**No actionable CTA (>30 TPM at this statistic):** ACC, BL, BLCA, BRCA, BRCA_Basal, BRCA_HER2, BRCA_LumA, BRCA_LumB, BRCA_Normal, B_ALL, CESC, CHOL, CLL, COAD, DLBC, ESCA, GBM, HEPB, HNSC, HNSC_HPVneg, HNSC_HPVpos, KICH, KIRC, KIRP, LAML, LAML_APL, LAML_ELNadv, LAML_ELNfav, LAML_ELNint, LGG, LIHC, LUSC, MBL, MBL_G3, MBL_G4, MBL_SHH, MBL_WNT, MCL, MDS, MESO, MM, MPN, NET_LUNG, NET_MIDGUT, NET_PANCREAS, NET_RECTAL, PAAD, PCPG, PRAD, RB, READ, SARC_CHON, SARC_CHOR, SARC_DDLPS, SARC_EWS, SARC_GIST, SARC_KS, SARC_LMS, SARC_LPS_UNSPEC, SARC_MPNST, SARC_MYXFIB, SARC_OS, SARC_PEC, SARC_RMS_ARMS, SARC_UPS, SARC_WDLPS, SCLC, SCLC_ASCL1, STAD, THCA, THYM, T_ALL, UVM.

# Statistic: q3 (≥25% of patients)

## Covering set by cancer TYPE (each code = 1)

Coverable: 44/96 codes (46% of weight). Full cover needs 8 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | PRAME | 27% (26 codes) | ADCC, BRCA_Basal, LAML_APL, LUSC, MBL_G3, NBL… |
| 2 | XAGE1A | 33% (32 codes) | LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11, SARC_EWS, SARC_RMS_ARMS |
| 3 | MAGEA3 | 39% (37 codes) | BLCA, ESCA, HNSC, HNSC_HPVneg, NEC_LUNG_LARGECELL |
| 4 | EGFL6 | 41% (39 codes) | MTC, SARC_PLEOLPS |
| 5 | PHF7 | 43% (41 codes) | LAML, LAML_ELNfav |
| 6 | FATE1 | 44% (42 codes) | ACC |
| 7 | MAGEA9B | 45% (43 codes) | DLBC |
| 8 | PSG5 | 46% (44 codes) | KICH |

**Milestones:** target not reached

## Covering set by PATIENTS (≈US incidence)

Coverable: 44/96 codes (38% of weight). Full cover needs 8 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | MAGEA6 | 19% (10 codes) | BLCA, ESCA, HNSC, HNSC_HPVneg, LUSC, NEC_LUNG_LARGECELL… |
| 2 | XAGE1A | 28% (18 codes) | LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11, SARC_EWS, SARC_MYXLPS… |
| 3 | PRAME | 35% (37 codes) | ADCC, BRCA_Basal, LAML_APL, MBL_G3, NBL, NBL_MYCNamp… |
| 4 | MAGEA9B | 36% (38 codes) | DLBC |
| 5 | PHF7 | 37% (40 codes) | LAML, LAML_ELNfav |
| 6 | PSG5 | 38% (41 codes) | KICH |
| 7 | EGFL6 | 38% (43 codes) | MTC, SARC_PLEOLPS |
| 8 | FATE1 | 38% (44 codes) | ACC |

**Milestones:** target not reached

**No actionable CTA (>30 TPM at this statistic):** BL, BRCA, BRCA_HER2, BRCA_LumA, BRCA_LumB, BRCA_Normal, B_ALL, CESC, CHOL, CLL, COAD, GBM, HEPB, HNSC_HPVpos, KIRC, KIRP, LAML_ELNadv, LAML_ELNint, LGG, LIHC, MBL, MBL_G4, MBL_SHH, MBL_WNT, MCL, MDS, MESO, MM, MPN, NET_LUNG, NET_MIDGUT, NET_PANCREAS, NET_RECTAL, PAAD, PCPG, PRAD, RB, READ, SARC_CHON, SARC_CHOR, SARC_DDLPS, SARC_GIST, SARC_KS, SARC_LMS, SARC_LPS_UNSPEC, SARC_MYXFIB, SARC_PEC, SARC_UPS, SARC_WDLPS, STAD, THCA, T_ALL.
