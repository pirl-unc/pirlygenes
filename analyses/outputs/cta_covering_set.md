# Covering set of CTAs — actionable target for most cancers

For each cancer code: its most-gene-rich source; a CTA covers the code if clean-TPM > 30 at the given statistic. Greedy set cover. CTAs are subset antigens, so the **q3** view (target in ≥25% of patients) is the clinically relevant one — median understates them.

96 cancer codes with CTA data.

# Statistic: median (≥50% of patients)

## Covering set by cancer TYPE (each code = 1)

Coverable: 25/96 codes (26% of weight). Full cover needs 6 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | PRAME | 17% (16 codes) | ADCC, LUSC, NBL, NBL_MYCN_amp, NBL_MYCN_nonamp, OV… |
| 2 | XAGE1A | 21% (20 codes) | LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11 |
| 3 | DPPA5 | 23% (22 codes) | MTC, TGCT |
| 4 | MAGEA3 | 24% (23 codes) | LUNG_NET_LCNEC |
| 5 | PHF7 | 25% (24 codes) | LAML_APL |
| 6 | RGPD4 | 26% (25 codes) | SARC_PLEOLPS |

**Milestones:** target not reached

## Covering set by PATIENTS (≈US incidence)

Coverable: 25/96 codes (23% of weight). Full cover needs 6 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | PRAME | 14% (16 codes) | ADCC, LUSC, NBL, NBL_MYCN_amp, NBL_MYCN_nonamp, OV… |
| 2 | XAGE1A | 22% (20 codes) | LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11 |
| 3 | DPPA5 | 23% (22 codes) | MTC, TGCT |
| 4 | MAGEA3 | 23% (23 codes) | LUNG_NET_LCNEC |
| 5 | PHF7 | 23% (24 codes) | LAML_APL |
| 6 | RGPD4 | 23% (25 codes) | SARC_PLEOLPS |

**Milestones:** target not reached

**No actionable CTA (>30 TPM at this statistic):** ACC, BL, BLCA, BRCA, BRCA_Basal, BRCA_HER2, BRCA_LumA, BRCA_LumB, BRCA_Normal, B_ALL, CESC, CHOL, CLL, COAD, DLBC, ESCA, GBM, HEPB, HNSC, HNSC_HPVneg, HNSC_HPVpos, KICH, KIRC, KIRP, LAML, LAML_ELN_Adv, LAML_ELN_Fav, LAML_ELN_Int, LGG, LIHC, LUNG_NET_LC, MBL, MBL_G3, MBL_G4, MBL_SHH, MBL_WNT, MCL, MDS, MESO, MID_NET, MM, MPN, PAAD, PANNET, PCPG, PRAD, RB, READ, REC_NET, SARC_CHON, SARC_CHOR, SARC_DDLPS, SARC_EWS, SARC_GIST, SARC_KS, SARC_LMS, SARC_LPS_UNSPEC, SARC_MPNST, SARC_MYXFIB, SARC_OS, SARC_PEC, SARC_RMS_ARMS, SARC_UPS, SARC_WDLPS, SCLC, SCLC_ASCL1, STAD, THCA, THYM, T_ALL, UVM.

# Statistic: q3 (≥25% of patients)

## Covering set by cancer TYPE (each code = 1)

Coverable: 47/96 codes (49% of weight). Full cover needs 8 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | PRAME | 28% (27 codes) | ADCC, BRCA_Basal, LAML_APL, LUSC, MBL, MBL_G3… |
| 2 | XAGE1A | 35% (34 codes) | LIHC, LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11, SARC_EWS… |
| 3 | MAGEA3 | 41% (39 codes) | BLCA, ESCA, HNSC, HNSC_HPVneg, LUNG_NET_LCNEC |
| 4 | PHF7 | 45% (43 codes) | LAML, LAML_ELN_Fav, LAML_ELN_Int, MTC |
| 5 | ACTL8 | 46% (44 codes) | SARC_PLEOLPS |
| 6 | FATE1 | 47% (45 codes) | ACC |
| 7 | MAGEA9B | 48% (46 codes) | DLBC |
| 8 | PSG5 | 49% (47 codes) | KICH |

**Milestones:** target not reached

## Covering set by PATIENTS (≈US incidence)

Coverable: 47/96 codes (40% of weight). Full cover needs 8 CTAs.

| # | CTA | cum. coverage | newly covered codes |
| ---: | --- | ---: | --- |
| 1 | MAGEA6 | 18% (10 codes) | BLCA, ESCA, HNSC, HNSC_HPVneg, LUNG_NET_LCNEC, LUSC… |
| 2 | XAGE1A | 29% (19 codes) | LIHC, LUAD, LUAD_EGFR, LUAD_KRAS, LUAD_STK11, SARC_EWS… |
| 3 | PRAME | 36% (39 codes) | ADCC, BRCA_Basal, LAML_APL, MBL, MBL_G3, NBL… |
| 4 | PHF7 | 38% (43 codes) | LAML, LAML_ELN_Fav, LAML_ELN_Int, MTC |
| 5 | MAGEA9B | 40% (44 codes) | DLBC |
| 6 | PSG5 | 40% (45 codes) | KICH |
| 7 | ACTL8 | 40% (46 codes) | SARC_PLEOLPS |
| 8 | FATE1 | 40% (47 codes) | ACC |

**Milestones:** target not reached

**No actionable CTA (>30 TPM at this statistic):** BL, BRCA, BRCA_HER2, BRCA_LumA, BRCA_LumB, BRCA_Normal, B_ALL, CESC, CHOL, CLL, COAD, GBM, HEPB, HNSC_HPVpos, KIRC, KIRP, LAML_ELN_Adv, LGG, LUNG_NET_LC, MBL_G4, MBL_SHH, MBL_WNT, MCL, MDS, MESO, MID_NET, MM, MPN, PAAD, PANNET, PCPG, PRAD, RB, READ, REC_NET, SARC_CHON, SARC_CHOR, SARC_DDLPS, SARC_GIST, SARC_KS, SARC_LMS, SARC_LPS_UNSPEC, SARC_MYXFIB, SARC_PEC, SARC_UPS, SARC_WDLPS, STAD, THCA, T_ALL.
