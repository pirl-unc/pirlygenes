# CTA expression — top cohorts × top CTAs

Top 30 cancer cohorts (n≥10) × top 30 CTAs. Values are median TPM unless noted; see the per-stat CSVs for Q1/Q3. White cells in the heatmaps are genes **not measured** in that cohort (NaN), not zero.

## Standout CTAs (by peak median TPM across these cohorts)

| CTA | peak median TPM | peak cohort | cohorts measured |
| --- | ---: | --- | ---: |
| CTAG1A | 466.6 | SARC_MYXLPS | 27/30 |
| CTAG1B | 466.6 | SARC_MYXLPS | 28/30 |
| PRAME | 361.7 | NBL_MYCN_amp | 30/30 |
| DPPA5 | 341.1 | TGCT | 27/30 |
| FKBP6 | 229.4 | PANNET | 30/30 |
| XAGE1A | 222.5 | LUAD_KRAS | 26/30 |
| XAGE1B | 143.4 | LUAD_KRAS | 25/30 |
| RGPD4 | 107.1 | SARC_PLEOLPS | 29/30 |
| MAGEA3 | 95.6 | LUNG_NET_LCNEC | 29/30 |
| EGFL6 | 92.0 | SARC_MYXLPS | 30/30 |
| MAGEA6 | 77.0 | LUNG_NET_LCNEC | 30/30 |
| OOEP | 75.9 | REC_NET | 27/30 |
| CT47B1 | 67.9 | MTC | 27/30 |
| SPATA4 | 61.2 | MTC | 27/30 |
| TRIML2 | 60.3 | TGCT | 27/30 |

Notes:
- PRAME is the single highest-expressed CTA here (peak in SKCM), measured in every cohort — the clearest broad target signal.
- The MAGE-A cluster (MAGEA3/A6/A4/A12/…) is the next tier, strongest in melanoma / lung / HNSC / bladder.
- A column like H1-6 (a testis histone) ranks high by *max* but is NaN in most cohorts — read it as sparsely-measured, not broadly on.
