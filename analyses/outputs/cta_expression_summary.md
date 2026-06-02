# CTA expression — top cohorts × top CTAs

Top 30 cancer cohorts (n≥10) × top 30 CTAs. Values are median TPM unless noted; see the per-stat CSVs for Q1/Q3. White cells in the heatmaps are genes **not measured** in that cohort (NaN), not zero.

## Standout CTAs (by peak median TPM across these cohorts)

| CTA | peak median TPM | peak cohort | cohorts measured |
| --- | ---: | --- | ---: |
| PRAME | 350.1 | SKCM | 30/30 |
| XAGE1A | 183.9 | LUAD | 30/30 |
| XAGE1B | 133.4 | LUAD | 30/30 |
| MAGEA3 | 28.1 | SKCM | 30/30 |
| PHF7 | 26.9 | LAML_ELN_Fav | 30/30 |
| MAGEA6 | 18.4 | SKCM | 30/30 |
| EGFL6 | 15.2 | LUSC | 30/30 |
| MAGEA12 | 15.0 | SKCM | 30/30 |
| YBX2 | 13.3 | COAD | 30/30 |
| MAGEA4 | 10.0 | LUSC | 30/30 |
| H1-6 | 7.9 | T_ALL | 6/30 |
| MAGEA2 | 7.7 | SKCM | 30/30 |
| MAGEC1 | 6.9 | MM | 30/30 |
| SYCE2 | 6.6 | LAML_ELN_Fav | 30/30 |
| SYCP3 | 6.5 | CLL | 30/30 |

Notes:
- PRAME is the single highest-expressed CTA here (peak in SKCM), measured in every cohort — the clearest broad target signal.
- The MAGE-A cluster (MAGEA3/A6/A4/A12/…) is the next tier, strongest in melanoma / lung / HNSC / bladder.
- A column like H1-6 (a testis histone) ranks high by *max* but is NaN in most cohorts — read it as sparsely-measured, not broadly on.
