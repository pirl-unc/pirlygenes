# anti-PD-1 ORR × TMB × expression — population-match audit

The causal-factor model (`apd1_causal_factors.py`) joins three numbers per
cohort that come from **three different studies on three different patient
populations**:

| factor | source population |
|---|---|
| **aPD1 ORR** | a specific trial — a *line of therapy*, often *biomarker-selected* (PD-L1/CPS), in *metastatic/pretreated* disease |
| **TMB** | a WES/panel cohort (Lawrence 2013, Chalmers 2017, TCGA) — usually *primary, treatment-naive, all-comer* |
| **expression** | TCGA / Treehouse reference — *primary, treatment-naive, all-comer* |

The model is only valid where these three describe the *same* population. This
audit flags every mismatch. Verdict key: ✓ matched · ⚠ partial · ✗ mismatch.

## Per-cohort verdicts

| cohort | ORR | ORR population (PMID) | TMB (code used) | TMB pop | verdict | issue |
|---|---|---|---|---|---|---|
| SKCM | 42 | nivo mono 1L all-comer (28889792) | 12.9 (SKCM) | all-comer WES | ✓ | gold standard |
| BLCA | 29 | 1L cis-ineligible all-comer (32552471) | 7.1 (BLCA) | all-comer | ✓ | |
| KIRC | 25 | pretreated all-comer ccRCC (26406148) | 1.9 (KIRC) | ccRCC all-comer | ✓ | line diff only |
| KIRP | 25 | papillary 1L (34272311) | 1.3 (KIRP) | papillary | ✓ | |
| KICH | 10 | chromophobe 1L (34272311) | 0.7 (KICH) | chromophobe | ✓ | small n |
| OV | 8 | recurrent all-comer (31218020) | 1.7 (OV) | all-comer | ✓ | |
| GBM | 8 | recurrent all-comer (32437507) | 2.2 (GBM) | all-comer | ✓ | |
| PRAD | 4 | pretreated mCRPC (31774688) | 0.7 (PRAD) | all-comer | ✓ | |
| UVM | 4 | metastatic uveal pooled (27533448) | 0.34 (UVM) | uveal | ✓ | low-conf ORR |
| **LUAD** | **45** | **1L PD-L1 TPS≥50% (27718847)** | 8.1 (LUAD) | all-comer | **✗** | **ORR is a PD-L1-high slice; all-comer mono ≈19% (CheckMate-057)** |
| **LUSC** | **45** | **1L PD-L1 TPS≥50% (27718847)** | 9.9 (LUSC) | all-comer | **✗** | **all-comer mono ≈20% (CheckMate-017)** |
| **COAD_MSI** | **45** | 1L MSI-H/dMMR (33264544) | **3.1 (COAD bulk)** | MSS bulk | **✗** | **TMB is MSS fallback; MSI-H ≈46/Mb** |
| **READ_MSI** | **45** | 1L MSI-H/dMMR (33264544) | **3.1 (READ bulk)** | MSS bulk | **✗** | **same MSI TMB fallback** |
| CESC | 15 | pretreated PD-L1+ CPS≥1 (30943124) | 5.0 (CESC) | all-comer | ⚠ | PD-L1-selected; all-comer ~12% |
| NPC | 25 | pretreated PD-L1+ (28837405) | 0.95 (NPC) | all-comer | ⚠ | PD-L1-selected |
| HNSC | 19 | 1L R/M CPS≥1 mono (33052747) | 3.9 (HNSC) | all-comer | ⚠ | CPS≥1; total pop ~17% |
| LUAD_STK11 | 10 | STK11/KEAP1-mut subset (30955977) | 8.1 (LUAD bulk) | all-comer | ⚠ | TMB+expr are bulk LUAD, not the STK11-mut subset (which is higher-TMB but cold) |
| UCEC_CNL | 6 | pMMR/MSS NSMP (34990208) | 2.5 (UCEC_CNL) | TCGA EC | ⚠ | KEYNOTE-158 non-MSI-H ~7% **not** split CNL vs CNH — both assigned ~6 (interpolation) |
| UCEC_CNH | 6 | pMMR/MSS p53-abn (34990208) | 2.8 (UCEC_CNH) | TCGA EC | ⚠ | same interpolation |
| ESCA | 18 | pretreated ESCC (31582355) | 4.0 (ESCA) | mixed adeno+SCC | ⚠ | ORR is SCC-only; TMB/expr mixed histology |
| STAD | 12 | 3L all-comer (29260193) | 5.0 (STAD) | all-comer | ⚠ | EBV/MSI hot subsets not split (viral=subset flag only) |
| LIHC | 17 | mixed sorafenib exposure (28434648) | 4.0 (LIHC) | all-comer | ⚠ | HBV/HCV subset etiology |
| SCLC | 13 | pretreated all-comer (27269732) | 8.6 (SCLC) | all-comer | ✓ | line diff |
| HL | 71 | r/r post-ASCT (29584546) | 2.0 (HL) | all-comer | ⚠ | response is 9p24/PD-L1-amp-driven, *not* TMB — low TMB but very high ORR by design |
| BRCA_Basal | 5 | pretreated TNBC all-comer (30475950) | 1.8 (BRCA_Basal) | TCGA basal | ✓ | TNBC≈basal |
| CHOL | 6 | advanced biliary (32319072) | 2.0 (CHOL) | all-comer | ✓ | |
| MESO | 10 | pretreated all-comer (33125908) | 1.8 (MESO) | all-comer | ✓ | |
| COAD/READ (bulk) | 5 | all-comer MSI-dependent (33264544) | 3.1 | MSS bulk | ✓ | bulk≈MSS-dominated |
| COAD_MSS/READ_MSS | 0 | MSS (26028255) | 3.1 | MSS | ✓ | |
| PAAD | 1 | pooled basket MSS (no PMID) | 2.1 (PAAD) | all-comer | ⚠ | no single trial; low-conf |

## Systematic findings

1. **Two true mismatches that distort the model (FIXED):**
   - **LUAD/LUSC ORR=45** was PD-L1 TPS≥50%-selected (KEYNOTE-024). Replaced
     with all-comer monotherapy ORR — **LUAD→19** (CheckMate-057, Borghaei
     2015, PMID 26412456), **LUSC→20** (CheckMate-017, Brahmer 2015, PMID
     26028407) — to match the all-comer TMB and expression. (Old PD-L1-high
     figure kept in `notes`.)
   - **COAD_MSI/READ_MSI TMB=3.1** was the bulk-COAD/READ (MSS-dominated)
     fallback. Added MSI-H/dMMR rows at **46/Mb** (Chalmers 2017, PMID
     28420421; MSI-H CRC is hypermutated, ≫20/Mb).

2. **Biomarker-selected ORRs (⚠, left as-is, flagged):** CESC/NPC/HNSC are
   PD-L1/CPS-enriched, so their ORR slightly overstates the all-comer
   population the TMB/expression describe. Magnitude is small (a few ORR
   points); documented rather than re-curated.

3. **Subtype-scope gaps (⚠):** LUAD_STK11 borrows bulk-LUAD TMB+expression;
   UCEC_CNL vs CNH share a single interpolated non-MSI-H ORR (KEYNOTE-158 did
   not split them); ESCA ORR is SCC-only over mixed-histology TMB/expression.

4. **Pervasive line/stage mismatch (accepted caveat):** nearly every ORR is
   metastatic/pretreated while TMB+expression are TCGA primary,
   treatment-naive. This is unavoidable with public reference data and applies
   roughly uniformly, so it shifts the intercept, not the cross-cohort ranking.

5. **HL is mechanistically off-model:** its 71% ORR comes from near-universal
   9p24.1/PD-L1 amplification, not antigen load (TMB=2). It is the clearest
   single-cohort reason the antigen axis alone can't rank response.
