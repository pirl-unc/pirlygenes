# Stage-0 reasoning: signals, rules, and evidence synthesis

The Stage-0 gate answers "what kind of tissue is this, and is there
evidence of cancer?" as the coarsest step of the pirlygenes pipeline.
This document enumerates every signal the gate reads, the ordered
rule list that drives the `cancer_hint` decision, and the
multi-channel evidence score that supports the call.

## Signals

The gate reads ten named observations from the sample expression
table, cached onto `TissueCompositionSignal`:

1. **top_normal_tissues** — Spearman-on-log-TPM correlation of the
   sample against each of the 50 HPA normal-tissue columns; top 3
   returned with their ρ.
2. **top_tcga_cohorts** — same correlation against the 33 TCGA
   `FPKM_<code>` columns; top 3 returned.
3. **correlation_margin** — `top_normal[0].ρ - top_tcga[0].ρ`.
   Positive margin = HPA beats TCGA. Bounded roughly -0.2 to +0.15.
4. **proliferation_log2_mean** — geomean on log2(TPM+1) of the
   13-gene proliferation panel (`proliferation_panel_gene_names()`).
   Tumors sit ≥ 4.5, quiescent tissue ≤ 2.
5. **cta_panel_hits** — count of CTAs (~257 genes via `CTA_gene_names()`)
   expressed above 3 TPM; also sum TPM + top-3 hit list. Guarded
   against reproductive-tissue normals (testis/placenta/ovary).
6. **oncofetal_strict_hits** — count of strict-panel hits
   (`oncofetal_strict_gene_names()` — 11 genes: AFP, LIN28A/B, TPBG,
   PLAC1, CGB family, NANOG, POU5F1) above 3 TPM. Guarded against
   testis/placenta/ovary/liver.
7. **type_specific_hits** — count of hits from the top-TCGA-cohort's
   own tumor-up-vs-matched-normal panel (`tumor_up_vs_matched_normal`).
8. **hypoxia_ca9_tpm** — single-gene hypoxia marker (CA9 above
   5 TPM → non-zero score, ≥ 50 TPM → saturated).
9. **glycolysis_geomean_fold** — panel geomean TPM divided by a
   conservative baseline (`glycolysis_panel_gene_names()`).
10. **structural-ambiguity flags**:
    - `lymphoid_ambiguity` — top HPA ∈ lymphoid-normal-set AND top
      TCGA ∈ heme-lymphoid-cohort-set
    - `mesenchymal_ambiguity` — top HPA ∈ mesenchymal-normal-set AND
      top TCGA ∈ sarcoma-cohort-set

## Evidence score

Six of the signals contribute to a unified tumor-evidence score in
`TumorEvidenceScore`. Each channel produces a sub-score in [0, 1]
via a linear ramp between a baseline and a saturation point. The
aggregate is the un-clamped sum — ≥ 1.0 means either one strong
channel alone or several soft channels co-occurring.

| Channel | Metric | Baseline → Saturation |
|---|---|---|
| CTA | count above 3 TPM | 0 → 5 hits |
| Oncofetal-strict | count above 3 TPM | 0 → 2 hits |
| Type-specific | count above 3 TPM | 0 → 2 hits |
| Proliferation | panel log2-TPM geomean | 2.0 → 5.0 |
| Hypoxia | CA9 TPM | 5.0 → 50.0 |
| Glycolysis | panel geomean fold over baseline 50 TPM | 1.0 → 3.0 |

## Rule list

Stage-0 picks a `cancer_hint` by running these rules in order. The
first rule whose precondition matches fires and writes to the
`reasoning_trace` for audit.

- **R1 — strong tumor evidence**: `aggregate ≥ 1.0` OR strong single
  channel (CTA ≥ 4 hits / oncofetal ≥ 2 hits / type-specific ≥ 2 hits)
  → **tumor-consistent**. The aggregate catches the low-purity case
  where no single channel crosses the strong threshold but multiple
  soft channels co-occur.
- **R2 — lymphoid ambiguity**: top HPA lymphoid AND top TCGA heme-
  lymphoid cohort → **possibly-tumor / structural-ambiguity**.
  Banner notes bulk-RNA-correlation cannot distinguish normal
  lymphoid from lymphoid malignancy; downstream cancer-specific
  analysis proceeds under the tumor-sample prior.
- **R3 — mesenchymal ambiguity**: top HPA mesenchymal AND top TCGA
  sarcoma cohort → **possibly-tumor / structural-ambiguity**.
  Analogous to R2 for well-differentiated SARC vs smooth muscle /
  adipose / skeletal muscle / endometrial myometrium.
- **R4 — elevated proliferation alone**: panel log2-TPM ≥ 4.5 →
  **tumor-consistent** (one strong channel is enough).
- **R5a — confident healthy**: quiet proliferation (log2 < 3.5) AND
  strong healthy-margin (≥ 0.05) AND no soft tumor evidence →
  **healthy-dominant**.
- **R5b — demoted healthy**: same as R5a but with soft CTA /
  oncofetal / type-specific signals → **possibly-tumor**.
- **R6 — weak healthy margin**: margin ≥ 0.02 → **possibly-tumor**
  (soft caveat).
- **R7 — TCGA-dominant correlation**: neither HPA nor proliferation
  enough → **tumor-consistent** (default).

The ordering is deliberate: R1 (strong tumor evidence) fires before
R2/R3 (structural ambiguity) because a sample with overwhelming
CTA signal is definitively tumor even in a lineage-ambiguous
correlation regime. pfo004 (SARC, 58 CTA hits at 5000+ TPM) is the
canonical R1-wins-over-R3 case.

## Banner suppression

Banners are displayed to the clinician in the brief / actionable.
They can be suppressed when downstream evidence (Stage 1 signature +
Stage 2 purity) corroborates the cancer call:

- **Healthy-dominant banner**: suppressed only under very strong
  corroboration (purity ≥ 0.5 AND signature ≥ 0.75).
- **Possibly-tumor banner**: suppressed when purity ≥ 0.3 OR
  signature ≥ 0.75.
- **Structural-ambiguity banners** (lymphoid / mesenchymal): NEVER
  suppressed — the downstream purity / signature estimates are
  themselves spurious in these regimes because the TCGA reference
  is dominated by the ambiguous tissue.

## Public-API panels

All gene panels are exposed via `pirlygenes.gene_sets_cancer`:

- `CTA_gene_names()` — ~257 cancer-testis antigens
- `oncofetal_strict_gene_names()` — 11 strict oncofetal markers
- `proliferation_panel_gene_names()` — 13-gene mitotic panel
- `hypoxia_panel_gene_names()` — 5-gene hypoxia panel
- `glycolysis_panel_gene_names()` — 8-gene Warburg panel
- `ddr_activation_panel_gene_names()` — 6-gene DDR panel
- `tumor_up_vs_matched_normal(cancer_code=None)` — 99 solid-tumor
  cohort-specific markers
- `heme_tumor_up_vs_matched_normal(cancer_code=None)` — 20 DLBC / LAML
  tumor-up markers

Consumers can reuse these panels for downstream scoring, signature
calibration, or cross-cohort comparison — same definitions the
Stage-0 gate uses.
