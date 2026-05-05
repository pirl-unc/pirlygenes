# Lineage-Aware Decomposition

> Status: proposal / design doc. Targets the per-gene attribution pipeline in
> `plot_tumor_expr.py` and the decomposition engine in
> `pirlygenes/decomposition/engine.py`.

## TL;DR

Replace the current per-gene `attr_tumor = observed − effective_non_tumor`
heuristic with a three-stage decomposition that

1. fits an **explicit matched-normal-epithelium compartment** for genes whose
   expression is shared between tumor and tissue-of-origin;
2. enforces **per-compartment TPM mass balance** via Iterative Proportional
   Fitting, so attribution sums to the fitted compartment fractions;
3. **regularizes per-gene tumor expression** with priors from cohort
   references (TCGA per-cancer-type) and active pathway / therapy
   signatures already curated in `therapy_response.py` and the cancer
   gene-set registry.

The output keeps all existing TSV columns (`attr_tumor_tpm`,
`attr_tumor_fraction`, etc.) and adds `tumor_fold_over_matched_normal[g]`,
which is what the report actually wants for prostate-lineage targets.

## Motivation

For the canonical case — a low-purity prostate biopsy where tumor cells are
of prostate-epithelial lineage — the report needs three things the current
pipeline cannot produce:

- **A non-zero, defensible tumor share for canonical PRAD lineage genes
  (KLK3, KLK2, FOLH1, NKX3-1, STEAP1/2, TMPRSS2, HOXB13)**, because tumor
  cells in PRAD inherit the AR-driven program. The current binary
  tumor-vs-TME split forces these into either over-claim (all tumor) or
  silent erasure (`attr_tumor = 0`, "TME-dominant" fallback).
- **A fold-change against matched-normal prostate epithelium**, not against
  TME, because that ratio is the clinically interpretable number for
  de-differentiation status, target selection, and CRPC/NEPC subtype
  reasoning.
- **A purity-invariant readout** for therapy targets so capped genes don't
  all collapse onto the same `0.45 × observed` ceiling. Without this, the
  17 capped therapy targets in a 15%-purity sample lose all relative
  ordering on tumor-source signal.

These needs are not solvable by tuning the existing flow; they require a
matched-normal compartment in the decomposition and a per-gene fit that
shares signal between the tumor and matched-normal axes.

## Errors this is intended to fix

Concrete failure modes observed on a low-purity PRAD case
(15% purity, AR-suppressed CRPC/NEPC-emerging):

| # | Symptom | Root cause |
|---|---|---|
| E1 | `matched_normal_tissue` field empty in the output TSV despite the sample being PRAD-context | `decomposition/templates.py` deliberately excludes the matched-normal-epithelium column to "avoid stealing retained lineage signal from tumor cells" (templates.py top comment). This is correct for purity estimation but wrong for per-gene attribution of lineage genes. |
| E2 | `attribution_raw_sum_tpm = 0` for ~all therapy targets | `per_compartment_tpm_by_symbol` is built only from genes in the fitted signature matrix; non-lineage surface markers fall through and silently drop to the breadth-floor heuristic. |
| E3 | `attr_tumor` collapses to `(1−purity) × HPA mean_top_N_healthy` for these genes | Breadth-floor fallback uses HPA whole-tissue means, which underestimate stromal contribution in cancer-context tissue. Reference is biologically wrong for the subtraction it is asked to perform. |
| E4 | All 17 capped therapy targets sit at exactly `0.45 × observed` | Uniform 3× headroom in the `observed × purity × 3` cap flattens distinctions between targets at low purity. The cap is a ceiling, not an estimator. |
| E5 | Lineage genes flagged "likely de-differentiated" with `attr_tumor` 0–5 TPM | Matched-normal prostate signal *exceeds* observed TPM for collapsed lineage genes (e.g., `0.085 × HPA_KLK3 ≈ 255 TPM ≫ 24 TPM observed`). The model has no way to express "tumor lost the gene" cleanly; it falls into TME-dominant. |
| E6 | No fold-change against matched-normal | `tumor_fold_over_matched_normal[g] = T[g] / HPA_prostate_epi[g]` is never computed because there is no `MN[g]` latent in the model. |

E5 and E6 are the key product gaps. E1–E4 are the mechanism gaps that
preclude solving them.

## Design principles

1. **Two latents per gene, not one.** For lineage-aware genes, fit
   `T[g]` (per-tumor-cell expression rate) *and* `MN[g]`
   (matched-normal-epithelium per-cell rate) jointly. The current single
   latent (`attr_tumor`) cannot represent the shared-lineage case.
2. **Anchor latents to references with weights, don't replace them.** Both
   `T[g]` and `MN[g]` carry a regularizer toward a reference; the data
   pulls them away when it has the strength to do so. The reference for
   `T[g]` is the TCGA cohort prior (cancer-type-specific bulk median,
   purity-adjusted); the reference for `MN[g]` is the HPA per-cell-type
   single-cell mean for the matched normal lineage (e.g., prostate
   glandular cell), not the whole-tissue mean.
3. **Enforce TPM invariants explicitly.** Per-gene mass balance
   (`Σ_c attr[g,c] + tumor[g] = observed[g]`) and per-compartment mass
   balance (`Σ_g attr[g,c] = f[c] · Σ observed`) must both hold at the
   end. The current pipeline enforces the first but not the second; the
   second is the missing constraint that makes attributions defensible.
4. **Lineage-awareness is gene-set-scoped, not global.** Apply the
   matched-normal compartment only to genes flagged as lineage-relevant
   for the inferred cancer type. For non-lineage genes, the model
   collapses back to the standard tumor + stromal-compartments split, so
   the change is non-disruptive for the rest of the report.
5. **Pathway/therapy priors nudge, they don't override.** When a pathway
   signature is "active" (high MPAS-like score, high IFN, high
   AR-suppression flag), the prior on `T[g]` for genes in that pathway
   shifts in the implied direction. This is a soft constraint, scaled by
   the pathway score's own confidence.

## Algorithm overview

Three stages, run after `decomposition_results` is populated:

### Stage 1 — Joint per-gene NNLS fit

For each gene g, with fitted compartment fractions
`{f_T, f_MN, f_c1, f_c2, …}`:

```
minimize  λ_T   · (T[g]   − tcga_cohort_prior[g])²
        + λ_MN  · (MN[g]  − hpa_matched_normal[g])²
        + λ_S   · Σ_c (S[g,c] − hpa_per_cell_type[g,c])²
        + λ_P   · Σ_p pathway_priors[g,p]      ← from therapy_response.py

subject to
  f_T · T[g] + f_MN · MN[g] + Σ_c f_c · S[g,c] = observed[g]
  T[g], MN[g], S[g,c] ≥ 0
```

Solve as a constrained QP per gene. Vectorizable across genes; ~10⁴ genes
fit in ≪1s with `scipy.optimize.nnls` or `cvxpy`.

The four regularization terms encode the priors:

- `λ_T`: tumor expression should be near the cohort median for this
  cancer type, after deconvolving TCGA's own cohort impurity.
- `λ_MN`: matched-normal expression should be near HPA single-cell
  reference for the lineage cell type (prostate glandular for PRAD,
  alveolar epithelium for LUAD, ductal epithelium for BRCA, etc.).
- `λ_S`: stromal/immune compartments stay near HPA per-cell-type means.
- `λ_P`: pathway priors. If MAPK is active in the sample, the prior on
  `T[g]` for `MAPK_ACTIVITY_GENES` is multiplied by the pathway's
  fold-change above cohort baseline, weighted by pathway-score
  confidence. Same machinery for IFN, hypoxia, EMT, AR axis (see
  *Pathway / therapy priors as nudges* below).

### Stage 2 — IPF rebalance for per-compartment mass balance

After per-gene fits, the per-compartment column sums won't generally
satisfy `Σ_g attr[g,c] = f[c] · Σ observed`. Iterative Proportional
Fitting (matrix scaling) restores both invariants in alternation:

```
for iter in 1..N:
    # Column scaling: pull each non-tumor compartment to its target mass
    for c in non_tumor_components:
        target = f[c] · Σ_g observed[g]
        actual = Σ_g attr[g,c]
        if actual > 0:
            scale = clip(target/actual, 1/MAX_DEV, MAX_DEV)
            attr[*, c] *= scale
    # Row scaling: re-enforce per-gene sum == observed
    for g:
        s = Σ_c attr[g,c] + tumor[g]
        if s > 0:
            attr[g,*] *= observed[g]/s
            tumor[g]  *= observed[g]/s
```

`MAX_DEV` (default 3.0) bounds how far IPF can push any single
attribution from its reference, preventing fibroblast EGFR being scaled
to nonsense to satisfy mass balance.

The column-scaling step needs an *eligibility filter*: a gene can only
absorb additional mass into compartment c if (a) HPA reference for that
gene in that cell type is non-zero, OR (b) the gene is in a curated
compartment-marker set (e.g., CAF gene set for fibroblasts, TAM gene
set for macrophages). This prevents tumor-restricted genes (KLK3, FOLH1)
from being recruited to fill stromal mass gaps.

Convergence: typically 3–10 iterations.

### Stage 3 — Per-gene specificity-aware cap

Replace the uniform `cap = observed × purity × 3` with a gene-specific
headroom derived from HPA tissue specificity:

```
specificity[g] = log2(max_tissue_TPM[g] / mean_tissue_TPM[g])
headroom[g]    = clip(specificity[g] / median_specificity, 1.0, 5.0)
cap[g]         = purity · observed[g] · headroom[g]
```

Highly tissue-restricted genes (KLK3 in prostate, NEUROD1 in brain,
SFTPC in lung) get higher headroom because per-tumor-cell expression
*should* exceed bulk levels by a wide margin in their lineage. Broadly
expressed genes get a cap close to `purity × observed`, which is the
biological floor.

For genes carrying a TCGA cohort prior (Stage 1's `tcga_cohort_prior[g]`),
the cap is redundant; the prior already prevents runaway. The cap kicks
in mainly for therapy targets without a cohort prior.

## Pathway / therapy priors as nudges

`pirlygenes/therapy_response.py` already curates the gene panels and
score machinery for MAPK, IFN, hypoxia, EMT, and AR-axis activity (and
the registry has CTAs, tumor-source genes, lineage panels per cancer
type via `gene_sets_cancer.py`). The Stage-1 `λ_P` regularizer is a
direct hookup:

```
for each scored pathway p with score s_p (active / suppressed):
    fold_target_p = sample_score / cohort_baseline_score
    for g in p.support_genes:
        prior_T[g] *= fold_target_p · w_p
```

`w_p` is the pathway-score confidence (from `TherapyAxisScore`). This
encodes "if MAPK is 4.8× active, the prior on tumor MAPK targets is
shifted ~5× above cohort median; if AR-axis is suppressed, prior on
tumor AR-target genes is shifted toward zero."

This composes with — does not replace — the per-cohort cohort prior. A
gene under multiple active pathways picks up multiple multiplicative
nudges; one inactive pathway leaves the prior unchanged.

## Cancer-context references for non-tumor compartments

Separate, composable improvement (independent of the algorithm):

- **CAF / TAM / CAE atlases** replace HPA columns when a cancer-type-
  specific atlas is loaded. For PRAD: Tuong/Karthaus/Joseph prostate
  scRNA-seq references for fibroblast/endothelial/myeloid columns.
  Ship as optional bundled datasets (`pirlygenes/data/atlases/prad/`).
- **Per-cohort residual learning**: from TCGA-PRAD bulk samples with
  ABSOLUTE-derived purity, fit an additive shift `δ_PRAD[g,c]` for each
  gene × compartment, applied as a learned prior to Stage 1 when the
  sample's report scope matches.

These plug into Stage 1's `λ_S` regularizer by replacing the reference
mean for the relevant compartment columns.

## Implementation plan

### Phase 0 — Lineage-aware compartment plumbing (1–2 days)

- **`decomposition/templates.py`**: add a per-cancer-type flag
  `enable_matched_normal_for_attribution` (default `True` for
  epithelial primaries) that controls whether the matched-normal column
  is included in the *attribution* fit. Keep the *purity* fit excluding
  it (preserve the existing E1 design choice).
- **`decomposition/engine.py`**: refactor `_fit_one_hypothesis` so the
  attribution fit and the purity fit can run with different compartment
  sets. Today they share. Surface the matched-normal fraction in
  `DecompositionResult.fractions` for the attribution path.
- **`tumor_purity.EPITHELIAL_MATCHED_NORMAL_TISSUE`**: verify coverage
  for all epithelial cancer types (PRAD, BRCA, LUAD, LUSC, COAD, READ,
  PAAD, STAD, BLCA, OV, HNSC, ESCA, KIRC, etc.). Add missing entries.

**Test**: rerun PRAD case, assert `matched_normal_tissue == "prostate"`
and `matched_normal_fraction > 0` in the output TSV.

### Phase 1 — HPA per-cell-type matched-normal reference (1 day)

- **New module `pirlygenes/decomposition/matched_normal_ref.py`**: load
  HPA single-cell prostate-glandular, breast-glandular,
  alveolar-epithelial, etc. Provide `matched_normal_reference(cancer_type)
  -> dict[symbol, tpm]`.
- The current `signature.build_signature_matrix` uses HPA whole-tissue
  by default; extend to accept a `cell_type_specific=True` flag for the
  matched-normal column.

**Test**: KLK3 reference for PRAD matched-normal returns ~3000 TPM,
not the whole-tissue prostate mean of ~800.

### Phase 2 — Stage 1 NNLS solver (2–3 days)

- **New module `pirlygenes/decomposition/refine.py`** with entry point:

  ```
  refine_per_gene_attribution(
      observed_tpm: pd.Series,
      fractions: dict,          # {tumor: f_T, matched_normal_<X>: f_MN, fibroblast: f_S1, ...}
      hpa_per_cell_type: pd.DataFrame,
      matched_normal_ref: pd.Series,
      tcga_cohort_prior: pd.Series,
      pathway_priors: dict,     # {gene: multiplicative_nudge}
      lineage_gene_set: set,    # genes that get the matched-normal column
      lambda_T: float = 0.5,
      lambda_MN: float = 1.0,
      lambda_S: float = 0.3,
      lambda_P: float = 0.2,
  ) -> AttributionRefineResult
  ```

- Implement the per-gene QP using scipy. Vectorize across genes by
  batching into a dense system; for genes outside `lineage_gene_set`,
  solve the simpler model without `MN[g]`. Parallelize across genes
  with joblib if needed.
- Output: per-gene `T[g]`, `MN[g]`, `S[g,c]`, plus diagnostics (active
  constraints, residual, prior-deviation).

**Test**: synthetic mixture (known T, MN, fractions) recovers all
parameters within 5% relative error.

### Phase 3 — IPF rebalance (1 day)

- **`pirlygenes/decomposition/refine.py::ipf_rebalance`**: as in the
  algorithm sketch. Input is the attribution matrix from Phase 2;
  output is the same shape with both invariants satisfied.
- Eligibility filter for column scaling: gene must have non-zero
  reference for the compartment being filled, *or* be in the
  compartment's curated marker set.
- Bounded scaling factor (`MAX_DEV` parameter) to prevent unrealistic
  rescues.

**Test**: after rebalance, assert
`abs(Σ_g attr[g,c] − f[c]·Σ observed) / (f[c]·Σ observed) < 0.05` for
each non-tumor compartment.

### Phase 4 — Per-gene specificity-aware cap (½ day)

- **`plot_tumor_expr.py::_attribution_candidate`**: replace uniform 3×
  headroom with `headroom[g]` derived from HPA tissue specificity.
- Gate: only apply the cap when no cohort prior is active for the gene.
  (Cohort prior already prevents runaway.)

**Test**: capped therapy targets are no longer collinear at
`0.45 × observed` — they spread across a range.

### Phase 5 — Pathway/therapy prior wiring (1 day)

- **New helper in `therapy_response.py::pathway_attribution_priors`**:
  return `{gene: multiplicative_nudge}` from current therapy axis
  scores. Each scored pathway contributes one multiplicative factor per
  support gene.
- Wire into `refine_per_gene_attribution` as `pathway_priors`.

**Test**: in a synthetic sample with MAPK score = 5×, ETV5 / EPHA4 /
DUSP6 / SPRY4 tumor priors are shifted ~5× upward; `T[g]` for those
genes lands within 30% of the shifted prior when observed expression
supports it.

### Phase 6 — Output integration (1 day)

- Add column `tumor_fold_over_matched_normal` to
  `low-purity-prad-tumor-expression-ranges.tsv`, computed as
  `T[g] / matched_normal_ref[g]` when `MN[g]` is fitted, else NA.
- Add a new analysis section to `*-analysis.md`:
  *"Lineage-share attribution"* — table of lineage genes with columns
  `observed`, `T[g]`, `MN[g]`, `f_T·T[g]`, `f_MN·MN[g]`, `tumor/normal`,
  `interpretation` (retained / amplified / partial-loss / lost).
- Update `pirlygenes-summary.md` to surface the per-target
  `tumor/matched-normal` ratio for the top therapy candidates instead
  of the raw `attr_tumor_tpm`.

**Test**: report regenerates with new columns; existing columns
unchanged for non-lineage genes.

### Phase 7 — Cancer-context references (deferred)

- Bundle CAF/TAM/CAE references for top cancer types as optional
  datasets.
- TCGA per-cohort residual fits as a separate offline pipeline,
  results bundled.
- This phase is independent and can ship later.

## Validation plan

Three layers:

1. **Unit tests on synthetic data.** Mixtures with known
   `(T, MN, S, fractions)` recovered to 5% relative error.
2. **Regression tests on the curated case set.** Run on a fixed list
   of TCGA-PRAD, TCGA-BRCA, TCGA-LUAD samples with known purity
   (ABSOLUTE) and known lineage status. Check
   `tumor_fold_over_matched_normal` for canonical lineage genes
   matches the expected pattern (PRAD primaries: KLK3 ~1× retained;
   CRPC: KLK3 ~0×; NEPC: KLK3 0× *and* AR ~0.3×).
3. **A/B comparison on the low-purity PRAD case.** Side-by-side report of
   `attr_tumor` (legacy) vs `T[g]` and `tumor_fold_over_matched_normal`
   (refined) for the lineage panel and the therapy targets. Manual
   review with the user; commit the refined path as default once the
   STEAP1/STEAP2/KLK3/AR pattern matches the CRPC/NEPC interpretation
   without manual cancer-type forcing.

## Open questions

- **Multiple matched-normal references when sample matches multiple
  tissues** (e.g., a metastasis biopsy with both prostate and bone
  marrow context). Default to the cancer-type's primary tissue;
  surface a warning if Step-0 places the sample in a non-canonical
  tissue.
- **Reference-prior weights** (`λ_T`, `λ_MN`, `λ_S`, `λ_P`). Default
  values need tuning against the validation set in (2). Expose as
  CLI knobs for now.
- **Genes with no cohort prior and no HPA reference for any
  compartment.** Currently fall through to `attr_tumor = observed`
  (everything tumor). Probably correct as a default — it's a true
  unknown — but flag in the output so downstream consumers know.
- **Compatibility with non-epithelial cancers** (sarcomas, gliomas,
  hematologic). These don't have a clean matched-normal-epithelium.
  Phase 0's per-cancer-type flag handles this; default off for these
  cohorts so they keep current behavior.
