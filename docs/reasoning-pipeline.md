# Reasoning pipeline — steps and information flow

`pirlygenes analyze` runs a coarse-to-fine pipeline. Each step
produces a named result that downstream steps consume. This page
documents the steps and the contract between them — what each step
writes, and what each step reads from its predecessors.

The guiding principle: **no step overrides an earlier step's
output**. Later steps refine the reading by adding orthogonal
evidence; if a sample is flagged as ambiguous at Step 0, that flag
travels all the way to the brief. The reader sees the full chain.

## Step inventory (coarsest → finest)

```
  Step 0   Sample context      library prep + preservation + degradation
  Step 0   Tissue composition  top HPA normals + top TCGA cohorts + cancer-hint
  Step 1   Cancer-type call    top-k TCGA candidates with signature + purity + lineage
  Step 2   Tumor purity        point + CI + confidence tier
  Step 3   Broad decomposition NNLS fit of tumor + TME compartments (template-aware)
  Step 4   Therapy-axis state  AR / EMT / hypoxia / IFN / HER2 / ER up/down calls
  Step 5   Tumor-value core    9-point per-gene tumor-attributed TPM with TME + purity ranges
  Step 6   Report synthesis    brief / actionable / summary / analysis / targets / provenance
```

Steps 0 run before Step 1; the rest form a strictly ordered chain
from Step 1 onward. Each step emits its result into the in-memory
`analysis` dict under a known key, so Step 6 (report synthesis) can
read the full chain to surface every piece of evidence the clinician
needs.

---

## Step 0a — Sample context

Module: `pirlygenes.sample_context`
Entry: `infer_sample_context(df_expr) -> SampleContext`

Reads the expression table alone and infers:

- **library prep** — poly-A capture / ribo-depletion / total RNA /
  exome capture, with a confidence score
- **preservation** — fresh-frozen / FFPE / partial degradation
  (conditional on library prep)
- **degradation severity** — none / mild / moderate / severe
- diagnostic signals: mitochondrial fraction, ribosomal-protein
  fraction, transcript-length-bias index

Writes: `analysis["sample_context"]`

Consumed by: every downstream step reads `sample_context` as a
keyword argument. Step 3 adjusts marker-panel weightings when prep
is exome-capture (skip non-coding genes), Step 5 widens 9-point
ranges when degradation is severe, Step 6 surfaces the prep-specific
"mitochondrial and non-coding absence is expected by design" caveat.

---

## Step 0b — Tissue composition + cancer hint

Module: `pirlygenes.healthy_vs_tumor`
Entry: `assess_tissue_composition(df_expr) -> TissueCompositionSignal`

Races the sample against:

- 50 HPA normal-tissue columns (`nTPM_<tissue>`) in
  `pan_cancer_expression()`
- 33 TCGA cancer columns (`FPKM_<code>`) in the same reference

Produces:

- `top_normal_tissues`: top-3 HPA tissues with Spearman ρ on log-TPM
- `top_tcga_cohorts`: top-3 TCGA cohorts with ρ
- `proliferation_log2_mean`: 5-gene panel geomean (MKI67, TOP2A,
  CCNB1, BIRC5, AURKA)
- `cancer_hint`: `"tumor-consistent"` / `"possibly-tumor"` /
  `"healthy-dominant"`

Writes: `analysis["healthy_vs_tumor"]`

Consumed by: Step 6 (brief / summary banner). Explicitly does NOT
override the cancer call; it gives the reader a coarse "what kind of
tissue + any hint of cancer" context so they can judge downstream
confidence. Low-purity tumors and healthy tissue look similar here —
the gate surfaces that ambiguity rather than guessing.

Known limitation: lymphoid-tissue-normal correlates almost identically
with DLBC (the TCGA reference is itself >90% lymphoid tissue + the
malignant clone). Step 0b flags such cases as tumor-consistent /
ambiguous rather than over-committing; Step 3's lineage-marker
check is better positioned to distinguish.

---

## Step 1 — Cancer-type classification

Module: `pirlygenes.tumor_purity`
Entry: `rank_cancer_type_candidates(df_expr, candidate_codes=None,
top_k=6) -> list[dict]`

Scores every TCGA cancer type against the sample's expression profile
using five composable factors:

1. **signature** — z-scored match to cancer-type-enriched genes
2. **purity** — per-candidate tumor-purity estimate (lineage-weighted)
3. **support** — lineage-gene pattern concordance
4. **stability** — signature-gene dispersion / stability
5. **family-factor** — carries through to the final `geomean` score

Reads: `sample_context` (library prep adjusts marker weights),
`tissue_composition` (the Step-0b top TCGA cohorts inform the
candidate-codes set when caller doesn't override).

Writes: `analysis["cancer_candidates"]` — a list of
`{code, name, signature_score, purity_result, support_norm, geomean,
 normalized}` rows. Head of the list is the working call; rows 2-6
are the alternatives surfaced in the *analysis.md* table.

Consumed by: Step 2 pulls the top candidate's `purity_result` as
the starting point; Step 3 runs decomposition per (cancer_type,
template) hypothesis across the top-k list.

---

## Step 2 — Tumor purity refinement

Module: `pirlygenes.tumor_purity`
Entry: `estimate_tumor_purity(df_expr, cancer_code, sample_context)
         -> dict`

Combines three orthogonal estimators:

1. **signature-gene** — TCGA-cohort signature-gene fold-change
2. **ESTIMATE-style** — stromal / immune enrichment penalty
3. **lineage** — lineage-gene tumor-fraction estimator (per-gene
   agreement across the curated lineage panel)

Produces:

- `overall_estimate` / `overall_lower` / `overall_upper` — point + CI
- `purity_source` — which estimator dominated
- `purity_confidence` tier — high / moderate / low / very_low with
  per-reason explanations (wide CI, low-purity regime, inconsistency
  across estimators)

Reads: `sample_context`, top candidate from Step 1.

Writes: `analysis["purity"]`, `analysis["purity_confidence"]`.

Consumed by: Step 3 uses purity as the external anchor for the NNLS
fit; Step 5 uses CI to widen 9-point ranges; Step 6 surfaces tier
+ reasons in the brief's purity line.

---

## Step 3 — Broad-compartment decomposition

Module: `pirlygenes.decomposition`
Entry: `decompose_sample(df_expr, cancer_types, templates=None,
                          purity_override=None, sample_context=None,
                          candidate_rows=None) -> list[DecompositionResult]`

For each (cancer_type, template) hypothesis in the top-k candidates:

1. Select template — `solid_primary`, `met_bone`, `met_lymph_node`,
   `heme_marrow`, etc. Templates declare what compartments to fit
   (T_cell, B_cell, fibroblast, endothelial, matched_normal_<tissue>,
   host-specific: osteoblast + marrow_stroma for bone, mesothelial
   for peritoneal, etc.)
2. Select markers per compartment (HPA-derived specificity ranks,
   guarded against lineage-irrelevant auto-markers)
3. Fit weighted NNLS under a soft sum-to-one constraint, with the
   external purity from Step 2 as anchor
4. Attribute per gene: the TME + matched-normal contributions are
   subtracted from observed TPM to produce tumor-attributed TPM

Reads: `sample_context`, `candidate_rows` (reuses Step 1's ranking
to avoid re-running purity estimation per candidate — #85).

Writes: `analysis["decomp_results"]` (list of candidates),
`best_decomp = decomp_results[0]` — the winning hypothesis has
`gene_attribution` DataFrame + `fractions` dict + `component_trace`.

Consumed by: Step 5 reads the gene_attribution for per-target tumor
TPM; Step 6 renders the component breakdown in provenance.md.

---

## Step 4 — Therapy-response axis state

Module: `pirlygenes.therapy_response`
Entry: `score_therapy_signatures(sample_tpm_by_symbol, cancer_code)
         -> dict[axis, TherapyAxisScore]`

For each therapy axis (AR, EMT, hypoxia, IFN response, HER2, ER,
glycolysis), scores up-panel + down-panel geomean fold-change vs
TCGA cohort and emits a state call:

- `active` — up-panel elevated ≥2× cohort
- `suppressed` — up-panel ≤0.5× AND down-panel elevated (signals
  therapy exposure, e.g. AR-axis suppressed in ADT-treated PRAD)
- `intact` — neither

Writes: `analysis["therapy_response_scores"]`.

Consumed by: Step 6 synthesises disease-state narrative from axis
states ("AR axis suppressed consistent with ADT exposure ...
hypoxia active ... IFN-driven inflation of MHC-I / ISG surface
targets").

---

## Step 5 — Tumor-value adjustment + 9-point expression ranges

Module: `pirlygenes.cli` (inside `analyze()` after decomposition)

Per gene, computes a 9-point per-gene tumor-TPM range crossing
three purity levels (low / point / high) × three TME-background
levels. Attribution flags:

- `tme_explainable` — observed TPM dominantly explained by a TME
  compartment; don't credit to tumor
- `tme_dominant` — ≥70% of observed signal attributed to non-tumor
- `matched_normal_over_predicted` — matched-normal compartment
  over-absorbs (typical for KLK3 in CRPC PRAD at low purity); #134
  purity-weighted fallback kicks in
- `amplified` — observed fold over TCGA cohort median; passes
  the breadth floor if the gene's tumor-specific signal is strong

Writes: `analysis["expression_ranges"]` — DataFrame with one row
per target gene.

Consumed by: Step 6 renders targets.md deep tables + actionable
therapy landscape + brief top-3 candidates.

---

## Step 6 — Report synthesis

Modules: `pirlygenes.brief` (build_brief + build_actionable),
`pirlygenes.cli` (summary.md, analysis.md, targets.md, provenance.md)

Reads the entire `analysis` dict and produces six markdown
artefacts:

- `*-brief.md` — ≤40-line tumor-board handoff (Step-0 banner,
  cancer call, purity, disease state, top-3 therapies, caveats)
- `*-actionable.md` — oncologist-facing review with therapy
  landscape + biomarker panel
- `*-summary.md` — narrative summary (Step-0 tissue composition
  line at top, then cancer call, therapy state, purity)
- `*-analysis.md` — full pipeline detail (every step's output)
- `*-targets.md` — per-gene deep tables
- `*-provenance.md` — step-by-step deduction chain

Each of these surfaces evidence from every prior step so a reader
can follow the reasoning from Step 0 context down to the final
per-gene tumor-TPM call without losing intermediate evidence.

---

## Information flow diagram

```
  df_expr
    │
    ├─► Step 0a: SampleContext ──────┐
    │                                 │
    ├─► Step 0b: TissueComposition ──┤
    │                                 ▼
    └─► Step 1: CancerCandidates ───► Step 2: Purity ───► Step 3: Decomp
                                                                   │
                                                                   ├─► Step 4: TherapyAxes
                                                                   │
                                                                   └─► Step 5: ExpressionRanges
                                                                                   │
                                                                                   ▼
                                                                           Step 6: Reports
```

All steps write to the same `analysis` dict; Step 6 is a pure reader.
No step mutates another step's output. This means any step's result
can be re-run independently (e.g. to debug a bad purity call) without
corrupting upstream reasoning.
