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
  Step 5b  Subtype disambiguation activation-gated degenerate-pair resolution
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

## Step 5b — Within-family subtype disambiguation

Module: `pirlygenes.degenerate_subtype`
Entry: `resolve_degenerate_subtype(winning_subtype, site_template, tumor_tpm_by_symbol) -> dict`

Several within-family subtypes carry the same gene-expression
signature and cannot be distinguished on expression alone. Examples:

- **OS vs DDLPS** — both carry the 12q13-15 amplicon (MDM2 + CDK4 + FRS2).
- **Ewing vs DSRCT vs ARMS** — all CD99+ small-blue-round-cell tumors.
- **PANNET vs MID_NET vs LUNG_NET** — all CHGA/SYP/ENO2-positive
  neuroendocrine tumors; site distinguishes origin.
- **LUSC vs HNSC vs CESC** — all KRT5/6 + TP63 + SOX2 squamous.
- **CLL vs MCL vs FL** — all B-cell small-cell lymphomas.
- **NUTM vs squamous** — both squamous-ish; NUTM1 mRNA is the only
  reliable disambiguator (PRAME / MAGE-A panels are broadly expressed
  in LUSC and are NOT NUTM-specific).

Two declarative catalogs drive resolution:

- `degenerate-subtype-pairs.csv` — pair members, shared signature,
  activation signature (gene thresholds that must be met for the pair
  to fire), tiebreaker rule (`site_template`, `fusion_surrogate`, or
  `marker_combo`), and tiebreaker mapping.
- `fusion-surrogate-expression.csv` — genes whose expression is a
  deterministic surrogate for a specific fusion/translocation class
  (FATE1/NR0B1 for EWS-FLI1; NUTM1 for BRD4-NUTM1; CCNB3 for BCOR-CCNB3;
  ALK/NTRK/ROS1 expression as ectopic-fusion markers; etc.).

### Activation gating — the "is this ambiguity actually present" check

Each pair declares an `activation_signature` (e.g. `MDM2:100|CDK4:50`
for OS_vs_DDLPS, `CD99:20` for Ewing/DSRCT/ARMS, `NUTM1:1` for
NUTM_vs_squamous). The resolver evaluates the signature against the
observed tumor-attributed TPMs **before** consulting the tiebreaker;
if the shared signature isn't present, the pair is *inactive* and
the upstream classifier's call stands unchanged. A LUSC sample with
clear squamous lineage and silent NUTM1 is never pulled into the
NUTM_vs_squamous resolution path.

### Contextual inputs

The resolver reasons over three inputs:

1. `winning_subtype` — the classifier's pick (from Step 2's mixture-
   cohort lineage summary).
2. `site_template` — Step 3's top-ranked decomposition template
   (`met_bone`, `primary_retroperitoneum`, `primary_lung`, ...).
3. `tumor_tpm_by_symbol` — Step 5's per-gene tumor-attributed TPM
   dict, built from `ranges_df` at render time.

When a subtype sits in multiple pairs (e.g. HNSC ∈
{LUSC_vs_HNSC_vs_CESC, NUTM_vs_squamous}) the resolver prefers
pairs whose tiebreaker rule has the context inputs available. This
makes pair selection deterministic rather than CSV-row-ordered.

### Output

```python
{
  "final_subtype":  <str>,
  "status":         "no_pair" | "pair_inactive" | "confirmed" |
                    "corrected" | "degenerate",
  "reason":         <str>,
  "pair_id":        <str | None>,
  "alternatives":   <list[str]>,
}
```

| Status | Meaning | Markdown rendering |
|---|---|---|
| `no_pair` | Subtype not in any pair | Classifier's call rendered unchanged |
| `pair_inactive` | Pair exists but activation signature not met | Classifier's call rendered unchanged; no subtype note |
| `confirmed` | Tiebreaker agrees with classifier | Classifier's call rendered unchanged |
| `corrected` | Tiebreaker swapped to another pair member | Swapped label + `**Subtype note:**` explaining the swap |
| `degenerate` | Pair active but tiebreaker inconclusive | `(subtype: degenerate — X vs Y/Z)` + `**Subtype note:**` |

Reads: `analysis["candidate_trace"][0]["winning_subtype"]`,
`analysis["decomposition"]["best_template"]`, `ranges_df` (to build
the TPM dict).

Writes: nothing to `analysis` — the resolver returns its decision to
the caller (`build_summary`), which applies it at render time.

Consumed by: Step 6's brief / summary / actionable renderers. The
resolver's output drives the cancer-call line and the subtype note.

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
                                                                          Step 5b: SubtypeDisambiguation
                                                                                   │
                                                                                   ▼
                                                                           Step 6: Reports
```

All steps write to the same `analysis` dict; Step 6 is a pure reader.
No step mutates another step's output. This means any step's result
can be re-run independently (e.g. to debug a bad purity call) without
corrupting upstream reasoning.

---

## Uniformity principle

Every cancer type should get the same kind of analysis. The pipeline
is a uniform loop over the 76 leaf codes in the registry — no
cancer-specific code branches should gate which figures emit, which
steps run, or what depth of report gets produced. Asymmetry belongs
in *data* (the registry, the signature files, the pair catalog), not
in *code*.

Two consequences:

1. **Registry completeness is a contract** — #199 requires every leaf
   code to carry a minimum package (expression, lineage, biomarker,
   therapy, matched-normal, therapy-response axis panel). Missing
   fields are enumerated in `_MISSING_MATCHED_NORMAL` /
   `_MISSING_THERAPY_AXIS` allowlists; shrinking those lists is the
   measure of progress. A clinician should read the same structure of
   markdown for a PRAD sample as for a PANNET or a NUT carcinoma
   sample, differing only in the curated content.

2. **Conditional figure emission is data-driven, not code-driven** —
   `sample-matched-normal-*.png` emits for any code with a
   `tumor-up-vs-matched-normal` row; `sample-subtype-signature.png`
   emits for any code with a cancer-specific
   `therapy-response-signatures` row; degenerate-pair resolution
   fires for any code listed in `degenerate-subtype-pairs.csv`. The
   rendering loop is identical across cancers; which outputs land
   depends only on what the registry has for that code.

Known asymmetry (tracked, not yet resolved): the disease-state
synthesis in `cli._synthesize_disease_state` still contains hardcoded
PRAD and BRCA branches (castrate-resistant pattern detection, HER2+
BRCA pattern). These should move to a `disease-state-narratives.csv`
data file that declares "when axis X is state S AND gene Y is
retained, emit narrative Z" per cancer. Tracked in the follow-up
issue for data-ification of narrative rules.
