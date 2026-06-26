# Reasoning Pipeline

Historical note: this document describes the per-sample reasoning flow that now
belongs in [`trufflepig`](https://github.com/pirl-unc/trufflepig). It remains
here as migration context for pirlygenes-owned reference data and API
contracts.

The analysis runner builds an auditable inference state from RNA and any
optional clinical or molecular inputs. The report is not a single thresholded
label. It is a chain of evidence: what was supplied, what RNA suggests, where
those sources agree or disagree, how uncertain each estimate is, and which
downstream conclusions depend on that uncertainty.

The main design rule is simple: **later steps add evidence; they do not
erase earlier evidence**. If an external label disagrees with RNA, both
remain visible. If a rare-marker rule adds a hypothesis, it widens the
hypothesis set instead of silently replacing pathology. If purity is
uncertain, target expression and therapy interpretation inherit that
uncertainty.

## The State Model

Every step writes named state into the in-memory `analysis` dictionary.
Report synthesis is a reader of that state.

| State layer | Main keys | What it means |
|---|---|---|
| Supplied context | `cancer_type_source`, `analysis_constraints`, `fusion_records`, `alteration_records` | User- or file-supplied facts and constraints. |
| RNA context | `expression_scale_qc`, `sample_context`, `healthy_vs_tumor` | Whether the expression table looks usable, what assay context is inferred, and broad tissue/tumor evidence. |
| Cancer hypotheses | `candidate_trace`, `fit_quality`, `reference_cancer_type`, `report_scope_cancer_type` | Ranked RNA cancer-code hypotheses, fit/confidence context, and the distinction between expression reference and report label. |
| Rare/refined hypotheses | `rare_report_scope_inference`, `fusion_report_scope_inference`, `rare_marker_hypotheses` | Data-backed hypotheses from rare markers or direct fusion evidence. |
| Quantitative estimates | `purity`, `purity_confidence`, `decomposition`, `expression_ranges` | Tumor fraction, tumor/TME composition, and tumor-attributed target TPM ranges. |
| Biology state | `therapy_response_scores`, `pathway_activity_inferences`, `fusion_expression_hypotheses`, `mutation_expression_hypotheses` | Active pathways, therapy-response axes, and alteration-effect consistency checks. |
| Report products | `call_summary`, markdown and figure outputs | The human-facing synthesis of the state above. |

The important split is `reference_cancer_type` versus
`report_scope_cancer_type`. A registry-only or rare label can be the
report scope while a broader TCGA or Treehouse cohort remains the numeric
expression reference used for purity, ranges, plots, and cohort-relative
fold changes.

## Step Map

```
  Input expression + optional context
      |
      v
  Preflight: expression-scale QC, fusion/alteration parsing, sample context
      |
      v
  Step 0: tissue context + tumor-evidence gate
      |
      v
  Step 1: ranked cancer-code hypotheses and confidence
      |
      v
  Scope harmonization: supplied label, rare/fusion hypotheses, parent refs
      |
      v
  Step 2: purity estimate with confidence and interval
      |
      v
  Step 3: decomposition across cancer/template hypotheses
      |
      v
  Step 4: biology state, pathway activity, fusion/mutation-expression checks
      |
      v
  Step 5: tumor-attributed target expression and subtype/degenerate handling
      |
      v
  Step 6: markdown, tables, and figures
```

## Preflight

The command first validates and annotates the input:

- **Expression-scale QC** checks whether the table looks like TPM-like
  expression rather than log-transformed or severely truncated values.
- **Sample context** (`pirlygenes.sample_context`) infers library prep,
  preservation/degradation patterns, and assay caveats from expression
  structure.
- **Optional molecular inputs** parse fusion, alteration, and HLA data
  when supplied. Missing optional inputs are tracked so reports can ask
  for them only when they would change interpretation.

Writes: `expression_scale_qc`, `sample_context`, `fusion_records`,
`alteration_records`, input-presence flags.

Consumed by: every downstream step that needs assay caveats, missing-data
prompts, or direct molecular evidence.

## Step 0: Tissue Context

Module: `pirlygenes.healthy_vs_tumor`

Step 0 asks the coarsest RNA questions:

- Which HPA normal tissues are closest?
- Which broad cancer cohorts are closest?
- Is there tumor-associated expression from CTA, oncofetal,
  proliferation, hypoxia, glycolysis, or tumor-up marker panels?
- Is the result structurally ambiguous, especially lymphoid or
  mesenchymal?

Writes: `healthy_vs_tumor`, including ranked normal tissues, ranked TCGA
cohorts, `cancer_hint`, tumor-evidence scores, and a named
`reasoning_trace`.

Consumed by: report banners, call-confidence checks, and the reader-facing
explanation of whether RNA is acting like tumor, normal tissue, or an
ambiguous mixture. Step 0 does not set the final cancer type.

More detail: [step0-reasoning.md](step0-reasoning.md).

## Step 1: Cancer Hypotheses

Module: `pirlygenes.tumor_purity`

`rank_cancer_type_candidates(...)` scores cancer-code hypotheses using
signature fit, purity plausibility, lineage support, stability, and
family-aware support factors. The result is a ranked `candidate_trace`.

The top row is the working RNA expression hypothesis. The remaining rows
are retained as alternatives, especially when scores are close, lineage
support is weak, or Step 0's nearest TCGA cohort disagrees.

Writes:

- `candidate_trace`: ranked rows with score components, purity estimate,
  lineage evidence, and family context.
- `fit_quality`: whether the RNA fit is strong, weak, ambiguous, or
  contested enough to preserve alternatives. Report builders also compute
  a concise call-confidence suffix from `candidate_trace` and related
  state.

Consumed by: purity, decomposition, rare-marker cross-checks, summary RNA
alternatives, cancer-hypothesis plots, and report caveats.

## Scope Harmonization

This is the step that prevents cancer labels from being mixed together
without provenance.

- If `--cancer-type` is supplied, it sets the report scope. RNA
  classification still runs when possible and is reported as concordant,
  discordant, or ambiguous.
- If the supplied label is a refined registry child without its own
  numeric expression cohort, analysis uses the parent/reference cohort
  for quantitative work while preserving the refined label in the report.
- If direct fusion evidence or a rare-marker RNA rule supports a rare
  cancer hypothesis, that can set or widen report scope, but the nearest
  expression cohort remains labeled as context rather than diagnosis.
- If no external label or rare-scope rule applies, the top RNA candidate
  is explicitly rendered as an RNA-inferred hypothesis.

Writes: `reference_cancer_type`, `report_scope_cancer_type`,
`report_scope_parent_cancer_type`, `rare_report_scope_inference`,
`fusion_report_scope_inference`, and `cancer_type_source`.

Consumed by: every downstream report line that names a cancer type, every
cohort-relative expression calculation, and every plot title that needs
to distinguish label scope from numeric reference scope.

## Step 2: Purity

Module: `pirlygenes.tumor_purity`

Purity combines signature-gene fit, stromal/immune penalties, and
lineage-marker evidence. The output is a point estimate plus interval and
confidence tier, not just a percentage.

Writes: `purity`, `purity_confidence`.

Consumed by: decomposition anchoring, tumor-attributed TPM ranges, target
source attribution, therapy ranking caveats, and summary-level purity
language.

## Step 3: Decomposition

Module: `pirlygenes.decomposition`

The decomposition engine evaluates multiple `(cancer_type, template)`
hypotheses, such as primary solid tumor, bone metastasis, lymph-node
metastasis, or heme/marrow contexts. It fits tumor plus TME components
and produces per-gene source attribution.

Writes: `decomposition_results` and `decomposition`, including the best
hypothesis, component fractions, and gene attribution.

Consumed by: tumor-specific target TPM estimates, target source traces,
composition plots, and the "parallel hypotheses still alive" report
section.

## Step 4: Biology State

This layer asks what active biology is visible after cancer context and
purity have been established:

- `therapy_response_scores` score cancer-aware axes such as hypoxia, EMT,
  IFN, ER, AR, HER2, glycolysis, and related panels.
- `pathway_activity_inferences` summarize pan-cancer pathway activity,
  including MAPK/ERK activity and possible upstream sources when relevant.
- `fusion_expression_hypotheses` and `mutation_expression_hypotheses`
  compare RNA patterns with curated downstream effects of fusions,
  mutations, or CNVs.

These are hypothesis-level findings. They can support biology narratives
or prompt orthogonal testing, but they should not be treated as direct
mutation or fusion calls without molecular evidence.

Consumed by: disease-state narrative, target prioritization, alteration
consistency sections, and therapy caveats.

## Step 5: Tumor-Attributed Expression

Target expression is reported as a tumor-attributed range rather than a
single bulk TPM whenever decomposition and purity are available. The range
crosses purity uncertainty and TME-background uncertainty.

Writes: `expression_ranges`, `tumor_tpm_by_symbol`.

Consumed by: target tables, therapy shortlists, HLA-restricted therapy
prompts, source-attribution traces, and subtype/rare-marker reasoning.

## Step 5b: Refined And Degenerate Labels

Some distinctions cannot be resolved cleanly by bulk expression alone:
OS versus liposarcoma with 12q amplification, squamous cohorts versus
NUTM-rearranged cancer, neuroendocrine sites, or related sarcoma
subtypes. The resolver treats these as a small hypothesis set until a
tiebreaker is available.

Data files such as `degenerate-subtype-pairs.csv`,
`fusion-surrogate-expression.csv`, `rare-cancer-rna-surrogates.csv`,
`fusion-expression-effects.csv`, and `mutation-expression-effects.csv`
encode the conditions that add, constrain, or leave those hypotheses
unresolved.

Consumed by: the cancer-call line, subtype notes, rare-marker sections,
and confirmatory-testing prompts.

## Step 6: Reports

Report synthesis reads the full state and writes:

- `*-summary.md`: top-line call, basis, RNA alternatives, purity,
  shortlist, and caveats.
- `*-analysis.md`: interpreted analysis with reasoning trace and therapy
  landscape.
- `*-evidence.md`: deeper tables, source attribution, fusion/alteration
  evidence, and deduction chain.
- figures and TSVs: hypothesis plots, embedding plots, decomposition,
  target context, pathway/signature panels, and figure audit entries.

The intended reader should be able to tell:

- what was supplied externally versus inferred from RNA;
- which cancer-type or subtype alternatives remain plausible;
- how purity and decomposition uncertainty affect target expression;
- which therapy or biology statements require missing molecular, HLA,
  imaging, pathology, or clinical context.

## Uniformity Principle

Cancer-specific behavior belongs in data, not bespoke control flow. The
registry, expression references, marker panels, therapy-target curation,
rare-marker rules, fusion-effect rules, mutation-effect rules, and
response-axis panels should determine what the reports can say for each
cancer type.

The goal is that a common report structure appears for common TCGA
cancers, rare cancers, sarcomas, pediatric tumors, and registry-only
refined labels. Differences should be visible as data coverage,
confidence, and caveats, not as missing sections or hidden special cases.
