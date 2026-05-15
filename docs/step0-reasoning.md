# Step-0 Reasoning

Step 0 is the coarse tissue-context gate for `pirlygenes analyze`. It
does not diagnose a cancer type and it does not override a supplied
clinical label. It answers two narrower questions from RNA alone:

1. Which normal tissues and broad TCGA cohorts does the sample most
   resemble?
2. Is there enough tumor-associated expression to treat the sample as
   tumor-consistent, possibly tumor, or healthy-dominant?

The output is a `TissueCompositionSignal`: ranked normal-tissue matches,
ranked TCGA matches, a tumor-evidence score, a `cancer_hint`, and a
`reasoning_trace` naming the rule that fired. Later stages read this as
context and uncertainty, not as a final diagnosis.

## Mental Model

Step 0 keeps three ideas separate:

- **Correlation context**: broad RNA similarity to normal tissues and
  TCGA cancer cohorts.
- **Tumor evidence**: coordinated tumor programs such as CTA,
  oncofetal, proliferation, hypoxia, glycolysis, and tumor-up markers.
- **Structural ambiguity**: tissue contexts where bulk RNA cannot cleanly
  separate normal lineage from malignancy, especially lymphoid and
  mesenchymal samples.

The gate is conservative in both directions. Strong tumor-specific
markers can override a lineage-ambiguous correlation result, but
lineage-ambiguous cases remain visibly flagged because downstream purity
and decomposition can inherit the same ambiguity.

## Inputs And Outputs

Module: `pirlygenes.healthy_vs_tumor`

Primary entry point:

```python
assess_tissue_composition(df_expr) -> TissueCompositionSignal
```

The signal exposes:

- `top_normal_tissues`: top HPA normal-tissue references with Spearman
  correlation on log-TPM.
- `top_tcga_cohorts`: top TCGA cohort references with Spearman
  correlation on log-TPM.
- `correlation_margin`: top normal correlation minus top TCGA
  correlation. Positive values lean normal; negative values lean tumor.
- `proliferation_log2_mean`: log2(TPM + 1) geomean over the public
  13-gene mitotic panel returned by `proliferation_panel_gene_names()`.
- `tumor_evidence`: channel-level and aggregate tumor-evidence scores.
- `cancer_hint`: `"tumor-consistent"`, `"possibly-tumor"`, or
  `"healthy-dominant"`.
- `reasoning_trace`: ordered audit messages, including the named rule
  that set `cancer_hint`.

## Signal Groups

### Correlation Context

The sample is compared against the HPA normal-tissue columns
(`nTPM_<tissue>`) and TCGA cancer columns (`FPKM_<code>`) in the bundled
pan-cancer expression reference. These are broad context signals, not
diagnostic labels. A low-purity tumor, a differentiated tumor, and its
normal parent tissue may all correlate similarly.

### Tumor Evidence

Tumor evidence is aggregated from six channels. Each channel ramps from
0 to 1 between a baseline and a saturation point; the aggregate is the
unclamped sum. An aggregate score >= 1.0 means one strong channel or
several softer channels are present.

| Channel | Metric | Baseline -> Saturation |
|---|---|---|
| CTA | Count above 3 TPM | 0 -> 5 hits |
| Oncofetal-strict | Count above 3 TPM | 0 -> 2 hits |
| Type-specific tumor-up | Count above 3 TPM | 0 -> 2 hits |
| Proliferation | 13-gene log2-TPM geomean | 2.0 -> 5.0 |
| Hypoxia | CA9 TPM | 5.0 -> 50.0 |
| Glycolysis | Panel geomean fold over 50 TPM baseline | 1.0 -> 3.0 |

Guardrails prevent normal reproductive or trophoblastic tissues from
being overcalled by CTA/oncofetal programs, and prevent physiologic
lymphoid proliferation from being treated as an unambiguous tumor call.

### Structural Ambiguity

Two ambiguity flags are first-class state:

- `lymphoid_ambiguity`: top HPA tissue is lymphoid and top TCGA cohort is
  a heme/lymphoid cohort.
- `mesenchymal_ambiguity`: top HPA tissue is mesenchymal and top TCGA
  cohort is a sarcoma-like cohort.

These flags are not failures. They tell downstream reports to keep a
visible caveat because correlation, purity, and decomposition can all be
less decisive in those regimes.

## Ordered Rule Set

`pirlygenes.reasoning` runs nine named rules in order. The first matching
rule sets the `cancer_hint` and writes its hyphenated rule name to the
reasoning trace. The Python functions use underscores; the trace uses the
`@rule(...)` names shown below. Rules are pure functions of
`(signal, flags)`. `DerivedFlags` computes thresholded booleans once, and
the decorator stamps the rule name and structural-ambiguity status into
the trace.

1. **`tumor-marker-overrides-ambiguity`**:
   Strong tumor-specific marker evidence in a lymphoid or mesenchymal
   ambiguity regime -> `tumor-consistent`.
2. **`lymphoid-tissue-tumor-indistinguishable`**:
   Lymphoid normal-tissue context plus heme/lymphoid TCGA context ->
   `possibly-tumor` with a structural-ambiguity caveat.
3. **`mesenchymal-tissue-tumor-indistinguishable`**:
   Mesenchymal normal-tissue context plus sarcoma-like TCGA context ->
   `possibly-tumor` with a structural-ambiguity caveat.
4. **`aggregate-tumor-evidence`**:
   Non-ambiguous tissue plus aggregate score >= 1.0, or a strong single
   tumor-marker category -> `tumor-consistent`.
5. **`high-proliferation-panel`**:
   13-gene mitotic panel >= 4.5 log2-TPM -> `tumor-consistent`.
6. **`confident-healthy-tissue`**:
   Quiet proliferation, strong healthy correlation margin, and no soft
   tumor evidence -> `healthy-dominant`.
7. **`healthy-tissue-with-soft-tumor-signal`**:
   Healthy-leaning correlation plus quiet proliferation, but soft tumor
   marker evidence -> `possibly-tumor`.
8. **`weak-healthy-lean`**:
   Weak healthy correlation margin without stronger evidence ->
   `possibly-tumor`.
9. **`tcga-dominant-correlation`**:
   Default when no earlier rule fires -> `tumor-consistent`.

The ordering is deliberate. Tumor-specific marker evidence is allowed to
beat structural ambiguity; structural ambiguity is preserved before
aggregate scoring; healthy calls occur only after meaningful tumor
signals have had a chance to fire.

## Reporting Behavior

Step 0 can display report banners, but those banners are caveats rather
than final calls:

- **Healthy-dominant banner**: suppressed only when downstream evidence
  is very strong: purity >= 0.5 and signature >= 0.75.
- **Possibly-tumor banner**: suppressed when purity >= 0.3 or signature
  >= 0.75.
- **Structural-ambiguity banners**: never suppressed. The ambiguity is a
  property of the reference comparison itself, so downstream estimates
  may not resolve it cleanly.

## Public Panels

All Step-0 gene panels are exposed through `pirlygenes.gene_sets_cancer`
so plots, reports, tests, and downstream tools use the same definitions:

- `CTA_gene_names()`
- `oncofetal_strict_gene_names()`
- `proliferation_panel_gene_names()`
- `hypoxia_panel_gene_names()`
- `glycolysis_panel_gene_names()`
- `ddr_activation_panel_gene_names()`

Consumers should call these APIs rather than copying hard-coded gene
lists. That keeps Step-0 scoring, pathway plots, and report narratives
aligned when a panel is revised.
