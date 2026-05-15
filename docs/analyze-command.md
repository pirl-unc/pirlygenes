# The `analyze` Command

`pirlygenes analyze` is the main CLI entry point for personalized
cancer sample analysis. It takes a gene expression file (CSV, TSV,
or Excel) and produces:

- Figures (moved to `figures/` subdir)
- A combined PDF (`{prefix}-all-figures.pdf`)
- Text reports (`{prefix}-summary.md`, `{prefix}-analysis.md`, `{prefix}-evidence.md`)

Raw sample QC is computed before any expression rescue. By default,
downstream biology uses a technical-RNA-normalized expression view:
mitochondrial transcripts, NUMT-like mitochondrial pseudogenes, rRNA-like
features, and rRNA-pseudogene rows are zeroed and the remaining sample TPM
is renormalized. Downstream reference comparisons use the same normalized
analysis view; raw sample/reference values are retained for QC and provenance.
Use `--expression-qc-rescue off` to preserve raw TPM for downstream
analysis. `--expression-qc-remove-noncoding` additionally removes
noncoding-biotype rows when a biotype column is available, while keeping
protein-coding, immunoglobulin, and TCR genes; this is off by default.

## Usage

```bash
pirlygenes analyze input.csv \
    --output-dir results/ \
    --cancer-type PRAD \          # optional, auto-detected if omitted
    --label-genes "FOLH1,PSMA" \  # genes to always label in plots
    --output-dpi 200
```

## Output Files

### Text Reports

| File | Contents |
|------|----------|
| `{prefix}-summary.md` | Distilled reader-facing summary |
| `{prefix}-analysis.md` | Main interpreted report: cancer type, purity, therapy landscape, context, and reasoning |
| `{prefix}-evidence.md` | Stepwise/raw appendix: attribution chain plus full biomarker and target tables |
| `README.md` | Output directory index |

### Figures (in `figures/` subdir)

| Figure | Description |
|--------|-------------|
| `{prefix}-sample-summary.png` | Sample overview: cancer type, purity, tissue context |
| `{prefix}-decomposition-composition.png` | Standalone best-hypothesis composition bar |
| `{prefix}-decomposition-components.png` | Standalone best-hypothesis TME component breakdown |
| `{prefix}-decomposition-candidates.png` | Standalone candidate composition comparison |
| `{prefix}-purity.png` | Tumor purity estimation detail |
| `{prefix}-immune.png` | Immune gene expression strip plot |
| `{prefix}-tumor.png` | Tumor marker gene expression |
| `{prefix}-antigens.png` | Antigen gene expression (CTAs + surface) |
| `{prefix}-treatments.png` | Therapy target gene expression |
| `{prefix}-actionable-targets.png` | Canonical actionable-target screen: observed expression, tumor-source estimate, normal-tissue context, and readiness caveats |
| `{prefix}-priority-targets.png` | Priority ranking only: integrated score across tumor support, readiness, safety, and tumor level |
| `{prefix}-priority-target-context.png` | Separate evidence page: tumor range plus tumor-source, healthy-tissue, and maturity context |
| `{prefix}-target-tissues.pdf` | Per-gene tissue expression heatmaps |
| `{prefix}-purity-ctas.png`, `{prefix}-purity-surface.png` | 9-point tumor expression ranges for non-target expression categories |
| `{prefix}-cancer-types-genes.png` | Gene set heatmap vs TCGA cancer types |
| `{prefix}-cancer-types-disjoint.png` | Disjoint gene counts per cancer type |
| `{prefix}-reference-mds.png` | MDS embedding of TCGA cancer medians, subtype references, normal tissues, and the sample |
| `{prefix}-reference-neighborhood.png` | Nearest cancer/subtype/normal reference distance ranking; preserves full feature-space distance |
| `{prefix}-vs-cancer/` | Per-category scatter plots: sample vs TCGA cancer type |
| `{prefix}-vs-cancer.pdf` | Combined scatter plot PDF |
| `{prefix}-all-figures.pdf` | All figures in one PDF |
| `{prefix}-figure-audit.pdf` | Figure packet grouped by redundancy/value; each page carries the source filename |

Use `--deprecated-figures` to also emit older comparison/debug target views
under `figures/deprecated/` (`target-safety`, target-tissue PNG fan-out,
`curated-target-evidence`, and `purity-targets`). They are excluded from the
main figure packet because the canonical target figures integrate more of the
sample context: cancer type/subtype curation, observed expression,
tumor-source deconvolution, purity uncertainty, healthy-tissue context,
eligibility gates such as HLA or required alterations, and therapy-state
cautions such as expected target modulation after treatment.

### Target Ranking Evidence Contract

The priority target figures should be interpreted as a structured evidence
synthesis, not a TPM ranking. The score combines:

- cancer-type and subtype curation for expected lineage genes and
  disease-matched therapy targets;
- observed expression and tumor-source attribution across the purity interval;
- healthy-tissue / matched-normal context;
- HLA, fusion, mutation, amplification, or other required eligibility gates
  when supplied;
- therapy-state caveats, such as AR-target attenuation after ADT/ARPI in PRAD
  or IFN-driven inflation of MHC-related targets;
- clinical maturity and treatment-path tier;
- optional structured `benefit_tier` / `toxicity_tier` fields when those are
  curated for an agent.

Do not infer survival benefit or toxicity from expression alone. External data
sources that can support future curation include FDA/DailyMed structured product
labels for approved-drug adverse reactions, openFDA label/adverse-event APIs,
ClinicalTrials.gov results tables for response/survival/adverse events,
CTCAE terminology for toxicity severity grading, and side-effect resources such
as SIDER/OnSIDES. Those sources need normalization by indication, line of
therapy, endpoint (ORR/DOR/PFS/OS/RFS), dose, and grade 3+ adverse-event rate
before they can safely become a benefit-vs-toxicity score.

## Cancer Type Labels

`--cancer-type` accepts both TCGA-style coarse labels and registry labels.
These are deliberately separated in the analysis state:

- TCGA-backed labels such as `BLCA` or `SARC` constrain the expression
  reference directly.
- Fine-grained registry labels with a TCGA parent, such as `SARC_SYN`,
  keep the supplied subtype as report scope but run purity, decomposition,
  ranges, and plots against the parent expression reference (`SARC`).
  The summary cross-check reports this as concordance or discordance
  through the parent reference instead of pretending the child subtype is
  a standalone TCGA cohort.
- Non-TCGA rare labels without a resolvable parent remain report scope
  hypotheses. The nearest TCGA expression reference is shown only as
  context and should not be read as the diagnosis.

Fine-grained cohort medians and subtype expression references are analysis
resources, not bundled pirlygenes data. They should live with the pipeline
that owns their cohort ingestion, sample-level QC, and provenance.

## Expression Reference Distributions

`pan-cancer-expression.csv` is a compatibility table with TCGA and HPA
point estimates. Its HPA normal `nTPM_*` columns are consensus values,
so empirical q1/q3 cannot be reconstructed from that file alone.
Sample-level distribution references should be generated and versioned in
the analysis layer that consumes them.
- Normal-tissue q1/q3 should be generated from sample-level normal
  resources, such as GTEx/TOIL tissue matrices, then stored in a
  separate normal-expression summary table rather than inferred from a
  consensus point estimate.
- HPA cell-type tables are aggregate cell-type references. If future
  sample/cell-level HPA downloads are imported, their distributions
  should be labeled by provenance; otherwise any spread is only a proxy,
  not empirical cohort IQR.

## Tumor Purity Estimation

Purity is estimated by combining three independent methods:

### 1. Signature genes

30 cancer-type-specific genes are selected by z-score across TCGA
cancer types. Each gene's HK-normalized ratio (gene / housekeeping
median) in the sample is compared to the same ratio in the TCGA
reference, calibrated for known TCGA cohort purity (Aran et al. 2015).

### 2. ESTIMATE enrichment

Stromal and immune gene sets (Yoshihara et al. 2013) are scored as
enrichment ratios vs the TCGA reference. High stromal/immune
enrichment implies lower tumor purity.

### 3. Lineage gene calibration

Cancer-type-specific lineage genes (e.g. STEAP1, STEAP2, FOLH1, KLK3
for prostate) independently estimate purity using HK-normalized ratios.
Lineage genes are curated for all 33 TCGA cancer types.

The **upper-half median** of per-gene purity estimates is used to
resist de-differentiation artifacts: in metastatic tumors, some
lineage genes lose expression (e.g. KLK3 in prostate met), giving
artificially low purity estimates. These de-differentiated genes are
identified and excluded from the purity estimate, but reported in the
analysis with their interpretation.

The analysis report (`{prefix}-analysis.md`) includes a lineage gene
calibration table showing each gene's purity estimate and whether it
was classified as "retained — reliable" or "likely de-differentiated".

## Purity-Adjusted Expression (9-Point Ranges)

For each gene, tumor-specific expression is estimated as:

```
tumor_expr = max(0, (observed - (1-purity) * tme_background) / purity)
```

All values are HK-normalized (fold-over-housekeeping) before
subtraction, then converted back to TPM scale. This corrects for
unit differences between sample TPM, reference nTPM, and TCGA FPKM.

### TME reference

TME background is computed from curated immune and stromal tissues in
the HPA normal tissue atlas:

- **Immune**: bone marrow, lymph node, spleen, thymus, tonsil, appendix
- **Stromal**: smooth muscle, skeletal muscle, heart muscle, adipose tissue

Per-gene TME background is reported as 25th / 50th / 75th percentile
across these tissues (HK-normalized).

### 9-point grid

Uncertainty is captured by crossing 3 TME levels (25th/50th/75th
percentile) with 3 purity levels (lower/estimate/upper bound),
producing 9 estimates per gene. The median of the 9 is the point
estimate.

### % of cancer type median

Each gene's median tumor estimate is compared to the purity-adjusted
TCGA median for the matched cancer type. This answers "how does this
tumor's expression compare to a typical tumor of this type?"

For example:
- STEAP1 = 103% of PRAD median → at expected level
- KLK3 = 4% of PRAD median → lost expression (de-differentiated)
- CD74 = 33x PRAD median → likely TME-driven, not tumor

TCGA values are themselves purity-adjusted using published cohort
purity estimates (Aran et al. 2015).

## Therapeutic Target Categories

The evidence appendix (`evidence.md`) carries the full target tables and categorizes genes as:

- **CTA** (Cancer-Testis Antigens): expressed in tumor but not normal
  adult tissue. Any expressed CTA is a potential vaccination target,
  even without active clinical trials.
- **Surface proteins**: targetable by ADC, CAR-T, or bispecific
  T-cell engagers. Cross-referenced with the human surfaceome and
  tumor-specific cancer surfaceome (TCSA).
- **Therapy targets**: genes with active clinical trials (ADC, CAR-T,
  TCR-T, bispecific antibodies, radioligand therapy).
- **Intracellular targets**: non-surface proteins presentable via
  MHC-I, targetable by TCR-T or peptide vaccination.

MHC-I status is assessed to determine viability of intracellular
targeting approaches.

## Other Commands

### `pirlygenes data`

Lists all bundled datasets and available cancer types.

### `pirlygenes plot-cancer-cohorts`

Multi-sample cohort analysis. Takes multiple expression files and
produces comparative heatmaps, PCA/MDS embeddings, therapy target
summaries, and CTA/surface protein panels across the cohort.

```bash
pirlygenes plot-cancer-cohorts sample1.tsv sample2.tsv sample3.tsv \
    --cancer-type PRAD \
    --output-image-prefix cohort
```

### `pirlygenes plot-expression` (deprecated)

Legacy single-sample plotting command. Use `pirlygenes analyze` instead.

## Design principles

### Uniformity — every cancer runs the same pipeline

The analyze command runs a uniform loop over 76+ leaf codes in the
registry. Per-cancer differences live in *data* (the registry CSVs,
the signature panels, the rule catalogs) — not in `if cancer_code == ...`
code branches. A clinician reading the markdown for a PRAD sample sees
the same structure as a PANNET or NUT carcinoma report, differing only
in the curated content.

Three data-driven catalogs power the per-cancer differentiation:

| Catalog | Purpose | Contract |
|---|---|---|
| `disease-state-rules.csv` + `narrative-gene-sets.csv` | Per-cancer disease-state narrative rules (castrate-resistant PRAD, HER2+ BRCA, ER-suppressed BRCA, plus pan-cancer EMT/hypoxia/IFN) | Adding a narrative = adding a CSV row; no Python change. Rule engine: `pirlygenes.disease_state_rules.synthesize_disease_state`. |
| `degenerate-subtype-pairs.csv` + `fusion-surrogate-expression.csv` | Within-family subtype disambiguation (OS vs DDLPS, Ewing vs DSRCT vs ARMS, PANNET vs MID_NET vs lung NET, squamous trio, B-cell lymphomas, NUTM vs squamous). Activation-gated — pairs only fire when the shared signature is actually present in the sample. | Resolver: `pirlygenes.degenerate_subtype.resolve_degenerate_subtype`. |
| `cancer-type-registry.csv` | The canonical leaf-code list with family / primary_tissue / primary_template / subtype_key / mixture_cohort columns | Registry completeness contract (#199) requires every leaf to carry lineage + biomarker + therapy + therapy-response axis panel. Expression and matched-normal coverage live in downstream analysis tests. |

Cancer-type curation is layered. For each registry row, the goal is to
curate:

- **Identity and grouping:** canonical code, display name, clinical
  group/family, parent subtype relationship, primary tissue, and
  decomposition template.
- **Expression reference:** TCGA or non-TCGA tumor-expression medians
  used for cancer-type comparison, subtype refinement, purity, and
  target range context. This can come from multiple datasets; it is not
  the same thing as the registry/clinical curation source.
- **Lineage markers:** genes used for lineage/purity calibration and
  to decide whether the sample retains or loses expected tumor identity.
- **Biomarkers and therapy targets:** marker rows and treatment-target
  rows in `cancer-key-genes.csv`, including agent, maturity, indication,
  eligibility caveats, and source references where available.
- **Matched-normal context:** tumor-vs-parent-normal marker rows used
  to avoid mistaking benign parent-tissue signal for tumor-cell signal.
- **Response and biology axes:** cancer-specific therapy-response or
  activation signatures used in the disease-state narrative.
- **Rare/subtype reasoning hooks:** fusion rules, RNA surrogates,
  degenerate subtype pairs, mutation-expression effects, and expected
  downstream expression effects when curated.

Run `pirlygenes cancers` to see the current coverage audit and a
per-cancer table. Its `Curation source` column is intentionally separate
from `Expression source`: a row may be clinically/literature curated
while its numeric expression median comes from Treehouse, TARGET, TCGA,
BeatAML, GEO, or another public cohort.

### Conditional figures and reports — data-driven

Figure emission is gated by what the registry has for the called
cancer, not by code branches:

- `sample-subtype-signature.png` emits when the cancer has a
  cancer-specific `cancer_context` in `therapy-response-signatures.csv`.
- `sample-subtype-attribution-*.png` emits when within-family
  subtype refinement fires on the sample. These are audit/provenance
  figures and are mainly surfaced in `figure-audit.pdf`.
- Degenerate-pair resolution surfaces a `Subtype note:` line in
  `summary.md` whenever the resolver corrects or flags ambiguity.

### Public data-discovery surface

All bundled catalogs are discoverable via `pirlygenes.gene_sets_cancer`:

```python
from pirlygenes.gene_sets_cancer import (
    narrative_gene_set, narrative_gene_set_names,
    disease_state_rules_df,
    degenerate_subtype_pairs_df,
    fusion_surrogate_expression_df,
    fusion_surrogate_genes_for_cancer,
)
```

`pirlygenes data` lists every bundled CSV with a source description.

See also:
- `docs/reasoning-pipeline.md` — step-by-step information flow through the analyze pipeline (Steps 0–6 including Step 5b subtype disambiguation).
