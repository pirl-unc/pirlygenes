# pirlygenes

Curated gene sets, pan-cancer expression references, and bulk RNA-seq decomposition tools for cancer immunotherapy.

**What it ships:** ~30 curated CSVs covering therapy targets (ADC, CAR-T, TCR-T, bispecific, radioligand), cancer-testis antigens, surface proteins, cancer-driver genes, housekeeping genes, immune/stromal marker panels, and pan-cancer expression across 33 TCGA types and 50 normal tissues. All data are bundled in the package — no downloads needed.

**What it does with them:** The gene sets are the building blocks for decomposing a bulk tumor expression profile into its constituent parts — tumor, immune, stromal, and site-specific host tissue — so that per-gene expression can be conservatively attributed to the tumor rather than naively read from raw TPM. The `analyze` command runs this full pipeline end-to-end and produces reports, figures, and structured data.

## Contents

- [Quick start](#quick-start) — install, run `analyze`, use gene sets in Python
- [Gene sets](#gene-sets) — what's bundled and how to access it
- [Analyze command](#the-analyze-command) — full single-sample pipeline
  - [Attribution flow](#attribution-flow-make-it-make-sense) — the 5-step decomposition logic
  - [CLI arguments](#cli-arguments)
  - [Sample quality](#sample-quality-assessment)
  - [Decomposition](#broad-compartment-decomposition)
  - [Purity estimation](#tumor-purity-estimation)
  - [9-point expression ranges](#9-point-expression-ranges)
  - [Output files](#output-files)
- [Python API](#python-api) — expression data, surface proteins, purity, quality, decomposition, plotting
- [Therapy modalities](#therapy-modalities) — ADC, CAR-T, TCR-T, bispecific, radioligand references
- [Cancer-testis antigens](#cancer-testis-antigens-ctas) — curation pipeline and evidence columns
- [Other gene sets](#class-i-mhc-antigen-presentation) — MHC presentation, IFN-gamma, cancer drivers

## Quick start

```bash
pip install pirlygenes
```

### Use gene sets directly

```python
from pirlygenes.gene_sets_cancer import (
    CTA_gene_names,                   # ~257 cancer-testis antigens
    surface_protein_gene_names,       # 2,799 surfaceome genes
    cancer_surfaceome_gene_names,     # 147 tumor-specific surface targets
    therapy_target_gene_names,        # targets by modality: "ADC", "CAR-T", etc.
    pan_cancer_expression,            # ~3,100 genes x 83 columns (50 tissues + 33 cancers)
    cancer_expression,                # expression for one cancer type
    cancer_enriched_genes,            # genes enriched in one cancer type
)
```

### Run full sample analysis

```bash
# Full sample analysis: cancer type, purity, decomposition, targets, embeddings
pirlygenes analyze gene_expression.tsv

# Specify cancer type and output directory
pirlygenes analyze gene_expression.tsv --cancer-type prostate --output-dir results/

# Force a metastatic template with a site hint
pirlygenes analyze gene_expression.tsv --cancer-type COAD --tumor-context met --site-hint liver

# Explicit decomposition template list
pirlygenes analyze gene_expression.tsv --decomposition-templates "solid_primary,met_liver"

# Force-label specific genes in plots
pirlygenes analyze gene_expression.tsv --cancer-type PRAD --label-genes "FOLH1,STEAP1,CD276"

# List all bundled datasets and cancer types
pirlygenes data

# Multi-sample cohort analysis
pirlygenes plot-cancer-cohorts sample1.tsv sample2.tsv sample3.tsv --cancer-type PRAD
```

## Gene sets

All gene sets ship as CSVs in `pirlygenes/data/` and are accessible via `pirlygenes.gene_sets_cancer`. See [docs/gene-sets.md](docs/gene-sets.md) for the full API reference.

| Gene set | Genes | Access | Description |
|----------|------:|--------|-------------|
| Cancer-testis antigens | 257 | `CTA_gene_names()` | Reproductive-restricted, filter-passing and expressed CTAs from CTpedia, CTexploreR, literature. `CTA_filtered_gene_names()` (278) adds never-expressed filter-passers; `CTA_unfiltered_gene_names()` (358) is the full candidate pool from source databases |
| Surface proteins | 2,799 | `surface_protein_gene_names()` | Human surfaceome (SURFY + CSPA); 1,410 mass-spec validated |
| Tumor-specific surface | 147 | `cancer_surfaceome_gene_names()` | TCSA L3-tier surface targets |
| ADC targets | 59 (13 approved) | `therapy_target_gene_id_to_name("ADC")` | Approved + trial ADC antigens |
| CAR-T targets | 2 (2 approved) | `therapy_target_gene_id_to_name("CAR-T")` | Approved CAR-T antigens |
| TCR-T targets | 14 (1 approved) | `therapy_target_gene_id_to_name("TCR-T")` | Approved + trial TCR-T antigens |
| Bispecific / TCE | 11 (11 approved) | `therapy_target_gene_id_to_name("bispecific-antibodies")` | Bispecific T-cell engager targets |
| Multispecific TCE (trials) | 30 | `therapy_target_gene_id_to_name("multispecific-TCE")` | TCE trials across multispecific formats |
| Radioligand targets | 20 | `therapy_target_gene_id_to_name("radioligand")` | RLT target genes |
| Cancer-key-genes panel | 441 (23 cancer types) | `cancer_key_genes_df()` / `cancer_biomarker_genes()` / `cancer_therapy_targets()` | Clinician-relevant biomarker + therapy-target rows per cancer type. SARC is subtype-tiled (LMS, DDLPS, myxoid LPS, synovial, DSRCT, GIST, Ewing); LAML has an APL subtype tile. |
| Cancer drivers | 739 | `cancer-driver-genes.csv` | Recurrently mutated genes (Bailey et al. 2018) |
| Housekeeping genes | 30 | `housekeeping-genes.csv` | Cross-platform normalization reference |
| Mitochondrial genes | 15 | `mitochondrial_gene_names()` | MT-encoded transcripts (quality / FFPE signal) |
| TME markers | 19 | `tme_marker_gene_names()` | Minimal immune + stromal markers for cell-line vs tissue |
| Culture stress | 23 | `culture_stress_gene_names()` | Cell-line adaptation signature |
| Degradation gene pairs | 20 | `degradation_gene_pairs()` | Matched short / long transcript pairs for FFPE length-bias index |
| Pan-cancer expression | 19,784 | `pan_cancer_expression()` | Expression reference: 33 TCGA cancers × 50 HPA normal tissues (HPA nTPM + GDC FPKM). Filtered to the ~3,100 gene-set universe when callers pass `genes=`. |
| TCGA deconvolved tumor-only | 33 codes | `tcga_deconvolved_expression()` | Per-(TCGA code, symbol) tumor-only TPM (median + IQR + N) after running the pirlygenes decomposition on every Xena TOIL TCGA sample (#21). Merged into `pan_cancer_expression()` as `tcga_<CODE>` columns when shipped. |
| Subtype-stratified tumor-only | 7 cohorts × 14 subtypes | `subtype_deconvolved_expression()` | BRCA × PAM50, BeatAML × ELN2017 + APL, TARGET AML / NBL (MYCN) / WT / RT, SCLC (Rudin 2015, ASCL1-dominant) — tumor-only medians for subtype-aware downstream reasoning. |
| Cancer-type registry | 105 codes / 26 families | `cancer_type_registry()` / `cancer_types_in_family()` / `cancer_types_by_tissue()` / `cancer_type_subtypes_of()` | Richer superset of TCGA — 33 TCGA codes, 10 SARC subtype tiles, PAM50 × BRCA, ELN/APL × LAML, non-TCGA heme (CLL, MM, MCL, FL, HL, BL, MDS, MPN, CML, HCL), pediatric (OS, EWS, RMS, NBL, WILMS, RT, MBL, ATRT, RB, HEPB), NET axis (PANNET, MID_NET, LUNG_NET, SCLC + 4 subtype TFs, MEC), and rare (NUTM, ADCC, MTC, CHOR, NPC). Each row carries primary tissue + decomposition template + parent code + expression source. |

These gene sets serve two purposes: **(1)** as standalone curated lists for target selection, enrichment analysis, or annotation, and **(2)** as the reference panels that power the decomposition pipeline — immune and stromal marker genes define the TME signature matrix, housekeeping genes anchor cross-platform normalization, and therapy target sets structure the final report.

## The `analyze` command

The main entry point for single-sample analysis. Takes a gene expression file (CSV, TSV, or Excel with a TPM column) and produces a comprehensive output directory with:

- **Sample context inference** — library prep (poly-A / ribo-depleted / total RNA / exome capture) and preservation (fresh / FFPE / degraded) detected from the expression table; propagated to every downstream step as the base layer of expression expectations
- **Sample quality assessment** — RNA degradation / FFPE detection (transcript-length gene pairs + tissue-matched MT/RP baselines) and cell line / cell culture detection
- **Cancer type identification** — auto-detected or specified, scored against 33 TCGA types
- **Tumor purity estimation** — three methods combined: cancer-type signature genes, ESTIMATE stromal/immune enrichment, and lineage gene calibration
- **Broad-compartment decomposition** — weighted NNLS decomposition of the non-tumor fraction into immune, stromal, and site-specific host components with template-based scoring
- **Purity-adjusted expression** — 9-point tumor expression ranges crossing (low/med/high) TME background with (low/med/high) purity, with % of cancer type median comparison
- **Therapeutic target analysis** — CTAs, ADC/CAR-T/TCR-T/bispecific/radioligand targets, surface proteins
- **Tissue context** — normal tissue similarity scoring
- **PCA/MDS embeddings** — sample positioned among 33 TCGA cancer types
- **Combined PDF** with all figures

See [docs/analyze-command.md](docs/analyze-command.md) for full output file reference.

### Reasoning pipeline: coarse-to-fine with explicit steps

See [docs/reasoning-pipeline.md](docs/reasoning-pipeline.md) for the complete step inventory and information-flow contract. Short version:

```
  Step 0a  Sample context        library prep + preservation + degradation
  Step 0b  Tissue composition    top HPA normals + top TCGA cohorts + cancer-hint
  Step 1   Cancer-type call      top-k TCGA candidates (signature + purity + lineage)
  Step 2   Tumor purity          point + CI + confidence tier
  Step 3   Broad decomposition   NNLS fit of tumor + TME (template-aware)
  Step 4   Therapy-axis state    AR / EMT / hypoxia / IFN / HER2 / ER up/down calls
  Step 5   Tumor-value core      9-point per-gene tumor-TPM
  Step 6   Report synthesis      brief / actionable / summary / analysis / targets / provenance
```

Each step writes a named result into a shared `analysis` dict; downstream steps read but never overwrite. Contracts are explicit — a reader of the final brief can trace any claim back through the step chain. The guiding principle: if a sample is flagged ambiguous at Step 0, that flag propagates through to Step 6; no downstream step silently promotes a soft-confidence call to a confident one.

### Attribution flow: "make it make sense"

`pirlygenes` treats every TPM in a bulk tumor sample as a claim that must be *attributed* to some source — and then explains away as much of it as possible before crediting the tumor. The goal is a conservative, defensible core of tumor-specific expression, not a lift of raw TPM straight into therapeutic target recommendations.

The flow runs in five steps:

1. **Sample context** (runs *first*, before any cancer-type inference). A `SampleContext` is inferred from the expression table alone — what **library prep** produced the data (poly-A capture, ribo-depletion, total RNA, or exome capture) and what **preservation / degradation** state the RNA is in (fresh-frozen, FFPE, partial degradation). This becomes a **base layer of expression expectations**: which genes are expected to be over- or under-represented for *artifactual* reasons, independent of biology. The context is propagated forward and every downstream step reads from it.
2. **Coarse healthy-tissue decomposition.** Broad non-tumor compartments (T cell, B cell, myeloid, fibroblast, endothelial, plus site-specific host tissue for met samples) are fit by weighted NNLS against curated marker panels, anchored to an externally-estimated tumor purity.
3. **Fine TME-subtype / activation-state dissection** (in progress: CAF vs healthy fibroblast, TAM polarisation, exhausted T, tumor endothelium, etc.). Activated-state references refine the coarse compartment call into biologically actionable subsets.
4. **Tumor-value adjustment.** Per gene, the TME and matched-normal contributions are subtracted from the observed TPM before dividing by purity. Genes whose observed signal is dominantly explained by non-tumor compartments are flagged `tme_explainable` rather than credited to the tumor.
5. **Conservative tumor-specific core.** What remains after steps 1–4 is reported as a bounded per-gene tumor-expression range (9-point across low/median/high TME and low/median/high purity), with a low-purity caveat for samples below 20% purity where TME residuals would otherwise be amplified ≥5×.

Each step *adds* to the attribution chain; none replaces earlier steps. The `analyze` report (summary.md, analysis.md, targets.md) surfaces the chain so a reader can see *why* a number is what it is, not just the final value.

Reports and figures are ordered high-level → specific: start with what data we're looking at and how it was prepared (QC), then the coarse cancer-type call and what else is in the sample, then the deeper per-gene / per-target detail.

### CLI arguments

Key decomposition-related options:

| Argument | Description |
|---|---|
| `--cancer-type` | TCGA code or alias (e.g. `PRAD`, `prostate`). If omitted, auto-detected. |
| `--sample-mode` | `auto` (default), `solid` (solid tumor biopsy), `heme` (hematologic malignancy), `pure` (cell line / sorted population) |
| `--tumor-context` | `auto` (default), `primary` (restricts to primary-site templates), `met` (restricts to metastatic templates) |
| `--site-hint` | Metastatic site (e.g. `liver`, `lung`, `brain`, `bone`) — selects the matching met template |
| `--met-site` | Biopsy site for TME background augmentation (`primary`, `lymph_node`, `liver`, `brain`, `lung`, `bone`). Adds the host tissue to the TME reference so tumor-expression estimates aren't inflated by unsubtracted host signal (e.g. CD74 in a lymph-node met). |
| `--decomposition-templates` | Comma-separated explicit template list (e.g. `solid_primary,met_liver`) |
| `--label-genes` | Comma-separated genes to always label in plots |

### Sample quality assessment

Before decomposition, the sample is evaluated for two quality issues that would undermine downstream interpretation. Quality metrics are inferred from the expression matrix alone (no raw reads or BAM files needed).

**RNA degradation / FFPE detection.** Three complementary signals:

1. **Transcript-length gene pairs** — 20 matched pairs of a short gene (<1.2 kb coding) and a long gene (>6.9 kb coding), selected for stable expression ratios (CV < 0.35) across 50 normal tissues and 33 TCGA cancer types. In FFPE/degraded RNA, long transcripts are preferentially fragmented, so the observed/expected ratio drops. The median across pairs is the degradation index (1.0 = normal, < 0.3 = severe).
2. **Mitochondrial fraction** — MT-encoded transcripts are short and abundant; their fraction rises in degraded samples. Compared against the matched normal tissue baseline (kidney at 56% MT is biological, not degradation).
3. **Ribosomal protein fraction** — short RP transcripts survive degradation, so their share of non-MT expression rises. Also compared to tissue-matched baselines.

Degradation is called as *moderate* or *severe* when multiple signals agree. The call flows into downstream analysis: decomposition warnings, purity caveats, and report narrative.

**Cell line / cell culture detection.** Two signals:

1. **TME absence** — mean TPM of immune + stromal markers (CD3D, CD68, COL1A1, VWF, etc.) near zero
2. **Culture stress signature** — elevated heat-shock, glycolysis, proliferation, ER stress, and glutamine metabolism genes (Yu et al., *Nature Communications* 2019)

Both absent + stress high → likely cell line. TME absent but stress normal → could be immune-desert tumor or sorted population.

### Broad-compartment decomposition

After purity anchoring, the non-tumor fraction is decomposed across broad, reference-supported compartments using weighted non-negative least squares (NNLS) on component-enriched marker genes.

**Algorithm.**

1. **Template selection** — the sample mode determines which templates are evaluated:
    - `solid` → solid primary + 9 metastatic site templates (liver, lung, brain, bone, lymph node, adrenal, peritoneal, skin, soft tissue)
    - `heme` → nodal, blood, marrow
    - `pure` → pure population (just tumor, no TME)
2. **Signature matrix** — for each template, build a reference matrix from HPA single-cell profiles (immune/stromal components) and bulk tissue references (site-specific components). Site-specific components use a hierarchical best-match approach: for brain mets, the CNS category compares the sample against cerebral cortex, cerebellum, hippocampus, etc. and picks the best-matching tissue as the reference column. This avoids HK-normalization distortion seen in HPA single-cell neural references (astrocyte HK median ~20 nTPM vs ~350 nTPM for immune/stroma).
3. **Marker selection** — for each component, pick 12 genes with highest specificity × expression in the HK-normalized signature matrix. Component-specific curated markers (e.g., CD3D/CD3E for T cells, IGKC/JCHAIN for plasma) are always included.
4. **Weighted NNLS fit** — solve for component fractions summing to 1, with:
    - `1/b` row weighting (proportional error, not absolute — prevents Ig genes from dominating the residual in plasma-heavy samples)
    - Soft sum-to-one penalty and light ridge
    - External purity anchor (tumor fraction fixed from the purity estimate)
5. **Template scoring** — each (cancer_type, template) hypothesis gets:
    - `fit_score = 1 / (1 + residual)` — how well the marker expression is explained
    - `template_factor` — site_factor × extra_component_factor (penalizes met templates with weak site evidence or underused site-specific components)
    - Final score = `fit_score × (base + gain × cancer_support) × template_factor`
6. **Ranking** — hypotheses are sorted by combined score. Top-6 are reported; the best is used for gene-level attribution.

**Per-gene attribution.** For each expressed gene, the decomposition computes:
- Component-wise TPM contribution: `(1 - purity) × mix[comp] × signature[gene, comp]`
- Residual tumor TPM: `observed - sum(TME contributions)`
- Overexplained TPM: when the TME model predicts more than observed (flagged in warnings)

**Quality caveats.** If RNA degradation is detected, decomposition adds a warning to the result and the reports flag that component fractions involving long-transcript markers (e.g., fibroblast via COL6A3, endothelial via VWF) may be systematically underestimated.

### Tumor purity estimation

Purity is estimated by combining three independent methods:

1. **Signature genes** — 30 cancer-type-specific genes selected by z-score across TCGA; per-gene HK-ratio compared to TCGA reference (calibrated for cohort purity)
2. **ESTIMATE enrichment** — stromal and immune gene set enrichment ratios vs TCGA
3. **Lineage gene calibration** — cancer-type-specific lineage markers (e.g. STEAP1/2, KLK3 for prostate) that anchor purity to known biology. Uses upper-half median to resist de-differentiation artifacts in metastatic samples

All comparisons are HK-normalized (fold-over-housekeeping) for cross-platform robustness (TPM, FPKM, nTPM).

### 9-point expression ranges

For each gene, tumor-specific expression is estimated as:

```
tumor = (observed - (1-purity) * tme_background) / purity
```

The TME background is the median expression across curated immune + stromal reference tissues (bone marrow, lymph node, spleen, thymus, tonsil, appendix, smooth muscle, skeletal muscle, heart muscle, adipose tissue), HK-normalized.

Uncertainty is captured by a 3x3 grid: (25th/50th/75th percentile TME) x (lower/estimate/upper purity). The median of these 9 estimates is the point estimate; the full range is shown in the strip plot.

Each gene's estimate is compared to the purity-adjusted TCGA median for the matched cancer type (e.g. "STEAP1 = 103% of PRAD median"), providing a biologically meaningful reference point.

### Output files

Every `analyze` run produces a directory with these files (prefixed by the input basename):

**Reports** (markdown):

| File | Description |
|---|---|
| `*-summary.md` | One-paragraph natural language summary — cancer type, purity, key findings, quality warnings |
| `*-analysis.md` | Structured analysis — sample quality, candidate trace, purity components, decomposition hypotheses, background signatures, embedding features |
| `*-targets.md` | Therapeutic targets — tumor context, therapy landscape at a glance, CTAs, surface proteins, intracellular targets, safety context |

**Structured data** (TSV / JSON):

| File | Description |
|---|---|
| `*-analysis-parameters.json` | All free parameters, selected sample mode, embedding methods, sample quality flags |
| `*-cancer-candidates.tsv` | Candidate cancer-type support trace (family, signature, purity, lineage scores) |
| `*-decomposition-hypotheses.tsv` | Ranked decomposition hypotheses with fit quality, site score, template factor |
| `*-decomposition-components.tsv` | Component-level fit for best decomposition (fraction, marker score, top markers) |
| `*-decomposition-markers.tsv` | Marker-gene evidence for best decomposition (specificity, reference HK, observed TPM, sample/ref ratio) |
| `*-decomposition-gene-attribution.tsv` | Per-gene TME vs tumor attribution (how much of each gene's observed TPM is explained by each component) |
| `*-tumor-expression-ranges.tsv` | Purity-adjusted tumor-expression ranges with TCGA context (9-point grid, TCGA percentile) |

**Figures** (PNG + combined PDF):

Prefer the standalone decomposition figures when reviewing or sharing a case. They replace the crowded legacy composite by splitting composition, component breakdown, and candidate comparison into separate PNGs.

| File | Description |
|---|---|
| `*-sample-context.png` | Step 1 diagnostic: library prep + preservation inference with thresholds used for the call |
| `*-degradation-index.png` | Gene-pair scatter: expected vs observed long/short ratios, diagonal = no degradation |
| `*-sample-summary.png` | Quick overview: cancer type, purity, background signatures |
| `*-decomposition-composition.png` | Standalone composition bar for the best hypothesis |
| `*-decomposition-components.png` | Standalone TME component breakdown for the best hypothesis |
| `*-decomposition-candidates.png` | Standalone per-candidate composition comparison |
| `*-purity.png` | Tumor purity estimation detail |
| `*-immune.png`, `*-tumor.png`, `*-antigens.png`, `*-treatments.png` | Gene expression strip plots by category |
| `*-target-safety.png`, `*-purity-targets.png`, `*-purity-ctas.png`, `*-purity-surface.png` | Therapy target expression with normal tissue context |
| `*-pca-hierarchy.png`, `*-mds-hierarchy.png` | Embeddings in hierarchical support space |
| `*-pca-tme.png`, `*-mds-tme.png` | Embeddings in TME-low gene space |
| `*-cancer-types-genes.png`, `*-cancer-types-disjoint.png` | Cancer-type gene signature heatmaps |
| `*-all-figures.pdf` | All figures combined into a single PDF |

## Python API

### Pan-cancer expression data

Expression data for ~3,100 therapy-relevant genes across 50 normal tissues (HPA v23 consensus nTPM) and 33 TCGA cancer types (median FPKM/TPM). All data ships with the package — no downloads needed.

#### Cancer types

| Code | Cancer type | Aliases |
|------|------------|---------|
| ACC | Adrenocortical Carcinoma | adrenocortical, adrenal |
| BLCA | Bladder Urothelial Carcinoma | bladder |
| BRCA | Breast Invasive Carcinoma | breast |
| CESC | Cervical Squamous Cell Carcinoma | cervical, cervix |
| CHOL | Cholangiocarcinoma | cholangiocarcinoma, bile_duct |
| COAD | Colon Adenocarcinoma | colon, colorectal |
| DLBC | Diffuse Large B-Cell Lymphoma | dlbcl, lymphoma |
| ESCA | Esophageal Carcinoma | esophageal, esophagus |
| GBM | Glioblastoma Multiforme | glioblastoma, gbm |
| HNSC | Head and Neck Squamous Cell Carcinoma | head_neck, hnscc |
| KICH | Kidney Chromophobe | kidney_chromophobe |
| KIRC | Kidney Renal Clear Cell Carcinoma | kidney, kidney_clear |
| KIRP | Kidney Renal Papillary Cell Carcinoma | kidney_papillary |
| LAML | Acute Myeloid Leukemia | leukemia, aml |
| LGG | Brain Lower Grade Glioma | glioma, lgg, low_grade_glioma |
| LIHC | Liver Hepatocellular Carcinoma | liver |
| LUAD | Lung Adenocarcinoma | lung_adeno |
| LUSC | Lung Squamous Cell Carcinoma | lung_squamous |
| MESO | Mesothelioma | mesothelioma |
| OV | Ovarian Serous Cystadenocarcinoma | ovarian, ovary |
| PAAD | Pancreatic Adenocarcinoma | pancreatic, pancreas |
| PCPG | Pheochromocytoma and Paraganglioma | pheochromocytoma, paraganglioma |
| PRAD | Prostate Adenocarcinoma | prostate |
| READ | Rectum Adenocarcinoma | rectal |
| SARC | Sarcoma | sarcoma |
| SKCM | Skin Cutaneous Melanoma | melanoma, skin |
| STAD | Stomach Adenocarcinoma | stomach, gastric |
| TGCT | Testicular Germ Cell Tumor | testicular, testis |
| THCA | Thyroid Carcinoma | thyroid |
| THYM | Thymoma | thymoma |
| UCEC | Uterine Corpus Endometrial Carcinoma | endometrial, uterine |
| UCS | Uterine Carcinosarcoma | uterine_carcinosarcoma |
| UVM | Uveal Melanoma | uveal_melanoma |

```python
from pirlygenes.gene_sets_cancer import (
    pan_cancer_expression,      # full expression matrix
    cancer_types,               # list of available TCGA codes
    cancer_expression,          # expression for one cancer type
    cancer_enriched_genes,      # genes enriched in one cancer type
    cancer_surfaceome_evidence, # tumor-specific surface targets (TCSA)
    surface_protein_evidence,   # all surface proteins (SURFY/CSPA)
)

# Full matrix: ~3,100 genes x 83 columns (50 tissues + 33 cancers)
df = pan_cancer_expression()

# Housekeeping-normalized, log-scaled (for heatmaps)
df = pan_cancer_expression(normalize="housekeeping", log_transform=True)

# Expression in prostate cancer (housekeeping-normalized)
df = cancer_expression("prostate")

# Genes enriched in prostate vs other cancers (>3x fold-change)
df = cancer_enriched_genes("prostate", min_fold=3.0)

# Specific genes across all cancers
df = pan_cancer_expression(genes=["EGFR", "ERBB2", "MSLN", "FOLR1"])
```

### Surface proteins

```python
from pirlygenes.gene_sets_cancer import (
    surface_protein_gene_names,       # 2,799 surfaceome genes
    surface_protein_gene_ids,
    cancer_surfaceome_gene_names,     # 147 tumor-specific surface targets
    cancer_surfaceome_evidence,       # with druggability + cancer types
)

# CSPA mass-spec validated only
validated = surface_protein_gene_names(validated_only=True)  # 1,410 genes
```

### Tumor purity estimation

```python
from pirlygenes.tumor_purity import estimate_tumor_purity

result = estimate_tumor_purity(df_expr, cancer_type="PRAD")
print(result["overall_estimate"])  # e.g. 0.10
print(result["components"]["lineage"]["per_gene"])  # per-gene purity estimates
```

### Sample quality assessment

```python
from pirlygenes.sample_quality import assess_sample_quality

# Optionally pass tissue_scores from analyze_sample for tissue-matched baselines
quality = assess_sample_quality(df_expr, tissue_scores=[("prostate", 0.9, 20)])

print(quality["degradation"]["level"])         # "normal" | "mild" | "moderate" | "severe"
print(quality["degradation"]["long_short_ratio"])  # transcript-length pair index
print(quality["culture"]["level"])             # "normal" | "tme_absent" | "possible_cell_line" | "likely_cell_line"
print(quality["flags"])                        # human-readable warnings
```

### Broad-compartment decomposition

```python
from pirlygenes.decomposition import decompose_sample

results = decompose_sample(
    df_expr,
    cancer_types=["PRAD", "BRCA"],          # optional — ranked automatically if omitted
    sample_mode="auto",                      # "auto" | "solid" | "heme" | "pure"
    tumor_context="auto",                    # "auto" | "primary" | "met"
    site_hint=None,                          # e.g. "liver", "brain", "bone"
    templates=None,                          # or explicit list: ["solid_primary", "met_liver"]
    top_k=6,
)

best = results[0]
print(best.cancer_type, best.template, best.score)
print(best.fractions)                        # {"tumor": 0.65, "T_cell": 0.08, "fibroblast": 0.12, ...}
print(best.component_trace)                  # DataFrame: per-component fractions + top markers
print(best.gene_attribution)                 # DataFrame: per-gene TME vs tumor TPM split
print(best.warnings)                         # warnings for overexplained genes, weak site support, etc.
```

### Purity-adjusted expression

```python
from pirlygenes.plot import estimate_tumor_expression_ranges

ranges = estimate_tumor_expression_ranges(df_expr, "PRAD", purity_result)
# DataFrame with est_1..est_9 (9-point grid), median_est, pct_cancer_median
```

### Plotting

```python
from pirlygenes import plot_gene_expression, plot_sample_vs_cancer

# Strip plot: gene expression by category
plot_gene_expression(df_expr, save_to_filename="summary.png")

# Scatter: sample vs cancer cohort (PDF + per-category PNGs)
plot_sample_vs_cancer(df_expr, cancer_type="prostate", save_to_filename="vs_prad.pdf")
```

---

## Therapy modalities

Curated target lists and literature references for each therapy modality. Target gene sets are accessible via `therapy_target_gene_names(key)` (see [Gene sets](#gene-sets) above).

### TCR-T 

#### Clinical trials

Last updated: September 17th, 2024

Sources: 

- [Toward a comprehensive solution for treating solid tumors using T-cell receptor therapy: A review](https://www.sciencedirect.com/science/article/pii/S0959804924008803#sec0055)

### CAR-T

#### Approved therapies

Last updated: September 17th, 2024

Sources: 

- [CAR-T: What Is Next? ](https://www.mdpi.com/2072-6694/15/3/663)

### Multi-specific antibodies and T-cell engagers

#### Clinical trials

Last updated: September 11th, 2024

Sources:

- [Progresses of T-cell-engaging bispecific antibodies in treatment of solid tumors](https://www.sciencedirect.com/science/article/abs/pii/S1567576924011305)

### Antibody-drug conjugates (ADCs)

#### Approved

Last updated: September 19th, 2024

Sources:

- [Development of antibody-drug conjugates in cancer: Overview and prospects](https://onlinelibrary.wiley.com/doi/full/10.1002/cac2.12517)


#### Clinical trials 
Last updated: September 11th, 2024

Sources:

- [Pan-cancer analysis of antibody-drug conjugate targets and putative predictors of treatment response](<https://www.ejcancer.com/article/S0959-8049(23)00681-0/fulltext>)

### Radioligand therapies (RLTs)

#### Current target list

Last updated: February 11th, 2026

Sources:

- [Radioligand therapy in precision oncology: opportunities and challenges](https://www.mdpi.com/2072-6694/17/21/3412)
- [FDA approves Pluvicto for metastatic castration-resistant prostate cancer](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-pluvicto-metastatic-castration-resistant-prostate-cancer)
- [FDA approves Lutetium Lu 177 dotatate for gastroenteropancreatic neuroendocrine tumors](https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-lutetium-lu-177-dotatate-gastroenteropancreatic-neuroendocrine-tumors)
- [Emerging molecular targets and agents in radioligand therapy for solid tumors](https://pmc.ncbi.nlm.nih.gov/articles/PMC12239088/)
- [Early evidence for anti-CD20 targeted alpha-radiation approaches in B-cell malignancies](https://pmc.ncbi.nlm.nih.gov/articles/PMC10964383/)

Methodology:

- `pirlygenes/data/radioligand-targets.csv` is a curated target-level list (gene targets, Ensembl IDs, and target status buckets) intended to power gene-set visualization while trial-level `v1.4.0` curation is in progress.

CLI plotting notes:

- Treatment plots now include a `Radio` category label (capitalized consistently with other treatment labels).
- Use `--label-genes` to force annotation of genes that should always be text-labeled, for example: `--label-genes FAP,CD276`.
- PNG output defaults are larger/higher resolution (`--plot-height 12.0`, `--plot-aspect 1.4`, `--output-dpi 300`), and can be overridden from CLI.

## Cancer-testis antigens (CTAs)

Last updated: March 23rd, 2026

### Quick start

```python
from pirlygenes.gene_sets_cancer import (
    CTA_gene_names,              # recommended: filtered, reproductive-restricted CTAs
    CTA_gene_ids,                # same, as Ensembl gene IDs
    CTA_unfiltered_gene_names,   # full superset from all source databases
    CTA_unfiltered_gene_ids,     # same, as Ensembl gene IDs
    CTA_evidence,                # full DataFrame with all evidence columns
)

# Default: expressed, reproductive-restricted CTAs (~257 genes)
cta_genes = CTA_gene_names()

# Full unfiltered superset from all sources (~358 genes)
all_ctas = CTA_unfiltered_gene_names()

# Partition ALL protein-coding genes into CTA / never-expressed / non-CTA
from pirlygenes.gene_sets_cancer import CTA_partition_gene_ids
p = CTA_partition_gene_ids()   # p.cta, p.cta_never_expressed, p.non_cta

# Evidence table with per-gene HPA tissue restriction data
df = CTA_evidence()
```

### Pipeline overview

The CTA gene set is built as an unbiased union of genes from multiple CT antigen databases and literature sources, then systematically filtered using Human Protein Atlas tissue expression data.

**Step 1: Collect** — union of protein-coding CT genes from multiple source databases (358 genes):

| Source | Genes | Reference |
|---|---|---|
| [CTpedia](http://www.cta.lncc.br/) | 167 | [Almeida et al. 2009](https://doi.org/10.1093/nar/gkn673), *NAR* |
| [CTexploreR/CTdata](https://www.bioconductor.org/packages/release/bioc/html/CTexploreR.html) | 62 new | [Loriot et al. 2025](https://doi.org/10.1371/journal.pgen.1011734), *PLOS Genetics* |
| Protein-level CT genes (136 total, 46 overlap) | 89 new | [da Silva et al. 2017](https://doi.org/10.18632/oncotarget.21715), *Oncotarget* |
| EWSR1-FLI1 CT gene binding sites | 12 | [Gallegos et al. 2019](https://doi.org/10.1128/MCB.00138-19), *Mol Cell Biol* |
| Meiosis, piRNA, spermatogenesis genes | 28 | Multiple sources (see [docs](docs/cta-curation.md#literature-scan-28-genes)) |

Each gene is tracked with a `source_databases` column indicating which databases include it (CTpedia, CTexploreR_CT, CTexploreR_CTP, daSilva2017, daSilva2017_protein). Only protein-coding genes (Ensembl biotype) are included. Genes with outdated HGNC symbols are renamed to current symbols with old names kept as aliases.

**Step 2: Annotate** — each gene is scored against [Human Protein Atlas](https://www.proteinatlas.org/) v23 tissue expression:

- **RNA**: [HPA RNA tissue consensus](https://www.proteinatlas.org/about/download) (`rna_tissue_consensus.tsv`) — normalized transcripts per million (nTPM) across 50 normal tissues
- **Protein**: [HPA normal tissue IHC](https://www.proteinatlas.org/about/download) (`normal_tissue.tsv`) — immunohistochemistry detection levels (Not detected / Low / Medium / High) across 63 tissues with antibody reliability scores (Enhanced / Supported / Approved / Uncertain)

**Step 3: Filter** — protein-coding + tiered thresholds based on protein antibody confidence (278 of 358 pass):

| Protein evidence | Deflated RNA threshold |
|---|---|
| Enhanced (orthogonal validation) | ≥ 80% |
| Supported (consistent characterization) | ≥ 90% |
| Approved (basic validation) | ≥ 95% |
| Uncertain or no protein data | ≥ 99% |

Genes with protein detected in non-reproductive tissues always fail. Thymus is excluded from all restriction checks (AIRE-driven mTEC expression is expected for CTAs).

### Gene set counts

| Function | Description | Count |
|---|---|---|
| `CTA_gene_names()` | **Recommended default.** Expressed, reproductive-restricted CTAs | ~257 |
| `CTA_never_expressed_gene_names()` | CTAs from databases but no HPA expression (max nTPM < 2, no protein) | ~21 |
| `CTA_filtered_gene_names()` | All filter-passing CTAs (= expressed + never_expressed) | ~278 |
| `CTA_excluded_gene_names()` | CTAs that fail filter (somatic expression) | ~80 |
| `CTA_unfiltered_gene_names()` | Full superset from all source databases | 358 |
| `CTA_evidence()` | Full DataFrame with all evidence columns | 358 rows |
| `CTA_partition_gene_ids()` | Partition all protein-coding genes (dataclass with `.cta`, `.cta_never_expressed`, `.non_cta` sets) | ~20k |
| `CTA_partition_gene_names()` | Same, as gene symbols | ~20k |
| `CTA_partition_dataframes()` | Same, as DataFrames with evidence columns | ~20k |

### Evidence columns

Each gene in `cancer-testis-antigens.csv` carries identity and HPA-derived evidence:

| Column | Description |
|---|---|
| `Ensembl_Gene_ID` | Ensembl gene ID (validated against release 112) |
| `source_databases` | Semicolon-separated list of source databases (CTpedia, CTexploreR_CT, CTexploreR_CTP, daSilva2017) |
| `biotype` | Ensembl gene biotype (must be `protein_coding` to pass filter) |
| `Canonical_Transcript_ID` | Longest protein-coding transcript (Ensembl 112) |
| `protein_reproductive` | IHC detected only in {testis, ovary, placenta} (excl. thymus), or `"no data"` |
| `protein_thymus` | IHC detected in thymus |
| `protein_reliability` | Best HPA antibody reliability: Enhanced / Supported / Approved / Uncertain / `"no data"` |
| `protein_strict_expression` | Semicolon-separated tissues with IHC detection (excl. thymus) |
| `rna_reproductive` | All tissues with ≥1 nTPM (excl. thymus) are in {testis, ovary, placenta} |
| `rna_thymus` | Thymus nTPM ≥ 1 |
| `rna_reproductive_frac` | Fraction of total nTPM (excl. thymus) in core reproductive tissues |
| `rna_deflated_reproductive_frac` | `(1 + Σ_repro max(0, nTPM−1)) / (1 + Σ_all max(0, nTPM−1))` |
| `rna_deflated_reproductive_and_thymus_frac` | Same but thymus added to reproductive numerator |
| `rna_80/90/95/99_pct_filter` | Whether deflated reproductive fraction ≥ threshold |
| `filtered` | Final inclusion flag (see tiered thresholds above) |

For full details on the curation process, evidence columns, and filter logic, see [docs/cta-curation.md](docs/cta-curation.md).

### Deflated RNA metric

The deflated metric `max(0, nTPM − 1)` per tissue suppresses low-level basal transcription noise before computing the reproductive fraction. A `+1` pseudocount on numerator and denominator prevents 0/0 for very-low-expression genes. Example: CTCFL/BORIS has raw reproductive fraction 54% (diluted by sub-1 nTPM noise across ~40 tissues) but deflated fraction 100% (only testis has ≥1 nTPM).

## Class I MHC antigen presentation

Last updated: July 21st, 2018

Sources:

- [Frequent HLA class I alterations in human prostate cancer: molecular mechanisms and clinical relevance](https://link.springer.com/article/10.1007/s00262-015-1774-5):
  - LMP2/7
  - peptide transporters TAP1/2
  - chaperones calreticulin, calnexin, ERP57, and tapasin
  - IFR-1 and NLRC5
- [Expression of Antigen Processing and Presenting Molecules in Brain Metastasis of Breast Cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3365630/)
  - "β2-microgloblin, transporter associated with antigen processing (TAP) 1, TAP2 and calnexin are down-regulated in brain lesions compared with unpaired breast lesions"
- [NLRC5/MHC class I transactivator is a target for immune evasion in cancer](http://www.pnas.org/content/early/2016/05/05/1602069113.short)
- [TAPBPR: a new player in the MHC class I presentation pathway](https://www.ncbi.nlm.nih.gov/pubmed/25720504)

## Interferon-gamma response

Last updated: July 21st, 2018

Sources:

- [Interferon Receptor Signaling Pathways Regulating PD-L1 and PD-L2 Expression](https://www.sciencedirect.com/science/article/pii/S2211124717305259)
  - "JAK1/JAK2-STAT1/STAT2/STAT3-IRF1 axis primarily regulates PD-L1 expression, with IRF1 binding to its promoter"
  - "PD-L2 responded equally to interferon beta and gamma and is regulated through both IRF1 and STAT3, which bind to the PD-L2 promoter"
  - "the suppressor of cytokine signaling protein family (SOCS; mostly SOCS1 and SOCS3) are involved in negative feedback regulation of cytokines that signal mainly through JAK2 binding, thereby modulating the activity of both STAT1 and STAT3"
- [Mutations Associated with Acquired Resistance to PD-1 Blockade in Melanoma](https://www.nejm.org/doi/full/10.1056/NEJMoa1604958)
  - "resistance-associated loss-of-function mutations in the genes encoding interferon-receptor–associated Janus kinase 1 (JAK1) or Janus kinase 2 (JAK2), concurrent with deletion of the wild-type allele"
- [SOCS, inflammation, and cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3772102/)
  - "Abnormal expression of SOCS1 and SOCS3 in cancer cells has been reported in human carcinoma associated with dysregulation of signals from cytokine receptors"

## Recurrently mutated cancer genes

Last updated: July 21st, 2018

Cancer genes and recurrent mutations extract from [Comprehensive Characterization of Cancer Driver Genes and Mutations](<https://www.cell.com/cell/fulltext/S0092-8674(18)30237-X>).

Genes extracted from Table S1 into `cancer-driver-genes.csv`. Mutations extracted from Table S4 into `cancer-driver-variants.csv`.

Both datasets were annotated with Ensembl IDs using Ensembl release 92.
