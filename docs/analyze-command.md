# The `analyze` Command

`pirlygenes analyze` is the main CLI entry point for personalized
cancer sample analysis. It takes a gene expression file (CSV, TSV,
or Excel) and produces:

- Figures (moved to `figures/` subdir)
- A combined PDF (`{prefix}-all-figures.pdf`)
- Text reports (`{prefix}-summary.md`, `{prefix}-analysis.md`, `{prefix}-targets.md`)

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
| `{prefix}-summary.md` | One-paragraph natural language summary |
| `{prefix}-analysis.md` | Structured analysis: cancer type, purity (with lineage gene calibration), MHC, tissue context, embedding genes |
| `{prefix}-targets.md` | Therapeutic target analysis with purity-adjusted expression |
| `README.md` | Output directory index |

### Figures (in `figures/` subdir)

| Figure | Description |
|--------|-------------|
| `{prefix}-summary.png` | Sample overview: cancer type, purity, tissue context |
| `{prefix}-purity.png` | Tumor purity estimation detail |
| `{prefix}-immune.png` | Immune gene expression strip plot |
| `{prefix}-tumor.png` | Tumor marker gene expression |
| `{prefix}-antigens.png` | Antigen gene expression (CTAs + surface) |
| `{prefix}-treatments.png` | Therapy target gene expression |
| `{prefix}-target-safety.png` | Therapy target normal tissue expression |
| `{prefix}-target-tissues.pdf` | Per-gene tissue expression heatmaps |
| `{prefix}-purity-adjusted.png` | 9-point tumor expression ranges with % of cancer type median |
| `{prefix}-cancer-types-genes.png` | Gene set heatmap vs TCGA cancer types |
| `{prefix}-cancer-types-disjoint.png` | Disjoint gene counts per cancer type |
| `{prefix}-pca-bottleneck.png` | PCA embedding (bottleneck gene set) |
| `{prefix}-mds-bottleneck.png` | MDS embedding (bottleneck gene set) |
| `{prefix}-pca-tme.png` | PCA embedding (TME-low gene set — preferred at low purity) |
| `{prefix}-mds-tme.png` | MDS embedding (TME-low gene set — preferred at low purity) |
| `{prefix}-vs-cancer/` | Per-category scatter plots: sample vs TCGA cancer type |
| `{prefix}-vs-cancer.pdf` | Combined scatter plot PDF |
| `{prefix}-all-figures.pdf` | All figures in one PDF |

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

The target report (`targets.md`) categorizes genes as:

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
