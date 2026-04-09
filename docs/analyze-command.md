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
    --output-image-prefix out \
    --cancer-type PRAD \          # optional, auto-detected if omitted
    --label-genes "FOLH1,PSMA" \  # genes to always label in plots
    --output-dpi 200
```

## Output Files

### Text Reports

| File | Contents |
|------|----------|
| `{prefix}-summary.md` | One-paragraph natural language summary |
| `{prefix}-analysis.md` | Structured analysis: cancer type, purity, MHC, tissue context, embedding genes |
| `{prefix}-targets.md` | Therapeutic target analysis with purity-adjusted expression |

### Figures (in `figures/` subdir)

| Figure | Description |
|--------|-------------|
| `{prefix}-summary.png` | Sample overview: cancer type, purity, tissue context |
| `{prefix}-purity.png` | Tumor purity estimation detail |
| `{prefix}-immune.png` | Immune gene expression strip plot |
| `{prefix}-tumor.png` | Tumor marker gene expression |
| `{prefix}-antigens.png` | Antigen gene expression |
| `{prefix}-treatments.png` | Therapy target gene expression |
| `{prefix}-target-safety.png` | Therapy target normal tissue expression |
| `{prefix}-purity-adjusted.png` | Purity-adjusted expression by target category |
| `{prefix}-cancer-types-genes.png` | Gene set heatmap vs TCGA cancer types |
| `{prefix}-cancer-types-disjoint.png` | Disjoint gene counts per cancer type |
| `{prefix}-pca-bottleneck.png` | PCA embedding among TCGA cancer types |
| `{prefix}-mds-bottleneck.png` | MDS embedding among TCGA cancer types |

## Purity-Adjusted Expression

For each gene, the observed TPM is corrected for TME contamination:

```
tumor_expr = (observed_TPM - (1 - purity) * tme_reference) / purity
```

Where `tme_reference` is the mean expression across immune and stromal
tissues in the HPA normal tissue atlas. This estimate is then compared
to the TCGA distribution for the inferred cancer type.

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
