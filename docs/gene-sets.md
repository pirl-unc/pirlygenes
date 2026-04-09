# Available Gene Sets and Lists

PIRLy Genes provides curated gene sets for cancer biology analysis.
All are accessible via `pirlygenes.gene_sets_cancer`.

## Therapy Target Gene Sets

Each therapy type has trial and approved variants. Access via
`therapy_target_gene_id_to_name(therapy_type)` or
`therapy_target_gene_names(therapy_type)` /
`therapy_target_gene_ids(therapy_type)`.

| Therapy Type | Key | Description |
|-------------|-----|-------------|
| ADC | `"ADC"`, `"ADC-approved"`, `"ADC-trials"` | Antibody-drug conjugates |
| CAR-T | `"CAR-T"`, `"CAR-T-approved"` | Chimeric antigen receptor T cells |
| TCR-T | `"TCR-T"`, `"TCR-T-approved"`, `"TCR-T-trials"` | T-cell receptor therapy |
| Bispecific antibodies | `"bispecific-antibodies"`, `"bispecific-antibodies-approved"` | Bispecific T-cell engagers |
| Radioligand | `"radioligand"` | Radioligand therapy |
| Multispecific TCE | `"multispecific-TCE"` | Multispecific T-cell engagers |

## Cancer-Testis Antigens (CTAs)

CTAs are genes normally expressed only in germline tissues (testis,
placenta) but aberrantly activated in tumors. They are prime
vaccination and immunotherapy targets.

| Function | Returns |
|----------|---------|
| `CTA_gene_names()` | Filtered + expressed CTA gene symbols |
| `CTA_gene_ids()` | Filtered + expressed CTA Ensembl IDs |
| `CTA_gene_id_to_name()` | `{ensembl_id: symbol}` mapping |
| `CTA_filtered_gene_names()` | Filtered CTAs (includes never-expressed) |
| `CTA_unfiltered_gene_names()` | All CTAs before filtering |
| `CTA_excluded_gene_names()` | CTAs that failed quality filters |
| `CTA_never_expressed_gene_names()` | CTAs with no detectable expression |
| `CTA_evidence()` | DataFrame with expression evidence per CTA |
| `CTA_partition_gene_ids()` | `CTAPartitionSets(cta, cta_never_expressed, non_cta)` |
| `CTA_partition_gene_names()` | Same as above, using gene symbols |
| `CTA_partition_dataframes()` | `CTAPartitionDataFrames` with full evidence |

## Surface Proteins

| Function | Returns |
|----------|---------|
| `surface_protein_gene_names()` | Human surfaceome gene symbols |
| `surface_protein_gene_ids()` | Human surfaceome Ensembl IDs |
| `cancer_surfaceome_gene_id_to_name()` | Tumor-specific surface proteins (TCSA, L3-tier) |

## pMHC and Surface TCE Targets

| Function | Returns |
|----------|---------|
| `pMHC_TCE_target_gene_id_to_name()` | pMHC-targeting T-cell engager targets |
| `surface_TCE_target_gene_id_to_name()` | Surface-targeting T-cell engager targets |

## Pan-Cancer Expression Reference

`pan_cancer_expression()` returns a DataFrame with median expression
across 33 TCGA cancer types (FPKM) and 50 HPA normal tissues (nTPM).

```python
from pirlygenes.gene_sets_cancer import pan_cancer_expression

ref = pan_cancer_expression()
# Columns: Ensembl_Gene_ID, Symbol, FPKM_ACC, FPKM_BLCA, ..., nTPM_adipose_tissue, nTPM_adrenal_gland, ...
```

Supports normalization:
- `pan_cancer_expression(normalize="housekeeping")` — fold over housekeeping median
- `pan_cancer_expression(normalize="percentile")` — percentile ranks
- `pan_cancer_expression(log2=True)` — log2 transform

## Embedding Gene Sets

The bottleneck gene selection (`_select_embedding_genes_bottleneck()`)
produces ~158 genes for PCA/MDS visualization. See
[embedding-gene-selection.md](embedding-gene-selection.md) for the
methodology and evaluation.

## Cancer Type Identifiers

33 TCGA cancer types are supported. Use `cancer_types()` for the full
list or `CANCER_TYPE_NAMES` for `{code: full_name}` mapping.

Common codes: ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM,
HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV,
PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC,
UCS, UVM.
