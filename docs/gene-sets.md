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
across 33 TCGA cancer types (raw FPKM plus deterministic TPM companions)
and 50 HPA normal tissues (nTPM).

By default, `normalize="tpm_clean"`: the accessor preserves raw TCGA
`*_FPKM`, adds derived TCGA `*_TPM`, preserves HPA `*_nTPM`, cleans
TPM-scale analysis columns, and adds the clean values as
`*_TPM_clean` / `*_nTPM_clean`.

```python
from pirlygenes.expression import pan_cancer_expression

ref = pan_cancer_expression()
# Columns: Ensembl_Gene_ID, Symbol, adipose_tissue_nTPM, ..., ACC_FPKM, ..., ACC_TPM, ...
```

Supports normalization with `None`, a string, or a list of strings:
- `pan_cancer_expression(normalize=None)` — raw/provenance view with
  `*_FPKM` and `*_nTPM` columns unchanged, plus derived `*_TPM` analysis
  columns
- `pan_cancer_expression(normalize="tpm")` or `normalize="TPM"` — add
  deterministic `*_TPM` companions derived from `*_FPKM`
- `pan_cancer_expression(normalize="tpm_log1p")` — add natural-log
  `*_TPM_log1p` and `*_nTPM_log1p` analysis columns
- `pan_cancer_expression(normalize="tpm_clean")` — TPM scale plus zero
  mitochondrial, NUMT-like, rRNA-like, and MALAT1/NEAT1 rows, then pin each
  clean analysis column to sum to 1e6. Base `*_nTPM`, `*_TPM`, and
  `*_FPKM` columns remain unchanged; clean values are added as
  `*_nTPM_clean` and `*_TPM_clean`.
- `pan_cancer_expression(normalize="tpm_clean_log1p")` — add clean
  TPM/nTPM columns, then add natural-log `*_TPM_clean_log1p` and
  `*_nTPM_clean_log1p` columns
- `pan_cancer_expression(normalize="hk")` — fold TPM-scale analysis
  columns over housekeeping median, added as `*_nTPM_hk` and `*_TPM_hk`
- `pan_cancer_expression(normalize="percentile")` — percentile ranks
  on TPM-scale analysis columns, added as `*_nTPM_percentile` and
  `*_TPM_percentile`
- `pan_cancer_expression(normalize=["tpm_clean", "hk", "percentile"])` —
  add all requested normalized column families in one call. `tpm_clean`,
  `hk`, and `percentile` imply `tpm`.
- `pan_cancer_expression(log_transform=True)` — log2 transform on
  the active normalized columns, or on base `*_nTPM` / `*_TPM` columns
  when `normalize=None` or `normalize="tpm"`. Raw `*_FPKM` provenance
  columns stay raw; prefer the `*_log1p` normalize modes for additive
  logged columns.

`normalize_expression()` in `pirlygenes.expression` implements the shared
transform for samples and references. The default removal set is intentionally
narrow: mitochondrial transcripts, NUMT-like mitochondrial pseudogenes, and
rRNA/rRNA-pseudogene features, plus the nuclear-retained lncRNAs MALAT1 and
NEAT1. `remove_noncoding=True` additionally removes noncoding-biotype rows when
biotype metadata is available, while keeping protein-coding, immunoglobulin, and
TCR biotypes. That option is off by default because noncoding RNAs can be real
biology in some assays.

Current curated gene-set impact: the default transform silences the dedicated
mitochondrial QC gene set (`MT-CO1/2/3`, `MT-ND*`, `MT-CYB`, `MT-ATP6/8`,
`MT-RNR1/2`) in downstream biology, but raw QC still reports it. The local
symbol audit did not find therapy-target or cancer-marker rows matching the
default rRNA/NUMT removal rules. The optional noncoding-biotype gate can silence
noncoding transcript annotations such as retained-intron helper mappings, so it
should be used only when a protein-coding-centered analysis view is intended.

Other useful FFPE/degradation artifact categories to track but not remove by
default: ribosomal protein pseudogenes, small nuclear/snoRNA/Y-RNA/miRNA
features, replication-dependent histones, hemoglobin/globin transcripts,
extremely short transcripts, low-complexity processed pseudogenes, and
alignment-level rDNA repeat/decoy contig enrichment.

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
