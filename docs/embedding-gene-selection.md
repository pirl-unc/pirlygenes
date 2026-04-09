# Embedding Gene Selection

How genes are chosen for the PCA/MDS cancer-type embedding plots.

## Problem

Bulk RNA-seq from tumors is a mixture of tumor cells and the tumor
microenvironment (TME) — immune cells and stromal cells. At low tumor
purity (e.g., 5-10%), most of the RNA signal comes from the TME, not
the tumor. A naive gene selection based on cancer-type specificity alone
will pick genes that are also expressed in normal tissues, making the
sample look like whatever tissue dominates the non-tumor fraction.

## Approach: Bottleneck Scoring

For each gene and each cancer type, we compute two z-scores:

- **z_tme** — How far is this gene's cancer-type expression above
  the distribution of TME tissue expression? TME tissues are identified
  data-driven: immune tissues where PTPRC (CD45) > median across all HPA
  normal tissues, plus stromal tissues (smooth/skeletal/heart muscle,
  adipose). Reproductive tissues are excluded from the denominator so
  cancer-testis antigens are not penalized.

- **z_other** — How far is this gene's cancer-type expression above
  the distribution of all 33 TCGA cancer types? This measures
  cancer-type specificity.

The combined score is `min(z_tme, z_other)` — the **bottleneck**. A gene
ranks high only if it is *both* visible above TME background *and*
specific to the cancer type. This avoids hard thresholds on either axis:
a gene with extreme TME silence but poor specificity (or vice versa) is
naturally downranked.

Top 5 genes per cancer type are selected (yielding ~158 genes total),
IG/TR gene families are excluded (somatically rearranged, don't follow
normal transcriptional regulation).

The embedding uses z-score of log2(1 + expression) with clipping at +/-3.

## Evaluation

Evaluated on 160 individual TCGA RNA-seq samples (5 per cancer type,
32 types) from the UCSC Xena TOIL recompute. To test purity robustness,
each tumor sample was mixed with a random GTEx whole blood or spleen
sample at varying ratios.

### Gene set comparison (full-space nearest-neighbor accuracy)

21 gene sets were compared, including:

| Gene set | #genes | Pure | 10% purity | 5% purity |
|----------|-------:|:----:|:----------:|:---------:|
| **min(z_tme, z_other) @5** | **158** | **85%** | **71%** | **56% / 76%** |
| z x log(tme) x log(other) @10 | 318 | 89% | 71% | 52% / 83% |
| all-discrim-10 (no TME filter) | 330 | 88% | 65% | 53% / 82% |
| composite-10 (z x log(S/N+1)) | 293 | 85% | 63% | 50% / 72% |
| z10-tme<25 (loose TME filter) | 325 | 85% | 63% | 48% / 71% |
| z10-tme<1 (strict TME silence) | 239 | 79% | 28% | 11% / 36% |
| z10-imm<1 (strict immune silence) | 251 | 79% | 28% | 10% / 37% |

Values are top-1 accuracy (and top-1 / top-5 for 5% purity).

### Key findings

1. **Strict TME silence is counterproductive.** Requiring tme < 1 nTPM
   eliminates the most discriminating genes, achieving only 11% at 5%
   purity — worse than no TME filtering at all (53%).

2. **Bottleneck scoring is the most purity-robust.** 56% at 5% purity
   with only 158 genes. The `min()` naturally balances both axes without
   any threshold.

3. **z-score of log2(1+expr) is the best normalization.** Tested against
   log, rank, HK-normalized, raw z-score, and various combinations.
   HK normalization was catastrophic at low purity.

4. **More genes help at high purity but hurt at low purity.** 330 genes
   (all-discrim-10) achieves 88% at pure but only 53% at 5%. The compact
   158-gene bottleneck set degrades more gracefully.

5. **PCA 2D projection loses substantial signal.** Best full-space
   accuracy is 85-89%, but PCA 2D is only 46-56%. The 2D plots are for
   visualization, not classification.

### Normalization comparison (on the top 3 gene sets)

| Normalization | Pure | 10% | 5% |
|--------------|:----:|:---:|:--:|
| **z-score of log2(1+x)** | **88%** | **65%** | **53%** |
| log2(1+x) only | 84% | 59% | 37% |
| percentile rank | 82% | 47% | 42% |
| HK-normalized z-score | 68% | 23% | 13% |
| raw z-score (no log) | 84% | 46% | 24% |

### Dilution method

Purity simulation: `mixed = purity * tumor_TPM + (1-purity) * gtex_immune_TPM`

GTEx immune samples: 20 whole blood + 10 spleen samples from the UCSC
Xena GTEx TOIL recompute (uniformly processed with the same pipeline as
TCGA).

## Implementation

`pirlygenes.plot._select_embedding_genes_bottleneck(n_genes_per_type=5)`

Called via `_cancer_type_feature_matrix(df, method="bottleneck")`, which
is the default for `plot_cancer_type_pca()` and `plot_cancer_type_mds()`.
