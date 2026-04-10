# v4.0.0 Plan: Fine-Grained Expression Decomposition

## Problem

At low purity (e.g. a PRAD lymph node met at ~10%), the current TME subtraction
can't distinguish tumor from TME. GPNMB shows 156x PRAD median, CD74 at 33x,
FN1 at 72x — but we can't tell what's from the tumor vs the lymph node. We
need a decomposition that explains each gene's expression by attributing it to
specific cell types and tissue components.

## Goals

1. **Sample type detection**: cell line, sorted leukemia, solid primary, met
   (lymph node / liver / brain / lung / bone), mixed adjacent tissue
2. **Cell-type decomposition**: T cells, B cells, macrophages, fibroblasts,
   endothelial, etc. — with fractions and per-gene attribution
3. **Broader cancer coverage**: AML subtypes, CLL, sarcoma subtypes (GIST,
   leiomyosarcoma, liposarcoma, synovial, Ewing)
4. **Deconvolved TCGA reference**: apply decomposition to individual TCGA
   samples, extract tumor-only expression, use that as the comparison baseline
5. **Sample-type purity priors**: 3% for PDAC, <10% for lymph node mets, etc.
6. **Significance-aware**: don't sweat 0.1 vs 0.3 TPM, but flag 7 TPM when
   only tumor explains it

## Approach: Custom NNLS with Curated Signature Matrices

Build a custom decomposition using `scipy.optimize.nnls` rather than wrapping
R tools (CIBERSORTx, BayesPrism, EPIC). Rationale:

- pirlygenes is a pure Python pip package — no R dependency
- The math is trivial (~10 lines of scipy); the value is in reference data
- HK-normalization (our existing cross-platform strategy) slots in naturally
- CIBERSORTx requires web registration; BayesPrism is R-only

### Core model

For each gene g in sample s:

```
expression_g = sum_k (fraction_k * reference_g_k)
```

where k = {tumor, CD8_T, CD4_T, Treg, B_cell, plasma, NK, monocyte_macrophage,
dendritic, neutrophil, fibroblast, endothelial, site_tissue}

Solved via constrained NNLS with fractions summing to 1.

### Signature matrix

~300-500 genes x 12 cell types, HK-normalized. Sources:

- HPA single-cell RNA data (immune + stromal types)
- quanTIseq reference profiles (Finotello et al. 2019)
- LM22-equivalent collapsed to major types (Newman et al. 2015)
- HPA tissue nTPM for stromal (fibroblast, endothelial)

Shipped as `pirlygenes/data/immune-signature-matrix.csv` (~50KB).

### Per-gene explanatory decomposition

For each gene, after decomposition:

```python
{
    "symbol": "CD74",
    "observed_tpm": 1200.0,
    "decomposition": {
        "tumor_PRAD": 2.1,        # TPM from tumor
        "B_cells": 890.5,         # TPM from B cells (lymph node)
        "macrophages": 180.2,     # TPM from macrophages
        "T_cells": 95.0,          # TPM from T cells
        "other": 32.2,            # residual
    },
    "tumor_fraction_of_total": 0.002,
    "vs_tcga_tumor_only": 0.8,    # vs deconvolved TCGA median
    "significance": "not_tumor",
}
```

## Implementation Phases

### Phase 1: Immune Signature Matrix + NNLS (v3.18-3.19)

No breaking changes. Incremental.

1. Curate `immune-signature-matrix.csv` from public data
2. Create `pirlygenes/decomposition.py` with `estimate_cell_fractions()`
3. Add cell fraction output to `analyze_sample()` return dict
4. Add "Cell Composition" panel to summary plot
5. Validate against CIBERSORT on TCGA eval samples

New dependency: scipy (for `nnls`).

### Phase 2: Sample Type Detection + Purity Priors (v3.20)

No breaking changes.

1. Create `pirlygenes/sample_type.py` with `detect_sample_type()`
2. Create `pirlygenes/data/sample-type-priors.csv`
3. Integrate into `analyze_sample()` and `estimate_tumor_purity()`
4. Detection signals: ESTIMATE scores, lineage markers, tissue signatures,
   cell fraction patterns

### Phase 3: Per-Gene Decomposition (v3.21)

No breaking changes. Replaces current 9-point grid.

1. Extend `decomposition.py` with `decompose_gene_expression()`
2. New plot: stacked bar of expression sources per gene
3. Update `estimate_tumor_expression_ranges()` to use decomposition
4. Decomposition column group in output DataFrame

### Phase 4: TCGA Per-Sample Deconvolution (offline, pre-v4.0)

Batch job, not shipped.

1. Create `pirlygenes/tcga_decompose.py` (offline script)
2. Process ~10K TCGA samples through `estimate_cell_fractions()`
3. Compute tumor-only expression per sample, aggregate per type
4. Output `pirlygenes/data/tcga-deconvolved-expression.csv` (~2MB)
5. Validate against Aran et al. 2015 consensus purity

Runtime: ~1.5 hours (parallelizable).

### Phase 5: v4.0.0 Release

Breaking change: deconvolved TCGA replaces raw FPKM as reference.

1. `pan_cancer_expression()` gains deconvolved columns
2. `pct_cancer_median` compares to tumor-only TCGA reference
3. Cancer type detection uses deconvolved reference
4. Expanded cancer types (sarcoma subtypes, AML subtypes)
5. Full integration test suite

### Phase 6: Extended Cancer Coverage (v4.1+)

1. CLL (external reference, not in TCGA)
2. Sarcoma subtype detection from lineage panels
3. AML subtype classification
4. External cohort integration for rare types

## New Files

| File | Purpose | Size | Bundled |
|------|---------|------|---------|
| `pirlygenes/decomposition.py` | NNLS decomposition engine | ~500 lines | Yes |
| `pirlygenes/sample_type.py` | Sample type detection | ~300 lines | Yes |
| `pirlygenes/tcga_decompose.py` | Offline TCGA batch job | ~200 lines | No |
| `data/immune-signature-matrix.csv` | Cell type reference profiles | ~50KB | Yes |
| `data/sample-type-priors.csv` | Purity priors by sample type | ~2KB | Yes |
| `data/tcga-deconvolved-expression.csv` | Tumor-only TCGA reference | ~2MB | Yes |

## Key Design Decisions

- **12 cell types**, not 22: at bulk resolution, fine distinctions (naive vs
  memory CD4) are unreliable. Major types only.
- **HK-normalized signature matrix**: same normalization as everything else
  in pirlygenes. Avoids TPM/FPKM bridging problems.
- **Incremental delivery**: phases 1-3 are non-breaking (v3.18-3.21). Only
  phase 5 is a major version bump.
- **No R dependency**: pure Python + scipy. The algorithm is trivial; the
  value is in the curated reference data.
- **User override via `--sample-type`**: detection is best-effort; clinicians
  always know the biopsy site.

## Expanded Cancer Type Coverage

### Sarcoma subtypes (within SARC)
- Leiomyosarcoma: DES, ACTA2, TAGLN
- Liposarcoma: PPARG, FABP4, ADIPOQ, MDM2
- GIST: KIT, DOG1 (ANO1), PDGFRA
- Synovial sarcoma: TLE1, SS18-SSX markers
- Ewing sarcoma: NKX2-2, NR0B1

### AML subtypes (within LAML)
- APL (M3): PML-RARA targets
- Monocytic: CD14, CD64
- Myeloid: MPO, CD34

### New types
- CLL: CD5, CD23, CD19, CD20, BTK
