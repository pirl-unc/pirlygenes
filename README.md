# pirlygenes

Gene lists related to cancer immunotherapy

## TCR-T 

### Clinical trials

Last updated: September 17th, 2024

Sources: 

- [Toward a comprehensive solution for treating solid tumors using T-cell receptor therapy: A review](https://www.sciencedirect.com/science/article/pii/S0959804924008803#sec0055)

## CAR-T

### Approved therapies

Last updated: September 17th, 2024

Sources: 

- [CAR-T: What Is Next? ](https://www.mdpi.com/2072-6694/15/3/663)

## Multi-specific antibodies and T-cell engagers

### Clinical trials

Last updated: September 11th, 2024

Sources:

- [Progresses of T-cell-engaging bispecific antibodies in treatment of solid tumors](https://www.sciencedirect.com/science/article/abs/pii/S1567576924011305)

## Antibody-drug conjugates (ADCs)

### Approved

Last updated: September 19th, 2024

Sources:

- [Development of antibody-drug conjugates in cancer: Overview and prospects](https://onlinelibrary.wiley.com/doi/full/10.1002/cac2.12517)


### Clinical trials 
Last updated: September 11th, 2024

Sources:

- [Pan-cancer analysis of antibody-drug conjugate targets and putative predictors of treatment response](<https://www.ejcancer.com/article/S0959-8049(23)00681-0/fulltext>)

## Radioligand therapies (RLTs)

### Current target list

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

# Partition ALL protein-coding genes into CTA / never-expressed / excluded / non-CTA
partition = CTA_partition()  # returns dict of Ensembl gene ID sets

# Evidence table with per-gene HPA tissue restriction data
df = CTA_evidence()
```

### Pipeline overview

The CTA gene set is built as an unbiased union of genes from multiple CT antigen databases and literature sources, then systematically filtered using Human Protein Atlas tissue expression data.

**Step 1: Collect** — union of protein-coding CT genes from multiple source databases (358 genes):

| Source | Genes | Reference |
|---|---|---|
| [CTpedia](http://www.cta.lncc.br/) | 167 | [Almeida et al. 2009](https://doi.org/10.1093/nar/gkn673), *NAR* |
| [CTexploreR/CTdata](https://www.bioconductor.org/packages/release/bioc/html/CTexploreR.html) | 62 new | [Loriot et al. 2024](https://doi.org/10.1371/journal.pgen.1011734), *PLOS Genetics* |
| [da Silva et al. 2017](https://doi.org/10.18632/oncotarget.21715) protein-level CT genes | 89 new | Tumor mass spec proteomics (136 genes, 46 overlap) |
| EWSR1-FLI1 CT gene binding sites | 12 | [Grünewald et al.](https://doi.org/10.1158/0008-5472.CAN-14-2908), *Cancer Research* |
| Meiosis, piRNA, spermatogenesis literature | 28 | Multiple sources |

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
| `CTA_partition()` | Partition all protein-coding genes into cta/never_expressed/excluded/non_cta | ~20k |

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
