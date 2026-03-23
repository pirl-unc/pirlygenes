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

Last updated: March 20th, 2026

Sources:

- [CTpedia](http://www.cta.lncc.br/)
- [Human Protein Atlas v23](https://www.proteinatlas.org/) — RNA tissue consensus & normal tissue IHC
- [EWSR1-FLI1 Activation of the Cancer/Testis Antigen FATE1 Promotes Ewing Sarcoma Survival](https://pubmed.ncbi.nlm.nih.gov/) — additional CT gene candidates

### Gene sets

The CTA data includes **207 genes** with two access tiers:

| Function | Returns | Count |
|---|---|---|
| `CTA_gene_names()` / `CTA_gene_ids()` | All CTAs (unfiltered) | 207 |
| `CTA_filtered_gene_names()` / `CTA_filtered_gene_ids()` | Reproductive-tissue-restricted CTAs | ~186 |
| `CTA_evidence()` | Full DataFrame with all evidence columns | 207 rows |

### Evidence columns

Each gene in `cancer-testis-antigens.csv` carries identity and HPA-derived tissue-restriction evidence:

| Column | Description |
|---|---|
| `Canonical_Transcript_ID` | Ensembl canonical (longest protein-coding) transcript ID |
| `protein_reproductive` | IHC detected only in testis/ovary/placenta (excluding thymus), or `"no data"` |
| `protein_thymus` | IHC detected in thymus |
| `protein_reliability` | Best HPA antibody reliability: Enhanced, Supported, Approved, Uncertain, or `"no data"` |
| `rna_reproductive` | All tissues with ≥1 nTPM (excluding thymus) are testis/ovary/placenta |
| `rna_thymus` | Thymus nTPM ≥ 1 |
| `protein_strict_expression` | Semicolon-separated tissues with IHC detection (excluding thymus) |
| `rna_reproductive_frac` | Fraction of total nTPM (excluding thymus) in core reproductive tissues |
| `rna_reproductive_and_thymus_frac` | Same, but thymus nTPM added to numerator and denominator |
| `rna_deflated_reproductive_frac` | `(1 + Σ repro max(0, nTPM−1)) / (1 + Σ all max(0, nTPM−1))` — deflated with +1 pseudocount |
| `rna_deflated_reproductive_and_thymus_frac` | Same but thymus added to reproductive numerator |
| `rna_80_pct_filter` / `rna_90_pct_filter` / `rna_95_pct_filter` | Deflated reproductive fraction ≥ threshold |
| `filtered` | Final inclusion flag (see below) |

### Filter logic

The `filtered` column uses tiered RNA thresholds based on protein data confidence. A gene passes when protein is detected only in reproductive tissues (thymus excluded) and the deflated RNA reproductive fraction meets the threshold for the antibody reliability tier:

| Protein evidence | RNA threshold |
|---|---|
| Enhanced (orthogonal validation) | ≥ 80% |
| Supported (consistent characterization) | ≥ 90% |
| Approved (basic validation) | ≥ 95% |
| Uncertain or no protein data | ≥ 99% |

Genes with protein detected in non-reproductive tissues always fail regardless of RNA.

Thymus is excluded from restriction checks because AIRE-driven expression in medullary thymic epithelial cells (mTECs) is expected for CTAs.

The deflated metric (`max(0, nTPM − 1)` per tissue) suppresses low-level basal transcription noise that would otherwise dilute the reproductive fraction for genes like CTCFL/BORIS (raw 54% → deflated 100%).

### Methodology

- Base list: intersection of CTpedia genes with HPA tissue antibody staining restricted to testis and placenta.
- Extended with CT genes from EWSR1-FLI1 binding site analysis (ADAM29, CABYR, CCDC33, CTCFL, DDX53, DPPA2, FTHL17, HORMAD2, LEMD1, SYCP1, TDRD1, TEX15).
- Extended with testis-specific genes from meiosis, piRNA pathway, and CT antigen literature that pass the HPA reproductive-tissue filter (ACTL7A, ACTL7B, BOLL, BRDT, CALR3, DDX4, DMRTB1, DPPA3, DPPA5, FKBP6, GAGE1, LDHC, MAEL, MAGEA8, MAGEA12, MAGEB10, MEIOB, NANOS2, PASD1, PIWIL1, RAD21L1, SMC1B, SYCE2, SYCP3, TEX14, UTF1, ZPBP, ZPBP2).
- HPA v23 RNA tissue consensus and normal tissue IHC data used to compute reproductive tissue restriction evidence for all genes.

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
