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

Use `therapy_benefit_toxicity_evidence()` for curated clinical
benefit/toxicity rows keyed by agent, cancer code, subtype, and line of
therapy. These rows are separate from expression data; target expression is
not evidence of survival benefit or toxicity by itself.

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

`pan_cancer_expression()` returns a DataFrame with median expression across
33 TCGA cancer types (raw FPKM plus deterministic TPM companions), five
computed tumor rollups (`BTC`, `CRC`, `NET`, `NSCLC`, `SGC`), and 50 HPA normal
tissues (nTPM). The rollups are sample-weighted combinations of cohort-level
TPM medians, so they are TPM-only: FPKM cannot be recovered from a TPM summary
and is not synthesized.

The uniform analysis contract is TPM-based. Every TCGA cancer and computed
rollup has `*_TPM` plus every requested `*_TPM_clean`, `*_TPM_hk`,
`*_TPM_percentile`, or log derivative. `*_FPKM` is optional source provenance,
present only for the 33 inputs that were actually supplied in FPKM units; it is
not a required analysis column.

The five rollups are baked from the selected cohort sources into a small
canonical artifact and joined with the same oncoref Ensembl alias map used for
the base pan matrix. A gene unavailable in a rollup remains `NaN` in both
`*_TPM` and every derived column; it is never reported as a measured zero.

By default, `normalize="tpm_clean"`: the accessor preserves raw TCGA
`*_FPKM`, exposes TPM for TCGA and computed rollups, preserves HPA `*_nTPM`,
cleans TPM-scale analysis columns, and adds the clean values as
`*_TPM_clean` / `*_nTPM_clean`.

```python
from pirlygenes.expression import pan_cancer_expression

ref = pan_cancer_expression()
# Columns: Ensembl_Gene_ID, Symbol, adipose_tissue_nTPM, ..., ACC_FPKM, ..., ACC_TPM, ...
```

Supports normalization with `None`, a string, or a list of strings:
- `pan_cancer_expression(normalize=None)` — raw/provenance view with
  `*_FPKM` and `*_nTPM` columns unchanged, deterministic TCGA `*_TPM`
  companions, and TPM-only computed rollups
- `pan_cancer_expression(normalize="tpm")` or `normalize="TPM"` — expose
  the uniform `*_TPM` analysis columns for every tumor entity while preserving
  any available FPKM provenance
- `pan_cancer_expression(normalize="tpm_log1p")` — add natural-log
  `*_TPM_log1p` and `*_nTPM_log1p` analysis columns
- `pan_cancer_expression(normalize="tpm_clean")` — TPM scale plus the
  canonical clean-TPM 16/9/75 transform. Ribosomal-protein censored rows are
  pinned to 16% of each column budget, other technical censored rows to 9%,
  and biological rows to 75%. Base `*_nTPM`, `*_TPM`, and `*_FPKM` columns
  remain unchanged; clean values are added as `*_nTPM_clean` and
  `*_TPM_clean`.
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

`cancer_expression(cancer_type)` presents the same default analysis view for a
single cancer type, regardless of source. TCGA-backed cancers and packaged
non-TCGA references both default to clean TPM (`normalize="tpm_clean"`).
Housekeeping-normalized TCGA values are returned only when callers explicitly
request `normalize="hk"` or `normalize="housekeeping"`. Fine-grained registry
labels without their own packaged cohort, such as `BRCA_Basal`, `SARC_GIST`,
or `PCN`, load through their documented parent expression reference. Use
`cancer_expression_reference_status()` to see whether a code is backed by a
direct reference, TCGA pan-cancer column, parent reference, or only a candidate
source.

`cancer_reference_expression()` exposes oncoref-owned tumor references in long
or wide form with pirlygenes-compatible normalization names. Current references
include CLL-map (`CLL`), MMRF CoMMpass (`MM`), and TARGET ALL (`B_ALL`,
`T_ALL`), GEO/CGCI heme references including CTCL scRNA/TCR pseudobulk nTPM,
plus BeatAML/TARGET subtype summaries and selected Treehouse/GEO
cancer-specific cohorts such as osteosarcoma (`OS`), `PANNET`, `CHON`, `SCLC`,
`RB`, and sarcoma subtypes. Callers can inspect available sources with
`available_cancer_expression_references()`. Imported symbol-only cohort
summaries are exposed only for genes that map unambiguously to current Ensembl
IDs, including conservative rescues through older Ensembl gene names whose IDs
still resolve in the current release. Those sources provide median/Q1/Q3 but
not recoverable per-sample min/max.

`cancer_expression_source_candidates()` exposes the source-acquisition register
for missing or parent-backed references. Each row records `cancer_code`,
`source_status`, parent/reference code, accession/URL, assay, source scope,
estimated samples when known, and the intended processing/gene-ID/normalization
scheme. The register is intentionally loadable data rather than prose so
downstream consumers can distinguish `parent_sample_split_ready`,
`bulk_candidate_ready`, `scRNA_candidate_ready`,
`scRNA_candidate_needs_malignant_selection`, `parent_reference_only`, and
`source_needed`.

Current high-confidence candidates include:

| Codes | Candidate | Plan |
|---|---|---|
| `BRCA_Basal`, `BRCA_HER2`, `BRCA_LumA`, `BRCA_LumB`, `BRCA_Normal` | TCGA BRCA + PAM50 labels | sample-level GDC STAR-count TPM aggregation by PAM50 subtype |
| `HNSC_HPV_pos`, `HNSC_HPV_neg` | TCGA HNSC + HPV labels | sample-level GDC STAR-count TPM aggregation by HPV status |
| `LUAD_EGFR`, `LUAD_KRAS`, `LUAD_STK11` | TCGA LUAD + mutation labels | sample-level GDC STAR-count TPM aggregation by mutation class |
| `MBL_G3`, `MBL_G4`, `MBL_SHH`, `MBL_WNT` | GEO `GSE155446` | scRNA neoplastic-cell pseudobulk by medulloblastoma subgroup |
| `FL` | GEO `GSE261917` | scRNA pseudobulk after reproducible malignant B-cell selection |
| `NPC` | GEO `GSE102349` | bulk RNA-seq import of treatment-naive NPC tumors |
| `MEC` | GEO `GSE235092` | Merkel-cell carcinoma bulk RNA-seq import; note this registry code is not salivary mucoepidermoid carcinoma |
| `ADCC`, `ACINIC` | GEO `GSE294016` | salivary gland bulk TPM import by sample-table histology |
| `SARC_ASPS` | GEO `GSE54729` | bulk RNA-seq import of human ASPS tumor rows only |
| `SARC_GIST` | GEO `GSE162115` | scRNA tumor-cell pseudobulk from GIST tumor samples |
| `SARC_MPNST` | GEO `GSE207400` | bulk RNA-seq import of malignant PNST rows |

Rows marked `source_needed` are explicitly documented gaps, not silent
omissions. They still load via parent reference where the registry has a
defensible parent; otherwise `cancer_expression_reference_status()` reports
`candidate_or_missing`.

Packaged cancer expression coverage:

- TCGA/HPA pan-cancer reference via `pan_cancer_expression()`:
  `ACC`, `BLCA`, `BRCA`, `CESC`, `CHOL`, `COAD`, `DLBC`, `ESCA`, `GBM`,
  `HNSC`, `KICH`, `KIRC`, `KIRP`, `LAML`, `LGG`, `LIHC`, `LUAD`, `LUSC`,
  `MESO`, `OV`, `PAAD`, `PCPG`, `PRAD`, `READ`, `SARC`, `SKCM`, `STAD`,
  `TGCT`, `THCA`, `THYM`, `UCEC`, `UCS`, `UVM`.

| Code | Source project | Source cohort | Samples |
|---|---|---|---:|
| `ATRT` | Treehouse | `TREEHOUSE_POLYA_25_01` | 4 |
| `BL` | CGCI Burkitt Lymphoma Genome Sequencing Project | `CGCI_BLGSP` | 184 |
| `B_ALL` | TARGET ALL | `TARGET_ALL_2018` | 154 |
| `CHON` | GEO | `GSE299759_MEIJER_2026` | 54 |
| `CHOR` | Treehouse/RiboD | `TREEHOUSE_RIBOD_25_01` | 3 |
| `CLL` | CLL-map | `CLLMAP_2022` | 708 |
| `CML` | GEO | `GSE100026_DING_2017` | 5 |
| `CTCL` | GEO | `GSE171811_ECCITE_CTCL` | 7 |
| `EWS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 101 |
| `HEPB` | Treehouse | `TREEHOUSE_POLYA_25_01` | 20 |
| `LAML_APL` | BeatAML | `BEATAML_OHSU_2022` | 18 |
| `LAML_ELN_Adv` | BeatAML | `BEATAML_OHSU_2022` | 175 |
| `LAML_ELN_Fav` | BeatAML | `BEATAML_OHSU_2022` | 140 |
| `LAML_ELN_Int` | BeatAML | `BEATAML_OHSU_2022` | 100 |
| `MBL` | Treehouse | `TREEHOUSE_POLYA_25_01` | 125 |
| `MCL` | GEO | `GSE271664_BODOR_2025` | 51 |
| `MDS` | GEO | `GSE114922_SHIOZAWA_2018` | 82 |
| `MM` | MMRF CoMMpass | `MMRF_COMMPASS` | 764 |
| `MPN` | GEO | `GSE283710_WASHU_2024` | 45 |
| `NBL_MYCN_amp` | TARGET | `TARGET_NBL_2018` | 29 |
| `NBL_MYCN_nonamp` | TARGET | `TARGET_NBL_2018` | 113 |
| `NUTM` | Treehouse | `TREEHOUSE_POLYA_25_01` | 1 |
| `OS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 262 |
| `PANNET` | GEO | `GSE118014_ALVAREZ_2018` | 33 |
| `RB` | Treehouse/RiboD | `TREEHOUSE_RIBOD_25_01` | 15 |
| `RMS_ARMS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 73 |
| `RMS_ERMS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 95 |
| `RMS_PRMS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 6 |
| `RMS_SSRMS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 8 |
| `RT` | TARGET | `TARGET_RT_2017` | 43 |
| `SARC_DDLPS` | GEO | `GSE75885_DELESPAUL_2017` | 19 |
| `SARC_LGFMS` | GEO | `GSE75885_DELESPAUL_2017` | 2 |
| `SARC_LMS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 151 |
| `SARC_LPS_UNSPEC` | Treehouse | `TREEHOUSE_POLYA_25_01` | 92 |
| `SARC_MYXFIB` | Treehouse | `TREEHOUSE_POLYA_25_01` | 41 |
| `SARC_PLEOLPS` | GEO | `GSE75885_DELESPAUL_2017` | 4 |
| `SARC_SYN` | Treehouse | `TREEHOUSE_POLYA_25_01` | 50 |
| `SARC_UPS` | Treehouse | `TREEHOUSE_POLYA_25_01` | 110 |
| `SCLC` | University of Cologne | `SCLC_UCOLOGNE_2015` | 81 |
| `T_ALL` | TARGET ALL | `TARGET_ALL_2018` | 264 |
| `WILMS` | TARGET | `TARGET_WT_2015` | 130 |

`tumor_up_vs_matched_normal()` and
`heme_tumor_up_vs_matched_normal()` expose compact marker panels for
tumor-up-vs-matched-normal comparisons. They are marker tables, not full
expression matrices. Solid-tumor marker codes are `ACC`, `ATRT`, `BLCA`,
`BRCA`, `CESC`, `CHOL`, `CHON`, `COAD`, `ESCA`, `EWS`, `GBM`, `HEPB`,
`HNSC`, `KICH`, `KIRC`, `KIRP`, `LGG`, `LIHC`, `LUAD`, `LUSC`, `MBL`,
`MESO`, `NBL`, `OS`, `OV`, `PAAD`, `PANNET`, `PCPG`, `PRAD`, `RB`, `READ`,
`RMS_ARMS`, `RMS_ERMS`, `RMS_SSRMS`, `RT`, `SCLC`, `SKCM`, `STAD`, `TGCT`,
`THCA`, `THYM`, `UCEC`, `UCS`, `UVM`, and `WILMS`; heme marker codes are
`DLBC` and `LAML`.

`clean_tpm_matrix()` in `pirlygenes.expression` is the shared clean-TPM
transform for samples and references. Its censored compartments come from
oncoref's `clean-tpm-censored-genes` dataset, re-exported by pirlygenes for
compatibility, not from the supporting family tables:
`category == "ribosomal_protein"` is the 16% ribosomal compartment,
`category == "technical"` is the 9% other-technical compartment, and every
gene absent from that table remains biological.

`normalize_expression()` remains available as a lower-level strict
technical-RNA zero path. Its default removal set is intentionally narrower:
mitochondrial transcripts, NUMT-like mitochondrial pseudogenes,
rRNA/rRNA-pseudogene features, plus the nuclear-retained lncRNAs MALAT1 and
NEAT1. `remove_noncoding=True` additionally removes noncoding-biotype rows when
biotype metadata is available, while keeping protein-coding, immunoglobulin,
and TCR biotypes. That option is off by default because noncoding RNAs can be
real biology in some assays.

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

## Proteoform-level panels

Curated panels are stored in **member space** — each row carries its own gene
`symbol` and unversioned `Ensembl_Gene_ID`. That is the right primitive for the
~99% of genes that are single-locus and for every locus/allele-level row
(mutations, fusion breakpoints, amplifications). But a handful of genes encode
the **identical protein from ≥2 loci**, so a quantifier splits the reads across
them and per-locus TPM under-counts the protein:

- **cancer-testis X-paralog families** — `CTAG1A/CTAG1B` (NY-ESO-1),
  `SSX2/SSX2B`, `SSX4/SSX4B`, the `CT47A` cluster;
- **pseudoautosomal-region (PAR) genes** — `CRLF2`, `P2RY8`, `CSF2RA`,
  `IL3RA` (CD123), `CD99` (PAR1) and `IL9R` (PAR2): an identical X- *and*
  Y-encoded copy.

For these, the **protein** — not the locus — is what a TCR/CAR/ADC or a
protein-abundance threshold sees, so you must operate in *proteoform space*.

### The one rule

There is **one canonical space per analysis**. Map *both* your panel and your
expression data into it **up front**, then match — never compare raw member
symbols/ENSGs to collapsed data. Pick the space with `kind=`:

- `kind="protein"` — byte-identical canonical **protein** (the protein-abundance
  / antigen fold). The default for panel work.
- `kind="cdna"` — byte-identical canonical **cDNA** (the read-recovery fold for
  expression matrices; a few curated overrides force-collapse antigens like the
  `CT47A` cluster).

### The API, by what you hold

| You have | Call | Result |
|---|---|---|
| a panel, want every row's proteoform-space **key** | `add_proteoform_columns(df, kind="protein", id_col=...)` | adds `Proteoform_ID` (proteoform id if grouped, else own ENSG) + `Member_Ensembl_Gene_IDs` |
| a panel, want to **see** what's grouped | `annotate_panel_proteoforms(df, id_col=..., kind="protein")` | adds `proteoform_id`, `proteoform_members`, `proteoform_member_ensembl_ids`, `proteoform_n_members` (blank/0 for single-locus) |
| a single ENSG | `proteoform_group_of(ensg, kind="protein")` | the group dict, or `None` |
| a list of symbols / ENSGs | `fold_symbols(...)` / `fold_ids(...)` | deduped proteoform keys to match a collapsed matrix |
| an expression matrix | `collapse_protein_identical_loci(tpm_df)` | identical loci summed in **linear TPM** space |

```python
from pirlygenes.load_dataset import get_data
from pirlygenes.expression.protein_groups import (
    add_proteoform_columns, annotate_panel_proteoforms, collapse_protein_identical_loci)

panel = get_data("cancer-key-genes")

# "always operate in proteoform space" — one call gives every row its key:
keyed = add_proteoform_columns(panel, kind="protein", id_col="ensembl_gene_id")
collapsed = collapse_protein_identical_loci(tpm_df)        # match keyed.Proteoform_ID -> collapsed key

# inspect which curated genes are multi-locus proteoforms:
ann = annotate_panel_proteoforms(panel, id_col="ensembl_gene_id")
ann[ann.proteoform_n_members >= 2]    # NY-ESO-1 (CTAG1A/B), CD99, CD123/IL3RA, ...
```

Built-in panels can also ship **pre-folded** so the consumer never folds at the
selection site — e.g. `gene_sets_cancer.CTA_proteoform_symbols()` /
`CTA_proteoform_ids()`. `gene_names.display_name()` is the single id→label
authority (`CTAG1A/B` → `NY-ESO-1`). `n_members` counts distinct *loci*, so
same-symbol PAR X/Y pairs (CD99, IL3RA) correctly report 2.

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
