# Plan: v1.4.0 (Radioligand Expansion)

## Goals

1. Add radioligand therapy datasets to `pirlygenes/data/`.
2. Ensure all requested radioligand targets are represented.
3. Add a `Radio` modality/category to treatment plotting.
4. Add a `categories` column to a gene summary CSV.

## References (starting set)

1. MDPI review provided by user: https://www.mdpi.com/2072-6694/17/21/3412
2. FDA approval (Pluvicto, ^177Lu-PSMA-617):
   https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-pluvicto-metastatic-castration-resistant-prostate-cancer
3. FDA approval (Lutathera, ^177Lu-DOTATATE):
   https://www.fda.gov/drugs/resources-information-approved-drugs/fda-approves-lutetium-lu-177-dotatate-gastroenteropancreatic-neuroendocrine-tumors
4. Broad target review (radioligand targets incl. CAIX/GRPR/HER2/CXCR4/etc):
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12239088/

Note: for trial-row ingestion and counts, use ClinicalTrials.gov API v2 with dated snapshots for reproducibility.

## Required Targets (must be present)

| Target Label | Canonical Gene Symbol | Ensembl Gene ID | Status Bucket |
|---|---|---|---|
| PSMA | FOLH1 | ENSG00000086205 | FDA-approved + trials |
| SSTR2 | SSTR2 | ENSG00000180616 | FDA-approved + trials |
| CD20 | MS4A1 | ENSG00000156738 | FDA-approved historical + trials |
| FAP | FAP | ENSG00000078098 | Trials |
| CAIX | CA9 | ENSG00000107159 | Trials |
| GRPR | GRPR | ENSG00000126010 | Trials |
| HER2 | ERBB2 | ENSG00000141736 | Trials |
| CXCR4 | CXCR4 | ENSG00000121966 | Trials |
| DLL3 | DLL3 | ENSG00000090932 | Trials |
| TROP2 | TACSTD2 | ENSG00000184292 | Trials |
| Nectin-4 | PVRL4 (NECTIN4 alias) | ENSG00000143217 | Trials |
| IGF-1R | IGF1R | ENSG00000140443 | Trials |
| MC1R | MC1R | ENSG00000258839 | Trials |
| GPC3 | GPC3 | ENSG00000147257 | Trials |

## Data Deliverables

1. `pirlygenes/data/radioligand-approved.csv`
2. `pirlygenes/data/radioligand-trials.csv`
3. `pirlygenes/data/gene-summary.csv` (new or regenerated), with `categories` column.

### Proposed schema: `radioligand-trials.csv`

- `NCT_Number`
- `Tumor_Site`
- `Disease_Setting`
- `Therapy`
- `Target_Symbols`
- `Target_Labels`
- `Radionuclide`
- `Ligand_or_Vector`
- `Phase`
- `Status`
- `Source`
- `Ensembl_Gene_IDs`
- `notes`

### Proposed schema: `gene-summary.csv`

- `Symbol`
- `Ensembl_Gene_ID`
- `categories`

`categories` is a semicolon-delimited list of modalities where the gene appears, e.g.:
- `ADCs;CAR-T;Radio`

## Code Changes

1. Add radioligand gene-set accessors in `pirlygenes/gene_sets_cancer.py`:
   - `radioligand_trial_target_gene_names()`
   - `radioligand_trial_target_gene_ids()`
   - `radioligand_target_gene_names()`
   - `radioligand_target_gene_ids()`
2. Update `pirlygenes/cli.py` treatment plot mapping to include:
   - `"Radio": radioligand_target_gene_names()`
3. Ensure plotting shows a separate treatment column/category for `Radio` in the therapies figure.
4. Add summary-generation utility/script (or function) that writes `gene-summary.csv` with `categories`.

## Validation Checklist

1. Gene symbol ↔ Ensembl ID consistency check for all rows.
2. NCT uniqueness checks and duplicate detection (`NCT_Number`, `Therapy`, `Target_Symbols`).
3. Required targets checklist above must all be present in `radioligand-trials.csv`.
4. CLI `plot-expression` produces `*-treatments.png` with a `Radio` category.
5. `gene-summary.csv` includes non-empty `categories` for all rows.

## Execution Order

1. Build `radioligand-trials.csv` from MDPI + ClinicalTrials.gov API extraction.
2. Build `radioligand-approved.csv` from FDA/label sources.
3. Wire `Radio` category into `gene_sets_cancer.py` and `cli.py`.
4. Generate/update `gene-summary.csv` with `categories`.
5. Validate + test + release bump for v1.4.0.
