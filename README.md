# pirlygenes

> Curated cancer gene sets and reference expression data.

Analysis, plotting, and the `analyze` CLI moved to
[`trufflepig`](https://github.com/pirl-unc/trufflepig) in v5.0. This
package now ships **data only**:

- curated gene-set CSVs (therapy targets, CTAs, cancer-driver genes,
  housekeeping genes, surface proteins, immune/stromal marker panels,
  lineage and matched-normal panels, fusion/mutation expression-effect
  rules, narrative gene sets, …)
- pan-cancer TCGA expression reference (~3,100 genes × 83 columns: 50
  normal tissues + 33 TCGA cancer types)
- subtype-deconvolved expression references for non-TCGA cohorts
  (Treehouse pediatric, GEO sarcoma, panNET, …)
- the cancer-type registry and gene-symbol/Ensembl-ID resolvers
- cohort-baseline constants (e.g. `TCGA_MEDIAN_PURITY`)

## Install

```bash
pip install pirlygenes
```

Run analyses with [trufflepig](https://github.com/pirl-unc/trufflepig):

```bash
pip install trufflepig
trufflepig run --sample expr.tsv --workspace out --cancer-type PRAD
```

## Python API

```python
from pirlygenes.gene_sets_cancer import (
    CTA_gene_names,                   # ~257 cancer-testis antigens
    surface_protein_gene_names,       # 2,799 surfaceome genes
    cancer_surfaceome_gene_names,     # 147 tumor-specific surface targets
    therapy_target_gene_names,        # by modality: "ADC", "CAR-T", "TCR-T", "bispecific", ...
    pan_cancer_expression,            # 3,100 genes x 83 columns (50 tissues + 33 cancers)
    cancer_expression,                # one cancer type
    cancer_enriched_genes,            # enriched genes for one cancer type
    cancer_type_registry,             # cancer-type registry DataFrame
    lineage_genes_by_cancer_type,     # lineage panels
    cancer_family_panels,             # broad-family aggregate panels
    housekeeping_gene_ids,
    mitochondrial_gene_ids,
    tme_marker_gene_ids,              # tumor microenvironment markers
    degradation_gene_pairs,           # for RNA degradation index
    cancer_family_panel,
    TCGA_MEDIAN_PURITY,               # per-cohort median tumor purity (Aran et al., 2015)
)
from pirlygenes.gene_sets_cancer import (
    fusion_expression_effect_rules_df,
    mutation_expression_effect_rules_df,
    rare_cancer_fusion_rules_df,
    rare_cancer_rna_surrogate_rules_df,
    degenerate_subtype_pairs_df,
    fusion_surrogate_expression_df,
    narrative_gene_sets_df,
    narrative_gene_set,
    disease_state_rules_df,
)
from pirlygenes.load_dataset import get_data, get_all_csv_paths
from pirlygenes.gene_ids import (
    find_canonical_gene_ids_and_names,
    gene_id_aliases,
)
from pirlygenes.gene_names import display_name, short_gene_name, aliases
from pirlygenes.expression_qc import (
    normalize_expression,
    normalize_technical_rna_long_table,
)
```

## What's bundled (`pirlygenes/data/`)

| Category | Files |
|---|---|
| Therapy targets | `ADC-approved.csv`, `ADC-trials.csv`, `CAR-T-approved.csv`, `TCR-T-trials.csv`, `TCR-T-approved.csv`, `bispecific-antibodies-approved.csv`, `radioligand-approved.csv`, `radioligand-trials.csv` |
| Surface proteins | `cancer-surfaceome.csv`, `surface-proteins.csv` |
| Cancer-testis antigens | `cancer-testis-antigens.csv` |
| Driver genes | `cancer-driver-genes.csv`, `cancer-driver-variants.csv`, `cancer-key-genes.csv` |
| Cancer-type registry | `cancer-type-registry.csv`, `cancer-family-panels.csv`, `cancer-type-genes.csv` |
| Lineage / matched-normal | `lineage-genes.csv`, `tumor-up-vs-matched-normal.csv`, `heme-tumor-up-vs-matched-normal.csv` |
| Reference expression | `pan-cancer-expression.csv`, `subtype-deconvolved-expression.csv.gz`, `tcga-deconvolved-expression.csv` |
| Rule sets | `mutation-expression-effects.csv`, `fusion-expression-effects.csv`, `rare-cancer-fusion-rules.csv`, `rare-cancer-rna-surrogates.csv`, `degenerate-subtype-pairs.csv`, `fusion-surrogate-expression.csv`, `disease-state-rules.csv`, `narrative-gene-sets.csv` |
| QC panels | `housekeeping-genes.csv`, `mitochondrial-genes.csv`, `culture-stress-genes.csv`, `tme-markers.csv`, `degradation-gene-pairs.csv` |
| Therapy response axes | `therapy-response-signatures.csv` |
| Misc | `ensembl-id-aliases.csv`, `extra-tx-mappings.csv`, `estimate-signatures.csv`, `ffpe-sensitive-markers.csv`, `artifact-expectations.csv` |

The full curated set is the surface area `trufflepig` calls into.

## Migration history

- **v5.0.0** — `analyze`, `compare-analyze`, plotting, and reporting
  moved to `trufflepig`. The `pirlygenes` CLI is removed; data and
  Python API are unchanged.
- **v4.x** — combined data + analysis package.

See [`pirl-unc/trufflepig#1`](https://github.com/pirl-unc/trufflepig/issues/1)
for the migration umbrella and
[`pirl-unc/pirlygenes#119`](https://github.com/pirl-unc/pirlygenes/issues/119)
for the deprecation tracking.

## License

Apache 2.0 — see [LICENSE](LICENSE).
