# pirlygenes

> Curated cancer gene-knowledge data.

Analysis, plotting, the `analyze` CLI, and all expression matrices
moved to [`trufflepig`](https://github.com/pirl-unc/trufflepig) in
v5.0. This package now ships **gene-knowledge data only**:

- curated gene-set CSVs (therapy targets, CTAs, cancer-driver genes,
  housekeeping genes, surface proteins, immune/stromal marker panels,
  lineage and matched-normal panels, fusion/mutation expression-effect
  rules, narrative gene sets, …)
- the cancer-type registry and gene-symbol/Ensembl-ID resolvers
- cohort-baseline constants (e.g. `TCGA_MEDIAN_PURITY`)

Expression matrices (pan-cancer TCGA reference, subtype-deconvolved
non-TCGA cohorts, TCGA deconvolution, HPA cell-type expression,
tumor-up-vs-matched-normal panels, ESTIMATE signatures) ship with
`trufflepig` — use `trufflepig.reference.<accessor>()` to read them.

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
from pirlygenes.qc_feature_groups import (
    qc_class_for_ensembl_id,       # ENSG → QcFeatureClass (mt_dna, rrna_like, ...)
    qc_class_for_symbol,           # Symbol → QcFeatureClass
    qc_feature_groups,             # {group_name: DataFrame}
    qc_feature_ensembl_ids,        # set of ENSGs for one group
)
```

The `qc_feature_groups` panels are ENSG-keyed QC categorisations
derived from every installed Ensembl release; `trufflepig.expression_qc`
reads them as the source of truth for its
:func:`classify_gene_qc` lookup. Regenerate with
``python scripts/generate_qc_gene_sets.py`` after the upstream regex
changes.

Expression matrices and QC normalization moved to `trufflepig` in v5.0:

```python
from trufflepig.reference import (
    pan_cancer_expression,            # 3,100 genes x 83 columns (50 tissues + 33 cancers)
    cancer_expression,                # one cancer type
    cancer_enriched_genes,            # enriched genes for one cancer type
    subtype_deconvolved_expression,   # non-TCGA cohorts (Treehouse, GEO sarcoma, ...)
    tcga_deconvolved_expression,
    tumor_up_vs_matched_normal,
    heme_tumor_up_vs_matched_normal,
    hpa_cell_type_expression,
    estimate_signatures,
)
from trufflepig.expression_qc import (
    normalize_expression,
    normalize_technical_rna_long_table,
)
```

## What's bundled (`pirlygenes/data/`)

| Category | Files |
|---|---|
| Therapy targets | `ADC-approved.csv`, `ADC-trials.csv`, `ADC-withdrawn.csv`, `CAR-T-approved.csv`, `TCR-T-trials.csv`, `TCR-T-approved.csv`, `bispecific-antibodies-approved.csv`, `multispecific-tcell-engager-trials.csv`, `radioligand-targets.csv` |
| Surface proteins | `cancer-surfaceome.csv`, `surface-proteins.csv` |
| Cancer-testis antigens | `cancer-testis-antigens.csv` |
| Driver / key genes | `cancer-driver-genes.csv`, `cancer-driver-variants.csv`, `cancer-key-genes.csv` |
| Cancer-type registry | `cancer-type-registry.csv`, `cancer-family-panels.csv`, `cancer-type-genes.csv` |
| Lineage panel | `lineage-genes.csv` |
| Rule sets | `mutation-expression-effects.csv`, `fusion-expression-effects.csv`, `rare-cancer-fusion-rules.csv`, `rare-cancer-rna-surrogates.csv`, `degenerate-subtype-pairs.csv`, `fusion-surrogate-expression.csv`, `disease-state-rules.csv`, `narrative-gene-sets.csv` |
| QC panels | `housekeeping-genes.csv`, `mitochondrial-genes.csv`, `culture-stress-genes.csv`, `tme-markers.csv`, `degradation-gene-pairs.csv`, `ffpe-sensitive-markers.csv`, `artifact-expectations.csv` |
| QC feature groups (ENSG-keyed, derived) | `qc-mt-dna.csv`, `qc-mt-like-pseudogene.csv`, `qc-polya-bias-lncrna.csv`, `qc-rrna-like.csv`, `qc-ribosomal-protein.csv`, `qc-ribosomal-protein-pseudogene.csv`, `qc-small-ncrna.csv`, `qc-histone.csv`, `qc-hemoglobin.csv`, `qc-immune-receptor.csv` |
| Gene-set catalog | `gene-sets.csv` |
| Therapy response axes | `therapy-response-signatures.csv` |
| Misc | `ensembl-id-aliases.csv`, `extra-tx-mappings.csv` |

Expression matrices (`pan-cancer-expression.csv`,
`subtype-deconvolved-expression.csv.gz`, `tcga-deconvolved-expression.csv.gz`,
`hpa-cell-type-expression.csv`, `tumor-up-vs-matched-normal.csv`,
`heme-tumor-up-vs-matched-normal.csv`, `estimate-signatures.csv`)
moved to `trufflepig/data/` in v5.0 — access via
`trufflepig.reference.<accessor>()`.

The full curated set is the surface area `trufflepig` calls into.

## Migrating from pirlygenes 4.x

Most data-side imports are unchanged. Anything that ran analysis,
plotting, or sample-context inference moved to `trufflepig`:

| Was in pirlygenes 4.x | Now in 5.0 |
|---|---|
| `pirlygenes` CLI (`analyze`, `compare-analyze`, `plot-expression`, `plot-cancer-cohorts`, `data`, `cancers`) | `trufflepig run`, `trufflepig compare`, `trufflepig plot-cancer-cohorts`, `trufflepig data`, `trufflepig cancers` |
| `from pirlygenes import infer_sample_context, SampleContext, plot_sample_context, plot_degradation_index` | `from trufflepig.sample_context import infer_sample_context, SampleContext, plot_sample_context, plot_degradation_index` |
| `from pirlygenes import plot_gene_expression, plot_sample_vs_cancer, plot_geneset_vs_vital_tissues, plot_ctas_vs_cancer_type_detail` | `from trufflepig.plot import ...` |
| `from pirlygenes import pan_reference_embedding_genes, get_embedding_feature_metadata` | `from trufflepig.plot_embedding import ...` |
| `from pirlygenes.tumor_purity import TCGA_MEDIAN_PURITY` | `from pirlygenes.gene_sets_cancer import TCGA_MEDIAN_PURITY` *(moved into data-side; trufflepig re-exports)* |
| `from pirlygenes.cli import analyze, compare_analyze` (Python API) | `from trufflepig.main import analyze, compare_analyze` |

Unchanged (still in pirlygenes):
- `gene_sets_cancer.*` accessors (CTAs, surfaceome, panels, registry, etc.) **except** the expression accessors listed below
- `load_dataset.get_data`, `load_all_dataframes`, `load_all_dataframes_dict`
- `gene_ids.*`, `gene_names.*`

Moved to trufflepig in v5.0:
- `pirlygenes.gene_sets_cancer.pan_cancer_expression` → `trufflepig.reference.pan_cancer_expression`
- `pirlygenes.gene_sets_cancer.cancer_expression` → `trufflepig.reference.cancer_expression`
- `pirlygenes.gene_sets_cancer.cancer_enriched_genes` → `trufflepig.reference.cancer_enriched_genes`
- `pirlygenes.gene_sets_cancer.tcga_deconvolved_expression` → `trufflepig.reference.tcga_deconvolved_expression`
- `pirlygenes.gene_sets_cancer.subtype_deconvolved_expression` → `trufflepig.reference.subtype_deconvolved_expression`
- `pirlygenes.gene_sets_cancer.tumor_up_vs_matched_normal` → `trufflepig.reference.tumor_up_vs_matched_normal`
- `pirlygenes.gene_sets_cancer.heme_tumor_up_vs_matched_normal` → `trufflepig.reference.heme_tumor_up_vs_matched_normal`
- `pirlygenes.expression_qc.classify_gene_qc` → `trufflepig.expression_qc.classify_gene_qc` *(now ENSG-aware via `pirlygenes.qc_feature_groups`)*
- `pirlygenes.expression_qc.normalize_expression` → `trufflepig.expression_normalize.normalize_expression`
- `pirlygenes.expression_qc.normalize_technical_rna_long_table` → `trufflepig.expression_normalize.normalize_technical_rna_long_table`

If the `pirlygenes` console-script is still on PATH from a prior install, it now prints a one-line "moved to trufflepig" notice and exits 2.

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
