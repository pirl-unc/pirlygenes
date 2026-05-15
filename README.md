# pirlygenes

> Curated cancer gene knowledge + reference expression data.

`pirlygenes` is the data layer for cancer RNA analysis. It ships:

- **Curated gene-set CSVs** — therapy targets, cancer-testis antigens,
  surfaceome, cancer-driver / cancer-key panels, lineage panels, rule
  tables, the cancer-type registry, narrative gene sets, …
- **Curated gene families keyed by Ensembl ID** — mtDNA, NUMTs, rRNA
  + pseudogenes, ribosomal proteins, histones, hemoglobins,
  immune-receptor segments, small ncRNAs, MALAT1/NEAT1.
- **Reference expression matrices** — pan-cancer TCGA × HPA panel,
  HPA cell-type expression, ESTIMATE signatures.
- **Mechanical transforms on the data** — `normalize_expression`,
  `fpkm_to_tpm`, `renormalize_to_million`,
  `tpm_to_housekeeping_normalized`, `classify_gene_qc`, and
  `aggregate_gene_expression`.
- Cohort-baseline constants (`TCGA_MEDIAN_PURITY`).

Analysis-layer code (CLI, plotting, sample-QC narration,
deconvolution, signature scoring) lives in
[`trufflepig`](https://github.com/pirl-unc/trufflepig).

## Install

```bash
pip install pirlygenes
```

Run analyses with [trufflepig](https://github.com/pirl-unc/trufflepig)
(distributed on PyPI as `pirl-trufflepig` — the bare `trufflepig`
name is owned by an unrelated package; the command + Python import
are both still `trufflepig`):

```bash
pip install pirl-trufflepig
trufflepig run --sample expr.tsv --workspace out --cancer-type PRAD
```

## Python API

Most accessors are re-exported from the top-level package, so
`from pirlygenes import pan_cancer_expression` works for any of the
~75 names in `pirlygenes.__all__`. The submodule paths below are the
canonical home and stay stable across versions.

### Gene-set panels and resolvers

```python
from pirlygenes.gene_sets_cancer import (
    CTA_gene_names,                   # ~258 cancer-testis antigens
    surface_protein_gene_names,       # 2,799 surfaceome genes
    cancer_surfaceome_gene_names,     # 147 tumor-specific surface targets
    therapy_target_gene_names,        # modality: "ADC", "CAR-T", "TCR-T",
                                      #   "bispecific-antibodies", "radioligand",
                                      #   "multispecific-TCE" (plus trial / approved
                                      #   sub-keys, e.g. "ADC-trials")
    cancer_type_registry,             # cancer-type registry DataFrame (125 rows)
    resolve_cancer_type,              # "prostate" / "PRAD" / "SARC_DDLPS" → registry code
    CANCER_TYPE_NAMES,                # registry-backed {code: display_name} view
    lineage_genes_by_cancer_type,     # lineage panels
    cancer_family_panels,             # broad-family aggregate panels (keys: PROSTATE,
                                      #   CRC, GASTRIC, ESCA_SQ, SQUAMOUS, MESENCHYMAL,
                                      #   RENAL, GLIAL, MELANOCYTIC)
    cancer_family_panel,              # e.g. cancer_family_panel("MESENCHYMAL")
    housekeeping_gene_ids,
    mitochondrial_gene_ids,
    tme_marker_gene_ids,              # tumor microenvironment markers
    degradation_gene_pairs,           # for RNA degradation index
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
    get_alias_as_list,                # symbol → list of known synonyms
    get_reverse_alias_as_list,        # canonical → all symbols that map to it
)
from pirlygenes.gene_names import display_name, short_gene_name, aliases
from pirlygenes.gene_families import (
    gene_family_for_ensembl_id,    # ENSG → family name (or None)
    gene_family_for_symbol,        # Symbol → family name (or None)
    gene_family_names,             # list of every shipped family
    gene_family_ids,               # set of ENSGs in one named family
    gene_family_symbols,           # set of Symbols in one named family
    gene_family_table,             # long-form DataFrame across all families
    # Typed per family (ID and symbol variants for each)
    numt_pseudogene_ids,
    numt_pseudogene_symbols,
    nuclear_retained_lncrna_ids,   # MALAT1, NEAT1 (ENE-stabilized)
    nuclear_retained_lncrna_symbols,
    rrna_and_pseudogene_ids,
    rrna_and_pseudogene_symbols,
    ribosomal_protein_ids,
    ribosomal_protein_pseudogene_ids,
    small_noncoding_rna_ids,       # snoRNAs, snRNAs, miRNAs, Y RNAs, ...
    histone_gene_ids,
    hemoglobin_gene_ids,
    immune_receptor_segment_ids,   # IG/TR V/D/J/C segments
)
```

### Expression matrices and transforms

```python
from pirlygenes.expression import (
    # Reference matrices (long- and wide-form)
    pan_cancer_expression,            # 19,784 genes × (50 nTPM tissues + 33 FPKM cancers + 33 TPM companions)
    cancer_expression,                # one cancer type, housekeeping-normalized
    cancer_enriched_genes,            # genes enriched in one cancer vs the others
    hpa_cell_type_expression,         # HPA single-cell consensus
    estimate_signatures,              # Yoshihara 2013 stromal/immune sigs

    # Rescaling primitives — pure math on expression matrices
    add_tpm_columns_from_fpkm,        # preserve FPKM and append TPM companions
    normalize_expression,             # zero technical-RNA rows + renormalize
    fpkm_to_tpm,                      # rescale each column to sum to 10⁶
    percentile_rank_expression,       # within-column percentile ranks
    renormalize_to_million,           # bare utility for column rescaling
    tpm_to_housekeeping_normalized,   # divide each column by housekeeping geomean
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,

    # Classifier — symbol/ENSG → QC class for tech-RNA flagging
    classify_gene_qc,
    is_rescue_feature,
    GeneQcClass,

    # Transcript → gene rollup
    aggregate_gene_expression,
)
```

`pan_cancer_expression()` exposes a single `normalize=` preset so callers
don't have to chain the primitives by hand. The default is
`normalize="clean_tpm"`: TCGA `FPKM_*` columns are preserved for provenance,
deterministic TCGA `TPM_*` companions are added, HPA `nTPM_*` columns are
preserved, TPM-scale analysis columns are cleaned and pinned to 1e6, and
pre-clean TPM-scale values are kept as `TPM_raw_*` / `nTPM_raw_*`.

```python
# Zero mtDNA / NUMT / rRNA / MALAT1+NEAT1 rows across TPM-scale analysis
# columns (nTPM_*, TPM_*) and pin each column sum back at 1e6. Raw FPKM
# columns remain unchanged as provenance; pre-clean nTPM/TPM values are kept
# as nTPM_raw_* and TPM_raw_* companion columns.
pan_cancer_expression()                          # normalize="clean_tpm"

# Raw/provenance view: raw FPKM_<code> from TCGA and nTPM_<tissue> from HPA,
# plus deterministic TPM_<code> companions derived from the FPKM columns.
pan_cancer_expression(normalize=None)

# Explicit alias for the raw/provenance TPM-companion view.
pan_cancer_expression(normalize="tpm")

# Divide TPM-scale analysis columns by their housekeeping-gene median.
# Percentile ranks are also available via normalize="percentile".
pan_cancer_expression(normalize="hk")
```

The three older kwargs (`technical_rna_normalize`, `remove_noncoding`,
`renormalize_to_million`) introduced in 5.1.1 still work but emit a
`DeprecationWarning`. Use `normalize="clean_tpm"` for the new
TPM-scaled, technical-RNA-cleaned view; compose
`normalize_expression()` / `renormalize_to_million()` when you need
exact legacy column names or semantics. The kwargs will be removed in a
later 5.x release.

The gene-family panels are ENSG-keyed sets derived from every
installed Ensembl release (`numt-pseudogenes.csv`,
`nuclear-retained-lncrnas.csv`, etc.); `pirlygenes.expression.qc`
reads them as the source of truth for `classify_gene_qc` lookup.
Mitochondrial-DNA membership is sourced from the curated
`mitochondrial-genes.csv` (with a semantic `Role` column).
Regenerate the derived CSVs with `python
scripts/generate_gene_family_sets.py` after the upstream regex panel
changes.

## What's bundled (`pirlygenes/data/`)

Every CSV ships in the wheel under `pirlygenes/data/`. The "Primary
accessor" column points at the typed Python entry point; any CSV
listed as `get_data("…")` has no named accessor and is meant to be
read raw via the generic loader.

| File | Primary accessor |
|---|---|
| `ADC-approved.csv`, `ADC-trials.csv`, `ADC-withdrawn.csv`, `CAR-T-approved.csv`, `TCR-T-trials.csv`, `TCR-T-approved.csv`, `bispecific-antibodies-approved.csv`, `multispecific-tcell-engager-trials.csv`, `radioligand-targets.csv` | `therapy_target_gene_names(modality)` / `therapy_target_gene_ids(modality)` |
| `cancer-surfaceome.csv` | `cancer_surfaceome_gene_names()`, `cancer_surfaceome_evidence()` |
| `surface-proteins.csv` | `surface_protein_gene_names()`, `surface_protein_evidence()` |
| `cancer-testis-antigens.csv` | `CTA_gene_names()`, `CTA_evidence()` |
| `cancer-driver-genes.csv` | `get_data("cancer-driver-genes")` |
| `cancer-driver-variants.csv` | `get_data("cancer-driver-variants")` |
| `cancer-key-genes.csv` | `cancer_key_genes_df()` |
| `cancer-type-registry.csv` | `cancer_type_registry()`, `CANCER_TYPE_NAMES`, `resolve_cancer_type()`, `cancer_types_in_family()`, `cancer_types_by_tissue()`, `cancer_type_subtypes_of()` |
| `cancer-family-panels.csv` | `cancer_family_panels()`, `cancer_family_panel(name)`, `cancer_family_panels_df()` |
| `cancer-type-genes.csv` | `cancer_type_gene_sets(cancer_type)` |
| `lineage-genes.csv` | `lineage_genes_df()`, `lineage_genes_by_cancer_type()`, `lineage_gene_ids(cancer_type)`, `lineage_gene_symbols(cancer_type)` |
| `mutation-expression-effects.csv` | `mutation_expression_effect_rules_df()` |
| `fusion-expression-effects.csv` | `fusion_expression_effect_rules_df()` |
| `rare-cancer-fusion-rules.csv` | `rare_cancer_fusion_rules_df()` |
| `rare-cancer-rna-surrogates.csv` | `rare_cancer_rna_surrogate_rules_df()` |
| `degenerate-subtype-pairs.csv` | `degenerate_subtype_pairs_df()` |
| `fusion-surrogate-expression.csv` | `fusion_surrogate_expression_df()` |
| `disease-state-rules.csv` | `disease_state_rules_df()` |
| `narrative-gene-sets.csv` | `narrative_gene_sets_df()`, `narrative_gene_set(name)` |
| `housekeeping-genes.csv` | `housekeeping_gene_names()`, `housekeeping_gene_ids()` |
| `mitochondrial-genes.csv` | `mitochondrial_genes_df(role=...)`, `mitochondrial_gene_ids()`, `mitochondrial_gene_names()` |
| `culture-stress-genes.csv` | `culture_stress_genes_df()`, `culture_stress_gene_ids()`, `culture_stress_gene_names()` |
| `tme-markers.csv` | `tme_markers_df()`, `tme_marker_gene_ids()`, `tme_marker_gene_names()` |
| `degradation-gene-pairs.csv` | `degradation_gene_pairs_df()`, `degradation_gene_pairs()` |
| `ffpe-sensitive-markers.csv` | `ffpe_sensitive_markers_df(direction=...)` |
| `artifact-expectations.csv` | `get_data("artifact-expectations")` |
| `numt-pseudogenes.csv` | `numt_pseudogene_ids()`, `numt_pseudogene_symbols()` |
| `nuclear-retained-lncrnas.csv` | `nuclear_retained_lncrna_ids()`, `nuclear_retained_lncrna_symbols()` |
| `rrna-and-pseudogenes.csv` | `rrna_and_pseudogene_ids()`, `rrna_and_pseudogene_symbols()` |
| `ribosomal-protein-genes.csv` | `ribosomal_protein_ids()`, `ribosomal_protein_symbols()` |
| `ribosomal-protein-pseudogenes.csv` | `ribosomal_protein_pseudogene_ids()`, `ribosomal_protein_pseudogene_symbols()` |
| `small-noncoding-rnas.csv` | `small_noncoding_rna_ids()`, `small_noncoding_rna_symbols()` |
| `histone-genes.csv` | `histone_gene_ids()`, `histone_gene_symbols()` |
| `hemoglobin-genes.csv` | `hemoglobin_gene_ids()`, `hemoglobin_gene_symbols()` |
| `immune-receptor-segments.csv` | `immune_receptor_segment_ids()`, `immune_receptor_segment_symbols()` |
| `pan-cancer-expression.csv` | `pan_cancer_expression()`, `cancer_expression(cancer_type)`, `cancer_enriched_genes(cancer_type)` |
| `hpa-cell-type-expression.csv` | `hpa_cell_type_expression()` |
| `estimate-signatures.csv` | `estimate_signatures()` |
| `gene-sets.csv` | `get_data("gene-sets")` (catalog of named sets) |
| `therapy-response-signatures.csv` | `get_data("therapy-response-signatures")` |
| `ensembl-id-aliases.csv` | `get_data("ensembl-id-aliases")` (consumed internally by `gene_ids`) |
| `extra-tx-mappings.csv` | `get_data("extra-tx-mappings")` (consumed internally by `gene_ids`, `aggregate_gene_expression`) |

Gene-family CSVs (`numt-pseudogenes.csv`, `nuclear-retained-lncrnas.csv`,
`rrna-and-pseudogenes.csv`, ribosomal-protein splits, `small-noncoding-rnas.csv`,
`histone-genes.csv`, `hemoglobin-genes.csv`, `immune-receptor-segments.csv`)
are **derived** — generated by `scripts/generate_gene_family_sets.py`
walking every installed Ensembl release. Re-run the script after the
upstream regex panel in `pirlygenes.expression.qc.classify_gene_qc`
changes. `mitochondrial-genes.csv` is curated by hand (the 37-row
mtDNA set with a semantic `Role` column).

## Where the boundary is

`pirlygenes` owns curated reference data and the *mechanical* operations
on it — gene-set lookups, expression-matrix accessors, FPKM↔TPM,
housekeeping rescaling, technical-RNA masking, transcript→gene rollup.
Anything that requires interpretive judgment (per-sample QC narration,
library-prep auto-detection, deconvolution pipelines, signature
scoring, rescue heuristics) lives in
[`pirl-trufflepig`](https://github.com/pirl-unc/trufflepig), which
depends on this package.

Downstream consumers pick their level:
- "I just want the data and its obvious transforms" → `pirlygenes` only.
- "I want to run a deconvolution / signature / report pipeline" →
  `pirl-trufflepig` (which pulls in `pirlygenes`).

## Migrating from pirlygenes 4.x or 5.0.x

> Only relevant if you're upgrading. Fresh 5.1+ installs can skip this.

The expression matrices and CLI ran a brief migration through 5.0.0–5.0.2
where the data lived in trufflepig. As of 5.1.0 the expression data is
back in pirlygenes and the boundary is the one described above.

| Was somewhere | Is now |
|---|---|
| `pirlygenes` CLI (4.x: `analyze`, `compare-analyze`, `plot-*`, `data`, `cancers`) | `trufflepig run`, `trufflepig compare`, `trufflepig plot-*`, `trufflepig data`, `trufflepig cancers` |
| 4.x: `from pirlygenes import infer_sample_context, SampleContext, plot_*` | `from trufflepig.sample_context import …`, `from trufflepig.plot import …` |
| 4.x: `from pirlygenes.tumor_purity import TCGA_MEDIAN_PURITY` | `from pirlygenes.gene_sets_cancer import TCGA_MEDIAN_PURITY` |
| 4.x or 5.0.x: `from pirlygenes.gene_sets_cancer import pan_cancer_expression` | `from pirlygenes.expression import pan_cancer_expression` (or `from pirlygenes import pan_cancer_expression`) |
| 5.0.x: `from trufflepig.reference import pan_cancer_expression` | `from pirlygenes.expression import pan_cancer_expression` |
| 5.0.x: `from trufflepig.expression_normalize import normalize_expression, fpkm_to_tpm` | `from pirlygenes.expression import normalize_expression, fpkm_to_tpm` |
| 5.0.x: `from trufflepig.expression_qc import classify_gene_qc` | `from pirlygenes.expression import classify_gene_qc` |
| `from pirlygenes.cli import analyze, compare_analyze` (Python API) | `from trufflepig.main import analyze, compare_analyze` |

Unchanged across all versions: `gene_sets_cancer.*` accessors,
`load_dataset.*`, `gene_ids.*`, `gene_names.*`.

If the `pirlygenes` console-script is still on PATH from a prior install, it now prints a one-line "moved to trufflepig" notice and exits 2.

## Migration history

- **v5.2.0** — add `normalize=` presets for TPM-scaled and
  technical-RNA-cleaned expression accessors, derive `TPM_<TCGA>`
  columns from the ID-keyed pan-cancer `FPKM_<TCGA>` columns, and remove
  deconvolution-derived reference tables from the package.
- **v5.1.0** — restore expression matrices to
  pirlygenes and add `pirlygenes.expression` with the rescaling
  primitives, the QC classifier, and the transcript→gene aggregator.
  Closes pirlygenes#246 and #247.
- **v5.0.0 – v5.0.2** — analysis CLI, plotting, and expression
  matrices briefly moved to trufflepig. 5.0.0's migration message
  pointed at `pip install trufflepig`, which on PyPI is an unrelated
  package; 5.0.1 corrected it to `pip install pirl-trufflepig`;
  5.0.2 added perf / caching polish.
- **v4.x** — combined data + analysis package.

See [`pirl-unc/trufflepig#1`](https://github.com/pirl-unc/trufflepig/issues/1)
for the analysis-migration umbrella and
[`pirl-unc/pirlygenes#119`](https://github.com/pirl-unc/pirlygenes/issues/119)
for the original deprecation thread.

## License

Apache 2.0 — see [LICENSE](LICENSE).
