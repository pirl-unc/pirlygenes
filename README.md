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
~85 names in `pirlygenes.__all__`. The submodule paths below are the
canonical home and stay stable across versions.

### Gene-set panels and resolvers

```python
from pirlygenes.gene_sets_cancer import (
    CTA_gene_names,                   # ~260 cancer-testis antigens
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
    therapy_benefit_toxicity_evidence,# curated clinical benefit/toxicity rows
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
    available_cancer_expression_references,
    cancer_expression_reference_status, # direct/parent/candidate coverage
    cancer_expression_source_candidates,# candidate source acquisition register
    cancer_reference_expression,       # source-agnostic non-TCGA references
    pan_cancer_expression,            # 19,784 genes × expression reference columns
    cancer_expression,                # one cancer type, clean TPM by default
    cancer_enriched_genes,            # genes enriched in one cancer vs the others
    tumor_up_vs_matched_normal,       # compact solid tumor-vs-normal markers
    heme_tumor_up_vs_matched_normal,  # compact heme tumor-vs-normal markers
    hpa_cell_type_expression,         # HPA single-cell consensus
    estimate_signatures,              # Yoshihara 2013 stromal/immune sigs

    # Rescaling primitives — pure math on expression matrices
    add_tpm_columns_from_fpkm,        # preserve FPKM and append TPM companions
    normalize_expression,             # zero technical-RNA rows + renormalize
    fpkm_to_tpm,                      # rescale each column to sum to 10⁶
    percentile_rank_expression,       # within-column percentile ranks
    renormalize_to_million,           # bare utility for column rescaling
    tpm_to_housekeeping_normalized,   # divide each column by housekeeping geomean
    log1p_transform,                  # natural log1p over selected value columns
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

`pan_cancer_expression()` exposes a single `normalize=` keyword so callers
don't have to chain the primitives by hand. It accepts `None`, a string, or a
list of strings. The default is `normalize="tpm_clean"` (equivalent to
`normalize=["tpm_clean"]`): TCGA `*_FPKM` columns are preserved for provenance,
deterministic TCGA `*_TPM` companions are added, HPA `*_nTPM` columns are
preserved, and cleaned TPM-scale analysis columns are added as
`*_TPM_clean` / `*_nTPM_clean`.

```python
# Zero mtDNA / NUMT / rRNA / MALAT1+NEAT1 rows across TPM-scale analysis
# columns and pin each column sum back at 1e6. Base *_nTPM, *_TPM, and
# *_FPKM columns remain unchanged; clean values are added as *_nTPM_clean
# and *_TPM_clean companion columns.
pan_cancer_expression()                          # normalize="tpm_clean"

# Raw/provenance view: raw <code>_FPKM from TCGA, <tissue>_nTPM from HPA,
# and deterministic <code>_TPM companions for analysis.
pan_cancer_expression(normalize=None)

# Add deterministic <code>_TPM companions derived from the FPKM columns.
pan_cancer_expression(normalize="tpm")

# Add explicit natural-log analysis columns while preserving raw/base values.
pan_cancer_expression(normalize="tpm_log1p")
pan_cancer_expression(normalize="tpm_clean_log1p")

# Add housekeeping-normalized TPM-scale columns. Percentile ranks are also
# available via normalize="percentile" as *_percentile columns.
pan_cancer_expression(normalize="hk")

# Combine modes in one call; tpm_clean, hk, and percentile each imply "tpm".
pan_cancer_expression(normalize=["tpm_clean", "hk", "percentile"])
```

`cancer_expression(cancer_type)` uses the same default analysis view across
reference sources: clean TPM (`normalize="tpm_clean"`). For TCGA-backed cancer
types, housekeeping-normalized values are available only when explicitly
requested with `normalize="hk"` or `normalize="housekeeping"`.
`cancer_reference_expression()` exposes packaged non-TCGA tumor references
through the same raw TPM / clean TPM contract; current bundled sources include
CLL-map (`CLL`), MMRF CoMMpass (`MM`), TARGET ALL (`B_ALL`, `T_ALL`),
CGCI/GDC Burkitt lymphoma (`BL`), GEO heme cohorts (`CML`, `MCL`, `MDS`,
`MPN`), a CTCL scRNA/TCR pseudobulk nTPM reference (`CTCL`),
BeatAML/TARGET subtype summaries, and selected Treehouse/GEO
cancer-specific cohorts such as `OS`, `PANNET`, `CHON`, `SCLC`, `RB`, and
sarcoma subtypes.
Imported symbol-only summaries are exposed only for genes that map
unambiguously to current Ensembl IDs, including conservative rescues through
older Ensembl gene names whose IDs still resolve in the current release. The
full source list is documented in `docs/gene-sets.md` and is available
programmatically with `available_cancer_expression_references()`.
`tumor_up_vs_matched_normal()` and `heme_tumor_up_vs_matched_normal()` expose
compact marker panels for tumor-up-vs-matched-normal comparisons.

The older `pan_cancer_expression()` kwargs (`technical_rna_normalize`,
`remove_noncoding`, and `renormalize_to_million`) have been removed. Use
`normalize="tpm_clean"` for the TPM-scaled, technical-RNA-cleaned view;
compose `normalize_expression()` / `renormalize_to_million()` directly when
you need lower-level transforms.

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
| `therapy-benefit-toxicity-evidence.csv` | `therapy_benefit_toxicity_evidence()` |
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
| `cancer-reference-expression.csv.gz` | `cancer_reference_expression()`, `available_cancer_expression_references()` |
| `tumor-up-vs-matched-normal.csv` | `tumor_up_vs_matched_normal()` |
| `heme-tumor-up-vs-matched-normal.csv` | `heme_tumor_up_vs_matched_normal()` |
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
  technical-RNA-cleaned expression accessors, derive `<TCGA>_TPM`
  columns from the ID-keyed pan-cancer `<TCGA>_FPKM` columns, and remove
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
