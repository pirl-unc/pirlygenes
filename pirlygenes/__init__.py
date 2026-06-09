# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curated cancer gene knowledge + reference expression data.

``pirlygenes`` is the data layer for cancer RNA analysis. It ships:

* curated gene-set CSVs (``gene_sets_cancer``) — therapy targets,
  CTAs, surfaceome, cancer-driver / cancer-key panels, lineage panels,
  rule tables, the cancer-type registry, narrative gene sets.
* curated gene families keyed by Ensembl ID (``gene_families``) —
  mtDNA, NUMTs, rRNA + pseudogenes, ribosomal proteins, histones,
  hemoglobins, immune-receptor segments, small ncRNAs, MALAT1/NEAT1.
* reference expression matrices + mechanical transforms
  (``expression``) — pan-cancer TCGA + HPA, HPA cell-type expression,
  ESTIMATE signatures; plus
  ``normalize_expression``, ``fpkm_to_tpm``,
  ``tpm_to_housekeeping_normalized``, ``classify_gene_qc``, and
  ``aggregate_gene_expression``.
* the bundled-dataset loader (``load_dataset``).
* canonical gene-id / gene-name helpers (``gene_ids``, ``gene_names``).

Analysis-layer code (CLI, plotting, sample QC narration, signature
scoring) lives in
`pirl-trufflepig <https://github.com/pirl-unc/trufflepig>`_, which
depends on this package.
"""

from .expression import (
    GeneQcClass,
    add_tpm_columns_from_fpkm,
    aggregate_gene_expression,
    available_cancer_expression_references,
    available_percentile_cohorts,
    available_representative_cohorts,
    cancer_enriched_genes,
    cancer_expression,
    CohortExpressionViews,
    cancer_reference_expression,
    cohort_expression_views,
    cohort_gene_percentiles,
    classify_gene_qc,
    clean_tpm_matrix,
    clean_tpm_removal_mask,
    drop_technical_genes,
    estimate_signatures,
    filter_technical_rna,
    filter_to_genes,
    fpkm_to_tpm,
    heme_tumor_up_vs_matched_normal,
    hpa_cell_type_expression,
    is_rescue_feature,
    log2_transform,
    normalize_expression,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
    normalize_to_housekeeping,
    pan_cancer_expression,
    percentile_rank_expression,
    rank_normalize,
    renormalize_to_million,
    representative_cohort_samples,
    select_representative_samples,
    technical_rna_gene_ids,
    tumor_up_vs_matched_normal,
    zscore_normalize,
    tpm_to_housekeeping_normalized,
)
from .gene_families import (
    GENE_FAMILIES,
    GeneFamily,
    gene_family_for_ensembl_id,
    gene_family_for_symbol,
    gene_family_ids,
    gene_family_names,
    gene_family_symbols,
    gene_family_table,
    hemoglobin_gene_ids,
    hemoglobin_gene_symbols,
    histone_gene_ids,
    histone_gene_symbols,
    immune_receptor_segment_ids,
    immune_receptor_segment_symbols,
    nuclear_retained_lncrna_ids,
    nuclear_retained_lncrna_symbols,
    numt_pseudogene_ids,
    numt_pseudogene_symbols,
    ribosomal_protein_ids,
    ribosomal_protein_pseudogene_ids,
    ribosomal_protein_pseudogene_symbols,
    ribosomal_protein_symbols,
    rrna_and_pseudogene_ids,
    rrna_and_pseudogene_symbols,
    small_noncoding_rna_ids,
    small_noncoding_rna_symbols,
)
from .gene_sets_cancer import (
    cancer_family_panel,
    cancer_family_panels,
    cancer_family_panels_df,
    cancer_lineage_panel,
    cancer_lineage_panels,
    cancer_lineage_panels_df,
    culture_stress_gene_ids,
    culture_stress_gene_names,
    culture_stress_genes_df,
    degradation_gene_pairs,
    degradation_gene_pairs_df,
    housekeeping_gene_ids,
    housekeeping_gene_names,
    lineage_gene_ids,
    lineage_gene_symbols,
    lineage_genes_by_cancer_type,
    lineage_genes_df,
    mitochondrial_gene_ids,
    mitochondrial_gene_names,
    mitochondrial_genes_df,
    therapy_benefit_toxicity_evidence,
    tme_marker_gene_ids,
    tme_marker_gene_names,
    tme_markers_df,
)
from .load_dataset import get_data, load_all_dataframes, load_all_dataframes_dict
from . import data_inventory, downloads
from .version import __version__


__all__ = [
    "__version__",
    # data loading
    "load_all_dataframes",
    "load_all_dataframes_dict",
    "get_data",
    # gene-set panels
    "housekeeping_gene_ids",
    "housekeeping_gene_names",
    "mitochondrial_genes_df",
    "mitochondrial_gene_ids",
    "mitochondrial_gene_names",
    "culture_stress_genes_df",
    "culture_stress_gene_ids",
    "culture_stress_gene_names",
    "tme_markers_df",
    "tme_marker_gene_ids",
    "tme_marker_gene_names",
    "degradation_gene_pairs_df",
    "degradation_gene_pairs",
    "lineage_genes_df",
    "lineage_gene_symbols",
    "lineage_gene_ids",
    "lineage_genes_by_cancer_type",
    "cancer_family_panels_df",
    "cancer_family_panel",
    "cancer_family_panels",
    "cancer_lineage_panels_df",
    "cancer_lineage_panel",
    "cancer_lineage_panels",
    "therapy_benefit_toxicity_evidence",
    # gene families
    "GeneFamily",
    "GENE_FAMILIES",
    "gene_family_names",
    "gene_family_table",
    "gene_family_ids",
    "gene_family_symbols",
    "gene_family_for_ensembl_id",
    "gene_family_for_symbol",
    "numt_pseudogene_ids",
    "numt_pseudogene_symbols",
    "nuclear_retained_lncrna_ids",
    "nuclear_retained_lncrna_symbols",
    "rrna_and_pseudogene_ids",
    "rrna_and_pseudogene_symbols",
    "ribosomal_protein_ids",
    "ribosomal_protein_symbols",
    "ribosomal_protein_pseudogene_ids",
    "ribosomal_protein_pseudogene_symbols",
    "small_noncoding_rna_ids",
    "small_noncoding_rna_symbols",
    "histone_gene_ids",
    "histone_gene_symbols",
    "hemoglobin_gene_ids",
    "hemoglobin_gene_symbols",
    "immune_receptor_segment_ids",
    "immune_receptor_segment_symbols",
    # expression: reference accessors
    "pan_cancer_expression",
    "cancer_reference_expression",
    "available_cancer_expression_references",
    "representative_cohort_samples",
    "cohort_expression_views",
    "CohortExpressionViews",
    "available_representative_cohorts",
    "cohort_gene_percentiles",
    "available_percentile_cohorts",
    "cancer_expression",
    "cancer_enriched_genes",
    "hpa_cell_type_expression",
    "estimate_signatures",
    "tumor_up_vs_matched_normal",
    "heme_tumor_up_vs_matched_normal",
    # expression: rescaling primitives
    "normalize_expression",
    "fpkm_to_tpm",
    "add_tpm_columns_from_fpkm",
    "percentile_rank_expression",
    "renormalize_to_million",
    "tpm_to_housekeeping_normalized",
    "normalize_technical_rna_columns",
    "normalize_technical_rna_long_table",
    "clean_tpm_matrix",
    "clean_tpm_removal_mask",
    "drop_technical_genes",
    "rank_normalize",
    "zscore_normalize",
    "select_representative_samples",
    # expression: reference-frame convenience
    "normalize_to_housekeeping",
    "log2_transform",
    "filter_technical_rna",
    "filter_to_genes",
    "technical_rna_gene_ids",
    # expression: classifier
    "classify_gene_qc",
    "is_rescue_feature",
    "GeneQcClass",
    # expression: aggregation
    "aggregate_gene_expression",
    # cohort-level downloads + cache
    "downloads",
    # cohort-level bundled-data inventory
    "data_inventory",
]
