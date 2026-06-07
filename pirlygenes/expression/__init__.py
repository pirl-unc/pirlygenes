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

"""Reference expression matrices + mechanical transforms on them.

``pirlygenes.expression`` ships the curated cross-cohort expression
panels and the data-side operations needed to put two expression
columns on the same scale. Anything that requires interpretive
judgment (per-sample QC narration, library-prep classification,
signature scoring) lives in :mod:`trufflepig`.

Subpackage layout::

    pirlygenes/expression/
      accessors.py   # pan_cancer_expression, cancer_expression, ...
      qc.py          # classify_gene_qc, GeneQcClass, family classifier
      normalize.py   # normalize_expression, fpkm_to_tpm, ...
      aggregate.py   # aggregate_gene_expression (tx -> gene rollup)

Public surface — re-exported from this ``__init__`` so the common
imports are flat::

    from pirlygenes.expression import (
        pan_cancer_expression,
        cancer_reference_expression,
        cohort_expression_views,
        normalize_expression,
        fpkm_to_tpm,
        tpm_to_housekeeping_normalized,
        classify_gene_qc,
        aggregate_gene_expression,
    )
"""

from .accessors import (
    CohortExpressionViews,
    available_cancer_expression_references,
    available_representative_cohorts,
    cancer_enriched_genes,
    cancer_expression,
    cancer_expression_reference_status,
    cancer_expression_source_candidates,
    cancer_reference_expression,
    cohort_expression_views,
    estimate_signatures,
    filter_technical_rna,
    filter_to_genes,
    heme_tumor_up_vs_matched_normal,
    hpa_cell_type_expression,
    log1p_transform,
    log2_transform,
    normalize_to_housekeeping,
    pan_cancer_expression,
    representative_cohort_samples,
    technical_rna_gene_ids,
    tumor_up_vs_matched_normal,
)
from .aggregate import (
    aggregate_gene_expression,
    extra_tx_mappings,
)
from .normalize import (
    add_tpm_columns_from_fpkm,
    clean_tpm_matrix,
    clean_tpm_removal_mask,
    drop_technical_genes,
    fpkm_to_tpm,
    normalize_expression,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
    percentile_rank_expression,
    rank_normalize,
    renormalize_to_million,
    technical_rna_mask,
    tpm_to_housekeeping_normalized,
    zscore_normalize,
)
from .qc import (
    GeneQcClass,
    classify_gene_qc,
    is_rescue_feature,
)
from .representatives import select_representative_samples


__all__ = [
    # Reference-data accessors
    "pan_cancer_expression",
    "cancer_reference_expression",
    "available_cancer_expression_references",
    "cancer_expression_reference_status",
    "cancer_expression_source_candidates",
    "representative_cohort_samples",
    "cohort_expression_views",
    "CohortExpressionViews",
    "available_representative_cohorts",
    "tumor_up_vs_matched_normal",
    "heme_tumor_up_vs_matched_normal",
    "cancer_expression",
    "cancer_enriched_genes",
    "hpa_cell_type_expression",
    "estimate_signatures",
    # Rescaling primitives
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
    "technical_rna_mask",
    "rank_normalize",
    "zscore_normalize",
    "select_representative_samples",
    # Reference-frame convenience wrappers
    "normalize_to_housekeeping",
    "log1p_transform",
    "log2_transform",
    "filter_technical_rna",
    "filter_to_genes",
    "technical_rna_gene_ids",
    # Classifier
    "classify_gene_qc",
    "is_rescue_feature",
    "GeneQcClass",
    # Aggregation
    "aggregate_gene_expression",
    "extra_tx_mappings",
]
