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
deconvolution, signature scoring) lives in :mod:`trufflepig`.

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
        normalize_expression,
        fpkm_to_tpm,
        tpm_to_housekeeping_normalized,
        classify_gene_qc,
        aggregate_gene_expression,
    )
"""

from .accessors import (
    cancer_enriched_genes,
    cancer_expression,
    estimate_signatures,
    filter_technical_rna,
    filter_to_genes,
    heme_tumor_up_vs_matched_normal,
    hpa_cell_type_expression,
    log2_transform,
    normalize_to_housekeeping,
    pan_cancer_expression,
    subtype_deconvolved_expression,
    tcga_deconvolved_expression,
    technical_rna_gene_ids,
    tumor_up_vs_matched_normal,
)
from .aggregate import (
    aggregate_gene_expression,
    extra_tx_mappings,
)
from .normalize import (
    fpkm_to_tpm,
    normalize_expression,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
    renormalize_to_million,
    tpm_to_housekeeping_normalized,
)
from .qc import (
    GeneQcClass,
    classify_gene_qc,
    is_rescue_feature,
)


__all__ = [
    # Reference-data accessors
    "pan_cancer_expression",
    "cancer_expression",
    "cancer_enriched_genes",
    "tcga_deconvolved_expression",
    "subtype_deconvolved_expression",
    "tumor_up_vs_matched_normal",
    "heme_tumor_up_vs_matched_normal",
    "hpa_cell_type_expression",
    "estimate_signatures",
    # Rescaling primitives
    "normalize_expression",
    "fpkm_to_tpm",
    "renormalize_to_million",
    "tpm_to_housekeeping_normalized",
    "normalize_technical_rna_columns",
    "normalize_technical_rna_long_table",
    # Reference-frame convenience wrappers
    "normalize_to_housekeeping",
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
