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

"""Curated cancer gene knowledge — data-only package.

Analysis, plotting, the ``analyze`` CLI, and all expression matrices
have moved to `trufflepig <https://github.com/pirl-unc/trufflepig>`_
in v5.0; this package is now a **data-only** dependency providing:

* curated gene-set CSVs (``gene_sets_cancer``)
* the bundled-dataset loader (``load_dataset``)
* canonical gene-id / gene-name helpers (``gene_ids``, ``gene_names``)
* QC feature panels keyed by Ensembl Gene ID (``qc_feature_groups``)

Install ``trufflepig`` to run RNA tumor analyses; it imports the
accessors here as a library.
"""

from .gene_sets_cancer import (
    cancer_family_panel,
    cancer_family_panels,
    cancer_family_panels_df,
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
    tme_marker_gene_ids,
    tme_marker_gene_names,
    tme_markers_df,
)
from .load_dataset import get_data, load_all_dataframes, load_all_dataframes_dict
from .qc_feature_groups import (
    QcFeatureClass,
    qc_class_for_ensembl_id,
    qc_class_for_symbol,
    qc_feature_ensembl_ids,
    qc_feature_groups,
    qc_feature_symbols,
    qc_feature_table,
)
from .version import __version__

__all__ = [
    "__version__",
    "load_all_dataframes",
    "load_all_dataframes_dict",
    "get_data",
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
    "QcFeatureClass",
    "qc_feature_table",
    "qc_feature_groups",
    "qc_feature_ensembl_ids",
    "qc_feature_symbols",
    "qc_class_for_ensembl_id",
    "qc_class_for_symbol",
]
