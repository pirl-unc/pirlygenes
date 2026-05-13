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

"""Curated cancer gene sets and reference expression data.

Analysis, plotting, and the ``analyze`` CLI have moved to
`trufflepig <https://github.com/pirl-unc/trufflepig>`_; this package is
now a **data-only** dependency providing:

* curated gene-set CSVs (``gene_sets_cancer``)
* the bundled-dataset loader (``load_dataset``)
* canonical gene-id / gene-name helpers (``gene_ids``, ``gene_names``)

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
    pan_cancer_expression,
    tme_marker_gene_ids,
    tme_marker_gene_names,
    tme_markers_df,
)
from .load_dataset import get_data, load_all_dataframes, load_all_dataframes_dict
from .version import __version__

__all__ = [
    "__version__",
    "load_all_dataframes",
    "load_all_dataframes_dict",
    "get_data",
    "pan_cancer_expression",
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
]
