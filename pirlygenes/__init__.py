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
from .load_dataset import load_all_dataframes, load_all_dataframes_dict, get_data
from .sample_context import (
    SampleContext,
    infer_sample_context,
    plot_degradation_index,
    plot_sample_context,
)
from .plot import (
    plot_ctas_vs_cancer_type_detail,
    plot_gene_expression,
    plot_geneset_vs_vital_tissues,
    plot_sample_vs_cancer,
)
from .version import __version__

__all__ = [
    "__version__",
    # Dataset access
    "load_all_dataframes",
    "load_all_dataframes_dict",
    "get_data",
    # Plotting
    "plot_gene_expression",
    "plot_sample_vs_cancer",
    "plot_geneset_vs_vital_tissues",
    "plot_ctas_vs_cancer_type_detail",
    # Pan-cancer reference
    "pan_cancer_expression",
    # Housekeeping
    "housekeeping_gene_ids",
    "housekeeping_gene_names",
    # Mitochondrial
    "mitochondrial_genes_df",
    "mitochondrial_gene_ids",
    "mitochondrial_gene_names",
    # Culture stress
    "culture_stress_genes_df",
    "culture_stress_gene_ids",
    "culture_stress_gene_names",
    # TME markers
    "tme_markers_df",
    "tme_marker_gene_ids",
    "tme_marker_gene_names",
    # Degradation pairs
    "degradation_gene_pairs_df",
    "degradation_gene_pairs",
    # Lineage
    "lineage_genes_df",
    "lineage_gene_symbols",
    "lineage_gene_ids",
    "lineage_genes_by_cancer_type",
    # Family panels
    "cancer_family_panels_df",
    "cancer_family_panel",
    "cancer_family_panels",
    # Sample context (stage 1 of unified attribution flow)
    "SampleContext",
    "infer_sample_context",
    "plot_sample_context",
    "plot_degradation_index",
]
