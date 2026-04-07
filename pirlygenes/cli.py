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

from argh import named, dispatch_commands
from typing import Optional, Set

from .version import print_name_and_version
from .load_dataset import load_all_dataframes
from .gene_sets_cancer import (
    ADC_target_gene_names,
    CAR_T_target_gene_names,
    pMHC_TCE_target_gene_names,
    surface_TCE_target_gene_names,
    bispecific_antibody_target_gene_names,
    radio_target_gene_names,
    TCR_T_target_gene_names,
)
from .load_expression import load_expression_data
from .plot import plot_gene_expression, plot_sample_vs_cancer, default_gene_sets


@named("data")
def print_dataset_sizes():
    for csv_file, df in load_all_dataframes():
        print("%s: %d rows" % (csv_file, len(df)))


def _parse_always_label_genes(always_label_genes: Optional[str]) -> Set[str]:
    if always_label_genes is None:
        return set()
    return {token.strip() for token in always_label_genes.split(",") if token.strip()}


@named("plot-expression")
def plot_expression(
    input_path: str,
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    label_genes: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    gene_id_col: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 12.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
):
    df_expr = load_expression_data(
        input_path,
        aggregate_gene_expression=aggregate_gene_expression,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
    )
    forced_labels = _parse_always_label_genes(label_genes)
    prefix = output_image_prefix or ""

    # Strip plots: summary + treatments
    for name, gene_sets in [
        ("summary", default_gene_sets),
        (
            "treatments",
            {
                "TCR-T": TCR_T_target_gene_names(),
                "CAR-T": CAR_T_target_gene_names(),
                "bispecifics": bispecific_antibody_target_gene_names(),
                "pMHC-TCEs": pMHC_TCE_target_gene_names(),
                "surface-TCEs": surface_TCE_target_gene_names(),
                "ADCs": ADC_target_gene_names(),
                "Radio": radio_target_gene_names(),
            },
        ),
    ]:
        output_image = (
            "%s-%s.png" % (prefix, name) if prefix else "%s.png" % name
        )
        plot_gene_expression(
            df_expr,
            gene_sets=gene_sets,
            save_to_filename=output_image,
            save_dpi=output_dpi,
            plot_height=plot_height,
            plot_aspect=plot_aspect,
            always_label_genes=forced_labels,
        )

    # Scatter plots: sample vs pan-cancer reference
    scatter_pdf = (
        "%s-vs-cancer.pdf" % prefix if prefix else "vs-cancer.pdf"
    )
    plot_sample_vs_cancer(
        df_expr,
        cancer_type=cancer_type,
        save_to_filename=scatter_pdf,
        save_dpi=output_dpi,
        always_label_genes=forced_labels,
    )


def main():
    print_name_and_version()
    print("---")
    dispatch_commands([print_dataset_sizes, plot_expression])


if __name__ == "__main__":
    main()
