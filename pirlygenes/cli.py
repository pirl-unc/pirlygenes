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

from argh import named, dispatch_commands, arg
import pandas as pd

from .version import print_name_and_version
from .load_dataset import load_all_dataframes, get_data
from .load_expression import load_expression_data
from .plot import plot_gene_expression, default_gene_sets
from .gene_expression import aggregate_gene_expression as tx2gene


@named("data")
def print_dataset_sizes():
    for csv_file, df in load_all_dataframes():
        print("%s: %d rows" % (csv_file, len(df)))


@named("plot-expression")
def plot_expression(
    input_path: str,
    output_image_prefix: str | None = None,
    aggregate_gene_expression: bool = False,
):
    df_expr = load_expression_data(
        input_path, aggregate_gene_expression=aggregate_gene_expression
    )

    df_mite_trials = get_data("multispecific-tcell-engager-trials")
    df_approved_bispecifics = get_data("approved-bispecific-antibodies")
    df_adc_trials = get_data("ADC-trials.csv")
    print(df_mite_trials.head())
    print(df_approved_bispecifics.head())
    print(df_adc_trials.head())

    for name, gene_sets in [
        ("summary", default_gene_sets),
        (
            "treatments",
            {
                "bispecific-approved": set(df_approved_bispecifics.Symbol_1),
                "MiTE-trials": set(df_mite_trials.Tumor_Target_Symbols),
                "ADC-trials": set(df_adc_trials.Symbols),
            },
        ),
    ]:
        output_image = (
            "%s-%s.png" % (output_image_prefix, name)
            if output_image_prefix
            else "%s.png" % name
        )
        plot_gene_expression(
            df_expr, gene_sets=gene_sets, save_to_filename=output_image
        )


def main():
    print_name_and_version()
    print("---")
    dispatch_commands([print_dataset_sizes, plot_expression])


if __name__ == "__main__":
    main()
