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
    cancer_types,
    cancer_type_gene_sets,
)
from .load_expression import load_expression_data
from .plot import (
    plot_gene_expression,
    plot_sample_vs_cancer,
    plot_cancer_type_genes,
    plot_cancer_type_pca,
    default_gene_sets,
    CANCER_TYPE_ALIASES,
    CANCER_TYPE_NAMES,
)

_DATASET_SOURCES = {
    "ADC-approved": "Wiley, doi:10.1002/cac2.12517",
    "ADC-trials": "EJC, doi:10.1016/j.ejca.2023.113342",
    "ADC-withdrawn": "FDA records",
    "bispecific-antibodies-approved": "FDA/EMA approvals",
    "cancer-driver-genes": "Bailey et al. 2018, Cell",
    "cancer-driver-variants": "Bailey et al. 2018, Cell",
    "cancer-surfaceome": "Hu et al. 2021, Nature Cancer (TCSA L3)",
    "cancer-testis-antigens": "CTpedia, CTexploreR, daSilva2017 + HPA v23",
    "CAR-T-approved": "FDA approvals",
    "class1-mhc-presentation-pathway": "Literature curation",
    "housekeeping-genes": "Eisenberg & Levanon 2013",
    "gene-sets": "Pre-resolved gene sets (immune, oncogenic, DNA repair)",
    "interferon-response": "Literature curation",
    "multispecific-tcell-engager-trials": "Literature curation 2024",
    "pan-cancer-expression": "HPA v23 (nTPM) + GDC/STAR (median TPM), 33 TCGA types",
    "radioligand-targets": "Literature curation 2026",
    "surface-proteins": "Bausch-Fluck et al. 2018, PNAS (SURFY/CSPA)",
    "TCR-T-approved": "FDA approvals",
    "TCR-T-trials": "Literature curation 2024",
}


@named("data")
def print_dataset_info():
    """List all bundled datasets with row counts and sources."""
    import os
    from pathlib import Path

    total_size = 0
    print("\nBundled datasets (shipped with pip install, no downloads needed):\n")
    print(f"  {'Dataset':<40s} {'Rows':>6s}  {'Size':>8s}  Source")
    print(f"  {'─'*40}  {'─'*6}  {'─'*8}  {'─'*40}")
    for csv_file, df in load_all_dataframes():
        name = Path(csv_file).stem
        size = os.path.getsize(csv_file)
        total_size += size
        if size >= 1024 * 1024:
            size_str = "%.1f MB" % (size / 1024 / 1024)
        else:
            size_str = "%.0f KB" % (size / 1024)
        source = _DATASET_SOURCES.get(name, "")
        print(f"  {name:<40s} {len(df):>6d}  {size_str:>8s}  {source}")
    print(f"\n  Total: {total_size / 1024 / 1024:.1f} MB\n")

    # Cancer types
    types = cancer_types()
    print(f"  Cancer types ({len(types)}):\n")
    print(f"  {'Code':<6s}  {'Full name':<45s}  Aliases")
    print(f"  {'─'*6}  {'─'*45}  {'─'*30}")
    # Build reverse alias map: code -> [aliases]
    code_to_aliases = {}
    for alias, code in CANCER_TYPE_ALIASES.items():
        code_to_aliases.setdefault(code, []).append(alias)
    for code in types:
        full_name = CANCER_TYPE_NAMES.get(code, "")
        aliases = ", ".join(sorted(code_to_aliases.get(code, [])))
        print(f"  {code:<6s}  {full_name:<45s}  {aliases}")
    print()


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

    # Cancer type signature plots
    genes_png = "%s-cancer-types-genes.png" % prefix if prefix else "cancer-types-genes.png"
    plot_cancer_type_genes(df_expr, save_to_filename=genes_png, save_dpi=output_dpi)

    pca_png = "%s-cancer-types-scatter.png" % prefix if prefix else "cancer-types-scatter.png"
    plot_cancer_type_pca(df_expr, save_to_filename=pca_png, save_dpi=output_dpi)

    # Cancer-type-specific gene set plot (only when --cancer-type specified)
    if cancer_type:
        from .plot import resolve_cancer_type
        code = resolve_cancer_type(cancer_type)
        ct_gene_sets = cancer_type_gene_sets(cancer_type)
        if ct_gene_sets:
            ct_png = "%s-%s-genes.png" % (prefix, code.lower()) if prefix else "%s-genes.png" % code.lower()
            plot_gene_expression(
                df_expr,
                gene_sets=ct_gene_sets,
                save_to_filename=ct_png,
                save_dpi=output_dpi,
                plot_height=plot_height,
                plot_aspect=plot_aspect,
                always_label_genes=forced_labels,
            )


def main():
    print_name_and_version()
    print("---")
    dispatch_commands([print_dataset_info, plot_expression])


if __name__ == "__main__":
    main()
