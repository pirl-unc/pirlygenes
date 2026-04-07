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
    therapy_target_gene_id_to_name,
    pMHC_TCE_target_gene_names,
    surface_TCE_target_gene_names,
    cancer_types,
    cancer_type_gene_sets,
)
import matplotlib.pyplot as plt
from PIL import Image
from .load_expression import load_expression_data
from .plot import (
    plot_gene_expression,
    plot_sample_vs_cancer,
    plot_cancer_type_genes,
    plot_cancer_type_disjoint_genes,
    plot_cancer_type_pca,
    plot_cohort_heatmap,
    plot_cohort_disjoint_counts,
    plot_cohort_pca,
    plot_cohort_therapy_targets,
    plot_cohort_surface_proteins,
    plot_cohort_ctas,
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
    # class1-mhc-presentation-pathway and interferon-response merged into gene-sets.csv
    "housekeeping-genes": "Eisenberg & Levanon 2013",
    "gene-sets": "Pre-resolved gene sets (immune, oncogenic, DNA repair, MHC1 presentation, IFN response)",
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
                "TCR-T": therapy_target_gene_id_to_name("TCR-T"),
                "CAR-T": therapy_target_gene_id_to_name("CAR-T"),
                "bispecifics": therapy_target_gene_id_to_name("bispecific-antibodies"),
                "pMHC-TCEs": pMHC_TCE_target_gene_names(),
                "surface-TCEs": surface_TCE_target_gene_names(),
                "ADCs": therapy_target_gene_id_to_name("ADC"),
                "Radio": therapy_target_gene_id_to_name("radioligand"),
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

    disjoint_png = "%s-cancer-types-disjoint.png" % prefix if prefix else "cancer-types-disjoint.png"
    plot_cancer_type_disjoint_genes(df_expr, save_to_filename=disjoint_png, save_dpi=output_dpi)

    pca_png = "%s-cancer-types-scatter.png" % prefix if prefix else "cancer-types-scatter.png"
    plot_cancer_type_pca(df_expr, save_to_filename=pca_png, save_dpi=output_dpi)

    # Cancer-type-specific gene set plot (only when --cancer-type specified)
    ct_png = None
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

    # Collect all figures into one PDF (native resolution)
    from pathlib import Path
    from PIL import Image

    all_pdf = "%s-all-figures.pdf" % prefix if prefix else "all-figures.pdf"
    png_files = [
        "%s-summary.png" % prefix if prefix else "summary.png",
        "%s-treatments.png" % prefix if prefix else "treatments.png",
        genes_png,
        disjoint_png,
        pca_png,
    ]
    if ct_png:
        png_files.append(ct_png)

    # Add per-category scatter PNGs from the vs-cancer output dir
    scatter_dir = Path(scatter_pdf).parent / Path(scatter_pdf).stem
    if scatter_dir.is_dir():
        png_files.extend(sorted(str(p) for p in scatter_dir.glob("*.png")))

    images = []
    for png_path in png_files:
        p = Path(png_path)
        if p.exists():
            img = Image.open(p).convert("RGB")
            images.append(img)

    if images:
        images[0].save(all_pdf, save_all=True, append_images=images[1:], resolution=output_dpi)
        print(f"Saved {all_pdf} ({len(images)} pages)")
    else:
        print("No images to collect into PDF")


@named("plot-cancer-cohorts")
def plot_cancer_cohorts(
    output_prefix: Optional[str] = None,
    output_dpi: int = 300,
):
    """Visualize curated gene sets across all 33 TCGA cancer types (no sample needed)."""
    from pathlib import Path

    prefix = output_prefix or "cohort"
    png_files = []

    plots = [
        ("heatmap", plot_cohort_heatmap),
        ("disjoint-counts", plot_cohort_disjoint_counts),
        ("pca", plot_cohort_pca),
        ("therapy-targets", plot_cohort_therapy_targets),
        ("surface-proteins", plot_cohort_surface_proteins),
        ("ctas", plot_cohort_ctas),
    ]
    for name, fn in plots:
        out = f"{prefix}-{name}.png"
        fn(save_to_filename=out, save_dpi=output_dpi)
        png_files.append(out)

    # Collect into PDF (native resolution)
    pdf_path = f"{prefix}-all.pdf"
    images = []
    for png_path in png_files:
        if Path(png_path).exists():
            images.append(Image.open(png_path).convert("RGB"))
    if images:
        images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=output_dpi)
        print(f"Saved {pdf_path} ({len(images)} pages)")


def main():
    print_name_and_version()
    print("---")
    dispatch_commands([print_dataset_info, plot_expression, plot_cancer_cohorts])


if __name__ == "__main__":
    main()
