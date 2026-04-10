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
from .tumor_purity import analyze_sample, plot_sample_summary, plot_tumor_purity
from .gene_sets_cancer import (
    therapy_target_gene_id_to_name,
    pMHC_TCE_target_gene_id_to_name,
    surface_TCE_target_gene_id_to_name,
    cancer_types,
    cancer_type_gene_sets,
)
from PIL import Image
from .load_expression import load_expression_data
from .plot import (
    plot_gene_expression,
    plot_sample_vs_cancer,
    plot_cancer_type_genes,
    plot_cancer_type_disjoint_genes,
    plot_cancer_type_pca,
    plot_cancer_type_mds,
    plot_therapy_target_tissues,
    plot_therapy_target_safety,
    plot_cohort_heatmap,
    plot_cohort_disjoint_counts,
    plot_cohort_pca,
    plot_cohort_therapy_targets,
    plot_cohort_surface_proteins,
    plot_cohort_ctas,
    default_gene_sets,
    _select_embedding_genes_bottleneck,
    estimate_tumor_expression,
    estimate_tumor_expression_ranges,
    plot_tumor_expression_ranges,
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


@named("analyze")
def analyze(
    input_path: str,
    output_dir: str = "pirlygenes-output",
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    label_genes: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    gene_id_col: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 14.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
    therapy_target_top_k: int = 10,
    therapy_target_tpm_threshold: float = 30.0,
):
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Writing to {out_dir}/")

    # Build prefix: output_dir / image_prefix (or just output_dir/)
    if output_image_prefix:
        prefix = str(out_dir / output_image_prefix)
    else:
        prefix = str(out_dir / "sample")

    df_expr = load_expression_data(
        input_path,
        aggregate_gene_expression=aggregate_gene_expression,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
    )
    forced_labels = _parse_always_label_genes(label_genes)

    # Strip plots: split into focused panels for readability
    # Immune microenvironment
    immune_sets = {k: default_gene_sets[k] for k in
                   ["Immune_checkpoints", "MHC1_presentation", "Interferon_response", "TLR"]
                   if k in default_gene_sets}
    # Tumor biology
    tumor_sets = {k: default_gene_sets[k] for k in
                  ["Oncogenes", "Tumor_suppressors", "DNA_repair", "Growth_receptors"]
                  if k in default_gene_sets}
    # Tumor antigens
    antigen_sets = {k: default_gene_sets[k] for k in
                    ["CTAs", "Cancer_surfaceome"]
                    if k in default_gene_sets}
    # Therapy modalities
    therapy_sets = {
        "TCR-T": therapy_target_gene_id_to_name("TCR-T"),
        "CAR-T": therapy_target_gene_id_to_name("CAR-T"),
        "bispecifics": therapy_target_gene_id_to_name("bispecific-antibodies"),
        "pMHC-TCEs": pMHC_TCE_target_gene_id_to_name(),
        "surface-TCEs": surface_TCE_target_gene_id_to_name(),
        "ADCs": therapy_target_gene_id_to_name("ADC"),
        "Radio": therapy_target_gene_id_to_name("radioligand"),
    }

    strip_plots = [
        ("immune", immune_sets),
        ("tumor", tumor_sets),
        ("antigens", antigen_sets),
        ("treatments", therapy_sets),
    ]
    for i, (name, gene_sets) in enumerate(strip_plots):
        output_image = (
            "%s-%s.png" % (prefix, name) if prefix else "%s.png" % name
        )
        print(f"[plot] Generating {name} strip plot...")
        plot_gene_expression(
            df_expr,
            gene_sets=gene_sets,
            save_to_filename=output_image,
            save_dpi=output_dpi,
            plot_height=plot_height,
            plot_aspect=plot_aspect,
            always_label_genes=forced_labels,
            verbose=(i == 0),  # only log remaps on first call
            source_file=input_path,
        )

    import matplotlib.pyplot as _plt
    _plt.close("all")

    # Sample composition analysis
    print("[analysis] Running sample composition analysis...")
    analysis = analyze_sample(df_expr, cancer_type=cancer_type)
    cancer_code = analysis["cancer_type"]
    purity = analysis["purity"]
    print(f"[analysis] Cancer type: {analysis['cancer_name']} ({cancer_code}), "
          f"score={analysis['cancer_score']:.3f}")
    print(f"[analysis] Tumor purity: {purity['overall_estimate']:.0%} "
          f"[{purity['overall_lower']:.0%}-{purity['overall_upper']:.0%}]")
    print(f"[analysis] Stromal enrichment: {purity['components']['stromal']['enrichment']:.1f}x vs TCGA")
    print(f"[analysis] Immune enrichment: {purity['components']['immune']['enrichment']:.1f}x vs TCGA")
    top_tissues = analysis["tissue_scores"][:3]
    tissue_str = ", ".join(f"{t} ({s:.2f})" for t, s, _ in top_tissues)
    print(f"[analysis] Top tissue context: {tissue_str}")
    mhc1 = analysis["mhc1"]
    print(f"[analysis] MHC-I: HLA-A={mhc1.get('HLA-A',0):.0f}, "
          f"HLA-B={mhc1.get('HLA-B',0):.0f}, "
          f"HLA-C={mhc1.get('HLA-C',0):.0f}, "
          f"B2M={mhc1.get('B2M',0):.0f} TPM")

    summary_png = "%s-sample-summary.png" % prefix if prefix else "sample-summary.png"
    plot_sample_summary(df_expr, cancer_type=cancer_code, save_to_filename=summary_png, save_dpi=output_dpi)

    print("[plot] Generating tumor purity detail plot...")
    purity_png = "%s-purity.png" % prefix if prefix else "purity.png"
    plot_tumor_purity(df_expr, cancer_type=cancer_code, save_to_filename=purity_png, save_dpi=output_dpi)
    _plt.close("all")

    # Scatter plots: sample vs pan-cancer reference
    print("[plot] Generating sample vs cancer scatter plots...")
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
    _plt.close("all")

    # Therapy target tissue expression / safety
    print("[plot] Generating therapy target tissue expression...")
    tissue_pdf = "%s-target-tissues.pdf" % prefix if prefix else "target-tissues.pdf"
    plot_therapy_target_tissues(
        df_expr,
        top_k=therapy_target_top_k,
        tpm_threshold=therapy_target_tpm_threshold,
        save_to_filename=tissue_pdf,
        save_dpi=output_dpi,
    )

    print("[plot] Generating therapy target safety plot...")
    safety_png = "%s-target-safety.png" % prefix if prefix else "target-safety.png"
    plot_therapy_target_safety(
        df_expr,
        top_k=therapy_target_top_k,
        tpm_threshold=therapy_target_tpm_threshold,
        save_to_filename=safety_png,
        save_dpi=output_dpi,
    )
    _plt.close("all")

    # Cancer type signature plots
    print("[plot] Generating cancer type signature gene plots...")
    genes_png = "%s-cancer-types-genes.png" % prefix if prefix else "cancer-types-genes.png"
    plot_cancer_type_genes(df_expr, save_to_filename=genes_png, save_dpi=output_dpi)

    disjoint_png = "%s-cancer-types-disjoint.png" % prefix if prefix else "cancer-types-disjoint.png"
    plot_cancer_type_disjoint_genes(df_expr, save_to_filename=disjoint_png, save_dpi=output_dpi)
    _plt.close("all")

    # PCA and MDS — bottleneck (general) + TME-low (preferred at low purity)
    methods = ["bottleneck", "tme"]
    print(f"[plot] Generating PCA/MDS embeddings ({len(methods)} methods x 2 embeddings)...")
    embedding_pngs = []
    for method in methods:
        pca_png = "%s-pca-%s.png" % (prefix, method)
        plot_cancer_type_pca(df_expr, method=method, save_to_filename=pca_png, save_dpi=output_dpi)
        embedding_pngs.append(pca_png)

        mds_png = "%s-mds-%s.png" % (prefix, method)
        plot_cancer_type_mds(df_expr, method=method, save_to_filename=mds_png, save_dpi=output_dpi)
        embedding_pngs.append(mds_png)
    _plt.close("all")

    # Generate text reports
    print("[report] Generating text reports...")
    _gene_meta = _select_embedding_genes_bottleneck()[1]
    _generate_text_reports(analysis, _gene_meta, prefix)


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
                source_file=input_path,
            )

    # Purity-adjusted tumor expression analysis (9-point ranges, one PNG per category)
    print("[plot] Generating tumor expression range analysis...")
    purity_dict = analysis["purity"]
    adj_pngs = []
    _adj_categories = [
        ("therapy_target", "targets"),
        ("CTA", "ctas"),
        ("surface", "surface"),
    ]
    try:
        ranges_df = estimate_tumor_expression_ranges(
            df_expr,
            cancer_type=analysis["cancer_type"],
            purity_result=purity_dict,
        )
        for cat_key, cat_slug in _adj_categories:
            cat_png = "%s-purity-%s.png" % (prefix, cat_slug) if prefix else "purity-%s.png" % cat_slug
            plot_tumor_expression_ranges(
                ranges_df,
                purity_result=purity_dict,
                cancer_type=analysis["cancer_type"],
                top_n=15,
                categories=[cat_key],
                save_to_filename=cat_png,
                save_dpi=output_dpi,
            )
            adj_pngs.append(cat_png)
            _plt.close("all")

        # Generate therapeutic target report (legacy single-point for table)
        purity_est = purity_dict["overall_estimate"]
        adj_df = estimate_tumor_expression(
            df_expr,
            cancer_type=analysis["cancer_type"],
            purity=purity_est,
        )
        _generate_target_report(adj_df, analysis, prefix)
    except Exception as e:
        print(f"[warn] Purity-adjusted analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Collect all figures into one PDF (native resolution)
    from pathlib import Path
    from PIL import Image

    all_pdf = "%s-all-figures.pdf" % prefix if prefix else "all-figures.pdf"
    print("[output] Collecting figures into PDF...")
    png_files = [
        summary_png,
        purity_png,
        "%s-immune.png" % prefix if prefix else "immune.png",
        "%s-tumor.png" % prefix if prefix else "tumor.png",
        "%s-antigens.png" % prefix if prefix else "antigens.png",
        "%s-treatments.png" % prefix if prefix else "treatments.png",
        safety_png,
        genes_png,
        disjoint_png,
    ] + embedding_pngs
    if ct_png:
        png_files.append(ct_png)

    # Add per-category scatter PNGs from the vs-cancer output dir
    scatter_dir = Path(scatter_pdf).parent / Path(scatter_pdf).stem
    if scatter_dir.is_dir():
        png_files.extend(sorted(str(p) for p in scatter_dir.glob("*.png")))

    # Purity-adjusted plots go last (different RNA measure)
    for adj_p in adj_pngs:
        if Path(adj_p).exists():
            png_files.append(adj_p)

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

    # Move PNGs and per-figure PDFs into figures/ subdir,
    # keeping all-figures.pdf and markdown reports in place.
    fig_out_dir = Path(prefix).parent
    figures_dir = fig_out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    moved = 0
    for png_path in png_files:
        p = Path(png_path)
        if p.exists() and p.suffix == ".png":
            p.rename(figures_dir / p.name)
            moved += 1
    # Move scatter dir contents and per-plot PDFs
    if scatter_dir.is_dir():
        for p in scatter_dir.glob("*.png"):
            p.rename(figures_dir / p.name)
            moved += 1
        # Remove empty scatter dir
        try:
            scatter_dir.rmdir()
        except OSError:
            pass
    for extra in [scatter_pdf, tissue_pdf]:
        p = Path(extra) if isinstance(extra, str) else extra
        if p.exists():
            p.rename(figures_dir / p.name)
            moved += 1
    if moved:
        print(f"[output] Moved {moved} figures to {figures_dir}/")

    # Write README explaining output files
    readme_path = Path(prefix).parent / "README.md"
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    readme = f"""# PIRLy Genes Analysis Output

Sample analyzed as **{cancer_code}** ({cancer_name}).

## Reports

| File | Description |
|------|-------------|
| `*-summary.md` | One-paragraph natural language summary — cancer type, purity, key findings |
| `*-analysis.md` | Structured analysis — cancer type ranking, purity components, MHC expression, tissue context, embedding genes |
| `*-targets.md` | Therapeutic targets — CTAs (vaccination), surface proteins (ADC/CAR-T), intracellular (TCR-T), purity-adjusted expression |
| `*-all-figures.pdf` | All figures combined into a single PDF |

## Figures (in `figures/`)

| Figure | Description |
|--------|-------------|
| `*-sample-summary.png` | Overview: cancer type, purity, tissue context |
| `*-purity.png` | Tumor purity estimation detail |
| `*-immune.png` | Immune microenvironment gene expression |
| `*-tumor.png` | Tumor biology gene expression |
| `*-antigens.png` | Tumor antigen expression (CTAs, surfaceome) |
| `*-treatments.png` | Therapy target expression by modality |
| `*-target-safety.png` | Therapy target normal tissue expression |
| `*-purity-adjusted.png` | Purity-adjusted expression by target category |
| `*-pca-bottleneck.png` | PCA: sample among TCGA cancer types (bottleneck genes) |
| `*-mds-bottleneck.png` | MDS: sample among TCGA cancer types (bottleneck genes) |
| `*-pca-tme.png` | PCA: TME-low genes — preferred when purity is low |
| `*-mds-tme.png` | MDS: TME-low genes — preferred when purity is low |
| `*-cancer-types-genes.png` | Cancer-type gene signature heatmap |
| `*-cancer-types-disjoint.png` | Disjoint (unique) gene counts per cancer type |
"""
    readme_path.write_text(readme)
    print(f"[output] Wrote {readme_path}")


def _generate_text_reports(analysis, gene_meta, prefix):
    """Write summary and detailed analysis markdown reports."""
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    purity = analysis["purity"]
    mhc1 = analysis["mhc1"]
    top_tissues = analysis["tissue_scores"][:5]

    # --- Summary report ---
    tissue_str = ", ".join(f"{t} ({s:.2f})" for t, s, _ in top_tissues[:3])
    hla_a = mhc1.get("HLA-A", 0)
    hla_b = mhc1.get("HLA-B", 0)
    b2m = mhc1.get("B2M", 0)
    mhc_level = "high" if min(hla_a, hla_b, b2m) > 20 else (
        "low" if max(hla_a, hla_b, b2m) < 5 else "moderate"
    )
    summary = (
        f"The sample most closely matches **{cancer_name} ({cancer_code})** "
        f"(score {analysis['cancer_score']:.3f}). "
        f"Estimated tumor purity is **{purity['overall_estimate']:.0%}** "
        f"(range {purity['overall_lower']:.0%}\u2013{purity['overall_upper']:.0%}), "
        f"with {purity['components']['stromal']['enrichment']:.1f}x stromal "
        f"and {purity['components']['immune']['enrichment']:.1f}x immune enrichment "
        f"vs TCGA median. "
        f"Top tissue context: {tissue_str}. "
        f"MHC-I expression is {mhc_level} "
        f"(HLA-A={hla_a:.0f}, HLA-B={hla_b:.0f}, B2M={b2m:.0f} TPM)."
    )
    summary_path = "%s-summary.md" % prefix if prefix else "summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Sample Analysis Summary\n\n{summary}\n")
    print(f"[report] Saved {summary_path}")

    # --- Detailed report ---
    lines = ["# Detailed Sample Analysis\n"]

    # Cancer type identification
    lines.append("## Cancer Type Identification\n")
    top_cancers = analysis.get("top_cancers", [(cancer_code, analysis["cancer_score"])])
    for code, score in top_cancers[:5]:
        name = CANCER_TYPE_NAMES.get(code, code)
        lines.append(f"- **{code}** ({name}): {score:.3f}")
    lines.append("")

    # Gene selection
    lines.append("## Embedding Gene Selection\n")
    lines.append(f"- **Method**: {gene_meta.get('method', 'unknown')}")
    lines.append(f"- **Total genes**: {gene_meta['n_genes']}")
    lines.append(f"- **Cancer types represented**: {gene_meta['n_types']}/33")
    if gene_meta.get("fallback_types"):
        lines.append(f"- **Fallback types** (z-score only, no S/N filter): "
                      f"{', '.join(gene_meta['fallback_types'])}")
    if gene_meta.get("cta_added"):
        lines.append(f"- **Curated CTAs added**: {', '.join(gene_meta['cta_added'])}")
    lines.append("")
    lines.append("### Genes per cancer type\n")
    lines.append("| Cancer | Genes |")
    lines.append("|--------|-------|")
    for ct in sorted(gene_meta["per_type"]):
        genes = gene_meta["per_type"][ct]
        if genes:
            lines.append(f"| {ct} | {', '.join(genes)} |")
    lines.append("")

    # Tumor purity
    lines.append("## Tumor Purity\n")
    lines.append(f"- **Overall estimate**: {purity['overall_estimate']:.0%} "
                  f"({purity['overall_lower']:.0%}\u2013{purity['overall_upper']:.0%})")
    components = purity.get("components", {})
    for comp_name in ("stromal", "immune"):
        comp = components.get(comp_name, {})
        if isinstance(comp, dict):
            enrichment = comp.get("enrichment", 0)
            lines.append(f"- **{comp_name.title()}** enrichment: {enrichment:.1f}x vs TCGA")

    # Lineage gene narrative
    lineage = components.get("lineage", {})
    lineage_genes = lineage.get("per_gene", [])
    if lineage_genes:
        lines.append("")
        lines.append("### Lineage Gene Calibration\n")
        lines.append(
            "Purity was refined using cancer-type lineage genes — genes with "
            "known high expression in this tumor type and low TME background. "
            "Each gene independently estimates purity by comparing the sample's "
            "HK-normalized expression to the TCGA reference (adjusted for "
            "TCGA cohort purity).\n"
        )

        # Sort genes into clusters
        sorted_genes = sorted(lineage_genes, key=lambda g: g["purity"], reverse=True)
        median_p = lineage.get("purity")

        # Identify retained vs de-differentiated
        if median_p is not None and median_p > 0:
            retained = [g for g in sorted_genes if g["purity"] >= median_p * 0.5]
            lost = [g for g in sorted_genes if g["purity"] < median_p * 0.5]
        else:
            retained = sorted_genes
            lost = []

        # Not found in sample
        from .tumor_purity import LINEAGE_GENES
        cancer_code_local = purity.get("cancer_type", cancer_code)
        all_lineage = LINEAGE_GENES.get(cancer_code_local, [])
        found_names = {g["gene"] for g in lineage_genes}
        not_found = [g for g in all_lineage if g not in found_names]

        lines.append("| Gene | Purity est. | Interpretation |")
        lines.append("|------|------------|----------------|")
        for g in sorted_genes:
            if g in retained:
                interp = "retained — reliable"
            else:
                interp = "likely de-differentiated"
            lines.append(
                f"| {g['gene']} | {g['purity']:.1%} | {interp} |"
            )
        for g in not_found:
            lines.append(f"| {g} | — | not detected |")

        lines.append("")

        if retained:
            retained_names = ", ".join(g["gene"] for g in retained)
            lines.append(
                f"**Reliable cluster** ({lineage['purity']:.0%}, "
                f"IQR {lineage['lower']:.0%}\u2013{lineage['upper']:.0%}): "
                f"{retained_names}. "
                "These genes are expressed at levels consistent with their "
                "TCGA reference, indicating retained tumor lineage identity."
            )
        if lost:
            lost_names = ", ".join(g["gene"] for g in lost)
            lines.append(
                f"\n**Possible de-differentiation**: {lost_names}. "
                "These genes give much lower purity estimates, suggesting "
                "the tumor may have lost expression of these markers — "
                "common in metastatic or treatment-resistant disease. "
                "These are excluded from the purity estimate."
            )
        if not_found:
            lines.append(
                f"\n**Not detected**: {', '.join(not_found)}."
            )

    lines.append("")

    # MHC expression
    lines.append("## MHC Expression\n")
    lines.append("| Gene | TPM |")
    lines.append("|------|-----|")
    for gene in ["HLA-A", "HLA-B", "HLA-C", "B2M"]:
        lines.append(f"| {gene} | {mhc1.get(gene, 0):.0f} |")
    mhc2 = analysis.get("mhc2", {})
    if mhc2:
        for gene, val in sorted(mhc2.items()):
            lines.append(f"| {gene} | {val:.0f} |")
    lines.append("")

    # Normal tissue context
    lines.append("## Normal Tissue Context\n")
    lines.append("| Tissue | Score | N genes |")
    lines.append("|--------|-------|---------|")
    for tissue, score, n in top_tissues:
        lines.append(f"| {tissue} | {score:.3f} | {n} |")
    lines.append("")

    analysis_path = "%s-analysis.md" % prefix if prefix else "analysis.md"
    with open(analysis_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] Saved {analysis_path}")


def _generate_target_report(adj_df, analysis, prefix):
    """Write purity-adjusted therapeutic target analysis report."""
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    purity_est = analysis["purity"]["overall_estimate"]

    lines = [f"# Therapeutic Target Analysis — {cancer_code} ({cancer_name})\n"]
    lines.append(f"Purity-adjusted expression (estimated purity: {purity_est:.0%}).\n")
    lines.append("Expression values are corrected for TME contamination: "
                 "`tumor_expr = (observed - (1-purity) * tme_reference) / purity`\n")

    # --- CTAs: vaccination targets ---
    ctas = adj_df[adj_df["is_cta"] & (adj_df["tumor_adjusted"] > 0.5)].copy()
    lines.append("## Cancer-Testis Antigens (Vaccination Targets)\n")
    lines.append("CTAs are expressed in tumor but not normal adult tissue (except testis/placenta). "
                 "Any expressed CTA is a potential vaccination target regardless of trial status.\n")
    if len(ctas):
        lines.append("| Gene | Tumor TPM | Observed | TCGA %ile | Surface | Therapies |")
        lines.append("|------|-----------|----------|-----------|---------|-----------|")
        for _, row in ctas.head(20).iterrows():
            surf = "yes" if row["is_surface"] else ""
            lines.append(
                f"| **{row['symbol']}** | {row['tumor_adjusted']:.1f} | "
                f"{row['observed_tpm']:.1f} | {row['tcga_percentile']:.0%} | "
                f"{surf} | {row['therapies']} |"
            )
        high_ctas = ctas[ctas["tcga_percentile"] > 0.7]
        if len(high_ctas):
            names = ", ".join(high_ctas["symbol"].head(5))
            lines.append(f"\n**Above TCGA median for {cancer_code}**: {names}")
    else:
        lines.append("No CTAs detected above threshold.\n")
    lines.append("")

    # --- Surface therapy targets ---
    surface_targets = adj_df[
        adj_df["is_surface"] & (adj_df["tumor_adjusted"] > 1) & ~adj_df["is_cta"]
    ].copy()
    lines.append("## Surface Protein Targets (ADC / CAR-T / Bispecific)\n")
    lines.append("Surface proteins with high purity-adjusted expression. "
                 "These can be targeted by antibody-drug conjugates, CAR-T, "
                 "or bispecific T-cell engagers.\n")
    if len(surface_targets):
        lines.append("| Gene | Tumor TPM | Observed | TCGA %ile | Therapies |")
        lines.append("|------|-----------|----------|-----------|-----------|")
        for _, row in surface_targets.head(20).iterrows():
            bold = "**" if row["therapies"] else ""
            lines.append(
                f"| {bold}{row['symbol']}{bold} | {row['tumor_adjusted']:.1f} | "
                f"{row['observed_tpm']:.1f} | {row['tcga_percentile']:.0%} | "
                f"{row['therapies']} |"
            )
    else:
        lines.append("No surface targets above threshold.\n")
    lines.append("")

    # --- Cytosolic / intracellular targets (TCR-T, pMHC) ---
    intracellular = adj_df[
        ~adj_df["is_surface"] & (adj_df["tumor_adjusted"] > 5)
        & (adj_df["category"].isin(["therapy_target", "CTA"]))
    ].copy()
    lines.append("## Intracellular Targets (TCR-T / pMHC Vaccination)\n")
    lines.append("Intracellular proteins presented via MHC-I. Targetable by "
                 "TCR-T cell therapy or peptide vaccination.\n")
    if len(intracellular):
        lines.append("| Gene | Tumor TPM | TCGA %ile | CTA | Therapies |")
        lines.append("|------|-----------|-----------|-----|-----------|")
        for _, row in intracellular.head(15).iterrows():
            cta_flag = "yes" if row["is_cta"] else ""
            lines.append(
                f"| {row['symbol']} | {row['tumor_adjusted']:.1f} | "
                f"{row['tcga_percentile']:.0%} | {cta_flag} | {row['therapies']} |"
            )
    else:
        lines.append("No intracellular targets above threshold.\n")
    lines.append("")

    # --- Top recommendation summary ---
    lines.append("## Recommended Targets Summary\n")

    # Best surface
    best_surface = surface_targets.head(3)
    if len(best_surface):
        lines.append("**Best surface targets** (ADC/CAR-T/bispecific):")
        for _, row in best_surface.iterrows():
            therapy_note = f" — active in {row['therapies']}" if row["therapies"] else ""
            lines.append(f"- **{row['symbol']}** ({row['tumor_adjusted']:.0f} TPM){therapy_note}")
        lines.append("")

    # Best CTAs
    best_cta = ctas.head(3)
    if len(best_cta):
        lines.append("**Best CTA targets** (vaccination even without active trials):")
        for _, row in best_cta.iterrows():
            lines.append(f"- **{row['symbol']}** ({row['tumor_adjusted']:.0f} TPM, "
                         f"TCGA {row['tcga_percentile']:.0%})")
        lines.append("")

    # MHC context for intracellular targeting
    mhc1 = analysis.get("mhc1", {})
    b2m = mhc1.get("B2M", 0)
    hla_mean = sum(mhc1.get(g, 0) for g in ["HLA-A", "HLA-B", "HLA-C"]) / 3
    if hla_mean > 50 and b2m > 100:
        lines.append(f"**MHC-I status**: adequate (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) "
                     "— intracellular targets are presentable.")
    elif hla_mean > 10:
        lines.append(f"**MHC-I status**: reduced (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) "
                     "— intracellular targeting may have limited efficacy.")
    else:
        lines.append(f"**MHC-I status**: low/absent (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) "
                     "— intracellular targets unlikely to be presented. Prioritize surface targets.")
    lines.append("")

    target_path = "%s-targets.md" % prefix if prefix else "targets.md"
    with open(target_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] Saved {target_path}")


@named("plot-expression")
def plot_expression(
    input_path: str,
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    label_genes: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    gene_id_col: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 14.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
    therapy_target_top_k: int = 10,
    therapy_target_tpm_threshold: float = 30.0,
):
    """Deprecated: use 'analyze' instead."""
    import warnings
    warnings.warn(
        "plot-expression is deprecated, use 'analyze' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return analyze(
        input_path,
        output_image_prefix=output_image_prefix,
        aggregate_gene_expression=aggregate_gene_expression,
        label_genes=label_genes,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        output_dpi=output_dpi,
        plot_height=plot_height,
        plot_aspect=plot_aspect,
        cancer_type=cancer_type,
        therapy_target_top_k=therapy_target_top_k,
        therapy_target_tpm_threshold=therapy_target_tpm_threshold,
    )


@named("plot-cancer-cohorts")
def plot_cancer_cohorts(
    output_prefix: Optional[str] = None,
    output_dpi: int = 300,
):
    """Visualize curated gene sets across all 33 TCGA cancer types (no sample needed)."""
    from pathlib import Path

    prefix = output_prefix or "cohort"
    png_files = []

    # Plots without zscore parameter
    simple_plots = [
        ("disjoint-counts", plot_cohort_disjoint_counts),
        ("pca", plot_cohort_pca),
    ]
    for name, fn in simple_plots:
        out = f"{prefix}-{name}.png"
        fn(save_to_filename=out, save_dpi=output_dpi)
        png_files.append(out)

    # Heatmaps: emit both z-score and HK-normalized versions
    heatmap_plots = [
        ("heatmap", plot_cohort_heatmap),
        ("therapy-targets", plot_cohort_therapy_targets),
        ("surface-proteins", plot_cohort_surface_proteins),
        ("ctas", plot_cohort_ctas),
    ]
    for name, fn in heatmap_plots:
        for suffix, zs in [("zscore", True), ("hk", False)]:
            out = f"{prefix}-{name}-{suffix}.png"
            fn(save_to_filename=out, save_dpi=output_dpi, zscore=zs)
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
    dispatch_commands([print_dataset_info, analyze, plot_expression, plot_cancer_cohorts])


if __name__ == "__main__":
    main()
