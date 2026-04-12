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
import json
from typing import Optional, Set

from .version import print_name_and_version
from .load_dataset import load_all_dataframes
from .tumor_purity import (
    analyze_sample,
    get_tumor_purity_parameters,
    plot_sample_summary,
    plot_tumor_purity,
)
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
    get_embedding_feature_metadata,
    estimate_tumor_expression_ranges,
    plot_tumor_expression_ranges,
    CANCER_TYPE_ALIASES,
    CANCER_TYPE_NAMES,
)
from .decomposition import (
    decompose_sample,
    get_decomposition_parameters,
    infer_sample_mode,
    plot_decomposition_summary,
)
from .sample_quality import assess_sample_quality

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


def _parse_csv_tokens(arg_value: Optional[str]):
    if arg_value is None:
        return None
    tokens = [token.strip() for token in str(arg_value).split(",") if token.strip()]
    return tokens or None


@named("analyze")
def analyze(
    input_path: str,
    output_dir: str = "pirlygenes-output",
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    label_genes: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    gene_id_col: Optional[str] = None,
    sample_id_col: Optional[str] = None,
    sample_id_value: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 14.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
    sample_mode: str = "auto",
    tumor_context: str = "auto",
    site_hint: Optional[str] = None,
    decomposition_templates: Optional[str] = None,
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
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
    )
    forced_labels = _parse_always_label_genes(label_genes)
    template_overrides = _parse_csv_tokens(decomposition_templates)

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
    analysis["sample_mode"] = infer_sample_mode(
        candidate_rows=analysis.get("candidate_trace"),
        cancer_types=[analysis["cancer_type"]] if analysis.get("cancer_type") else ([cancer_type] if cancer_type else None),
        sample_mode=sample_mode,
    )
    analysis["analysis_constraints"] = _analysis_constraints(
        cancer_type=cancer_type,
        sample_mode=sample_mode,
        tumor_context=tumor_context,
        site_hint=site_hint,
        decomposition_templates=template_overrides,
    )
    cancer_code = analysis["cancer_type"]
    purity = analysis["purity"]
    fit_quality = analysis.get("fit_quality", {})
    print(f"[analysis] Cancer type: {analysis['cancer_name']} ({cancer_code}), "
          f"score={analysis['cancer_score']:.3f}")
    if fit_quality.get("label"):
        print(f"[analysis] Fit quality: {fit_quality['label']} — {fit_quality.get('message', '')}")
    print(f"[analysis] Sample mode: {_sample_mode_display(analysis['sample_mode'])}")
    if analysis["analysis_constraints"]:
        print(f"[analysis] Constraints: {analysis['analysis_constraints']}")
    print(f"[analysis] {_purity_metric_label(analysis['sample_mode']).capitalize()}: {purity['overall_estimate']:.0%} "
          f"[{purity['overall_lower']:.0%}-{purity['overall_upper']:.0%}]")
    print(f"[analysis] Stromal enrichment: {purity['components']['stromal']['enrichment']:.1f}x vs TCGA")
    print(f"[analysis] Immune enrichment: {purity['components']['immune']['enrichment']:.1f}x vs TCGA")
    top_tissues = analysis["tissue_scores"][:3]
    tissue_str = ", ".join(f"{t} ({s:.2f})" for t, s, _ in top_tissues)
    print(f"[analysis] Top background signatures: {tissue_str}")
    mhc1 = analysis["mhc1"]
    print(f"[analysis] MHC-I: HLA-A={mhc1.get('HLA-A',0):.0f}, "
          f"HLA-B={mhc1.get('HLA-B',0):.0f}, "
          f"HLA-C={mhc1.get('HLA-C',0):.0f}, "
          f"B2M={mhc1.get('B2M',0):.0f} TPM")

    # Sample quality assessment — run after analysis so tissue_scores
    # are available for tissue-matched degradation baselines.
    quality = assess_sample_quality(df_expr, tissue_scores=analysis.get("tissue_scores"))
    analysis["quality"] = quality
    for flag in quality["flags"]:
        qtag = "[quality]" if not quality["has_issues"] else "[quality WARNING]"
        print(f"{qtag} {flag}")

    summary_png = "%s-sample-summary.png" % prefix if prefix else "sample-summary.png"
    plot_sample_summary(
        df_expr,
        cancer_type=cancer_code,
        sample_mode=analysis["sample_mode"],
        save_to_filename=summary_png,
        save_dpi=output_dpi,
    )

    print("[analysis] Running broad-compartment decomposition...")
    decomp_png = None
    candidate_codes = [row["code"] for row in analysis.get("candidate_trace", [])[:4]]
    candidate_tsv = "%s-cancer-candidates.tsv" % prefix if prefix else "cancer-candidates.tsv"
    import pandas as pd
    pd.DataFrame(
        [
            {
                "rank": idx + 1,
                "cancer_type": row["code"],
                "signature_score": row["signature_score"],
                "purity_estimate": row["purity_estimate"],
                "lineage_purity": row.get("lineage_purity"),
                "lineage_concordance": row.get("lineage_concordance"),
                "lineage_detection_fraction": row.get("lineage_detection_fraction"),
                "lineage_support_factor": row.get("lineage_support_factor"),
                "family_label": row.get("family_label"),
                "family_score": row.get("family_score"),
                "family_presence": row.get("family_presence"),
                "family_specificity": row.get("family_specificity"),
                "family_factor": row.get("family_factor"),
                "signature_stability": row.get("signature_stability"),
                "support_score": row["support_score"],
                "support_norm": row["support_norm"],
            }
            for idx, row in enumerate(analysis.get("candidate_trace", []))
        ]
    ).to_csv(candidate_tsv, sep="\t", index=False)

    params_json = "%s-analysis-parameters.json" % prefix if prefix else "analysis-parameters.json"
    with open(params_json, "w") as f:
        json.dump(
            {
                "input": {
                    "path": input_path,
                    "aggregate_gene_expression": aggregate_gene_expression,
                    "gene_name_col": gene_name_col,
                    "gene_id_col": gene_id_col,
                    "sample_id_col": sample_id_col,
                    "sample_id_value": sample_id_value,
                    "cancer_type": cancer_type,
                    "sample_mode": sample_mode,
                    "tumor_context": tumor_context,
                    "site_hint": site_hint,
                    "decomposition_templates": template_overrides,
                },
                "tumor_purity": get_tumor_purity_parameters(),
                "decomposition": get_decomposition_parameters(),
                "selected_sample_mode": analysis["sample_mode"],
                "embedding_methods": ["hierarchy", "tme"],
                "sample_quality": {
                    "degradation_level": quality["degradation"]["level"],
                    "degradation_pair_index": quality["degradation"]["long_short_ratio"],
                    "culture_level": quality["culture"]["level"],
                    "culture_stress_score": quality["culture"]["stress_score"],
                    "has_issues": quality["has_issues"],
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )

    decomp_results = decompose_sample(
        df_expr,
        cancer_types=candidate_codes or [cancer_code],
        top_k=6,
        sample_mode=analysis["sample_mode"],
        tumor_context=tumor_context,
        site_hint=site_hint,
        templates=template_overrides,
    )
    call_summary = _summarize_sample_call(
        analysis,
        decomp_results,
        sample_mode=analysis["sample_mode"],
    )
    analysis["call_summary"] = call_summary
    effective_cancer_type = cancer_code
    effective_purity = purity
    if decomp_results:
        best_decomp = decomp_results[0]
        effective_cancer_type = best_decomp.cancer_type
        effective_purity = best_decomp.purity_result or purity
        if call_summary.get("site_indeterminate"):
            print(
                f"[analysis] Possible labels: {call_summary['label_display']}; "
                f"site/template indeterminate"
            )
        else:
            print(
                f"[analysis] Best decomposition: {best_decomp.cancer_type} / {best_decomp.template}, "
                f"{_decomposition_fraction_label(analysis['sample_mode'])}={best_decomp.purity:.0%}, "
                f"score={best_decomp.score:.3f}"
            )

        # Quality-informed caveats on decomposition
        if quality["degradation"]["level"] in ("moderate", "severe"):
            print(
                "[quality WARNING] RNA degradation detected — decomposition component "
                "fractions and purity estimate may be unreliable. Long-transcript "
                "genes (>7 kb) are systematically underrepresented."
            )
            best_decomp.warnings.append(
                f"RNA degradation detected (pair index="
                f"{quality['degradation']['long_short_ratio']})"
            )
        if quality["culture"]["level"] in ("likely_cell_line", "possible_cell_line"):
            print(
                "[quality WARNING] Sample appears to be a cell line — "
                "decomposition TME components are not meaningful."
            )

        decomp_png = "%s-decomposition.png" % prefix if prefix else "decomposition.png"
        plot_decomposition_summary(
            decomp_results,
            call_summary=call_summary,
            save_to_filename=decomp_png,
            save_dpi=output_dpi,
        )

        hypotheses_tsv = "%s-decomposition-hypotheses.tsv" % prefix if prefix else "decomposition-hypotheses.tsv"
        pd.DataFrame(
            [
                {
                    "rank": idx + 1,
                    "cancer_type": row.cancer_type,
                    "template": row.template,
                    "score": row.score,
                    "purity": row.purity,
                    "reconstruction_error": row.reconstruction_error,
                    "cancer_signature_score": row.cancer_signature_score,
                    "cancer_purity_score": row.cancer_purity_score,
                    "cancer_support_score": row.cancer_support_score,
                    "template_tissue_score": row.template_tissue_score,
                    "template_origin_tissue_score": row.template_origin_tissue_score,
                    "template_site_factor": row.template_site_factor,
                    "template_extra_fraction": row.template_extra_fraction,
                    "warnings": "; ".join(row.warnings),
                }
                for idx, row in enumerate(decomp_results)
            ]
        ).to_csv(hypotheses_tsv, sep="\t", index=False)

        if not best_decomp.component_trace.empty:
            best_decomp.component_trace.to_csv(
                "%s-decomposition-components.tsv" % prefix if prefix else "decomposition-components.tsv",
                sep="\t",
                index=False,
            )
        if not best_decomp.marker_trace.empty:
            best_decomp.marker_trace.to_csv(
                "%s-decomposition-markers.tsv" % prefix if prefix else "decomposition-markers.tsv",
                sep="\t",
                index=False,
            )
        if not best_decomp.gene_attribution.empty:
            best_decomp.gene_attribution.to_csv(
                "%s-decomposition-gene-attribution.tsv" % prefix if prefix else "decomposition-gene-attribution.tsv",
                sep="\t",
                index=False,
            )

    print("[plot] Generating tumor purity detail plot...")
    purity_png = "%s-purity.png" % prefix if prefix else "purity.png"
    plot_tumor_purity(
        df_expr,
        cancer_type=effective_cancer_type,
        sample_mode=analysis["sample_mode"],
        save_to_filename=purity_png,
        save_dpi=output_dpi,
    )
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
    methods = ["hierarchy", "tme"]
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
    _embedding_meta = get_embedding_feature_metadata(method="hierarchy")
    _generate_text_reports(analysis, _embedding_meta, prefix, decomp_results=decomp_results)


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
    purity_dict = effective_purity
    adj_pngs = []
    ranges_df = None
    _adj_categories = [
        ("therapy_target", "targets"),
        ("CTA", "ctas"),
        ("surface", "surface"),
    ]
    try:
        ranges_df = estimate_tumor_expression_ranges(
            df_expr,
            cancer_type=effective_cancer_type,
            purity_result=purity_dict,
            decomposition_results=decomp_results,
        )
        ranges_tsv = "%s-tumor-expression-ranges.tsv" % prefix if prefix else "tumor-expression-ranges.tsv"
        ranges_df.to_csv(ranges_tsv, sep="\t", index=False)
        for cat_key, cat_slug in _adj_categories:
            cat_png = "%s-purity-%s.png" % (prefix, cat_slug) if prefix else "purity-%s.png" % cat_slug
            plot_tumor_expression_ranges(
                ranges_df,
                purity_result=purity_dict,
                cancer_type=effective_cancer_type,
                top_n=15,
                categories=[cat_key],
                save_to_filename=cat_png,
                save_dpi=output_dpi,
            )
            adj_pngs.append(cat_png)
            _plt.close("all")

        _generate_target_report(
            ranges_df,
            analysis,
            prefix,
            cancer_type=effective_cancer_type,
            purity_result=purity_dict,
        )
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
        decomp_png,
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
        if not png_path:
            continue
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
        if not png_path:
            continue
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
| `*-analysis.md` | Structured analysis — candidate trace, purity components, decomposition, background signatures, embedding features |
| `*-targets.md` | Therapeutic targets — CTAs (vaccination), surface proteins (ADC/CAR-T), intracellular (TCR-T), tumor-expression ranges |
| `*-analysis-parameters.json` | Free model parameters plus selected sample mode and embedding methods |
| `*-all-figures.pdf` | All figures combined into a single PDF |
| `*-cancer-candidates.tsv` | Candidate cancer-type support trace |
| `*-decomposition-hypotheses.tsv` | Ranked decomposition hypotheses |
| `*-decomposition-components.tsv` | Component-level fit for best decomposition |
| `*-decomposition-markers.tsv` | Marker-gene evidence for best decomposition |
| `*-decomposition-gene-attribution.tsv` | Per-gene TME/tumor attribution for best decomposition |
| `*-tumor-expression-ranges.tsv` | Purity-adjusted tumor-expression ranges with TCGA context |

## Figures (in `figures/`)

| Figure | Description |
|--------|-------------|
| `*-sample-summary.png` | Overview: cancer type, purity, background signatures |
| `*-decomposition.png` | Broad-compartment decomposition hypotheses and marker logic |
| `*-purity.png` | Tumor purity estimation detail |
| `*-immune.png` | Immune microenvironment gene expression |
| `*-tumor.png` | Tumor biology gene expression |
| `*-antigens.png` | Tumor antigen expression (CTAs, surfaceome) |
| `*-treatments.png` | Therapy target expression by modality |
| `*-target-safety.png` | Therapy target normal tissue expression |
| `*-purity-targets.png` | Tumor-expression ranges for therapeutic targets |
| `*-purity-ctas.png` | Tumor-expression ranges for CTAs |
| `*-purity-surface.png` | Tumor-expression ranges for surface proteins |
| `*-pca-hierarchy.png` | PCA: sample among TCGA cancer types (hierarchical support space) |
| `*-mds-hierarchy.png` | MDS: sample among TCGA cancer types (hierarchical support space) |
| `*-pca-tme.png` | PCA: TME-low genes — preferred when purity is low |
| `*-mds-tme.png` | MDS: TME-low genes — preferred when purity is low |
| `*-cancer-types-genes.png` | Cancer-type gene signature heatmap |
| `*-cancer-types-disjoint.png` | Disjoint (unique) gene counts per cancer type |
"""
    readme_path.write_text(readme)
    print(f"[output] Wrote {readme_path}")


def _sample_mode_display(sample_mode):
    labels = {
        "solid": "solid tumor / metastatic bulk",
        "heme": "hematologic / lymphoid bulk",
        "pure": "pure population / cell culture",
        "auto": "auto",
    }
    return labels.get(sample_mode, sample_mode)


def _purity_metric_label(sample_mode):
    if sample_mode == "heme":
        return "malignant-lineage fraction proxy"
    if sample_mode == "pure":
        return "population purity consistency"
    return "tumor purity"


def _summary_mode_clause(sample_mode, purity, top_tissues):
    tissue_str = ", ".join(f"{t} ({s:.2f})" for t, s, _ in top_tissues[:3])
    if sample_mode == "pure":
        return (
            f"The sample was analyzed in **pure-population mode**. The reported "
            f"purity-like estimate (**{purity['overall_estimate']:.0%}**, range "
            f"{purity['overall_lower']:.0%}–{purity['overall_upper']:.0%}) is best read as "
            "a coherence check against the matched lineage profile rather than as a bulk admixture fraction. "
            f"Residual background signatures are limited ({tissue_str}). "
        )
    if sample_mode == "heme":
        return (
            f"The estimated **malignant-lineage fraction proxy** is **{purity['overall_estimate']:.0%}** "
            f"(range {purity['overall_lower']:.0%}–{purity['overall_upper']:.0%}). "
            "In heme mode this is not a strict tumor-vs-immune split; it reflects how strongly the sample "
            "resembles the matched malignant program relative to hematopoietic background. "
            f"Top lineage/background contexts: {tissue_str}. "
        )
    return (
        f"Estimated tumor purity is **{purity['overall_estimate']:.0%}** "
        f"(range {purity['overall_lower']:.0%}–{purity['overall_upper']:.0%}), "
        f"with {purity['components']['stromal']['enrichment']:.1f}x stromal "
        f"and {purity['components']['immune']['enrichment']:.1f}x immune enrichment "
        f"vs TCGA median. "
        f"Top background signatures: {tissue_str}. "
        f"These tissue matches describe residual non-tumor background and are not literal site calls. "
    )


def _background_section_config(sample_mode):
    if sample_mode == "pure":
        return (
            "Residual Background Signatures",
            "In pure-population mode these scores are mostly a contamination check. "
            "Small residual matches do not imply a real anatomical site.\n",
        )
    if sample_mode == "heme":
        return (
            "Lineage / Background Context",
            "These scores summarize hematopoietic or tissue-context programs that remain in the sample. "
            "For heme specimens they should be read as lineage/background context rather than metastatic site calls.\n",
        )
    return (
        "Background Tissue Signatures",
        "These scores summarize which residual non-tumor backgrounds the sample "
        "resembles after normalization. They are useful context, not literal "
        "anatomical site calls in mixed samples.\n",
    )


def _decomposition_section_title(sample_mode):
    if sample_mode == "pure":
        return "Pure-Population Fit"
    if sample_mode == "heme":
        return "Heme Context Decomposition"
    return "Broad Decomposition"


def _decomposition_fraction_label(sample_mode):
    if sample_mode == "pure":
        return "population purity"
    if sample_mode == "heme":
        return "malignant-lineage fraction"
    return "tumor fraction"


def _target_report_mode_intro(sample_mode, cancer_code, p_lo, p_mid, p_hi):
    if sample_mode == "pure":
        return (
            f"Population-expression range using purity-like consistency **{p_lo:.0%} / {p_mid:.0%} / {p_hi:.0%}** "
            f"(low / estimate / high). In pure mode these values are close to direct observed cellular expression rather "
            f"than a bulk tumor-vs-background deconvolution against {cancer_code} TCGA.\n"
        )
    if sample_mode == "heme":
        return (
            f"Malignant-lineage expression range using fraction proxy **{p_lo:.0%} / {p_mid:.0%} / {p_hi:.0%}** "
            f"(low / estimate / high). In heme mode these values reflect malignant-lineage-enriched expression under "
            f"hematopoietic background assumptions rather than a strict epithelial tumor-vs-immune subtraction.\n"
        )
    return (
        f"Purity-adjusted tumor expression range using purity **{p_lo:.0%} / {p_mid:.0%} / {p_hi:.0%}** "
        f"(low / estimate / high).\n"
    )


def _target_value_label(sample_mode):
    if sample_mode == "pure":
        return "Cellular TPM"
    if sample_mode == "heme":
        return "Malignant TPM"
    return "Tumor TPM"


def _candidate_label_options(analysis):
    candidate_trace = analysis.get("candidate_trace", [])
    fit_quality = analysis.get("fit_quality", {})
    if not candidate_trace:
        return []
    labels = [candidate_trace[0]["code"]]
    if len(candidate_trace) >= 2 and fit_quality.get("label") in {"weak", "ambiguous"}:
        labels.append(candidate_trace[1]["code"])
    return labels[:2]


def _template_site_display(template_name):
    mapping = {
        "met_adrenal": "adrenal",
        "met_bone": "bone",
        "met_brain": "brain",
        "met_liver": "liver",
        "met_lung": "lung",
        "met_lymph_node": "lymph node",
        "met_peritoneal": "peritoneum",
        "met_skin": "skin",
        "met_soft_tissue": "soft tissue",
        "solid_primary": "primary site",
        "pure_population": "pure population",
        "heme_blood": "blood",
        "heme_marrow": "marrow",
        "heme_nodal": "lymphoid / nodal",
    }
    return mapping.get(template_name, template_name.replace("_", " "))


def _analysis_constraints(
    cancer_type=None,
    sample_mode="auto",
    tumor_context="auto",
    site_hint=None,
    decomposition_templates=None,
):
    constraints = {}
    if cancer_type:
        constraints["cancer_type"] = cancer_type
    if sample_mode and sample_mode != "auto":
        constraints["sample_mode"] = sample_mode
    if tumor_context and tumor_context != "auto":
        constraints["tumor_context"] = tumor_context
    if site_hint:
        constraints["site_hint"] = site_hint
    if decomposition_templates:
        constraints["decomposition_templates"] = list(decomposition_templates)
    return constraints


def _summarize_sample_call(analysis, decomp_results, sample_mode):
    fit_quality = analysis.get("fit_quality", {})
    label_options = _candidate_label_options(analysis)
    best = decomp_results[0] if decomp_results else None
    hypothesis_options = []
    if decomp_results:
        hypothesis_options.append(decomp_results[0])
        if len(decomp_results) >= 2:
            second = decomp_results[1]
            score_ratio = (
                float(decomp_results[0].score / second.score)
                if second.score not in (0, None)
                else None
            )
            if fit_quality.get("label") in {"weak", "ambiguous"} or (score_ratio is not None and score_ratio < 1.2):
                hypothesis_options.append(second)

    site_indeterminate = False
    context_indeterminate = False
    reported_context = None
    reported_site = None
    site_note = None

    if best is not None:
        if sample_mode == "pure":
            reported_context = "pure"
        elif sample_mode == "heme":
            reported_context = "heme"
        elif best.template == "solid_primary":
            reported_context = "primary"
        elif best.template.startswith("met_"):
            reported_context = "met"

        if fit_quality.get("label") == "weak" and best.template.startswith("met_"):
            site_indeterminate = True
            context_indeterminate = True
            site_note = "Weak subtype fit prevents a reliable metastatic site call."
        elif best.template.startswith("met_"):
            if (
                "Primary tissue support exceeds metastatic-site support" in best.warnings
                or (best.template_site_factor or 0.0) < 0.75
                or (best.template_tissue_score or 0.0) < 0.4
            ):
                site_indeterminate = True
                context_indeterminate = True
                site_note = "Host-site evidence is not strong enough to trust a specific metastatic template."

        if not site_indeterminate:
            reported_site = _template_site_display(best.template)

    label_display = " or ".join(label_options) if label_options else analysis.get("cancer_type")
    hypothesis_display = [
        f"{row.cancer_type} / {row.template}"
        for row in hypothesis_options[:2]
    ]
    return {
        "label_options": label_options,
        "label_display": label_display,
        "reported_context": None if context_indeterminate else reported_context,
        "reported_site": reported_site,
        "site_indeterminate": site_indeterminate,
        "site_note": site_note,
        "hypothesis_display": hypothesis_display,
    }


def _generate_text_reports(analysis, embedding_meta, prefix, decomp_results=None):
    """Write summary and detailed analysis markdown reports."""
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    purity = analysis["purity"]
    mhc1 = analysis["mhc1"]
    top_tissues = analysis["tissue_scores"][:5]
    candidate_trace = analysis.get("candidate_trace", [])
    family_summary = analysis.get("family_summary", {})
    fit_quality = analysis.get("fit_quality", {})
    sample_mode = analysis.get("sample_mode", "auto")
    constraints = analysis.get("analysis_constraints", {})
    call_summary = analysis.get("call_summary") or _summarize_sample_call(
        analysis,
        decomp_results or [],
        sample_mode=sample_mode,
    )
    best_decomp = decomp_results[0] if decomp_results else None

    # --- Summary report ---
    ambiguity_clause = ""
    if len(call_summary.get("label_options", [])) == 2:
        ambiguity_clause = (
            f" Possible labels: **{call_summary['label_options'][0]}** or "
            f"**{call_summary['label_options'][1]}**."
        )
    hla_a = mhc1.get("HLA-A", 0)
    hla_b = mhc1.get("HLA-B", 0)
    b2m = mhc1.get("B2M", 0)
    mhc_level = "high" if min(hla_a, hla_b, b2m) > 20 else (
        "low" if max(hla_a, hla_b, b2m) < 5 else "moderate"
    )
    family_display = family_summary.get("display")
    subtype_clause = family_summary.get("subtype_clause")
    lead_candidate = candidate_trace[0]["code"] if candidate_trace else None
    constrained_cancer = constraints.get("cancer_type")
    if family_display:
        intro = f"The sample most closely matches **{family_display}**"
        if subtype_clause:
            if constrained_cancer and lead_candidate and constrained_cancer != lead_candidate:
                intro += f", with **{cancer_code}** as the constrained working subtype"
            else:
                intro += f", with **{cancer_code}** as the current best subtype hypothesis"
        intro += f" (score {analysis['cancer_score']:.3f}). "
    else:
        intro = (
            f"The sample most closely matches **{cancer_name} ({cancer_code})** "
            f"(score {analysis['cancer_score']:.3f}). "
        )
    summary = intro + _summary_mode_clause(sample_mode, purity, top_tissues) + (
        f"MHC-I expression is {mhc_level} "
        f"(HLA-A={hla_a:.0f}, HLA-B={hla_b:.0f}, B2M={b2m:.0f} TPM). "
        f"Analysis mode: **{_sample_mode_display(sample_mode)}**."
    )
    if fit_quality.get("message"):
        summary += f" {fit_quality['message']}"
    if constraints:
        constraint_parts = []
        if constraints.get("cancer_type"):
            constraint_parts.append(f"cancer type fixed to **{constraints['cancer_type']}**")
        if constraints.get("tumor_context"):
            constraint_parts.append(
                f"template context restricted to **{constraints['tumor_context']}**"
            )
        if constraints.get("site_hint"):
            constraint_parts.append(f"site hint **{constraints['site_hint']}**")
        if constraints.get("decomposition_templates"):
            constraint_parts.append(
                "template list fixed to **"
                + ", ".join(constraints["decomposition_templates"])
                + "**"
            )
        if constraint_parts:
            summary += " Analysis constraints: " + "; ".join(constraint_parts) + "."
    if call_summary.get("site_indeterminate"):
        summary += " Decomposition recovered broad admixture structure, but site/template assignment is indeterminate."
    elif best_decomp is not None:
        summary += (
            f" Best {_decomposition_section_title(sample_mode).lower()}: "
            f"**{best_decomp.cancer_type} / {best_decomp.template}** "
            f"with {_decomposition_fraction_label(sample_mode)} **{best_decomp.purity:.0%}**."
        )
    summary += ambiguity_clause

    # Quality flags in summary
    quality = analysis.get("quality")
    if quality and quality.get("has_issues"):
        summary += "\n\n**Quality warnings**: " + "; ".join(quality["flags"]) + "."
    elif quality and quality["degradation"]["level"] != "normal":
        summary += "\n\n**Quality note**: " + "; ".join(quality["flags"]) + "."

    summary_path = "%s-summary.md" % prefix if prefix else "summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Sample Analysis Summary\n\n{summary}\n")
    print(f"[report] Saved {summary_path}")

    # --- Detailed report ---
    lines = ["# Detailed Sample Analysis\n"]

    # Sample quality
    quality = analysis.get("quality")
    if quality:
        lines.append("## Sample Quality\n")
        deg = quality["degradation"]
        cul = quality["culture"]
        lines.append(f"**RNA degradation**: {deg['level']}")
        lines.append(f"- Mitochondrial fraction: {deg['mt_fraction']:.1%}")
        lines.append(f"- Ribosomal protein fraction: {deg['rp_fraction']:.1%}")
        if deg.get("matched_tissue"):
            lines.append(f"- Matched tissue baseline: {deg['matched_tissue']} "
                         f"(MT={deg['baseline_mt']:.1%}, RP={deg['baseline_rp']:.1%})")
            if deg.get("mt_fold") is not None:
                lines.append(f"- Fold over baseline: MT {deg['mt_fold']:.1f}×, RP {deg['rp_fold']:.1f}×")
        if deg["level"] != "normal":
            lines.append(f"- *{deg['message']}*")
        lines.append("")
        lines.append(f"**Cell culture / cell line**: {cul['level'].replace('_', ' ')}")
        lines.append(f"- Culture-stress z-score: {cul['stress_score']:.1f}")
        lines.append(f"- TME marker mean: {cul['tme_mean_tpm']:.1f} TPM "
                     f"({'absent' if cul['tme_absent'] else 'present'})")
        if cul["top_stress_genes"]:
            top_genes_str = ", ".join(f"{g}={t:.0f}" for g, t in cul["top_stress_genes"][:5])
            lines.append(f"- Top stress genes: {top_genes_str}")
        if cul["level"] != "normal":
            lines.append(f"- *{cul['message']}*")
        lines.append("")

    # Cancer type identification
    lines.append("## Cancer Type Identification\n")
    lines.append(f"- **Sample mode**: {_sample_mode_display(sample_mode)}")
    if fit_quality.get("label"):
        lines.append(f"- **Fit quality**: {fit_quality['label']}")
    if fit_quality.get("message"):
        lines.append(f"- **Fit note**: {fit_quality['message']}")
    if call_summary.get("label_options"):
        if len(call_summary["label_options"]) == 1:
            lines.append(f"- **Resolved label**: {call_summary['label_options'][0]}")
        else:
            lines.append(
                f"- **Possible labels**: {call_summary['label_options'][0]} or {call_summary['label_options'][1]}"
            )
    if family_display:
        lines.append(f"- **Family-level call**: {family_display}")
        if subtype_clause:
            lines.append(f"- **Subtype ordering within family**: {subtype_clause}")
    if constraints:
        if constraints.get("cancer_type"):
            lines.append(f"- **User-constrained cancer type**: {constraints['cancer_type']}")
        if constraints.get("sample_mode"):
            lines.append(
                f"- **Requested sample mode**: {_sample_mode_display(constraints['sample_mode'])}"
            )
        if constraints.get("tumor_context"):
            lines.append(f"- **Requested tumor context**: {constraints['tumor_context']}")
        if constraints.get("site_hint"):
            lines.append(f"- **Requested site hint**: {constraints['site_hint']}")
        if constraints.get("decomposition_templates"):
            lines.append(
                "- **Requested decomposition templates**: "
                + ", ".join(constraints["decomposition_templates"])
            )
    top_cancers = analysis.get("top_cancers", [(cancer_code, analysis["cancer_score"])])
    for code, score in top_cancers[:5]:
        name = CANCER_TYPE_NAMES.get(code, code)
        lines.append(f"- **{code}** ({name}): {score:.3f}")
    lines.append("")

    if candidate_trace:
        lines.append("### Candidate Trace\n")
        lines.append("| Cancer | Family | Support | Signature | Purity | Lineage | Concordance |")
        lines.append("|--------|--------|---------|-----------|--------|---------|-------------|")
        for row in candidate_trace[:8]:
            lineage = row.get("lineage_purity")
            concordance = row.get("lineage_concordance")
            lines.append(
                f"| {row['code']} | {row.get('family_label') or '—'} | {row['support_score']:.3f} | {row['signature_score']:.3f} | "
                f"{row['purity_estimate']:.3f} | "
                f"{'%.3f' % lineage if lineage is not None else '—'} | "
                f"{'%.3f' % concordance if concordance is not None else '—'} |"
            )
        lines.append("")

    # Embedding features
    lines.append("## Embedding Features\n")
    lines.append(f"- **Method**: {embedding_meta.get('method', 'unknown')}")
    feature_kind = embedding_meta.get("feature_kind")
    if feature_kind == "hierarchical_scores":
        lines.append(
            f"- **Feature space**: {embedding_meta['n_features']} features built from "
            "family-aware cancer support scores, site/background context, and a purity anchor"
        )
        if embedding_meta.get("families"):
            lines.append(
                f"- **Families represented**: {', '.join(embedding_meta['families'])}"
            )
        if embedding_meta.get("sites"):
            lines.append(
                f"- **Site/background axes**: {', '.join(embedding_meta['sites'][:8])}"
                + (" ..." if len(embedding_meta["sites"]) > 8 else "")
            )
        lines.append(
            f"- **Cancer types represented**: {embedding_meta['n_types']}/33"
        )
        lines.append("")
        lines.append(
            "The hierarchy embedding uses the same broad-family gating and subtype "
            "support logic as the main classifier, while adding host/background "
            "context axes. The 2D embedding therefore reflects the same evidence "
            "hierarchy shown in the candidate trace rather than a flat gene-only space."
        )
    else:
        lines.append(f"- **Total genes**: {embedding_meta['n_genes']}")
        lines.append(f"- **Cancer types represented**: {embedding_meta['n_types']}/33")
        if embedding_meta.get("fallback_types"):
            lines.append(
                f"- **Fallback types** (z-score only, no S/N filter): "
                f"{', '.join(embedding_meta['fallback_types'])}"
            )
        if embedding_meta.get("cta_added"):
            lines.append(f"- **Curated CTAs added**: {', '.join(embedding_meta['cta_added'])}")
        lines.append("")
        lines.append("### Genes per cancer type\n")
        lines.append("| Cancer | Genes |")
        lines.append("|--------|-------|")
        for ct in sorted(embedding_meta["per_type"]):
            genes = embedding_meta["per_type"][ct]
            if genes:
                lines.append(f"| {ct} | {', '.join(genes)} |")
    lines.append("")

    # Purity / composition
    lines.append(f"## {_purity_metric_label(sample_mode).title()}\n")
    lines.append(f"- **Overall estimate**: {purity['overall_estimate']:.0%} "
                  f"({purity['overall_lower']:.0%}\u2013{purity['overall_upper']:.0%})")
    components = purity.get("components", {})
    for comp_name in ("stromal", "immune"):
        comp = components.get(comp_name, {})
        if isinstance(comp, dict):
            enrichment = comp.get("enrichment", 0)
            lines.append(f"- **{comp_name.title()}** enrichment: {enrichment:.1f}x vs TCGA")
    if sample_mode == "pure":
        lines.append(
            "- **Interpretation**: in pure-population mode this estimate is a consistency check "
            "against the matched lineage profile, not a bulk admixture fraction."
        )
    elif sample_mode == "heme":
        lines.append(
            "- **Interpretation**: in heme mode this estimate is a malignant-lineage proxy, "
            "not a strict tumor-vs-immune split."
        )

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
            if all(lineage.get(key) is not None for key in ("purity", "lower", "upper")):
                lines.append(
                    f"**Reliable cluster** ({lineage['purity']:.0%}, "
                    f"IQR {lineage['lower']:.0%}\u2013{lineage['upper']:.0%}): "
                    f"{retained_names}. "
                    "These genes are expressed at levels consistent with their "
                    "TCGA reference, indicating retained tumor lineage identity."
                )
            else:
                lines.append(
                    f"**Reliable cluster**: {retained_names}. "
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

    # Background / context signatures
    bg_title, bg_intro = _background_section_config(sample_mode)
    lines.append(f"## {bg_title}\n")
    lines.append(bg_intro)
    lines.append("| Tissue | Score | N genes |")
    lines.append("|--------|-------|---------|")
    for tissue, score, n in top_tissues:
        lines.append(f"| {tissue} | {score:.3f} | {n} |")
    lines.append("")

    if decomp_results or call_summary.get("site_indeterminate") or call_summary.get("hypothesis_display"):
        lines.append(f"## {_decomposition_section_title(sample_mode)}\n")
        if call_summary.get("site_indeterminate"):
            lines.append("Reported site/template call: **indeterminate**.\n")
            if call_summary.get("site_note"):
                lines.append(call_summary["site_note"] + "\n")
        elif call_summary.get("reported_site") is not None:
            lines.append(
                f"Reported template/site call: **{call_summary['reported_site']}**"
                + (
                    f" ({call_summary['reported_context']})\n"
                    if call_summary.get("reported_context")
                    else "\n"
                )
            )
        if len(call_summary.get("hypothesis_display", [])) == 2:
            lines.append(
                f"Top broad possibilities: **{call_summary['hypothesis_display'][0]}** "
                f"or **{call_summary['hypothesis_display'][1]}**.\n"
            )
        if decomp_results:
            lines.append("| Hypothesis | Score | Fraction | Tissue | Warnings |")
            lines.append("|------------|-------|--------|--------|----------|")
            for row in decomp_results[:6]:
                warnings = "; ".join(row.warnings) if row.warnings else ""
                lines.append(
                    f"| {row.cancer_type} / {row.template} | {row.score:.3f} | "
                    f"{row.purity:.3f} | {row.template_tissue_score:.3f} | {warnings} |"
                )
            lines.append("")

        if best_decomp is not None and not best_decomp.component_trace.empty:
            lines.append("### Best-Fit Components\n")
            lines.append("| Component | Fraction | Marker score | Top markers |")
            lines.append("|-----------|----------|--------------|-------------|")
            for _, row in best_decomp.component_trace.iterrows():
                lines.append(
                    f"| {row['component']} | {row['fraction']:.3f} | "
                    f"{row['marker_score'] if row['marker_score'] is not None else '—'} | "
                    f"{row['top_markers']} |"
                )
            lines.append("")

    analysis_path = "%s-analysis.md" % prefix if prefix else "analysis.md"
    with open(analysis_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] Saved {analysis_path}")


def _generate_target_report(ranges_df, analysis, prefix, cancer_type, purity_result):
    """Write tumor-expression range report using purity/decomposition bounds."""
    import pandas as pd

    cancer_code = cancer_type
    cancer_name = CANCER_TYPE_NAMES.get(cancer_code, cancer_code)
    sample_mode = analysis.get("sample_mode", "solid")
    value_label = _target_value_label(sample_mode)
    p_lo = purity_result["overall_lower"]
    p_mid = purity_result["overall_estimate"]
    p_hi = purity_result["overall_upper"]

    lines = [f"# Therapeutic Target Analysis — {cancer_code} ({cancer_name})\n"]
    lines.append(_target_report_mode_intro(sample_mode, cancer_code, p_lo, p_mid, p_hi))
    if sample_mode == "pure":
        lines.append(
            "Each gene is reported as a bounded expression estimate around the observed sample value, "
            f"then contextualized against the matched {cancer_code} TCGA cohort.\n"
        )
    elif sample_mode == "heme":
        lines.append(
            "Each gene is reported as a bounded malignant-lineage-enriched expression estimate across "
            f"hematopoietic background assumptions, then contextualized against the matched {cancer_code} "
            "TCGA cohort.\n"
        )
    else:
        lines.append(
            "Each gene is reported as a bounded deconvolution across purity and "
            "TME-background assumptions, then contextualized against the matched "
            f"{cancer_code} TCGA cohort.\n"
        )

    # --- CTAs: vaccination targets ---
    ctas = ranges_df[ranges_df["is_cta"] & (ranges_df["median_est"] > 0.5)].copy()
    lines.append("## Cancer-Testis Antigens (Vaccination Targets)\n")
    lines.append("CTAs are expressed in tumor but not normal adult tissue (except testis/placenta). "
                 "Any expressed CTA is a potential vaccination target regardless of trial status.\n")
    if len(ctas):
        lines.append(f"| Gene | {value_label} | Range | Observed | vs TCGA | TCGA %ile | Surface | Therapies |")
        lines.append("|------|-----------|-------|----------|---------|-----------|---------|-----------|")
        for _, row in ctas.head(20).iterrows():
            surf = "yes" if row["is_surface"] else ""
            vs_tcga = row["pct_cancer_median"]
            lines.append(
                f"| **{row['symbol']}** | {row['median_est']:.1f} | "
                f"{row['est_1']:.1f}\u2013{row['est_9']:.1f} | {row['observed_tpm']:.1f} | "
                f"{'%.1fx' % vs_tcga if pd.notna(vs_tcga) else '—'} | {row['tcga_percentile']:.0%} | "
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
    surface_targets = ranges_df[
        ranges_df["is_surface"] & (ranges_df["median_est"] > 1) & ~ranges_df["is_cta"]
    ].copy()
    lines.append("## Surface Protein Targets (ADC / CAR-T / Bispecific)\n")
    lines.append("Surface proteins with high purity-adjusted expression. "
                 "These can be targeted by antibody-drug conjugates, CAR-T, "
                 "or bispecific T-cell engagers.\n")
    if len(surface_targets):
        lines.append(f"| Gene | {value_label} | Range | Observed | vs TCGA | TCGA %ile | Therapies |")
        lines.append("|------|-----------|-------|----------|---------|-----------|-----------|")
        for _, row in surface_targets.head(20).iterrows():
            bold = "**" if row["therapies"] else ""
            vs_tcga = row["pct_cancer_median"]
            lines.append(
                f"| {bold}{row['symbol']}{bold} | {row['median_est']:.1f} | "
                f"{row['est_1']:.1f}\u2013{row['est_9']:.1f} | {row['observed_tpm']:.1f} | "
                f"{'%.1fx' % vs_tcga if pd.notna(vs_tcga) else '—'} | {row['tcga_percentile']:.0%} | "
                f"{row['therapies']} |"
            )
    else:
        lines.append("No surface targets above threshold.\n")
    lines.append("")

    # --- Cytosolic / intracellular targets (TCR-T, pMHC) ---
    intracellular = ranges_df[
        ~ranges_df["is_surface"] & (ranges_df["median_est"] > 5)
        & (ranges_df["category"].isin(["therapy_target", "CTA"]))
    ].copy()
    lines.append("## Intracellular Targets (TCR-T / pMHC Vaccination)\n")
    lines.append("Intracellular proteins presented via MHC-I. Targetable by "
                 "TCR-T cell therapy or peptide vaccination.\n")
    if len(intracellular):
        lines.append(f"| Gene | {value_label} | Range | vs TCGA | TCGA %ile | CTA | Therapies |")
        lines.append("|------|-----------|-------|---------|-----------|-----|-----------|")
        for _, row in intracellular.head(15).iterrows():
            cta_flag = "yes" if row["is_cta"] else ""
            vs_tcga = row["pct_cancer_median"]
            lines.append(
                f"| {row['symbol']} | {row['median_est']:.1f} | "
                f"{row['est_1']:.1f}\u2013{row['est_9']:.1f} | "
                f"{'%.1fx' % vs_tcga if pd.notna(vs_tcga) else '—'} | "
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
            lines.append(
                f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
                f"range {row['est_1']:.0f}\u2013{row['est_9']:.0f}){therapy_note}"
            )
        lines.append("")

    # Best CTAs
    best_cta = ctas.head(3)
    if len(best_cta):
        lines.append("**Best CTA targets** (vaccination even without active trials):")
        for _, row in best_cta.iterrows():
            lines.append(f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
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
    sample_id_col: Optional[str] = None,
    sample_id_value: Optional[str] = None,
    output_dpi: int = 300,
    plot_height: float = 14.0,
    plot_aspect: float = 1.4,
    cancer_type: Optional[str] = None,
    sample_mode: str = "auto",
    tumor_context: str = "auto",
    site_hint: Optional[str] = None,
    decomposition_templates: Optional[str] = None,
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
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        output_dpi=output_dpi,
        plot_height=plot_height,
        plot_aspect=plot_aspect,
        cancer_type=cancer_type,
        sample_mode=sample_mode,
        tumor_context=tumor_context,
        site_hint=site_hint,
        decomposition_templates=decomposition_templates,
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
    import sys
    # Handle --version / -V before dispatching to subcommands.  argh's
    # dispatch_commands uses argparse under the hood and rejects unknown
    # top-level flags, so without this, `pirlygenes --version` would print
    # the banner and then error with "unrecognized arguments: --version".
    if len(sys.argv) >= 2 and sys.argv[1] in ("--version", "-V"):
        print_name_and_version()
        return
    print_name_and_version()
    print("---")
    dispatch_commands([print_dataset_info, analyze, plot_expression, plot_cancer_cohorts])


if __name__ == "__main__":
    main()
