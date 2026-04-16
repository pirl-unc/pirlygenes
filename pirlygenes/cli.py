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
from pathlib import Path
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
    plot_matched_normal_attribution,
    plot_tumor_expression_ranges,
    CANCER_TYPE_ALIASES,
    CANCER_TYPE_NAMES,
)
from .decomposition import (
    decompose_sample,
    epithelial_matched_normal_component,
    get_decomposition_parameters,
    infer_sample_mode,
    plot_decomposition_candidates,
    plot_decomposition_component_breakdown,
    plot_decomposition_composition,
)
from .sample_context import (
    infer_sample_context,
    plot_degradation_index,
    plot_sample_context,
)
from .sample_quality import assess_sample_quality
from .therapy_response import score_therapy_signatures

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


_MT_EXPECTED_MISSING_PREPS = frozenset({"poly_a", "exome_capture"})

# Genes we treat as "AR-transactivation output" — the downstream AR-
# target program that collapses in castrate-resistant PRAD. Used by
# the synthesis layer (#78) to detect the AR-retained-but-targets-
# collapsed pattern typical of ADT-treated CRPC.
_PRAD_AR_TARGET_GENES = frozenset({
    "KLK3", "KLK2", "NKX3-1", "HOXB13", "FOLH1", "TMPRSS2",
    "SLC45A3", "NDRG1", "PMEPA1", "FKBP5",
})

# Genes annotated in the therapy_response AR_signaling *up* panel but
# also core ISGs — used to tag per-gene surface-target fold-changes as
# "IFN-driven" when the IFN_response axis is active.
_CORE_ISG_SURFACE = frozenset({
    "HLA-A", "HLA-B", "HLA-C", "HLA-F", "HLA-E",
    "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DRA", "HLA-DRB1",
    "B2M", "TAP1", "TAP2",
    "STAT1", "IRF1", "ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "OAS2",
    "CXCL9", "CXCL10",
})


def compose_disease_state_narrative(analysis) -> str:
    """Synthesize a one-line disease-state narrative from combined
    signals (issue #78).

    Draws on the candidate trace, lineage-gene calibration, and
    therapy-response axis scores. Cancer-type-family-specific rules;
    PRAD covers the castrate-resistant / NEPC pattern first since
    that's the validation case (Tempus FFPE PRAD sample: AR retained
    at 51% + KLK3/KLK2/NKX3-1/HOXB13 all < 2% + NE markers elevated
    = textbook ADT-treated CRPC trending toward NEPC).

    Returns an empty string when no family rule matches — callers
    skip rendering in that case.
    """
    cancer_code = analysis.get("cancer_type")
    therapy_scores = analysis.get("therapy_response_scores") or {}
    purity = analysis.get("purity") or {}
    components = purity.get("components") or {}
    lineage = components.get("lineage") or {}
    per_gene = lineage.get("per_gene") or []

    # Bucket lineage genes by their independent purity estimate.
    retained: set[str] = set()
    collapsed: set[str] = set()
    for g in per_gene:
        sym = g.get("gene")
        est = g.get("purity")
        if not sym or est is None:
            continue
        if est >= 0.30:
            retained.add(sym)
        elif est < 0.05:
            collapsed.add(sym)

    # Axis states
    def _state(cls):
        s = therapy_scores.get(cls)
        return s.state if s is not None else None

    ar_state = _state("AR_signaling")
    ne_state = _state("NE_differentiation")
    emt_state = _state("EMT")
    ifn_state = _state("IFN_response")
    hypoxia_state = _state("hypoxia")
    er_state = _state("ER_signaling")
    her2_state = _state("HER2_signaling")

    parts: list[str] = []

    if cancer_code == "PRAD":
        ar_retained = "AR" in retained
        ar_targets_collapsed = len(_PRAD_AR_TARGET_GENES & collapsed) >= 3
        if ar_retained and ar_targets_collapsed and ar_state == "down":
            verb = (
                "**Castrate-resistant pattern**: AR receptor retained "
                "while AR-transactivation targets "
                f"({', '.join(sorted(_PRAD_AR_TARGET_GENES & collapsed))}) "
                "are collapsed — consistent with prior ADT exposure"
            )
            if ne_state == "up":
                verb += (
                    " with **emerging neuroendocrine differentiation** "
                    "(NE markers elevated; workup for NEPC warranted)"
                )
            verb += "."
            parts.append(verb)
        elif ar_state == "down":
            parts.append(
                "**AR axis suppressed** — consistent with ADT exposure. "
                "Lineage AR was not retained in the sample; insufficient "
                "signal for a castrate-resistant call."
            )
    elif cancer_code in ("BRCA",):
        if er_state == "down" and "ESR1" in collapsed:
            parts.append(
                "**ER-axis suppressed / endocrine-exposed pattern** "
                "(ESR1 low, classic ER targets collapsed)."
            )
        if her2_state == "up":
            parts.append(
                "**HER2-amplification pattern** (ERBB2 / GRB7 / STARD3 "
                "co-elevated)."
            )

    # Cross-axis observation: EMT + hypoxia together typically signal
    # an aggressive / treatment-resistant phenotype regardless of
    # cancer type.
    if emt_state == "up" and hypoxia_state == "up":
        parts.append(
            "EMT and hypoxia programs are both active — aggressive-"
            "phenotype pattern."
        )
    elif emt_state == "up":
        parts.append("EMT program is active (mesenchymal switch).")

    # Active IFN response affects how we read high-fold-change
    # surface / MHC targets; mention so the reader knows the targets
    # table carries IFN-driven inflation.
    if ifn_state == "up":
        parts.append(
            "**Active IFN response** — MHC-I / ISG surface fold-changes "
            "in the therapy-target tables carry IFN-driven inflation "
            "and are not tumor-cell-specific."
        )

    return " ".join(parts)


def annotate_surface_targets_with_cross_signals(ranges_df, therapy_scores):
    """Return a map ``{symbol: note}`` annotating surface / intracellular
    targets whose elevation is likely *driven by* an active therapy-
    response axis rather than tumor-cell specificity (issue #78).

    Currently covers IFN: when ``IFN_response`` axis is active and the
    gene is a core ISG, tag with "IFN-driven". The renderer attaches
    these notes inline in the target tables so a high fold-change
    isn't read as pure tumor-cell selectivity.
    """
    ifn = therapy_scores.get("IFN_response")
    if ifn is None or getattr(ifn, "state", None) != "up":
        return {}
    return {sym: "IFN-driven" for sym in _CORE_ISG_SURFACE}


def _filter_quality_flags_against_context(flags, sample_context):
    """Drop / rewrite quality flags that the stage-1 SampleContext
    already explains (issue #77).

    Specifically: the "Suspicious MT fraction" warning fires whenever
    MT detection is near zero, but that's *expected* under the poly-A
    and exome-capture library preps we already inferred. In those
    cases the warning is noise — replace it with an informational
    line so readers see the reason rather than a false alarm.
    """
    if sample_context is None:
        return list(flags)
    prep = getattr(sample_context, "library_prep", None)
    out = []
    for flag in flags:
        if "Suspicious MT fraction" in flag and prep in _MT_EXPECTED_MISSING_PREPS:
            prep_label = prep.replace("_", " ")
            out.append(
                f"MT fraction near zero — consistent with {prep_label} "
                "library prep; degradation signal from MT fold is not "
                "assessable (this is informational, not a warning)"
            )
        else:
            out.append(flag)
    return out


def _ci_confidence_tier(overall_lower, overall_upper):
    """Map a (lower, upper) span to a confidence tier (issue #79).

    Reader-facing tags on the purity estimate so a 19–100% CI is
    visibly different from a 58–70% CI in the report.
    """
    try:
        span = float(overall_upper) - float(overall_lower)
    except (TypeError, ValueError):
        return "unknown"
    if span < 0.15:
        return "high"
    if span < 0.35:
        return "moderate"
    return "low"


def _default_output_dir() -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"pirlygenes-{ts}"


def _clean_prefix_outputs(out_dir: Path, prefix_path: str) -> int:
    """Delete stale files from a prior run sharing the same output prefix.

    Removes files in ``out_dir`` (and the ``figures/`` subdir) whose name
    starts with ``{prefix_basename}-``, so a partial rerun cannot leave
    misleading artifacts from an earlier successful run. Only touches
    files matching our prefix — anything else the user placed in the
    directory is preserved (issue #30).

    Also removes ``{prefix_basename}-vs-cancer`` scatter subdirectories.
    Returns the number of files deleted.
    """
    base = Path(prefix_path).name
    if not base:
        return 0
    stale_prefix = f"{base}-"
    removed = 0
    for directory in (out_dir, out_dir / "figures"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.name.startswith(stale_prefix):
                path.unlink()
                removed += 1
            elif path.is_dir() and path.name.startswith(stale_prefix):
                for child in path.iterdir():
                    if child.is_file():
                        child.unlink()
                        removed += 1
                try:
                    path.rmdir()
                except OSError:
                    pass
    return removed


_TX_HEADER_TOKENS = frozenset({
    "name", "target_id", "transcript_id", "transcript",
    "transcriptid", "targetid", "effectivelength", "numreads",
})
_GENE_HEADER_TOKENS = frozenset({
    "gene", "gene symbol", "gene_symbol", "genesymbol",
    "geneid", "gene_id", "ensembl_gene_id",
})


def _sniff_input_level(path: str) -> str:
    """Read the header of a tabular file and guess whether it contains
    transcript-level or gene-level quantification.

    Returns ``"transcript"`` or ``"gene"``.
    """
    import csv
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t" if path.endswith((".sf", ".tsv")) else ",")
        try:
            header = next(reader)
        except StopIteration:
            return "gene"
    lower = {h.strip().lower() for h in header}
    if lower & _TX_HEADER_TOKENS:
        return "transcript"
    return "gene"


@named("analyze")
def analyze(
    input_path: str,
    output_dir: str = "pirlygenes-output",
    output_image_prefix: Optional[str] = None,
    aggregate_gene_expression: bool = False,
    genes: Optional[str] = None,
    transcripts: Optional[str] = None,
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
    met_site: Optional[str] = None,
    decomposition_templates: Optional[str] = None,
    therapy_target_top_k: int = 10,
    therapy_target_tpm_threshold: float = 30.0,
):
    """Analyze gene expression from a quantification file.

    The positional argument auto-detects whether the input is
    transcript-level (salmon quant.sf, kallisto abundance.tsv) or
    gene-level. Transcript-level inputs are aggregated to gene-level
    automatically. Use --genes / --transcripts for explicit control
    or to supply both simultaneously::

        pirlygenes analyze quant.sf                    # auto-detect: transcript → aggregate
        pirlygenes analyze gene_tpm.csv                # auto-detect: gene-level
        pirlygenes analyze --genes g.csv --transcripts quant.sf  # explicit both
    """
    from pathlib import Path

    # Validate met_site before any I/O so bad values fail fast.
    if met_site is not None:
        from .plot import MET_SITE_TISSUE_AUGMENTATION as _MET_SITE_MAP
        if met_site not in _MET_SITE_MAP:
            raise ValueError(
                f"--met-site must be one of {sorted(_MET_SITE_MAP.keys())}, got {met_site!r}"
            )

    # --- Resolve input -------------------------------------------------
    if not genes and not transcripts:
        if aggregate_gene_expression:
            transcripts = input_path
        else:
            level = _sniff_input_level(input_path)
            if level == "transcript":
                transcripts = input_path
                aggregate_gene_expression = True
                print("[input] Auto-detected transcript-level input, will aggregate to gene level")
            else:
                genes = input_path

    if transcripts and not genes:
        aggregate_gene_expression = True

    gene_input = genes or transcripts
    transcript_input = transcripts if genes else None

    if not output_dir or output_dir == "pirlygenes-output":
        output_dir = _default_output_dir()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Writing to {out_dir}/")

    _analyze_body(
        input_path=gene_input,
        out_dir=out_dir,
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
        met_site=met_site,
        transcript_path=transcript_input,
        decomposition_templates=decomposition_templates,
        therapy_target_top_k=therapy_target_top_k,
        therapy_target_tpm_threshold=therapy_target_tpm_threshold,
    )


def _analyze_body(
    input_path: str,
    out_dir: Path,
    output_image_prefix: Optional[str],
    aggregate_gene_expression: bool,
    label_genes: Optional[str],
    gene_name_col: Optional[str],
    gene_id_col: Optional[str],
    sample_id_col: Optional[str],
    sample_id_value: Optional[str],
    output_dpi: int,
    plot_height: float,
    plot_aspect: float,
    cancer_type: Optional[str],
    sample_mode: str,
    tumor_context: str,
    site_hint: Optional[str],
    met_site: Optional[str],
    transcript_path: Optional[str],
    decomposition_templates: Optional[str],
    therapy_target_top_k: int,
    therapy_target_tpm_threshold: float,
):
    if output_image_prefix:
        prefix = str(out_dir / output_image_prefix)
    else:
        prefix = str(out_dir / "sample")

    stale_removed = _clean_prefix_outputs(out_dir, prefix)
    if stale_removed:
        print(f"[output] Removed {stale_removed} stale files from prior run")

    df_expr = load_expression_data(
        input_path,
        aggregate_gene_expression=aggregate_gene_expression,
        gene_name_col=gene_name_col,
        gene_id_col=gene_id_col,
        sample_id_col=sample_id_col,
        sample_id_value=sample_id_value,
        transcript_path=transcript_path,
    )
    forced_labels = _parse_always_label_genes(label_genes)
    template_overrides = _parse_csv_tokens(decomposition_templates)

    # Stage 1 of the unified attribution flow: infer SampleContext BEFORE
    # cancer-type inference. Downstream stages (purity CIs, decomposition,
    # tumor-value adjustment, reporting) read from it as the base layer
    # of expression expectations.
    print("[context] Inferring sample context (library prep + preservation)...")
    sample_context = infer_sample_context(df_expr)
    print(f"[context] {sample_context.summary_line()}")
    for flag in sample_context.flags:
        print(f"[context] {flag}")

    context_png = "%s-sample-context.png" % prefix if prefix else "sample-context.png"
    try:
        plot_sample_context(sample_context, save_to_filename=context_png, save_dpi=output_dpi)
        print(f"[plot] Saved sample-context diagnostic to {context_png}")
    except Exception as exc:  # noqa: BLE001 — plotting must not break analyze
        print(f"[plot] sample-context plot failed: {exc}")

    # #27: gene-pair degradation index scatter. Emitted whenever any
    # degradation signal is available (including the "none" call, so
    # users can visually confirm a non-degraded sample lies on the
    # diagonal).
    degradation_png = (
        "%s-degradation-index.png" % prefix if prefix else "degradation-index.png"
    )
    try:
        out = plot_degradation_index(
            df_expr, sample_context,
            save_to_filename=degradation_png, save_dpi=output_dpi,
        )
        if out:
            print(f"[plot] Saved degradation-index scatter to {degradation_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] degradation-index plot failed: {exc}")

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
    analysis["sample_context"] = sample_context
    analysis["cancer_type_source"] = "user-specified" if cancer_type else "auto-detected"

    # Stage 1 propagation: widen purity confidence intervals under
    # detected degradation (#26). A noisier sample has a noisier purity
    # estimate; we don't re-estimate, just scale the reported band and
    # attach a ``degradation_caveat`` so downstream consumers (reports,
    # downstream analyses) can cite the reason for the wider band
    # without having to re-derive it from the raw sample_context.
    ci_factor = sample_context.purity_ci_widening_factor()
    if ci_factor > 1.0 and "purity" in analysis:
        purity_block = analysis["purity"]
        est = purity_block.get("overall_estimate")
        lo = purity_block.get("overall_lower")
        hi = purity_block.get("overall_upper")
        if est is not None and lo is not None and hi is not None:
            half_lo = max(0.0, est - lo) * ci_factor
            half_hi = max(0.0, hi - est) * ci_factor
            purity_block["overall_lower"] = round(max(0.0, est - half_lo), 4)
            purity_block["overall_upper"] = round(min(1.0, est + half_hi), 4)
            purity_block["ci_widening_factor"] = round(ci_factor, 3)
            purity_block["degradation_caveat"] = {
                "severity": sample_context.degradation_severity,
                "index": sample_context.degradation_index,
                "message": (
                    f"Purity confidence interval widened ×{ci_factor:.2f} "
                    f"to reflect {sample_context.degradation_severity} "
                    "RNA degradation — tumor-specific genes with long "
                    "transcripts are under-represented, biasing the "
                    "point estimate low and the precision high."
                ),
            }
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
        met_site=met_site,
    )
    cancer_code = analysis["cancer_type"]
    purity = analysis["purity"]
    fit_quality = analysis.get("fit_quality", {})
    candidate_trace_for_print = analysis.get("candidate_trace", [])
    top_row = candidate_trace_for_print[0] if candidate_trace_for_print else None
    if top_row:
        parts = [
            f"signature={top_row.get('signature_score', 0.0):.2f}",
            f"geomean={top_row.get('support_geomean', 0.0):.2f}",
            f"normalized={top_row.get('support_norm', 0.0):.2f}",
        ]
        if len(candidate_trace_for_print) > 1:
            runner = candidate_trace_for_print[1]
            parts.append(
                f"(runner-up {runner['code']} {runner.get('support_norm', 0.0):.2f})"
            )
        print(f"[analysis] Cancer type: {analysis['cancer_name']} ({cancer_code}), " + ", ".join(parts))
    else:
        print(f"[analysis] Cancer type: {analysis['cancer_name']} ({cancer_code})")
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
    # #77: filter quality flags against the stage-1 SampleContext — the
    # "Suspicious MT fraction" warning is a false alarm when the
    # library prep we already inferred (exome capture / poly-A) legitimately
    # strips MT. Same filtered list is used by the markdown reports,
    # so the three documents agree on the same set of concerns.
    filtered_flags = _filter_quality_flags_against_context(
        quality["flags"], sample_context
    )
    quality["filtered_flags"] = filtered_flags
    for flag in filtered_flags:
        qtag = "[quality]" if not quality["has_issues"] else "[quality WARNING]"
        print(f"{qtag} {flag}")

    # Therapy-response signatures (#57) — score each applicable axis
    # (AR / ER / HER2 / MAPK-EGFR / NE / EMT / hypoxia / IFN) so the
    # report can explain *why* individual genes are high or low
    # (e.g. KLK3 ↓ + FOLH1 ↑ → AR-suppressed, consistent with ADT).
    try:
        from .common import build_sample_tpm_by_symbol
        sample_tpm_by_symbol = build_sample_tpm_by_symbol(df_expr)
        therapy_scores = score_therapy_signatures(sample_tpm_by_symbol, cancer_code)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"[therapy-response] scoring skipped: {exc}")
        therapy_scores = {}
    analysis["therapy_response_scores"] = therapy_scores
    for cls, score in therapy_scores.items():
        if score.state in ("up", "down"):
            tag = "[therapy-state]"
            print(f"{tag} {cls}: {score.state} — {score.message}")

    summary_png = "%s-sample-summary.png" % prefix if prefix else "sample-summary.png"
    plot_sample_summary(
        df_expr,
        cancer_type=cancer_code,
        sample_mode=analysis["sample_mode"],
        save_to_filename=summary_png,
        save_dpi=output_dpi,
        analysis=analysis,  # #84: skip redundant analyze_sample call
    )

    print("[analysis] Running broad-compartment decomposition...")
    decomp_png = None
    composition_png = None
    components_png = None
    candidates_png = None
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
                "support_geomean": row.get("support_geomean"),
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
                "embedding_methods": ["tme"],
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
        sample_context=sample_context,
        # #85: hand off the already-ranked candidate rows so
        # decompose_sample reuses analyze_sample's work instead of
        # re-ranking (which internally re-runs estimate_tumor_purity
        # per candidate).
        candidate_rows=analysis.get("candidate_trace"),
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

        # Propagate a lineage-panel purity override back into
        # ``analysis["purity"]`` so every downstream report is
        # consistent (bug 2026-04-14: user saw 23% in the headline
        # and 64% in the decomposition-hypotheses table). When the
        # decomposition's best hypothesis used a non-signature purity
        # source we promote it, preserve the original estimate as
        # ``signature_based_estimate``, and reset the CI widening
        # and the downstream pct_cancer_median math to use the new
        # anchor.
        purity_source_best = (
            effective_purity.get("purity_source")
            if isinstance(effective_purity, dict) else None
        )
        if (
            purity_source_best in ("lineage_panel",)
            and isinstance(effective_purity, dict)
            and "overall_estimate" in effective_purity
        ):
            orig_purity = dict(analysis["purity"])
            analysis["purity"]["signature_based_estimate"] = orig_purity.get(
                "overall_estimate"
            )
            analysis["purity"]["signature_based_lower"] = orig_purity.get(
                "overall_lower"
            )
            analysis["purity"]["signature_based_upper"] = orig_purity.get(
                "overall_upper"
            )
            analysis["purity"]["overall_estimate"] = effective_purity["overall_estimate"]
            analysis["purity"]["overall_lower"] = effective_purity.get(
                "overall_lower", effective_purity["overall_estimate"]
            )
            analysis["purity"]["overall_upper"] = effective_purity.get(
                "overall_upper", effective_purity["overall_estimate"]
            )
            analysis["purity"]["purity_source"] = purity_source_best
            analysis["purity"]["lineage_tumor_fraction"] = effective_purity.get(
                "lineage_tumor_fraction"
            )
            # Refresh locals that were captured above the override.
            purity = analysis["purity"]
            print(
                f"[analysis] Adopted lineage-panel purity "
                f"{analysis['purity']['overall_estimate']:.0%} "
                f"(signature-based estimate was "
                f"{orig_purity.get('overall_estimate', 0):.0%})"
            )
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

        # Prefer standalone decomposition plots over the crowded legacy
        # 4-panel composite so each figure reads cleanly on its own.
        # Keep ``plot_decomposition_summary`` available in the API, but
        # do not emit it by default from ``analyze``.
        # Standalone presentation-ready plots for the best hypothesis
        composition_png = "%s-decomposition-composition.png" % prefix if prefix else "decomposition-composition.png"
        plot_decomposition_composition(
            best_decomp,
            save_to_filename=composition_png,
            save_dpi=output_dpi,
        )
        components_png = "%s-decomposition-components.png" % prefix if prefix else "decomposition-components.png"
        plot_decomposition_component_breakdown(
            best_decomp,
            save_to_filename=components_png,
            save_dpi=output_dpi,
        )
        # Sample decomposition candidate bars — one row per (cancer × template)
        # candidate, showing the 3-segment composition (tumor / template-
        # specific / shared immune+stroma). Replaces the "extra=..." text
        # annotation on the composite summary with a structural picture.
        candidates_png = (
            "%s-decomposition-candidates.png" % prefix if prefix else "decomposition-candidates.png"
        )
        plot_decomposition_candidates(
            decomp_results,
            save_to_filename=candidates_png,
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
        # #86: render the harmonized purity (may be lineage-panel-
        # adopted) so the figure agrees with the rest of the report
        # instead of silently recomputing a signature-only estimate.
        purity_result=analysis["purity"],
    )
    _plt.close("all")

    # Scatter plots: sample vs pan-cancer reference
    print("[plot] Generating sample vs cancer scatter plots...")
    scatter_pdf = (
        "%s-vs-cancer.pdf" % prefix if prefix else "vs-cancer.pdf"
    )
    plot_sample_vs_cancer(
        df_expr,
        # #83: use the resolved (possibly auto-detected and decomp-
        # promoted) cancer type so the scatter agrees with the
        # summary/purity outputs. Previously fell back to the raw CLI
        # argument, which for the default auto-detect path meant the
        # pan-cancer mean instead of the inferred tumor type.
        cancer_type=effective_cancer_type,
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

    # Sample-among-TCGA embedding: MDS in the TME-low gene space is the
    # preferred view — robust at low purity (where hierarchy-method plots
    # can cluster the sample by infiltrate instead of by tumor biology),
    # and MDS preserves pairwise distances better than PCA for samples
    # that sit between canonical cancer-type centroids.  Other
    # method/algorithm combinations are available in the Python API but
    # are not emitted by default; see pirl-unc/pirlygenes#... for the
    # design rationale.
    print("[plot] Generating MDS embedding (TME-low genes)...")
    mds_png = "%s-mds-tme.png" % prefix
    plot_cancer_type_mds(df_expr, method="tme", save_to_filename=mds_png, save_dpi=output_dpi)
    embedding_pngs = [mds_png]
    _plt.close("all")

    # Generate text reports
    print("[report] Generating text reports...")
    _embedding_meta = get_embedding_feature_metadata(method="hierarchy")
    _generate_text_reports(
        analysis, _embedding_meta, prefix,
        decomp_results=decomp_results,
        input_path=input_path,
    )


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
            met_site=analysis.get("analysis_constraints", {}).get("met_site"),
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

        # Per-gene matched-normal attribution (issue #55). One PNG per
        # category, only emitted when matched-normal subtraction was
        # active. Separate plots rather than a composite figure per the
        # project's crowding preference.
        if (
            "matched_normal_tpm" in ranges_df.columns
            and (ranges_df["matched_normal_tpm"].astype(float) > 0).any()
        ):
            for cat_key, cat_slug in _adj_categories:
                mn_png = (
                    "%s-matched-normal-%s.png" % (prefix, cat_slug)
                    if prefix else "matched-normal-%s.png" % cat_slug
                )
                fig = plot_matched_normal_attribution(
                    ranges_df,
                    cancer_type=effective_cancer_type,
                    category=cat_key,
                    top_n=15,
                    save_to_filename=mn_png,
                    save_dpi=output_dpi,
                )
                if fig is not None:
                    adj_pngs.append(mn_png)
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
    # Report-flow order (user direction 2026-04-14): QC first
    # (context_png + degradation_png), then the headline cancer call
    # (summary_png), then deeper detail (decomposition / purity /
    # strip plots / embeddings). Plots missing from this list didn't
    # make it into all-figures.pdf and got left out of the moved-to-
    # figures/ step.
    png_files = [
        context_png,
        degradation_png,
        summary_png,
        decomp_png,
        # Standalone decomposition PNGs — composition / component breakdown
        # / candidate bars. Historically missed the move-to-figures/ step
        # because they weren't listed here.
        composition_png,
        components_png,
        candidates_png,
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
| `*-targets.md` | Therapeutic targets — tumor context, therapy landscape at a glance, CTAs, surface proteins, intracellular targets, tumor-expression ranges |
| `*-analysis-parameters.json` | Free model parameters plus selected sample mode and embedding methods |
| `*-all-figures.pdf` | All figures combined into a single PDF |
| `*-cancer-candidates.tsv` | Candidate cancer-type support trace |
| `*-decomposition-hypotheses.tsv` | Ranked decomposition hypotheses |
| `*-decomposition-components.tsv` | Component-level fit for best decomposition |
| `*-decomposition-markers.tsv` | Marker-gene evidence for best decomposition |
| `*-decomposition-gene-attribution.tsv` | Per-gene TME/tumor attribution for best decomposition |
| `*-tumor-expression-ranges.tsv` | Purity-adjusted tumor-expression ranges with TCGA context |

## Figures (in `figures/`)

Prefer the standalone decomposition figures for review and sharing. They replace the crowded legacy composite by splitting composition, component breakdown, and candidate comparison into separate PNGs.

| Figure | Description |
|--------|-------------|
| `*-sample-summary.png` | Quick overview: cancer type, purity, background signatures |
| `*-decomposition-composition.png` | Standalone composition bar (tumor + TME) for the best hypothesis |
| `*-decomposition-components.png` | Standalone TME cell-type breakdown for the best hypothesis |
| `*-decomposition-candidates.png` | Standalone per-candidate composition bars (tumor / template-specific / shared host) across top decomposition candidates |
| `*-purity.png` | Tumor purity estimation detail |
| `*-immune.png` | Immune microenvironment gene expression |
| `*-tumor.png` | Tumor biology gene expression |
| `*-antigens.png` | Tumor antigen expression (CTAs, surfaceome) |
| `*-treatments.png` | Therapy target expression by modality |
| `*-target-safety.png` | Therapy target normal tissue expression |
| `*-purity-targets.png` | Tumor-expression ranges for therapeutic targets |
| `*-purity-ctas.png` | Tumor-expression ranges for CTAs |
| `*-purity-surface.png` | Tumor-expression ranges for surface proteins |
| `*-mds-tme.png` | MDS: sample among TCGA cancer types (TME-low gene space) |
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


def _purity_ci_phrase(purity):
    """Render the purity estimate and CI with an explicit low-confidence
    tag when the interval is so wide it provides almost no constraint
    (issue #79). A 19%-100% CI should NOT look the same to a reader as
    a 58%-70% CI.
    """
    est = purity["overall_estimate"]
    lo = purity["overall_lower"]
    hi = purity["overall_upper"]
    tier = _ci_confidence_tier(lo, hi)
    core = f"**{est:.0%}** (range {lo:.0%}–{hi:.0%})"
    if tier == "low":
        core += (
            " — **⚠ low-confidence**: the CI spans "
            f"{(hi - lo):.0%}, so per-gene tumor-expression estimates "
            "derived from this purity carry wide error bars"
        )
    elif tier == "moderate":
        core += " (moderate-width CI)"
    return core


def _summary_mode_clause(sample_mode, purity, top_tissues):
    tissue_str = ", ".join(f"{t} ({s:.2f})" for t, s, _ in top_tissues[:3])
    ci_phrase = _purity_ci_phrase(purity)
    if sample_mode == "pure":
        return (
            f"The sample was analyzed in **pure-population mode**. The reported "
            f"purity-like estimate ({ci_phrase}) is best read as "
            "a coherence check against the matched lineage profile rather than as a bulk admixture fraction. "
            f"Residual background signatures are limited ({tissue_str}). "
        )
    if sample_mode == "heme":
        return (
            f"The estimated **malignant-lineage fraction proxy** is {ci_phrase}. "
            "In heme mode this is not a strict tumor-vs-immune split; it reflects how strongly the sample "
            "resembles the matched malignant program relative to hematopoietic background. "
            f"Top lineage/background contexts: {tissue_str}. "
        )
    return (
        f"Estimated tumor purity is {ci_phrase}, "
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


def _mhc1_status_text(mhc1):
    mhc1 = mhc1 or {}
    b2m = float(mhc1.get("B2M", 0) or 0)
    hla_mean = sum(float(mhc1.get(g, 0) or 0) for g in ("HLA-A", "HLA-B", "HLA-C")) / 3
    if hla_mean > 50 and b2m > 100:
        return (
            "adequate",
            f"adequate (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) — "
            "intracellular targets are plausibly presentable.",
        )
    if hla_mean > 10:
        return (
            "reduced",
            f"reduced (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) — "
            "intracellular targeting may have limited efficacy.",
        )
    return (
        "low/absent",
        f"low/absent (HLA mean={hla_mean:.0f}, B2M={b2m:.0f} TPM) — "
        "surface-directed strategies are safer than TCR-style approaches.",
    )


def _lineage_caveat_text(sample_mode, cancer_code):
    if sample_mode == "pure":
        return (
            "these are lineage-identity genes. In pure-population mode they help "
            "confirm that the population still resembles the expected lineage, but "
            "they do not prove every expressing cell is malignant."
        )
    if sample_mode == "heme":
        return (
            "these are lineage-identity genes. They help separate the malignant "
            "program from unrelated hematopoietic background, but reactive cells of "
            "the same lineage can express them too, so they do not by themselves "
            "distinguish malignant from benign same-lineage cells."
        )
    if epithelial_matched_normal_component(cancer_code) is not None:
        return (
            "these are lineage-identity genes. They help confirm tissue-of-origin "
            "against unrelated TME background, but epithelial primaries can share "
            "them with admixed benign parent tissue. They do NOT by themselves "
            "distinguish tumor cells from benign cells of the same lineage."
        )
    return (
        "these are lineage-identity genes. They help confirm tissue-of-origin "
        "against unrelated host background, but benign cells of the same lineage "
        "can also express them. They do NOT by themselves distinguish malignant "
        "from benign cells when both share the same program."
    )


def _matched_normal_split_summary(ranges_df):
    if "matched_normal_tissue" not in ranges_df.columns:
        return None
    import pandas as pd

    mn_values = ranges_df["matched_normal_tissue"].dropna().astype(str)
    mn_nonempty = [v for v in mn_values.unique() if v]
    if not mn_nonempty:
        return None
    mn_tissue = mn_nonempty[0]
    mn_frac_series = (
        ranges_df.get("matched_normal_fraction", pd.Series(dtype=float))
        .dropna()
    )
    mn_frac = float(mn_frac_series.iloc[0]) if len(mn_frac_series) else 0.0
    return (
        f"`matched_normal_{mn_tissue}` at **{mn_frac:.2%}** of the sample. "
        "Per-gene estimates subtract both stromal/immune TME and benign parent-tissue "
        "signal before dividing by purity."
    )


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
    met_site=None,
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
    if met_site:
        constraints["met_site"] = met_site
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


def _next_best_support_gap(candidate_trace):
    """Return (next_best_code, support_ratio) — ratio of top support_norm
    over the second candidate's support_norm, or None when unavailable.
    """
    if not candidate_trace or len(candidate_trace) < 2:
        return None, None
    top = candidate_trace[0]
    runner = candidate_trace[1]
    top_n = float(top.get("support_norm", 0.0) or 0.0)
    runner_n = float(runner.get("support_norm", 0.0) or 0.0)
    if runner_n <= 0:
        return runner.get("code"), None
    return runner.get("code"), top_n / runner_n


def _generate_text_reports(
    analysis, embedding_meta, prefix, decomp_results=None, input_path=None,
):
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

    # Cancer-type call line. Qualitative, not raw composite-score (#32):
    # show "top match, X× over next-best <CODE>" built from support_norm,
    # and whether the call was auto-detected vs user-specified (#33).
    source_label = {
        "auto-detected": "auto-detected",
        "user-specified": "user-specified",
    }.get(analysis.get("cancer_type_source"), "")
    source_suffix = f" ({source_label})" if source_label else ""
    next_best_code, support_ratio = _next_best_support_gap(candidate_trace)
    if family_display:
        intro = f"The sample most closely matches **{family_display}**"
        if subtype_clause:
            if constrained_cancer and lead_candidate and constrained_cancer != lead_candidate:
                intro += f", with **{cancer_code}** as the constrained working subtype{source_suffix}"
            else:
                intro += f", with **{cancer_code}** as the current best subtype hypothesis{source_suffix}"
        intro += ". "
    else:
        intro = (
            f"The sample most closely matches **{cancer_name} ({cancer_code})**"
            f"{source_suffix}. "
        )
    if next_best_code and support_ratio is not None and support_ratio > 1.0:
        intro += (
            f"Support is **{support_ratio:.1f}× over next-best {next_best_code}**. "
        )
    if fit_quality.get("label") and fit_quality.get("message"):
        intro += f"Fit quality: *{fit_quality['label']}* — {fit_quality['message']} "

    # Report flow (user direction 2026-04-14):
    #   1. What data do we have + sample QC (this block)
    #   2. What kind of cancer (the intro above, written before this)
    #   3. What else is in the sample (purity + TME + tissues)
    #   4. Deeper detail (analysis.md, targets.md, figures)
    #
    # Build ordering below assembles the summary string in that flow —
    # *not* in the order variables were defined — so the narrative
    # starts with QC and ends with ambiguity / constraints.

    # Disease-state synthesis (#78): one-line narrative up top.
    disease_state_paragraph = compose_disease_state_narrative(analysis)

    # Stage 1: input & sample-QC framing.
    sample_context = analysis.get("sample_context")
    context_paragraph = ""
    if sample_context is not None:
        ctx_signals = sample_context.signals or {}
        context_line = f"**Sample context**: {sample_context.summary_line()}."
        n_det_1 = ctx_signals.get("genes_detected_above_1_tpm")
        if n_det_1 is not None:
            context_line += f" {n_det_1} genes at TPM > 1"
            top50 = ctx_signals.get("top_50_share_of_total_tpm")
            if top50 is not None:
                context_line += f"; top-50 share {top50:.0%} of total"
            context_line += "."
        if ctx_signals.get("likely_targeted_panel"):
            context_line += " ⚠ Input looks like a targeted panel rather than whole-transcriptome."
        if sample_context.missing_mt and getattr(sample_context, "library_prep", None) not in _MT_EXPECTED_MISSING_PREPS:
            # #77: suppress the MT caveat when the inferred library
            # prep already explains the absence — avoids double-counting.
            context_line += " ⚠ MT genes missing from quant — degradation signal unreliable."
        context_paragraph = context_line

    # Quality flags land right after sample_context in the QC block.
    quality = analysis.get("quality")
    quality_paragraph = ""
    if quality and quality.get("has_issues"):
        # Prefer the SampleContext-filtered flag list (#77) so the MT
        # warning doesn't double-count an exome-capture library prep
        # we already explained.
        flags_to_show = quality.get("filtered_flags", quality["flags"])
        # Only emit a "Quality warnings" paragraph when real issues
        # remain after filtering — otherwise skip entirely.
        if flags_to_show and any(not f.startswith("MT fraction near zero") for f in flags_to_show):
            quality_paragraph = "**Quality warnings**: " + "; ".join(flags_to_show) + "."
    elif quality and quality["degradation"]["level"] != "normal":
        flags_to_show = quality.get("filtered_flags", quality["flags"])
        if flags_to_show:
            quality_paragraph = "**Quality note**: " + "; ".join(flags_to_show) + "."

    # Stage 2: headline call (already built above as `intro`).
    headline = intro

    # Stage 3: what else is in the sample — purity clause + background
    # tissues (from _summary_mode_clause), MHC-I level, analysis mode.
    composition = (
        _summary_mode_clause(sample_mode, purity, top_tissues)
        + f"MHC-I expression is {mhc_level} "
        + f"(HLA-A={hla_a:.0f}, HLA-B={hla_b:.0f}, B2M={b2m:.0f} TPM). "
        + f"Analysis mode: **{_sample_mode_display(sample_mode)}**."
    )
    if purity.get("components", {}).get("integration", {}).get("signature_deprioritized"):
        composition += (
            " The tumor-specific signature panel was materially weaker and less stable than "
            "the lineage panel, so it was downweighted rather than used as a hard lower anchor."
        )

    # Therapy-response state (#57): surface any axis that is clearly
    # up or down vs cohort so the reader sees *why* individual genes
    # might be off-baseline — ADT suppression of AR-transactivated
    # genes, endocrine resistance, etc.
    therapy_scores = analysis.get("therapy_response_scores") or {}
    therapy_paragraph = ""
    active_states = [
        (cls, s) for cls, s in therapy_scores.items()
        if s.state in ("up", "down")
    ]
    if active_states:
        lines_ts = ["**Therapy-response state**:"]
        for cls, s in active_states:
            verb = "active" if s.state == "up" else "suppressed"
            fold_phrase = ""
            if s.up_geomean_fold is not None:
                fold_phrase = f" (up-panel {s.up_geomean_fold:.2f}× cohort"
                if s.down_geomean_fold is not None:
                    fold_phrase += f", down-panel {s.down_geomean_fold:.2f}×"
                fold_phrase += ")"
            lines_ts.append(
                f"  - {cls.replace('_', ' ')}: {verb}{fold_phrase}"
            )
        therapy_paragraph = "\n".join(lines_ts)

    # Stage 4: constraints / decomposition detail / ambiguity.
    detail_parts = []
    if therapy_paragraph:
        detail_parts.append(therapy_paragraph)
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
        if constraints.get("met_site"):
            constraint_parts.append(
                f"biopsy site **{constraints['met_site']}** (TME reference augmented)"
            )
        if constraint_parts:
            detail_parts.append("Analysis constraints: " + "; ".join(constraint_parts) + ".")
    # Only surface the decomposition line when its call materially differs
    # from the headline call (#33: the tumor fraction % is identical to the
    # already-stated purity, so repeating it is noise).
    if call_summary.get("site_indeterminate"):
        detail_parts.append(
            "Decomposition recovered broad admixture structure, but "
            "site/template assignment is indeterminate."
        )
    elif best_decomp is not None:
        decomp_piece = (
            f"Decomposition template: **{best_decomp.cancer_type} / {best_decomp.template}**"
        )
        if (
            getattr(best_decomp, "cancer_type", None)
            and best_decomp.cancer_type != cancer_code
        ):
            decomp_piece += " (differs from headline call)"
        detail_parts.append(decomp_piece + ".")
    if ambiguity_clause.strip():
        detail_parts.append(ambiguity_clause.strip())
    # Tissue-score caveat (#33): readers were interpreting similarity
    # scores as composition percentages. One short line clarifies.
    if top_tissues:
        detail_parts.append(
            "*Note: background tissue scores are normalised similarity "
            "to reference nTPM profiles, not composition percentages.*"
        )

    # Assemble in the user-requested order: disease-state synthesis
    # (#78) → QC → coarse call → detail. The synthesis line lands
    # *before* the QC block so a reader immediately sees the clinical
    # framing ("ADT-treated CRPC with emerging NEPC") before
    # descending into sequencing QC or the raw cancer-type number.
    sections = []
    if disease_state_paragraph:
        sections.append(f"**Disease state**: {disease_state_paragraph}")
    qc_block = "\n\n".join([b for b in (context_paragraph, quality_paragraph) if b])
    if qc_block:
        sections.append(qc_block)
    sections.append(headline + composition)
    if detail_parts:
        sections.append("\n\n".join(detail_parts))
    summary = "\n\n".join(sections)

    summary_path = "%s-summary.md" % prefix if prefix else "summary.md"
    header = "# Sample Analysis Summary\n"
    if input_path:
        header += f"\n*Input*: `{input_path}`\n"
    with open(summary_path, "w") as f:
        f.write(f"{header}\n{summary}\n")
    print(f"[report] Saved {summary_path}")

    # --- Detailed report ---
    lines = ["# Detailed Sample Analysis\n"]

    # Disease-state synthesis (#78) — matches the summary.md top line
    # so readers arriving from either report see the same framing.
    if disease_state_paragraph:
        lines.append(f"**Disease state**: {disease_state_paragraph}\n")

    # Input characterization (#68) — surfaced before quality/decomp so
    # readers see whether the file we analysed was transcript-level vs
    # gene-level, whole-transcriptome vs panel, poly-A vs total RNA,
    # before they scrutinise downstream numbers.
    sample_context = analysis.get("sample_context")
    if sample_context is not None:
        ctx_signals = sample_context.signals or {}
        lines.append("## Input characterization\n")
        if input_path:
            lines.append(f"- **Source file**: `{input_path}`")
        lines.append(f"- **Library prep**: {sample_context.library_prep.replace('_', ' ')} "
                     f"(confidence {sample_context.library_prep_confidence:.0%})")
        lines.append(f"- **Preservation**: {sample_context.preservation.replace('_', ' ')}"
                     + (f" (degradation index {sample_context.degradation_index:.2f})"
                        if sample_context.degradation_index is not None else ""))
        n_det = ctx_signals.get("genes_detected_above_1_tpm")
        if n_det is not None:
            lines.append(f"- **Detection breadth**: {n_det} genes with TPM > 1 "
                         f"({ctx_signals.get('genes_detected_above_10_tpm', 0)} with TPM > 10, "
                         f"{ctx_signals.get('genes_detected_above_0p5_tpm', 0)} with TPM > 0.5)")
        top50 = ctx_signals.get("top_50_share_of_total_tpm")
        top2000 = ctx_signals.get("top_2000_share_of_total_tpm")
        if top50 is not None:
            concentration = f"- **Concentration**: top 50 genes carry {top50:.0%} of total TPM"
            if top2000 is not None:
                concentration += f"; top 2000 carry {top2000:.0%}"
            lines.append(concentration)
        if ctx_signals.get("likely_targeted_panel"):
            lines.append(
                "- ⚠ **Likely targeted panel** (few detected genes or >90% TPM "
                "concentrated in top 2000 genes) — downstream scores assume "
                "whole-transcriptome input; interpret carefully."
            )
        log2_med = ctx_signals.get("log2_tpm_median")
        if log2_med is not None:
            lines.append(f"- **Expression range**: log2(TPM+1) median={log2_med:.2f}, "
                         f"IQR={ctx_signals.get('log2_tpm_iqr', 0):.2f}, "
                         f"p95={ctx_signals.get('log2_tpm_p95', 0):.2f}")
        if sample_context.missing_mt:
            lines.append(
                "- ⚠ **Mitochondrial genes missing** from quant table — "
                "degradation signal from MT fraction is unreliable."
            )
        lines.append("")

    # Sample quality — driven by the stage-1 SampleContext so the
    # three report sections (summary / analysis / targets) agree
    # (#77). The raw sample_quality output is still used for its
    # tissue-matched baselines, but the top-level "is this sample
    # degraded?" answer comes from sample_context.preservation.
    quality = analysis.get("quality")
    if quality:
        lines.append("## Sample Quality\n")
        deg = quality["degradation"]
        cul = quality["culture"]

        if sample_context is not None:
            # Preservation call is the sample_context call; the
            # sample_quality raw-signal read is shown as supporting
            # detail under it, not as a competing verdict.
            preservation_label = sample_context.preservation.replace("_", " ")
            prep = getattr(sample_context, "library_prep", "unknown")
            prep_label = prep.replace("_", " ")
            lines.append(
                f"**Preservation**: {preservation_label} "
                f"(library prep inferred as *{prep_label}*, confidence "
                f"{sample_context.library_prep_confidence:.0%})"
            )
            if sample_context.degradation_severity and sample_context.degradation_severity != "none":
                lines.append(
                    f"- Severity: {sample_context.degradation_severity}"
                    + (
                        f" (length-pair index {sample_context.degradation_index:.2f})"
                        if sample_context.degradation_index is not None else ""
                    )
                )
        else:
            lines.append(f"**RNA degradation**: {deg['level']}")
        lines.append(f"- Mitochondrial fraction: {deg['mt_fraction']:.1%}")
        lines.append(f"- Ribosomal protein fraction: {deg['rp_fraction']:.1%}")
        if deg.get("long_short_ratio") is not None:
            lines.append(f"- Long/short transcript index: {deg['long_short_ratio']:.2f}")
        if deg.get("matched_tissue"):
            lines.append(f"- Matched tissue baseline: {deg['matched_tissue']} "
                         f"(MT={deg['baseline_mt']:.1%}, RP={deg['baseline_rp']:.1%})")
            if deg.get("mt_fold") is not None:
                lines.append(f"- Fold over baseline: MT {deg['mt_fold']:.1f}×, RP {deg['rp_fold']:.1f}×")
        # #77: only echo the raw sample_quality "MT filtered" message
        # when the inferred library prep doesn't already explain it.
        prep = getattr(sample_context, "library_prep", None) if sample_context else None
        mt_expected_missing = prep in _MT_EXPECTED_MISSING_PREPS
        if deg["level"] not in ("normal", "unknown"):
            lines.append(f"- *{deg['message']}*")
        elif deg["level"] == "unknown" and not mt_expected_missing:
            lines.append(f"- *{deg['message']}*")
        lines.append("")

        # Cell-culture / stress section is only meaningful when the
        # sample may plausibly be a cell line. Skip for solid-tumor
        # biopsies (#77) — the HSP90AA1 / GLS elevated pattern is
        # almost always hypoxia / tumor metabolism, already covered
        # by the therapy-response hypoxia axis.
        skip_culture = sample_mode == "solid"
        if not skip_culture:
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
    if candidate_trace:
        lines.append("- **Top candidates** (geomean · normalized):")
        for row in candidate_trace[:5]:
            name = CANCER_TYPE_NAMES.get(row["code"], row["code"])
            lines.append(
                f"  - **{row['code']}** ({name}): "
                f"{row.get('support_geomean', 0.0):.2f} · {row.get('support_norm', 0.0):.2f}"
            )
    else:
        top_cancers = analysis.get("top_cancers", [(cancer_code, analysis["cancer_score"])])
        for code, score in top_cancers[:5]:
            name = CANCER_TYPE_NAMES.get(code, code)
            lines.append(f"- **{code}** ({name}): {score:.3f}")
    lines.append("")

    if candidate_trace:
        lines.append("### Cancer Type Inference — Candidate Ranking\n")
        lines.append(
            "Each row is a TCGA cancer-type hypothesis considered by the classifier. "
            "Three scores summarize the match; the top row is the working call.\n\n"
            "- **Signature** (0–1): raw match quality between the sample's expression and "
            "this cancer type's TCGA-derived signature genes, computed from z-scored "
            "expression of cancer-type-enriched genes. Interpretable on its own — higher "
            "means the lineage pattern is strongly present. Does not account for purity "
            "or lineage concordance.\n"
            "- **Geomean** (0–1): the geometric mean of the five factors that feed the "
            "ranking (signature × purity × lineage support × signature stability × family "
            "factor). Stays bounded on [0, 1] so it's comparable across samples, unlike "
            "the raw product which collapses toward zero.\n"
            "- **Normalized** (0–1, top = 1.0): each candidate's composite support score "
            "divided by the top candidate's. Use this to judge separation — if the runner-up "
            "is ≪ 1.0, the top call is well-isolated; values near 1.0 mean the call is "
            "ambiguous between rows.\n\n"
            "Supporting columns: **Purity** is the overall tumor-purity estimate under "
            "this hypothesis; **Lineage** is an orthogonal per-lineage-gene purity; "
            "**Concordance** is how well the sample's lineage-gene pattern matches the "
            "expected pattern for that cancer type.\n"
        )
        lines.append("| Cancer | Family | Signature | Geomean | Normalized | Purity | Lineage | Concordance |")
        lines.append("|--------|--------|-----------|---------|------------|--------|---------|-------------|")
        for row in candidate_trace[:8]:
            lineage = row.get("lineage_purity")
            concordance = row.get("lineage_concordance")
            lines.append(
                f"| {row['code']} | {row.get('family_label') or '—'} | "
                f"{row.get('signature_score', 0.0):.3f} | "
                f"{row.get('support_geomean', 0.0):.3f} | "
                f"{row.get('support_norm', 0.0):.3f} | "
                f"{row.get('purity_estimate', 0.0):.3f} | "
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
    integration = components.get("integration", {})
    if integration.get("signature_deprioritized"):
        lines.append(
            "- **Integration note**: the tumor-specific signature panel was weaker and less stable "
            "than the lineage panel, so it was downweighted rather than used as a hard lower bound."
        )
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

        lines.append(
            "**Purity est.** per row = an independent purity estimate from that "
            "gene alone: sample expression ÷ the TCGA reference, after TCGA's own "
            "cohort impurity has been deconvolved from the reference. A value of "
            "20% means the sample expresses this gene at 20% of pure-tumor levels.\n"
        )
        lines.append(
            f"**Lineage caveat**: {_lineage_caveat_text(sample_mode, cancer_code_local)}\n"
        )
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
                    "TCGA reference, indicating retained lineage identity (does not distinguish tumor cells from normal cells of the same lineage)."
                )
            else:
                lines.append(
                    f"**Reliable cluster**: {retained_names}. "
                    "These genes are expressed at levels consistent with their "
                    "TCGA reference, indicating retained lineage identity (does not distinguish tumor cells from normal cells of the same lineage)."
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
            # #79: collapse components with < 0.5% fraction into a
            # single summary row so the useful entries are legible.
            trace_df = best_decomp.component_trace
            shown = trace_df[trace_df["fraction"] >= 0.005]
            hidden = trace_df[trace_df["fraction"] < 0.005]
            for _, row in shown.iterrows():
                comp = row["component"]
                marker_score = row["marker_score"]
                score_cell = (
                    f"{marker_score:.3f}" if isinstance(marker_score, (int, float))
                    and marker_score is not None else (marker_score if marker_score else "—")
                )
                top_markers_cell = row["top_markers"]
                # Matched-normal compartments have no decomposition
                # markers by design (see panels.py); annotate rather
                # than leave the empty cell unexplained (#79).
                if str(comp).startswith("matched_normal_") and (
                    not top_markers_cell or str(top_markers_cell).strip() == ""
                ):
                    top_markers_cell = "*matched-normal compartment — fraction derived from lineage-panel estimate (#52), not discriminative markers*"
                    if score_cell in ("—", "", "0.000", "0.0"):
                        score_cell = "n/a"
                lines.append(
                    f"| {comp} | {row['fraction']:.3f} | {score_cell} | {top_markers_cell} |"
                )
            if len(hidden):
                lines.append(
                    f"| *other components* | *{hidden['fraction'].sum():.3f} total across "
                    f"{len(hidden)} components < 0.5%* | — | — |"
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

    # Low-purity TME-inflation caveat (#35). Below 20% purity, every
    # residual TPM is amplified ≥5× by the tumor-value division.
    # Combined with incomplete TME subtraction, this can rank classic
    # stromal / ECM genes (FN1, COL1A1/2, DCN) as high-expressing tumor
    # markers. Users must read the caveat before interpreting the CAR-T
    # / ADC / radioligand target tables.
    if p_mid is not None and p_mid < 0.20:
        lines.append(
            f"> **⚠ Low-purity caveat**: estimated purity is "
            f"**{p_mid:.0%}**, so residual TPM is divided by a small "
            "number and amplified ≥5×. Genes heavily expressed in "
            "fibroblast / endothelial / immune compartments (FN1, "
            "COL1A1/2, DCN, etc.) can appear as high tumor-expressed "
            "even when most of the signal is stromal. Treat the "
            "`tme_explainable=true` column in the TSV as the primary "
            "filter for therapy-target safety; cross-check any "
            "`median_est` > 30 TPM against `tme_fold_med` before acting.\n"
        )

    # Sample-context caveat (stage 1 propagation): when the sample was
    # flagged as degraded / FFPE, every marker-selection step down the
    # pipeline is noisier and the target medians are correspondingly
    # less reliable.
    sample_context = analysis.get("sample_context")
    if sample_context is not None and sample_context.is_degraded:
        index_str = (
            f" (length-pair index {sample_context.degradation_index:.2f})"
            if sample_context.degradation_index is not None else ""
        )
        lines.append(
            f"> **⚠ Degradation caveat**: sample flagged as "
            f"`{sample_context.degradation_severity}` degradation"
            f"{index_str}. Long-transcript TPMs are under-represented; "
            "tumor-expression estimates for long-gene targets carry "
            "higher uncertainty than the reported CIs suggest.\n"
        )
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

    def _flag_series(df, column):
        if column in df.columns:
            return df[column].fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    def _format_target_stub(row, *, include_tcga=False):
        parts = [f"{row['median_est']:.0f} {value_label}"]
        therapies = str(row.get("therapies") or "").strip()
        if therapies:
            parts.append(therapies)
        if include_tcga and pd.notna(row.get("tcga_percentile")):
            parts.append(f"TCGA {row['tcga_percentile']:.0%}")
        if row.get("tme_explainable"):
            parts.append("single-tissue-explainable")
        return f"{row['symbol']} ({'; '.join(parts)})"

    therapy_scores = analysis.get("therapy_response_scores") or {}
    ts_to_show = [
        (cls, s) for cls, s in therapy_scores.items()
        if s.state in ("up", "down") and s.per_gene
    ]
    call_summary = analysis.get("call_summary") or _summarize_sample_call(
        analysis,
        [],
        sample_mode=sample_mode,
    )
    fit_quality = analysis.get("fit_quality", {})
    family_display = (analysis.get("family_summary") or {}).get("display")
    disease_state = compose_disease_state_narrative(analysis)
    matched_normal_summary = _matched_normal_split_summary(ranges_df)
    mhc_status_label, mhc_status_text = _mhc1_status_text(analysis.get("mhc1"))

    # Pre-sort the key target categories so the summary and the full
    # tables agree on what "top" means.
    ctas = (
        ranges_df[ranges_df["is_cta"] & (ranges_df["median_est"] > 0.5)]
        .sort_values("median_est", ascending=False)
        .copy()
    )
    _excluded = ranges_df.get(
        "excluded_from_ranking", pd.Series(False, index=ranges_df.index)
    )
    surface_targets = (
        ranges_df[
            ranges_df["is_surface"]
            & (ranges_df["median_est"] > 1)
            & ~ranges_df["is_cta"]
            & ~_excluded
        ]
        .sort_values("median_est", ascending=False)
        .copy()
    )
    intracellular = (
        ranges_df[
            ~ranges_df["is_surface"]
            & (ranges_df["median_est"] > 5)
            & (ranges_df["category"].isin(["therapy_target", "CTA"]))
            & ~_excluded
        ]
        .sort_values("median_est", ascending=False)
        .copy()
    )

    safe_surface = surface_targets[~_flag_series(surface_targets, "tme_dominant")].head(3)
    best_cta = ctas.head(3)
    clean_intracellular = intracellular[~_flag_series(intracellular, "tme_explainable")].head(3)

    lines.append("## Tumor context for interpretation\n")
    if call_summary.get("label_options"):
        if len(call_summary["label_options"]) == 2:
            lines.append(
                f"- **Working label**: provisional between "
                f"**{call_summary['label_options'][0]}** and **{call_summary['label_options'][1]}**."
            )
        else:
            lines.append(f"- **Working label**: **{call_summary['label_options'][0]}**.")
    else:
        lines.append(f"- **Working label**: **{cancer_code}** ({cancer_name}).")
    if family_display:
        lines.append(f"- **Family-level framing**: {family_display}.")
    if fit_quality.get("label"):
        fit_line = f"- **Fit quality**: {fit_quality['label']}"
        if fit_quality.get("message"):
            fit_line += f" — {fit_quality['message']}"
        lines.append(fit_line + ".")
    if call_summary.get("site_indeterminate"):
        lines.append("- **Context / site template**: indeterminate; treat site-specific decomposition as provisional.")
    elif call_summary.get("reported_site"):
        lines.append(f"- **Context / site template**: {call_summary['reported_site']}.")
    lines.append(f"- **Analysis mode**: {_sample_mode_display(sample_mode)}.")
    lines.append(
        f"- **{_purity_metric_label(sample_mode).title()}**: "
        f"{_purity_ci_phrase(purity_result)}."
    )
    if disease_state:
        lines.append(f"- **Disease-state synthesis**: {disease_state.rstrip('.')}.")
    lines.append(f"- **MHC-I status**: {mhc_status_text}")
    if matched_normal_summary:
        lines.append(f"- **Matched-normal split**: {matched_normal_summary}")
        lines.append(
            "- **Per-gene provenance**: see TSV `estimation_path` "
            "(`tme_only`, `matched_normal_split`, `clamped`, `tme_fold_fallback`)."
        )

    context_cautions = []
    integration = purity_result.get("components", {}).get("integration", {})
    if integration.get("signature_deprioritized"):
        context_cautions.append(
            "tumor-specific signature evidence was weaker than lineage/background evidence, so purity leaned more on the latter"
        )
    if p_mid is not None and p_mid < 0.20:
        context_cautions.append("low purity amplifies residual host/background signal")
    if sample_context is not None and sample_context.is_degraded:
        context_cautions.append("RNA degradation widens uncertainty for long transcripts")
    if therapy_scores.get("IFN_response") is not None and therapy_scores["IFN_response"].state == "up":
        context_cautions.append("active IFN response can inflate HLA/MHC-class and other interferon-stimulated targets")
    if context_cautions:
        lines.append(f"- **Interpretation caveats**: {'; '.join(context_cautions)}.")
    lines.append("")

    lines.append("## Therapy landscape at a glance\n")
    if len(safe_surface):
        lines.append(
            "- **Surface-directed modalities**: "
            + ", ".join(_format_target_stub(row) for _, row in safe_surface.iterrows())
            + "."
        )
    elif len(surface_targets):
        lines.append(
            "- **Surface-directed modalities**: no clean surface target passed the current reliability filter; "
            "top rows were TME-dominant or otherwise host-explainable."
        )
    else:
        lines.append("- **Surface-directed modalities**: no surface target rose above the reporting threshold.")
    if len(best_cta):
        lines.append(
            "- **CTA / vaccine ideas**: "
            + ", ".join(_format_target_stub(row, include_tcga=True) for _, row in best_cta.iterrows())
            + "."
        )
    else:
        lines.append("- **CTA / vaccine ideas**: no CTA rose above the reporting threshold.")
    if mhc_status_label == "adequate" and len(clean_intracellular):
        lines.append(
            "- **Intracellular / TCR-style ideas**: "
            + ", ".join(_format_target_stub(row) for _, row in clean_intracellular.iterrows())
            + "."
        )
    elif mhc_status_label == "adequate" and len(intracellular):
        lines.append(
            "- **Intracellular / TCR-style ideas**: MHC-I is adequate, but the current intracellular rows are mostly "
            "explainable by host-lineage/background signal."
        )
    elif len(intracellular):
        lines.append(
            "- **Intracellular / TCR-style ideas**: detectable candidates exist, but reduced antigen presentation makes "
            "surface-directed strategies safer to prioritize first."
        )
    else:
        lines.append("- **Intracellular / TCR-style ideas**: no intracellular target rose above the reporting threshold.")

    landscape_cautions = []
    if len(surface_targets) and _flag_series(surface_targets.head(10), "tme_dominant").any():
        landscape_cautions.append("some of the numerically highest surface rows are TME-dominant and should not be treated as tumor-cell targets")
    if matched_normal_summary:
        landscape_cautions.append("benign parent-tissue admixture is active")
    if (
        "therapy_supported" in ranges_df.columns
        and "therapy_support_note" in ranges_df.columns
    ):
        fn1_blocked = ranges_df[
            ranges_df["symbol"].eq("FN1")
            & ~ranges_df["therapy_supported"].fillna(True).astype(bool)
            & ranges_df["therapy_support_note"].fillna("").astype(str).str.len().gt(0)
        ]
        if len(fn1_blocked):
            landscape_cautions.append(
                fn1_blocked.sort_values("observed_tpm", ascending=False).iloc[0]["therapy_support_note"]
            )
    if landscape_cautions:
        lines.append(f"- **Landscape cautions**: {'; '.join(landscape_cautions)}.")
    lines.append("")

    # Therapy-state context (#57): enumerate the chain of evidence
    # behind any active / suppressed signaling axis so the reader can
    # see *why* specific genes are off-baseline rather than having to
    # reconstruct the pattern from per-gene tables.
    if ts_to_show:
        lines.append("## Therapy-state context\n")
        lines.append(
            "Cohort-referenced fold changes across curated signaling-axis "
            "panels (#57). Interpretation: a ≥2× up-panel elevation "
            "signals active signaling; a ≤0.5× up-panel drop with elevated "
            "down-panel genes signals therapy exposure (e.g. ADT in PRAD).\n"
        )
        for cls, s in ts_to_show:
            verb = "active" if s.state == "up" else "suppressed"
            lines.append(f"**{cls.replace('_', ' ')}** — {verb}. {s.message}\n")
            # Show the top 8 most-divergent per-gene rows so the chain
            # of evidence is concrete without overflowing the report.
            entries = sorted(
                s.per_gene,
                key=lambda e: abs((e["fold_vs_cohort"] or 1.0) - 1.0),
                reverse=True,
            )[:8]
            lines.append("| Gene | Direction | Sample TPM | Cohort median | Fold | Mechanism |")
            lines.append("|------|-----------|------------|---------------|------|-----------|")
            for e in entries:
                lines.append(
                    f"| {e['symbol']} | {e['direction']} | "
                    f"{e['sample_tpm']:.1f} | {e['cohort_median']:.1f} | "
                    f"{e['fold_vs_cohort']:.2f}× | {e['mechanism']} |"
                )
            lines.append("")

    # --- CTAs: vaccination targets ---
    lines.append("## Cancer-Testis Antigens (Vaccination Targets)\n")
    lines.append("CTAs are expressed in tumor but not normal adult tissue (except testis/placenta). "
                 "Any expressed CTA is a potential vaccination target regardless of trial status.\n")
    if len(ctas):
        lines.append(f"| Gene | {value_label} | Range | Observed | vs TCGA | TCGA %ile | TME | Surface | Therapies |")
        lines.append("|------|-----------|-------|----------|---------|-----------|-----|---------|-----------|")
        for _, row in ctas.head(20).iterrows():
            surf = "yes" if row["is_surface"] else ""
            vs_tcga = row["pct_cancer_median"]
            tme_warn = "⚠" if row.get("tme_explainable") else ""
            lines.append(
                f"| **{row['symbol']}** | {row['median_est']:.1f} | "
                f"{row['est_1']:.1f}\u2013{row['est_9']:.1f} | {row['observed_tpm']:.1f} | "
                f"{'%.1fx' % vs_tcga if pd.notna(vs_tcga) else '—'} | {row['tcga_percentile']:.0%} | "
                f"{tme_warn} | {surf} | {row['therapies']} |"
            )
        high_ctas = ctas[ctas["tcga_percentile"] > 0.7]
        if len(high_ctas):
            names = ", ".join(high_ctas["symbol"].head(5))
            lines.append(f"\n**Above TCGA median for {cancer_code}**: {names}")
        # Propagate healthy-tissue-explainability warnings across all target
        # tables, not just surface (issue #50 point 5). For a CTA this is
        # usually a false alarm (CTAs activated in a subset of tumors),
        # but the flag still conveys real ambiguity — e.g. testis-
        # retained genes in TGCT samples — so surface it consistently.
        if (
            "tme_explainable" in ctas.columns
            and ctas.head(20)["tme_explainable"].any()
        ):
            lines.append(
                "\n⚠ = sample signal could be entirely explained by a single healthy "
                "tissue's expression. For CTAs this is usually benign (cohort-median "
                "≈ 0 is normal for CTAs), but verify the flagged gene is not a germline "
                "/ germ-cell lineage marker in this patient's context."
            )
    else:
        lines.append("No CTAs detected above threshold.\n")
    lines.append("")

    # --- Surface therapy targets ---
    # #60: drop extended-housekeeping symbols (excluded_from_ranking)
    # from the ranked output so they can't appear as spurious high
    # tumor-expressed targets. They remain in the TSV with the flag.
    lines.append("## Surface Protein Targets (ADC / CAR-T / Bispecific)\n")
    lines.append("Surface proteins with high purity-adjusted expression. "
                 "These can be targeted by antibody-drug conjugates, CAR-T, "
                 "or bispecific T-cell engagers.\n")
    if len(surface_targets):
        lines.append(
            f"| Gene | {value_label} | Range | Observed | vs TCGA | TCGA %ile | TME | Therapies |"
        )
        lines.append(
            "|------|-----------|-------|----------|---------|-----------|-----|-----------|"
        )
        # #78: cross-signal annotations — flag IFN-driven surface /
        # MHC targets when the IFN_response axis is active, so the
        # reader knows a 287× HLA-F fold change isn't tumor-specific.
        cross_notes = annotate_surface_targets_with_cross_signals(
            ranges_df, analysis.get("therapy_response_scores") or {}
        )
        for _, row in surface_targets.head(20).iterrows():
            bold = "**" if row["therapies"] else ""
            vs_tcga = row["pct_cancer_median"]
            # TME flag — compact key:
            #   ⚠⚠ = ``tme_dominant`` (TME alone accounts for ≥70% of
            #   the observed signal — almost certainly a stromal /
            #   immune origin, #35); ⚠ = ``tme_explainable`` (a single
            #   healthy reference tissue could explain ≥50% of signal,
            #   #45).
            if row.get("tme_dominant"):
                tme_warn = "⚠⚠"
            elif row.get("tme_explainable"):
                tme_warn = "⚠"
            else:
                tme_warn = ""
            # Append IFN-driven cross-signal note to the therapies
            # column for core ISGs when IFN is active.
            therapies_cell = row["therapies"]
            cross = cross_notes.get(row["symbol"])
            if cross:
                therapies_cell = f"{therapies_cell} ({cross})" if therapies_cell else f"*{cross}*"
            lines.append(
                f"| {bold}{row['symbol']}{bold} | {row['median_est']:.1f} | "
                f"{row['est_1']:.1f}\u2013{row['est_9']:.1f} | {row['observed_tpm']:.1f} | "
                f"{'%.1fx' % vs_tcga if pd.notna(vs_tcga) else '—'} | {row['tcga_percentile']:.0%} | "
                f"{tme_warn} | {therapies_cell} |"
            )
        head20 = surface_targets.head(20)
        any_dominant = (
            "tme_dominant" in surface_targets.columns
            and head20["tme_dominant"].any()
        )
        any_explainable = (
            "tme_explainable" in surface_targets.columns
            and head20["tme_explainable"].any()
        )
        if any_dominant:
            lines.append(
                "\n⚠⚠ = **TME-dominant** (≥70% of observed signal is "
                "explained by the TME reference alone). These are very "
                "likely stromal / immune rather than tumor-cell expressed "
                "— excluding them from therapy-target consideration is "
                "the safest default (#35)."
            )
        if any_explainable:
            lines.append(
                "\n⚠ = sample signal could be entirely explained by a single healthy "
                "tissue's expression (max across non-reproductive tissues ≥ 50% of "
                "observed TPM). The tumor-cell attribution for these genes is "
                "unreliable — consider stromal / immune origin."
            )
    else:
        lines.append("No surface targets above threshold.\n")
    lines.append("")

    # --- Cytosolic / intracellular targets (TCR-T, pMHC) ---
    lines.append("## Intracellular Targets (TCR-T / pMHC Vaccination)\n")
    lines.append("Intracellular proteins presented via MHC-I. Targetable by "
                 "TCR-T cell therapy or peptide vaccination.\n")
    if len(intracellular):
        lines.append(f"| Gene | {value_label} | Range | vs TCGA | TCGA %ile | TME | CTA | Therapies |")
        lines.append("|------|-----------|-------|---------|-----------|-----|-----|-----------|")
        for _, row in intracellular.head(15).iterrows():
            cta_flag = "yes" if row["is_cta"] else ""
            vs_tcga = row["pct_cancer_median"]
            tme_warn = "⚠" if row.get("tme_explainable") else ""
            lines.append(
                f"| {row['symbol']} | {row['median_est']:.1f} | "
                f"{row['est_1']:.1f}\u2013{row['est_9']:.1f} | "
                f"{'%.1fx' % vs_tcga if pd.notna(vs_tcga) else '—'} | "
                f"{row['tcga_percentile']:.0%} | {tme_warn} | {cta_flag} | {row['therapies']} |"
            )
        if (
            "tme_explainable" in intracellular.columns
            and intracellular.head(15)["tme_explainable"].any()
        ):
            lines.append(
                "\n⚠ = sample signal could be entirely explained by a single healthy "
                "tissue's expression (max across non-reproductive tissues ≥ 50% of "
                "observed TPM). The tumor-cell attribution for these genes is "
                "unreliable — consider lineage-retained / stromal origin."
            )
    else:
        lines.append("No intracellular targets above threshold.\n")
    lines.append("")

    # --- Top recommendation summary ---
    # #79: Recommendations must respect the reliability flags from the
    # per-category tables. Previously the top-3 surface targets were
    # chosen by TPM rank alone, which promoted genes flagged ⚠⚠ (TME-
    # dominant / very likely stromal) as "best surface targets" — the
    # opposite of the safe default. Now: skip ⚠⚠-flagged rows
    # entirely from the summary; ⚠-flagged retained rows carry an
    # inline caveat.
    lines.append("## Recommended Targets Summary\n")

    def _reliability_badge(row):
        if row.get("tme_dominant"):
            return "⚠⚠"  # caller filters these out
        if row.get("tme_explainable"):
            return "⚠"
        return ""

    # Best surface — strict filter on ⚠⚠.
    safe_surface = surface_targets[~surface_targets.get(
        "tme_dominant",
        pd.Series(False, index=surface_targets.index),
    )].head(3)
    dropped_dominant = int(
        surface_targets.head(3).get("tme_dominant", pd.Series(False)).sum()
    )
    if len(safe_surface):
        lines.append("**Best surface targets** (ADC/CAR-T/bispecific):")
        for _, row in safe_surface.iterrows():
            badge = _reliability_badge(row)
            caveat = f", {badge} single-tissue-explainable" if badge == "⚠" else ""
            therapy_note = f" — active in {row['therapies']}" if row["therapies"] else ""
            lines.append(
                f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
                f"range {row['est_1']:.0f}\u2013{row['est_9']:.0f}"
                f"{caveat}){therapy_note}"
            )
        if dropped_dominant:
            lines.append(
                f"- *{dropped_dominant} top-ranked row"
                + ("s" if dropped_dominant != 1 else "")
                + " excluded from this summary for being TME-dominant"
                " (⚠⚠, see Surface Protein Targets table).*"
            )
        lines.append("")
    elif dropped_dominant:
        lines.append(
            "**Best surface targets** (ADC/CAR-T/bispecific): "
            "all top-ranked rows were flagged TME-dominant (⚠⚠) and "
            "are not safe to recommend. See Surface Protein Targets "
            "table for full context."
        )
        lines.append("")

    # Best CTAs — same flag respect (though CTA flags usually benign,
    # cohort-median ≈ 0 is normal for CTAs).
    best_cta = ctas.head(3)
    if len(best_cta):
        lines.append("**Best CTA targets** (vaccination even without active trials):")
        for _, row in best_cta.iterrows():
            badge = _reliability_badge(row)
            caveat = f" — {badge} check vs germline" if badge else ""
            lines.append(f"- **{row['symbol']}** ({row['median_est']:.0f} TPM, "
                         f"TCGA {row['tcga_percentile']:.0%}){caveat}")
        lines.append("")

    # MHC context for intracellular targeting
    lines.append(f"**MHC-I status**: {mhc_status_text}")
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
