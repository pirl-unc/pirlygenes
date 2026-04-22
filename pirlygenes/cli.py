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
from .tumor_purity import (
    analyze_sample,
    get_tumor_purity_parameters,
    plot_sample_summary,
    plot_tumor_purity,
    plot_purity_method_comparison,
    plot_cancer_type_hypotheses,
    plot_background_tissues,
    plot_mhc_expression,
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
    plot_cancer_type_mds,
    plot_therapy_target_tissues,
    plot_therapy_target_safety,
    plot_cohort_heatmap,
    plot_cohort_disjoint_counts,
    plot_cohort_pca,
    plot_cohort_therapy_targets,
    plot_cohort_surface_proteins,
    plot_cohort_ctas,
    get_embedding_feature_metadata,
    estimate_tumor_expression_ranges,
    plot_matched_normal_attribution,
    plot_target_attribution,
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
from .format import (
    render_fold,
    render_fraction_no_decimal,
    render_tpm,
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
    "disease-state-rules": "Per-cancer narrative rule catalog (#202)",
    "narrative-gene-sets": "Named gene sets for #202 disease-state rules",
    "degenerate-subtype-pairs": "Subtype disambiguation catalog (#198)",
    "fusion-surrogate-expression": "Fusion-surrogate gene catalog (#198)",
}


@named("data")
def print_dataset_info():
    """List all bundled datasets with row counts and sources."""
    import pandas as pd

    from .load_dataset import get_all_csv_paths

    total_size = 0
    print("\nBundled datasets (shipped with pip install, no downloads needed):\n")
    print(f"  {'Dataset':<40s} {'Rows':>6s}  {'Size':>8s}  Source")
    print(f"  {'─'*40}  {'─'*6}  {'─'*8}  {'─'*40}")
    for csv_path in get_all_csv_paths():
        df = pd.read_csv(str(csv_path))
        name = csv_path.name.removesuffix(".gz").removesuffix(".csv")
        size = csv_path.stat().st_size
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


@named("cancers")
def print_cancer_registry(family: str = None, tissue: str = None, show_all: bool = False):
    """List cancer types in the registry with data-availability markers.

    For each code the output shows which data sources are curated:
      bm=biomarkers   tg=targets   tcga=tcga_deconvolved   sub=subtype_deconvolved

    :param family: Restrict to one family (e.g. "sarcoma", "heme-myeloid", "net").
    :param tissue: Restrict to one primary tissue (e.g. "bone", "lymph_node").
    :param show_all: Show full registry including rows without expression data.
    """
    from .gene_sets_cancer import (
        cancer_type_registry,
        cancer_biomarker_genes,
        cancer_therapy_targets,
        tcga_deconvolved_expression,
        subtype_deconvolved_expression,
    )

    df = cancer_type_registry()
    if family:
        df = df[df["family"] == family]
    if tissue:
        df = df[df["primary_tissue"] == tissue]
    if df.empty:
        print("No registry entries match the given filters.")
        return

    tcga_deconv = tcga_deconvolved_expression()
    tcga_codes = set(tcga_deconv["cancer_code"].unique()) if tcga_deconv is not None else set()
    sub_deconv = subtype_deconvolved_expression()
    sub_codes = set() if sub_deconv is None else set(
        sub_deconv["cancer_code"].astype(str).unique()
    )

    def _safe_count(fn, *args, **kwargs):
        try:
            return len(fn(*args, **kwargs) or [])
        except Exception:
            return 0

    def _clean(value):
        """Normalise pandas NaN / blank / 'nan' to the empty string."""
        if value is None:
            return ""
        s = str(value).strip()
        if s.lower() == "nan":
            return ""
        return s

    def _resolve_lookup(row):
        """Hierarchical key-genes lookup.

        Prefers the explicit ``subtype_key`` column when non-empty
        (e.g. SARC_LMS -> SARC + 'leiomyosarcoma'). Falls back to the
        code-suffix heuristic only when the code abbreviation already
        matches the key-genes subtype literally (SARC_GIST -> 'gist');
        otherwise returns the code itself with no subtype so the
        lookup falls back to the parent's full-panel union."""
        code = _clean(row["code"])
        parent_code = _clean(row.get("parent_code"))
        subtype_key = _clean(row.get("subtype_key"))
        if subtype_key and parent_code:
            return parent_code, subtype_key
        if parent_code and code.startswith(parent_code + "_"):
            suffix = code[len(parent_code) + 1:].lower()
            # Only use the suffix as a subtype if the caller actually
            # has a row for it; the _clean() above already rejects NaN.
            if suffix:
                return parent_code, suffix
        return code, None

    # Render family by family; parent codes listed before children (subtypes).
    print(f"\nCancer-type registry — {len(df)} entries\n")
    print(f"  {'Code':<14s} {'Name':<42s} {'Primary tissue':<18s} {'Data':<22s} Source")
    print(f"  {'─'*14} {'─'*42} {'─'*18} {'─'*22} {'─'*30}")

    # Sort within family: parents first (no parent_code), then children.
    df = df.assign(_is_child=df["parent_code"].fillna("").astype(str).ne(""))
    df = df.sort_values(["family", "_is_child", "code"])

    current_family = None
    for _, row in df.iterrows():
        code = str(row["code"])
        if row["family"] != current_family:
            current_family = row["family"]
            print(f"\n  [{current_family}]")

        lookup_code, subtype = _resolve_lookup(row)
        if subtype is not None:
            bm_count = _safe_count(cancer_biomarker_genes, lookup_code, subtype=subtype)
            tg_df = cancer_therapy_targets(lookup_code, subtype=subtype)
        else:
            bm_count = _safe_count(cancer_biomarker_genes, lookup_code)
            tg_df = cancer_therapy_targets(lookup_code) if code else None
        tg_count = 0 if tg_df is None else len(tg_df)
        data_flags = []
        if bm_count:
            data_flags.append(f"bm={bm_count}")
        if tg_count:
            data_flags.append(f"tg={tg_count}")
        if code in tcga_codes:
            data_flags.append("tcga")
        if code in sub_codes:
            data_flags.append("sub")
        # Check if any subtype row below this code has sub-deconv
        if sub_deconv is not None:
            sub_children = sub_deconv[
                sub_deconv.get("subtype", "").astype(str).str.startswith(code + "_")
            ]
            if not sub_children.empty:
                data_flags.append("sub-child")
        data_cell = " ".join(data_flags) if data_flags else "—"

        if not show_all and data_cell == "—":
            continue

        indent = "    " if row["_is_child"] else "  "
        name_s = str(row["name"])[:42]
        tissue_s = str(row.get("primary_tissue") or "")[:18]
        source_s = str(row.get("source_cohort") or "")[:30]
        print(f"  {indent+code:<14s} {name_s:<42s} {tissue_s:<18s} {data_cell:<22s} {source_s}")

    print("\n  Legend: bm=biomarkers  tg=therapy-targets  tcga=deconvolved-TCGA-median  sub=subtype-stratified-median  sub-child=subtype-tiles-below  — = no curation/data yet\n")


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

# The AR-transactivation output panel used by the CRPC-pattern
# narrative rule moved to ``narrative-gene-sets.csv`` (#202). The
# disease-state rule engine looks it up by name (``AR_targets``) so
# the synthesis layer stays data-driven.

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


def _metabolic_axes_rows(evidence) -> list[tuple[str, str, str]]:
    """Return (axis, signal, therapy) rows for active metabolic programs (#158).

    Step-0 already computes proliferation / hypoxia / glycolysis
    channel scores on :class:`TumorEvidenceScore`. This helper converts
    those scores into therapy-report rows so readers see actionable
    metabolic axes (CA9-directed ADC, MCT inhibitors, CDK4/6) without
    having to cross-reference Step-0 narrative against the target
    landscape. Rows emit only when the corresponding channel is
    meaningfully elevated so the section is silent on non-metabolic
    tumors.
    """
    if evidence is None:
        return []
    rows: list[tuple[str, str, str]] = []

    hypoxia = float(getattr(evidence, "hypoxia", 0.0) or 0.0)
    ca9_tpm = float(getattr(evidence, "ca9_tpm", 0.0) or 0.0)
    if hypoxia >= 0.5 or ca9_tpm >= 20.0:
        rows.append((
            "Hypoxia / CA9",
            f"CA9 observed {ca9_tpm:.0f} TPM (hypoxia score {hypoxia:.2f})",
            "Acetazolamide; CA9-directed ADCs (DS-6157a / trials). Consider "
            "HIF-2α inhibitors (belzutifan) in ccRCC context.",
        ))

    glycolysis = float(getattr(evidence, "glycolysis", 0.0) or 0.0)
    glyc_fold = float(getattr(evidence, "glycolysis_geomean_fold", 0.0) or 0.0)
    if glycolysis >= 0.5:
        rows.append((
            "Glycolysis / MCT",
            f"Panel geomean {glyc_fold:.1f}× over median (score {glycolysis:.2f})",
            "MCT1/4 inhibitors (AZD3965 trials); LDHA / HK2 inhibitors "
            "(preclinical). Metformin where comorbidity supports it.",
        ))

    prolif = float(getattr(evidence, "proliferation", 0.0) or 0.0)
    prolif_log2 = float(getattr(evidence, "prolif_log2", 0.0) or 0.0)
    if prolif >= 0.6:
        rows.append((
            "Proliferation / cell-cycle",
            f"Panel log2-TPM {prolif_log2:.2f} (score {prolif:.2f})",
            "CDK4/6 inhibitors (palbociclib / ribociclib) where cancer-type "
            "context supports; WEE1 inhibitors (adavosertib trials).",
        ))

    return rows


def compose_disease_state_narrative(analysis) -> str:
    """Synthesize the disease-state narrative line (#78).

    As of #202 the per-cancer logic lives in ``disease-state-rules.csv``
    and ``narrative-gene-sets.csv``. This function only prepares the
    three inputs the rule engine consumes (axis states + lineage-
    retained / lineage-collapsed gene sets) and dispatches.

    Adding a new cancer-specific narrative is a CSV edit — no Python
    change. This keeps the analyze pipeline uniform across cancer
    types; differences live in data, not in ``if/elif cancer_code``
    branches.
    """
    from .disease_state_rules import synthesize_disease_state

    cancer_code = analysis.get("cancer_type")
    therapy_scores = analysis.get("therapy_response_scores") or {}
    purity = analysis.get("purity") or {}
    components = purity.get("components") or {}
    lineage = components.get("lineage") or {}
    per_gene = lineage.get("per_gene") or []

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

    axis_states: dict[str, str | None] = {}
    for axis_name, score in therapy_scores.items():
        axis_states[axis_name] = getattr(score, "state", None)

    return synthesize_disease_state(
        cancer_code=cancer_code,
        axis_states=axis_states,
        retained=retained,
        collapsed=collapsed,
    )


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
    """Drop / rewrite quality flags that the step-1 SampleContext
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


def _render_vs_tcga_cell(row):
    """Render the "vs TCGA" column for a target-table row.

    The naive ``render_fold(row["pct_cancer_median"])`` produces ``∞×``
    whenever the TCGA cohort's tumor-component deconvolves to ~0 for
    this gene — which happens in two biologically distinct cases:

    - **not_in_cohort**: the raw cohort median is itself ~0 (the gene
      genuinely isn't expressed at cohort median — the CTA-in-solid-
      cohort case). Here ``∞×`` is not wrong but is degenerate for a
      clinician; instead show the raw cohort TPM so the reader sees
      "gene is absent from the reference cohort".
    - **tme_explained**: the raw cohort was non-trivial but the TME
      deconvolution zeroed the tumor component. The ratio isn't
      meaningful because the deconvolution subtracted everything;
      label as "TME-only" with the raw cohort TPM for context.

    Keeps the column narrow; preserves information rather than capping.
    """
    import math as _math
    import pandas as _pd
    state = row.get("tcga_ref_state")
    vs_tcga = row.get("pct_cancer_median")
    cohort_tpm = row.get("tcga_cohort_median_tpm")

    if state == "finite" and _pd.notna(vs_tcga):
        return render_fold(vs_tcga)
    if state == "not_in_cohort":
        if cohort_tpm is not None and cohort_tpm > 0:
            return f"ref {cohort_tpm:.2f} TPM"
        return "ref 0"
    if state == "tme_explained":
        if cohort_tpm is not None:
            return f"TME-only ({cohort_tpm:.1f} TPM)"
        return "TME-only"
    if state == "both_absent":
        return "—"
    # Back-compat: when tcga_ref_state is missing (older ranges_df),
    # fall back to the raw fold. Inf still lands here and renders ∞×;
    # new runs always set the state.
    if vs_tcga is None or (isinstance(vs_tcga, float) and _math.isinf(vs_tcga)):
        if cohort_tpm is not None and cohort_tpm > 0:
            return f"ref {cohort_tpm:.2f} TPM"
        return "ref 0"
    return render_fold(vs_tcga) if _pd.notna(vs_tcga) else "—"


def _format_attribution_cell(row):
    """Compact per-target Attribution cell for targets.md (#108, #128).

    Renders "tumor T / comp C" when the decomposition produced a per-
    compartment attribution for this gene; returns "—" when attribution
    isn't available.

    For genes flagged ``broadly_expressed`` (#128), appends a brief
    "broadly expr." tag so the reader can't mistake a high
    tumor-attribution number for a specificity claim.
    """
    try:
        observed = float(row.get("observed_tpm") or 0.0)
    except (TypeError, ValueError):
        observed = 0.0
    if observed <= 0:
        return "—"
    attribution = row.get("attribution")
    try:
        attr_tumor = float(row.get("attr_tumor_tpm") or 0.0)
        top_comp = row.get("attr_top_compartment") or ""
        top_tpm = float(row.get("attr_top_compartment_tpm") or 0.0)
    except (TypeError, ValueError):
        return "—"
    broadly = bool(row.get("broadly_expressed"))
    amplified = bool(row.get("amplified_over_healthy"))
    over_predicted = bool(row.get("matched_normal_over_predicted"))
    sm_leakage = bool(row.get("smooth_muscle_stromal_leakage"))
    try:
        amp_fold = float(row.get("amplification_fold") or 0.0)
    except (TypeError, ValueError):
        amp_fold = 0.0

    def _tag():
        # Over-prediction is the strongest caveat — when matched-normal
        # alone predicts more of the gene than we observed, the
        # attribution math can't say anything reliable about the tumor
        # contribution (#131). Surface this first so a reader looking
        # at e.g. KLK3 / TACSTD2 on a CRPC sample knows the "tumor 0"
        # number isn't independent evidence that tumor cells aren't
        # expressing the gene — it means the reference over-predicted.
        if over_predicted:
            return "matched-normal over-predicted"
        # #59 item 1: smooth-muscle stromal leakage. Fibromuscular-
        # stroma density varies sample-to-sample; matched-normal
        # references carry the cohort-average and under-subtract
        # SM-lineage genes when the biopsy is SM-rich. The tumor
        # story for TAGLN / ACTA2 / MYH11 / CNN1 should be read with
        # that caveat.
        if sm_leakage:
            return "likely smooth-muscle stromal leakage"
        # Amplification is a positive specificity signal and takes
        # precedence over the broadly-expressed caution (#128). A gene
        # that's broadly expressed at baseline but observed >= 5× over
        # the peak healthy tissue is telling an amplification /
        # overexpression story — HER2 in HER2+ BRCA, MDM2 in WD/DD-LPS.
        if amplified and amp_fold >= 1.5:
            return f"amplified {amp_fold:.1f}\u00d7"
        if broadly:
            return "broadly expr."
        return ""

    tag = _tag()
    if not attribution or not isinstance(attribution, dict):
        if tag:
            return tag
        return "—"
    if not top_comp:
        base = f"tumor {attr_tumor:.0f}"
    else:
        comp_label = top_comp.replace("_", " ")
        base = f"tumor {attr_tumor:.0f} / {comp_label} {top_tpm:.0f}"
    if tag:
        base += f" · {tag}"
    return base


def _ci_confidence_tier(overall_lower, overall_upper):
    """Map a (lower, upper) span to a confidence tier (issue #79).

    Reader-facing tags on the purity estimate so a 19–100% CI is
    visibly different from a 58–70% CI in the report.

    A *zero-width* CI (``lower == upper == estimate``) is degenerate —
    the estimator saw no per-gene variation, which happens on synthetic
    / deterministic inputs (TCGA cohort medians, decomposition
    templates, expected-expression probes). Surfacing that as "high
    confidence" is misleading — the estimator couldn't produce
    uncertainty, not because the answer is certain but because the
    input has no spread to bound it with (#161).
    """
    try:
        span = float(overall_upper) - float(overall_lower)
    except (TypeError, ValueError):
        return "unknown"
    if span <= 1e-9:
        return "degenerate"
    if span < 0.15:
        return "high"
    if span < 0.35:
        return "moderate"
    return "low"


def _default_output_dir() -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"pirlygenes-{ts}"


# ── Output-directory lock (#82) ─────────────────────────────────────────
#
# Two concurrent ``analyze`` invocations pointed at the same
# ``--output-dir`` would happily overwrite each other's artifacts
# because ``_clean_prefix_outputs`` wipes stale files at start and both
# runs race to write to the same filenames. The result is a silently
# inconsistent report (figures from one run, markdowns from the other).
#
# Cheap fix: write an advisory ``.pirlygenes.lock`` with the runner's
# pid + start time. A second invocation into the same directory while
# the lock-owner is still alive bails out with a clear error directing
# the user to ``--force`` or a different ``--output-dir``.
#
# Not a true concurrency guarantee — a race still exists between the
# stale-lock unlink and the new write — but the common accidental
# double-launch case is caught and diagnosed instead of silently
# corrupting the output.
_LOCKFILE_NAME = ".pirlygenes.lock"


def _pid_is_alive(pid: int) -> bool:
    """Cheap liveness check.

    ``os.kill(pid, 0)`` raises :class:`ProcessLookupError` when the
    process does not exist and :class:`PermissionError` when it
    exists but the caller can't signal it (which still counts as
    alive for our purposes — we don't want a lock held by another
    user to look stale just because we can't ping it).
    """
    if pid <= 0:
        return False
    import os as _os

    try:
        _os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_output_dir_lock(out_dir, force: bool = False):
    """Claim the output dir for this process. Returns the lockfile Path
    so the caller can unlink on exit.

    Raises :class:`RuntimeError` with a clear message when another
    pirlygenes process is already writing there and ``force`` is
    False. Stale locks (pid no longer alive) are reclaimed with a
    console note.
    """
    import json
    import os as _os
    from datetime import datetime

    lock_path = out_dir / _LOCKFILE_NAME
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text() or "{}")
            holder_pid = int(payload.get("pid") or 0)
            holder_started = str(payload.get("started_at") or "unknown time")
        except (ValueError, OSError):
            holder_pid = 0
            holder_started = "unknown time"
        if holder_pid and _pid_is_alive(holder_pid):
            if not force:
                raise RuntimeError(
                    f"Another pirlygenes analyze process (pid={holder_pid}, "
                    f"started {holder_started}) is writing to {out_dir}. "
                    "Use a different --output-dir, wait for it to finish, "
                    "or pass --force to override (only do this if you're "
                    "sure the lock is stale)."
                )
            print(
                f"[output] --force: ignoring live lock held by pid={holder_pid} "
                "(started {holder_started})"
            )
        else:
            print(
                f"[output] Cleaning stale lockfile (pid={holder_pid} "
                f"not running, started {holder_started})"
            )

    lock_path.write_text(
        json.dumps(
            {
                "pid": _os.getpid(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    return lock_path


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
    force: bool = False,
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

    # #82: advisory lock so a second concurrent analyze into the same
    # output dir fails fast instead of silently corrupting artifacts.
    lock_path = _acquire_output_dir_lock(out_dir, force=force)
    try:
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
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


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

    # Step 1 of the unified attribution flow: infer SampleContext BEFORE
    # cancer-type inference. Downstream steps (purity CIs, decomposition,
    # tumor-value adjustment, reporting) read from it as the base layer
    # of expression expectations.
    print("[context] Inferring sample context (library prep + preservation)...")
    sample_context = infer_sample_context(df_expr)
    print(f"[context] {sample_context.summary_line()}")
    for flag in sample_context.flags:
        print(f"[context] {flag}")

    # #149: Step-0 healthy-vs-tumor gate. Races the sample against
    # the 50 HPA normal-tissue columns + 33 TCGA cancer columns in
    # pan_cancer_expression() and checks a proliferation-panel
    # (MKI67+TOP2A+CCNB1+BIRC5+AURKA) geomean. Fixes the GTEx-style
    # mis-classification where a healthy sample gets force-called
    # into a TCGA cohort. Informational only — doesn't override
    # cancer-type inference; surfaces as a banner in brief/actionable
    # when the call is "healthy" or "ambiguous".
    from .healthy_vs_tumor import assess_healthy_vs_tumor
    try:
        healthy_vs_tumor = assess_healthy_vs_tumor(df_expr)
        print(f"[step-0] {healthy_vs_tumor.verdict}")
    except Exception as exc:  # noqa: BLE001
        print(f"[step-0] healthy-vs-tumor gate failed: {exc}")
        healthy_vs_tumor = None

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

    # Strip plot: therapy modalities. The per-category strip plots
    # (Immune_checkpoints / Oncogenes / CTAs / ...) are emitted
    # elsewhere; the aggregate immune / tumor / antigens overview
    # panels were retired as redundant in v4.46.0.
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

    # The ``immune`` / ``tumor`` / ``antigens`` overview strip plots
    # duplicate the 10 curated category strip plots (Immune_checkpoints,
    # Oncogenes, Tumor_suppressors, CTAs, Cancer_surfaceome, …) that
    # this CLI also emits. Per the figure audit (docs/figure-audit.md),
    # the overview set is the redundant one — retired in 4.46.0. The
    # ``treatments`` plot stays because it's organized by therapy
    # modality (ADC / TCR-T / CAR-T / bispecific / …), not by gene-set
    # category, so it's not covered by the per-category plots.
    strip_plots = [
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

    # Step 1 propagation: widen purity confidence intervals under
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
    print(f"[analysis] Stromal enrichment: {render_fold(purity['components']['stromal']['enrichment'])} vs TCGA")
    print(f"[analysis] Immune enrichment: {render_fold(purity['components']['immune']['enrichment'])} vs TCGA")
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
    # #77: pass the step-1 library_prep so the assessor skips the
    # "Suspicious MT fraction" override (and doesn't clobber the
    # length-pair-derived degradation level) when MT being near-zero is
    # explained by the inferred prep.
    quality = assess_sample_quality(
        df_expr,
        tissue_scores=analysis.get("tissue_scores"),
        library_prep=getattr(sample_context, "library_prep", None)
        if sample_context is not None else None,
    )
    analysis["quality"] = quality
    # #77: filter quality flags against the step-1 SampleContext — the
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
    # #149: make the Step-0 healthy-vs-tumor call available to
    # brief / actionable / summary renderers downstream.
    analysis["healthy_vs_tumor"] = healthy_vs_tumor
    for cls, score in therapy_scores.items():
        if score.state in ("up", "down"):
            tag = "[therapy-state]"
            print(f"{tag} {cls}: {score.state} — {score.message}")

    # Individual summary panels (replaces the crowded 4-panel composite, #97)
    summary_png = "%s-sample-summary.png" % prefix if prefix else "sample-summary.png"
    # Keep the composite for backward compatibility but also emit standalone PNGs
    plot_sample_summary(
        df_expr,
        cancer_type=cancer_code,
        sample_mode=analysis["sample_mode"],
        save_to_filename=summary_png,
        save_dpi=output_dpi,
        analysis=analysis,
    )
    hypotheses_png = "%s-cancer-hypotheses.png" % prefix
    plot_cancer_type_hypotheses(analysis, save_to_filename=hypotheses_png, save_dpi=output_dpi)
    tissues_png = "%s-background-tissues.png" % prefix
    plot_background_tissues(analysis, save_to_filename=tissues_png, save_dpi=output_dpi)
    mhc_png = "%s-mhc-expression.png" % prefix
    plot_mhc_expression(analysis, save_to_filename=mhc_png, save_dpi=output_dpi)

    # #136: one-figure therapy-pathway state readout. Renders the
    # disease-state narrative visually — dumbbells for each AR / NE /
    # EMT / hypoxia / IFN axis, caption restating the synthesis
    # sentence. Emitted only when at least one axis has a measurable
    # up/down geomean; skipped for empty therapy-scores dicts.
    pathway_state_png = (
        "%s-therapy-pathway-state.png" % prefix
        if prefix else "therapy-pathway-state.png"
    )
    try:
        from .plot_therapy import plot_therapy_pathway_state

        fig_ps = plot_therapy_pathway_state(
            therapy_response_scores=therapy_scores,
            cancer_code=cancer_code,
            disease_state_caption=compose_disease_state_narrative(analysis),
            save_to_filename=pathway_state_png,
            save_dpi=output_dpi,
        )
        if fig_ps is None:
            pathway_state_png = None
    except Exception as _tps_err:
        print(f"[warn] Therapy-pathway state plot failed: {_tps_err}")
        pathway_state_png = None

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
        # #198: surface the decomposition top template into the analysis
        # dict so the degenerate-subtype resolver (invoked from brief.py)
        # can consult it as a tiebreaker context. ``best_template`` is
        # the top-ranked template name (e.g. ``met_bone``); full ranked
        # list available for future tiebreakers that need the runner-up.
        analysis["decomposition"] = {
            "best_template": best_decomp.template,
            "best_cancer_type": best_decomp.cancer_type,
            "hypotheses": [
                {"template": d.template, "cancer_type": d.cancer_type,
                 "score": d.score}
                for d in decomp_results[:5]
            ],
        }
        # The classifier's top call (``cancer_code``) is the authoritative
        # cancer-type identification across every downstream report; the
        # decomposer only fits *subtraction templates* to separate tumor
        # from TME. When the best-fit template belongs to a different
        # cancer type than the classifier's call that's a decomposition-
        # fit observation, not a re-classification — do NOT overwrite
        # effective_cancer_type here (prior behavior leaked the template
        # label into brief/actionable/targets and recommended BRCA agents
        # on a COAD sample).
        #
        # The decomp's purity is only safe to adopt when its cancer_type
        # matches the classifier AND its template has non-tumor
        # compartments. A template without TME compartments trivially
        # returns fraction=1.0 (everything maps to tumor by construction),
        # which is not a purity measurement — the canonical failure
        # case is a CRC sample whose classifier said 36% (COAD) but
        # whose best-fit decomposition template was BRCA / solid_primary
        # with "No non-tumor components in template"; fraction=100%
        # propagated as the headline purity.
        decomp_agrees = (best_decomp.cancer_type == cancer_code)
        decomp_has_tme = not any(
            "No non-tumor components in template" in w
            for w in (best_decomp.warnings or [])
        )
        if decomp_agrees and decomp_has_tme and best_decomp.purity_result:
            effective_purity = best_decomp.purity_result

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

    # #124: side-by-side comparison of every purity estimation method
    # on a single purity axis. The existing plot_tumor_purity panel
    # mixes enrichment scales with purity % on the same row, which
    # hides how much the methods agree / disagree. This dedicated
    # figure renders signature / lineage / ESTIMATE stromal / ESTIMATE
    # immune / ESTIMATE combined / decomposition / adopted overall on
    # one purity axis with CI bars, plus TCGA cohort median reference.
    print("[plot] Generating purity-method comparison plot...")
    methods_png = (
        "%s-purity-methods.png" % prefix if prefix else "purity-methods.png"
    )
    best_for_methods = (
        decomp_results[0] if decomp_results else None
    )
    plot_purity_method_comparison(
        analysis["purity"],
        save_to_filename=methods_png,
        save_dpi=output_dpi,
        decomposition_result=best_for_methods,
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

    # Cancer-type signature gene grids (``plot_cancer_type_genes`` +
    # ``plot_cancer_type_disjoint_genes``) were removed from the default
    # plot set in 4.40.1 — they duplicated the candidate-ranking table
    # in analysis.md without adding interpretive value. The functions
    # remain in ``pirlygenes.plot_embedding`` for Python-API consumers.

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

    # Deep-dive therapy target + CTA + subtype plots
    from .plot_target_deep_dive import (
        plot_actionable_targets,
        plot_cta_deep_dive,
    )
    from .plot_subtype_signature import plot_subtype_signature

    p_est = purity.get("overall_estimate")
    try:
        targets_deep_png = "%s-targets-deep-dive.png" % prefix
        plot_actionable_targets(
            df_expr, cancer_type=effective_cancer_type,
            purity_estimate=p_est,
            save_to_filename=targets_deep_png, save_dpi=output_dpi,
        )
        print(f"[plot] Saved actionable targets deep dive to {targets_deep_png}")
    except Exception as exc:
        print(f"[plot] actionable targets deep dive failed: {exc}")
        targets_deep_png = None

    try:
        cta_deep_png = "%s-cta-deep-dive.png" % prefix
        plot_cta_deep_dive(
            df_expr, cancer_type=effective_cancer_type,
            purity_estimate=p_est,
            save_to_filename=cta_deep_png, save_dpi=output_dpi,
        )
        print(f"[plot] Saved CTA deep dive to {cta_deep_png}")
    except Exception as exc:
        print(f"[plot] CTA deep dive failed: {exc}")
        cta_deep_png = None

    # The pre-#108 ``plot_tumor_attribution`` (reference-based 2-color
    # tumor-vs-TME view) was removed from the default plot set in
    # v4.46.0 — ``plot_target_attribution`` (per-compartment stacked
    # bars keyed on the #108 attribution dict) is strictly more
    # informative and already emits per-category ``target-attribution-
    # *.png``. The older function remains in
    # ``pirlygenes.plot_target_deep_dive`` for Python-API consumers.
    attrib_targets_png = None
    attrib_cta_png = None

    subtype_png = None
    try:
        subtype_png = "%s-subtype-signature.png" % prefix
        fig = plot_subtype_signature(
            df_expr, cancer_type=effective_cancer_type,
            save_to_filename=subtype_png, save_dpi=output_dpi,
        )
        if fig is None:
            subtype_png = None
        else:
            print(f"[plot] Saved subtype signature to {subtype_png}")
    except Exception as exc:
        print(f"[plot] subtype signature failed: {exc}")
        subtype_png = None

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

        # Per-target compositional attribution (#108). One PNG per
        # category showing the per-gene tumor-core + compartment
        # breakdown. Emitted only when decomposition produced an
        # attribution; the function returns None otherwise and no file
        # is written, which the CLI respects by not appending to
        # adj_pngs.
        if (
            "attribution" in ranges_df.columns
            and ranges_df["attribution"].apply(
                lambda v: isinstance(v, dict) and len(v) > 0
            ).any()
        ):
            for cat_key, cat_slug in _adj_categories:
                attr_png = (
                    "%s-target-attribution-%s.png" % (prefix, cat_slug)
                    if prefix else "target-attribution-%s.png" % cat_slug
                )
                fig = plot_target_attribution(
                    ranges_df,
                    cancer_type=effective_cancer_type,
                    category=cat_key,
                    top_n=15,
                    save_to_filename=attr_png,
                    save_dpi=output_dpi,
                )
                if fig is not None:
                    adj_pngs.append(attr_png)
                _plt.close("all")

        # Per-gene subtype-refinement before / after bars (#56 / #58,
        # figure-audit C2). Only emitted when at least one gene in the
        # category got refined by the CAF / TAM / ... reference swap.
        # Separate PNG per category per the plot-crowding preference.
        if (
            "subtype_refined" in ranges_df.columns
            and ranges_df["subtype_refined"].astype(bool).any()
        ):
            from .plot_tumor_expr import plot_subtype_attribution
            for cat_key, cat_slug in _adj_categories:
                sub_png = (
                    "%s-subtype-attribution-%s.png" % (prefix, cat_slug)
                    if prefix else "subtype-attribution-%s.png" % cat_slug
                )
                fig = plot_subtype_attribution(
                    ranges_df,
                    category=cat_key,
                    top_n=15,
                    save_to_filename=sub_png,
                    save_dpi=output_dpi,
                )
                if fig is not None:
                    adj_pngs.append(sub_png)
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

        # #111: two-tier markdown handoff. Emitted after the detailed
        # reports so both can reference each other. The brief is the
        # doc a clinician pastes into a note; the actionable is the
        # one they read before a tumor board.
        try:
            from .brief import build_summary, build_actionable

            disease_state_for_summary = compose_disease_state_narrative(analysis)
            sample_id = prefix if prefix else None
            summary_md = build_summary(
                analysis,
                ranges_df,
                cancer_code=effective_cancer_type,
                disease_state=disease_state_for_summary,
                sample_id=sample_id,
            )
            actionable_md = build_actionable(
                analysis,
                ranges_df,
                cancer_code=effective_cancer_type,
                disease_state=disease_state_for_summary,
                sample_id=sample_id,
            )
            # The 1-page clinician-facing file is emitted as
            # ``*-summary.md``. The old free-form paragraph that used to
            # live at that name was retired in 4.41.0 — its disease-state,
            # step-0, and cancer-type text was ~80% redundant with
            # analysis.md.
            summary_path = "%s-summary.md" % prefix if prefix else "summary.md"
            actionable_path = (
                "%s-actionable.md" % prefix if prefix else "actionable.md"
            )
            with open(summary_path, "w") as f:
                f.write(summary_md)
            with open(actionable_path, "w") as f:
                f.write(actionable_md)
            print(f"[report] Saved {summary_path}")
            print(f"[report] Saved {actionable_path}")

            # Back-compat: emit ``*-brief.md`` as a duplicate of
            # ``*-summary.md`` for one deprecation window (removed in 5.0).
            # Downstream pipelines that glob for ``*-brief.md`` keep
            # working; new code should read ``*-summary.md``.
            brief_path = "%s-brief.md" % prefix if prefix else "brief.md"
            with open(brief_path, "w") as f:
                f.write(
                    "<!-- DEPRECATED in 4.41.0, removed in 5.0. "
                    "This file is a copy of *-summary.md; update your "
                    "pipelines to read the new name. -->\n\n"
                )
                f.write(summary_md)
            print(f"[report] Saved {brief_path} (deprecated copy of summary.md)")

            # #106: one-page provenance chain (library prep -> tumor
            # core). Emits *-provenance.md alongside a simple stacked-
            # bar figure showing the compartment composition.
            from .provenance import build_provenance_md, plot_provenance_funnel

            provenance_md = build_provenance_md(
                analysis,
                ranges_df,
                decomp_results,
                cancer_code=effective_cancer_type,
                sample_id=sample_id,
            )
            prov_path = "%s-provenance.md" % prefix if prefix else "provenance.md"
            with open(prov_path, "w") as f:
                f.write(provenance_md)
            print(f"[report] Saved {prov_path}")
            prov_png = "%s-provenance.png" % prefix if prefix else "provenance.png"
            fig_out = plot_provenance_funnel(
                analysis,
                ranges_df,
                decomp_results,
                save_to_filename=prov_png,
                save_dpi=output_dpi,
            )
            if fig_out:
                print(f"[plot] Saved {prov_png}")
        except Exception as brief_err:
            print(f"[warn] Brief / actionable rendering failed: {brief_err}")
    except Exception as e:
        print(f"[warn] Purity-adjusted analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # Collect all figures into one PDF (native resolution)
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

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
        # Standalone analysis panels (cancer hypotheses, background
        # tissues, MHC) — new outputs added while splitting the legacy
        # 4-panel composite. Without being listed here they never made
        # it into the PDF or the figures/ move step.
        hypotheses_png,
        tissues_png,
        mhc_png,
        # Purity-method comparison (#124) and sample-provenance (#106)
        # — added later and initially missed the figures/ move list.
        methods_png,
        # Provenance PNG may not exist when decomposition / range
        # computation was skipped; resolve by path rather than a
        # possibly-undefined variable.
        "%s-provenance.png" % prefix if prefix else "provenance.png",
        # #136: therapy-pathway-state dumbbell figure.
        pathway_state_png,
        "%s-treatments.png" % prefix if prefix else "treatments.png",
        safety_png,
    ] + embedding_pngs
    if ct_png:
        png_files.append(ct_png)
    # Deep-dive plots
    for _ddp in [targets_deep_png, cta_deep_png, attrib_targets_png, attrib_cta_png, subtype_png]:
        if _ddp and Path(_ddp).exists():
            png_files.append(_ddp)

    # Add per-category scatter PNGs from the vs-cancer output dir
    scatter_dir = Path(scatter_pdf).parent / Path(scatter_pdf).stem
    if scatter_dir.is_dir():
        png_files.extend(sorted(str(p) for p in scatter_dir.glob("*.png")))

    # Purity-adjusted plots go last (different RNA measure)
    for adj_p in adj_pngs:
        if Path(adj_p).exists():
            png_files.append(adj_p)

    def _with_filename_caption(img, filename):
        """Add a light-gray filename caption in a small strip below
        the figure so readers can locate the source PNG later. Caption
        sits in its own strip so it never overlaps the plot."""
        caption_h = 18
        new_w, new_h = img.width, img.height + caption_h
        canvas = Image.new("RGB", (new_w, new_h), color="white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # Bottom-right, light gray.
        text = filename
        if font is not None:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
            except AttributeError:  # pragma: no cover — very old PIL
                tw = len(text) * 6
        else:
            tw = len(text) * 6
        draw.text(
            (max(4, new_w - tw - 8), img.height + 3),
            text, fill="#AAAAAA", font=font,
        )
        return canvas

    images = []
    for png_path in png_files:
        if not png_path:
            continue
        p = Path(png_path)
        if p.exists():
            img = Image.open(p).convert("RGB")
            images.append(_with_filename_caption(img, p.name))

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
| `*-summary.md` | One-page clinician-facing summary (≤ 40 lines) — cancer call, purity, top therapies, caveats |
| `*-actionable.md` | Oncologist treatment-review — agents with cancer-type-indicated approvals + trials, cross-referenced against this sample |
| `*-analysis.md` | Structured deep-dive — disease-state, step-0 evidence, candidate trace, purity components, decomposition, background signatures |
| `*-targets.md` | Biomarker panel + full therapy-target landscape + tumor-expression ranges |
| `*-provenance.md` | Attribution chain — library-prep → preservation → TME → tumor-core step by step |
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
| `*-treatments.png` | Therapy target expression by modality |
| `*-target-safety.png` | Therapy target normal tissue expression |
| `*-purity-targets.png` | Tumor-expression ranges for therapeutic targets |
| `*-purity-ctas.png` | Tumor-expression ranges for CTAs |
| `*-purity-surface.png` | Tumor-expression ranges for surface proteins |
| `*-mds-tme.png` | MDS: sample among TCGA cancer types (TME-low gene space) |
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
    if tier == "degenerate":
        core += (
            " — **degenerate CI**: input had no per-gene variation so "
            "uncertainty could not be estimated (typical of synthetic "
            "/ cohort-median inputs)"
        )
    elif tier == "low":
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
        f"with {render_fold(purity['components']['stromal']['enrichment'])} stromal "
        f"and {render_fold(purity['components']['immune']['enrichment'])} immune enrichment "
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
        "These scores are normalised similarity to reference nTPM profiles "
        "— they summarize which residual non-tumor backgrounds the sample "
        "resembles after normalization. Useful context, not literal "
        "anatomical site calls or composition percentages in mixed samples.\n",
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
    """Write the detailed ``*-analysis.md`` report."""
    cancer_code = analysis["cancer_type"]
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
    family_display = family_summary.get("display")
    subtype_clause = family_summary.get("subtype_clause")

    # Disease-state synthesis (#78) — still used by analysis.md.
    disease_state_paragraph = compose_disease_state_narrative(analysis)

    # The old free-form ``*-summary.md`` paragraph (disease-state +
    # QC + headline + composition + therapy-response state) was
    # retired in 4.41.0 — every block already lived in analysis.md.
    # The name ``summary.md`` now carries what used to be
    # ``brief.md`` (the 1-page clinician file).

    # --- Detailed report ---
    lines = ["# Detailed Sample Analysis\n"]
    if input_path:
        # Input path at the top so the file is self-identifying even
        # without sample_context downstream. Propagated from
        # ``analyze()`` along with the rest of the analysis dict.
        lines.append(f"*Input*: `{input_path}`\n")

    # Disease-state synthesis (#78) — rendered at the top so a reader
    # sees the clinical framing before descending into pipeline detail.
    if disease_state_paragraph:
        lines.append(f"**Disease state**: {disease_state_paragraph}\n")

    # Step-0 tissue composition (#149) — same signal that drives the
    # brief banner and the summary top-line, surfaced here with the
    # full top-3 tissues / top-3 cohorts so a reader doing a deep
    # analysis.md read sees the Step-0 prior inline.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None and hvt.top_normal_tissues:
        from .gene_sets_cancer import proliferation_panel_gene_names
        _prolif_panel_size = len(proliferation_panel_gene_names())
        lines.append("## Step-0 tissue composition\n")
        lines.append(f"- **Cancer hint**: {hvt.cancer_hint}")
        lines.append(
            f"- **Proliferation panel**: "
            f"{hvt.proliferation_log2_mean:.2f} log2-TPM mean across "
            f"{hvt.proliferation_genes_observed}/{_prolif_panel_size} genes observed"
        )
        hpa_line = ", ".join(
            f"{t.replace('nTPM_', '').replace('_', ' ')} (ρ={rho:.2f})"
            for t, rho in hvt.top_normal_tissues[:3]
        )
        lines.append(f"- **Top normal-tissue matches**: {hpa_line}")
        tcga_line = ", ".join(
            f"{t.replace('FPKM_', '')} (ρ={rho:.2f})"
            for t, rho in hvt.top_tcga_cohorts[:3]
        )
        lines.append(f"- **Top TCGA cohort matches**: {tcga_line}")
        # Surface the reasoning trace (which rule fired + rationale)
        # so a reader can audit how Step-0 arrived at the hint.
        if hvt.reasoning_trace:
            lines.append(
                f"- **Reasoning rule**: {' → '.join(hvt.reasoning_trace)}"
            )
        # Full tumor-evidence breakdown (per-channel scores + aggregate)
        # so the hint isn't a black box.
        lines.append(f"- **Evidence**: {hvt.evidence.synthesis()}")
        lines.append("")

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

    # Sample quality — driven by the step-1 SampleContext so the
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
    # #33 source attribution — show whether the call was auto-detected
    # or user-specified. Used to live only in the retired summary.md.
    source_label = {
        "auto-detected": "auto-detected",
        "user-specified": "user-specified",
    }.get(analysis.get("cancer_type_source"))
    if source_label:
        lines.append(f"- **Call source**: {source_label}")
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
    # #109: compute the sample-level confidence tier once and surface it
    # inline so readers see a 19–100% CI render visibly different from a
    # 58–70% CI (#79). The tier consumes purity CI width, point estimate,
    # degradation severity, and sample-context flags.
    from .confidence import compute_purity_confidence

    sample_ctx_for_tier = analysis.get("sample_context")
    deg_for_tier = (
        getattr(sample_ctx_for_tier, "degradation_severity", "none")
        if sample_ctx_for_tier else "none"
    )
    purity_tier = compute_purity_confidence(
        purity,
        sample_context=sample_ctx_for_tier,
        degradation_severity=deg_for_tier,
    )
    analysis["purity_confidence"] = purity_tier
    if purity_tier.tier == "degenerate":
        tier_suffix = f" — **degenerate CI**: {purity_tier.inline_note}"
    elif purity_tier.tier in {"low", "moderate"} and purity_tier.reasons:
        tier_suffix = (
            f" — **{purity_tier.tier} confidence** "
            f"({purity_tier.inline_note})"
        )
    else:
        tier_suffix = ""
    lines.append(f"- **Overall estimate**: {purity['overall_estimate']:.0%} "
                  f"({purity['overall_lower']:.0%}\u2013{purity['overall_upper']:.0%})"
                  f"{tier_suffix}")
    components = purity.get("components", {})
    for comp_name in ("stromal", "immune"):
        comp = components.get(comp_name, {})
        if isinstance(comp, dict):
            enrichment = comp.get("enrichment", 0)
            lines.append(f"- **{comp_name.title()}** enrichment: {render_fold(enrichment)} vs TCGA")
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

        # Not found in sample — but distinguish genuinely absent from
        # detected-but-uninformative. The estimator skips lineage genes
        # whose TME bleed-through in the reference exceeds their tumor
        # contribution; those genes are still present in the sample at
        # real TPM, they just can't anchor a lineage purity. The
        # canonical case is SARC-ACTA2: ACTA2 at ~190 TPM in the
        # sample but TME-dominated at the TCGA SARC reference.
        from .tumor_purity import LINEAGE_GENES
        cancer_code_local = purity.get("cancer_type", cancer_code)
        all_lineage = LINEAGE_GENES.get(cancer_code_local, [])
        found_names = {g["gene"] for g in lineage_genes}
        skipped_detected = {
            entry["gene"]: entry
            for entry in lineage.get("skipped_detected", [])
        }
        not_found = [
            g for g in all_lineage
            if g not in found_names and g not in skipped_detected
        ]

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
        for gene_name, entry in skipped_detected.items():
            s_tpm = entry["sample_tpm"]
            # Keep resolution proportional to magnitude so a 0.3 TPM
            # gene doesn't render as "0 TPM".
            tpm_str = f"{s_tpm:.1f}" if s_tpm < 10 else f"{s_tpm:.0f}"
            lines.append(
                f"| {gene_name} | — | detected {tpm_str} TPM — "
                "uninformative (TME background exceeds tumor contribution "
                "for this cancer type) |"
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

    # Sample-context caveat (step 1 propagation): when the sample was
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

    # #110: cancer-type-scoped biomarker panel + therapy landscape.
    # Emitted before the general surface/intracellular tables so a
    # clinician reading top-down sees the curated, decision-relevant
    # rows first. Silently omitted for cancer types that aren't yet
    # curated in ``data/cancer-key-genes.csv`` — that's explicit, not
    # a fallback to the general tables.
    try:
        from .gene_sets_cancer import (
            cancer_biomarker_genes,
            cancer_therapy_targets,
            cancer_key_genes_cancer_types,
        )

        if cancer_code in cancer_key_genes_cancer_types():
            # Build symbol → row lookup from ranges_df for biomarker
            # expression levels.
            sym_to_row = {}
            for _, rrow in ranges_df.iterrows():
                sym_to_row[str(rrow["symbol"])] = rrow

            lines.append(f"## Biomarker Panel — {cancer_code}\n")
            lines.append(
                "Clinician-relevant biomarkers for this cancer type: "
                "lineage confirmation, disease-state indicators, and "
                "diagnostic markers. **Not therapy targets** — see the "
                "Therapy Landscape below for druggable genes.\n"
            )
            biomarker_syms = cancer_biomarker_genes(cancer_code)
            if biomarker_syms:
                lines.append("| Gene | Observed TPM | Tumor-attributed | Attribution |")
                lines.append("|------|--------------|------------------|-------------|")
                for sym in biomarker_syms:
                    row = sym_to_row.get(sym)
                    if row is None:
                        lines.append(f"| {sym} | *not measured* | — | — |")
                        continue
                    obs = float(row.get("observed_tpm") or 0.0)
                    attr_tumor = float(row.get("attr_tumor_tpm") or 0.0)
                    attribution_cell = _format_attribution_cell(row)
                    tumor_cell = (
                        f"{attr_tumor:.0f}" if row.get("attribution")
                        else "—"
                    )
                    lines.append(
                        f"| {sym} | {obs:.1f} | {tumor_cell} | {attribution_cell} |"
                    )
                lines.append("")
            else:
                lines.append("*No biomarker genes curated for this cancer type.*\n")

            lines.append(f"## Therapy Target Landscape — {cancer_code}\n")
            lines.append(
                "Approved and trialed agents with an indication for this "
                "cancer type, cross-referenced against sample expression. "
                "Rows where the target is absent from the sample are "
                "still shown to make that explicit.\n"
            )
            targets_df = cancer_therapy_targets(cancer_code)
            if len(targets_df):
                lines.append(
                    "| Target | Agent | Class | Phase | Indication | "
                    "Observed | Tumor-attr. | Attribution |"
                )
                lines.append(
                    "|--------|-------|-------|-------|------------|"
                    "----------|-------------|-------------|"
                )
                # Approved first, then phase_3, phase_2, phase_1,
                # preclinical. Within phase, agent name for stability.
                phase_order = {
                    "approved": 0, "phase_3": 1, "phase_2": 2,
                    "phase_1": 3, "preclinical": 4,
                }
                targets_sorted = targets_df.assign(
                    _phase_key=targets_df["phase"].map(
                        lambda p: phase_order.get(str(p), 99)
                    )
                ).sort_values(["_phase_key", "symbol", "agent"])
                def _cell(value):
                    if value is None:
                        return "—"
                    s = str(value).strip()
                    if s == "" or s.lower() == "nan":
                        return "—"
                    return s

                for _, trow in targets_sorted.iterrows():
                    sym = _cell(trow.get("symbol"))
                    agent = _cell(trow.get("agent"))
                    agent_class = _cell(trow.get("agent_class"))
                    phase = _cell(trow.get("phase")).replace("_", " ")
                    indication = _cell(trow.get("indication"))
                    # Agent-only rows (no gene target — e.g. doxorubicin,
                    # trabectedin for sarcoma) carry a blank symbol; skip the
                    # expression lookup so we don't render a "nan" row with
                    # false tumor TPM claims.
                    if sym == "—":
                        obs_cell = "*not measured*"
                        tumor_cell = "—"
                        attr_cell = "—"
                    else:
                        expr = sym_to_row.get(sym)
                        if expr is None:
                            obs_cell = "*not measured*"
                            tumor_cell = "—"
                            attr_cell = "—"
                        else:
                            obs_cell = f"{float(expr.get('observed_tpm') or 0.0):.1f}"
                            attr_tumor = float(expr.get("attr_tumor_tpm") or 0.0)
                            tumor_cell = (
                                f"{attr_tumor:.0f}" if expr.get("attribution")
                                else "—"
                            )
                            attr_cell = _format_attribution_cell(expr)
                    bold = "**" if phase == "approved" and sym != "—" else ""
                    lines.append(
                        f"| {bold}{sym}{bold} | {agent} | {agent_class} | "
                        f"{phase} | {indication} | {obs_cell} | "
                        f"{tumor_cell} | {attr_cell} |"
                    )
                lines.append("")
            else:
                lines.append(
                    "*No clinician-validated therapy targets curated for "
                    "this cancer type yet. The general Surface / "
                    "Intracellular tables below are ranked by raw "
                    "expression, not by approved indication.*\n"
                )
    except Exception as _lm_err:
        # Never fail the whole report on curation / loader issues.
        print(f"[warn] Could not render cancer-key-genes panel: {_lm_err}")

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
            # Treat missing therapy_supported as True (the common case for
            # rows without any therapy flag set); avoid .fillna on the
            # object-dtype series because pandas 2.x emits a FutureWarning
            # about silent downcasting. Mask + default keeps the intent
            # explicit.
            & ~ranges_df["therapy_supported"].map(
                lambda v: True if v is None or (isinstance(v, float) and pd.isna(v)) else bool(v)
            )
            & ranges_df["therapy_support_note"].fillna("").astype(str).str.len().gt(0)
        ]
        if len(fn1_blocked):
            landscape_cautions.append(
                fn1_blocked.sort_values("observed_tpm", ascending=False).iloc[0]["therapy_support_note"]
            )
    if landscape_cautions:
        lines.append(f"- **Landscape cautions**: {'; '.join(landscape_cautions)}.")
    lines.append("")

    # Metabolic axes (#158): Step-0 already computes proliferation /
    # hypoxia / glycolysis channel scores on TumorEvidenceScore but only
    # uses them for the Step-0 narrative. Surface any meaningfully-
    # elevated metabolic programs here so the therapy report names the
    # axes a reader could act on (CA9-directed ADCs, MCT inhibitors,
    # CDK4/6). Emits nothing when no channel clears the threshold.
    metabolic_rows = _metabolic_axes_rows(
        getattr(analysis.get("healthy_vs_tumor"), "evidence", None)
    )
    if metabolic_rows:
        lines.append("## Metabolic axes\n")
        lines.append(
            "Metabolic programs active on Step-0 scoring (proliferation / "
            "hypoxia / glycolysis). Rows emit only when the corresponding "
            "channel is meaningfully elevated.\n"
        )
        lines.append("| Axis | Signal | Therapy considerations |")
        lines.append("|------|--------|------------------------|")
        for axis, signal, therapy in metabolic_rows:
            lines.append(f"| {axis} | {signal} | {therapy} |")
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
            tme_warn = "⚠" if row.get("tme_explainable") else ""
            lines.append(
                f"| **{row['symbol']}** | {render_tpm(row['median_est'])} | "
                f"{render_tpm(row['est_1'])}\u2013{render_tpm(row['est_9'])} | "
                f"{render_tpm(row['observed_tpm'])} | "
                f"{_render_vs_tcga_cell(row)} | "
                f"{render_fraction_no_decimal(row['tcga_percentile'])} | "
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
            f"| Gene | {value_label} | Range | Observed | vs TCGA | TCGA %ile | TME | Attribution | Therapies |"
        )
        lines.append(
            "|------|-----------|-------|----------|---------|-----------|-----|-------------|-----------|"
        )
        # #78: cross-signal annotations — flag IFN-driven surface /
        # MHC targets when the IFN_response axis is active, so the
        # reader knows a 287× HLA-F fold change isn't tumor-specific.
        cross_notes = annotate_surface_targets_with_cross_signals(
            ranges_df, analysis.get("therapy_response_scores") or {}
        )
        for _, row in surface_targets.head(20).iterrows():
            bold = "**" if row["therapies"] else ""
            # TME flag — compact key:
            #   ⚠⚠ = ``tme_dominant`` (observed signal is mostly non-
            #   tumor per the decomposition attribution, #108); ⚠ =
            #   ``tme_explainable`` (a single healthy reference tissue
            #   could explain ≥50% of signal, #45).
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
            attribution_cell = _format_attribution_cell(row)
            lines.append(
                f"| {bold}{row['symbol']}{bold} | {render_tpm(row['median_est'])} | "
                f"{render_tpm(row['est_1'])}\u2013{render_tpm(row['est_9'])} | "
                f"{render_tpm(row['observed_tpm'])} | "
                f"{_render_vs_tcga_cell(row)} | "
                f"{render_fraction_no_decimal(row['tcga_percentile'])} | "
                f"{tme_warn} | {attribution_cell} | {therapies_cell} |"
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
                "\n⚠⚠ = **TME-dominant**: the decomposition attribution "
                "assigns less than 30% of the observed TPM to the tumor "
                "compartment (#108). These targets are very likely "
                "stromal / immune rather than tumor-cell expressed — "
                "excluding them from therapy-target consideration is the "
                "safest default (#35)."
            )
            lines.append(
                "\nAttribution column shows `tumor {tpm} / {dominant "
                "non-tumor compartment} {tpm}` from the decomposition "
                "fit; `—` means no decomposition attribution was "
                "available for this gene. A trailing `· broadly expr.` "
                "flag (#128) means the gene is detected above 5 nTPM "
                "in many non-reproductive HPA tissues and has no "
                "strongly enriched tissue — tumor-cell specificity is "
                "not supported regardless of the numeric tumor-attributed "
                "TPM."
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
        lines.append(f"| Gene | {value_label} | Range | vs TCGA | TCGA %ile | TME | Attribution | CTA | Therapies |")
        lines.append("|------|-----------|-------|---------|-----------|-----|-------------|-----|-----------|")
        for _, row in intracellular.head(15).iterrows():
            cta_flag = "yes" if row["is_cta"] else ""
            tme_warn = "⚠" if row.get("tme_explainable") else ""
            attribution_cell = _format_attribution_cell(row)
            lines.append(
                f"| {row['symbol']} | {render_tpm(row['median_est'])} | "
                f"{render_tpm(row['est_1'])}\u2013{render_tpm(row['est_9'])} | "
                f"{_render_vs_tcga_cell(row)} | "
                f"{render_fraction_no_decimal(row['tcga_percentile'])} | "
                f"{tme_warn} | {attribution_cell} | {cta_flag} | {row['therapies']} |"
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
    dispatch_commands([print_dataset_info, print_cancer_registry, analyze, plot_expression, plot_cancer_cohorts])


if __name__ == "__main__":
    main()
