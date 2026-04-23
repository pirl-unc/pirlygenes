"""Shared report-facing helpers for evidence and attribution reliability.

These helpers are intentionally narrow: they do not change the raw
attribution math, only how downstream markdown decides whether a row is
safe to summarize as credible tumor-core signal. The goal is to keep the
headline markdown blocks aligned with the caveats already present in the
tables and TSVs.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache


def _truthy(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    try:
        return bool(value)
    except Exception:
        return False


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _safe_float(value, default=0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def tumor_attribution_context(row):
    """Summarize how securely a row stays tumor-attributed."""
    observed = _safe_float(row.get("observed_tpm"), 0.0)
    mid_tpm = _safe_float(row.get("attr_tumor_tpm"), 0.0)
    low_tpm = _safe_float(row.get("attr_tumor_tpm_low"), mid_tpm)
    high_tpm = _safe_float(row.get("attr_tumor_tpm_high"), mid_tpm)
    mid_frac = _safe_float(row.get("attr_tumor_fraction"), 0.0)
    low_frac = _safe_float(row.get("attr_tumor_fraction_low"), mid_frac)
    high_frac = _safe_float(row.get("attr_tumor_fraction_high"), mid_frac)
    support_fraction = _safe_float(
        row.get("attr_support_fraction"),
        1.0 if (mid_tpm >= 1.0 and mid_frac >= 0.30) else 0.0,
    )
    low_tpm, mid_tpm, high_tpm = sorted([max(0.0, low_tpm), max(0.0, mid_tpm), max(0.0, high_tpm)])
    low_frac, mid_frac, high_frac = sorted(
        [
            max(0.0, min(1.0, low_frac)),
            max(0.0, min(1.0, mid_frac)),
            max(0.0, min(1.0, high_frac)),
        ]
    )
    support_fraction = max(0.0, min(1.0, support_fraction))

    notes = []
    if _truthy(row.get("matched_normal_over_predicted")):
        notes.append("matched-normal reference overshoots the sample")
    if _truthy(row.get("low_purity_cap_applied")):
        notes.append("low-purity cap is active")
    if _truthy(row.get("tme_explainable")) and support_fraction < 1.0:
        notes.append("healthy-tissue-only explanations remain plausible")
    if _truthy(row.get("tme_dominant")):
        notes.append("most fitted signal remains non-tumor")

    if high_tpm < 1.0 or high_frac < 0.30 or _truthy(row.get("tme_dominant")):
        tier = "background_dominant"
        label = "background-dominant"
        summary = "non-tumor compartments remain the simpler explanation"
    elif (
        low_tpm >= 1.0
        and low_frac >= 0.30
        and support_fraction >= 0.67
        and not _truthy(row.get("matched_normal_over_predicted"))
    ):
        tier = "tumor_supported"
        label = "tumor-supported"
        summary = "tumor-attributed signal stays material across the uncertainty band"
    else:
        tier = "mixed_source"
        label = "mixed-source"
        summary = "both tumor and benign/background sources remain plausible"

    if observed > 0:
        band = f"{mid_tpm:.0f} TPM (band {low_tpm:.0f}-{high_tpm:.0f}; {mid_frac:.0%} tumor, {low_frac:.0%}-{high_frac:.0%} band)"
    else:
        band = f"{mid_tpm:.0f} TPM"

    return {
        "tier": tier,
        "label": label,
        "summary": summary,
        "notes": notes,
        "band": band,
        "observed_tpm": observed,
        "attr_tumor_tpm": mid_tpm,
        "attr_tumor_tpm_low": low_tpm,
        "attr_tumor_tpm_high": high_tpm,
        "attr_tumor_fraction": mid_frac,
        "attr_tumor_fraction_low": low_frac,
        "attr_tumor_fraction_high": high_frac,
        "attr_support_fraction": support_fraction,
    }


def tumor_attribution_band_text(row):
    return tumor_attribution_context(row)["band"]


def target_reliability_reasons(row, *, category=None):
    """Return ordered reader-facing caveats for a target/expression row."""
    source = tumor_attribution_context(row)
    reasons = []
    if source["tier"] == "background_dominant":
        reasons.append("background-dominant")
    elif source["tier"] == "mixed_source":
        reasons.append("mixed-source")
    if _truthy(row.get("matched_normal_over_predicted")):
        reasons.append("matched-normal over-predicted")
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        reasons.append("possible smooth-muscle stromal leakage")
    if _truthy(row.get("tme_explainable")) and source["tier"] != "tumor_supported":
        reasons.append("healthy-tissue-explainable")
    if _truthy(row.get("low_purity_cap_applied")):
        reasons.append("low-purity capped")
    return reasons


def target_reliability_status(row, *, category=None):
    """Classify a row as ``supported``, ``provisional``, or ``unsupported``."""
    source = tumor_attribution_context(row)
    if source["tier"] == "background_dominant":
        return "unsupported"
    if _truthy(row.get("matched_normal_over_predicted")):
        return "provisional"
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        return "provisional"
    if source["tier"] == "mixed_source":
        return "provisional"
    return "supported"


_ESSENTIAL_TISSUE_COLS = {
    "brain": [
        "nTPM_cerebral_cortex", "nTPM_cerebellum", "nTPM_basal_ganglia",
        "nTPM_hippocampal_formation", "nTPM_amygdala", "nTPM_midbrain",
        "nTPM_hypothalamus", "nTPM_spinal_cord", "nTPM_choroid_plexus",
    ],
    "heart": ["nTPM_heart_muscle"],
    "liver": ["nTPM_liver"],
    "lung": ["nTPM_lung"],
    "kidney": ["nTPM_kidney"],
    "bone_marrow": ["nTPM_bone_marrow"],
    "spleen": ["nTPM_spleen"],
    "pancreas": ["nTPM_pancreas"],
    "colon": ["nTPM_colon", "nTPM_rectum", "nTPM_duodenum", "nTPM_small_intestine"],
    "stomach": ["nTPM_stomach"],
}


@lru_cache(maxsize=1)
def _pan_cancer_normal_expression_index():
    try:
        from .gene_sets_cancer import pan_cancer_expression

        df = pan_cancer_expression().copy()
        if "Ensembl_Gene_ID" not in df.columns:
            return None
        df["Ensembl_Gene_ID"] = df["Ensembl_Gene_ID"].astype(str).str.split(".", n=1).str[0]
        df = df.drop_duplicates(subset="Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")
        return df
    except Exception:
        return None


def _healthy_expression_profile(gene_id):
    gene_id = _clean_text(gene_id).split(".", 1)[0]
    if not gene_id:
        return {}
    ref = _pan_cancer_normal_expression_index()
    if ref is None or gene_id not in ref.index:
        return {}
    row = ref.loc[gene_id]
    profile = {}
    for tissue, columns in _ESSENTIAL_TISSUE_COLS.items():
        values = []
        for column in columns:
            if column not in row.index:
                continue
            try:
                value = float(row[column])
            except Exception:
                continue
            if value == value:
                values.append(value)
        if values:
            profile[tissue] = max(values)
    return profile


def normal_expression_context(row):
    """Return a coarse reader-facing normal-expression context label."""
    category_label = str(row.get("category") or "").upper()
    if _truthy(row.get("is_cta")) or category_label == "CTA":
        return {
            "tier": "cta_restricted",
            "label": "CTA / immune-privileged",
            "summary": "CTA-style restricted normal expression",
            "details": [],
        }

    matched_tissue = _clean_text(row.get("matched_normal_tissue")).replace("_", " ")
    matched_normal_tpm = _safe_float(row.get("matched_normal_tpm"), 0.0)
    tumor_tpm = _safe_float(row.get("attr_tumor_tpm"), 0.0)
    attr_top = _clean_text(row.get("attr_top_compartment")).replace("_", " ")
    details = []
    same_lineage = False
    if matched_tissue:
        same_lineage = (
            attr_top == f"matched normal {matched_tissue}"
            or _truthy(row.get("matched_normal_over_predicted"))
            or matched_normal_tpm >= max(1.0, tumor_tpm * 0.5)
        )

    healthy_profile = _healthy_expression_profile(row.get("gene_id"))
    vital_detail = ""
    if healthy_profile:
        top_tissue, top_value = max(healthy_profile.items(), key=lambda item: item[1])
        if top_value >= 10.0:
            vital_detail = f"{top_tissue.replace('_', ' ')} expression is appreciable"

    if _truthy(row.get("broadly_expressed")) or _safe_int(row.get("n_healthy_tissues_expressed"), 0) >= 15:
        details.append("broader healthy-tissue signal is present outside the matched lineage")
    if vital_detail:
        details.append(vital_detail)

    if same_lineage:
        return {
            "tier": "same_lineage_expected",
            "label": "same-lineage expected",
            "summary": f"benign {matched_tissue} lineage expression is expected",
            "details": details,
        }

    if vital_detail:
        return {
            "tier": "vital_tissue_concern",
            "label": "vital-tissue concern",
            "summary": vital_detail,
            "details": details[1:] if details else [],
        }

    if details:
        return {
            "tier": "broad_healthy_expression",
            "label": "broad healthy expression",
            "summary": details[0],
            "details": details[1:],
        }

    return {
        "tier": "restricted_outside_lineage",
        "label": "restricted outside matched lineage",
        "summary": "limited healthy-tissue signal outside the matched lineage",
        "details": [],
    }


def clinical_maturity_info(target_row, target_panel=None):
    """Summarize maturity from the current row and its curated siblings."""
    phase = _clean_text(target_row.get("phase"))
    phase_label = {
        "approved": "approved",
        "phase_3": "late clinical",
        "phase_2": "mid-clinical",
        "phase_1": "early clinical",
        "preclinical": "preclinical",
    }.get(phase, phase or "curated")
    agent_class = _clean_text(target_row.get("agent_class")).replace("_", " ")
    summary = phase_label
    if agent_class:
        summary += f" {agent_class}"

    if target_panel is None or len(target_panel) == 0:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 0,
            "n_modalities": 0,
        }

    symbol = _clean_text(target_row.get("symbol"))
    if not symbol or "symbol" not in target_panel.columns:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 0,
            "n_modalities": 0,
        }

    try:
        sub = target_panel[target_panel["symbol"].astype(str) == symbol]
    except Exception:
        sub = None

    if sub is None or len(sub) <= 1:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 1 if sub is not None and len(sub) == 1 else 0,
            "n_modalities": 1 if sub is not None and len(sub) == 1 else 0,
        }

    n_agents = (
        sub["agent"].fillna("").astype(str).str.strip()
        .replace("nan", "").loc[lambda s: s.ne("")].nunique()
        if "agent" in sub.columns else 0
    )
    n_modalities = (
        sub["agent_class"].fillna("").astype(str).str.strip()
        .replace("nan", "").loc[lambda s: s.ne("")].nunique()
        if "agent_class" in sub.columns else 0
    )
    extras = []
    if n_agents > 1:
        extras.append(f"{n_agents} curated agents")
    if n_modalities > 1:
        extras.append(f"{n_modalities} modalities")
    if extras:
        summary += f" ({', '.join(extras)})"
    return {
        "tier": phase_label,
        "summary": summary,
        "n_agents": int(n_agents),
        "n_modalities": int(n_modalities),
    }


def clinical_maturity_summary(target_row, target_panel=None):
    return clinical_maturity_info(target_row, target_panel=target_panel)["summary"]


def target_interpretation_summary(target_row, expression_row, target_panel=None):
    """Return a compact integrated summary for a curated target row."""
    source = tumor_attribution_context(expression_row)
    normal = normal_expression_context(expression_row)
    maturity = (
        clinical_maturity_summary(target_row, target_panel=target_panel)
        if target_row is not None else ""
    )
    parts = [source["label"], source["band"], normal["label"]]
    details = list(normal.get("details") or [])
    if details:
        parts.append(details[0])
    if maturity:
        parts.append(maturity)
    return "; ".join(part for part in parts if part)


def partition_tumor_core_rows(ranges_df, min_tumor_tpm=1.0):
    """Split expression rows by report-facing tumor-core reliability."""
    if ranges_df is None or len(ranges_df) == 0 or "attr_tumor_tpm" not in ranges_df.columns:
        empty = ranges_df.iloc[0:0] if ranges_df is not None else None
        return empty, empty, empty

    eligible = ranges_df[ranges_df["attr_tumor_tpm"].astype(float) >= float(min_tumor_tpm)].copy()
    if eligible.empty:
        empty = eligible.iloc[0:0]
        return empty, empty, empty

    statuses = [
        target_reliability_status(row)
        for _, row in eligible.iterrows()
    ]
    eligible["_report_reliability"] = statuses
    supported = eligible[eligible["_report_reliability"] == "supported"].copy()
    provisional = eligible[eligible["_report_reliability"] == "provisional"].copy()
    unsupported = eligible[eligible["_report_reliability"] == "unsupported"].copy()
    return supported, provisional, unsupported


def summarize_reliability_reasons(rows, top_n=3):
    """Summarize the most common reliability caveats in a row set."""
    if rows is None or len(rows) == 0:
        return ""
    counter = Counter()
    for _, row in rows.iterrows():
        counter.update(target_reliability_reasons(row))
    if not counter:
        return ""
    return ", ".join(reason for reason, _ in counter.most_common(top_n))


def resolved_subtype_code_for_analysis(analysis, ranges_df=None):
    """Return the final subtype/cancer code implied by the analysis."""
    candidate_trace = analysis.get("candidate_trace") or []
    if not candidate_trace:
        return None
    winning_subtype = _clean_text(candidate_trace[0].get("winning_subtype"))
    if not winning_subtype:
        return None

    tumor_tpm_by_symbol = analysis.get("tumor_tpm_by_symbol")
    if not tumor_tpm_by_symbol and ranges_df is not None:
        try:
            tumor_tpm_by_symbol = {
                str(row["symbol"]): float(row.get("attr_tumor_tpm") or 0.0)
                for _, row in ranges_df.iterrows()
                if str(row.get("symbol") or "")
            }
        except Exception:
            tumor_tpm_by_symbol = None

    try:
        from .degenerate_subtype import resolve_degenerate_subtype

        decomposition = analysis.get("decomposition") or {}
        site_template = decomposition.get("best_template")
        resolution = resolve_degenerate_subtype(
            winning_subtype,
            site_template=site_template,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        winning_subtype = _clean_text(
            resolution.get("final_subtype")
        ) or winning_subtype
    except Exception:
        pass
    return winning_subtype or None


def _match_curated_subtype(parent_code, *candidates):
    """Return the exact curated subtype string matching any candidate."""
    if not parent_code:
        return None
    try:
        from .gene_sets_cancer import cancer_key_genes_subtypes

        curated_subtypes = [
            _clean_text(item)
            for item in cancer_key_genes_subtypes(parent_code)
        ]
    except Exception:
        return None

    curated_subtypes = [item for item in curated_subtypes if item]
    if not curated_subtypes:
        return None

    exact_lookup = {item: item for item in curated_subtypes}
    lower_lookup = {item.lower(): item for item in curated_subtypes}

    for candidate in candidates:
        raw = _clean_text(candidate)
        if not raw:
            continue
        normalized = raw.replace("-", "_")
        variants = (
            raw,
            normalized,
            raw.lower(),
            normalized.lower(),
            raw.upper(),
            normalized.upper(),
        )
        for variant in variants:
            if variant in exact_lookup:
                return exact_lookup[variant]
        for variant in variants:
            match = lower_lookup.get(str(variant).lower())
            if match:
                return match
    return None


def cancer_key_genes_lookup_for_analysis(cancer_code, analysis, ranges_df=None):
    """Return ``(panel_code, panel_subtype)`` for curated report lookups.

    Most samples use their top-level cancer code directly. Some need a
    subtype-filtered panel (for example ``SARC`` + ``leiomyosarcoma``),
    while others resolve onto a different curated code entirely (for
    example a ``SARC`` umbrella call that degeneracy-resolution upgrades
    to ``OS``).
    """
    default_code = _clean_text(cancer_code)
    resolved_code = resolved_subtype_code_for_analysis(
        analysis,
        ranges_df=ranges_df,
    )

    try:
        from .gene_sets_cancer import (
            cancer_key_genes_cancer_types,
            cancer_type_registry,
        )

        curated_codes = {
            _clean_text(code)
            for code in cancer_key_genes_cancer_types()
        }
        if resolved_code and resolved_code in curated_codes:
            return resolved_code, None

        if resolved_code:
            reg = cancer_type_registry()
            match = reg[reg["code"] == resolved_code]
            if not match.empty:
                row = match.iloc[0]
                parent_code = _clean_text(row.get("parent_code"))
                subtype_key = _clean_text(row.get("subtype_key"))
                suffix = ""
                if parent_code and resolved_code.startswith(parent_code + "_"):
                    suffix = resolved_code[len(parent_code) + 1:]
                matched_subtype = _match_curated_subtype(
                    parent_code,
                    subtype_key,
                    suffix,
                )
                if parent_code and matched_subtype:
                    return parent_code, matched_subtype

        if default_code and default_code in curated_codes:
            return default_code, None
    except Exception:
        pass

    return default_code or resolved_code, None


def subtype_key_for_analysis(analysis, ranges_df=None):
    """Return the curated subtype key implied by the current analysis."""
    winning_subtype = resolved_subtype_code_for_analysis(
        analysis,
        ranges_df=ranges_df,
    )
    if not winning_subtype:
        return None

    try:
        from .gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == winning_subtype]
        if match.empty:
            return None
        row = match.iloc[0]
        subtype_key = str(row.get("subtype_key") or "").strip()
        if subtype_key and subtype_key.lower() != "nan":
            return subtype_key
        parent_code = str(row.get("parent_code") or "").strip()
        code = str(row.get("code") or "").strip()
        if parent_code and code.startswith(parent_code + "_"):
            suffix = code[len(parent_code) + 1:].strip().lower()
            return suffix or None
    except Exception:
        return None
    return None
