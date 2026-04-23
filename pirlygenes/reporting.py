"""Shared report-facing helpers for evidence and attribution reliability.

These helpers are intentionally narrow: they do not change the raw
attribution math, only how downstream markdown decides whether a row is
safe to summarize as credible tumor-core signal. The goal is to keep the
headline markdown blocks aligned with the caveats already present in the
tables and TSVs.
"""

from __future__ import annotations

from collections import Counter


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


def target_reliability_reasons(row, *, category=None):
    """Return ordered reader-facing caveats for a target/expression row."""
    reasons = []
    if _truthy(row.get("tme_dominant")):
        reasons.append("TME-dominant")
    if _truthy(row.get("matched_normal_over_predicted")):
        reasons.append("matched-normal over-predicted")
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        reasons.append("possible smooth-muscle stromal leakage")
    if _truthy(row.get("broadly_expressed")):
        reasons.append("broadly expressed")
    if _truthy(row.get("tme_explainable")):
        reasons.append("single-tissue-explainable")
    if _truthy(row.get("low_purity_cap_applied")):
        reasons.append("low-purity capped")
    return reasons


def target_reliability_status(row, *, category=None):
    """Classify a row as ``supported``, ``provisional``, or ``unsupported``."""
    category_label = str(category or row.get("category") or "").upper()

    if _truthy(row.get("tme_dominant")):
        return "unsupported"

    if _truthy(row.get("matched_normal_over_predicted")):
        return "provisional"

    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        return "provisional"

    if _truthy(row.get("low_purity_cap_applied")):
        return "provisional"

    if _truthy(row.get("broadly_expressed")):
        return "provisional"

    if _truthy(row.get("tme_explainable")):
        # CTA rows often carry this because cohort medians are near zero
        # by design; keep them provisional rather than unsupported.
        return "provisional" if category_label == "CTA" else "provisional"

    return "supported"


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
