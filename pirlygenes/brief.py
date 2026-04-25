# Licensed under the Apache License, Version 2.0

"""Two-tier markdown handoff documents (#111).

Audience distinction:

- ``*-summary.md`` — one-page summary, ≤ 40 lines. For a clinician
  skimming before a tumor board, or an LLM asked for a 3-sentence
  referral-note paragraph. Strict structure; no internal jargon.
  (Named ``brief.md`` through 4.40; the earlier free-form
  ``summary.md`` paragraph was retired as redundant with analysis.md.)
- ``*-actionable.md`` — longer treatment-review document. For an
  oncologist preparing a treatment discussion, reading carefully.

Both consume:

- ``analysis`` (the shared dict produced by ``analyze_sample`` and
  enriched by the CLI pipeline), including ``purity_confidence`` from
  :mod:`pirlygenes.confidence`.
- ``ranges_df`` (per-gene tumor-expression + #108 attribution).
- The disease-state narrative from ``compose_disease_state_narrative``.
- The curated cancer-key-genes panels from ``gene_sets_cancer``.

No JSON is produced — consumers read markdown. (See
``feedback_markdown_not_json`` in the memory: the audience test for a
new output is "who reads this and when?", not "is it machine-
parseable?".)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .reporting import (
    cancer_code_display_name,
    cancer_key_genes_lookup_for_analysis,
    clinical_maturity_summary,
    expression_independent_indication,
    expression_independent_interpretation,
    expression_independent_rna_context,
    normal_expression_context,
    report_disease_state_text,
    same_lineage_material_target_candidate,
    subtype_curation_scope_note,
    therapy_path_context,
    therapy_path_rank,
    therapy_state_caution,
    tumor_band_available,
    tumor_band_cell,
    target_reliability_status,
    tpm_semantics_note,
    tumor_attribution_context,
)
from .confidence import concise_confidence_reasons
from .sample_context import library_prep_clause, library_prep_display_label

logger = logging.getLogger(__name__)


def _display_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    text = str(sample_id).strip()
    if not text:
        return None
    if "/" in text or "\\" in text:
        text = Path(text).name.strip()
    return text or None


def _display_subtype_code(code: Optional[str]) -> str:
    text = str(code or "").strip()
    if not text:
        return "the alternate subtype"
    try:
        from .gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == text]
        if not match.empty:
            row = match.iloc[0]
            subtype_key = row.get("subtype_key")
            if isinstance(subtype_key, str) and subtype_key and subtype_key.lower() != "nan":
                return subtype_key.replace("_", " ")
            name = row.get("name")
            if isinstance(name, str) and name:
                return name.split("(")[0].strip().lower()
    except Exception:
        logger.debug("subtype display lookup failed", exc_info=True)
    return text.replace("_", " ").lower()


def _call_confidence_suffix(call_tier, *, concise: bool = True) -> str:
    """Render cancer-call confidence consistently across Markdown reports."""
    if call_tier.tier not in {"low", "moderate"} or not call_tier.reasons:
        return ""
    tier_text = f"{call_tier.tier} confidence"
    if call_tier.tier == "low":
        tier_text += ", provisional"
    note = concise_confidence_reasons(call_tier) if concise else call_tier.inline_note
    return f" — **{tier_text}** ({note})" if note else f" — **{tier_text}**"


def _site_template_note_label(template_name: Optional[str]) -> str:
    mapping = {
        "met_adrenal": "adrenal-associated",
        "met_bone": "bone-associated",
        "met_brain": "brain-associated",
        "met_liver": "liver-associated",
        "met_lung": "lung-associated",
        "met_lymph_node": "lymph-node-associated",
        "met_peritoneal": "peritoneal-associated",
        "met_skin": "skin-associated",
        "met_soft_tissue": "soft-tissue-associated",
        "solid_primary": "primary-site-compatible",
    }
    text = str(template_name or "").strip()
    if not text:
        return "site"
    return mapping.get(text, text.replace("_", " "))


def _shared_signature_note(shared_signature: Optional[str]) -> str:
    text = str(shared_signature or "").strip()
    if not text:
        return "shared lineage pattern"
    text = text.replace("_", " ")
    text = text.replace("+", " / ")
    text = text.replace(" amp", " amplification")
    return text


def _render_subtype_note(
    resolution: dict,
    *,
    original_subtype: Optional[str],
    site_template: Optional[str],
) -> str:
    if not resolution:
        return ""
    status = str(resolution.get("status") or "")
    rule = str(resolution.get("rule") or "")
    shared_signature = _shared_signature_note(resolution.get("shared_signature"))
    final_label = _display_subtype_code(resolution.get("final_subtype"))
    original_label = _display_subtype_code(original_subtype)
    alternatives = [
        _display_subtype_code(code)
        for code in (resolution.get("alternatives") or [])
    ]

    if status == "corrected":
        if rule == "site_template":
            site_label = _site_template_note_label(site_template)
            return (
                f"{site_label.capitalize()} context favors {final_label} over {original_label}; "
                f"both can share the {shared_signature}."
            )
        if rule == "fusion_surrogate":
            return (
                f"Fusion-surrogate expression favors {final_label} over {original_label}; "
                f"both can share the {shared_signature}."
            )
        if rule == "marker_combo":
            return (
                f"The marker combination is more consistent with {final_label} than {original_label}; "
                f"both can share the {shared_signature}."
            )
        return (
            f"Additional subtype evidence favors {final_label} over {original_label}; "
            f"both can share the {shared_signature}."
        )

    if status == "degenerate":
        option_text = " vs ".join([final_label] + alternatives) if alternatives else final_label
        if rule == "site_template":
            return (
                f"Subtype remains unresolved between {option_text}; the available site context does not break the tie, "
                f"and these options can share the {shared_signature}."
            )
        if rule == "fusion_surrogate":
            return (
                f"Subtype remains unresolved between {option_text}; the available fusion-surrogate expression does not break the tie, "
                f"and these options can share the {shared_signature}."
            )
        if rule == "marker_combo":
            return (
                f"Subtype remains unresolved between {option_text}; the available marker combination does not break the tie, "
                f"and these options can share the {shared_signature}."
            )
        return (
            f"Subtype remains unresolved between {option_text}; the available evidence does not break the tie, "
            f"and these options can share the {shared_signature}."
        )

    return str(resolution.get("reason") or "").strip()


def _phase_label(phase: str) -> str:
    return {
        "approved": "Approved",
        "phase_3": "Phase 3",
        "phase_2": "Phase 2",
        "phase_1": "Phase 1",
        "preclinical": "Preclinical",
    }.get(phase, phase)


def _top_candidate_signature_score(analysis) -> float | None:
    """Return the top-ranked cancer candidate's signature match score.

    Used by the Step-0 banner to suppress noise: a confident TCGA
    signature match is evidence of tumor and nudges the banner to
    stay silent on soft "composition-ambiguous" cases.
    """
    candidates = (
        analysis.get("cancer_candidates")
        or analysis.get("candidate_trace")
        or []
    )
    if not candidates:
        return None
    top = candidates[0]
    # Different code paths store this under slightly different keys.
    for key in ("signature_score", "support_norm", "geomean", "normalized"):
        if key in top and top[key] is not None:
            try:
                return float(top[key])
            except (TypeError, ValueError):
                continue
    return None


def _format_therapy_bullet(
    target_row,
    expression_row,
    target_panel=None,
    *,
    analysis=None,
    disease_state=None,
) -> str:
    """One standardized therapy bullet for the brief."""
    sym = str(target_row.get("symbol") or "")
    agent = str(target_row.get("agent") or "—")
    phase = _phase_label(str(target_row.get("phase") or ""))
    indication = str(target_row.get("indication") or "")
    indication_clause = f", {indication}" if indication else ""
    expr_independent = expression_independent_indication(target_row)
    path_context = therapy_path_context(
        target_row,
        analysis=analysis,
        disease_state=disease_state,
    )
    path_prefix = f"{path_context}. " if path_context else ""
    state_caution = therapy_state_caution(
        target_row,
        analysis=analysis,
        disease_state=disease_state,
    )
    caution_suffix = (
        f" Current-therapy check: {state_caution}."
        if state_caution else ""
    )
    maturity = clinical_maturity_summary(target_row, target_panel=target_panel)

    def _sentence(parts, *, maturity: str | None = None) -> str:
        body = "; ".join(part for part in parts if part)
        if maturity:
            return f"{body}. Clinical maturity: {maturity}."
        return f"{body}."

    if expression_row is None:
        if expr_independent:
            parts = [
                expression_independent_interpretation(target_row),
                expression_independent_rna_context(None),
            ]
            if path_context:
                parts.append(path_context)
            return (
                f"- **{sym}** — {agent} ({phase}{indication_clause}). "
                f"{_sentence(parts, maturity=maturity)}{caution_suffix}"
            )
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"{path_prefix}Target **not measured** in this sample.{caution_suffix}"
        )
    observed = float(expression_row.get("observed_tpm") or 0.0)
    if observed < 1.0:
        if expr_independent:
            parts = [
                expression_independent_interpretation(target_row),
                expression_independent_rna_context(expression_row),
            ]
            if path_context:
                parts.append(path_context)
            return (
                f"- **{sym}** — {agent} ({phase}{indication_clause}). "
                f"{_sentence(parts, maturity=maturity)}{caution_suffix}"
            )
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"{path_prefix}Bulk target RNA {observed:.1f} TPM — "
            f"**target absent** in this sample.{caution_suffix}"
        )
    if not tumor_band_available(expression_row):
        parts = [f"Bulk TPM {observed:.0f}", "tumor-inferred model interval unavailable"]
        if path_context:
            parts.append(path_context)
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"{_sentence(parts, maturity=maturity)}{caution_suffix}"
        )
    source = tumor_attribution_context(expression_row)
    normal = normal_expression_context(expression_row)
    if expr_independent:
        interpretation_parts = [
            expression_independent_interpretation(target_row),
            expression_independent_rna_context(expression_row),
        ]
    else:
        interpretation_parts = [source["label"], source["band"], normal["label"]]
    notes = list(source.get("notes") or []) + list(normal.get("details") or [])
    if notes and not expr_independent:
        interpretation_parts.append(notes[0])
    if path_context:
        interpretation_parts.append(path_context)
    interpretation = "; ".join(part for part in interpretation_parts if part)
    maturity_sentence = f" Clinical maturity: {maturity}." if maturity else ""
    return (
        f"- **{sym}** — {agent} ({phase}{indication_clause}). "
        f"{interpretation}.{maturity_sentence}{caution_suffix}"
    )


def _top_therapies(
    targets_df,
    ranges_df,
    limit=3,
    *,
    analysis=None,
    disease_state=None,
):
    """Pick the top therapies to show in the brief.

    Priority: approved agents with present targets first, ranked by
    tumor-attributed TPM; then trial phases. Non-measured / absent
    targets are skipped from the brief (they still show in the full
    landscape in ``actionable.md`` / ``targets.md``).
    """
    if targets_df is None or len(targets_df) == 0 or ranges_df is None:
        return []
    sym_to_row = {}
    for _, rrow in ranges_df.iterrows():
        sym_to_row[str(rrow["symbol"])] = rrow

    phase_priority = {
        "approved": 0, "phase_3": 1, "phase_2": 2, "phase_1": 3, "preclinical": 4,
    }

    scored = []
    for _, t in targets_df.iterrows():
        sym = str(t.get("symbol") or "")
        expr = sym_to_row.get(sym)
        expr_independent = expression_independent_indication(t)
        if expr is None:
            if expr_independent:
                phase = str(t.get("phase") or "")
                scored.append(
                    (
                        (
                            therapy_path_rank(
                                t,
                                analysis=analysis,
                                disease_state=disease_state,
                            ),
                            phase_priority.get(phase, 99),
                            1,
                            1,
                            0.0,
                            sym,
                        ),
                        t,
                        None,
                    )
                )
            continue
        observed = float(expr.get("observed_tpm") or 0.0)
        if observed < 1.0 and not expr_independent:
            # Target absent — the brief reports presence, not absence.
            # The full landscape in targets.md has the absence noted.
            continue
        attr_tumor = float(expr.get("attr_tumor_tpm") or 0.0)
        attr_fraction = float(expr.get("attr_tumor_fraction") or 1.0)
        lineage_material = same_lineage_material_target_candidate(
            expr,
            target_row=t,
        )
        # Drop rows that are mostly non-tumor from the top-3 — they
        # don't belong in the clinician handoff per #79 semantics.
        # Same-lineage clinical targets are a special case: a prostate
        # lineage marker assigned partly to matched-normal prostate is
        # source-ambiguous, not equivalent to an immune/stromal target.
        if attr_fraction < 0.30 and not expr_independent and not lineage_material:
            continue
        reliability_status = target_reliability_status(expr, target_row=t)
        if reliability_status == "unsupported":
            continue
        # Note (#128): we deliberately do NOT filter on
        # ``broadly_expressed`` here. The caller's ``targets_df`` is
        # the **curated** cancer-key-genes panel (#110) — every row
        # in it has been evaluated by hand as a clinician-relevant
        # target, often because the targeting mechanism is
        # amplification or lineage-retained overexpression rather
        # than baseline expression breadth (ERBB2 for HER2+ BRCA,
        # MDM2 for WD/DD-LPS, GPC3 for HCC). Trust curation. The
        # broadly-expressed flag is enforced in the generic Surface
        # / Intracellular target tables where ranking is by raw
        # expression, not curation.
        phase = str(t.get("phase") or "")
        reliability_rank = 0 if reliability_status == "supported" else 1
        expression_rank = 1 if expr_independent else 0
        sort_key = (
            therapy_path_rank(
                t,
                analysis=analysis,
                disease_state=disease_state,
            ),
            phase_priority.get(phase, 99),
            expression_rank,
            reliability_rank,
            -attr_tumor,
            sym,
        )
        scored.append((sort_key, t, expr))

    # Sort by key only — avoid pandas Series comparison in tie-break.
    scored.sort(key=lambda item: item[0])

    deduped = []
    seen_symbols = set()
    for sort_key, t, expr in scored:
        sym = str(t.get("symbol") or "")
        if sym in seen_symbols:
            continue
        seen_symbols.add(sym)
        deduped.append((t, expr))
        if len(deduped) >= limit:
            break
    return deduped


def _brief_float(value, default=0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _brief_truthy(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _format_trace_tpm(value) -> str:
    value = _brief_float(value, 0.0)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_component_label(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "—"
    text = text.replace("_", " ")
    if text.startswith("matched normal "):
        text = text.replace("matched normal ", "matched-normal ", 1)
    return text


def _top_non_tumor_attribution(expression_row) -> tuple[str, float]:
    label = _format_component_label(expression_row.get("attr_top_compartment"))
    value = _brief_float(expression_row.get("attr_top_compartment_tpm"), 0.0)
    if label != "—" and label.lower() != "tumor" and value > 0:
        return label, value

    attribution = expression_row.get("attribution")
    if isinstance(attribution, str):
        try:
            import ast

            attribution = ast.literal_eval(attribution)
        except Exception:
            attribution = None
    if isinstance(attribution, dict):
        candidates = []
        for comp, comp_value in attribution.items():
            comp_label = _format_component_label(comp)
            if comp_label == "—" or comp_label.lower() == "tumor":
                continue
            comp_tpm = _brief_float(comp_value, 0.0)
            if comp_tpm > 0:
                candidates.append((comp_label, comp_tpm))
        if candidates:
            return max(candidates, key=lambda item: item[1])
    return "—", 0.0


def _trace_phase_label(target_row) -> str:
    phase = str(target_row.get("phase") or "")
    return {
        "approved": "approved",
        "phase_3": "phase 3",
        "phase_2": "phase 2",
        "phase_1": "phase 1 exploratory",
        "preclinical": "preclinical",
    }.get(phase, phase.replace("_", " ") or "curated")


def _source_trace_reason(target_row, expression_row, *, in_shortlist: bool) -> str:
    source = tumor_attribution_context(expression_row)
    reliability = target_reliability_status(expression_row, target_row=target_row)
    phase = _trace_phase_label(target_row)
    attr_fraction = _brief_float(expression_row.get("attr_tumor_fraction"), 0.0)
    comp_label, _comp_tpm = _top_non_tumor_attribution(expression_row)
    lineage_material = same_lineage_material_target_candidate(
        expression_row,
        target_row=target_row,
    )

    parts = []
    if phase and phase != "approved":
        parts.append(phase)

    if in_shortlist:
        if source["tier"] == "tumor_supported":
            parts.append("clears source gate")
        elif lineage_material:
            parts.append("same-lineage marker, provisional source")
        else:
            parts.append(f"{source['label']}, clears source gate")
    elif lineage_material:
        parts.append("same-lineage marker, provisional source")
    elif _brief_truthy(expression_row.get("matched_normal_over_predicted")):
        if comp_label != "—":
            background = (
                "lineage background"
                if comp_label.lower().startswith("matched-normal")
                else "non-tumor background"
            )
            parts.append(f"{comp_label} over-predicts / {background}")
        else:
            parts.append("matched-normal over-predicts / lineage background")
    elif reliability == "unsupported" and attr_fraction < 0.30:
        if comp_label != "—":
            parts.append(f"{attr_fraction:.0%} tumor; mostly {comp_label}")
        else:
            parts.append(f"{attr_fraction:.0%} tumor fraction")
    elif reliability == "unsupported":
        parts.append(
            f"mostly {comp_label}/background" if comp_label != "—" else "background-dominant"
        )
    elif reliability == "provisional":
        parts.append(source["label"])
    else:
        parts.append("ranked below top list")

    deduped = []
    for part in parts:
        if part and part not in deduped:
            deduped.append(part)
    return "; ".join(deduped)


def _shortlist_omission_note(targets_df, ranges_df, top_rows) -> str:
    """Trace source attribution for non-clean clinical target decisions."""
    if targets_df is None or ranges_df is None or not top_rows:
        return ""
    top_symbols = {str(t.get("symbol") or "") for t, _ in top_rows}
    sym_to_row = {
        str(row.get("symbol") or ""): row
        for _, row in ranges_df.iterrows()
    }
    omitted = []
    seen = set()
    for _, target in targets_df.iterrows():
        sym = str(target.get("symbol") or "").strip()
        if not sym or sym.lower() == "nan" or sym in top_symbols or sym in seen:
            continue
        seen.add(sym)
        expr = sym_to_row.get(sym)
        if expr is None or _brief_float(expr.get("observed_tpm"), 0.0) < 1.0:
            continue
        attr_fraction = _brief_float(expr.get("attr_tumor_fraction"), 1.0)
        reliability_status = target_reliability_status(expr, target_row=target)
        phase = str(target.get("phase") or "")
        if reliability_status != "supported" or phase in {"phase_1", "preclinical"}:
            comp_label, comp_tpm = _top_non_tumor_attribution(expr)
            omitted.append(
                {
                    "symbol": sym,
                    "bulk": _brief_float(expr.get("observed_tpm"), 0.0),
                    "tumor": _brief_float(expr.get("attr_tumor_tpm"), 0.0),
                    "fraction": attr_fraction,
                    "component": comp_label,
                    "component_tpm": comp_tpm,
                    "reason": _source_trace_reason(target, expr, in_shortlist=False),
                }
            )
        if len(omitted) >= 4:
            break
    shortlist_context = []
    for target, expr in top_rows:
        if expr is None:
            continue
        phase = str(target.get("phase") or "")
        source = tumor_attribution_context(expr)
        if phase == "approved" and source["tier"] == "tumor_supported":
            continue
        if phase == "approved":
            continue
        comp_label, comp_tpm = _top_non_tumor_attribution(expr)
        shortlist_context.append(
            {
                "symbol": str(target.get("symbol") or ""),
                "bulk": _brief_float(expr.get("observed_tpm"), 0.0),
                "tumor": _brief_float(expr.get("attr_tumor_tpm"), 0.0),
                "fraction": _brief_float(expr.get("attr_tumor_fraction"), 0.0),
                "component": comp_label,
                "component_tpm": comp_tpm,
                "reason": _source_trace_reason(target, expr, in_shortlist=True),
            }
        )
        if len(shortlist_context) >= 2:
            break

    rows = shortlist_context + omitted
    if not rows:
        return ""
    lines = [
        "**Target expression source trace**",
        "| Gene | Bulk TPM | Tumor-inferred TPM | Tumor fraction | Top non-tumor attribution | Component TPM | Main reason |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        component = row["component"] if row["component"] != "—" else "none modeled"
        lines.append(
            f"| {row['symbol']} | {_format_trace_tpm(row['bulk'])} | "
            f"{_format_trace_tpm(row['tumor'])} | {row['fraction']:.0%} | "
            f"{component} | {_format_trace_tpm(row['component_tpm'])} | "
            f"{row['reason']} |"
        )
    lines.append(
        "*Source attribution is a caveat, not an automatic exclusion; "
        "clinical maturity and eligibility still set the shortlist order.*"
    )
    return "\n".join(lines)


def _panel_display_label(panel_code, panel_subtype=None):
    if panel_subtype:
        return f"{panel_code} ({str(panel_subtype).replace('_', ' ')})"
    return panel_code


def _curated_target_panel_for_sample(cancer_code, analysis, ranges_df=None):
    from .gene_sets_cancer import cancer_therapy_targets

    panel_code, panel_subtype = cancer_key_genes_lookup_for_analysis(
        cancer_code,
        analysis,
        ranges_df=ranges_df,
    )
    if panel_subtype:
        targets_df = cancer_therapy_targets(panel_code, subtype=panel_subtype)
    else:
        targets_df = cancer_therapy_targets(panel_code)
    return panel_code, panel_subtype, targets_df.reset_index(drop=True)


def _caveats_from_purity_tier(purity_tier, sample_context) -> List[str]:
    """User-facing caveat lines for the brief.

    Converts the ``purity_tier.reasons`` (which are short internal
    strings) into full sentences without internal jargon.
    """
    if purity_tier is None:
        return []
    reasons = getattr(purity_tier, "reasons", []) or []
    out = []
    for reason in reasons:
        r = str(reason)
        if "wide purity CI" in r:
            out.append(
                "Purity range is wide — target TPMs "
                "could be over- or under-stated depending on the true "
                "purity."
            )
        elif "low-purity regime" in r:
            out.append(
                "Sample is in a low-purity regime — raw target TPMs "
                "tend to overstate tumor presence. Prefer the tumor-"
                "attributed values."
            )
        elif "severe RNA degradation" in r:
            out.append(
                "RNA is severely degraded — long-transcript targets "
                "are systematically under-counted; interpret "
                "negative results cautiously."
            )
        elif "moderate RNA degradation" in r:
            out.append(
                "RNA is partially degraded — long-transcript targets "
                "are under-counted to a moderate degree."
            )
        elif "targeted-panel" in r:
            out.append(
                "Input appears to be a targeted panel rather than "
                "whole-transcriptome — relative expression estimates "
                "should be interpreted within the panel only."
            )
    # Library prep / preservation note from sample_context.
    if sample_context is not None:
        prep = getattr(sample_context, "library_prep", None)
        preservation = getattr(sample_context, "preservation", None)
        prep_label = library_prep_display_label(prep) if prep else None
        if prep_label and preservation == "ffpe":
            out.append(
                f"FFPE preservation with {prep_label} library prep — "
                "short transcripts favored; some classes of targets "
                "are artificially depressed."
            )
        elif prep == "exome_capture":
            out.append(
                "RNA hybrid-capture / RNA-exome prep — rRNA and many "
                "non-polyadenylated RNAs are under-sampled by design; "
                "low MT fraction is expected, but measurable MT mRNAs "
                "should not be treated as filtered out."
            )
    return out


def build_summary(
    analysis,
    ranges_df,
    cancer_code: str,
    disease_state: str,
    sample_id: Optional[str] = None,
) -> str:
    """Return the one-page ``*-summary.md`` content (≤ 40 lines).

    Audience: clinician skimming before a tumor board; LLM asked for a
    short referral-note paragraph. Strict structure; no internal
    jargon. (Named ``build_brief`` through 4.40; the legacy name is
    still exported as an alias.)
    """
    from .gene_sets_cancer import cancer_key_genes_cancer_types

    purity = analysis.get("purity") or {}
    purity_tier = analysis.get("purity_confidence")
    sample_context = analysis.get("sample_context")
    cancer_name = analysis.get("cancer_name") or cancer_code

    lines: List[str] = []
    sample_id = _display_sample_id(sample_id)
    header_id = f": {sample_id}" if sample_id else ""
    lines.append(f"# Summary{header_id}\n")
    lines.append(
        "<!-- Audience: clinician skimming before tumor board. "
        "Intended length: ≤ 40 lines. -->"
    )
    lines.append("")

    # #149: Step-0 healthy-vs-tumor banner. Above the cancer call so
    # the reader sees the caveat before anchoring on the TCGA label.
    # Banner decision reads downstream tumor evidence (purity from
    # Step 2, signature score from Step 1) so a confident cancer
    # call doesn't trigger a spurious Step-0 warning.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None:
        banner = hvt.brief_banner(
            purity=purity.get("overall_estimate") if purity else None,
            signature_score=_top_candidate_signature_score(analysis),
        )
        if banner:
            lines.append(banner)
            lines.append("")

    # Cancer call — annotated with #169 contested-call confidence when
    # orthogonal signals (lineage concordance, runner-up gap, Step-0
    # top-ρ cohort) disagree with the classifier's pick.
    from .confidence import compute_call_confidence
    call_tier = compute_call_confidence(analysis)
    suffix = _call_confidence_suffix(call_tier, concise=True)

    # #171: for mixture cohorts, surface the winning subtype hypothesis
    # so the reader sees "Cancer call: SARC (subtype: rhabdomyosarcoma
    # -consistent)" instead of just SARC. The subtype is the registry
    # code's ``subtype_key`` field (e.g. "leiomyosarcoma") or the code
    # itself when no human-readable key is populated.
    subtype_annotation = ""
    winning_subtype = None
    candidate_trace = analysis.get("candidate_trace") or []
    if candidate_trace:
        winning_subtype = candidate_trace[0].get("winning_subtype")

    # #198: before rendering, consult the degenerate-subtype registry.
    # Several within-family subtypes share a gene signature (OS vs DDLPS
    # both carry 12q13-15 amplicon; Ewing vs DSRCT vs ARMS all CD99+)
    # and need a site or fusion-surrogate tiebreaker. The resolver
    # reasons over the full sample context — decomposition site
    # template AND the complete tumor-attributed TPM dict — so that
    # decisions aren't made from single-gene lookups in isolation.
    # Activation-signature gating ensures pairs only fire when the
    # shared signature is actually present; high-confidence clear-
    # winner calls bypass the resolver entirely.
    degenerate_status = None
    degenerate_alternatives = []
    degenerate_resolution = None
    original_winning_subtype = winning_subtype
    if winning_subtype:
        try:
            from .degenerate_subtype import resolve_degenerate_subtype
            decomposition = analysis.get("decomposition") or {}
            site_template = decomposition.get("best_template")
            tumor_tpm_by_symbol = analysis.get("tumor_tpm_by_symbol")
            if not tumor_tpm_by_symbol and ranges_df is not None:
                # Build from ``ranges_df`` — the per-gene attribution
                # stage's output. Propagates the full tumor-attributed
                # TPM context so activation signatures and multi-gene
                # tiebreakers get the full evidence, not a single-gene
                # slice.
                try:
                    import pandas as pd
                    if isinstance(ranges_df, pd.DataFrame) and "symbol" in ranges_df.columns and "attr_tumor_tpm" in ranges_df.columns:
                        tumor_tpm_by_symbol = dict(
                            zip(
                                ranges_df["symbol"].astype(str),
                                pd.to_numeric(
                                    ranges_df["attr_tumor_tpm"],
                                    errors="coerce",
                                ).fillna(0.0).astype(float),
                            )
                        )
                except Exception:
                    logger.debug(
                        "degenerate-subtype: failed to build tumor_tpm_by_symbol from ranges_df",
                        exc_info=True,
                    )
                    tumor_tpm_by_symbol = None
            resolution = resolve_degenerate_subtype(
                winning_subtype,
                site_template=site_template,
                tumor_tpm_by_symbol=tumor_tpm_by_symbol,
            )
            degenerate_resolution = resolution
            if resolution["status"] == "corrected":
                winning_subtype = resolution["final_subtype"]
            degenerate_status = resolution["status"]
            degenerate_alternatives = resolution["alternatives"]
        except Exception:
            logger.debug(
                "degenerate-subtype resolution failed; keeping classifier pick",
                exc_info=True,
            )

    if winning_subtype:
        try:
            from .gene_sets_cancer import cancer_type_registry
            reg = cancer_type_registry()
            match = reg[reg["code"] == winning_subtype]
            if not match.empty:
                row = match.iloc[0]
                subtype_key = row.get("subtype_key")
                label = None
                if isinstance(subtype_key, str) and subtype_key and subtype_key.lower() != "nan":
                    label = subtype_key.replace("_", " ")
                else:
                    # Fall back to the human-readable registry ``name``
                    # so aggregate-only rows without a ``subtype_key``
                    # still render cleanly (SARC_LPS_UNSPEC →
                    # "liposarcoma" rather than raw code).
                    name = row.get("name")
                    if isinstance(name, str) and name:
                        label = name.split("(")[0].strip().lower()
                if not label:
                    label = winning_subtype
                if degenerate_status == "degenerate":
                    subtype_annotation = (
                        f" (subtype: degenerate — {label} vs "
                        f"{'/'.join(degenerate_alternatives)})"
                    )
                else:
                    subtype_annotation = f" (subtype: {label}-consistent)"
        except Exception:
            subtype_annotation = f" (subtype: {winning_subtype}-consistent)"

    call_punctuation = suffix or "."
    lines.append(
        f"**Cancer call:** {cancer_code} ({cancer_name})"
        f"{subtype_annotation}{call_punctuation}"
    )
    # Surface a subtype note only when the resolver changed the call or
    # flagged irreducible ambiguity. ``pair_inactive`` means the pair
    # didn't apply — no reader-facing note needed.
    if degenerate_status in ("corrected", "degenerate"):
        subtype_note = _render_subtype_note(
            degenerate_resolution or {},
            original_subtype=original_winning_subtype,
            site_template=(analysis.get("decomposition") or {}).get("best_template"),
        ).strip()
        if subtype_note:
            lines.append(f"**Subtype note:** {subtype_note}")

    # Purity
    overall = purity.get("overall_estimate")
    lower = purity.get("overall_lower")
    upper = purity.get("overall_upper")
    if overall is not None and lower is not None and upper is not None:
        tier_label = getattr(purity_tier, "tier", "unknown") if purity_tier else "unknown"
        lines.append(
            f"**Purity:** {overall:.0%} (model interval {lower:.0%}–{upper:.0%}, "
            f"{tier_label} confidence)."
        )

    # Disease state
    disease_state_display = report_disease_state_text(disease_state, analysis=analysis)
    if disease_state_display:
        lines.append(f"**Disease state:** {disease_state_display}")

    # Sample context
    if sample_context is not None:
        prep_label = library_prep_clause(getattr(sample_context, "library_prep", "unknown"))
        pres_label = str(getattr(sample_context, "preservation", "unknown")).replace("_", " ")
        lines.append(f"**Sample:** {prep_label}, {pres_label} preservation.")

    lines.append("")

    # Top therapies — subtype/direct-code resolved when the umbrella
    # cancer call narrows onto a more specific curated panel.
    panel_code, panel_subtype, targets_df = _curated_target_panel_for_sample(
        cancer_code, analysis, ranges_df=ranges_df,
    )
    if panel_code in cancer_key_genes_cancer_types():
        top = _top_therapies(
            targets_df,
            ranges_df,
            limit=3,
            analysis=analysis,
            disease_state=disease_state_display,
        )
        lines.append("## Top candidate therapies\n")
        lines.append(
            "*Ranked by treatment-path maturity first, then tumor-source support; "
            "verify current therapy before acting on any row.*\n"
        )
        if panel_code != cancer_code or panel_subtype:
            lines.append(
                "*Subtype-resolved therapy curation:* "
                + subtype_curation_scope_note(
                    panel_code,
                    panel_subtype=panel_subtype,
                    base_code=cancer_code,
                    base_name=analysis.get("cancer_name") or cancer_code,
                    noun="therapy evidence",
                )
                + "\n"
            )
        if top:
            for target_row, expression_row in top:
                lines.append(
                    _format_therapy_bullet(
                        target_row,
                        expression_row,
                        target_panel=targets_df,
                        analysis=analysis,
                        disease_state=disease_state_display,
                    )
                )
            omission_note = _shortlist_omission_note(targets_df, ranges_df, top)
            if omission_note:
                lines.append(omission_note)
            lines.append("")
        else:
            lines.append(
                "*No approved or trialed agents with a measured, "
                "tumor-supported target in this sample.*\n"
            )
    else:
        lines.append(
            f"## Top candidate therapies\n"
            f"*Cancer type {cancer_code} is not yet in the curated "
            "key-genes panel — see the full tables below for a raw "
            "expression ranking.*\n"
        )

    # Caveats
    caveats = _caveats_from_purity_tier(purity_tier, sample_context)
    if caveats:
        lines.append("## Caveats")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append(
        "*Full detail: see the accompanying `*-analysis.md` and `*-evidence.md`.*"
    )

    return "\n".join(lines)


# Back-compat alias — ``build_brief`` was the public name through 4.40;
# removed in 5.0. External importers should migrate to ``build_summary``.
build_brief = build_summary


def build_actionable(
    analysis,
    ranges_df,
    cancer_code: str,
    disease_state: str,
    sample_id: Optional[str] = None,
) -> str:
    """Return the longer ``*-actionable.md`` content (~2-3 pages).

    Audience: oncologist preparing a treatment discussion, or LLM
    asked to draft a clinical summary. Full structure with the
    biomarker panel and therapy landscape inline; no pipeline-
    internal jargon.
    """
    from .gene_sets_cancer import cancer_key_genes_cancer_types

    purity = analysis.get("purity") or {}
    purity_tier = analysis.get("purity_confidence")
    sample_context = analysis.get("sample_context")
    cancer_name = analysis.get("cancer_name") or cancer_code

    lines: List[str] = []
    sample_id = _display_sample_id(sample_id)
    header_id = f" — {sample_id}" if sample_id else ""
    lines.append(f"# Actionable review{header_id}\n")
    lines.append(
        "<!-- Audience: oncologist preparing a treatment discussion; "
        "molecular tumor board member reading carefully. -->"
    )
    lines.append("")

    # Sample + confidence paragraph
    lines.append("## Sample and confidence\n")
    prep_label = (
        library_prep_clause(getattr(sample_context, "library_prep", "unknown"))
        if sample_context else "unknown"
    )
    pres_label = str(
        getattr(sample_context, "preservation", "unknown")
    ).replace("_", " ") if sample_context else "unknown"
    if sample_context:
        lines.append(
            f"Input: **{prep_label}**, **{pres_label}** "
            "preservation. "
            + _preservation_clinical_clause(sample_context)
        )

    overall = purity.get("overall_estimate")
    lower = purity.get("overall_lower")
    upper = purity.get("overall_upper")
    tier_label = getattr(purity_tier, "tier", "unknown") if purity_tier else "unknown"
    tier_reasons = getattr(purity_tier, "reasons", []) if purity_tier else []
    if overall is not None:
        confidence_clause = f"**{tier_label}** confidence"
        if tier_reasons and tier_label in {"low", "moderate"}:
            confidence_clause += " (" + "; ".join(tier_reasons) + ")"
        lines.append(
            f"\nPurity point estimate: **{overall:.0%}** "
            f"(model interval {lower:.0%}–{upper:.0%}). {confidence_clause.capitalize()}."
        )

    lines.append("")

    # Cancer call + disease state
    lines.append("## Cancer call and disease state\n")
    from .confidence import compute_call_confidence
    call_tier = compute_call_confidence(analysis)
    call_suffix = _call_confidence_suffix(call_tier, concise=True)
    call_punctuation = call_suffix or "."
    lines.append(
        f"Working call: **{cancer_code}** ({cancer_name}){call_punctuation}"
    )
    # Step-0 tissue-composition banner (if non-tumor-consistent) so
    # an actionable reader sees the Step-0 caveat attached to the
    # working call, not buried in the summary. Same evidence-gated
    # logic as the brief — strong tumor signal suppresses the banner.
    hvt = analysis.get("healthy_vs_tumor")
    if hvt is not None:
        banner = hvt.brief_banner(
            purity=purity.get("overall_estimate") if purity else None,
            signature_score=_top_candidate_signature_score(analysis),
        )
        if banner:
            lines.append(f"\n{banner}")
    disease_state_display = report_disease_state_text(disease_state, analysis=analysis)
    if disease_state_display:
        lines.append(f"\n{disease_state_display}")
    lines.append("")

    # Therapy landscape
    panel_code, panel_subtype, targets_df = _curated_target_panel_for_sample(
        cancer_code, analysis, ranges_df=ranges_df,
    )
    panel_label = _panel_display_label(panel_code, panel_subtype)
    if panel_code in cancer_key_genes_cancer_types():
        sym_to_row = {}
        for _, rrow in ranges_df.iterrows():
            sym_to_row[str(rrow["symbol"])] = rrow

        if len(targets_df):
            lines.append("## Therapy landscape\n")
            if panel_code != cancer_code or panel_subtype:
                lines.append(
                    "*Subtype-resolved therapy curation:* "
                    + subtype_curation_scope_note(
                        panel_code,
                        panel_subtype=panel_subtype,
                        base_code=cancer_code,
                        base_name=cancer_name or cancer_code,
                        noun="therapy evidence",
                    )
                    + "\n"
                )
            lines.append(
                "Agents with an approved or trialed indication for "
                f"{cancer_code_display_name(panel_code, panel_label)}, cross-referenced to this sample. "
                "Approved agents listed first. Interpretation separates "
                "tumor-source support from normal-expression context so "
                "lineage markers are not confused with tumor-exclusive "
                "targets. Treatment-path context flags standard options, "
                "later-line requirements, trial follow-ups, and possible "
                "current/prior therapy exposure."
            )
            lines.append(tpm_semantics_note())
            lines.append("")
            lines.append(
                "| Target | Agent | Class | Phase | Indication | "
                "Bulk TPM (measured) | Tumor-inferred TPM (model) | Interpretation |"
            )
            lines.append(
                "|--------|-------|-------|-------|------------|"
                "----------|------------|----------------|"
            )
            phase_order = {
                "approved": 0, "phase_3": 1, "phase_2": 2,
                "phase_1": 3, "preclinical": 4,
            }
            sorted_df = targets_df.assign(
                _path_key=[
                    therapy_path_rank(
                        t,
                        analysis=analysis,
                        disease_state=disease_state_display,
                    )
                    for _, t in targets_df.iterrows()
                ],
                _po=targets_df["phase"].map(
                    lambda p: phase_order.get(str(p), 99)
                )
            ).sort_values(["_path_key", "_po", "symbol", "agent"])
            def _cell(value):
                """Render a cell, turning NaN / blank / 'nan' into em-dash."""
                if value is None:
                    return "—"
                s = str(value).strip()
                if s == "" or s.lower() == "nan":
                    return "—"
                return s

            for _, t in sorted_df.iterrows():
                raw_sym = t.get("symbol")
                sym = _cell(raw_sym)
                # Agent-only rows (no gene target — e.g. doxorubicin, pazopanib,
                # trabectedin for sarcoma) have a blank ``symbol``; sym_to_row
                # keying by "nan" would always miss, so skip the lookup and
                # mark as not measurable rather than reporting the TPM of a
                # nonexistent gene.
                if sym == "—":
                    obs_cell = "*not measured*"
                    tumor_cell = "—"
                    interp_cell = "agent-only / no direct gene target"
                else:
                    expr = sym_to_row.get(sym)
                    if expr is None:
                        obs_cell = "*not measured*"
                        tumor_cell = "—"
                        if expression_independent_indication(t):
                            interp_cell = (
                                expression_independent_interpretation(t)
                                + "; "
                                + expression_independent_rna_context(None)
                            )
                        else:
                            interp_cell = "not measured"
                        path_context = therapy_path_context(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        state_caution = therapy_state_caution(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        extra_parts = []
                        if path_context:
                            extra_parts.append(path_context)
                        if state_caution:
                            extra_parts.append(
                                f"current-therapy check: {state_caution}"
                            )
                        if extra_parts:
                            interp_cell += "; " + "; ".join(extra_parts)
                    else:
                        obs_cell = f"{float(expr.get('observed_tpm') or 0):.1f}"
                        tumor_cell = tumor_band_cell(expr)
                        source = tumor_attribution_context(expr)
                        normal = normal_expression_context(expr)
                        expr_independent = expression_independent_indication(t)
                        if expr_independent:
                            interp_parts = [
                                expression_independent_interpretation(t),
                                expression_independent_rna_context(expr),
                            ]
                        else:
                            interp_parts = [source["label"], normal["label"]]
                        notes = (
                            list(source.get("notes") or [])
                            + list(normal.get("details") or [])
                        )
                        if notes and not expr_independent:
                            interp_parts.append(notes[0])
                        path_context = therapy_path_context(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        if path_context:
                            interp_parts.append(path_context)
                        state_caution = therapy_state_caution(
                            t,
                            analysis=analysis,
                            disease_state=disease_state_display,
                        )
                        if state_caution:
                            interp_parts.append(
                                f"current-therapy check: {state_caution}"
                            )
                        interp_parts.append(
                            clinical_maturity_summary(t, target_panel=targets_df)
                        )
                        interp_cell = "; ".join(
                            part for part in interp_parts if part
                        )
                phase = _phase_label(str(t.get("phase") or ""))
                bold = "**" if phase == "Approved" and sym != "—" else ""
                lines.append(
                    f"| {bold}{sym}{bold} | {_cell(t.get('agent'))} | "
                    f"{_cell(t.get('agent_class'))} | {phase} | "
                    f"{_cell(t.get('indication'))} | {obs_cell} | {tumor_cell} | {interp_cell} |"
                )
            lines.append("")
        else:
            lines.append(
                "## Therapy landscape\n"
                "*No curated therapy targets are available for this resolved panel.*\n"
            )
    else:
        lines.append(
            "## Therapy landscape\n"
            f"*Cancer type {cancer_code} is not yet in the curated "
            "key-genes panel — see `evidence.md` for the generic "
            "expression-ranked tables.*\n"
        )

    # Caveats
    caveats = _caveats_from_purity_tier(purity_tier, sample_context)
    if caveats:
        lines.append("## Caveats\n")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append(
        "*See also: `*-analysis.md` (full integrated interpretation) "
        "and `*-evidence.md` (stepwise deduction chain + full target tables).*"
    )
    return "\n".join(lines)


def _preservation_clinical_clause(sample_context) -> str:
    """One-sentence clinician-facing framing of preservation."""
    prep = getattr(sample_context, "library_prep", None)
    preservation = getattr(sample_context, "preservation", None)
    severity = getattr(sample_context, "degradation_severity", "none")
    if preservation == "ffpe" and severity in ("moderate", "severe"):
        return (
            f"FFPE with {severity} degradation — long-transcript "
            "quantification is biased; negative results for long genes "
            "should be interpreted cautiously."
        )
    if prep == "exome_capture":
        return (
            "RNA hybrid-capture / RNA-exome prep selectively enriches "
            "targeted transcripts; rRNA and many non-polyadenylated RNAs "
            "are under-sampled, while low-level MT mRNAs may still be real."
        )
    return ""
