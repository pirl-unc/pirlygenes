# Licensed under the Apache License, Version 2.0

"""Two-tier markdown handoff documents (#111).

Audience distinction:

- ``*-brief.md`` — one-page summary, ≤ 40 lines. For a clinician
  skimming before a tumor board, or an LLM asked for a 3-sentence
  referral-note paragraph. Strict structure; no internal jargon.
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

from typing import List, Optional


def _phase_label(phase: str) -> str:
    return {
        "approved": "Approved",
        "phase_3": "Phase 3",
        "phase_2": "Phase 2",
        "phase_1": "Phase 1",
        "preclinical": "Preclinical",
    }.get(phase, phase)


def _format_therapy_bullet(target_row, expression_row) -> str:
    """One standardized therapy bullet for the brief."""
    sym = str(target_row.get("symbol") or "")
    agent = str(target_row.get("agent") or "—")
    phase = _phase_label(str(target_row.get("phase") or ""))
    indication = str(target_row.get("indication") or "")
    indication_clause = f", {indication}" if indication else ""
    if expression_row is None:
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"Target **not measured** in this sample."
        )
    observed = float(expression_row.get("observed_tpm") or 0.0)
    attr_tumor = float(expression_row.get("attr_tumor_tpm") or 0.0)
    has_attribution = bool(expression_row.get("attribution"))
    attr_fraction = float(expression_row.get("attr_tumor_fraction") or 0.0)
    if observed < 1.0:
        return (
            f"- **{sym}** — {agent} ({phase}{indication_clause}). "
            f"Observed {observed:.1f} TPM — **target absent** in this sample."
        )
    if has_attribution:
        tumor_clause = f"tumor-attributed {attr_tumor:.0f} TPM"
    else:
        tumor_clause = f"observed {observed:.0f} TPM"
    if attr_fraction >= 0.5:
        conf = "high"
    elif attr_fraction >= 0.3 or not has_attribution:
        conf = "moderate"
    else:
        conf = "low (mostly non-tumor)"
    return (
        f"- **{sym}** — {agent} ({phase}{indication_clause}). "
        f"{tumor_clause.capitalize()}. Confidence: {conf}."
    )


def _top_therapies(
    targets_df,
    ranges_df,
    limit=3,
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
        if expr is None:
            continue
        observed = float(expr.get("observed_tpm") or 0.0)
        if observed < 1.0:
            # Target absent — the brief reports presence, not absence.
            # The full landscape in targets.md has the absence noted.
            continue
        attr_tumor = float(expr.get("attr_tumor_tpm") or 0.0)
        attr_fraction = float(expr.get("attr_tumor_fraction") or 1.0)
        # Drop rows that are mostly non-tumor from the top-3 — they
        # don't belong in the clinician handoff per #79 semantics.
        if attr_fraction < 0.30:
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
        sort_key = (phase_priority.get(phase, 99), -attr_tumor, sym)
        scored.append((sort_key, t, expr))

    # Sort by key only — avoid pandas Series comparison in tie-break.
    scored.sort(key=lambda item: item[0])

    deduped = []
    seen_symbols = set()
    for sort_key, t, expr in scored:
        sym = sort_key[2]
        if sym in seen_symbols:
            continue
        seen_symbols.add(sym)
        deduped.append((t, expr))
        if len(deduped) >= limit:
            break
    return deduped


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
                "Purity confidence interval is wide — target TPMs "
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
        prep_label = {
            "exome_capture": "exome capture",
            "poly_a": "poly-A capture",
            "ribo_depleted": "ribosomal depletion",
            "total_rna": "total RNA",
        }.get(prep, None)
        if prep_label and preservation == "ffpe":
            out.append(
                f"FFPE preservation with {prep_label} library prep — "
                "short transcripts favored; some classes of targets "
                "are artificially depressed."
            )
        elif prep_label == "exome capture":
            out.append(
                "Exome-capture library prep — mitochondrial and "
                "non-coding RNAs are absent by design; don't interpret "
                "their absence as sample quality issues."
            )
    return out


def build_brief(
    analysis,
    ranges_df,
    cancer_code: str,
    disease_state: str,
    sample_id: Optional[str] = None,
) -> str:
    """Return the one-page ``*-brief.md`` content (≤ 40 lines).

    Audience: clinician skimming before a tumor board; LLM asked for a
    short referral-note paragraph. Strict structure; no internal
    jargon.
    """
    from .gene_sets_cancer import (
        cancer_therapy_targets,
        cancer_key_genes_cancer_types,
    )

    purity = analysis.get("purity") or {}
    purity_tier = analysis.get("purity_confidence")
    sample_context = analysis.get("sample_context")
    cancer_name = analysis.get("cancer_name") or cancer_code

    lines: List[str] = []
    header_id = f": {sample_id}" if sample_id else ""
    lines.append(f"# Brief{header_id}\n")
    lines.append(
        "<!-- Audience: clinician skimming before tumor board. "
        "Intended length: ≤ 40 lines. -->"
    )
    lines.append("")

    # Cancer call
    lines.append(f"**Cancer call:** {cancer_code} ({cancer_name}).")

    # Purity
    overall = purity.get("overall_estimate")
    lower = purity.get("overall_lower")
    upper = purity.get("overall_upper")
    if overall is not None and lower is not None and upper is not None:
        tier_label = getattr(purity_tier, "tier", "unknown") if purity_tier else "unknown"
        lines.append(
            f"**Purity:** {overall:.0%} (CI {lower:.0%}–{upper:.0%}, "
            f"{tier_label} confidence)."
        )

    # Disease state
    if disease_state:
        lines.append(f"**Disease state:** {disease_state}")

    # Sample context
    if sample_context is not None:
        prep_label = str(getattr(sample_context, "library_prep", "unknown")).replace("_", " ")
        pres_label = str(getattr(sample_context, "preservation", "unknown")).replace("_", " ")
        lines.append(f"**Sample:** {prep_label} library, {pres_label} preservation.")

    lines.append("")

    # Top therapies — only if the cancer type is curated.
    if cancer_code in cancer_key_genes_cancer_types():
        targets_df = cancer_therapy_targets(cancer_code)
        top = _top_therapies(targets_df, ranges_df, limit=3)
        if top:
            lines.append("## Top candidate therapies\n")
            for target_row, expression_row in top:
                lines.append(_format_therapy_bullet(target_row, expression_row))
            lines.append("")
        else:
            lines.append(
                "## Top candidate therapies\n"
                "*No approved or trialed agents with a measured, "
                "tumor-attributed target in this sample.*\n"
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
        "*Full detail: see the accompanying `*-actionable.md`, "
        "`*-summary.md`, `*-analysis.md`, and `*-targets.md`.*"
    )

    return "\n".join(lines)


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
    from .gene_sets_cancer import (
        cancer_biomarker_genes,
        cancer_therapy_targets,
        cancer_key_genes_cancer_types,
    )

    purity = analysis.get("purity") or {}
    purity_tier = analysis.get("purity_confidence")
    sample_context = analysis.get("sample_context")
    cancer_name = analysis.get("cancer_name") or cancer_code

    lines: List[str] = []
    header_id = f" — {sample_id}" if sample_id else ""
    lines.append(f"# Actionable review{header_id}\n")
    lines.append(
        "<!-- Audience: oncologist preparing a treatment discussion; "
        "molecular tumor board member reading carefully. -->"
    )
    lines.append("")

    # Sample + confidence paragraph
    lines.append("## Sample and confidence\n")
    prep_label = str(
        getattr(sample_context, "library_prep", "unknown")
    ).replace("_", " ") if sample_context else "unknown"
    pres_label = str(
        getattr(sample_context, "preservation", "unknown")
    ).replace("_", " ") if sample_context else "unknown"
    if sample_context:
        lines.append(
            f"Input: **{prep_label}** library prep, **{pres_label}** "
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
            f"(CI {lower:.0%}–{upper:.0%}). {confidence_clause.capitalize()}."
        )

    lines.append("")

    # Cancer call + disease state
    lines.append("## Cancer call and disease state\n")
    lines.append(f"Working call: **{cancer_code}** ({cancer_name}).")
    if disease_state:
        lines.append(f"\n{disease_state}")
    lines.append("")

    # Therapy landscape
    if cancer_code in cancer_key_genes_cancer_types():
        targets_df = cancer_therapy_targets(cancer_code)
        sym_to_row = {}
        for _, rrow in ranges_df.iterrows():
            sym_to_row[str(rrow["symbol"])] = rrow

        if len(targets_df):
            lines.append("## Therapy landscape\n")
            lines.append(
                "Agents with an approved or trialed indication for "
                f"{cancer_code}, cross-referenced to this sample. "
                "Approved agents listed first."
            )
            lines.append("")
            lines.append(
                "| Target | Agent | Class | Phase | Indication | "
                "Observed | Tumor-attr. |"
            )
            lines.append(
                "|--------|-------|-------|-------|------------|"
                "----------|-------------|"
            )
            phase_order = {
                "approved": 0, "phase_3": 1, "phase_2": 2,
                "phase_1": 3, "preclinical": 4,
            }
            sorted_df = targets_df.assign(
                _po=targets_df["phase"].map(
                    lambda p: phase_order.get(str(p), 99)
                )
            ).sort_values(["_po", "symbol", "agent"])
            for _, t in sorted_df.iterrows():
                sym = str(t["symbol"])
                expr = sym_to_row.get(sym)
                if expr is None:
                    obs_cell = "*not measured*"
                    tumor_cell = "—"
                else:
                    obs_cell = f"{float(expr.get('observed_tpm') or 0):.1f}"
                    attr_tumor = float(expr.get("attr_tumor_tpm") or 0)
                    tumor_cell = (
                        f"{attr_tumor:.0f}" if expr.get("attribution") else "—"
                    )
                phase = _phase_label(str(t.get("phase") or ""))
                bold = "**" if phase == "Approved" else ""
                lines.append(
                    f"| {bold}{sym}{bold} | {t.get('agent', '—')} | "
                    f"{t.get('agent_class', '—')} | {phase} | "
                    f"{t.get('indication', '—')} | {obs_cell} | {tumor_cell} |"
                )
            lines.append("")

        # Biomarker panel
        biomarker_syms = cancer_biomarker_genes(cancer_code)
        if biomarker_syms:
            lines.append("## Biomarker panel\n")
            lines.append(
                "Lineage, diagnostic, and disease-state biomarkers for "
                f"{cancer_code}. Status in this sample:"
            )
            lines.append("")
            lines.append("| Gene | Observed TPM | Tumor-attributed |")
            lines.append("|------|--------------|------------------|")
            for sym in biomarker_syms:
                row = sym_to_row.get(sym)
                if row is None:
                    lines.append(f"| {sym} | *not measured* | — |")
                    continue
                obs = float(row.get("observed_tpm") or 0)
                attr = float(row.get("attr_tumor_tpm") or 0)
                tumor_cell = f"{attr:.0f}" if row.get("attribution") else "—"
                lines.append(f"| {sym} | {obs:.1f} | {tumor_cell} |")
            lines.append("")

    # Caveats
    caveats = _caveats_from_purity_tier(purity_tier, sample_context)
    if caveats:
        lines.append("## Caveats\n")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append(
        "*See also: `*-summary.md` (free-form narrative), "
        "`*-analysis.md` (full pipeline detail), `*-targets.md` "
        "(complete target list), `*-provenance.md` (sample-content chain).*"
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
            "Exome-capture prep selectively targets coding exons; "
            "mitochondrial and non-polyadenylated transcripts are "
            "absent by design."
        )
    return ""
