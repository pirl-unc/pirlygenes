# Licensed under the Apache License, Version 2.0

"""Confidence tiers for pirlygenes analyses (#109).

A single source-of-truth for "how much should the reader trust this
number" that every renderer consumes. The goal is structural: no
quantity shows up in any report identically regardless of how shaky
its underlying evidence is.

The tier system has three levels:

- ``"high"``: no caveats. CI tight, decomposition stable, no artefact
  flags. Renderer shows the number bare.
- ``"moderate"``: one signal is weaker than typical but the number is
  still usable. Renderer shows the number with a short inline note.
- ``"low"``: at least one core input is unreliable. Renderer shows the
  number with a prominent low-confidence tag and, for summary-level
  lists, excludes the row or requires the inline caveat.

Two computed tiers:

- ``compute_purity_confidence``: sample-level — purity CI width, point
  estimate, degradation severity, prep-specific caveats.
- ``compute_target_confidence``: per-target — rolls in the purity
  tier, per-gene attribution (#108), TME flags, purity amplification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ConfidenceTier:
    tier: str  # "high" | "moderate" | "low" | "degenerate" | "unknown"
    reasons: List[str] = field(default_factory=list)

    @property
    def badge(self) -> str:
        return {
            "low": "⚠⚠",
            "moderate": "⚠",
            "high": "",
            "degenerate": "—",
            "unknown": "?",
        }.get(self.tier, "")

    @property
    def inline_note(self) -> str:
        if not self.reasons:
            return ""
        return "; ".join(self.reasons)

    def render(self) -> str:
        if self.tier == "high":
            return ""
        note = self.inline_note
        prefix = f"**{self.tier} confidence**"
        return f"{prefix} ({note})" if note else prefix


def compute_purity_confidence(
    purity,
    sample_context=None,
    degradation_severity: str = "none",
) -> ConfidenceTier:
    """Tier the overall purity estimate.

    ``purity`` is the dict produced by ``estimate_tumor_purity`` /
    ``analyze_sample`` (keys ``overall_estimate``, ``overall_lower``,
    ``overall_upper``).
    """
    reasons: List[str] = []
    try:
        overall = float(purity.get("overall_estimate") or 0.0)
        lower = float(purity.get("overall_lower") or 0.0)
        upper = float(purity.get("overall_upper") or 0.0)
    except (TypeError, AttributeError, ValueError):
        return ConfidenceTier(tier="unknown", reasons=["no purity estimate available"])

    tier = "high"
    span = upper - lower
    if span <= 1e-9:
        # Zero-width CI means the estimator saw no per-gene variation
        # (synthetic / cohort-median / deterministic input). Surfacing
        # that as "high confidence" is misleading — the estimator
        # couldn't produce uncertainty, not because the answer is
        # certain but because the input has no spread to bound it.
        # #161: tier this as ``degenerate`` so renderers show a
        # specific "deterministic input — CI not estimated" message.
        return ConfidenceTier(
            tier="degenerate",
            reasons=[
                "deterministic input (no per-gene variation) — purity "
                "CI not estimated"
            ],
        )
    if span >= 0.35:
        tier = "low"
        reasons.append(f"wide purity CI ({lower:.0%}–{upper:.0%})")
    elif span >= 0.15:
        tier = "moderate"
        reasons.append(f"moderate purity CI span ({span * 100:.0f} pp)")

    if overall < 0.15:
        # Low-purity regime: even a tight CI is hard to act on because
        # any non-tumor signal dominates after dividing by purity.
        if tier != "low":
            tier = "moderate" if tier == "high" else tier
        reasons.append(f"low-purity regime ({overall:.0%})")
        if overall < 0.08:
            tier = "low"

    sev = (degradation_severity or "none").lower()
    if sev in ("moderate", "severe"):
        # Severe RNA degradation biases long-transcript quantification;
        # target TPMs downstream are less trustworthy.
        if sev == "severe":
            tier = "low"
        elif tier == "high":
            tier = "moderate"
        reasons.append(f"{sev} RNA degradation")

    if sample_context is not None:
        flags = getattr(sample_context, "flags", None) or []
        for flag in flags:
            # targeted-panel inputs are handled elsewhere but worth
            # flagging at the confidence layer so the tier stays honest.
            if "targeted_panel" in str(flag).lower():
                tier = "low"
                reasons.append("likely targeted-panel input")
                break

    return ConfidenceTier(tier=tier, reasons=reasons)


def compute_target_confidence(
    row,
    purity_tier: ConfidenceTier,
    sample_context=None,
) -> ConfidenceTier:
    """Tier a single therapy-target row.

    ``row`` is a dict- or Series-like with at least ``observed_tpm``,
    optionally ``tme_dominant``, ``tme_explainable``,
    ``attr_tumor_fraction``, and ``attribution`` (the new #108 columns).
    """
    reasons: List[str] = []
    tier = purity_tier.tier if purity_tier.tier in {"low", "moderate", "high"} else "moderate"
    if purity_tier.tier in {"low", "moderate"} and purity_tier.reasons:
        # Fold in the sample-level reasons so target rows carry them
        # when they're the dominant limit on confidence.
        reasons.extend(purity_tier.reasons)

    def _get(key, default=None):
        try:
            value = row.get(key) if hasattr(row, "get") else row[key]
        except (KeyError, AttributeError):
            return default
        return value if value is not None else default

    if _get("tme_dominant"):
        tier = "low"
        attribution = _get("attribution")
        if isinstance(attribution, dict) and attribution:
            top = max(attribution, key=lambda k: attribution[k])
            reasons.append(
                f"TME-dominant (tumor < 30% of observed; "
                f"top non-tumor compartment: {top.replace('_', ' ')})"
            )
        else:
            reasons.append("TME-dominant (≥70% of signal is non-tumor)")
    elif _get("tme_explainable"):
        if tier == "high":
            tier = "moderate"
        reasons.append("could be explained by a single healthy tissue's expression")

    # #131: matched-normal over-prediction. When the fitted
    # matched-normal (or any) compartment predicts more of the gene
    # than the sample actually contains, the attribution math's
    # tumor residual is zero by construction — but that's a ceiling
    # effect of the model, not evidence the tumor isn't expressing
    # the gene. Fire a moderate downgrade with a reader-facing reason
    # so clinically curated targets (KLK3 / TACSTD2 / FOLH1 on CRPC
    # samples) aren't silently dismissed.
    if _get("matched_normal_over_predicted"):
        if tier == "high":
            tier = "moderate"
        reasons.append(
            "matched-normal reference over-predicts this gene — "
            "tumor attribution hit the zero floor; the raw observed "
            "TPM is the better read than the attributed fraction"
        )

    # #128: broadly-expressed flag. A gene expressed across many
    # non-reproductive HPA tissues cannot be claimed as tumor-cell-
    # specific even when the residual attribution puts most of the
    # observed TPM into the tumor compartment, because the sample's
    # healthy cells alone carry a baseline broader than any single
    # compartment in the fitted decomposition.
    #
    # Amplification overrides: HER2 in HER2+ BRCA, MDM2 in WD/DD-LPS,
    # GPC3 in HCC — broadly expressed per HPA but observed well above
    # peak-healthy in the sample, which IS a tumor-specificity signal
    # regardless of breadth. The ``broadly_expressed`` column in
    # ranges_df is already gated on ``not amplified_over_healthy`` so
    # amplification-driven targets don't even trigger this branch.
    n_tissues = _get("n_healthy_tissues_expressed")
    amp_fold = _get("amplification_fold")
    if _get("broadly_expressed"):
        tier = "low"
        amp_note = ""
        if isinstance(amp_fold, (int, float)) and float(amp_fold) < 5.0:
            amp_note = f", peak-healthy fold only {float(amp_fold):.1f}×"
        if isinstance(n_tissues, (int, float)):
            reasons.append(
                f"broadly expressed across {int(n_tissues)} healthy "
                f"tissues{amp_note} — not tumor-cell-specific"
            )
        else:
            reasons.append(
                "broadly expressed across many healthy tissues — "
                "not tumor-cell-specific"
            )

    attr_fraction = _get("attr_tumor_fraction")
    if isinstance(attr_fraction, (int, float)) and 0.3 <= float(attr_fraction) < 0.5:
        if tier == "high":
            tier = "moderate"
        reasons.append(f"only {float(attr_fraction):.0%} of signal attributed to tumor core")

    # Deduplicate reasons while preserving order.
    seen = set()
    deduped: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    return ConfidenceTier(tier=tier, reasons=deduped)
