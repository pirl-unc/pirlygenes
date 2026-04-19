# Licensed under the Apache License, Version 2.0
"""Stage-by-stage reasoning framework.

Each pipeline stage emits a typed dataclass. An ``AnalysisState``
composes all the stage outputs so far, and reasoning rules are
standalone typed functions that take the state (plus the fields
they specifically need) and return a ``RuleOutcome`` or ``None``.

Design intent:

- Rules are testable in isolation — no global state, no hidden
  dependencies on upstream analysis dicts.
- Rules are ordered as a list (data, not code) so ordering changes
  are configurational, not edits-to-decision-logic.
- Stages accumulate: Stage-1 analysis reads Stage-0 signals + fresh
  Stage-1 fields; Stage-2 reads Stage-0 + Stage-1 + Stage-2.
- Every stage has the same ``cancer_hint / reasoning_trace`` contract
  so downstream code that only needs the hint doesn't care which
  stage produced the call.

For now this module implements the Stage-0 rule set; Stage-1 through
Stage-6 are planned follow-ups. The framework is ready to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .healthy_vs_tumor import TissueCompositionSignal


# Thresholds — duplicated from healthy_vs_tumor so the rules can be
# inspected / tested without importing the legacy constants. Keep in
# sync when the original module's thresholds change.
_PROLIFERATION_HIGH_LOG2 = 4.5
_PROLIFERATION_QUIET_LOG2 = 3.5
_HPA_MARGIN_STRONG = 0.05
_HPA_MARGIN_WEAK = 0.02
_CTA_STRONG_COUNT = 4
_CTA_STRONG_SUM_TPM = 30.0
_ONCOFETAL_STRONG_COUNT = 2
_TUMOR_UP_PANEL_STRONG_COUNT = 2


@dataclass
class RuleOutcome:
    """Standard outcome of a reasoning rule.

    ``hint`` is the cancer-call it asserts. ``rule_name`` is a short
    ID suitable for logging ("R1a-strong-tumor-evidence"). ``structural``
    marks structural-ambiguity outcomes (lymphoid / mesenchymal) that
    must not be suppressed by downstream tumor-evidence checks.
    ``rationale`` is a human-readable short string listing the
    evidence that fired the rule.
    """

    hint: str
    rule_name: str
    structural: bool = False
    rationale: str = ""


# Type alias for Stage-0 rules. Each rule is a pure function of the
# :class:`TissueCompositionSignal` returning a ``RuleOutcome`` (fires)
# or ``None`` (defers to the next rule).
Stage0Rule = Callable[["TissueCompositionSignal"], Optional[RuleOutcome]]


# ---- Stage-0 rules ----

def R1a_single_channel_in_ambiguity(s) -> Optional[RuleOutcome]:
    """Strong single-channel tumor evidence inside a structural-
    ambiguity regime overrides the ambiguity flag.

    Example: pfo004 (real SARC) with 58 CTA hits at 5000+ TPM is
    definitively tumor despite sitting in the mesenchymal-ambiguity
    regime by correlation.
    """
    lymphoid_ambiguity = getattr(s, "_lymphoid_ambiguity", False)
    mesenchymal_ambiguity = getattr(s, "_mesenchymal_ambiguity", False)
    if not (lymphoid_ambiguity or mesenchymal_ambiguity):
        return None
    strong = (
        s.cta_count_above_1_tpm >= _CTA_STRONG_COUNT
        or s.cta_panel_sum_tpm >= _CTA_STRONG_SUM_TPM
        or s.oncofetal_count_above_threshold >= _ONCOFETAL_STRONG_COUNT
        or len(s.type_specific_hits) >= _TUMOR_UP_PANEL_STRONG_COUNT
    )
    if not strong:
        return None
    reasons = []
    if s.cta_count_above_1_tpm >= _CTA_STRONG_COUNT:
        reasons.append(f"CTA_strong(n={s.cta_count_above_1_tpm})")
    if s.oncofetal_count_above_threshold >= _ONCOFETAL_STRONG_COUNT:
        reasons.append(f"oncofetal_strong(n={s.oncofetal_count_above_threshold})")
    if len(s.type_specific_hits) >= _TUMOR_UP_PANEL_STRONG_COUNT:
        reasons.append(f"type_specific_strong(n={len(s.type_specific_hits)})")
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name="R1a-single-channel-tumor-wins-in-ambiguity",
        rationale=",".join(reasons),
    )


def R2_lymphoid_ambiguity(s) -> Optional[RuleOutcome]:
    """Lymphoid structural ambiguity — bulk RNA can't distinguish
    normal lymphoid from lymphoid malignancy."""
    if getattr(s, "_lymphoid_ambiguity", False):
        return RuleOutcome(
            hint="possibly-tumor",
            rule_name="R2-lymphoid-ambiguity",
            structural=True,
            rationale=f"top_HPA={s.top_normal_tissues[0][0] if s.top_normal_tissues else '-'} vs top_TCGA={s.top_tcga_cohorts[0][0] if s.top_tcga_cohorts else '-'}",
        )
    return None


def R3_mesenchymal_ambiguity(s) -> Optional[RuleOutcome]:
    """Mesenchymal structural ambiguity — well-differentiated SARC
    can't be cleanly distinguished from smooth muscle / adipose /
    skeletal muscle / endometrial myometrium by correlation."""
    if getattr(s, "_mesenchymal_ambiguity", False):
        return RuleOutcome(
            hint="possibly-tumor",
            rule_name="R3-mesenchymal-ambiguity",
            structural=True,
            rationale=f"top_HPA={s.top_normal_tissues[0][0] if s.top_normal_tissues else '-'} vs top_TCGA={s.top_tcga_cohorts[0][0] if s.top_tcga_cohorts else '-'}",
        )
    return None


def R1b_strong_evidence_non_ambiguous(s) -> Optional[RuleOutcome]:
    """Non-ambiguous tissue: aggregate ≥ 1.0 OR any strong single
    channel → tumor-consistent."""
    agg = s.evidence.aggregate_score
    cta_strong = (
        s.cta_count_above_1_tpm >= _CTA_STRONG_COUNT
        or s.cta_panel_sum_tpm >= _CTA_STRONG_SUM_TPM
    )
    oncofetal_strong = s.oncofetal_count_above_threshold >= _ONCOFETAL_STRONG_COUNT
    type_specific_strong = len(s.type_specific_hits) >= _TUMOR_UP_PANEL_STRONG_COUNT
    if not (agg >= 1.0 or cta_strong or oncofetal_strong or type_specific_strong):
        return None
    reasons = []
    if agg >= 1.0:
        reasons.append(f"aggregate={agg:.2f}≥1.0")
    if cta_strong:
        reasons.append(f"CTA_strong(n={s.cta_count_above_1_tpm})")
    if oncofetal_strong:
        reasons.append(f"oncofetal_strong(n={s.oncofetal_count_above_threshold})")
    if type_specific_strong:
        reasons.append(f"type_specific_strong(n={len(s.type_specific_hits)})")
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name="R1b-strong-tumor-evidence",
        rationale=",".join(reasons),
    )


def R4_elevated_proliferation(s) -> Optional[RuleOutcome]:
    """Single-channel high proliferation → tumor-consistent."""
    if s.evidence.prolif_log2 >= _PROLIFERATION_HIGH_LOG2:
        return RuleOutcome(
            hint="tumor-consistent",
            rule_name="R4-proliferation-high",
            rationale=f"panel_log2={s.evidence.prolif_log2:.1f}",
        )
    return None


def R5a_confident_healthy(s) -> Optional[RuleOutcome]:
    """Quiet proliferation + strong healthy-margin + no soft tumor
    evidence → healthy-dominant."""
    margin = _margin(s)
    if not (s.evidence.prolif_log2 < _PROLIFERATION_QUIET_LOG2
            and margin >= _HPA_MARGIN_STRONG):
        return None
    cta_soft = s.cta_count_above_1_tpm >= 2
    oncofetal_soft = s.oncofetal_count_above_threshold >= 1
    type_specific_soft = len(s.type_specific_hits) >= 1
    if cta_soft or oncofetal_soft or type_specific_soft:
        return None
    return RuleOutcome(
        hint="healthy-dominant",
        rule_name="R5a-healthy-correlation-quiet-prolif",
        rationale=f"margin={margin:+.2f}",
    )


def R5b_demoted_healthy(s) -> Optional[RuleOutcome]:
    """R5a preconditions but soft tumor evidence fires → demoted to
    possibly-tumor."""
    margin = _margin(s)
    if not (s.evidence.prolif_log2 < _PROLIFERATION_QUIET_LOG2
            and margin >= _HPA_MARGIN_STRONG):
        return None
    cta_soft = s.cta_count_above_1_tpm >= 2
    oncofetal_soft = s.oncofetal_count_above_threshold >= 1
    type_specific_soft = len(s.type_specific_hits) >= 1
    if not (cta_soft or oncofetal_soft or type_specific_soft):
        return None
    demotes = []
    if cta_soft:
        demotes.append(f"CTA_soft(n={s.cta_count_above_1_tpm})")
    if oncofetal_soft:
        demotes.append(f"oncofetal_soft(n={s.oncofetal_count_above_threshold})")
    if type_specific_soft:
        demotes.append("type_specific_soft")
    return RuleOutcome(
        hint="possibly-tumor",
        rule_name="R5b-healthy-correlation-demoted",
        rationale=",".join(demotes),
    )


def R6_weak_margin(s) -> Optional[RuleOutcome]:
    """Weak healthy-margin without strong proliferation → possibly-tumor."""
    margin = _margin(s)
    if margin >= _HPA_MARGIN_WEAK:
        return RuleOutcome(
            hint="possibly-tumor",
            rule_name="R6-weak-healthy-margin",
            rationale=f"margin={margin:+.2f}",
        )
    return None


def R7_default_tumor_consistent(s) -> Optional[RuleOutcome]:
    """Default: neither HPA nor strong proliferation fires — correlation
    favours TCGA → tumor-consistent."""
    return RuleOutcome(
        hint="tumor-consistent",
        rule_name="R7-tcga-dominant-correlation",
    )


def _margin(s) -> float:
    """Correlation margin helper (best HPA ρ − best TCGA ρ)."""
    if not s.top_normal_tissues or not s.top_tcga_cohorts:
        return 0.0
    return s.top_normal_tissues[0][1] - s.top_tcga_cohorts[0][1]


# Ordered rule list. Rules are applied in order; the first to fire
# (return a non-None ``RuleOutcome``) wins and downstream rules are
# skipped. Re-orderable without editing rule bodies.
STAGE0_RULES: list[Stage0Rule] = [
    R1a_single_channel_in_ambiguity,
    R2_lymphoid_ambiguity,
    R3_mesenchymal_ambiguity,
    R1b_strong_evidence_non_ambiguous,
    R4_elevated_proliferation,
    R5a_confident_healthy,
    R5b_demoted_healthy,
    R6_weak_margin,
    R7_default_tumor_consistent,
]


def run_stage0_rules(
    signal, rules: list[Stage0Rule] = None,
) -> tuple[RuleOutcome, list[str]]:
    """Apply rules in order. Return (first-fire outcome, trace).

    The trace is a list of just-the-winning rule's name (one entry
    here). When multi-rule trace is useful (e.g. for post-hoc
    debugging), callers can extend this helper to log every rule's
    return value.
    """
    if rules is None:
        rules = STAGE0_RULES
    for rule in rules:
        outcome = rule(signal)
        if outcome is not None:
            return outcome, [outcome.rule_name]
    # Should be unreachable — R7 is unconditional default.
    return (
        RuleOutcome(hint="tumor-consistent", rule_name="default"),
        ["default"],
    )


# ---- Analysis state (composable; grows stage by stage) ----

@dataclass
class AnalysisState:
    """Accumulating typed container for stage outputs.

    Rules and downstream analyses read whichever stage fields they
    need. ``stage0`` is the Stage-0 tissue-composition signal;
    ``stage1``..``stage6`` will be populated as the pipeline runs
    (Stage-1 cancer candidates, Stage-2 purity, Stage-3 decomposition,
    etc.). A stage that hasn't run yet is ``None`` — rules that
    depend on it must check before reading.

    For now only stage0 is typed through; the other slots are kept
    as the raw ``analysis`` dict pirlygenes already threads through
    the pipeline. Future PRs can replace each slot with a typed
    dataclass as the corresponding stage is refactored.
    """

    stage0: object | None = None  # TissueCompositionSignal
    # Future slots — left as Any so pre-refactor pipeline still works:
    stage1_candidates: list | None = None
    stage2_purity: dict | None = None
    stage3_decomp: object | None = None
    stage4_therapy_axes: dict | None = None
    stage5_expression_ranges: object | None = None
    reasoning_trace: list[str] = field(default_factory=list)
