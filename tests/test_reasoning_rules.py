"""Tests for the Stage-0 reasoning rules as standalone pure functions.

Each rule in pirlygenes.reasoning takes a TissueCompositionSignal
and returns a RuleOutcome or None. The rules are testable without
running the full assess_tissue_composition pipeline — these tests
cover each rule's contract individually.
"""

import pytest

from pirlygenes.healthy_vs_tumor import TissueCompositionSignal
from pirlygenes.reasoning import (
    R1a_single_channel_in_ambiguity,
    R1b_strong_evidence_non_ambiguous,
    R2_lymphoid_ambiguity,
    R3_mesenchymal_ambiguity,
    R4_elevated_proliferation,
    R5a_confident_healthy,
    R5b_demoted_healthy,
    R6_weak_margin,
    R7_default_tumor_consistent,
    RuleOutcome,
    run_stage0_rules,
)
from pirlygenes.tumor_evidence import TumorEvidenceScore


def _make_signal(**kwargs):
    """Build a minimal TissueCompositionSignal for testing."""
    defaults = dict(
        top_normal_tissues=[],
        top_tcga_cohorts=[],
        proliferation_log2_mean=0.0,
        proliferation_genes_observed=0,
        cancer_hint="tumor-consistent",
        n_reference_genes=5000,
        verdict="",
        structural_ambiguity=False,
        cta_panel_sum_tpm=0.0,
        cta_count_above_1_tpm=0,
        cta_top_hits=[],
        oncofetal_count_above_threshold=0,
        oncofetal_top_hits=[],
        type_specific_cohort="",
        type_specific_hits=[],
        evidence=TumorEvidenceScore(),
    )
    defaults.update(kwargs)
    s = TissueCompositionSignal(**defaults)
    s._lymphoid_ambiguity = kwargs.pop("_lymphoid_ambiguity", False)
    s._mesenchymal_ambiguity = kwargs.pop("_mesenchymal_ambiguity", False)
    return s


# ---------- R1a ----------

def test_R1a_fires_on_strong_single_channel_in_lymphoid_ambiguity():
    s = _make_signal(cta_count_above_1_tpm=10, cta_panel_sum_tpm=500)
    s._lymphoid_ambiguity = True
    out = R1a_single_channel_in_ambiguity(s)
    assert out is not None
    assert out.hint == "tumor-consistent"
    assert "R1a" in out.rule_name


def test_R1a_does_not_fire_without_ambiguity_flag():
    s = _make_signal(cta_count_above_1_tpm=10)
    # No ambiguity flags set
    assert R1a_single_channel_in_ambiguity(s) is None


def test_R1a_does_not_fire_without_strong_signal():
    s = _make_signal(cta_count_above_1_tpm=1)  # below strong threshold
    s._lymphoid_ambiguity = True
    assert R1a_single_channel_in_ambiguity(s) is None


# ---------- R2 / R3 ----------

def test_R2_fires_on_lymphoid_ambiguity_and_marks_structural():
    s = _make_signal(
        top_normal_tissues=[("nTPM_lymph_node", 0.82)],
        top_tcga_cohorts=[("FPKM_DLBC", 0.83)],
    )
    s._lymphoid_ambiguity = True
    out = R2_lymphoid_ambiguity(s)
    assert out is not None
    assert out.hint == "possibly-tumor"
    assert out.structural is True


def test_R3_fires_on_mesenchymal_ambiguity():
    s = _make_signal(
        top_normal_tissues=[("nTPM_smooth_muscle", 0.85)],
        top_tcga_cohorts=[("FPKM_SARC", 0.80)],
    )
    s._mesenchymal_ambiguity = True
    out = R3_mesenchymal_ambiguity(s)
    assert out is not None
    assert out.hint == "possibly-tumor"
    assert out.structural is True


def test_R2_R3_skip_when_not_ambiguous():
    s = _make_signal()
    assert R2_lymphoid_ambiguity(s) is None
    assert R3_mesenchymal_ambiguity(s) is None


# ---------- R1b ----------

def test_R1b_fires_on_aggregate_above_1():
    evidence = TumorEvidenceScore(cta=0.5, oncofetal=0.3, proliferation=0.4)
    # aggregate = 1.2 ≥ 1.0
    assert evidence.aggregate_score >= 1.0
    s = _make_signal(evidence=evidence)
    out = R1b_strong_evidence_non_ambiguous(s)
    assert out is not None
    assert out.hint == "tumor-consistent"
    assert "aggregate" in out.rationale


def test_R1b_fires_on_strong_single_CTA():
    s = _make_signal(cta_count_above_1_tpm=10)
    out = R1b_strong_evidence_non_ambiguous(s)
    assert out is not None
    assert "CTA_strong" in out.rationale


def test_R1b_skips_below_thresholds():
    s = _make_signal(
        cta_count_above_1_tpm=1,
        evidence=TumorEvidenceScore(cta=0.2),  # aggregate only 0.2
    )
    assert R1b_strong_evidence_non_ambiguous(s) is None


# ---------- R4 ----------

def test_R4_fires_on_high_proliferation():
    s = _make_signal(
        evidence=TumorEvidenceScore(prolif_log2=5.0),
    )
    out = R4_elevated_proliferation(s)
    assert out is not None
    assert out.hint == "tumor-consistent"


def test_R4_skips_when_proliferation_quiet():
    s = _make_signal(evidence=TumorEvidenceScore(prolif_log2=2.0))
    assert R4_elevated_proliferation(s) is None


# ---------- R5a / R5b ----------

def test_R5a_fires_on_healthy_profile():
    s = _make_signal(
        top_normal_tissues=[("nTPM_cerebral_cortex", 0.95)],
        top_tcga_cohorts=[("FPKM_LGG", 0.85)],
        evidence=TumorEvidenceScore(prolif_log2=0.5),
    )
    # Margin 0.10, quiet prolif, no soft tumor evidence
    out = R5a_confident_healthy(s)
    assert out is not None
    assert out.hint == "healthy-dominant"


def test_R5b_demotes_healthy_when_soft_tumor_evidence_fires():
    s = _make_signal(
        top_normal_tissues=[("nTPM_cerebral_cortex", 0.95)],
        top_tcga_cohorts=[("FPKM_LGG", 0.85)],
        evidence=TumorEvidenceScore(prolif_log2=0.5),
        cta_count_above_1_tpm=3,  # soft CTA
    )
    out = R5b_demoted_healthy(s)
    assert out is not None
    assert out.hint == "possibly-tumor"
    assert "CTA_soft" in out.rationale


# ---------- R6 / R7 ----------

def test_R6_fires_on_weak_margin():
    s = _make_signal(
        top_normal_tissues=[("nTPM_lung", 0.82)],
        top_tcga_cohorts=[("FPKM_LUAD", 0.79)],
    )
    # Margin 0.03 — above weak threshold 0.02 but below strong 0.05
    out = R6_weak_margin(s)
    assert out is not None
    assert out.hint == "possibly-tumor"


def test_R7_is_unconditional_default():
    s = _make_signal()
    out = R7_default_tumor_consistent(s)
    assert out is not None
    assert out.hint == "tumor-consistent"


# ---------- Rule runner ----------

def test_run_stage0_rules_picks_first_match():
    """Rule runner returns the first rule that fires — ordering matters."""
    s = _make_signal(cta_count_above_1_tpm=10)  # R1b would fire
    s._lymphoid_ambiguity = True  # R1a would also fire; R2 would too
    outcome, trace = run_stage0_rules(s)
    # R1a comes first and matches (strong single-channel in ambiguity)
    assert outcome.rule_name.startswith("R1a")
    assert outcome.hint == "tumor-consistent"


def test_run_stage0_rules_falls_to_default_when_nothing_matches():
    """Rule runner reaches R7 when no earlier rule fires."""
    s = _make_signal()  # no tumor evidence, no ambiguity, no healthy margin
    outcome, trace = run_stage0_rules(s)
    assert outcome.rule_name.startswith("R7")
    assert outcome.hint == "tumor-consistent"


def test_run_stage0_rules_reorderable():
    """Custom rule order is honored."""
    s = _make_signal(cta_count_above_1_tpm=10)
    s._lymphoid_ambiguity = True
    # Put R2 first — lymphoid-ambiguity would then fire before R1a
    custom_order = [R2_lymphoid_ambiguity, R1a_single_channel_in_ambiguity,
                    R7_default_tumor_consistent]
    outcome, trace = run_stage0_rules(s, rules=custom_order)
    assert outcome.rule_name == "R2-lymphoid-ambiguity"
