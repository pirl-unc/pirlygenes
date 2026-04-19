"""Tests for the two-tier brief / actionable handoff (#111)."""

import pandas as pd

from pirlygenes.brief import build_brief, build_actionable
from pirlygenes.confidence import ConfidenceTier


def _make_analysis(purity_point=0.28, ci_low=0.19, ci_high=0.40,
                   purity_tier_label="moderate",
                   degradation="mild", library_prep="exome_capture",
                   preservation="ffpe"):
    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.library_prep = library_prep
    ctx.library_prep_confidence = 0.9
    ctx.preservation = preservation
    ctx.preservation_confidence = 0.85
    ctx.degradation_severity = degradation
    ctx.degradation_index = 0.6
    ctx.missing_mt = False
    ctx.signals = {}
    ctx.flags = []

    return {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate adenocarcinoma",
        "purity": {
            "overall_estimate": purity_point,
            "overall_lower": ci_low,
            "overall_upper": ci_high,
        },
        "purity_confidence": ConfidenceTier(
            tier=purity_tier_label,
            reasons=(
                ["moderate purity CI span (21 pp)", "low-purity regime (28%)"]
                if purity_tier_label in {"moderate", "low"} else []
            ),
        ),
        "sample_context": ctx,
        "therapy_response_scores": {},
    }


def _make_ranges_df():
    return pd.DataFrame([
        {
            "symbol": "FOLH1", "observed_tpm": 142.0,
            "attribution": {"endothelial": 12.0},
            "attr_tumor_tpm": 128.0, "attr_tumor_fraction": 0.90,
            "attr_top_compartment": "endothelial", "attr_top_compartment_tpm": 12.0,
            "tme_dominant": False, "tme_explainable": False,
        },
        {
            "symbol": "STEAP1", "observed_tpm": 78.0,
            "attribution": {"fibroblast": 10.0, "endothelial": 6.0},
            "attr_tumor_tpm": 62.0, "attr_tumor_fraction": 0.79,
            "attr_top_compartment": "fibroblast", "attr_top_compartment_tpm": 10.0,
            "tme_dominant": False, "tme_explainable": False,
        },
        {
            "symbol": "DLL3", "observed_tpm": 0.5,
            "attribution": {}, "attr_tumor_tpm": 0.0, "attr_tumor_fraction": 0.0,
            "attr_top_compartment": "", "attr_top_compartment_tpm": 0.0,
            "tme_dominant": False, "tme_explainable": False,
        },
        {
            "symbol": "AR", "observed_tpm": 50.0,
            "attribution": {"endothelial": 2.0},
            "attr_tumor_tpm": 48.0, "attr_tumor_fraction": 0.96,
            "attr_top_compartment": "endothelial", "attr_top_compartment_tpm": 2.0,
            "tme_dominant": False, "tme_explainable": False,
        },
    ])


def test_brief_is_compact():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="Castrate-resistant pattern.",
        sample_id="sample_X",
    )
    lines = md.splitlines()
    # ≤ 40 lines is the contract.
    assert len(lines) <= 40, f"brief is {len(lines)} lines, must be ≤ 40:\n{md}"

    # Key structural elements present.
    assert "# Brief" in md
    assert "**Cancer call:**" in md
    assert "**Purity:**" in md
    assert "**Disease state:**" in md
    assert "Top candidate therapies" in md


def test_brief_excludes_absent_targets():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="",
    )
    # DLL3 is absent (0.5 TPM) — must not appear in the top bullets.
    assert "DLL3" not in md, "brief should skip absent targets from the top list"


def test_brief_reports_tumor_attributed_for_present_targets():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="",
    )
    # FOLH1 has tumor-attr 128; the bullet should mention it.
    assert "FOLH1" in md
    assert "128" in md or "**FOLH1**" in md


def test_brief_no_internal_jargon():
    analysis = _make_analysis(purity_tier_label="low")
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="",
    )
    # Forbidden jargon — internal variable names and pipeline terms.
    for token in [
        "NNLS", "Spearman", "x1.10", "×1.10",
        "tme_fold_med", "_combine_purity_estimates",
        "overexplained_tpm", "sig_stability",
    ]:
        assert token not in md, f"jargon leak: {token}"


def test_brief_handles_uncurated_cancer_type():
    """The brief must gracefully handle any cancer code that isn't in
    the curated key-genes panel — not just TCGA codes. Uses a fake
    placeholder code so the test is independent of which TCGA codes
    we've curated (all 33 are curated as of #155; pick a non-existent
    code so the test remains valid as we expand)."""
    analysis = _make_analysis()
    analysis["cancer_type"] = "ZZUNCURATED"
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis, ranges_df, cancer_code="ZZUNCURATED",
        disease_state="",
    )
    assert "not yet in the curated key-genes panel" in md


def test_actionable_is_longer_but_structured():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_actionable(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="Castrate-resistant.",
        sample_id="sample_X",
    )
    # Actionable should be > 15 lines (more detail than the brief).
    assert len(md.splitlines()) > 15

    # Key section headings.
    for heading in ["Sample and confidence", "Cancer call and disease state",
                    "Therapy landscape", "Biomarker panel"]:
        assert heading in md, f"missing heading: {heading}"
