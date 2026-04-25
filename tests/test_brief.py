"""Tests for the two-tier brief / actionable handoff (#111)."""

import pandas as pd

from pirlygenes.brief import build_brief, build_actionable, _shortlist_omission_note
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
    # File was renamed brief → summary in 4.41.0; header tracks the name.
    assert "# Summary" in md
    assert "**Cancer call:**" in md
    assert "**Purity:**" in md
    assert "model interval" in md
    assert "(CI " not in md
    assert "**Disease state:**" in md
    assert "Top candidate therapies" in md


def test_low_confidence_call_punctuation_is_clean():
    analysis = _make_analysis()
    analysis["candidate_trace"] = [{
        "code": "PRAD",
        "support_geomean": 0.4,
        "signature_score": 0.4,
    }]
    analysis["fit_quality"] = {"label": "weak", "message": "flat signature"}
    ranges_df = _make_ranges_df()

    md = build_brief(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="",
    )
    cancer_line = next(line for line in md.splitlines() if line.startswith("**Cancer call:**"))
    assert "). —" not in cancer_line

    actionable = build_actionable(
        analysis, ranges_df, cancer_code="PRAD",
        disease_state="",
    )
    working_line = next(line for line in actionable.splitlines() if line.startswith("Working call:"))
    assert "). —" not in working_line


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


def test_brief_renders_no_pattern_disease_state_when_scores_exist():
    analysis = _make_analysis()
    analysis["therapy_response_scores"] = {"IFN_response": object()}
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "No strong RNA-defined therapy-exposure" in md


def test_brief_prioritizes_ar_path_and_flags_possible_current_therapy():
    from pirlygenes.therapy_response import TherapyAxisScore

    analysis = _make_analysis()
    analysis["therapy_response_scores"] = {
        "AR_signaling": TherapyAxisScore(
            therapy_class="AR_signaling",
            state="down",
            up_geomean_fold=0.39,
            down_geomean_fold=3.24,
        )
    }
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="**AR axis suppressed** — consistent with ADT exposure.",
    )
    assert md.index("**AR**") < md.index("**FOLH1**")
    ar_line = next(line for line in md.splitlines() if line.startswith("- **AR**"))
    assert "guideline-standard approved pathway" in ar_line
    assert "current/prior ADT or ARPI" in ar_line


def test_brief_uses_path_maturity_across_cancer_types_not_prad_special_case():
    analysis = _make_analysis()
    analysis["cancer_type"] = "BRCA"
    analysis["cancer_name"] = "Breast invasive carcinoma"
    ranges_df = pd.DataFrame([
        {
            "symbol": "TACSTD2", "observed_tpm": 260.0,
            "attr_tumor_tpm": 240.0, "attr_tumor_fraction": 0.92,
            "attr_top_compartment": "tumor", "attr_top_compartment_tpm": 240.0,
            "tme_dominant": False, "tme_explainable": False,
        },
        {
            "symbol": "ERBB2", "observed_tpm": 45.0,
            "attr_tumor_tpm": 40.0, "attr_tumor_fraction": 0.89,
            "attr_top_compartment": "tumor", "attr_top_compartment_tpm": 40.0,
            "tme_dominant": False, "tme_explainable": False,
        },
    ])
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="BRCA",
        disease_state="",
    )
    assert md.index("**ERBB2**") < md.index("**TACSTD2**")
    erbb2_line = next(line for line in md.splitlines() if line.startswith("- **ERBB2**"))
    assert "guideline-standard approved pathway" in erbb2_line
    tacstd2_line = next(line for line in md.splitlines() if line.startswith("- **TACSTD2**"))
    assert "approved later-line pathway" in tacstd2_line


def test_expression_independent_therapy_summary_keeps_rna_contextual():
    analysis = _make_analysis()
    analysis["cancer_type"] = "COAD"
    analysis["cancer_name"] = "Colon adenocarcinoma"
    ranges_df = pd.DataFrame([
        {
            "symbol": "CD274", "observed_tpm": 0.0,
            "attr_tumor_tpm": 0.0, "attr_tumor_fraction": 0.0,
            "attr_top_compartment": "", "attr_top_compartment_tpm": 0.0,
            "tme_dominant": False, "tme_explainable": False,
        },
    ])
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="COAD",
        disease_state="",
    )
    pdcd1_line = next(line for line in md.splitlines() if line.startswith("- **CD274**"))
    assert "expression-independent indication" in pdcd1_line
    assert "target RNA is contextual only" in pdcd1_line
    assert "Clinical maturity: approved antibody" in pdcd1_line
    assert "; Clinical maturity" not in pdcd1_line
    assert "model interval" not in pdcd1_line


def test_brief_flags_possible_current_endocrine_therapy_beyond_prad():
    from pirlygenes.therapy_response import TherapyAxisScore

    analysis = _make_analysis()
    analysis["cancer_type"] = "BRCA"
    analysis["cancer_name"] = "Breast invasive carcinoma"
    analysis["therapy_response_scores"] = {
        "ER_signaling": TherapyAxisScore(
            therapy_class="ER_signaling",
            state="down",
            up_geomean_fold=0.31,
            down_geomean_fold=2.4,
        )
    }
    ranges_df = pd.DataFrame([
        {
            "symbol": "ESR1", "observed_tpm": 80.0,
            "attr_tumor_tpm": 70.0, "attr_tumor_fraction": 0.88,
            "attr_top_compartment": "tumor", "attr_top_compartment_tpm": 70.0,
            "tme_dominant": False, "tme_explainable": False,
        },
    ])
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="BRCA",
        disease_state="**ER-axis suppressed / endocrine-exposed pattern**.",
    )
    esr1_line = next(line for line in md.splitlines() if line.startswith("- **ESR1**"))
    assert "current/prior endocrine therapy" in esr1_line


def test_brief_explains_bulk_present_targets_that_fail_source_gate():
    analysis = _make_analysis()
    ranges_df = pd.concat([
        _make_ranges_df(),
        pd.DataFrame([
            {
                "symbol": "STEAP2", "observed_tpm": 90.0,
                "attribution": {"matched_normal_prostate": 78.0},
                "attr_tumor_tpm": 13.0, "attr_tumor_fraction": 0.14,
                "attr_top_compartment": "matched_normal_prostate",
                "attr_top_compartment_tpm": 78.0,
                "tme_dominant": True, "tme_explainable": True,
            },
            {
                "symbol": "KLK2", "observed_tpm": 247.0,
                "attribution": {"matched_normal_prostate": 155.0},
                "attr_tumor_tpm": 57.0, "attr_tumor_fraction": 0.23,
                "attr_top_compartment": "matched_normal_prostate",
                "attr_top_compartment_tpm": 155.0,
                "tme_dominant": False, "tme_explainable": True,
                "matched_normal_over_predicted": True,
            },
        ]),
    ], ignore_index=True)
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "Target expression source trace" in md
    assert "Tumor-inferred TPM" in md
    assert "Top non-tumor attribution" in md
    assert "STEAP2" in md
    assert "KLK2" in md
    assert "matched-normal prostate" in md
    assert "phase 1 exploratory" in md


def test_source_trace_renders_when_top_trial_rows_are_mixed_source():
    target = pd.Series({
        "symbol": "TARGET1",
        "phase": "phase_2",
        "agent": "trial drug",
        "agent_class": "antibody",
        "treatment_path_tier": "trial_follow_up",
        "eligibility_note": "not default standard",
    })
    expr = pd.Series({
        "symbol": "TARGET1",
        "observed_tpm": 20.0,
        "attr_tumor_tpm": 8.0,
        "attr_tumor_fraction": 0.40,
        "attr_top_compartment": "",
        "attr_top_compartment_tpm": 0.0,
        "tme_dominant": False,
        "tme_explainable": False,
    })
    md = _shortlist_omission_note(
        pd.DataFrame([target]),
        pd.DataFrame([expr]),
        [(target, expr)],
    )
    assert "Target expression source trace" in md
    assert "none modeled" in md


def test_source_trace_does_not_call_non_lineage_component_lineage_background():
    top = pd.Series({
        "symbol": "TOP",
        "phase": "phase_2",
        "agent": "trial drug",
        "agent_class": "antibody",
    })
    top_expr = pd.Series({
        "symbol": "TOP",
        "observed_tpm": 20.0,
        "attr_tumor_tpm": 12.0,
        "attr_tumor_fraction": 0.60,
        "attr_top_compartment": "osteoblast",
        "attr_top_compartment_tpm": 1.0,
        "tme_dominant": False,
        "tme_explainable": False,
    })
    omitted = pd.Series({
        "symbol": "ERBB2",
        "phase": "phase_2",
        "agent": "trial drug",
        "agent_class": "ADC",
    })
    omitted_expr = pd.Series({
        "symbol": "ERBB2",
        "observed_tpm": 25.0,
        "attr_tumor_tpm": 0.0,
        "attr_tumor_fraction": 0.0,
        "attr_top_compartment": "osteoblast",
        "attr_top_compartment_tpm": 8.0,
        "matched_normal_over_predicted": True,
        "tme_dominant": True,
        "tme_explainable": True,
    })
    md = _shortlist_omission_note(
        pd.DataFrame([top, omitted]),
        pd.DataFrame([top_expr, omitted_expr]),
        [(top, top_expr)],
    )
    erbb2_line = next(line for line in md.splitlines() if line.startswith("| ERBB2 "))
    assert "osteoblast over-predicts / non-tumor background" in erbb2_line
    assert "osteoblast over-predicts / lineage background" not in erbb2_line


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

    # Actionable remains available as an internal builder, but its
    # cross-links should now point to the consolidated evidence.md
    # appendix rather than a standalone targets.md file.
    for heading in ["Sample and confidence", "Cancer call and disease state",
                    "Therapy landscape"]:
        assert heading in md, f"missing heading: {heading}"
    assert "model interval" in md
    assert "(CI " not in md
    assert "*-evidence.md*" in md or "`*-evidence.md`" in md, (
        "actionable should link to evidence.md as the target-table source"
    )


def test_brief_normalizes_path_like_sample_id():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
        sample_id="/tmp/run-123/rs",
    )
    assert md.splitlines()[0] == "# Summary: rs"


def test_brief_uses_tumor_band_without_attribution_dict():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    idx = ranges_df.index[ranges_df["symbol"] == "FOLH1"][0]
    ranges_df.at[idx, "attribution"] = {}
    md = build_brief(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "tumor-specific decomposition was unavailable" not in md
    assert "128 TPM (model interval 128-128" in md


def test_actionable_renders_tumor_band_without_attribution_dict():
    analysis = _make_analysis()
    ranges_df = _make_ranges_df()
    idx = ranges_df.index[ranges_df["symbol"] == "FOLH1"][0]
    ranges_df.at[idx, "attribution"] = {}
    md = build_actionable(
        analysis,
        ranges_df,
        cancer_code="PRAD",
        disease_state="",
    )
    assert "| **FOLH1** | 177Lu-PSMA-617 | radioligand | Approved | mCRPC | 142.0 | 128 (128-128) |" in md
