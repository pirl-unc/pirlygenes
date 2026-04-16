"""Tests for v4.1 report-clarity + met-site bundle (issues #13, #32, #33)."""

import pandas as pd
import pytest

from pirlygenes.cli import _next_best_support_gap
from pirlygenes.plot import MET_SITE_TISSUE_AUGMENTATION, estimate_tumor_expression_ranges
from pirlygenes.gene_sets_cancer import pan_cancer_expression


# ── #32: qualitative score language ────────────────────────────────────


def test_next_best_support_gap_returns_ratio_for_two_candidates():
    trace = [
        {"code": "PRAD", "support_norm": 1.00},
        {"code": "HNSC", "support_norm": 0.40},
    ]
    code, ratio = _next_best_support_gap(trace)
    assert code == "HNSC"
    assert ratio == pytest.approx(2.5, rel=1e-3)


def test_next_best_support_gap_handles_edge_cases():
    # One candidate — no gap to measure
    assert _next_best_support_gap([{"code": "PRAD", "support_norm": 1.0}]) == (None, None)
    # Empty
    assert _next_best_support_gap([]) == (None, None)
    # Runner-up has zero support — can't divide
    code, ratio = _next_best_support_gap(
        [
            {"code": "PRAD", "support_norm": 1.0},
            {"code": "HNSC", "support_norm": 0.0},
        ]
    )
    assert code == "HNSC"
    assert ratio is None


# ── #13: met-site augmentation ─────────────────────────────────────────


def test_met_site_augmentation_map_covers_all_sites_mentioned_in_issue():
    """Issue #13 lists primary / lymph_node / liver / brain / lung / bone.
    Any new met site exposed via CLI must live in the augmentation map
    so downstream knows how to augment the TME reference."""
    expected = {"primary", "lymph_node", "liver", "brain", "lung", "bone"}
    assert expected.issubset(set(MET_SITE_TISSUE_AUGMENTATION.keys()))
    # `primary` intentionally adds nothing — tumor is assumed in situ.
    assert MET_SITE_TISSUE_AUGMENTATION["primary"] == set()


def test_met_site_liver_includes_liver_tissue_in_tme_reference():
    """Validates #13: a liver-met ``estimate_tumor_expression_ranges``
    call should use a TME reference set that includes ``liver``, while
    the default (met_site=None) does not."""
    # Small synthetic sample drawn from the bundled reference so the
    # function can actually run.
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["nTPM_liver"].astype(float),  # a liver-rich synthetic
        }
    )
    purity = {
        "overall_estimate": 0.6, "overall_lower": 0.4, "overall_upper": 0.8,
        "components": {"stromal": {"enrichment": 1.0}, "immune": {"enrichment": 1.0}},
    }

    baseline = estimate_tumor_expression_ranges(
        df_gene_expr=df, cancer_type="COAD", purity_result=purity,
        met_site=None,
    )
    augmented = estimate_tumor_expression_ranges(
        df_gene_expr=df, cancer_type="COAD", purity_result=purity,
        met_site="liver",
    )

    # TME medians for liver-expressed genes (ALB, APOA1, HP) should be
    # materially higher under the liver-augmented reference, since
    # adding the liver column to the TME tissue set raises the
    # percentile for liver-high genes.
    def _tme_med(df_, symbol):
        sub = df_[df_["symbol"] == symbol]
        if sub.empty:
            return None
        # estimate_tumor_expression_ranges reports the TME background
        # as a fold-over-HK median; a higher value means the reference
        # TME set picks up more of this gene's signal.
        return float(sub.iloc[0]["tme_fold_med"])

    alb_baseline = _tme_med(baseline, "ALB")
    alb_augmented = _tme_med(augmented, "ALB")
    # Both runs should produce an ALB row (it's a reference gene).
    assert alb_baseline is not None
    assert alb_augmented is not None
    # Augmented reference must have higher or equal TME contribution
    # for the host-tissue gene.
    assert alb_augmented >= alb_baseline
    # And strictly higher when liver signal is the distinguishing
    # factor (ALB is nearly liver-only).
    assert alb_augmented > alb_baseline


def test_met_site_rejects_unknown_value():
    """Invalid met_site passed directly to the estimator is a no-op on
    the set-union rather than a crash (the CLI layer validates before
    reaching here). The function must not error on unknown strings."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    df = pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref["nTPM_liver"].astype(float),
        }
    )
    purity = {
        "overall_estimate": 0.6, "overall_lower": 0.4, "overall_upper": 0.8,
        "components": {"stromal": {"enrichment": 1.0}, "immune": {"enrichment": 1.0}},
    }
    # Must not crash on unknown met_site (augmentation map .get default
    # is an empty set). CLI validates up front in analyze().
    ranges = estimate_tumor_expression_ranges(
        df_gene_expr=df, cancer_type="COAD", purity_result=purity,
        met_site="moon",
    )
    assert len(ranges) > 0


def test_summary_md_structure_for_report_clarity(tmp_path):
    """#33: summary.md should include the input filename, the source
    attribution (auto vs user-specified), the tissue-score caveat, and
    no raw ``score 0.019``-style composite number in the prose."""
    from pirlygenes.cli import _generate_text_reports

    analysis = {
        "cancer_type": "PRAD",
        "cancer_name": "Prostate Adenocarcinoma",
        "cancer_score": 0.019,  # realistic low composite
        "cancer_type_source": "auto-detected",
        "top_cancers": [("PRAD", 0.019)],
        "purity": {
            "overall_estimate": 0.11,
            "overall_lower": 0.01,
            "overall_upper": 0.11,
            "components": {
                "stromal": {"enrichment": 4.3},
                "immune": {"enrichment": 2.5},
            },
        },
        "tissue_scores": [
            ("prostate", 0.90, 20),
            ("smooth_muscle", 0.72, 18),
            ("rectum", 0.71, 17),
        ],
        "mhc1": {"HLA-A": 25, "HLA-B": 30, "HLA-C": 22, "B2M": 3000},
        "mhc2": {},
        "candidate_trace": [
            {"code": "PRAD", "support_norm": 1.00, "signature_score": 0.67},
            {"code": "HNSC", "support_norm": 0.40, "signature_score": 0.35},
        ],
        "family_summary": {},
        "fit_quality": {"label": "strong_separation", "message": "Top match is clearly separated from alternatives."},
        "sample_mode": "solid",
        "analysis_constraints": {},
    }

    prefix = str(tmp_path / "sample")
    # Minimal embedding_meta shape — summary.md is written first, before
    # the more demanding analysis.md fields are consulted.
    embedding_meta = {
        "method": "hierarchy", "feature_kind": "hierarchical_scores",
        "n_genes": 0, "n_features": 0, "n_types": 33, "families": [],
    }
    try:
        _generate_text_reports(
            analysis, embedding_meta, prefix,
            decomp_results=[], input_path="/data/sample_BG002.tsv",
        )
    except KeyError:
        # analysis.md generation may need more context than this fixture
        # provides — we only assert on summary.md below.
        pass

    summary_md = (tmp_path / "sample-summary.md").read_text()
    # Input filename header (#33)
    assert "/data/sample_BG002.tsv" in summary_md
    # Source attribution (#33)
    assert "auto-detected" in summary_md
    # Raw composite score 0.019 should NOT appear in prose (#32).
    assert "0.019" not in summary_md, (
        "Raw composite cancer_score must not be in summary prose (#32)."
    )
    # Qualitative ratio instead (#32) — top match is 2.5× HNSC.
    assert "2.5" in summary_md and "HNSC" in summary_md
    # Tissue score caveat (#33)
    assert "similarity" in summary_md.lower()


def test_detailed_report_uses_generic_lineage_caveat(tmp_path):
    from pirlygenes.cli import _generate_text_reports

    analysis = {
        "cancer_type": "HNSC",
        "cancer_name": "Head and Neck Squamous Cell Carcinoma",
        "cancer_score": 0.18,
        "top_cancers": [("HNSC", 0.18), ("LUSC", 0.12)],
        "candidate_trace": [
            {
                "code": "HNSC",
                "support_score": 0.18,
                "support_geomean": 0.18,
                "support_norm": 1.0,
                "signature_score": 0.81,
                "purity_estimate": 0.42,
                "family_label": "SQUAMOUS",
                "lineage_purity": 0.44,
                "lineage_concordance": 0.89,
            },
            {
                "code": "LUSC",
                "support_score": 0.12,
                "support_geomean": 0.12,
                "support_norm": 0.67,
                "signature_score": 0.72,
                "purity_estimate": 0.38,
                "family_label": "SQUAMOUS",
                "lineage_purity": 0.41,
                "lineage_concordance": 0.82,
            },
        ],
        "fit_quality": {
            "label": "ambiguous",
            "message": "Top squamous candidates remain close; treat the exact site as provisional.",
        },
        "family_summary": {
            "display": "squamous family (HNSC > LUSC)",
            "subtype_clause": "HNSC > LUSC",
        },
        "purity": {
            "overall_estimate": 0.42,
            "overall_lower": 0.25,
            "overall_upper": 0.58,
            "cancer_type": "HNSC",
            "components": {
                "stromal": {"enrichment": 2.1},
                "immune": {"enrichment": 1.7},
                "lineage": {
                    "per_gene": [
                        {"gene": "KRT14", "purity": 0.52},
                        {"gene": "KRT5", "purity": 0.47},
                        {"gene": "TP63", "purity": 0.24},
                    ],
                    "purity": 0.49,
                    "lower": 0.47,
                    "upper": 0.52,
                },
            },
        },
        "tissue_scores": [("tongue", 0.93, 20), ("esophagus", 0.84, 20)],
        "mhc1": {"HLA-A": 26, "HLA-B": 31, "B2M": 220},
        "mhc2": {},
        "sample_mode": "solid",
        "call_summary": {
            "label_options": ["HNSC", "LUSC"],
            "label_display": "HNSC or LUSC",
            "reported_context": "primary",
            "reported_site": "primary site",
            "site_indeterminate": False,
            "site_note": None,
            "hypothesis_display": ["HNSC / solid_primary", "LUSC / solid_primary"],
        },
    }
    embedding_meta = {
        "method": "hierarchy",
        "feature_kind": "hierarchical_scores",
        "n_features": 12,
        "n_types": 3,
        "families": ["SQUAMOUS"],
        "sites": ["tongue", "lung", "esophagus"],
    }
    prefix = str(tmp_path / "hnsccase")

    _generate_text_reports(analysis, embedding_meta, prefix, decomp_results=[])

    detailed = (tmp_path / "hnsccase-analysis.md").read_text()
    assert "Lineage caveat" in detailed
    assert "prostate-lineage" not in detailed
    assert "do NOT by themselves distinguish tumor cells from benign cells of the same lineage" in detailed


def test_compose_disease_state_detects_crpc_nepc_pattern():
    """#78: AR retained + AR targets collapsed + NE up + AR axis
    down must produce the castrate-resistant + emerging-NEPC narrative.
    """
    from pirlygenes.cli import compose_disease_state_narrative
    from pirlygenes.therapy_response import TherapyAxisScore

    analysis = {
        "cancer_type": "PRAD",
        "purity": {
            "overall_estimate": 0.64,
            "components": {
                "lineage": {
                    "per_gene": [
                        {"gene": "AR", "purity": 0.51},
                        {"gene": "STEAP2", "purity": 0.16},
                        {"gene": "KLK3", "purity": 0.003},
                        {"gene": "KLK2", "purity": 0.015},
                        {"gene": "NKX3-1", "purity": 0.011},
                        {"gene": "HOXB13", "purity": 0.004},
                        {"gene": "FOLH1", "purity": 0.073},
                    ]
                }
            },
        },
        "therapy_response_scores": {
            "AR_signaling": TherapyAxisScore(
                therapy_class="AR_signaling", state="down",
                up_geomean_fold=0.33, down_geomean_fold=2.54,
            ),
            "NE_differentiation": TherapyAxisScore(
                therapy_class="NE_differentiation", state="up",
                up_geomean_fold=2.08,
            ),
            "EMT": TherapyAxisScore(
                therapy_class="EMT", state="up", up_geomean_fold=8.95,
            ),
            "hypoxia": TherapyAxisScore(
                therapy_class="hypoxia", state="up", up_geomean_fold=3.52,
            ),
            "IFN_response": TherapyAxisScore(
                therapy_class="IFN_response", state="up",
                up_geomean_fold=2.73,
            ),
        },
    }
    narrative = compose_disease_state_narrative(analysis)
    # Core clinical call must be present.
    assert "Castrate-resistant" in narrative
    assert "neuroendocrine" in narrative.lower()
    # Each collapsed AR target should be cited in evidence.
    for g in ("KLK3", "KLK2", "NKX3-1"):
        assert g in narrative
    # EMT + hypoxia cross-axis must land.
    assert "EMT" in narrative and "hypoxia" in narrative
    # IFN active must be flagged so users discount MHC-I fold changes.
    assert "IFN" in narrative


def test_compose_disease_state_empty_when_no_pattern():
    """Generic samples with nothing notable should not produce a
    disease-state narrative — callers skip the section when empty.
    """
    from pirlygenes.cli import compose_disease_state_narrative

    analysis = {
        "cancer_type": "PRAD",
        "purity": {"overall_estimate": 0.7, "components": {}},
        "therapy_response_scores": {},
    }
    assert compose_disease_state_narrative(analysis) == ""


def test_recommended_targets_skips_tme_dominant_rows():
    """#79: ⚠⚠ (tme_dominant) rows must not appear in the Recommended
    Targets Summary; they're called out as excluded."""
    import pandas as pd
    from pirlygenes.cli import _generate_target_report

    purity = {
        "overall_estimate": 0.6, "overall_lower": 0.5, "overall_upper": 0.7,
    }
    analysis = {
        "sample_mode": "solid", "cancer_type": "PRAD",
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 80, "B2M": 300},
    }
    ranges_df = pd.DataFrame([
        # TME-dominant top row → must be filtered from the summary
        {"symbol": "CD74", "median_est": 1580, "observed_tpm": 1580,
         "est_1": 1156, "est_9": 1580, "pct_cancer_median": 1.5,
         "tcga_percentile": 0.94, "is_surface": True, "is_cta": False,
         "tme_explainable": True, "tme_dominant": True,
         "excluded_from_ranking": False, "therapies": "",
         "max_healthy_tpm": 2000, "tme_fold_lo": 0.1, "tme_fold_med": 0.2,
         "tme_fold_hi": 0.3, "cohort_prior_tpm": 1400, "tme_only_tpm": 1100,
         "matched_normal_tpm": 0, "matched_normal_tissue": "",
         "matched_normal_fraction": 0.0, "estimation_path": "clamped",
         "low_confidence_tumor": True, "category": "therapy_target",
         **{f"est_{i+1}": 1156 + i*50 for i in range(9)},
        },
        # Clean ADAM9 → SHOULD appear in the summary
        {"symbol": "ADAM9", "median_est": 998, "observed_tpm": 825,
         "est_1": 696, "est_9": 2179, "pct_cancer_median": 7.5,
         "tcga_percentile": 1.0, "is_surface": True, "is_cta": False,
         "tme_explainable": False, "tme_dominant": False,
         "excluded_from_ranking": False, "therapies": "ADC",
         "max_healthy_tpm": 300, "tme_fold_lo": 0.1, "tme_fold_med": 0.2,
         "tme_fold_hi": 0.3, "cohort_prior_tpm": 100, "tme_only_tpm": 150,
         "matched_normal_tpm": 0, "matched_normal_tissue": "",
         "matched_normal_fraction": 0.0, "estimation_path": "tme_only",
         "low_confidence_tumor": False, "category": "therapy_target",
         **{f"est_{i+1}": 696 + i*150 for i in range(9)},
        },
    ])

    tmp_prefix = "/tmp/target_test"
    import os
    if os.path.exists(f"{tmp_prefix}-targets.md"):
        os.remove(f"{tmp_prefix}-targets.md")
    _generate_target_report(
        ranges_df, analysis, tmp_prefix, cancer_type="PRAD",
        purity_result=purity,
    )
    targets = open(f"{tmp_prefix}-targets.md").read()

    # The Recommended Targets section must not list CD74 as a best
    # surface target — it was ⚠⚠ flagged.
    recs_block = targets.split("## Recommended Targets Summary")[-1]
    assert "**Best surface targets**" in recs_block
    # Clean ADAM9 should be there
    assert "ADAM9" in recs_block
    # CD74 is in the full targets table above but NOT in the
    # recommendations block
    assert "CD74" not in recs_block.split("**Best CTA targets**")[0]


def test_target_report_explains_blocked_fn1_pyx201_call():
    """FN1 should carry an explicit report caveat when the curated
    PYX-201 hook is withheld for lack of EDB+ transcript support."""
    from pirlygenes.cli import _generate_target_report

    purity = {
        "overall_estimate": 0.6, "overall_lower": 0.5, "overall_upper": 0.7,
    }
    analysis = {
        "sample_mode": "solid", "cancer_type": "PRAD",
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 80, "B2M": 300},
    }
    ranges_df = pd.DataFrame([
        {
            "symbol": "ADAM9", "median_est": 998, "observed_tpm": 825,
            "est_1": 696, "est_9": 2179, "pct_cancer_median": 7.5,
            "tcga_percentile": 1.0, "is_surface": True, "is_cta": False,
            "tme_explainable": False, "tme_dominant": False,
            "excluded_from_ranking": False, "therapies": "ADC",
            "therapy_supported": True, "therapy_support_note": "",
            "max_healthy_tpm": 300, "tme_fold_lo": 0.1, "tme_fold_med": 0.2,
            "tme_fold_hi": 0.3, "cohort_prior_tpm": 100, "tme_only_tpm": 150,
            "matched_normal_tpm": 0, "matched_normal_tissue": "",
            "matched_normal_fraction": 0.0, "estimation_path": "tme_only",
            "low_confidence_tumor": False, "category": "therapy_target",
            **{f"est_{i+1}": 696 + i*150 for i in range(9)},
        },
        {
            "symbol": "FN1", "median_est": 260, "observed_tpm": 180,
            "est_1": 180, "est_9": 340, "pct_cancer_median": 0.9,
            "tcga_percentile": 0.65, "is_surface": False, "is_cta": False,
            "tme_explainable": False, "tme_dominant": False,
            "excluded_from_ranking": False, "therapies": "",
            "therapy_supported": False,
            "therapy_support_note": (
                "PYX-201 (NCT05720117) targets EDB+ FN1; bulk gene-level FN1 alone "
                "is not sufficient evidence because transcript-level data is unavailable."
            ),
            "max_healthy_tpm": 500, "tme_fold_lo": 0.1, "tme_fold_med": 0.2,
            "tme_fold_hi": 0.3, "cohort_prior_tpm": 120, "tme_only_tpm": 80,
            "matched_normal_tpm": 0, "matched_normal_tissue": "",
            "matched_normal_fraction": 0.0, "estimation_path": "tme_only",
            "low_confidence_tumor": False, "category": "other",
            **{f"est_{i+1}": 180 + i*20 for i in range(9)},
        },
    ])

    tmp_prefix = "/tmp/target_test_fn1"
    import os
    if os.path.exists(f"{tmp_prefix}-targets.md"):
        os.remove(f"{tmp_prefix}-targets.md")
    _generate_target_report(
        ranges_df, analysis, tmp_prefix, cancer_type="PRAD",
        purity_result=purity,
    )
    targets = open(f"{tmp_prefix}-targets.md").read()

    assert "PYX-201 (NCT05720117) targets EDB+ FN1" in targets
    assert "Landscape cautions" in targets


def test_ci_confidence_tier_buckets():
    from pirlygenes.cli import _ci_confidence_tier

    assert _ci_confidence_tier(0.58, 0.70) == "high"       # span 0.12
    assert _ci_confidence_tier(0.40, 0.70) == "moderate"   # span 0.30
    assert _ci_confidence_tier(0.19, 1.00) == "low"        # span 0.81
    assert _ci_confidence_tier(None, 0.5) == "unknown"


def test_filter_quality_flags_rewrites_mt_warning_under_exome():
    """#77: MT 'Suspicious' warning must be rewritten as informational
    when the inferred library prep (exome capture / poly-A) already
    explains MT absence."""
    from pirlygenes.cli import _filter_quality_flags_against_context
    from pirlygenes.sample_context import SampleContext

    flags = [
        "Suspicious MT fraction: 0.0% (n_mt_found=13/15) — mitochondrial "
        "genes appear filtered or renamed in the input",
        "Some other warning that must pass through",
    ]
    ctx = SampleContext(library_prep="exome_capture")
    out = _filter_quality_flags_against_context(flags, ctx)
    assert len(out) == 2
    assert "Suspicious MT fraction" not in out[0]
    assert "informational" in out[0]
    assert out[1] == "Some other warning that must pass through"

    # Under total_rna the warning should pass through unchanged.
    ctx2 = SampleContext(library_prep="total_rna")
    out2 = _filter_quality_flags_against_context(flags, ctx2)
    assert "Suspicious MT fraction" in out2[0]


def test_cli_analyze_rejects_invalid_met_site(monkeypatch, tmp_path):
    """CLI-level validation: analyze() should raise ValueError on
    an unknown --met-site value rather than silently ignoring it."""
    from pirlygenes import cli as cli_mod

    monkeypatch.setattr(cli_mod, "load_expression_data", lambda *a, **k: pd.DataFrame({"x": [1]}))
    out_dir = str(tmp_path / "out")
    with pytest.raises(ValueError):
        cli_mod.analyze(
            "input.csv",
            output_dir=out_dir,
            met_site="not_a_site",
        )
