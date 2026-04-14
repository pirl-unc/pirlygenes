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
