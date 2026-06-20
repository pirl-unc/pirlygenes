import pandas as pd

from pirlygenes.gene_sets_cancer import (
    THERAPY_BENEFIT_TIERS,
    THERAPY_TOXICITY_TIERS,
    cancer_therapy_targets,
    therapy_benefit_toxicity_evidence,
)


def test_therapy_benefit_toxicity_schema_and_tiers():
    df = therapy_benefit_toxicity_evidence()
    assert not df.empty

    required = {
        "agent",
        "agent_class",
        "target_symbol",
        "cancer_code",
        "subtype",
        "line_of_therapy",
        "endpoint_type",
        "endpoint_value",
        "benefit_tier",
        "toxicity_tier",
        "source_type",
        "source_id",
        "source_token",
        "source_url",
        "source_anchor",
        "evidence_transfer",
    }
    assert required.issubset(df.columns)
    assert set(df["benefit_tier"].dropna()) <= set(THERAPY_BENEFIT_TIERS)
    assert set(df["toxicity_tier"].dropna()) <= set(THERAPY_TOXICITY_TIERS)


def test_disease_matched_benefit_row_for_gist_imatinib():
    rows = therapy_benefit_toxicity_evidence(
        agent="imatinib",
        cancer_code="SARC",
        subtype="gist",
        line_of_therapy="standard_or_frontline",
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["benefit_tier"] == "major_survival"
    assert row["toxicity_tier"] == "moderate"

    targets = cancer_therapy_targets("SARC", subtype="gist")
    assert row["agent"] in set(targets["agent"].astype(str))


def test_subtype_only_filter_does_not_include_blank_subtype_rows():
    rows = therapy_benefit_toxicity_evidence(subtype="gist")

    assert len(rows) == 1
    assert rows.iloc[0]["agent"] == "imatinib"
    assert rows.iloc[0]["cancer_code"] == "SARC"
    assert rows.iloc[0]["subtype"] == "gist"


def test_high_toxicity_row_keeps_boxed_warning_context():
    targets = cancer_therapy_targets("SARC", subtype="synovial_sarcoma")
    target_agent = targets.loc[targets["symbol"] == "MAGE-A4", "agent"].iloc[0]

    rows = therapy_benefit_toxicity_evidence(
        agent=target_agent,
        cancer_code="SARC",
        subtype="synovial_sarcoma",
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["benefit_tier"] == "high_response"
    assert row["toxicity_tier"] == "high"
    assert "CRS" in str(row["boxed_warning"])


def test_cross_indication_rows_are_optional():
    default_rows = therapy_benefit_toxicity_evidence(
        agent="pembrolizumab",
        cancer_code="OV",
    )
    strict_rows = therapy_benefit_toxicity_evidence(
        agent="pembrolizumab",
        cancer_code="OV",
        include_transferred=False,
    )

    assert len(default_rows) == 1
    assert default_rows.iloc[0]["evidence_transfer"] == "cross_indication"
    assert strict_rows.empty


def test_postmarket_signal_rows_are_not_incidence_estimates():
    rows = therapy_benefit_toxicity_evidence(source_type="postmarket_signal")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["evidence_transfer"] == "safety_signal_only"
    assert pd.isna(row["grade3_plus_ae_rate"])
    assert pd.isna(row["discontinuation_rate"])


def test_source_tokens_are_structured():
    df = therapy_benefit_toxicity_evidence()
    assert df["source_token"].astype(str).str.len().gt(0).all()
    assert df["source_anchor"].astype(str).str.len().gt(0).all()
    assert df["source_url"].astype(str).str.startswith(("https://", "http://")).all()
