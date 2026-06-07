"""Unified cohort normalization-views object (#319): one object bundling the
tpm / clean_tpm / clean_tpm_biological stages so a consumer can't re-normalize
inconsistently."""

from pirlygenes.expression import (
    CohortExpressionViews,
    cohort_expression_views,
)


def test_views_bundle_three_stages_and_provenance():
    v = cohort_expression_views("CLL", genes=["MS4A1", "MALAT1", "RPL13A"])
    assert isinstance(v, CohortExpressionViews)
    for frame in (v.tpm, v.clean_tpm, v.clean_tpm_biological):
        assert {"Ensembl_Gene_ID", "Symbol"} <= set(frame.columns)
    # biological view drops the censored genes (MALAT1 technical, RPL13A ribo),
    # keeps real biology (MS4A1)
    bio = set(v.clean_tpm_biological["Symbol"])
    assert "MS4A1" in bio
    assert "MALAT1" not in bio and "RPL13A" not in bio
    # tpm/clean_tpm keep all requested genes (technical included)
    assert {"MS4A1", "MALAT1", "RPL13A"} <= set(v.clean_tpm["Symbol"])
    # provenance records the cohort + pipeline (native unit)
    assert "source_cohort" in v.provenance.columns
    assert "processing_pipeline" in v.provenance.columns
    assert len(v.provenance) >= 1


def test_views_clean_differs_from_tpm_for_technical_gene():
    """clean_tpm_v4 changes the technical gene's value vs plain TPM (the whole
    point of having both stages in one object)."""
    v = cohort_expression_views("CLL", genes=["MS4A1", "MALAT1"])
    tpm = dict(zip(v.tpm["Symbol"], v.tpm["CLL"]))
    clean = dict(zip(v.clean_tpm["Symbol"], v.clean_tpm["CLL"]))
    # MALAT1 (polyA-bias technical) is suppressed under clean_tpm_v4
    assert clean["MALAT1"] != tpm["MALAT1"]


def test_aggregate_code_expands_in_views():
    """An aggregate code (SARC) expands to its subtype cohorts in the views."""
    v = cohort_expression_views("SARC", genes=["TP53"])
    cohort_cols = [c for c in v.tpm.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert any(c.startswith("SARC_") for c in cohort_cols)
