"""Tests for computed cohort aggregates (SARC_RMS / SARC_LPS / TCGA_SARC + the
pan-sarcoma grand union)."""

from pirlygenes.gene_sets_cancer import (
    cohort_aggregate_members,
    cohort_aggregates,
    cohort_aggregates_df,
    sarcoma_lineage_codes,
)


def test_schema():
    df = cohort_aggregates_df()
    for col in ["aggregate_code", "member_code", "basis"]:
        assert col in df.columns
    assert len(df) > 0


def test_rms_rollup_pools_the_four_subtypes():
    m = cohort_aggregate_members("SARC_RMS")
    assert set(m) == {"SARC_RMS_ERMS", "SARC_RMS_ARMS", "SARC_RMS_PRMS",
                      "SARC_RMS_SSRMS"}


def test_lps_rollup():
    assert "SARC_DDLPS" in cohort_aggregate_members("SARC_LPS")


def test_sarc_is_pan_sarcoma_union_excluding_aggregates_and_self():
    # The bare SARC code resolves to the pan-sarcoma grand union (its TCGA-SARC
    # leiomyosarcoma samples are already folded into SARC_LMS).
    pan = cohort_aggregate_members("SARC")
    lineage = set(sarcoma_lineage_codes())
    aggregates = set(cohort_aggregates())
    # every pan member is a real sarcoma atom; no aggregate (incl SARC) leaks in
    assert set(pan) <= lineage
    assert not (set(pan) & aggregates)
    assert "SARC" not in pan  # no self-membership / circularity
    # the curated rollup members themselves are atoms in the pan union
    assert "SARC_RMS_ERMS" in pan and "SARC_LMS" in pan


def test_non_aggregate_returns_none():
    assert cohort_aggregate_members("GBM") is None
    assert cohort_aggregate_members("not-a-code") is None
