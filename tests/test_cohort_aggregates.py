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


def test_lps_and_tcga_rollups():
    assert "SARC_DDLPS" in cohort_aggregate_members("SARC_LPS")
    # TCGA-SARC source rollup includes the leiomyosarcoma slice (code "SARC")
    assert "SARC" in cohort_aggregate_members("TCGA_SARC")


def test_pan_sarcoma_is_family_union_excluding_aggregates():
    pan = cohort_aggregate_members("SARC_PAN")
    lineage = set(sarcoma_lineage_codes())
    aggregates = set(cohort_aggregates()) - {"SARC_PAN"}
    # every pan member is a real sarcoma atom, and no aggregate leaks in
    assert set(pan) <= lineage
    assert not (set(pan) & aggregates)
    # the curated rollup members themselves are atoms in the pan union
    assert "SARC_RMS_ERMS" in pan and "SARC_LMS" in pan


def test_non_aggregate_returns_none():
    assert cohort_aggregate_members("GBM") is None
    assert cohort_aggregate_members("not-a-code") is None
