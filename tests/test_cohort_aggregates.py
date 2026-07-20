"""Tests for computed cohort aggregates (SARC_RMS / SARC_LPS + the pan-sarcoma
grand union)."""

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
    # oncoref's broader WHO ontology contains non-sample-bearing grouping nodes;
    # those must never become members of the expression union.
    from pirlygenes.gene_sets_cancer import cancer_type_registry

    registry = cancer_type_registry()
    grouping_nodes = set(
        registry.loc[
            registry["ontology_level"].astype(str).eq("grouping"),
            "code",
        ].astype(str)
    )
    assert not (set(pan) & grouping_nodes)


def test_sarc_tier_rollups_stay_in_lockstep_with_registry_children():
    """The curated SARC histology rollups must equal oncoref's registry
    parent_code tiers. Aligned exactly once oncoref #326 finished the
    intermediate-tier wiring (1.8.103: SARC_LPS/ESS/RMS are computed_union tiers
    with the leaves reparented under them). This guards against the curated CSV
    silently drifting from the registry — if oncoref reparents or adds a subtype,
    the rollup must be updated in lockstep.

    CRC is deliberately NOT checked here: its cohort-aggregate members are the
    organ atoms (COAD/READ), but the registry also parents the molecular subtype
    CRC_MSI under CRC, so registry-derivation would wrongly pull it into the
    organ rollup — the reason this stays a curated CSV, not a pure derivation.
    """
    from pirlygenes.gene_sets_cancer import cancer_type_registry

    reg = cancer_type_registry()
    children: dict[str, set] = {}
    for code, parent in zip(
        reg["code"].astype(str), reg["parent_code"].fillna("").astype(str)
    ):
        p = parent.strip()
        if p and p.lower() not in ("nan", "none"):
            children.setdefault(p, set()).add(code)
    for tier in ("SARC_LPS", "SARC_ESS", "SARC_RMS"):
        assert set(cohort_aggregate_members(tier)) == children.get(tier, set()), (
            f"{tier} rollup drifted from registry parent_code children"
        )


def test_non_aggregate_returns_none():
    assert cohort_aggregate_members("GBM") is None
    assert cohort_aggregate_members("not-a-code") is None
