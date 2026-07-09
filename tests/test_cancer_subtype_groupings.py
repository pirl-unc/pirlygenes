"""Tests for the orthogonal cross-cutting subtype groupings
(oncoref-delegated cancer_subtype_groupings + the cancer_subtype_group
accessor).

These groupings cut ACROSS the organ parent_code tree (a leaf belongs to its
organ parent AND to a mechanism group like MSI), so the same four colorectal
leaves can be grouped two ways: by organ (COAD/READ via parent_code) and by
microsatellite status (MSI/MSS via the grouping).

As of oncoref 1.8.95 this table is owned by oncoref (pirlygenes re-exports it
via load_dataset; oncoref#325) — a lossless superset of the former local CSV
that also covers the gastric (STAD_*) subtypes, the CRC_MSI pooled node, and
an EBV axis.
"""

from pirlygenes import cancer_types as ct
from pirlygenes.gene_sets_cancer import (
    cancer_subtype_group,
    cancer_subtype_groupings,
    resolve_cancer_type,
)


def test_schema_and_members_are_registry_codes():
    df = cancer_subtype_groupings()
    assert list(df.columns) == ["group_code", "axis", "member_code", "basis"]
    for code in df["member_code"]:
        assert resolve_cancer_type(code) is not None  # every member is real


def test_msi_mss_cross_cut_all_cancers():
    """MSI/MSS pull subtypes from every cancer that has them, across organs
    (oncoref's set spans colorectal, endometrial, and gastric, plus the
    CRC-level CRC_MSI pooled node)."""
    assert set(cancer_subtype_group("MSI")) == {
        "COAD_MSI", "READ_MSI", "UCEC_MSI", "STAD_MSI", "CRC_MSI"}
    assert set(cancer_subtype_group("MSS")) == {
        "COAD_MSS", "READ_MSS", "STAD_CIN", "STAD_GS"}


def test_under_restricts_to_hierarchy_descendants():
    """``under`` intersects a group with a hierarchy node — the colorectal MSI
    cross-cut is exactly COAD_MSI + READ_MSI (not the endometrial UCEC_MSI)."""
    # CRC-level MSI descendants: the two organ leaves plus the CRC_MSI pooled
    # node (itself parented under CRC as a molecular subtype).
    assert set(cancer_subtype_group("MSI", under="CRC")) == {
        "COAD_MSI", "READ_MSI", "CRC_MSI"}
    assert set(cancer_subtype_group("MSS", under="CRC")) == {"COAD_MSS", "READ_MSS"}
    assert "UCEC_MSI" not in cancer_subtype_group("MSI", under="CRC")
    assert "STAD_MSI" not in cancer_subtype_group("MSI", under="CRC")


def test_other_mechanism_axes_present():
    assert set(cancer_subtype_group("HPV_POS")) == {"HNSC_HPVpos", "CESC"}
    assert cancer_subtype_group("MYCN_AMP") == ["NBL_MYCNamp"]
    assert cancer_subtype_group("POLE") == ["UCEC_POLE"]
    # oncoref's set adds the gastric EBV axis the former local CSV lacked.
    assert set(cancer_subtype_group("EBV_POS")) == {"STAD_EBV"}


def test_two_orthogonal_groupings_of_the_four_colorectal_leaves():
    """The four CRC leaves group one way by organ (parent_code) and another by
    microsatellite status (grouping) — the dual axis."""
    assert set(ct.subtypes_of("COAD")) == {"COAD_MSI", "COAD_MSS"}
    assert set(ct.subtypes_of("READ")) == {"READ_MSI", "READ_MSS"}
    by_status = set(cancer_subtype_group("MSI", under="CRC")) | set(
        cancer_subtype_group("MSS", under="CRC"))
    by_organ = set(ct.subtypes_of("COAD")) | set(ct.subtypes_of("READ"))
    assert by_organ == {"COAD_MSI", "COAD_MSS", "READ_MSI", "READ_MSS"}
    # the four organ leaves are reachable by both axes; the status axis
    # additionally surfaces the CRC-level CRC_MSI pooled node.
    assert by_organ <= by_status
    assert by_status - by_organ == {"CRC_MSI"}
