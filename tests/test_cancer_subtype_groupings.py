"""Tests for the orthogonal cross-cutting subtype groupings
(cancer-subtype-groupings.csv + the cancer_subtype_group accessor).

These groupings cut ACROSS the organ parent_code tree (a leaf belongs to its
organ parent AND to a mechanism group like MSI), so the same four colorectal
leaves can be grouped two ways: by organ (COAD/READ via parent_code) and by
microsatellite status (MSI/MSS via the grouping).
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
    """MSI/MSS pull subtypes from every cancer that has them, across organs."""
    assert set(cancer_subtype_group("MSI")) == {"COAD_MSI", "READ_MSI", "UCEC_MSI"}
    assert set(cancer_subtype_group("MSS")) == {"COAD_MSS", "READ_MSS"}


def test_under_restricts_to_hierarchy_descendants():
    """``under`` intersects a group with a hierarchy node — the colorectal MSI
    cross-cut is exactly COAD_MSI + READ_MSI (not the endometrial UCEC_MSI)."""
    assert set(cancer_subtype_group("MSI", under="CRC")) == {"COAD_MSI", "READ_MSI"}
    assert set(cancer_subtype_group("MSS", under="CRC")) == {"COAD_MSS", "READ_MSS"}
    assert "UCEC_MSI" not in cancer_subtype_group("MSI", under="CRC")


def test_other_mechanism_axes_present():
    assert set(cancer_subtype_group("HPV_POS")) == {"HNSC_HPVpos", "CESC"}
    assert cancer_subtype_group("MYCN_AMP") == ["NBL_MYCNamp"]
    assert cancer_subtype_group("POLE") == ["UCEC_POLE"]


def test_two_orthogonal_groupings_of_the_four_colorectal_leaves():
    """The four CRC leaves group one way by organ (parent_code) and another by
    microsatellite status (grouping) — the dual axis."""
    assert set(ct.subtypes_of("COAD")) == {"COAD_MSI", "COAD_MSS"}
    assert set(ct.subtypes_of("READ")) == {"READ_MSI", "READ_MSS"}
    by_status = set(cancer_subtype_group("MSI", under="CRC")) | set(
        cancer_subtype_group("MSS", under="CRC"))
    by_organ = set(ct.subtypes_of("COAD")) | set(ct.subtypes_of("READ"))
    assert by_status == by_organ == {"COAD_MSI", "COAD_MSS", "READ_MSI", "READ_MSS"}
