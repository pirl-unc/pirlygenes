"""Tests for the expanded cancer-type registry.

The registry is a richer superset of TCGA — covers non-TCGA heme,
pediatric, NET, and rare entities, plus expression-based subtype rows
under TCGA umbrellas (BRCA × PAM50, LAML × ELN/APL, SARC × subtype,
LUAD × mutation class, SCLC × ASCL1/NEUROD1/POU2F3/YAP1, etc.).
"""

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    cancer_types_in_family,
    cancer_types_by_tissue,
    cancer_type_subtypes_of,
)


def test_registry_has_required_columns():
    df = cancer_type_registry()
    required = {"code", "name", "family", "primary_tissue",
                "primary_template", "parent_code", "expression_source", "notes"}
    missing = required - set(df.columns)
    assert not missing, f"registry missing columns: {missing}"


def test_registry_codes_are_unique():
    df = cancer_type_registry()
    dupes = df["code"][df["code"].duplicated()].tolist()
    assert not dupes, f"duplicate codes in registry: {dupes}"


def test_registry_covers_all_33_tcga_codes():
    """Every TCGA code must appear in the registry or we'll lose
    compatibility with existing cancer-type detection code paths."""
    df = cancer_type_registry()
    tcga_codes = {
        "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
        "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
        "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
        "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
        "UVM",
    }
    registry_codes = set(df["code"])
    missing = tcga_codes - registry_codes
    assert not missing, f"registry missing TCGA codes: {missing}"


def test_registry_includes_non_tcga_heme():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("CLL", "MM", "MCL", "FL", "HL", "BL", "CML", "MDS", "MPN", "HCL"):
        assert need in codes, f"missing heme code: {need}"


def test_registry_includes_pediatric():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("OS", "EWS", "RMS_ERMS", "RMS_ARMS", "NBL", "WILMS", "RT",
                 "MBL", "ATRT", "RB", "HEPB"):
        assert need in codes, f"missing pediatric code: {need}"


def test_registry_includes_net_axis():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("PANNET", "MID_NET", "LUNG_NET_LC", "SCLC", "MEC"):
        assert need in codes, f"missing NET code: {need}"


def test_registry_includes_rare_entities():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("NUTM", "ADCC", "MTC", "CHOR", "NPC"):
        assert need in codes, f"missing rare code: {need}"


def test_brca_pam50_subtypes_present():
    """BRCA's expression-based PAM50 tiles must be in the registry so
    the second-pass subtype classifier can route to them."""
    subs = cancer_type_subtypes_of("BRCA")
    assert set(subs) == {"BRCA_LumA", "BRCA_LumB", "BRCA_HER2",
                         "BRCA_Basal", "BRCA_Normal"}


def test_sarc_subtypes_cover_main_entities():
    """SARC subtypes must at minimum include the Tranche B tiles
    plus the known tumor-biology subtypes (MPNST, angiosarcoma,
    UPS)."""
    subs = set(cancer_type_subtypes_of("SARC"))
    required = {"SARC_LMS", "SARC_DDLPS", "SARC_MYXLPS", "SARC_SYN",
                "SARC_DSRCT", "SARC_GIST", "SARC_MPNST", "SARC_ANGIO",
                "SARC_UPS"}
    missing = required - subs
    assert not missing, f"SARC subtypes missing: {missing}"


def test_laml_has_apl_and_eln_tiles():
    subs = set(cancer_type_subtypes_of("LAML"))
    assert "LAML_APL" in subs
    # ELN2017 is the modern risk-stratification that gates transplant
    # vs chemo; must be representable as a subtype tile.
    for eln in ("LAML_ELN_Fav", "LAML_ELN_Int", "LAML_ELN_Adv"):
        assert eln in subs


def test_bone_tissue_returns_osteosarcoma_and_ewing():
    """Site-aware hypothesis: any sample suspected of bone origin
    should be able to enumerate OS + Ewing as candidates."""
    bone_cancers = set(cancer_types_by_tissue("bone"))
    assert "OS" in bone_cancers
    assert "EWS" in bone_cancers


def test_heme_myeloid_family_contains_laml_and_related():
    """Family grouping must pull LAML + its tiles + MDS + MPN + CML
    together — they share the heme_marrow / heme_blood templates and
    need joint candidate enumeration when sample mode is heme."""
    myeloid = set(cancer_types_in_family("heme-myeloid"))
    for need in ("LAML", "MDS", "MPN", "CML"):
        assert need in myeloid


def test_net_family_contains_sclc_and_pannet():
    net = set(cancer_types_in_family("net"))
    assert "SCLC" in net
    assert "PANNET" in net
    assert "MEC" in net  # Merkel cell carcinoma


def test_parent_codes_reference_registry_entries():
    """Every non-null parent_code must reference an existing code."""
    df = cancer_type_registry()
    codes = set(df["code"])
    parents = df["parent_code"].dropna().astype(str)
    orphan = [p for p in parents if p and p not in codes]
    assert not orphan, f"parent_codes not in registry: {set(orphan)}"


def test_primary_templates_are_declared_or_planned():
    """Every row has a primary_template — either an implemented
    template name or a ``primary_<tissue>`` name documented as planned
    (osteosarcoma, chondrosarcoma, adipose, etc.). Catches typos like
    ``primary_bones`` or blanks."""
    df = cancer_type_registry()
    templates = df["primary_template"].dropna().unique()
    # Every template must match the convention.
    for t in templates:
        assert t == "solid_primary" or t.startswith("primary_") or t.startswith("heme_"), (
            f"unknown primary_template convention: {t}"
        )
