"""Exact IMT, DFSP, and PEComa therapy-panel contracts (#606)."""

from itertools import product

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    cancer_fusions,
    cancer_key_genes_cancer_types,
    cancer_therapy_targets,
)


def test_exact_sarcoma_panels_are_advertised_as_available():
    available = set(cancer_key_genes_cancer_types())

    assert {"SARC_IMT", "SARC_DFSP", "SARC_PEC"} <= available


def test_direct_gist_lookup_preserves_registry_subtype_mapping():
    direct = cancer_therapy_targets("SARC_GIST")
    historical = cancer_therapy_targets("SARC", subtype="gist")

    assert not direct.empty
    pd.testing.assert_frame_equal(direct, historical)


def test_imt_panel_is_exact_and_molecular_options_are_alteration_gated():
    panel = cancer_therapy_targets("SARC_IMT")

    assert not panel.empty
    assert set(panel["cancer_code"]) == {"SARC_IMT"}
    assert set(panel["agent"]) == {
        "crizotinib",
        "larotrectinib",
        "entrectinib",
        "repotrectinib",
    }
    assert panel["requires_verified_alteration"].all()
    assert set(panel["eligibility_basis"]) == {
        "histology_and_alteration",
        "tumor_agnostic_alteration",
    }
    assert panel["eligibility_note"].str.contains(
        "RNA abundance alone is not eligibility evidence", regex=False
    ).all()

    sibling_agents = {
        "imatinib",
        "sunitinib",
        "regorafenib",
        "ripretinib",
        "avapritinib",
        "trabectedin",
        "pazopanib",
    }
    assert set(panel["agent"]).isdisjoint(sibling_agents)


def test_imt_ntrk_panel_covers_every_gene_and_approved_agent():
    panel = cancer_therapy_targets("SARC_IMT")
    ntrk = panel[panel["symbol"].isin({"NTRK1", "NTRK2", "NTRK3"})]

    expected = set(
        product(
            {"NTRK1", "NTRK2", "NTRK3"},
            {"larotrectinib", "entrectinib", "repotrectinib"},
        )
    )
    assert set(zip(ntrk["symbol"], ntrk["agent"])) == expected
    assert set(ntrk["eligibility_basis"]) == {"tumor_agnostic_alteration"}
    assert ntrk["indication"].str.contains("tumor-agnostic", regex=False).all()


def test_imt_does_not_promote_case_report_kinases_to_named_therapies():
    panel = cancer_therapy_targets("SARC_IMT")

    assert set(panel["symbol"]).isdisjoint({"ROS1", "RET", "PDGFRB"})


def test_dfsp_panel_is_histology_gated_and_links_to_col1a1_pdgfb():
    panel = cancer_therapy_targets("SARC_DFSP")

    assert list(panel["agent"]) == ["imatinib"]
    row = panel.iloc[0]
    assert row["symbol"] == "PDGFB"
    assert row["eligibility_basis"] == "histology"
    assert not bool(row["requires_verified_alteration"])
    assert "not required by the FDA label" in row["eligibility_note"]

    fusions = cancer_fusions("SARC_DFSP")
    match = fusions[
        fusions["gene_5prime"].eq("COL1A1")
        & fusions["gene_3prime"].eq("PDGFB")
    ]
    assert len(match) == 1
    assert match.iloc[0]["fusion_family"] == "COL1A1-PDGFB"


def test_pecoma_panel_is_histology_gated_without_molecular_expression_gate():
    panel = cancer_therapy_targets("SARC_PEC")

    assert list(panel["agent"]) == ["nab-sirolimus (Fyarro)"]
    row = panel.iloc[0]
    assert pd.isna(row["symbol"])
    assert row["eligibility_basis"] == "histology"
    assert not bool(row["requires_verified_alteration"])
    assert "not label requirements or expression gates" in row["eligibility_note"]
    assert "TSC1" in row["eligibility_note"]
    assert "TSC2" in row["eligibility_note"]


def test_new_sarcoma_panels_have_structured_fda_provenance():
    for code in ("SARC_IMT", "SARC_DFSP", "SARC_PEC"):
        panel = cancer_therapy_targets(code)
        assert panel["source"].str.fullmatch(r"FDA_LABEL:[A-Z0-9_]+").all()
        assert panel["eligibility_basis"].notna().all()
        assert panel["requires_verified_alteration"].notna().all()
