"""Typed diagnosis-level eligibility for NUT carcinoma therapies (#615)."""

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    THERAPY_ELIGIBILITY_BASES,
    cancer_therapy_targets,
    cancer_types_with_fusion,
    rare_cancer_fusion_rules_df,
    rare_cancer_rna_surrogate_rules_df,
)


def test_every_nutm_therapy_has_a_typed_eligibility_contract():
    panel = cancer_therapy_targets("NUTM")

    assert not panel.empty
    assert panel["eligibility_basis"].notna().all()
    assert set(panel["eligibility_basis"]) <= THERAPY_ELIGIBILITY_BASES
    assert panel["requires_verified_alteration"].notna().all()


def test_nutm_small_molecule_rows_require_diagnosis_not_target_gene_hits():
    panel = cancer_therapy_targets("NUTM")
    diagnosis_scoped = panel[panel["agent_class"].eq("small_molecule")]

    assert diagnosis_scoped["symbol"].isna().all()
    assert diagnosis_scoped["ensembl_gene_id"].isna().all()
    assert set(diagnosis_scoped["eligibility_basis"]) == {
        "confirmed_nut_carcinoma_diagnosis"
    }
    assert not diagnosis_scoped["requires_verified_alteration"].any()
    assert diagnosis_scoped["eligibility_note"].str.contains(
        "nuclear NUT IHC", regex=False
    ).all()
    assert diagnosis_scoped["eligibility_note"].str.contains(
        "suspected compatible NUTM1 fusion is a testing prompt pending confirmation",
        regex=False,
    ).all()
    assert diagnosis_scoped["eligibility_note"].str.contains(
        "not eligibility evidence", regex=False
    ).all()


def test_nutm1_partner_pairs_do_not_degrade_to_same_gene_matching():
    assert cancer_types_with_fusion(
        fusion="BRD4-NUTM1", defining_only=True
    ) == ["NUTM"]
    assert cancer_types_with_fusion(
        fusion="BRD3-NUTM1", defining_only=True
    ) == ["NUTM"]
    assert cancer_types_with_fusion(
        fusion="NSD3-NUTM1", defining_only=True
    ) == ["NUTM"]
    assert cancer_types_with_fusion(
        fusion="ZNF592-NUTM1", defining_only=True
    ) == ["NUTM"]
    assert cancer_types_with_fusion(
        fusion="BRD4-LINC00486", defining_only=True
    ) == []


def test_noncanonical_nutm_fusions_remain_partner_and_context_specific():
    rules = rare_cancer_fusion_rules_df().set_index("rule_id")

    for rule_id in (
        "cic_nutm1_non_nut",
        "yap1_nutm1_context_dependent",
        "mga_nutm1_non_nut",
        "mxd4_nutm1_non_nut",
        "nutm1_rearranged_uncertain",
    ):
        assert not bool(rules.loc[rule_id, "promote_report_scope"])
        assert pd.isna(rules.loc[rule_id, "cancer_code"])


def test_nutm2_fusions_route_to_endometrial_stromal_sarcoma_not_nutm():
    for fusion in ("YWHAE-NUTM2A", "YWHAE-NUTM2B"):
        assert cancer_types_with_fusion(
            fusion=fusion, defining_only=True
        ) == ["SARC_ESS_HG"]


def test_expression_only_nutm_evidence_cannot_become_therapy_eligibility():
    rules = rare_cancer_rna_surrogate_rules_df()
    nutm1 = rules[rules["rule_id"].eq("nutm_nutm1")].iloc[0]

    assert nutm1["evidence_role"] == "report_scope"
    assert bool(nutm1["promote_report_scope"])
    assert "Hypothesis only from gene-level RNA" in nutm1["caveat"]
    assert "fusion/IHC/FISH is needed" in nutm1["caveat"]
    assert not rules["primary_gene"].astype(str).str.startswith("NUTM2").any()


def test_nutm_tcr_rows_use_hla_and_antigen_expression_not_alterations():
    panel = cancer_therapy_targets("NUTM")
    tcr = panel[panel["agent_class"].eq("TCR-T")]

    assert set(tcr["symbol"]) == {"PRAME", "MAGEA4"}
    assert set(tcr["eligibility_basis"]) == {"hla_and_antigen_expression"}
    assert not tcr["requires_verified_alteration"].any()
    assert tcr["eligibility_note"].str.contains("HLA-A*02", regex=False).all()
    assert tcr["eligibility_note"].str.contains(
        "validated", case=False, regex=False
    ).all()
