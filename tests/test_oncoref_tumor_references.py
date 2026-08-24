"""Compatibility contract for oncoref's canonical tumor references (#601)."""

import numpy as np
import oncoref
from oncoref.version import DATA_VERSION as ONCOREF_DATA_VERSION

from pirlygenes import gene_sets_cancer as gsc
from pirlygenes.expression import available_cancer_expression_references


def test_pinned_oncoref_exposes_canonical_tumor_reference_apis():
    assert oncoref.__version__ == "1.8.182"
    assert ONCOREF_DATA_VERSION == "5.23.22"

    tcga = oncoref.tcga_deconvolved_expression("ACC")
    assert not tcga.empty
    assert set(tcga["cancer_code"]) == {"ACC"}
    assert np.isclose(tcga["tumor_tpm_median"].sum(), 1_000_000.0)
    assert tcga.attrs["oncoref"] == {
        "dataset": "tcga-deconvolved-expression",
        "data_version": "5.23.22",
        "scale": "classifier_tpm",
        "derivation_method": "tme_deconvolution",
        "derivation_scope": "dataset",
        "provenance_dataset": "tumor-reference-expression-provenance",
    }

    beataml = oncoref.subtype_tumor_reference_expression(
        "LAML",
        subtype_code="LAML_APL",
        source_cohort="BEATAML_OHSU_2022",
    )
    assert not beataml.empty
    assert set(beataml["cancer_code"]) == {"LAML"}
    assert set(beataml["subtype"]) == {"LAML_APL"}
    assert set(beataml["source_cohort"]) == {"BEATAML_OHSU_2022"}
    assert np.isclose(beataml["tumor_tpm_median"].sum(), 1_000_000.0)

    provenance = oncoref.tumor_reference_expression_provenance(
        artifact="subtype-deconvolved-expression",
        cancer_code="LAML",
        source_cohort="BEATAML_OHSU_2022",
    )
    apl = provenance[provenance["subtype"].eq("LAML_APL")]
    assert len(apl) == 1
    assert apl.iloc[0]["derivation_method"] == "high_purity_passthrough"
    assert apl.iloc[0]["source_scale"] == "clean_tpm_16_9_75"
    assert apl.iloc[0]["sample_qc_policy"] == "pass"


def test_latest_owner_references_are_exposed_through_pirlygenes():
    available = available_cancer_expression_references()
    latest = available.loc[
        available["cancer_code"].astype(str).isin({"CRANIO", "DIPG", "EPN"})
    ]

    observed = {
        (str(row.cancer_code), str(row.source_cohort), int(row.n_samples))
        for row in latest.itertuples(index=False)
    }
    assert {
        ("CRANIO", "OPENPBTA_V23_CBTN_CRANIO", 29),
        ("DIPG", "OPENPBTA_V23_DIPG_H3K27", 32),
        ("EPN", "GSE141460_GOJO_2020_EPN", 11),
    } <= observed


def test_mmnst_directional_panel_survives_owner_upgrade():
    high = gsc.lineage_gene_symbols("SARC_MMNST")
    low = gsc.lineage_gene_symbols("SARC_MMNST", direction="low")

    assert high == ["TYR", "PMEL", "MLANA", "DCT", "MITF", "SOX10", "S100B"]
    assert low == ["PMP22", "PMP2", "MPZ", "PRKAR1A"]
    assert set(high).isdisjoint(low)


def test_owner_aggregate_availability_requires_complete_member_unions():
    expected = {
        "BTC": [
            ("CHOL", "aggregate_member", True, ""),
            ("GBC", "aggregate_member", True, ""),
        ],
        "SGC": [
            ("ACINIC", "aggregate_member", True, ""),
            ("ADCC", "aggregate_member", True, ""),
        ],
        "NSCLC": [
            ("LUAD", "aggregate_member", True, ""),
            ("LUSC", "aggregate_member", True, ""),
        ],
    }

    for code, records in expected.items():
        availability = oncoref.cancer_reference_expression_availability(
            code,
            normalize="tpm_clean",
        )
        observed = list(
            availability[
                ["cancer_code", "request_kind", "available", "missing_reason"]
            ].itertuples(index=False, name=None)
        )
        assert observed == records
