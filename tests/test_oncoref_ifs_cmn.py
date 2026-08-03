"""CMN/IFS owner-reference compatibility and provenance contracts."""

from __future__ import annotations

import oncoref

from pirlygenes.expression import available_cancer_expression_references
from pirlygenes.gene_sets_cancer import cancer_lineage_group


def test_cmn_and_ifs_reference_sources_remain_physically_separate():
    available = available_cancer_expression_references()
    selected = available.loc[
        available["cancer_code"].astype(str).isin({"CMN", "SARC_IFS"})
    ]
    sources = {
        code: set(rows["source_cohort"].astype(str))
        for code, rows in selected.groupby("cancer_code")
    }

    assert sources["CMN"] == {
        "GSE11482_GADD_2010_CMN",
        "TREEHOUSE_RIBOD_25_01",
    }
    assert sources["SARC_IFS"] == {
        "TREEHOUSE_POLYA_25_01",
        "TREEHOUSE_RIBOD_25_01",
    }
    counts = selected.set_index(["cancer_code", "source_cohort"])["n_samples"]
    assert int(counts.loc[("CMN", "GSE11482_GADD_2010_CMN")]) == 12
    assert int(counts.loc[("SARC_IFS", "TREEHOUSE_POLYA_25_01")]) == 2
    assert int(counts.loc[("SARC_IFS", "TREEHOUSE_RIBOD_25_01")]) == 3


def test_cmn_is_owner_marked_validation_only_and_shares_sarcoma_lineage():
    registry = oncoref.cancer_type_registry().set_index("code")

    assert bool(registry.loc["CMN", "is_classification_target"]) is False
    assert registry.loc["CMN", "source_cohort"] == "GSE11482_GADD_2010_CMN"
    assert cancer_lineage_group("CMN") == "Sarcoma"
    assert cancer_lineage_group("SARC_IFS") == "Sarcoma"


def test_diagnosis_only_expression_does_not_fabricate_sample_drivers():
    for code in ("CMN", "SARC_IFS"):
        provenance = oncoref.molecular_provenance_for_cancer_code(code)
        public_expression = provenance.loc[
            provenance["access_level"].astype(str).eq("public")
            & provenance["expression_available"]
        ]
        assert not public_expression.empty
        assert public_expression["driver_event"].isna().all()
        assert set(public_expression["molecular_status"]) == {"unknown"}


def test_cmn_and_ifs_driver_spectra_are_related_but_not_interchangeable():
    cmn = set(oncoref.cancer_driver_spectrum("CMN")["driver_event"])
    ifs = set(oncoref.cancer_driver_spectrum("SARC_IFS")["driver_event"])

    assert "EGFR kinase-domain ITD" in cmn
    assert "EGFR kinase-domain ITD" not in ifs
    assert "EML4-NTRK3" in ifs
    assert "EML4-NTRK3" not in cmn
    assert "ETV6-NTRK3" in cmn & ifs
