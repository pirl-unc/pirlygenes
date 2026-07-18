"""The centralized cancer-types metadata sub-library (pirlygenes.cancer_types):
synonym resolution, viral/fusion status, tissue of origin, and the registry
biology columns that back them."""

import pandas as pd

from pirlygenes import cancer_types as ct
from pirlygenes.gene_sets_cancer import _RENAMED_CODE_ALIASES
from pirlygenes.load_dataset import get_data

_VIRAL_ETIOLOGIES = {"defining", "subset", "none"}
_FUSION_STATES = {"defining", "subtype", "rare", "none"}


def test_registry_has_biology_columns():
    reg = get_data("cancer-type-registry.csv")
    for col in ("viral_etiology", "viral_agent", "fusion_driven", "fusion_driver"):
        assert col in reg.columns


def test_biology_columns_use_controlled_vocab():
    reg = get_data("cancer-type-registry.csv")
    assert set(reg["viral_etiology"].astype(str)) <= _VIRAL_ETIOLOGIES
    assert set(reg["fusion_driven"].astype(str)) <= _FUSION_STATES
    # A non-none viral etiology must name an agent; 'none' must not.
    for r in reg.itertuples():
        if str(r.viral_etiology) != "none":
            assert isinstance(r.viral_agent, str) and r.viral_agent.strip()


def test_info_includes_biology_fields():
    info = ct.info("RMS_ARMS")  # resolves to SARC_RMS_ARMS
    assert info["code"] == "SARC_RMS_ARMS"
    assert {"viral_etiology", "viral_agent", "fusion_driven", "fusion_driver"} <= set(info)


def test_viral_status_known_entities():
    assert ct.viral_status("HNSC_HPVpos") == {"etiology": "defining", "agent": "HPV"}
    assert ct.viral_status("NPC") == {"etiology": "defining", "agent": "EBV"}
    assert ct.viral_status("NEC_MERKEL") == {"etiology": "defining", "agent": "MCPyV"}
    assert ct.viral_status("SARC_KS") == {"etiology": "defining", "agent": "HHV8"}
    assert ct.viral_status("LIHC")["etiology"] == "subset"
    # A type with no viral role.
    assert ct.viral_status("PRAD") == {"etiology": "none", "agent": ""}


def test_fusion_status_known_entities():
    ews = ct.fusion_status("SARC_EWS")
    assert ews["status"] == "defining" and "EWSR1-FLI1" in ews["driver"]
    syn = ct.fusion_status("SARC_SYN")
    assert syn["status"] == "defining" and "SS18-SSX1" in syn["driver"]
    # Recurrent-but-not-defining subtype (prostate TMPRSS2-ERG).
    pr = ct.fusion_status("prostate")
    assert pr["status"] == "subtype" and "TMPRSS2-ERG" in pr["driver"]
    # Complex-karyotype sarcoma: not fusion-driven.
    assert ct.fusion_status("SARC_LMS") == {"status": "none", "driver": ""}


def test_fusion_calls_consistent_with_cancer_fusions_table():
    """Every selectable fusion-defined target has detailed fusion evidence.

    oncoref's broader WHO ontology can include non-selectable summary entities
    before their partner-level rows are curated (oncoref#391). Pirlygenes must
    not expose an unsupported *classification target*, while it also must not
    require its compatibility detail table to mirror every ontology-only row.
    """
    fus = get_data("cancer-fusions")
    defining_in_table = {
        str(c)
        for c, d in fus.groupby("cancer_code")
        if d["is_defining"].astype(str).str.lower().isin(["true", "1", "yes"]).any()
    }
    reg = get_data("cancer-type-registry.csv")
    target = reg["is_classification_target"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    called_defining = set(
        reg.loc[
            reg["fusion_driven"].eq("defining") & target,
            "code",
        ].astype(str)
    )
    assert called_defining <= defining_in_table


def test_synonyms_reverse_resolution():
    syns = ct.synonyms("OS")  # canonical SARC_OS
    assert "OS" in syns  # the pre-rename code resolves here
    assert "SARC_OS" not in syns  # the canonical code itself is excluded
    # Every returned synonym resolves back to the same canonical code.
    for s in syns:
        assert ct.resolve(s) == "SARC_OS"
    assert ct.synonyms("not-a-cancer") == []


def test_alcl_curated_consistently():
    """ALCL (anaplastic large cell lymphoma) added as a registry entity, curated
    consistently across registry + markers + fusion tables."""
    info = ct.info("ALCL")
    assert info["family"] == "heme-tcell"
    assert info["primary_tissue"] == "lymph_node"
    assert info["fusion_driven"] == "subtype" and info["fusion_driver"] == "NPM1-ALK"
    assert info["burden_category"] == "non_hodgkin_lymphoma"
    # defining markers present (CD30=TNFRSF8, ALK)
    from pirlygenes.gene_sets_cancer import lineage_gene_symbols
    assert {"TNFRSF8", "ALK"} <= set(lineage_gene_symbols("ALCL"))
    # fusion detail table + reverse lookup agree with the registry
    assert "ALCL" in ct.with_fusion(partner="ALK")
    assert ct.with_fusion("NPM1-ALK") == ["ALCL"]


def test_info_is_json_serializable():
    """info() must be JSON-serializable — numpy scalars (e.g. numpy.bool_ for
    pediatric) coerced to native Python types (#307)."""
    import json
    info = ct.info("SARC_RMS_ARMS")
    json.dumps(info)  # raises TypeError if a numpy scalar leaked through
    assert isinstance(info["pediatric"], bool)
    assert info["tmb"] is None or isinstance(info["tmb"], float)


def test_resolve_strict_flag():
    """resolve(strict=False) is the non-raising lookup; strict=True (default)
    raises on unknown (#307)."""
    assert ct.resolve("not-a-cancer-type", strict=False) is None
    assert ct.resolve("", strict=False) is None
    import pytest
    with pytest.raises(ValueError):
        ct.resolve("not-a-cancer-type")


def test_no_annotations_namespace_leak():
    """`from __future__ import annotations` must not leak `annotations` into the
    package namespace (#307)."""
    assert not hasattr(ct, "annotations")


def test_fusion_subtype_not_defining_for_bulk_umbrellas():
    """Bulk umbrellas whose fusion defines a rare SUBTYPE are 'subtype', not
    'defining' (#308): BRCA (secretory), LIHC (fibrolamellar), LGG (pilocytic),
    LAML (CBF/KMT2A AML)."""
    for code in ("BRCA", "LIHC", "LGG", "LAML"):
        assert ct.fusion_status(code)["status"] == "subtype"
    # genuine fusion-defined entities stay 'defining'
    assert ct.fusion_status("SARC_EWS")["status"] == "defining"


def test_family_display_names():
    """Families expose human-readable display names (#309)."""
    assert ct.family_name("heme-bcell") == "B-cell neoplasm"
    assert ct.family_name("sarcoma") == "Sarcoma"
    fams = ct.families()
    assert len(fams) >= 20
    # every registry family resolves to a non-empty display label
    assert all(isinstance(v, str) and v for v in fams.values())
    # unknown family falls back to a de-slugged title, never raises
    assert ct.family_name("some-new-slug") == "Some new slug"


def test_reverse_fusion_lookup():
    # by exact fusion
    assert ct.with_fusion("EWSR1-FLI1") == ["SARC_EWS"]
    assert ct.with_fusion("EWSR1::FLI1") == ["SARC_EWS"]  # :: notation accepted
    # by partner gene (either end)
    by_ewsr1 = ct.with_fusion(partner="EWSR1")
    assert "SARC_EWS" in by_ewsr1 and "SARC_DSRCT" in by_ewsr1
    # by partner family
    fet = ct.with_fusion(partner_family="FET")
    assert "SARC_EWS" in fet and "SARC_DSRCT" in fet
    ets = ct.with_fusion(partner_family="ETS")
    assert {"PRAD", "SARC_EWS"} <= set(ets)
    # defining_only narrows to entities where the ALK fusion is defining
    # (inflammatory myofibroblastic tumor + ALK+ ALCL)
    assert ct.with_fusion(partner="ALK", defining_only=True) == ["ALCL", "SARC_IMT"]
    # returns canonical codes that round-trip through the registry
    for code in by_ewsr1:
        assert ct.resolve(code) == code

    import pytest
    with pytest.raises(ValueError):
        ct.with_fusion("EWSR1-FLI1", partner="EWSR1")  # exactly one selector


def test_synonym_input_flows_through_accessors():
    # Old/alias inputs resolve through every accessor.
    assert ct.fusion_status("MID_NET") == ct.fusion_status("NET_MIDGUT")
    assert ct.viral_status("CHOR") == ct.viral_status("SARC_CHOR")
    assert ct.tissue_of_origin("prostate") == ct.tissue_of_origin("PRAD")
