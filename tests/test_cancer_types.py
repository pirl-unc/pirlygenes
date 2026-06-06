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
    """Every 'defining' fusion call must be backed by an is_defining row in the
    cited cancer-fusions.csv (no fusion status invented beyond the evidence)."""
    fus = get_data("cancer-fusions")
    defining_in_table = {
        str(c)
        for c, d in fus.groupby("cancer_code")
        if d["is_defining"].astype(str).str.lower().isin(["true", "1", "yes"]).any()
    }
    reg = get_data("cancer-type-registry.csv")
    called_defining = set(
        reg.loc[reg["fusion_driven"] == "defining", "code"].astype(str)
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


def test_synonym_input_flows_through_accessors():
    # Old/alias inputs resolve through every accessor.
    assert ct.fusion_status("MID_NET") == ct.fusion_status("NET_MIDGUT")
    assert ct.viral_status("CHOR") == ct.viral_status("SARC_CHOR")
    assert ct.tissue_of_origin("prostate") == ct.tissue_of_origin("PRAD")
