"""Compatibility tests for oncoref-owned cancer-burden references."""

import oncoref
import pandas as pd

from pirlygenes.load_dataset import get_data

from pirlygenes.gene_sets_cancer import (
    burden_category,
    cancer_burden,
    cancer_burden_df,
    cancer_code_burden_map,
)

_METRICS = ["us_incidence_pct", "us_mortality_pct",
            "world_incidence_pct", "world_mortality_pct"]


def test_schema():
    df = cancer_burden_df()
    for col in ["burden_category", *_METRICS, "source", "notes"]:
        assert col in df.columns
    assert df["burden_category"].is_unique


def test_owner_frame_and_richer_provenance_flow_through_unchanged():
    actual = cancer_burden_df()
    expected = oncoref.cancer_burden_df()

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(
        get_data("Cancer-Incidence-Mortality"),
        expected,
    )
    assert {
        "aggregation",
        "source_anchor",
        "us_incidence_count",
        "us_incidence_total",
        "us_source_locator",
        "derivation_basis",
    } <= set(actual.columns)
    assert actual["aggregation"].astype(str).str.strip().ne("").all()
    assert actual["source_anchor"].astype(str).str.strip().ne("").all()


def test_values_are_percent_shares():
    df = cancer_burden_df()
    for m in _METRICS:
        vals = df[m].astype(float)
        assert (vals >= 0).all() and (vals <= 100).all()


def test_accessor_and_metric_validation():
    inc = cancer_burden(metric="us_incidence_pct")
    assert isinstance(inc, dict) and inc.get("lung", 0) > 0
    # mortality:incidence diverges — pancreas/lung high, prostate/thyroid low
    assert cancer_burden("pancreas", metric="us_mortality_pct") > \
        cancer_burden("pancreas", metric="us_incidence_pct")
    assert cancer_burden("prostate", metric="us_mortality_pct") < \
        cancer_burden("prostate", metric="us_incidence_pct")
    try:
        cancer_burden("lung", metric="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_code_map_is_overrides_only():
    # The hand-map now carries only the registry-ontology exceptions; common
    # codes (LUAD, COAD, OS, ...) resolve from the registry, not this map.
    m = cancer_code_burden_map()
    assert m["SARC_KS"] == "kaposi_sarcoma"
    assert m["LAML"] == "leukemia_AML"
    assert m["HL"] == "hodgkin_lymphoma"
    assert "LUAD" not in m and "SARC_OS" not in m
    # every override category exists in the burden table
    cats = set(cancer_burden_df()["burden_category"])
    assert set(m.values()) <= cats

    assert m == oncoref.cancer_code_burden_map()


def test_burden_category_registry_driven():
    cats = set(cancer_burden_df()["burden_category"])
    # primary-tissue driven (no explicit map entry)
    assert burden_category("LUAD") == "lung"
    assert burden_category("Lung Adenocarcinoma") == "lung"
    assert burden_category("NET_PANCREAS") == "pancreas"   # NET -> its organ
    assert burden_category("BRCA_LumA") == "breast"  # subtype -> parent tissue
    # sarcoma family splits bone vs soft tissue on primary_tissue
    assert burden_category("SARC_OS") == "bone_and_joint"
    assert burden_category("SARC_EWS") == "bone_and_joint"
    assert burden_category("SARC_LMS") == "soft_tissue_sarcoma"
    # Kaposi / AML / Hodgkin are the curated overrides
    assert burden_category("SARC_KS") == "kaposi_sarcoma"
    assert burden_category("LAML") == "leukemia_AML"
    assert burden_category("HL") == "hodgkin_lymphoma"
    # plasma-cell by family; ALL is leukemia, not lymphoma
    assert burden_category("MM") == "multiple_myeloma"
    assert burden_category("B_ALL") == "leukemia_all_other"
    # every registry code resolves to a real burden category (no silent gaps)
    from pirlygenes.gene_sets_cancer import cancer_type_registry
    reg = cancer_type_registry()
    resolved = {c: burden_category(c) for c in reg["code"]}
    assert all(v is not None for v in resolved.values())
    assert set(resolved.values()) <= cats
    # unresolvable input returns None (no crash)
    assert burden_category("not-a-cancer-xyz") is None


def test_public_burden_helpers_are_owner_compatible():
    for metric in _METRICS:
        assert cancer_burden(metric=metric) == oncoref.cancer_burden(metric=metric)
    for query in ("SARC_OS", "HCL", "GCT_OV_YST", "not-a-cancer-xyz"):
        assert burden_category(query) == oncoref.burden_category(query)
