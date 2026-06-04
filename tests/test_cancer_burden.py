"""Tests for the curated cancer-burden reference (incidence + mortality)."""

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


def test_code_map_covers_coverage_cohorts():
    m = cancer_code_burden_map()
    # representative codes map to the expected broad categories
    assert m["LUAD"] == "lung" and m["LUSC"] == "lung"
    assert m["COAD"] == "colorectal" and m["READ"] == "colorectal"
    assert m["OS"] == "soft_tissue_and_bone_sarcoma"
    assert m["DLBC"] == "non_hodgkin_lymphoma"
    # every mapped category exists in the burden table
    cats = set(cancer_burden_df()["burden_category"])
    assert set(m.values()) <= cats


def test_burden_category_robust_resolution():
    # explicit code
    assert burden_category("LUAD") == "lung"
    # alias + display name (synonym space — don't give up on one code)
    assert burden_category("melanoma") == "melanoma"
    assert burden_category("Lung Adenocarcinoma") == "lung"
    # subtype -> parent chain
    assert burden_category("BRCA_LumA") == "breast"
    # primary_tissue fallback (not in explicit map)
    assert burden_category("PANNET") == "pancreas"
    # family fallback
    assert burden_category("MM") == "multiple_myeloma"
    # unresolvable input returns None (no crash) rather than dropping silently
    assert burden_category("not-a-cancer-xyz") is None
