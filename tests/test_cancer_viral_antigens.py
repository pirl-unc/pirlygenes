"""Curated cancer-viral-antigens reference (#282): targetable oncoviral
antigens for virally-driven cancers, complementing the registry
viral_etiology / viral_agent columns with antigen-level detail."""

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    cancer_viral_antigens,
    cancer_viral_antigens_df,
    viral_antigens_for_cancer,
)

_SCHEMA = {
    "virus",
    "integration_mode",
    "targetable_antigens",
    "associated_cohorts",
    "notes",
    "source",
    "association_source",
    "integration_source",
    "antigen_expression_source",
    "targetability_source",
    "support_scope",
}

_ASSERTION_SOURCE_COLUMNS = {
    "association_source",
    "integration_source",
    "antigen_expression_source",
    "targetability_source",
}


def test_schema_and_controlled_vocab():
    df = cancer_viral_antigens_df()
    assert _SCHEMA <= set(df.columns)
    assert len(df) >= 5
    assert set(df["integration_mode"].astype(str)) <= {"integrated", "episomal"}
    # every row is cited
    assert df["source"].astype(str).str.match(r"(PMID:|DOI:|https?:)").all()


def test_assertion_level_source_columns_are_populated():
    df = cancer_viral_antigens_df()
    assert df["support_scope"].astype(str).str.strip().ne("").all()
    has_assertion_source = df[list(_ASSERTION_SOURCE_COLUMNS)].fillna("").astype(str)
    assert has_assertion_source.apply(
        lambda row: any(value.strip() for value in row),
        axis=1,
    ).all()

    by_virus = df.set_index("virus")
    assert by_virus.loc["HPV", "targetability_source"] == "PMID:25823737"
    assert by_virus.loc["HBV", "integration_source"] == "PMID:22634756"
    assert "need separate" in by_virus.loc["HBV", "support_scope"]


def test_known_virus_antigens():
    assert cancer_viral_antigens("HPV") == ["E6", "E7"]
    assert cancer_viral_antigens("hpv") == ["E6", "E7"]  # case-insensitive
    assert cancer_viral_antigens("MCPyV") == ["LT", "sT"]
    assert set(cancer_viral_antigens("EBV")) == {"LMP1", "LMP2A", "EBNA1"}
    assert cancer_viral_antigens("not-a-virus") == []
    full = cancer_viral_antigens()
    assert "HHV8" in full and full["HHV8"] == ["LANA"]


def test_reverse_lookup_by_cancer_code():
    assert viral_antigens_for_cancer("CESC") == [("HPV", ["E6", "E7"])]
    assert viral_antigens_for_cancer("NEC_MERKEL") == [("MCPyV", ["LT", "sT"])]
    assert viral_antigens_for_cancer("SARC_KS") == [("HHV8", ["LANA"])]
    # a non-viral cancer has none
    assert viral_antigens_for_cancer("PRAD") == []
    # alias input resolves
    assert viral_antigens_for_cancer("prostate") == []


def test_associated_cohorts_are_registry_codes():
    reg = set(cancer_type_registry()["code"].astype(str))
    df = cancer_viral_antigens_df()
    for r in df.itertuples():
        raw = r.associated_cohorts
        if not isinstance(raw, str) or not raw.strip():
            continue  # HTLV-1 (ATLL not yet a registry code) is allowed empty
        for code in raw.split(";"):
            code = code.strip()
            if code and code.lower() != "nan":
                assert code in reg, f"{r.virus}: unknown cohort {code!r}"


def test_consistent_with_registry_viral_agents():
    """Every virus a registry cohort is flagged for must have an antigen row, so
    the two surfaces don't drift (a virally-driven cohort with no antigen detail
    would be a curation gap)."""
    reg = cancer_type_registry()
    table_viruses = {str(v).upper() for v in cancer_viral_antigens_df()["virus"]}
    # registry viral_agent can be 'HBV;HCV' etc.; HCV is RNA (no clonal antigen)
    # and intentionally not in the antigen table.
    agents = set()
    for v in reg.loc[reg["viral_etiology"].astype(str) != "none", "viral_agent"]:
        for a in str(v).split(";"):
            a = a.strip().upper()
            if a and a != "HCV":
                agents.add(a)
    missing = agents - table_viruses
    assert not missing, f"registry names viral agents with no antigen row: {missing}"
