"""Tests for the curated anti-PD-1 monotherapy ORR reference (cancer-apd1-response.csv)."""

import math

from pirlygenes.gene_sets_cancer import (
    cancer_apd1_response,
    cancer_apd1_response_df,
    resolve_cancer_type,
)

_EXPECTED_COLS = [
    "cancer_code",
    "apd1_orr_pct",
    "drug",
    "trial",
    "setting",
    "pmid_doi",
    "confidence",
    "notes",
    "drug_target",
]


def test_schema_and_unique_codes():
    df = cancer_apd1_response_df()
    assert list(df.columns) == _EXPECTED_COLS
    codes = df["cancer_code"].astype(str)
    assert codes.is_unique
    for code in codes:                      # every code must be a real registry code
        assert resolve_cancer_type(code) is not None


def test_values_are_response_rates():
    df = cancer_apd1_response_df()
    vals = df.dropna(subset=["apd1_orr_pct"])["apd1_orr_pct"].astype(float)
    assert (vals >= 0).all() and (vals <= 100).all()
    # Hodgkin / Merkel are the high responders; the immune-cold anchors are low.
    m = cancer_apd1_response()
    assert m["HL"] > 50 and m["NEC_MERKEL"] > 40
    assert m["GBM"] < 15 and m["PRAD"] < 15


def test_drug_matches_its_checkpoint_target():
    """Anti-PD-1 rows use pembrolizumab/nivolumab; PD-L1 *proxy* rows (used only
    where no anti-PD-1 monotherapy ORR exists, and denoted by
    ``drug_target='PD-L1'``) use an anti-PD-L1 agent."""
    df = cancer_apd1_response_df()
    assert set(df["drug_target"]) <= {"PD-1", "PD-L1"}
    pd1 = df[df["drug_target"] == "PD-1"]
    pdl1 = df[df["drug_target"] == "PD-L1"]
    assert set(pd1["drug"]).issubset({"pembrolizumab", "nivolumab"})
    assert set(pdl1["drug"]).issubset({"atezolizumab", "durvalumab", "avelumab"})
    # PD-L1 proxies must carry a citation (no unsourced proxies)
    assert pdl1["pmid_doi"].astype(str).str.startswith(("PMID", "DOI", "10.")).all()


def test_every_value_is_cited_or_flagged():
    """A value with confidence high/medium carries a source PMID/DOI; the only
    citation-light rows are explicitly low-confidence anchors."""
    df = cancer_apd1_response_df()
    for row in df.itertuples():
        assert row.confidence in {"high", "medium", "low"}
        has_pmid = isinstance(row.pmid_doi, str) and row.pmid_doi.strip().startswith(
            ("PMID", "DOI", "10.")
        )
        if row.confidence in {"high", "medium"}:
            assert has_pmid, f"{row.cancer_code}: {row.confidence} row needs a PMID/DOI"


def test_accessor_resolves_aliases_and_map():
    assert cancer_apd1_response("melanoma") == cancer_apd1_response("SKCM")
    m = cancer_apd1_response()
    assert isinstance(m, dict) and "SKCM" in m and "NEC_MERKEL" in m


def test_subtype_inherits_parent_orr():
    # subtypes with no curated row inherit the parent (SCLC_ASCL1 -> SCLC,
    # LUAD_KRAS -> LUAD); the STK11 immune-cold subtype gets its own lower row.
    assert cancer_apd1_response("SCLC_ASCL1") == cancer_apd1_response("SCLC")
    assert cancer_apd1_response("LUAD_KRAS") == cancer_apd1_response("LUAD")
    assert cancer_apd1_response("LUAD_STK11") < cancer_apd1_response("LUAD")
    # strict (no inherit) returns None for an uncurated subtype
    assert cancer_apd1_response("SCLC_ASCL1", inherit=False) is None
