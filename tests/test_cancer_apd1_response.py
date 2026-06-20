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
    "evidence_type",
    "therapy_regimen_class",
    "histology_match",
    "is_direct_cancer_code_evidence",
    "endpoint_population",
    "response_numerator",
    "response_denominator",
    "source_anchor",
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
    """Each row's drug must match its ``drug_target`` class:
      - ``PD-1``        — anti-PD-1 monotherapy (pembrolizumab/nivolumab/cemiplimab).
      - ``PD-L1``       — anti-PD-L1 *proxy*, used only where no anti-PD-1 ORR exists.
      - ``PD-1+CTLA-4`` — dual checkpoint *fallback* (nivolumab+ipilimumab), used
                          where no single-agent anti-PD-1/PD-L1 ORR exists.
    Both fallback classes (PD-L1, PD-1+CTLA-4) must carry a citation."""
    df = cancer_apd1_response_df()
    assert set(df["drug_target"]) <= {"PD-1", "PD-L1", "PD-1+CTLA-4"}
    pd1 = df[df["drug_target"] == "PD-1"]
    pdl1 = df[df["drug_target"] == "PD-L1"]
    dual = df[df["drug_target"] == "PD-1+CTLA-4"]
    assert set(pd1["drug"]).issubset({"pembrolizumab", "nivolumab", "cemiplimab"})
    assert set(pdl1["drug"]).issubset({"atezolizumab", "durvalumab", "avelumab"})
    assert set(dual["drug"]).issubset({"nivolumab+ipilimumab"})
    # fallback classes must carry a citation (no unsourced proxies)
    for fallback in (pdl1, dual):
        assert fallback["pmid_doi"].astype(str).str.startswith(("PMID", "DOI", "10.")).all()


def test_response_evidence_provenance_is_structured():
    df = cancer_apd1_response_df()
    assert df["endpoint_population"].astype(str).str.len().gt(0).all()
    assert set(df["therapy_regimen_class"]) <= {
        "single_agent_pd1",
        "single_agent_pdl1_proxy",
        "dual_checkpoint_proxy",
    }
    direct_values = set(
        df["is_direct_cancer_code_evidence"].astype(str).str.lower()
    )
    assert direct_values <= {"true", "false"}

    proxy = df[df["cancer_code"].isin(["GBC", "NEC_LUNG_LARGECELL", "ACINIC"])]
    assert set(proxy["histology_match"]) == {"proxy_or_pooled"}

    sarc = df.set_index("cancer_code")
    assert sarc.loc["SARC_UPS", "apd1_orr_pct"] == 40.0
    assert sarc.loc["SARC_DDLPS", "apd1_orr_pct"] == 20.0
    assert sarc.loc["SARC_CHON", "apd1_orr_pct"] == 20.0
    assert str(
        sarc.loc["SARC_DDLPS", "is_direct_cancer_code_evidence"]
    ).lower() == "false"


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
