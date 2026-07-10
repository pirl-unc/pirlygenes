"""Delegation-surface tests for the anti-PD-1 monotherapy ORR reference.

The curated table is now owned by oncoref and re-exported through
``cancer_apd1_response`` / ``cancer_apd1_response_df`` (pirlygenes#541). These
tests exercise the pirlygenes *delegation contract* — the accessor resolves
aliases, walks parent inheritance, omits/returns sane values, and surfaces the
columns pirlygenes consumes — rather than pinning oncoref's exact schema or
per-code curated ORRs (which oncoref validates upstream and re-curates on its
own release cadence)."""

from pirlygenes.gene_sets_cancer import (
    cancer_apd1_response,
    cancer_apd1_response_df,
    resolve_cancer_type,
)

# The columns pirlygenes' own analyses/plots read off the re-exported frame.
# oncoref may add columns; it must not drop these.
_REQUIRED_COLS = {
    "cancer_code",
    "apd1_orr_pct",
    "drug",
    "pmid_doi",
    "confidence",
}


def test_schema_and_unique_codes():
    df = cancer_apd1_response_df()
    assert _REQUIRED_COLS <= set(df.columns)
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


# The drug↔checkpoint-target taxonomy (PD-1/PD-L1/PD-1+CTLA-4 fallback classes,
# drug-name spellings) and the per-code evidence provenance (endpoint_population,
# therapy_regimen_class, histology_match, per-code ORRs like SARC_UPS) are
# oncoref's curation contract and are validated in oncoref (pirlygenes#541). We
# keep only the delegation-surface + biology-sanity checks here.


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
