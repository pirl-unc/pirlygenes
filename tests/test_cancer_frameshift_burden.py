"""Tests for the curated frameshift/indel-enrichment flag (cancer-frameshift-burden.csv).

This is an *ordinal mechanistic class* (0 baseline / 1 intermediate / 2 high),
not a measured per-Mb value: high = RCC lineage (Turajlic 2017, PMID 28694034)
+ dMMR/MSI-H (frameshift-at-microsatellites by definition). It backs the indel
antigen factor in analyses/apd1_causal_factors.py.
"""

from pirlygenes.gene_sets_cancer import resolve_cancer_type
from pirlygenes.load_dataset import get_data

_EXPECTED_COLS = [
    "cancer_code", "indel_class", "indel_score", "basis",
    "pmid_doi", "confidence", "notes",
]
_CLASS_TO_SCORE = {"baseline": 0, "intermediate": 1, "high": 2}


def _df():
    return get_data("cancer-frameshift-burden.csv")


def test_schema_unique_codes_resolve():
    df = _df()
    assert list(df.columns) == _EXPECTED_COLS
    codes = df["cancer_code"].astype(str)
    assert codes.is_unique
    for code in codes:
        assert resolve_cancer_type(code) is not None


def test_class_and_score_consistent_and_cited():
    df = _df()
    for row in df.itertuples():
        assert row.indel_class in _CLASS_TO_SCORE
        assert int(row.indel_score) == _CLASS_TO_SCORE[row.indel_class]
        assert isinstance(row.pmid_doi, str) and row.pmid_doi.strip()
        assert row.confidence in {"high", "medium", "low"}


def test_mechanistic_anchors():
    """The flag must encode the two established mechanisms; chromophobe RCC
    (genomically quiet) must NOT be flagged high, and MSS CRC must be baseline."""
    df = _df().set_index("cancer_code")
    # RCC lineage (ccRCC/papillary) + dMMR/MSI-H are the high-indel groups.
    for code in ("KIRC", "KIRP", "COAD_MSI", "READ_MSI", "UCEC_MSI"):
        assert df.loc[code, "indel_score"] == 2
    # chromophobe RCC is quiet (low TMB), not indel-enriched.
    assert df.loc["KICH", "indel_score"] == 0
    # microsatellite-stable CRC has low frameshift load.
    assert df.loc["COAD_MSS", "indel_score"] == 0
    assert df.loc["SKCM", "indel_score"] == 0  # high SNV-TMB, baseline indel fraction
