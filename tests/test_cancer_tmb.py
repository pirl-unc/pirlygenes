"""Tests for the curated per-cancer-type median-TMB reference (cancer-tmb.csv)."""

import math

import pandas as pd

from pirlygenes.gene_sets_cancer import cancer_tmb, cancer_tmb_df, resolve_cancer_type

_EXPECTED_COLS = [
    "cancer_code",
    "median_tmb_mut_mb",
    "n_samples",
    "source",
    "pmid_doi",
    "confidence",
    "notes",
]


def test_n_samples_present_for_small_cohort_estimates():
    """The cohort size behind each estimate is tracked in n_samples; the
    rare/weak-n entities added from the literature audit carry an explicit n
    (e.g. CRANIO n=3, HCL n=1) so low-precision estimates are transparent."""
    df = cancer_tmb_df().set_index("cancer_code")
    for code in ("CRANIO", "MTC", "HCL"):
        assert df.loc[code, "median_tmb_mut_mb"] > 0
        assert df.loc[code, "n_samples"] > 0
        assert df.loc[code, "confidence"] in {"high", "medium", "low"}


def test_schema_and_unique_codes():
    df = cancer_tmb_df()
    assert list(df.columns) == _EXPECTED_COLS
    codes = df["cancer_code"].astype(str)
    assert codes.is_unique
    # Every code must be a real registry code.
    for code in codes:
        assert resolve_cancer_type(code) is not None


def test_values_positive_and_plausible():
    df = cancer_tmb_df()
    vals = df.dropna(subset=["median_tmb_mut_mb"])["median_tmb_mut_mb"].astype(float)
    assert (vals > 0).all()
    # Median TMB by type should sit in a sane range (mut/Mb); UV-driven
    # non-melanoma skin (BCC ~65, cSCC ~45) is the high end, melanoma ~13,
    # pheo/retinoblastoma the low end (<0.1).
    assert vals.max() < 100
    assert vals.min() >= 0.01


def test_every_value_is_cited():
    """A non-blank TMB value must carry a source + PMID/DOI; a blank value must
    be explicitly flagged confidence=none (an honest gap, not a silent absence)."""
    df = cancer_tmb_df()
    for row in df.itertuples():
        has_value = isinstance(row.median_tmb_mut_mb, float) and not math.isnan(
            row.median_tmb_mut_mb
        )
        if has_value:
            assert isinstance(row.source, str) and row.source.strip()
            assert isinstance(row.pmid_doi, str) and row.pmid_doi.strip()
            assert row.confidence in {"high", "medium", "low"}
        else:
            assert row.confidence == "none"


def test_accessor_map_omits_blanks():
    mapping = cancer_tmb()
    assert isinstance(mapping, dict)
    # Blank-value codes (no published median) are absent from the map.
    assert "MPN" not in mapping
    assert "CML" not in mapping
    # Well-established values are present.
    assert "SKCM" in mapping and "PRAD" in mapping


def test_accessor_resolves_aliases():
    # melanoma -> SKCM (high TMB), prostate -> PRAD (low TMB)
    assert cancer_tmb("melanoma") == cancer_tmb("SKCM")
    assert cancer_tmb("melanoma") > cancer_tmb("prostate")
    # A code with no curated value returns None rather than raising.
    assert cancer_tmb("MPN") is None


def test_skcm_is_highest_among_common_types():
    mapping = cancer_tmb()
    assert mapping["SKCM"] == max(
        mapping[c] for c in ("SKCM", "LUAD", "LUSC", "BLCA", "BRCA", "PRAD")
    )


def test_subtype_inherits_parent_tmb():
    """Molecular / histology subtypes with no curated row inherit the nearest
    ancestor's TMB by walking parent_code (default inherit=True)."""
    # subtypes with NO curated row inherit the parent: LUAD_KRAS/STK11 -> LUAD
    # (smoking, ~= parent), SCLC_ASCL1 -> SCLC, SARC_EPITH -> pan-SARC.
    assert cancer_tmb("LUAD_KRAS") == cancer_tmb("LUAD")
    assert cancer_tmb("SCLC_ASCL1") == cancer_tmb("SCLC")
    assert cancer_tmb("SARC_EPITH") == cancer_tmb("SARC")
    # but a subtype that genuinely DIFFERS gets its own cited row and overrides
    # the parent: LUAD_EGFR (never-smoker, lower) and SARC_CIC (CIC-driven, low).
    assert cancer_tmb("LUAD_EGFR") == 3.5 and cancer_tmb("LUAD_EGFR") != cancer_tmb("LUAD")
    assert cancer_tmb("SARC_CIC") == 1.2 and cancer_tmb("SARC_CIC") != cancer_tmb("SARC")
    # inherit=False is the strict direct-only lookup (no parent walk)
    assert cancer_tmb("LUAD_KRAS", inherit=False) is None
    # a top-level code with a genuinely blank value stays None even with inherit
    assert cancer_tmb("MPN") is None
