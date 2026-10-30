"""Historical membership cannot drift with Tsarina's default selection."""

import pandas as pd
import pytest

from pirlygenes import _cta_compatibility as compatibility
from pirlygenes import gene_sets_cancer as public


@pytest.fixture
def historical_panel(monkeypatch):
    historical = pd.DataFrame({
        "Ensembl_Gene_ID": ["SELECTED", "NEVER", "EXCLUDED", "NONCODING"],
        "Symbol": ["Selected", "Never", "Excluded", "Noncoding"],
        "passes_filters": pd.Series([True, True, False, pd.NA], dtype="boolean"),
    })
    current = pd.DataFrame({
        "Ensembl_Gene_ID": ["SELECTED", "NEVER", "EXCLUDED", "NONCODING", "CURRENT_ONLY"],
        "Symbol": ["Different"] * 5,
        "is_current_cta": [False, False, False, False, True],
        "passes_filters": [False] * 5,
        "ms_restriction": ["NO_MS_DATA", "NO_MS_DATA", "SOMATIC", "NO_MS_DATA", "REPRODUCTIVE"],
        "ms_pmids": ["", "", "123", "", "456"],
    })
    monkeypatch.setattr(compatibility.cta, "cta_evidence", lambda: historical)
    monkeypatch.setattr(compatibility.cta, "cta_gene_ids", lambda: {"SELECTED"})
    monkeypatch.setattr(compatibility.cta, "cta_never_expressed_gene_ids", lambda: {"NEVER"})
    monkeypatch.setattr(compatibility, "_current_evidence", lambda: current)
    monkeypatch.setattr(compatibility, "_coding_gene_names", lambda release: {
        "SELECTED": "Selected", "NEVER": "Never", "EXCLUDED": "Excluded",
        "CURRENT_ONLY": "Current", "BACKGROUND": "Background",
    })
    return historical, current


def test_evidence_preserves_hpa_identity_unknowns_and_independent_ms(historical_panel):
    historical, current = historical_panel
    before_hpa, before_current = historical.copy(deep=True), current.copy(deep=True)
    evidence = public.CTA_evidence()
    pd.testing.assert_frame_equal(evidence[historical.columns], historical)
    pd.testing.assert_series_equal(evidence.passes_filters, evidence.filtered, check_names=False)
    assert "is_current_cta" not in evidence
    assert evidence.loc[evidence.Ensembl_Gene_ID.eq("EXCLUDED"), "ms_pmids"].tolist() == ["123"]
    assert evidence.loc[evidence.Ensembl_Gene_ID.eq("NONCODING"), "filtered"].isna().all()
    evidence.loc[0, "Symbol"] = "Changed"
    pd.testing.assert_frame_equal(historical, before_hpa)
    pd.testing.assert_frame_equal(current, before_current)


def test_all_partition_views_use_historical_membership(historical_panel):
    frames = public.CTA_partition_dataframes()
    ids = public.CTA_partition_gene_ids()
    names = public.CTA_partition_gene_names()
    assert isinstance(frames, public.CTAPartitionDataFrames)
    assert isinstance(ids, public.CTAPartitionSets)
    assert ids.cta == {"SELECTED"}
    assert ids.cta_never_expressed == {"NEVER"}
    assert ids.non_cta == {"EXCLUDED", "CURRENT_ONLY", "BACKGROUND"}
    for population in ("cta", "cta_never_expressed", "non_cta"):
        frame = getattr(frames, population)
        assert set(frame.Ensembl_Gene_ID) == getattr(ids, population)
        assert set(frame.Symbol) == getattr(names, population)
    assert set(frames.cta.ms_restriction) == {"NO_MS_DATA"}


def test_ambiguous_ms_gene_identity_is_rejected(historical_panel, monkeypatch):
    _, current = historical_panel
    duplicate = pd.concat([current, current.iloc[[0]]], ignore_index=True)
    monkeypatch.setattr(compatibility, "_current_evidence", lambda: duplicate)
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        public.CTA_evidence()


def test_missing_ms_identity_is_not_filled_with_negative_evidence(historical_panel, monkeypatch):
    _, current = historical_panel
    monkeypatch.setattr(compatibility, "_current_evidence", lambda: current.iloc[1:])
    with pytest.raises(ValueError, match="lacks historical CTA IDs: SELECTED"):
        public.CTA_evidence()
