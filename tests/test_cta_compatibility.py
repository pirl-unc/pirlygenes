"""Active OncoRef CTA evidence & partition contracts with HPA and Tsarina MS joins (#655)."""

import pandas as pd
import pytest

from pirlygenes import _cta_compatibility as compatibility
from pirlygenes import gene_sets_cancer as public


@pytest.fixture
def active_panel(monkeypatch):
    selection = pd.DataFrame({
        "Ensembl_Gene_ID": ["SELECTED", "CA_ONLY", "UNSAFE", "NONCODING"],
        "Symbol": ["Selected", "CaOnly", "Unsafe", "Noncoding"],
        "is_current_cta": [True, False, False, False],
        "selection_category": ["CTA", "Cancer Antigen only", "Unsafe", "Other"],
        "selection_policy_id": ["oncoref.target-selection.test"] * 4,
    })
    historical = pd.DataFrame({
        "Ensembl_Gene_ID": ["SELECTED", "UNSAFE", "NONCODING"],
        "Symbol": ["OldSelected", "OldUnsafe", "OldNoncoding"],
        "passes_filters": pd.Series([True, False, pd.NA], dtype="boolean"),
        "restriction": ["TESTIS", "SOMATIC", ""],
        "restriction_confidence": ["HIGH", "MODERATE", ""],
    })
    current_ms = pd.DataFrame({
        "Ensembl_Gene_ID": ["SELECTED", "CA_ONLY", "UNSAFE", "NONCODING", "EXTRA"],
        "Symbol": ["Different"] * 5,
        "is_current_cta": [False, False, False, False, True],
        "passes_filters": [False] * 5,
        "ms_restriction": ["NO_MS_DATA", "REPRODUCTIVE", "SOMATIC", "NO_MS_DATA", "REPRODUCTIVE"],
        "ms_pmids": ["", "789", "123", "", "456"],
    })
    monkeypatch.setattr(compatibility, "_active_selection_table", lambda: selection.copy())
    monkeypatch.setattr(compatibility, "_active_cta_gene_ids", lambda: {"SELECTED"})
    monkeypatch.setattr(compatibility.cta, "cta_evidence", lambda: historical.copy())
    monkeypatch.setattr(compatibility, "_current_evidence", lambda: current_ms.copy())
    monkeypatch.setattr(compatibility, "_coding_gene_names", lambda release: {
        "SELECTED": "Selected", "CA_ONLY": "CaOnly", "UNSAFE": "Unsafe",
        "EXTRA": "Extra", "BACKGROUND": "Background",
    })
    return selection, historical, current_ms


def test_evidence_uses_active_selection_with_hpa_and_independent_ms(active_panel):
    selection, historical, current_ms = active_panel
    before_sel = selection.copy(deep=True)
    before_hpa = historical.copy(deep=True)
    before_ms = current_ms.copy(deep=True)

    evidence = public.CTA_evidence()
    pd.testing.assert_frame_equal(evidence[selection.columns], selection)
    pd.testing.assert_series_equal(evidence.passes_filters, evidence.filtered, check_names=False)
    assert evidence.loc[evidence.Ensembl_Gene_ID.eq("UNSAFE"), "ms_pmids"].tolist() == ["123"]
    assert evidence.loc[evidence.Ensembl_Gene_ID.eq("NONCODING"), "filtered"].isna().all()
    assert evidence.loc[evidence.Ensembl_Gene_ID.eq("CA_ONLY"), "passes_filters"].isna().all()
    assert evidence.loc[evidence.Ensembl_Gene_ID.eq("CA_ONLY"), "restriction"].tolist() == ["NO_DATA"]
    assert evidence.attrs["policy_id"] == public.CTA_policy_id()

    evidence.loc[0, "Symbol"] = "Changed"
    pd.testing.assert_frame_equal(selection, before_sel)
    pd.testing.assert_frame_equal(historical, before_hpa)
    pd.testing.assert_frame_equal(current_ms, before_ms)


def test_all_partition_views_use_active_membership_with_empty_never_expressed(active_panel):
    frames = public.CTA_partition_dataframes()
    ids = public.CTA_partition_gene_ids()
    names = public.CTA_partition_gene_names()

    assert isinstance(frames, public.CTAPartitionDataFrames)
    assert isinstance(ids, public.CTAPartitionSets)
    assert ids.cta == {"SELECTED"}
    assert ids.cta_never_expressed == set()
    assert ids.non_cta == {"CA_ONLY", "UNSAFE", "EXTRA", "BACKGROUND"}
    for population in ("cta", "cta_never_expressed", "non_cta"):
        frame = getattr(frames, population)
        assert set(frame.Ensembl_Gene_ID) == getattr(ids, population)
        assert set(frame.Symbol) == getattr(names, population)
    assert set(frames.cta.ms_restriction) == {"NO_MS_DATA"}
    assert frames.cta.attrs["policy_id"] == public.CTA_policy_id()


def test_ambiguous_ms_gene_identity_is_rejected(active_panel, monkeypatch):
    _, _, current_ms = active_panel
    duplicate = pd.concat([current_ms, current_ms.iloc[[0]]], ignore_index=True)
    monkeypatch.setattr(compatibility, "_current_evidence", lambda: duplicate)
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        public.CTA_evidence()


def test_missing_active_cta_ms_identity_is_rejected(active_panel, monkeypatch):
    _, _, current_ms = active_panel
    monkeypatch.setattr(compatibility, "_current_evidence", lambda: current_ms.iloc[1:])
    with pytest.raises(ValueError, match="lacks active CTA IDs: SELECTED"):
        public.CTA_evidence()
