"""Source selection must be explicit when constructing wide references."""

import pandas as pd
import pytest

from pirlygenes.expression import accessors


@pytest.fixture
def sources(monkeypatch):
    frame = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000141510"] * 2,
        "Symbol": ["TP53"] * 2,
        "cancer_code": ["NUTM"] * 2,
        "source_cohort": ["A", "B"],
        "normalization": ["TPM_clean"] * 2,
        "expression": [10.0, 100.0],
    })

    def reference(**kwargs):
        selected = frame.copy()
        if kwargs.get("source_cohort"):
            selected = selected[selected.source_cohort.eq(kwargs["source_cohort"])]
        if kwargs.get("pool"):
            selected = selected.iloc[:1].assign(expression=55.0, source_cohort="POOLED")
        return selected

    monkeypatch.setattr(accessors, "_oncoref_reference_mode", reference)
    monkeypatch.setattr(accessors, "_oncoref_reference_code_set", lambda: {"NUTM"})
    return frame


def test_ambiguous_wide_reference_raises_in_both_source_orders(sources):
    for _ in range(2):
        with pytest.raises(ValueError, match="select a single source_cohort"):
            accessors.cancer_reference_expression("NUTM", format="wide")
        sources["expression"] = sources["expression"].iloc[::-1].to_numpy()
    long = accessors.cancer_reference_expression("NUTM", format="long")
    assert set(long.expression) == {10.0, 100.0}


def test_wide_reference_accepts_explicit_source_or_pool(sources):
    selected = accessors.cancer_reference_expression("NUTM", format="wide", source_cohort="B")
    assert selected.NUTM_TPM_clean.tolist() == [100.0]
    pooled = accessors.cancer_reference_expression("NUTM", format="wide", pool=True)
    assert pooled.NUTM_TPM_clean.tolist() == [55.0]


def test_empty_source_selection_preserves_wide_columns(sources):
    result = accessors.cancer_reference_expression("NUTM", format="wide", source_cohort="missing")
    assert result.empty
    assert result.columns.tolist() == ["Ensembl_Gene_ID", "Symbol", "NUTM_TPM_clean"]
