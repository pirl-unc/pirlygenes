"""Regression tests for CTA patient-count analysis helpers."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))

from cta_patient_counts import _mage_mask, _specific_9mer_weights  # noqa: E402
import cta_patient_counts as counts  # noqa: E402
from pirlygenes.expression import protein_groups  # noqa: E402


def test_symbol_keyed_metrics_align_to_ensembl_matrix_index():
    index = pd.Index(["ENSG_PRAME", "ENSG_MAGEA3", "ENSG_OTHER"])
    ensg_to_sym = {
        "ENSG_PRAME": "PRAME",
        "ENSG_MAGEA3": "MAGEA3",
        "ENSG_OTHER": "OTHER",
    }

    weights = _specific_9mer_weights(
        index,
        ensg_to_sym,
        {"PRAME": 123, "MAGEA3": 456},
    )

    np.testing.assert_array_equal(weights, [123.0, 456.0, 0.0])
    assert _mage_mask(index, ensg_to_sym).tolist() == [False, True, False]


def test_folded_weights_use_same_groups_as_expression():
    mapping = protein_groups.cdna_canonical_to_symbol()
    symbols = ["CTAG1A/B", "XAGE1A/B", "CXorf49/CXorf49B"]
    ids = [next(gene for gene, symbol in mapping.items() if symbol == wanted)
           for wanted in symbols]
    weights = _specific_9mer_weights(pd.Index(ids), dict(zip(ids, symbols)), {
        "CTAG1A": 150, "CTAG1B": 172, "XAGE1A": 73, "XAGE1B": 73,
        "CXorf49": 400, "CXorf49B": 421,
    })
    np.testing.assert_array_equal(weights, [172, 73, 421])


def test_owner_matrix_uses_clean_values_recovers_partners_and_preserves_missing(monkeypatch):
    import oncoref
    from oncoref import source_matrices

    m2c = protein_groups.cdna_member_to_canonical()
    canon = next(gene for gene, sym in protein_groups.cdna_canonical_to_symbol().items()
                 if sym == "CTAG1A/B")
    members = [gene for gene, group in m2c.items() if group == canon]
    assert len(members) == 2
    frame = pd.DataFrame({
        "Ensembl_Gene_ID": [*members, "ENSG_OTHER"],
        "Symbol": ["CTAG1A", "CTAG1B", "OTHER"],
        "s1": [10.0, 20.0, 70.0],
        "s2": [np.nan, np.nan, 70.0],
    })
    monkeypatch.setattr(source_matrices, "registry", lambda: pd.DataFrame({
        "cancer_code": ["NUTM"], "source_cohort": ["UNC_NUTM1"],
    }))
    calls = []

    def owner(code, **kwargs):
        calls.append((code, kwargs))
        return frame.copy()

    monkeypatch.setattr(oncoref, "per_sample_expression", owner)
    matrix, cohorts, cutoffs = counts.owner_cta_expression({members[0], "ENSG_ABSENT"}, percentiles=[50])
    merged, symbols = counts._merge_proteins(matrix, {})
    assert calls == [("NUTM", {"normalize": "tpm_clean", "sample_qc": "all"})]
    assert merged.loc[canon, "UNC_NUTM1::s1"] == 30.0
    assert np.isnan(merged.loc[canon, "UNC_NUTM1::s2"])
    assert merged.loc["ENSG_ABSENT"].isna().all()
    assert cutoffs.loc["UNC_NUTM1::s1", "p50"] == 50.0
    assert cohorts["NUTM"] == ["UNC_NUTM1::s1", "UNC_NUTM1::s2"]


def test_gene_counts_distinguish_measured_negatives_from_missing():
    matrix = pd.DataFrame({
        "s1": [100.0, 0.0, np.nan],
        "s2": [np.nan, 0.0, np.nan],
        "s3": [0.0, 0.0, np.nan],
    }, index=["positive", "negative", "absent"])
    result = counts.per_cohort_counts(matrix, {"cohort": list(matrix)}, {}).set_index("Ensembl_Gene_ID")
    assert result.loc["positive", "n_samples"] == 3
    assert result.loc["positive", "n_available"] == 2
    assert result.loc["positive", "pct_gt25"] == 50.0
    assert result.loc["negative", "n_available"] == 3
    assert result.loc["negative", "pct_gt25"] == 0.0
    assert result.loc["absent", "n_available"] == 0
    assert np.isnan(result.loc["absent", "pct_gt25"])


def test_union_reports_unknowns_and_load_ignores_unobserved_values():
    matrix = pd.DataFrame({
        "positive": [100.0, np.nan], "negative": [0.0, 0.0],
        "unknown": [0.0, np.nan],
    }, index=["A", "B"])
    result = counts.per_cohort_union_counts(matrix, {"C": list(matrix)}, {}).iloc[0]
    assert result.n_samples == 3
    assert result.n_any_gt25 == 1
    assert result.n_unknown_gt25 == 1
    load = counts._mean_total_cta_tpm(matrix, list(matrix), 25, None)
    assert load == pytest.approx(100 / 3)


def test_downstream_factor_table_keeps_folded_9mer_payload(tmp_path, monkeypatch):
    import _apd1_factors as factors

    output = tmp_path / "outputs"
    cache = output / "_cache"
    cache.mkdir(parents=True)
    pd.DataFrame({
        "cancer_code": ["NUTM"], "n_samples": [10], "Symbol": ["CTAG1A/B"],
        "n_p90": [5], "n_p95": [2],
    }).to_csv(output / "_cta_patient_counts.csv", index=False)
    pd.DataFrame({
        "Symbol": ["CTAG1A", "CTAG1B"], "n_specific_9mers": [150, 172],
    }).to_csv(cache / "cta_specific_9mers.csv", index=False)
    monkeypatch.setattr(factors, "__file__", str(tmp_path / "_apd1_factors.py"))
    factors.cta_metric_table.cache_clear()
    try:
        metrics = factors.cta_metric_table()
        assert metrics.loc["NUTM", "cta_9mer_load_p90"] == 86.0
        assert metrics.loc["NUTM", "cta_9mer_load_p95"] == pytest.approx(34.4)
    finally:
        factors.cta_metric_table.cache_clear()
