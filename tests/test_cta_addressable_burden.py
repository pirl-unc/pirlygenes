"""Burden estimates must use deduplicated samples and measured denominators."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))
import cta_addressable_burden as burden  # noqa: E402
import cta_patient_counts as counts  # noqa: E402


def test_overlapping_parent_and_subtypes_are_counted_once(tmp_path, monkeypatch):
    monkeypatch.setattr(counts.gsc, "burden_category", lambda code: "lung")
    cohorts, members = counts.burden_sample_cohorts({
        "LUAD": ["s1", "s2", "s3"], "LUAD_EGFR": ["s1", "s2"],
        "LUAD_KRAS": ["s2", "s4"],
    })
    assert cohorts == {"lung": ["s1", "s2", "s3", "s4"]}
    matrix = pd.DataFrame({
        "s1": [100.0, 0.0], "s2": [0.0, 100.0],
        "s3": [0.0, np.nan], "s4": [0.0, 0.0],
    }, index=["MAGEA3", "PRAME"])
    symbols = {gene: gene for gene in matrix.index}
    c = counts.per_cohort_counts(matrix, cohorts, symbols).rename(columns={"cancer_code": "category"})
    u = counts.per_cohort_union_counts(matrix, cohorts, symbols).rename(columns={"cancer_code": "category"})
    for table in (c, u):
        table["source_cancer_codes"] = ";".join(members["lung"])
    counts_path, union_path = tmp_path / "counts.csv", tmp_path / "union.csv"
    c.to_csv(counts_path, index=False)
    u.to_csv(union_path, index=False)
    monkeypatch.setattr(burden, "COUNTS", counts_path)
    monkeypatch.setattr(burden, "UNION", union_path)
    monkeypatch.setattr(burden.gsc, "cancer_burden", lambda **kwargs: {"lung": 100.0})

    table, cat_n, breadth = burden._prepare()
    assert cat_n.to_dict() == {"lung": 4}
    rates, _ = burden._addressable(table, cat_n, burden.Thr("tpm", 25), "us_incidence_pct")
    rates = rates.set_index("Symbol").addressable
    assert rates["MAGEA3"] == 25.0
    assert rates["PRAME"] == pytest.approx(100 / 3)
    assert burden._union_addressable(burden.Thr("tpm", 25), "us_incidence_pct", cat_n) == 50.0

    nomage, filtered_n, _ = burden._prepare(drop_mage=True)
    assert nomage.Symbol.tolist() == ["PRAME"]
    pd.testing.assert_series_equal(filtered_n, cat_n)
    assert burden._union_addressable(burden.Thr("tpm", 25), "us_incidence_pct", cat_n, drop_mage=True) == 25.0


def test_conflicting_category_assignments_are_rejected(monkeypatch):
    monkeypatch.setattr(counts.gsc, "burden_category", lambda code: code)
    with pytest.raises(ValueError, match="spans burden categories"):
        counts.burden_sample_cohorts({"lung": ["same_sample"], "bone": ["same_sample"]})
