"""Regression tests for the aPD1/ICI causal-factor contribution plot."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))

from apd1_ici_factor_contributions import (  # noqa: E402
    _available_cta_metric_columns,
    _driver_title,
)


def test_missing_cta_9mer_metrics_are_skipped():
    index = pd.Index(["A", "B", "C", "D"])
    cta = pd.DataFrame({
        "cta_coverage_p90": [10.0, 20.0, 30.0, 40.0],
        "cta_count_p90": [1.0, 2.0, 3.0, 4.0],
    }, index=index)

    cols, skipped = _available_cta_metric_columns(cta, index)

    assert set(cols) == {"CTA coverage p90", "CTA load p90"}
    assert "CTA 9mer load p90" in skipped
    assert "CTA 9mer load p95" in skipped


def test_all_nan_cta_metric_is_skipped():
    index = pd.Index(["A", "B", "C", "D"])
    cta = pd.DataFrame({
        "cta_9mer_load_p90": [np.nan, np.nan, np.nan, np.nan],
        "cta_9mer_load_p95": [1.0, 2.0, 3.0, 4.0],
    }, index=index)

    cols, skipped = _available_cta_metric_columns(cta, index)

    assert set(cols) == {"CTA 9mer load p95"}
    assert "CTA 9mer load p90" in skipped


def test_driver_title_reflects_available_cta_metrics():
    factors = ["median TMB", "CTA coverage p90", "CTA load p90", "viral status"]

    title = _driver_title(factors)

    assert "coverage p90" in title
    assert "load p90" in title
    assert "9mer" not in title
