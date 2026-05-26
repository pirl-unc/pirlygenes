"""Tests for the extended cancer-reference-expression schema, the
shared :mod:`pirlygenes.expression.stats` helpers, and the
``pirlygenes data list`` CLI surface."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

from pirlygenes import cli, data_inventory
from pirlygenes.expression.stats import (
    CLEAN_STAT_COLUMNS,
    COUNT_COLUMNS,
    REFERENCE_COLUMNS,
    STAT_COLUMNS,
    assign_stats,
    compute_cohort_stats,
    compute_count_columns,
    round_stat_columns,
)
from pirlygenes.load_dataset import get_data


# ---------- schema ----------


def test_reference_columns_starts_with_legacy_order():
    legacy = (
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "TPM_median",
        "TPM_q1",
        "TPM_q3",
        "TPM_mean",
        "TPM_clean_median",
        "TPM_clean_q1",
        "TPM_clean_q3",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
    )
    assert REFERENCE_COLUMNS[: len(legacy)] == legacy


def test_reference_columns_appends_v53_extension():
    extension = (
        "TPM_std",
        "TPM_min",
        "TPM_max",
        "TPM_p5",
        "TPM_p10",
        "TPM_p90",
        "TPM_p95",
        "TPM_clean_mean",
        "TPM_clean_std",
        "TPM_clean_min",
        "TPM_clean_max",
        "TPM_clean_p5",
        "TPM_clean_p10",
        "TPM_clean_p90",
        "TPM_clean_p95",
    )
    assert REFERENCE_COLUMNS[-len(extension):] == extension


def test_stat_columns_have_raw_and_clean_parity():
    # The clean column tuple is generated from STAT_COLUMNS; they must
    # stay paired so any future addition lands on both sides.
    assert len(STAT_COLUMNS) == len(CLEAN_STAT_COLUMNS)
    for raw, clean in zip(STAT_COLUMNS, CLEAN_STAT_COLUMNS):
        assert clean == "TPM_clean_" + raw.removeprefix("TPM_")


def test_bundled_csv_has_full_schema():
    df = get_data("cancer-reference-expression")
    for col in REFERENCE_COLUMNS:
        assert col in df.columns, f"missing column {col!r}"


# ---------- compute_cohort_stats ----------


def test_compute_cohort_stats_against_known_values():
    # Two genes, five samples. Picked so every stat is hand-checkable.
    values = pd.DataFrame(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],   # mean 2, std≈1.581, q1=1, median=2, q3=3
            [10.0, 10.0, 10.0, 10.0, 10.0],  # all-10
        ],
        index=["g1", "g2"],
    )
    stats = compute_cohort_stats(values)
    assert stats["TPM_median"].tolist() == [2.0, 10.0]
    assert stats["TPM_mean"].tolist() == [2.0, 10.0]
    assert stats["TPM_q1"].tolist() == [1.0, 10.0]
    assert stats["TPM_q3"].tolist() == [3.0, 10.0]
    assert stats["TPM_min"].tolist() == [0.0, 10.0]
    assert stats["TPM_max"].tolist() == [4.0, 10.0]
    # std for {0,1,2,3,4} with ddof=1 is sqrt(2.5) ≈ 1.5811
    assert stats["TPM_std"][0] == np.sqrt(2.5)
    assert stats["TPM_std"][1] == 0.0
    # p5/p10/p90/p95 on a 5-vector use linear interpolation
    assert stats["TPM_p10"][0] == 0.4
    assert stats["TPM_p90"][0] == 3.6


def test_compute_cohort_stats_clean_prefix():
    values = pd.DataFrame([[1.0, 2.0, 3.0]], index=["g1"])
    stats = compute_cohort_stats(values, prefix="TPM_clean_")
    assert set(stats.keys()) == set(CLEAN_STAT_COLUMNS)


def test_compute_cohort_stats_single_sample_has_nan_std():
    values = pd.DataFrame([[7.0]], index=["g1"])
    stats = compute_cohort_stats(values)
    assert np.isnan(stats["TPM_std"][0])
    assert stats["TPM_mean"][0] == 7.0


def test_compute_count_columns():
    values = pd.DataFrame(
        [
            [0.0, 0.0, 1.0],   # 1 of 3 detected
            [5.0, 5.0, 5.0],   # 3 of 3 detected
        ],
        index=["g1", "g2"],
    )
    counts = compute_count_columns(values)
    assert counts["n_samples"].tolist() == [3, 3]
    assert counts["n_detected"].tolist() == [1, 3]


def test_assign_stats_populates_full_suite():
    raw = pd.DataFrame([[0.0, 2.0, 4.0]], index=["g1"])
    clean = pd.DataFrame([[0.0, 2.0, 4.0]], index=["g1"])
    out = pd.DataFrame({"Ensembl_Gene_ID": ["ENSG1"], "Symbol": ["S1"]})
    assign_stats(out, raw, clean)
    for col in STAT_COLUMNS + CLEAN_STAT_COLUMNS + COUNT_COLUMNS:
        assert col in out.columns, f"assign_stats failed to populate {col!r}"


def test_round_stat_columns_only_touches_known_columns():
    out = pd.DataFrame({"TPM_median": [1.234567891], "other": [9.999999]})
    rounded = round_stat_columns(out)
    assert rounded["TPM_median"].iloc[0] == round(1.234567891, 6)
    assert rounded["other"].iloc[0] == 9.999999


# ---------- data inventory + CLI ----------


def test_summarize_inventory_smoke():
    snapshot = data_inventory.summarize_inventory()
    assert snapshot.total_rows > 0
    assert snapshot.unique_genes > 0
    assert snapshot.cohort_rows
    assert snapshot.registered_sources > 0
    cohort_codes = {row.cancer_code for row in snapshot.cohort_rows}
    assert "BL" in cohort_codes
    assert "MM" in cohort_codes


def test_render_inventory_contains_expected_lines():
    snapshot = data_inventory.summarize_inventory()
    rendered = data_inventory.render_inventory(snapshot)
    assert "cancer-reference-expression" in rendered
    assert "size on disk" in rendered
    assert "Per-cohort row counts" in rendered
    assert "BL" in rendered


def _run_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


def test_cli_data_list_smoke():
    rc, out, _ = _run_cli(["data", "list"])
    assert rc == 0
    assert "cancer-reference-expression" in out
    assert "BL" in out


def test_cli_data_with_no_action_usage():
    rc, _, err = _run_cli(["data"])
    assert rc == 2
    assert "data" in err
