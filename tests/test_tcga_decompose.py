"""Tests for the offline TCGA deconvolution helpers (#21).

The heavy decompose_sample path needs the full TPM matrix and is
exercised by the maintainer's offline batch run, not by CI. These
tests cover the pure helpers: sample-barcode parsing, primary-tumor
filter, selection with cancer-type / per-type caps, and the
per-type aggregation.
"""

import pandas as pd
import pytest

from pirlygenes.tcga_decompose import (
    aggregate_per_type,
    is_primary_tumor_sample,
    sample_barcode_to_project,
    select_samples,
)


def test_sample_barcode_extracts_patient_prefix():
    patient_to_project = {"TCGA-XX-YYYY": "BRCA"}
    assert sample_barcode_to_project("TCGA-XX-YYYY-01", patient_to_project) == "BRCA"
    assert sample_barcode_to_project("TCGA-XX-YYYY-11A", patient_to_project) == "BRCA"


def test_sample_barcode_unknown_returns_none():
    assert sample_barcode_to_project("TCGA-AA-BBBB-01", {}) is None


def test_sample_barcode_non_tcga_returns_none():
    assert sample_barcode_to_project("GTEX-N7MS-01", {"TCGA-XX-YYYY": "BRCA"}) is None


def test_primary_tumor_sample_codes():
    assert is_primary_tumor_sample("TCGA-XX-YYYY-01")
    assert is_primary_tumor_sample("TCGA-XX-YYYY-01A")
    # 03 / 09 = primary blood cancer types.
    assert is_primary_tumor_sample("TCGA-XX-YYYY-03")
    assert is_primary_tumor_sample("TCGA-XX-YYYY-09")
    # Normals and metastases must not pass the filter.
    assert not is_primary_tumor_sample("TCGA-XX-YYYY-11")  # solid normal
    assert not is_primary_tumor_sample("TCGA-XX-YYYY-10")  # blood normal
    assert not is_primary_tumor_sample("TCGA-XX-YYYY-06")  # met
    # Malformed barcode.
    assert not is_primary_tumor_sample("TCGA-XX-YYYY")


def test_select_samples_filters_normals_and_other_types():
    columns = pd.Index(
        [
            "TCGA-A1-A0SB-01",  # BRCA primary
            "TCGA-A1-A0SB-11",  # BRCA normal — skip
            "TCGA-19-1787-01",  # GBM primary
            "TCGA-OR-A5LF-06",  # ACC metastasis — skip
        ]
    )
    patient_to_project = {
        "TCGA-A1-A0SB": "BRCA",
        "TCGA-19-1787": "GBM",
        "TCGA-OR-A5LF": "ACC",
    }
    pairs = select_samples(columns, patient_to_project)
    assert ("TCGA-A1-A0SB-01", "BRCA") in pairs
    assert ("TCGA-19-1787-01", "GBM") in pairs
    assert not any(code == "ACC" for _, code in pairs)
    assert not any(b.endswith("-11") for b, _ in pairs)


def test_select_samples_per_type_cap():
    columns = pd.Index(
        [
            "TCGA-AA-0001-01",
            "TCGA-AA-0002-01",
            "TCGA-AA-0003-01",
            "TCGA-BB-0001-01",
        ]
    )
    patient_to_project = {
        "TCGA-AA-0001": "BRCA",
        "TCGA-AA-0002": "BRCA",
        "TCGA-AA-0003": "BRCA",
        "TCGA-BB-0001": "GBM",
    }
    pairs = select_samples(
        columns, patient_to_project, max_samples_per_type=2
    )
    brca = [b for b, code in pairs if code == "BRCA"]
    gbm = [b for b, code in pairs if code == "GBM"]
    assert len(brca) == 2
    assert len(gbm) == 1


def test_select_samples_cancer_type_filter():
    columns = pd.Index(["TCGA-A1-A0SB-01", "TCGA-19-1787-01"])
    patient_to_project = {
        "TCGA-A1-A0SB": "BRCA",
        "TCGA-19-1787": "GBM",
    }
    pairs = select_samples(
        columns, patient_to_project, cancer_types=["GBM"]
    )
    assert pairs == [("TCGA-19-1787-01", "GBM")]


def test_aggregate_per_type_basic_stats():
    per_sample = pd.DataFrame(
        [
            {"sample": "s1", "cancer_code": "BRCA", "symbol": "TP53", "tumor_tpm": 10.0},
            {"sample": "s2", "cancer_code": "BRCA", "symbol": "TP53", "tumor_tpm": 20.0},
            {"sample": "s3", "cancer_code": "BRCA", "symbol": "TP53", "tumor_tpm": 30.0},
            {"sample": "s4", "cancer_code": "BRCA", "symbol": "TP53", "tumor_tpm": 40.0},
            {"sample": "s1", "cancer_code": "GBM", "symbol": "TP53", "tumor_tpm": 5.0},
        ]
    )
    summary = aggregate_per_type(per_sample)
    brca = summary[(summary["cancer_code"] == "BRCA") & (summary["symbol"] == "TP53")].iloc[0]
    assert brca["tumor_tpm_median"] == pytest.approx(25.0)
    assert brca["tumor_tpm_q1"] == pytest.approx(17.5)
    assert brca["tumor_tpm_q3"] == pytest.approx(32.5)
    assert brca["n_samples"] == 4
    gbm = summary[summary["cancer_code"] == "GBM"].iloc[0]
    assert gbm["n_samples"] == 1


def test_aggregate_per_type_empty_input():
    empty = pd.DataFrame(
        columns=["sample", "cancer_code", "symbol", "tumor_tpm"]
    )
    summary = aggregate_per_type(empty)
    assert list(summary.columns) == [
        "symbol",
        "cancer_code",
        "tumor_tpm_median",
        "tumor_tpm_q1",
        "tumor_tpm_q3",
        "n_samples",
    ]
    assert len(summary) == 0
