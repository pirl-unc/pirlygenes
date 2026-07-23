"""Diagnosis-split GSE294016 salivary artifact contracts (#346)."""

import oncoref
import pytest

from pirlygenes.expression import (
    available_percentile_cohorts,
    available_representative_cohorts,
    cancer_reference_expression,
    cohort_gene_percentiles,
    pan_cancer_expression,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import cohort_registry_df


SALIVARY_SAMPLES = {"ADCC": 57, "ACINIC": 3}
SALIVARY_QC = {
    "ADCC": {"n_qc_pass": 56, "n_qc_fail": 1},
    "ACINIC": {"n_qc_pass": 2, "n_qc_fail": 1},
}
SALIVARY_SOURCE = "GSE294016_BARTL_2025_SGC"


def test_salivary_diagnosis_split_exposes_every_artifact_family():
    codes = set(SALIVARY_SAMPLES)
    summaries = cancer_reference_expression(
        codes, genes=["TP53"], format="long"
    )

    assert set(summaries["cancer_code"]) == codes
    assert set(summaries["source_cohort"]) == {SALIVARY_SOURCE}
    assert summaries.set_index("cancer_code")["n_samples"].to_dict() == (
        SALIVARY_SAMPLES
    )
    assert codes <= set(available_representative_cohorts())
    assert codes <= set(available_percentile_cohorts())

    for code in codes:
        representatives = representative_cohort_samples(code)
        assert not representatives.empty
        assert {"Ensembl_Gene_ID", "Symbol"} <= set(representatives.columns)

        percentiles = cohort_gene_percentiles(code)
        assert not percentiles.empty
        assert percentiles["Ensembl_Gene_ID"].is_unique
        assert {"p0", "p50", "p100"} <= set(percentiles.columns)

        availability = oncoref.representative_cohort_availability(code).iloc[0]
        for field, value in SALIVARY_QC[code].items():
            assert int(availability[field]) == value


def test_sgc_rollup_includes_adcc_and_acinic_with_sample_weights():
    summaries = cancer_reference_expression(
        SALIVARY_SAMPLES, genes=["TP53"], normalize="tpm", format="long"
    ).set_index("cancer_code")
    expected = sum(
        float(summaries.loc[code, "expression"]) * n_samples
        for code, n_samples in SALIVARY_SAMPLES.items()
    ) / sum(SALIVARY_SAMPLES.values())

    pan = pan_cancer_expression(genes=["TP53"], normalize="tpm").iloc[0]

    assert pan["SGC_TPM"] == pytest.approx(expected)


def test_salivary_registry_describes_only_the_released_partition():
    row = cohort_registry_df().set_index("cohort_id").loc[SALIVARY_SOURCE]

    assert row["assay"] == "bulk RNA-seq"
    assert int(row["n_samples"]) == sum(SALIVARY_SAMPLES.values())
    assert int(row["n_codes"]) == len(SALIVARY_SAMPLES)
    assert "57 ADCC and 3 ACINIC" in row["provenance"]
    assert "excluding 35 other histologies" in row["provenance"]
    assert "58 QC pass / 2 fail" in row["provenance"]
