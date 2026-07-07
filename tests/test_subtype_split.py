"""Tests for the shared TCGA molecular-subtype split helper (pirlygenes#529).

Covers the pure partition logic and a real end-to-end split of a synthetic
parent per-sample parquet into subtype cohorts (read -> split -> write per-sample
parquets -> per-code summary shard), with the cache redirected to a tmp dir.
"""
from __future__ import annotations

import pandas as pd

from pirlygenes.builders import subtype_split as ss


def test_case_id_extracts_tcga_case_submitter_id():
    assert ss.case_id("TCGA-AA-3520-01A-01R-1234-07") == "TCGA-AA-3520"
    assert ss.case_id("TCGA-AA-3520") == "TCGA-AA-3520"  # already a case id


def test_group_samples_by_code_partitions_and_drops_unclassified():
    df = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000141510"],
        "Symbol": ["TP53"],
        "TCGA-A1-0001-01A": [10.0],  # UCEC_MSI
        "TCGA-A1-0002-01A": [12.0],  # UCEC_MSI
        "TCGA-A1-0003-01A": [8.0],   # UCEC_CN_LOW -> UCEC_CNL
        "TCGA-A1-0008-01A": [7.0],   # labeled but subtype not in code map -> dropped
        "TCGA-A1-0009-01A": [9.0],   # unlabeled case -> dropped
    })
    label_by_case = {
        "TCGA-A1-0001": "UCEC_MSI", "TCGA-A1-0002": "UCEC_MSI",
        "TCGA-A1-0003": "UCEC_CN_LOW", "TCGA-A1-0008": "UCEC_POLE",
    }
    code_by_label = {"UCEC_MSI": "UCEC_MSI", "UCEC_CN_LOW": "UCEC_CNL"}  # no POLE
    by_code = ss.group_samples_by_code(
        df, label_by_case=label_by_case, code_by_label=code_by_label)
    assert by_code == {
        "UCEC_MSI": ["TCGA-A1-0001-01A", "TCGA-A1-0002-01A"],
        "UCEC_CNL": ["TCGA-A1-0003-01A"],
    }


def _parent_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000075624",
                            "ENSG00000111640"],
        "Symbol": ["TP53", "ACTB", "GAPDH"],
        "TCGA-A1-0001-01A": [10.0, 500.0, 450.0],
        "TCGA-A1-0002-01A": [12.0, 480.0, 460.0],
        "TCGA-A1-0003-01A": [8.0, 520.0, 470.0],
        "TCGA-A1-0009-01A": [9.0, 510.0, 455.0],  # unlabeled -> dropped
    })


def test_build_subtype_split_end_to_end(tmp_path, monkeypatch):
    from pirlygenes import cohorts
    from pirlygenes.expression.stats import build_reference_rows

    cache = tmp_path / "treehouse-polya-25-01"
    (cache / "derived").mkdir(parents=True)
    monkeypatch.setattr(cohorts.downloads, "source_cache_dir",
                        lambda *a, **k: cache)

    # Place the synthetic parent UCEC parquet exactly where read_per_sample looks.
    ucec = cohorts.cohorts_for_source("treehouse-polya-25-01")["UCEC"]
    _parent_frame().to_parquet(cohorts.parquet_path(ucec))

    label_by_case = {
        "TCGA-A1-0001": "UCEC_MSI", "TCGA-A1-0002": "UCEC_MSI",
        "TCGA-A1-0003": "UCEC_CN_LOW",  # -> UCEC_CNL
    }  # 0009 unlabeled -> dropped; no CNH/POLE this run
    code_by_label = {"UCEC_MSI": "UCEC_MSI", "UCEC_CN_LOW": "UCEC_CNL",
                     "UCEC_CN_HIGH": "UCEC_CNH", "UCEC_POLE": "UCEC_POLE"}

    def _row(gene_table, values, code):
        return build_reference_rows(
            gene_table, values, cancer_code=code, source_cohort="T",
            source_project="p", source_version="v", processing_pipeline="pp",
            notes="n", tumor_origin="primary")

    shard = tmp_path / "shard"
    written = ss.build_subtype_split(
        source_id="treehouse-polya-25-01", parent_code="UCEC",
        label_by_case=label_by_case, code_by_label=code_by_label,
        summary_cohort="T", summary_output=shard, make_summary_row=_row)

    assert set(written) == {"UCEC_MSI", "UCEC_CNL"}
    # Per-sample parquets persisted under each subtype stem, with the right n.
    msi = pd.read_parquet(cache / "derived" / "UCEC_MSI_per_sample_tpm.parquet")
    cnl = pd.read_parquet(cache / "derived" / "UCEC_CNL_per_sample_tpm.parquet")
    assert len(cohorts.sample_columns(msi)) == 2
    assert len(cohorts.sample_columns(cnl)) == 1
    # CNH/POLE matched zero samples -> no stray parquet written for them.
    assert not (cache / "derived" / "UCEC_CNH_per_sample_tpm.parquet").exists()
    assert not (cache / "derived" / "UCEC_POLE_per_sample_tpm.parquet").exists()


def test_ucec_subtype_cohorts_registered_in_group():
    """The four UCEC subtype cohorts are declared under one build group so the
    builder can enumerate them and the read path knows them without a cache."""
    from pirlygenes import cohorts

    group = {c.code for c in cohorts.cohorts_for_group("tcga_ucec_subtype")}
    assert group == {"UCEC_MSI", "UCEC_CNL", "UCEC_CNH", "UCEC_POLE"}
