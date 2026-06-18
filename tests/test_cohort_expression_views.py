"""Unified cohort normalization-views object (#319): one object bundling the
tpm / clean_tpm / clean_tpm_biological stages so a consumer can't re-normalize
inconsistently."""

from pirlygenes.expression import (
    CohortExpressionViews,
    cohort_expression_views,
)
from pirlygenes.expression import accessors


def _disable_precomputed_views(monkeypatch, tmp_path):
    accessors._load_precomputed_cohort_views.cache_clear()
    from pathlib import Path
    root = Path(tmp_path)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root / "missing")


def test_views_bundle_three_stages_and_provenance():
    v = cohort_expression_views("CLL", genes=["MS4A1", "MALAT1", "RPL13A"])
    assert isinstance(v, CohortExpressionViews)
    for frame in (v.tpm, v.clean_tpm, v.clean_tpm_biological):
        assert {"Ensembl_Gene_ID", "Symbol"} <= set(frame.columns)
    # biological view drops the censored genes (MALAT1 technical, RPL13A ribo),
    # keeps real biology (MS4A1)
    bio = set(v.clean_tpm_biological["Symbol"])
    assert "MS4A1" in bio
    assert "MALAT1" not in bio and "RPL13A" not in bio
    # tpm/clean_tpm keep all requested genes (technical included)
    assert {"MS4A1", "MALAT1", "RPL13A"} <= set(v.clean_tpm["Symbol"])
    # provenance records the cohort + pipeline (native unit)
    assert "source_cohort" in v.provenance.columns
    assert "processing_pipeline" in v.provenance.columns
    assert len(v.provenance) >= 1


def test_views_clean_differs_from_tpm_for_technical_gene():
    """clean_tpm_16_9_75 changes the technical gene's value vs plain TPM (the whole
    point of having both stages in one object)."""
    v = cohort_expression_views("CLL", genes=["MS4A1", "MALAT1"])
    tpm = dict(zip(v.tpm["Symbol"], v.tpm["CLL"]))
    clean = dict(zip(v.clean_tpm["Symbol"], v.clean_tpm["CLL"]))
    # MALAT1 (polyA-bias technical) is suppressed under clean_tpm_16_9_75
    assert clean["MALAT1"] != tpm["MALAT1"]


def test_aggregate_code_expands_in_views():
    """An aggregate code (SARC) expands to its subtype cohorts in the views."""
    v = cohort_expression_views("SARC", genes=["TP53"])
    cohort_cols = [c for c in v.tpm.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert any(c.startswith("SARC_") for c in cohort_cols)


def test_views_canonicalize_before_pivoting_symbol_drift(monkeypatch, tmp_path):
    import pandas as pd

    fake = pd.DataFrame(
        [
            {
                "Ensembl_Gene_ID": "ENSG00000141510",
                "Symbol": "old_tp53_alias",
                "cancer_code": "AAA",
                "source_cohort": "S1",
                "source_project": "fixture",
                "source_version": "fixture-v1",
                "TPM_median": 1.0,
                "TPM_q1": 1.0,
                "TPM_q3": 1.0,
                "TPM_clean_median": 10.0,
                "TPM_clean_q1": 10.0,
                "TPM_clean_q3": 10.0,
                "n_samples": 1,
                "n_detected": 1,
                "processing_pipeline": "fixture",
                "notes": "",
            },
            {
                "Ensembl_Gene_ID": "ENSG00000141510",
                "Symbol": "TP53",
                "cancer_code": "AAA",
                "source_cohort": "S1",
                "source_project": "fixture",
                "source_version": "fixture-v2",
                "TPM_median": 2.0,
                "TPM_q1": 2.0,
                "TPM_q3": 2.0,
                "TPM_clean_median": 20.0,
                "TPM_clean_q1": 20.0,
                "TPM_clean_q3": 20.0,
                "n_samples": 1,
                "n_detected": 1,
                "processing_pipeline": "fixture",
                "notes": "",
            },
        ]
    )
    accessors._REFERENCE_VIEW_CACHE.clear()
    monkeypatch.setattr(accessors, "_load_cancer_reference_expression", lambda: fake)
    _disable_precomputed_views(monkeypatch, tmp_path)

    v = cohort_expression_views()

    assert v.tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]
    assert v.tpm["AAA"].iloc[0] == 3.0
    assert v.clean_tpm["AAA"].iloc[0] == 30.0


def _fixture_row(ensg, code, version, tpm):
    return {
        "Ensembl_Gene_ID": ensg, "Symbol": ensg, "cancer_code": code,
        "source_cohort": "S1", "source_project": "fixture",
        "source_version": version, "TPM_median": tpm, "TPM_q1": tpm,
        "TPM_q3": tpm, "TPM_clean_median": tpm, "TPM_clean_q1": tpm,
        "TPM_clean_q3": tpm, "n_samples": 1, "n_detected": 1,
        "processing_pipeline": "fixture", "notes": "",
    }


def test_views_protein_coding_and_coverage_filters(monkeypatch, tmp_path):
    import pandas as pd

    # TP53 (protein_coding) in both cohorts; MALAT1 (lncRNA) in one cohort only.
    fake = pd.DataFrame([
        _fixture_row("ENSG00000141510", "AAA", "v1", 1.0),
        _fixture_row("ENSG00000141510", "BBB", "v1", 1.0),
        _fixture_row("ENSG00000251562", "AAA", "v1", 5.0),
    ])
    monkeypatch.setattr(
        accessors, "_load_cancer_reference_expression", lambda: fake
    )

    accessors._REFERENCE_VIEW_CACHE.clear()
    _disable_precomputed_views(monkeypatch, tmp_path)
    pc = cohort_expression_views(protein_coding=True)
    assert pc.clean_tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]

    accessors._REFERENCE_VIEW_CACHE.clear()
    _disable_precomputed_views(monkeypatch, tmp_path)
    cov = cohort_expression_views(min_cohort_coverage=1.0)
    # only TP53 is measured in every cohort
    assert cov.clean_tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]


def test_views_reject_invalid_min_cohort_coverage():
    import pytest

    with pytest.raises(ValueError, match="min_cohort_coverage"):
        cohort_expression_views("CLL", min_cohort_coverage=-0.1)
    with pytest.raises(ValueError, match="min_cohort_coverage"):
        cohort_expression_views("CLL", min_cohort_coverage=1.1)


def test_views_precomputed_artifact_fast_path(tmp_path, monkeypatch):
    import pandas as pd

    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000251562"],
            "Symbol": ["TP53", "MALAT1"],
            "CLL": [10.0, 20.0],
            "PRAD": [30.0, None],
        }
    ).to_parquet(tmp_path / "tpm.parquet", index=False)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000251562"],
            "Symbol": ["TP53", "MALAT1"],
            "CLL": [1.0, 2.0],
            "PRAD": [3.0, None],
        }
    ).to_parquet(tmp_path / "clean_tpm.parquet", index=False)
    pd.DataFrame(
        [
            {
                "cancer_code": "CLL",
                "source_cohort": "S1",
                "processing_pipeline": "fixture",
                "n_samples": 2,
            },
            {
                "cancer_code": "PRAD",
                "source_cohort": "S2",
                "processing_pipeline": "fixture",
                "n_samples": 3,
            },
        ]
    ).to_parquet(tmp_path / "provenance.parquet", index=False)
    (tmp_path / "_manifest.json").write_text('{"canonical_gene_ids": true}\n')
    accessors._load_precomputed_cohort_views.cache_clear()
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: tmp_path)
    monkeypatch.setattr(
        accessors,
        "_cohort_expression_views_from_reference",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("slow path used")),
    )

    v = cohort_expression_views("CLL", genes=["TP53"])

    assert v.tpm.columns.tolist() == ["Ensembl_Gene_ID", "Symbol", "CLL"]
    assert v.tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]
    assert v.clean_tpm["CLL"].tolist() == [1.0]
    assert v.provenance["source_cohort"].tolist() == ["S1"]


def test_views_precomputed_gene_filter_drops_empty_cohorts(tmp_path, monkeypatch):
    import pandas as pd

    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "Symbol": ["TP53"],
            "CLL": [10.0],
            "PRAD": [None],
        }
    ).to_parquet(tmp_path / "tpm.parquet", index=False)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "Symbol": ["TP53"],
            "CLL": [1.0],
            "PRAD": [None],
        }
    ).to_parquet(tmp_path / "clean_tpm.parquet", index=False)
    pd.DataFrame(
        [
            {"cancer_code": "CLL", "source_cohort": "S1",
             "processing_pipeline": "fixture", "n_samples": 2},
            {"cancer_code": "PRAD", "source_cohort": "S2",
             "processing_pipeline": "fixture", "n_samples": 3},
        ]
    ).to_parquet(tmp_path / "provenance.parquet", index=False)
    (tmp_path / "_manifest.json").write_text('{"canonical_gene_ids": true}\n')
    accessors._load_precomputed_cohort_views.cache_clear()
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: tmp_path)

    v = cohort_expression_views(genes=["TP53"])

    assert v.tpm.columns.tolist() == ["Ensembl_Gene_ID", "Symbol", "CLL"]
    assert v.provenance["source_cohort"].tolist() == ["S1"]
