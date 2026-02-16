from pathlib import Path

import pandas as pd
import pytest

import pirlygenes.load_expression as le


def test_get_canonical_gene_name_from_gene_ids_string(monkeypatch):
    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": None}.get(gid),
    )
    assert le.get_canonical_gene_name_from_gene_ids_string("ENSG1;ENSG2") == "A"
    assert le.get_canonical_gene_name_from_gene_ids_string(float("nan")) == ""


def test_load_expression_data_with_existing_ids(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "Gene": ["B7-H3", "TP53"],
            "Ensembl Gene ID": ["ENSG00000103855", "ENSG00000141510"],
            "TPM": [1.0, 2.0],
        }
    ).to_csv(p, index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG00000103855": "CD276", "ENSG00000141510": "TP53"}[gid],
    )

    out = le.load_expression_data(str(p), verbose=False, progress=False)
    assert "canonical_gene_name" in out.columns
    assert "gene_display_name" in out.columns
    assert list(out["canonical_gene_name"]) == ["CD276", "TP53"]
    assert list(out["gene_display_name"]) == ["B7-H3", "p53"]


def test_load_expression_data_without_ensembl_ids(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame({"Gene Symbol": ["A", "B", "C", "D"], "TPM": [1, 2, 3, 4]}).to_csv(
        p, index=False
    )

    monkeypatch.setattr(
        le,
        "find_canonical_gene_ids_and_names",
        lambda genes: (
            ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            ["A1", ["B1", "B2"], None, 123],
        ),
    )
    monkeypatch.setattr(le, "find_gene_name_from_ensembl_gene_id", lambda gid: gid)

    out = le.load_expression_data(str(p), verbose=False, progress=False)
    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    assert list(out["canonical_gene_name"]) == ["A1", "B1;B2", "", "?"]


def test_load_expression_aggregate_and_save(tmp_path, monkeypatch):
    p = tmp_path / "tx.csv"
    out_csv = tmp_path / "rolled.csv"
    pd.DataFrame({"transcript": ["tx1"], "tpm": [1.0]}).to_csv(p, index=False)

    monkeypatch.setattr(
        le,
        "tx2gene",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "gene": ["GENE1"],
                "TPM": [1.0],
                "gene_id": ["ENSG1"],
                "ensembl_release": [112],
            }
        ),
    )
    monkeypatch.setattr(le, "find_gene_name_from_ensembl_gene_id", lambda gid: "GENE1")

    out = le.load_expression_data(
        str(p),
        aggregate_gene_expression=True,
        aggregated_output_path=str(out_csv),
        verbose=False,
        progress=False,
    )
    assert out_csv.exists()
    assert list(out["ensembl_gene_id"]) == ["ENSG1"]


def test_load_expression_error_paths(tmp_path):
    p = tmp_path / "expr.csv"
    pd.DataFrame({"TPM": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        le.load_expression_data(str(p), verbose=False, progress=False)

    with pytest.raises(ValueError):
        le.load_expression_data(str(tmp_path / "expr.bad"), verbose=False, progress=False)
