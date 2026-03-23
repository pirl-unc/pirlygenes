from pathlib import Path

import pandas as pd
import pytest

from pirlygenes import load_dataset as ld
from pirlygenes import gene_sets_cancer as gsc


def test_get_data_from_provided_dict():
    fake = {"abc.csv": pd.DataFrame({"x": [1]})}
    df = ld.get_data("abc", _dataframes_dict=fake)
    assert list(df.columns) == ["x"]


def test_get_data_missing_raises():
    with pytest.raises(ValueError):
        ld.get_data("does-not-exist", _dataframes_dict={})


def test_get_all_csv_paths_contains_core_dataset():
    paths = ld.get_all_csv_paths()
    assert any(Path(p).name == "ADC-trials.csv" for p in paths)


def test_gene_set_field_lookup_variants(monkeypatch):
    # exercises plural/lower/upper/no-underscore candidate expansion
    fake_df = pd.DataFrame({"TUMORTARGETSYMBOLS": ["A;B"]})
    monkeypatch.setattr(gsc, "get_data", lambda name: fake_df)
    out = gsc.get_field_from_gene_set("x", ["Tumor_Target_Symbol"])
    assert out == {"A", "B"}


def test_all_gene_set_wrappers(monkeypatch):
    df_generic = pd.DataFrame(
        {
            "Symbol": ["GENE1;GENE2"],
            "Gene_ID": ["ENSG1;ENSG2"],
            "Tumor_Target_Symbols": ["GENE3"],
            "Tumor_Target_Ensembl_Gene_IDs": ["ENSG3"],
        }
    )

    def fake_get_data(name):
        return df_generic

    monkeypatch.setattr(gsc, "get_data", fake_get_data)

    # ADC
    assert gsc.ADC_trial_target_gene_names()
    assert gsc.ADC_trial_target_gene_ids()
    assert gsc.ADC_approved_target_gene_names()
    assert gsc.ADC_approved_target_gene_ids()
    assert gsc.ADC_target_gene_names()
    assert gsc.ADC_target_gene_ids()

    # TCR-T
    assert gsc.TCR_T_trial_target_get_names()
    assert gsc.TCR_T_trial_target_get_ids()
    assert gsc.TCR_T_target_gene_names()
    assert gsc.TCR_T_target_gene_ids()

    # CAR-T
    assert gsc.CAR_T_approved_target_gene_names()
    assert gsc.CAR_T_approved_target_gene_ids()
    assert gsc.CAR_T_target_gene_names()
    assert gsc.CAR_T_target_gene_ids()

    # MuTE
    assert gsc.multispecific_tcell_engager_trial_target_gene_names()
    assert gsc.multispecific_tcell_engager_trial_target_gene_ids()
    assert gsc.multispecific_tcell_engager_target_gene_names()
    assert gsc.multispecific_tcell_engager_target_gene_ids()

    # Bispecifics
    assert gsc.bispecific_antibody_approved_target_gene_names()
    assert gsc.bispecific_antibody_approved_target_gene_ids()
    assert gsc.bispecific_antibody_target_gene_names()
    assert gsc.bispecific_antibody_targets_gene_ids()

    # Radio + CTA
    assert gsc.radio_target_gene_names()
    assert gsc.radio_target_gene_ids()
    assert gsc.radioligand_target_gene_names()
    assert gsc.radioligand_target_gene_ids()
    assert gsc.CTA_gene_names()
    assert gsc.CTA_gene_ids()


def test_cta_filtered_and_evidence():
    # CTA_gene_names() now returns the filtered (preferred) set
    filtered_names = gsc.CTA_gene_names()
    filtered_ids = gsc.CTA_gene_ids()
    all_names = gsc.CTA_unfiltered_gene_names()
    all_ids = gsc.CTA_unfiltered_gene_ids()
    assert filtered_names
    assert filtered_ids
    assert all_names
    assert all_ids
    assert filtered_names < all_names  # strict subset

    # Backwards-compatible aliases
    assert gsc.CTA_filtered_gene_names() == filtered_names
    assert gsc.CTA_filtered_gene_ids() == filtered_ids

    evidence_df = gsc.CTA_evidence()
    assert len(evidence_df) == len(all_names)
    expected_cols = [
        "protein_reproductive",
        "protein_thymus",
        "protein_reliability",
        "rna_reproductive",
        "rna_thymus",
        "protein_strict_expression",
        "rna_reproductive_frac",
        "rna_reproductive_and_thymus_frac",
        "rna_deflated_reproductive_frac",
        "rna_deflated_reproductive_and_thymus_frac",
        "rna_80_pct_filter",
        "rna_90_pct_filter",
        "rna_95_pct_filter",
        "rna_99_pct_filter",
        "filtered",
        "source_databases",
        "biotype",
        "Canonical_Transcript_ID",
        "rna_max_ntpm",
        "never_expressed",
    ]
    for col in expected_cols:
        assert col in evidence_df.columns, f"Missing column: {col}"

    # filtered column should match CTA_gene_names (the default/filtered set)
    df_filtered_names = set(
        evidence_df[evidence_df["filtered"].astype(str).str.lower() == "true"]["Symbol"]
    )
    assert df_filtered_names == filtered_names
