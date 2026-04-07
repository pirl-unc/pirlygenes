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
    assert gsc.therapy_target_gene_id_to_name("bispecific-antibodies-approved")
    assert gsc.therapy_target_gene_id_to_name("CAR-T-approved")

    # Radio + CTA
    assert gsc.radio_target_gene_names()
    assert gsc.radio_target_gene_ids()
    assert gsc.radioligand_target_gene_names()
    assert gsc.radioligand_target_gene_ids()
    assert gsc.CTA_gene_names()
    assert gsc.CTA_gene_ids()


def test_cta_filtered_and_evidence():
    # CTA_gene_names() = filtered + expressed (excludes never_expressed)
    expressed_names = gsc.CTA_gene_names()
    expressed_ids = gsc.CTA_gene_ids()
    # CTA_filtered includes never_expressed
    filtered_names = gsc.CTA_filtered_gene_names()
    filtered_ids = gsc.CTA_filtered_gene_ids()
    # never_expressed = filtered - expressed
    never_expr_names = gsc.CTA_never_expressed_gene_names()
    never_expr_ids = gsc.CTA_never_expressed_gene_ids()
    # unfiltered = full superset
    all_names = gsc.CTA_unfiltered_gene_names()
    all_ids = gsc.CTA_unfiltered_gene_ids()
    # excluded = fail filter
    excluded_names = gsc.CTA_excluded_gene_names()

    assert expressed_names
    assert filtered_names
    assert all_names
    assert expressed_names < filtered_names  # expressed is strict subset of filtered
    assert filtered_names < all_names  # filtered is strict subset of unfiltered
    assert expressed_names & never_expr_names == set()  # no overlap
    assert expressed_names | never_expr_names == filtered_names  # partition
    assert filtered_names | excluded_names == all_names  # partition
    assert filtered_names & excluded_names == set()  # no overlap

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


def test_cta_gene_id_to_name_preserves_row_pairing():
    mapping = gsc.CTA_gene_id_to_name()
    assert mapping["ENSG00000181323"] == "SPEM1"
    assert mapping["ENSG00000230594"] == "CT47A4"
    assert mapping["ENSG00000236126"] == "CT47A3"


def test_cta_partition():
    # gene_ids
    p = gsc.CTA_partition_gene_ids()
    assert isinstance(p, gsc.CTAPartitionSets)
    assert len(p.cta) > 200
    assert len(p.non_cta) > 15000
    assert p.cta & p.cta_never_expressed == set()
    assert p.cta & p.non_cta == set()
    assert p.cta_never_expressed & p.non_cta == set()

    # gene_names
    p2 = gsc.CTA_partition_gene_names()
    assert isinstance(p2, gsc.CTAPartitionSets)
    assert "MAGEA4" in p2.cta
    assert "TP53" in p2.non_cta

    # dataframes
    p3 = gsc.CTA_partition_dataframes()
    assert isinstance(p3, gsc.CTAPartitionDataFrames)
    assert "rna_deflated_reproductive_frac" in p3.cta.columns
    assert "Ensembl_Gene_ID" in p3.non_cta.columns

    # cta_excluded genes are in non_cta
    excluded_ids = gsc.CTA_excluded_gene_ids()
    assert excluded_ids.issubset(p.non_cta)
