import numpy as np
import pandas as pd
import pytest

from pirlygenes.gene_canonicalization import (
    CANONICAL_GENE_MAP_VERSION,
    GeneIdentitySpaceViolation,
    canonical_gene_id,
    canonical_gene_id_map,
    canonical_gene_space_report,
    canonical_proteoform_id,
    canonical_proteoform_id_map,
    canonicalize_gene_table,
    validate_canonical_gene_table,
)
from pirlygenes.load_dataset import get_data


def test_canonical_gene_id_uses_bundled_ensembl_alias_map():
    aliases = get_data("ensembl-id-aliases")
    row = next(
        row for _, row in aliases.iterrows()
        if canonical_gene_id(row["alt_haplotype_id"]) == row["primary_contig_id"]
    )
    assert canonical_gene_id(row["alt_haplotype_id"]) == row["primary_contig_id"]
    assert (
        canonical_gene_id(f"{row['alt_haplotype_id']}.12")
        == row["primary_contig_id"]
    )


def test_canonical_gene_id_map_is_versioned():
    m = canonical_gene_id_map()
    assert not m.empty
    assert set(m["map_version"]) == {CANONICAL_GENE_MAP_VERSION}
    assert {
        "source_identifier",
        "canonical_gene_id",
        "canonical_symbol",
        "mapping_source",
    } <= set(m.columns)
    assert m["canonical_gene_id"].notna().all()


def test_canonicalize_gene_table_collapses_same_id_symbol_drift():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000141510"],
            "Symbol": ["p53_old_alias", "TP53"],
            "cohort_a": [1.0, np.nan],
            "cohort_b": [np.nan, 2.0],
        }
    )
    out = canonicalize_gene_table(df, value_cols=["cohort_a", "cohort_b"])
    assert len(out) == 1
    assert out["Ensembl_Gene_ID"].iloc[0] == "ENSG00000141510"
    assert out["cohort_a"].iloc[0] == 1.0
    assert out["cohort_b"].iloc[0] == 2.0


def test_canonicalize_gene_table_maps_old_ids_into_authority_release():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000277113", "ENSG00000196539"],
            "Symbol": ["OR2T3", "OR2T3"],
            "source_version": ["gencode-v36", "ensembl-112"],
            "cohort_a": [1.0, 2.0],
        }
    )
    out = canonicalize_gene_table(df, value_cols=["cohort_a"])
    assert len(out) == 1
    assert out["Ensembl_Gene_ID"].iloc[0] == "ENSG00000196539"
    assert out["Symbol"].iloc[0] == "OR2T3"
    assert out["cohort_a"].iloc[0] == 3.0


def test_validate_catches_duplicate_canonical_id_contexts():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000141510"],
            "Symbol": ["TP53", "TP53"],
            "cancer_code": ["A", "A"],
        }
    )
    with pytest.raises(GeneIdentitySpaceViolation, match="duplicate"):
        validate_canonical_gene_table(df, context_cols=["cancer_code"])


def test_validate_catches_versioned_ids_and_proteoform_leakage():
    versioned = pd.DataFrame(
        {"Ensembl_Gene_ID": ["ENSG00000141510.17"], "Symbol": ["TP53"]}
    )
    with pytest.raises(GeneIdentitySpaceViolation, match="versioned"):
        validate_canonical_gene_table(versioned)

    proteoform = pd.DataFrame(
        {"Ensembl_Gene_ID": ["CTAG1A/B"], "Symbol": ["CTAG1A/B"]}
    )
    with pytest.raises(GeneIdentitySpaceViolation, match="outside"):
        validate_canonical_gene_table(proteoform)
    assert validate_canonical_gene_table(
        proteoform, allow_proteoform_ids=True
    ).n_invalid_ids == 0

    arbitrary_symbol = pd.DataFrame(
        {"Ensembl_Gene_ID": ["TP53"], "Symbol": ["TP53"]}
    )
    with pytest.raises(GeneIdentitySpaceViolation, match="outside"):
        validate_canonical_gene_table(
            arbitrary_symbol, allow_proteoform_ids=True
        )


def test_report_surfaces_raw_ensg_symbols_without_failing_by_default():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "Symbol": ["ENSG00000141510"],
        }
    )
    report = canonical_gene_space_report(df)
    assert report.n_symbol_fallback_ids == 1
    with pytest.raises(GeneIdentitySpaceViolation, match="raw ENSG as Symbol"):
        validate_canonical_gene_table(df, forbid_symbol_fallback_ids=True)


def test_validate_catches_ids_outside_authority_release():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000999999"],
            "Symbol": ["made_up_gene"],
        }
    )
    report = canonical_gene_space_report(df)
    assert report.n_invalid_ids == 1
    with pytest.raises(GeneIdentitySpaceViolation, match="outside"):
        validate_canonical_gene_table(df)


def test_canonical_proteoform_id_uses_existing_protein_space():
    assert canonical_proteoform_id("ENSG00000184033") == "CTAG1A/B"
    assert canonical_proteoform_id("ENSG00000141510") == "ENSG00000141510"


def test_canonical_proteoform_id_map_is_versioned():
    m = canonical_proteoform_id_map(kind="protein")
    assert not m.empty
    assert {
        "map_version",
        "proteoform_kind",
        "canonical_gene_id",
        "proteoform_id",
    } <= set(m.columns)
