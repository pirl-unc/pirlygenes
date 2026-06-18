import numpy as np
import pandas as pd
import pytest

from pirlygenes.gene_canonicalization import (
    CANONICAL_ENSEMBL_RELEASE,
    CANONICAL_GENE_MAP_VERSION,
    GeneIdentitySpaceViolation,
    canonical_authority_release,
    canonical_gene_biotype,
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


def test_keep_as_self_accepts_well_formed_ensg_outside_authority():
    # A well-formed unversioned ENSG is a valid gene-space key even when it is
    # absent from the pinned authority release: keep-as-self genes (#465) are
    # kept, not dropped or flagged invalid.
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000999999"],
            "Symbol": ["MADEUP"],
        }
    )
    report = canonical_gene_space_report(df)
    assert report.n_invalid_ids == 0
    validate_canonical_gene_table(df)  # does not raise


def test_validate_catches_malformed_gene_ids():
    df = pd.DataFrame({"Ensembl_Gene_ID": ["not-an-ensg"], "Symbol": ["x"]})
    report = canonical_gene_space_report(df)
    assert report.n_invalid_ids == 1
    with pytest.raises(GeneIdentitySpaceViolation, match="outside"):
        validate_canonical_gene_table(df)


def test_keep_as_self_never_drops_well_formed_ensg():
    # canonical_gene_id keeps a well-formed ENSG (version-stripped) rather than
    # returning None, which canonicalize_gene_table would drop.
    assert canonical_gene_id("ENSG00000999999") == "ENSG00000999999"
    assert canonical_gene_id("ENSG00000999999.4") == "ENSG00000999999"


def test_canonical_authority_release_is_pinned():
    # The authority is the bundled offline snapshot's release, not whatever
    # pyensembl happens to be installed locally.
    assert canonical_authority_release() == CANONICAL_ENSEMBL_RELEASE


def test_canonical_gene_biotype_is_offline():
    assert canonical_gene_biotype("ENSG00000141510") == "protein_coding"  # TP53
    assert canonical_gene_biotype("ENSG00000999999") is None


def test_sequence_identity_group_members_collapse_consistently():
    groups = get_data("sequence-identical-gene-groups")
    if groups.empty:
        pytest.skip("sequence-identical-gene-groups not bundled")
    # Every member of a byte-identical-sequence group resolves to the same
    # terminal canonical as its recorded representative — the alias+sequence
    # equivalence closure must not fragment a group (#465).
    sample = groups.head(200)
    member_terminal = sample["member_ensembl_gene_id"].map(canonical_gene_id)
    repr_terminal = sample["canonical_ensembl_gene_id"].map(canonical_gene_id)
    assert (member_terminal == repr_terminal).all()


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


def test_cross_release_gene_name_is_offline():
    from pirlygenes.gene_canonicalization import _cross_release_gene_name

    # The bundled ENSG->symbol snapshot resolves a name without building the
    # pyensembl union index, so the cohort canonicalization path is
    # install-independent (#465).
    assert _cross_release_gene_name("ENSG00000141510") == "TP53"
