"""Tests for the curated characteristic-fusion reference (cancer-fusions.csv)."""

from pirlygenes.gene_sets_cancer import (
    cancer_fusions,
    cancer_fusions_df,
    fusion_partners,
    protein_family,
    resolve_cancer_type,
)

_EXPECTED_COLS = [
    "cancer_code", "fusion_family",
    "gene_5prime", "gene_5prime_family", "gene_5prime_ensembl_id",
    "gene_3prime", "gene_3prime_family", "gene_3prime_ensembl_id",
    "frequency", "is_defining",
    "pathognomonic", "rnaseq_detectable", "mechanism", "confidence",
    "pmid", "notes",
]


def test_schema_and_codes_resolve():
    df = cancer_fusions_df()
    assert list(df.columns) == _EXPECTED_COLS
    for code in df["cancer_code"].astype(str).unique():
        assert resolve_cancer_type(code) is not None  # every code is a registry code


def test_pathognomonic_implies_defining_and_unique():
    df = cancer_fusions_df()
    patho = df[df["pathognomonic"].astype(bool)]
    assert len(patho) > 0
    assert patho["is_defining"].astype(bool).all()
    # a pathognomonic gene-pair must map to exactly one cancer code
    for r in patho.itertuples():
        codes = set(df[(df.gene_5prime == r.gene_5prime)
                       & (df.gene_3prime == r.gene_3prime)]["cancer_code"])
        assert codes == {r.cancer_code}


def test_known_pathognomonic_and_shared():
    # pathognomonic: EWSR1-WT1 -> DSRCT only; PML-RARA -> APL; SS18-SSX -> synovial
    patho = set(zip(cancer_fusions(pathognomonic_only=True)["gene_5prime"],
                    cancer_fusions(pathognomonic_only=True)["gene_3prime"]))
    assert ("EWSR1", "WT1") in patho
    assert ("PML", "RARA") in patho
    # shared (defining but NOT pathognomonic): ETV6-NTRK3 spans many entities,
    # ZC3H7B-BCOR is in both BCOR-sarcoma and HG-ESS.
    df = cancer_fusions_df()
    etv6 = df[(df.gene_5prime == "ETV6") & (df.gene_3prime == "NTRK3")]
    assert not etv6["pathognomonic"].astype(bool).any()
    assert etv6["cancer_code"].nunique() > 1


def test_pax_and_fox_are_separate_families():
    assert protein_family("PAX3") == "PAX"
    assert protein_family("PAX7") == "PAX"
    assert protein_family("FOXO1") == "FOX"
    assert protein_family("PAX3") != protein_family("FOXO1")
    # FET partners share a family (why they're interchangeable 5' partners)
    assert protein_family("EWSR1") == protein_family("FUS") == "FET"
    assert protein_family("FLI1") == "ETS"


def test_fusion_partners_promiscuous_sets():
    # EWSR1 is a promiscuous 5' partner across sarcoma entities
    ewsr1_3p = fusion_partners("EWSR1", side="5prime")  # 3' partners of EWSR1
    assert {"FLI1", "WT1", "NR4A3", "ATF1", "POU5F1"} <= ewsr1_3p
    # NR4A3 takes multiple 5' partners (EMC)
    nr4a3_5p = fusion_partners("NR4A3", side="3prime")
    assert {"EWSR1", "TAF15", "TCF12"} <= nr4a3_5p


def test_round_cell_entities_present():
    # the entities the gaps surfaced: CIC / BCOR / myoepithelial / NUT
    for code, g5, g3 in [("SARC_CIC", "CIC", "DUX4"),
                         ("SARC_BCOR", "BCOR", "CCNB3"),
                         ("SARC_MYOEP", "EWSR1", "POU5F1"),
                         ("NUTM", "BRD4", "NUTM1")]:
        rows = cancer_fusions(code)
        assert ((rows.gene_5prime == g5) & (rows.gene_3prime == g3)).any()


def test_fusion_negative_entities_marked():
    # leiomyosarcoma / GIST / MTC are fusion-negative -> a single (none) row
    for code in ("SARC_LMS", "SARC_GIST", "MTC"):
        rows = cancer_fusions(code)
        assert (rows["fusion_family"] == "(none)").any()
        assert rows["is_defining"].astype(bool).sum() == 0
