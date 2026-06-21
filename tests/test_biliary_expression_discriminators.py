from pirlygenes.gene_sets_cancer import cancer_type_discriminators_df
from pirlygenes.load_dataset import get_data


def test_biliary_expression_candidates_are_explicit():
    candidates = get_data("cancer-expression-source-candidates").set_index("cancer_code")

    chol = candidates.loc["CHOL"]
    assert chol["source_status"] == "direct_reference_available"
    assert chol["accession"] == "TCGA-CHOL"
    assert "Direct CHOL reference" in chol["notes"]

    gbc = candidates.loc["GBC"]
    assert gbc["source_status"] == "bulk_candidate_ready"
    assert gbc["accession"] == "GSE139682"
    assert "small-n discriminator evidence" in gbc["processing_plan"]


def test_gbc_discriminators_are_pairwise_multi_marker_fallbacks():
    df = cancer_type_discriminators_df()
    expected = {
        "GBC_vs_PAAD": {"GBC", "PAAD"},
        "GBC_vs_STAD": {"GBC", "STAD"},
        "GBC_vs_NET_PANCREAS": {"GBC", "NET_PANCREAS"},
    }

    for contrast, sides in expected.items():
        rows = df[df["contrast"] == contrast]
        assert set(rows["favors"]) == sides
        assert rows.groupby("favors")["Symbol"].nunique().min() >= 3

    gbc_rows = df[df["contrast"].isin(expected) & df["favors"].eq("GBC")]
    assert set(gbc_rows["support_type"]) == {"gallbladder_expression_candidate"}
    assert set(gbc_rows["source"]) == {"GSE139682"}


def test_gbc_discriminators_keep_shared_markers_out_of_family_panels():
    families = get_data("cancer-family-panels")
    assert "BILIARY" not in set(families["Family"])

    gbc_symbols = set(
        cancer_type_discriminators_df("GBC", "PAAD")
        .query("favors == 'GBC'")["Symbol"]
    )
    assert {"KRT7", "KRT19", "CLDN4"} <= gbc_symbols
