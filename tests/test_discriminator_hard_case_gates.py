"""Exact representative discriminator/QC gate from #266 and #326."""

from pirlygenes import (
    cancer_discriminator_hard_cases_df,
    cancer_type_discriminator,
    representative_cohort_samples,
)
from pirlygenes.load_dataset import get_data


EXPECTED_GATE = {
    ("BRCA_rep02", "BRCA", "SARC_EPITH"),
    ("BRCA_Basal_rep02", "BRCA_Basal", "SARC_EPITH"),
    ("BRCA_Basal_rep05", "BRCA_Basal", "SARC_OS"),
    ("GBM_rep03", "GBM", "SARC_LPS_UNSPEC"),
    ("KICH_rep02", "KICH", "SARC_UPS"),
    ("NBL_rep03", "NBL", "MTC"),
    ("NBL_MYCNnonamp_rep02", "NBL_MYCNnonamp", "MTC"),
    ("NET_MIDGUT_rep04", "NET_MIDGUT", "SARC_LMS"),
    ("RB_rep03", "RB", "SARC_LPS_UNSPEC"),
    ("SARC_LPS_UNSPEC_rep02", "SARC_LPS_UNSPEC", "READ_MSS"),
    ("SARC_SYN_rep02", "SARC_SYN", "LAML"),
    ("STAD_GS_rep02", "STAD_GS", "BL"),
    ("THYM_rep04", "THYM", "CTCL"),
}


def test_exact_13_sample_gate_is_stable_and_registry_valid():
    gate = cancer_discriminator_hard_cases_df()
    actual = set(
        gate[["representative_id", "expected_code", "baseline_prediction"]].itertuples(
            index=False, name=None
        )
    )

    assert actual == EXPECTED_GATE
    assert gate["representative_id"].is_unique
    assert gate["must_keep"].all()
    assert set(gate["required_outcome"]) == {"lineage_correct"}
    assert gate["expected_positive_markers"].str.len().gt(0).all()
    assert gate["counter_lineage_markers"].str.len().gt(0).all()

    registry_codes = set(get_data("cancer-type-registry")["code"])
    gate_codes = set(gate["expected_code"]) | set(gate["baseline_prediction"])
    assert gate_codes <= registry_codes

    for row in gate.itertuples(index=False):
        representatives = representative_cohort_samples(
            row.expected_code, k=5, format="wide"
        )
        assert row.representative_id in representatives.columns


def test_every_hard_pair_has_two_sided_positive_and_negative_evidence():
    gate = cancer_discriminator_hard_cases_df()
    discriminators = get_data("cancer-type-discriminators")

    for row in gate.itertuples(index=False):
        pair = {row.expected_code, row.baseline_prediction}
        curated = cancer_type_discriminator(*pair)
        assert set(curated) == pair, row.representative_id
        for code in pair:
            assert any(direction == "high" for _, direction in curated[code])

        pair_rows = discriminators[
            discriminators.apply(
                lambda item: {item["type_a"], item["type_b"]} == pair,
                axis=1,
            )
        ]
        expected_high = set(
            pair_rows.loc[
                (pair_rows["favors"] == row.expected_code)
                & (pair_rows["direction"] == "high"),
                "Symbol",
            ]
        )
        counter_high = set(
            pair_rows.loc[
                (pair_rows["favors"] == row.baseline_prediction)
                & (pair_rows["direction"] == "high"),
                "Symbol",
            ]
        )
        assert set(row.expected_positive_markers.split(";")) <= expected_high
        assert set(row.counter_lineage_markers.split(";")) <= counter_high
        assert "low" in set(pair_rows["direction"]), row.representative_id
        assert set(pair_rows["support_type"]) == {"hard_case_discriminator_literature"}


def test_net_midgut_rep04_remains_a_source_qc_and_discriminator_sentinel():
    gate = cancer_discriminator_hard_cases_df().set_index("representative_id")
    case = gate.loc["NET_MIDGUT_rep04"]

    assert case["resolution_track"] == "source_qc_and_panel"
    assert {"CDX2", "TPH1", "INSM1"} <= set(
        case["expected_positive_markers"].split(";")
    )
    assert {"ACTG2", "MYH11", "LMOD1", "DES"} <= set(
        case["counter_lineage_markers"].split(";")
    )
    assert "issues/326" in case["source_issue"]
