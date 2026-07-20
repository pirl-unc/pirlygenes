"""General, cohort-backed sibling-entity evidence from pirlygenes #266."""

import pandas as pd
import pytest

from pirlygenes import (
    cancer_type_discriminator,
    cancer_type_discriminators_df,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import degenerate_subtype_pairs_df


NEW_CONTRASTS = {
    "CML_vs_LAML": ("CML", "LAML"),
    "MPN_vs_LAML": ("MPN", "LAML"),
    "B_ALL_vs_LAML": ("B_ALL", "LAML"),
    "FL_vs_LAML": ("FL", "LAML"),
    "STAD_vs_ESCA": ("STAD", "ESCA"),
    "STAD_vs_CHOL": ("STAD", "CHOL"),
    "ESCA_vs_CHOL": ("ESCA", "CHOL"),
}


@pytest.fixture(scope="module")
def representative_expression():
    codes = sorted({code for pair in NEW_CONTRASTS.values() for code in pair})
    return representative_cohort_samples(codes, k=5, format="long")


def _representative_wins(rows, expression):
    """Count which cited program fits each real representative best.

    Within-pair ranks make this a scale-free coherence check, not a production
    classifier or a new absolute marker threshold.
    """
    codes = set(rows["type_a"]) | set(rows["type_b"])
    observed = expression[
        expression["cancer_code"].isin(codes)
        & expression["Symbol"].isin(rows["Symbol"])
    ].copy()
    observed["rank"] = observed.groupby("Symbol")["expression"].rank(
        method="average",
        pct=True,
    )

    wins = {code: 0 for code in codes}
    for (expected, _), sample in observed.groupby(
        ["cancer_code", "representative_id"]
    ):
        rank_by_symbol = dict(zip(sample["Symbol"], sample["rank"]))
        scores = {}
        for favored, markers in rows.groupby("favors"):
            weighted_score = 0.0
            total_weight = 0.0
            for marker in markers.itertuples(index=False):
                weight = 2.0 if marker.tier == "primary" else 1.0
                rank = rank_by_symbol[marker.Symbol]
                fit = rank if marker.direction == "high" else 1.0 - rank
                weighted_score += weight * fit
                total_weight += weight
            scores[favored] = weighted_score / total_weight
        winner = max(scores, key=scores.get)
        wins[expected] += int(winner == expected)
    return wins


def test_sibling_programs_are_two_sided_and_honest_about_separability():
    discriminators = cancer_type_discriminators_df()
    new_rows = discriminators[discriminators["contrast"].isin(NEW_CONTRASTS)]

    assert set(new_rows["contrast"]) == set(NEW_CONTRASTS)
    assert set(new_rows["support_type"]) == {"sibling_discriminator_literature"}
    assert new_rows["source"].str.match(r"PMID:\d+").all()
    assert set(new_rows["direction"]) == {"high", "low"}

    within_myeloid = {"CML_vs_LAML", "MPN_vs_LAML"}
    for contrast, pair in NEW_CONTRASTS.items():
        rows = new_rows[new_rows["contrast"] == contrast]
        assert set(rows["favors"]) == set(pair)
        for code in pair:
            favored = rows[rows["favors"] == code]
            primary_high = favored[
                (favored["tier"] == "primary")
                & (favored["direction"] == "high")
            ]
            assert len(primary_high) >= 2
        if contrast not in within_myeloid:
            assert set(rows["direction"]) == {"high", "low"}

    separability = new_rows.groupby("contrast")["separability"].first().to_dict()
    assert separability["B_ALL_vs_LAML"] == "strong"
    assert separability["FL_vs_LAML"] == "strong"
    assert separability["STAD_vs_CHOL"] == "moderate"
    assert separability["CML_vs_LAML"] == "poor"
    assert separability["MPN_vs_LAML"] == "poor"
    assert separability["STAD_vs_ESCA"] == "poor"
    assert separability["ESCA_vs_CHOL"] == "poor"


def test_parent_programs_cover_observed_risk_and_molecular_children():
    expected_matches = {
        ("LAML_ELNadv", "CML"): {"LAML", "CML"},
        ("LAML_ELNadv", "MPN"): {"LAML", "MPN"},
        ("LAML_ELNint", "B_ALL"): {"LAML", "B_ALL"},
        ("LAML_APL", "FL"): {"LAML", "FL"},
        ("LAML_ELNfav", "FL"): {"LAML", "FL"},
        ("STAD_CIN", "ESCA"): {"STAD", "ESCA"},
        ("STAD_EBV", "CHOL"): {"STAD", "CHOL"},
    }

    for requested_pair, matched_pair in expected_matches.items():
        inherited = cancer_type_discriminator(
            *requested_pair,
            ancestor_fallback=True,
        )
        assert set(inherited) == matched_pair
        reversed_inherited = cancer_type_discriminator(
            *reversed(requested_pair),
            ancestor_fallback=True,
        )
        assert inherited == reversed_inherited

    # The fallback is opt-in, preserving the exact-pair public contract.
    assert cancer_type_discriminator("LAML_ELNadv", "CML") == {}


def test_programs_cohere_with_real_reference_representatives(
    representative_expression,
):
    for contrast, pair in NEW_CONTRASTS.items():
        rows = cancer_type_discriminators_df(*pair)
        wins = _representative_wins(rows, representative_expression)
        minimum = 4 if rows["separability"].iloc[0] == "strong" else 3
        assert wins[pair[0]] >= minimum, (contrast, wins)
        assert wins[pair[1]] >= minimum, (contrast, wins)


def test_colon_rectum_children_remain_site_resolved_not_marker_resolved():
    pairs = degenerate_subtype_pairs_df().set_index("pair_id")
    colorectal = pairs.loc["COAD_vs_READ"]

    assert set(colorectal["members"].split(";")) == {
        "COAD",
        "COAD_MSS",
        "READ",
        "READ_MSS",
    }
    assert colorectal["tiebreaker_rule"] == "site_template"
    assert pd.isna(colorectal["activation_signature"])
    assert colorectal["refs"] == "PMID:22810696"
    assert cancer_type_discriminator(
        "COAD_MSS",
        "READ_MSS",
        ancestor_fallback=True,
    ) == {}
