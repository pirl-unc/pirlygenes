"""Consumer-neutral safety contract for pairwise cancer discriminators."""

import pytest

from pirlygenes import cancer_type_discriminator_consensus
from pirlygenes import gene_sets_cancer as gsc


def _truthy(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def test_all_pairwise_discriminators_are_hypothesis_only():
    discriminators = gsc.cancer_type_discriminators_df()

    assert set(discriminators["evidence_role"]) == {"hypothesis_only"}
    assert not _truthy(discriminators["promote_report_scope"]).any()
    assert set(discriminators["validation_scope"]) == {"pairwise_only"}
    assert set(discriminators["conflict_policy"]) == {"abstain"}


def test_conflicting_gi_nominations_abstain():
    decision = cancer_type_discriminator_consensus(
        {
            "PAAD_vs_STAD": "PAAD",
            "GBC_vs_STAD": "GBC",
        }
    )

    assert decision == {
        "status": "conflict",
        "hypotheses": ("GBC", "PAAD"),
        "non_classification_targets": ("GBC",),
        "promote_report_scope": False,
        "report_code": None,
    }


def test_consistent_pairwise_nominations_remain_hypotheses():
    decision = cancer_type_discriminator_consensus(
        [
            ("PAAD_vs_STAD", "PAAD"),
            ("GBC_vs_PAAD", "PAAD"),
        ]
    )

    assert decision["status"] == "hypothesis_only"
    assert decision["hypotheses"] == ("PAAD",)
    assert decision["non_classification_targets"] == ()
    assert decision["promote_report_scope"] is False
    assert decision["report_code"] is None


def test_empty_pairwise_evidence_has_no_report_code():
    decision = cancer_type_discriminator_consensus({})

    assert decision["status"] == "no_evidence"
    assert decision["hypotheses"] == ()
    assert decision["report_code"] is None


def test_unknown_pairwise_nomination_is_rejected():
    with pytest.raises(ValueError, match="unknown.*GBC_vs_STAD.*PAAD"):
        cancer_type_discriminator_consensus(
            {"GBC_vs_STAD": "PAAD"}
        )


def test_accessor_rejects_pairwise_only_promotion(monkeypatch):
    real_get_data = gsc.get_data
    discriminators = real_get_data("cancer-type-discriminators").copy()
    promoted = discriminators["contrast"].eq("PAAD_vs_STAD")
    discriminators.loc[promoted, "evidence_role"] = "report_scope"
    discriminators.loc[promoted, "promote_report_scope"] = True

    def fake_get_data(name):
        if name == "cancer-type-discriminators":
            return discriminators
        return real_get_data(name)

    monkeypatch.setattr(gsc, "get_data", fake_get_data)
    with pytest.raises(ValueError, match="pairwise-only.*PAAD_vs_STAD"):
        gsc.cancer_type_discriminators_df()


def test_accessor_rejects_non_target_promotion_even_if_joint(monkeypatch):
    real_get_data = gsc.get_data
    discriminators = real_get_data("cancer-type-discriminators").copy()
    promoted = (
        discriminators["contrast"].eq("GBC_vs_STAD")
        & discriminators["favors"].eq("GBC")
    )
    discriminators.loc[promoted, "evidence_role"] = "report_scope"
    discriminators.loc[promoted, "promote_report_scope"] = True
    discriminators.loc[promoted, "validation_scope"] = "joint_multiclass"

    def fake_get_data(name):
        if name == "cancer-type-discriminators":
            return discriminators
        return real_get_data(name)

    monkeypatch.setattr(gsc, "get_data", fake_get_data)
    with pytest.raises(ValueError, match="non-classification.*GBC"):
        gsc.cancer_type_discriminators_df()
