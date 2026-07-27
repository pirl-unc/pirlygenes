"""Cross-table safety contract for rare-cancer RNA surrogate rules."""

import pytest

from pirlygenes import gene_sets_cancer as gsc


def _truthy(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def test_rna_surrogate_roles_match_registry_classification_targets():
    rules = gsc.rare_cancer_rna_surrogate_rules_df()
    registry = gsc.get_data("cancer-type-registry.csv").set_index("code")

    promotes = _truthy(rules["promote_report_scope"])
    assert promotes.equals(rules["evidence_role"].eq("report_scope"))

    is_target = _truthy(
        rules["cancer_code"].map(registry["is_classification_target"])
    )
    assert not (promotes & ~is_target).any()


def test_acinic_nr4a3_is_hypothesis_only():
    rules = gsc.rare_cancer_rna_surrogate_rules_df().set_index("rule_id")
    acinic = rules.loc["acinic_nr4a3"]

    assert acinic["evidence_role"] == "hypothesis_only"
    assert not bool(acinic["promote_report_scope"])


def test_accessor_fails_closed_on_non_target_promotion(monkeypatch):
    real_get_data = gsc.get_data
    rules = real_get_data("rare-cancer-rna-surrogates").copy()
    registry = real_get_data("cancer-type-registry.csv").copy()
    acinic = rules["rule_id"].eq("acinic_nr4a3")
    rules.loc[acinic, "promote_report_scope"] = True
    rules.loc[acinic, "evidence_role"] = "report_scope"
    registry.loc[registry["code"].eq("ACINIC"), "is_classification_target"] = False

    def fake_get_data(name):
        if name == "rare-cancer-rna-surrogates":
            return rules
        if name == "cancer-type-registry.csv":
            return registry
        return real_get_data(name)

    monkeypatch.setattr(gsc, "get_data", fake_get_data)
    with pytest.raises(ValueError, match="non-classification.*acinic_nr4a3"):
        gsc.rare_cancer_rna_surrogate_rules_df()
