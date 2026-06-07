"""Registry ↔ expression-data consistency (#316).

If a registry row claims a **concrete** expression source (a real
``expression_source`` like Treehouse/TCGA/GEO with a concrete
``source_cohort``), then that code must actually be backed by either a packaged
reference (``available_cancer_expression_references``) or a candidate row in
``cancer-expression-source-candidates`` documenting why it isn't built yet.
Otherwise the registry over-claims data that doesn't exist (the four
SARC_CIC/BCOR/MYOEP/SMARCA4 atoms did this — declared Treehouse-backed with
sample counts but had no rows and no candidate)."""

from pirlygenes.expression import available_cancer_expression_references
from pirlygenes.load_dataset import get_data

# expression_source values that mean "no concrete packaged matrix expected"
_ABSTRACT_SOURCES = {"curated", "computed", "", "nan"}
# source_cohort placeholders that are not a concrete built shard
_ABSTRACT_COHORTS = {"COMPUTED_PAN_SARCOMA", "LITERATURE_CURATED", "", "nan"}


def test_concrete_source_has_reference_or_candidate():
    reg = get_data("cancer-type-registry.csv")
    ref_codes = set(available_cancer_expression_references()["cancer_code"].astype(str))
    cand_codes = set(
        get_data("cancer-expression-source-candidates")["cancer_code"].astype(str))

    violations = []
    for _, r in reg.iterrows():
        source = str(r.get("expression_source") or "").strip()
        cohort = str(r.get("source_cohort") or "").strip()
        if source.lower() in _ABSTRACT_SOURCES:
            continue
        if not cohort or cohort in _ABSTRACT_COHORTS:
            continue
        code = str(r["code"])
        if code not in ref_codes and code not in cand_codes:
            violations.append(f"{code} (source={source!r}, cohort={cohort!r})")

    assert not violations, (
        "Registry rows claim a concrete expression source but have neither a "
        "packaged reference nor a candidate row:\n  " + "\n  ".join(violations)
        + "\n\nEither build/package the reference, or set expression_source to "
        "'curated' and add a candidate row documenting the gap."
    )


def test_sarc_round_cell_atoms_are_consistent():
    """The four atoms from #316 are now consistent: marked curated (not
    Treehouse), with a candidate row recording the unbuilt Treehouse samples."""
    reg = get_data("cancer-type-registry.csv").set_index("code")
    cand = get_data("cancer-expression-source-candidates")
    cand_codes = set(cand["cancer_code"].astype(str))
    for code in ("SARC_CIC", "SARC_BCOR", "SARC_MYOEP", "SARC_SMARCA4"):
        assert reg.loc[code, "expression_source"] == "curated"
        sc = reg.loc[code, "source_cohort"]
        assert sc != sc or str(sc).strip() in ("", "nan")  # NaN / empty
        assert code in cand_codes, f"{code} missing a candidate row"
