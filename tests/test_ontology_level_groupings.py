"""Consuming oncoref's ontology_level / ontology_kind grouping signal
(oncoref #322/#323 add the columns; #326 finishes the aggregate tier wiring —
requires oncoref >= 1.8.103).

These accessors are the authoritative "is this a taxonomic grouping" signal
that pirlygenes (and trufflepig, which re-exports these) should gate on
instead of re-deriving groupings from family / mixture_cohort heuristics.
"""

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    computed_union_codes,
    grouping_codes,
    is_grouping,
    mixture_cohort_codes,
)


def test_registry_carries_ontology_columns():
    reg = cancer_type_registry()
    assert {"ontology_level", "ontology_kind"} <= set(reg.columns)


def test_grouping_codes_include_known_aggregates():
    g = set(grouping_codes())
    assert {"SARC", "CRC", "NET", "BTC"} <= g


def test_grouping_excludes_source_scope_subtype_and_primary_types():
    # CRC_MSI is a molecular_source_scope subtype, NOT a taxonomic grouping —
    # the exact distinction oncoref 1.8.95 draws (source pooling != taxonomy).
    assert not is_grouping("CRC_MSI")
    # A primary type is not a grouping either.
    assert not is_grouping("COAD")
    # ...but a real aggregate node is.
    assert is_grouping("SARC")


def test_grouping_distinguishes_source_unions_from_ontology_only_groups():
    # mixture_cohort identifies source-pooling nodes, while ontology_level also
    # permits non-selectable taxonomy-only groups (for example the WHO round-
    # cell sarcoma family added in oncoref 1.8.131). Computed-union groupings
    # must remain poolable; ontology-only groups deliberately need not be.
    groupings = set(grouping_codes())
    mixture = set(mixture_cohort_codes())
    reg = cancer_type_registry().set_index("code")
    computed_groupings = {
        code
        for code in groupings
        if str(reg.loc[code, "ontology_kind"]) == "computed_union"
    }
    ontology_only = groupings - mixture
    assert computed_groupings <= mixture
    assert all(
        str(reg.loc[code, "ontology_kind"]) == "ontology_group"
        for code in ontology_only
    )
    assert "CRC_MSI" in mixture and "CRC_MSI" not in groupings


def test_computed_union_codes_span_grouping_and_type_tiers():
    # oncoref #326 ("aggregate ontology tier wiring") split computed-union
    # aggregates across two ontology levels: top-level taxonomic groupings
    # (SARC/CRC/NET) AND intermediate type-level rollup tiers (SARC_LPS/ESS/RMS,
    # NEC_LUNG). So the computed-union set is no longer a subset of the grouping
    # set — but every member is still a real aggregate node (grouping- or
    # type-level, never a primary leaf), and the grouping-level members are
    # exactly grouping codes.
    cu = set(computed_union_codes())
    assert {"CRC", "NET", "SARC"} <= cu
    reg = cancer_type_registry().set_index("code")
    levels = {c: str(reg.loc[c, "ontology_level"]) for c in cu if c in reg.index}
    assert set(levels.values()) <= {"grouping", "type"}
    grouping_level = {c for c, lvl in levels.items() if lvl == "grouping"}
    assert grouping_level <= set(grouping_codes())
    assert {"SARC", "CRC", "NET"} <= grouping_level
