"""Contract tests for the curated QC feature-group panels.

These tables are the source-of-truth that
:func:`trufflepig.expression_qc.classify_gene_qc` reads when it falls
back from regex to a stable ENSG lookup. The shape of
:class:`QcFeatureClass` (the four fields below) is part of the
public cross-package API — changing it without updating trufflepig
will silently break the lookup path.
"""

from __future__ import annotations

import pandas as pd

from pirlygenes.qc_feature_groups import (
    QC_FEATURE_FILES,
    QcFeatureClass,
    qc_class_for_ensembl_id,
    qc_class_for_symbol,
    qc_feature_ensembl_ids,
    qc_feature_groups,
    qc_feature_symbols,
    qc_feature_table,
)


# ---------- shape / schema ----------


def test_qc_feature_class_shape_is_stable_for_downstream_consumers():
    """``trufflepig.expression_qc.classify_gene_qc`` reads these four
    fields. Adding fields is fine; renaming or removing is a breaking
    change to the cross-package contract."""
    hit = qc_class_for_ensembl_id("ENSG00000251562")
    assert isinstance(hit, QcFeatureClass)
    assert isinstance(hit.ensembl_gene_id, str)
    assert isinstance(hit.symbol, str)
    assert isinstance(hit.group, str)
    assert isinstance(hit.label, str)


def test_qc_feature_table_has_required_columns():
    df = qc_feature_table()
    required = {
        "Ensembl_Gene_ID",
        "Symbol",
        "qc_group",
        "qc_label",
        "ensembl_releases",
        "biotypes",
    }
    missing = required - set(df.columns)
    assert not missing, f"qc_feature_table missing columns: {missing}"


def test_qc_feature_groups_returns_one_frame_per_panel_with_rows():
    groups = qc_feature_groups()
    assert set(groups.keys()) <= {name for name, _ in QC_FEATURE_FILES}
    # Every panel that ships a CSV must have rows.
    for name, _path in QC_FEATURE_FILES:
        assert name in groups, f"panel {name} has no rows on disk"
        assert not groups[name].empty


# ---------- canonical ENSG lookups ----------


def test_malat1_ensembl_id_classifies_to_polya_bias_lncrna():
    hit = qc_class_for_ensembl_id("ENSG00000251562")
    assert hit is not None
    assert hit.group == "polyadenylation_bias_lncrna"
    assert hit.symbol == "MALAT1"


def test_neat1_ensembl_id_classifies_to_polya_bias_lncrna():
    hit = qc_class_for_ensembl_id("ENSG00000245532")
    assert hit is not None
    assert hit.group == "polyadenylation_bias_lncrna"
    assert hit.symbol == "NEAT1"


def test_mt_co1_classifies_to_mt_dna_via_ensembl_id_and_symbol():
    by_ensg = qc_class_for_ensembl_id("ENSG00000198804")
    by_sym = qc_class_for_symbol("MT-CO1")
    assert by_ensg is not None and by_ensg.group == "mt_dna"
    assert by_sym is not None and by_sym.group == "mt_dna"
    assert by_ensg.ensembl_gene_id == by_sym.ensembl_gene_id


def test_unknown_lookup_returns_none():
    assert qc_class_for_ensembl_id("ENSG99999999999") is None
    assert qc_class_for_symbol("MYC") is None  # protein-coding/other


def test_blank_input_returns_none():
    assert qc_class_for_ensembl_id("") is None
    assert qc_class_for_ensembl_id(None) is None
    assert qc_class_for_symbol("") is None
    assert qc_class_for_symbol(None) is None


# ---------- version suffix handling ----------


def test_versioned_ensembl_id_is_stripped_to_unversioned():
    """Real-world input matrices (TCGA, GTEx) often carry versioned IDs.
    Lookups must succeed regardless of the ``.N`` suffix."""
    plain = qc_class_for_ensembl_id("ENSG00000251562")
    versioned = qc_class_for_ensembl_id("ENSG00000251562.5")
    assert plain is not None
    assert versioned is not None
    assert plain.ensembl_gene_id == versioned.ensembl_gene_id == "ENSG00000251562"


def test_historic_ensembl_id_still_resolves():
    """MALAT1 had ENSG00000278217 (biotype misc_RNA) in releases 77-87.
    The multi-release walk should preserve it so a sample quantified
    against an old annotation still classifies."""
    historic = qc_class_for_ensembl_id("ENSG00000278217")
    assert historic is not None
    assert historic.group == "polyadenylation_bias_lncrna"


# ---------- symbol lookup ----------


def test_symbol_lookup_is_case_insensitive():
    upper = qc_class_for_symbol("MALAT1")
    lower = qc_class_for_symbol("malat1")
    mixed = qc_class_for_symbol("Malat1")
    assert upper is not None
    assert lower is not None
    assert mixed is not None
    assert upper.group == lower.group == mixed.group


# ---------- group-level helpers ----------


def test_qc_feature_ensembl_ids_returns_unversioned_strings():
    ids = qc_feature_ensembl_ids("mt_dna")
    assert ids
    assert all("." not in i for i in ids)
    assert "ENSG00000198804" in ids  # MT-CO1


def test_qc_feature_symbols_returns_uppercase():
    syms = qc_feature_symbols("polyadenylation_bias_lncrna")
    assert syms == {s.upper() for s in syms}
    assert "MALAT1" in syms
    assert "NEAT1" in syms


def test_multi_group_ensg_resolves_deterministically():
    """Real annotation drift: a few ENSG IDs are HGNC-renamed across
    Ensembl releases (e.g. ENSG00000237973 went from ``MTCO1P12`` to
    ``MIR6723``), so the same ID appears in two QC groups. The runtime
    lookup must pick one deterministically — the first-match-wins
    ordering in :data:`QC_FEATURE_FILES` selects the more
    technical-RNA-like group, which matches what we want at QC time."""
    df = qc_feature_table()
    counts = df.groupby("Ensembl_Gene_ID")["qc_group"].nunique()
    multi_ids = counts[counts > 1].index.tolist()
    # Every multi-group ID must still resolve to exactly one
    # QcFeatureClass via the public accessor.
    for ensg in multi_ids:
        hit = qc_class_for_ensembl_id(ensg)
        assert hit is not None
        assert hit.ensembl_gene_id == ensg


def test_qc_feature_table_is_idempotent_under_repeat_calls():
    """Internal lru_cache shouldn't leak mutation back to callers."""
    a = qc_feature_table()
    a["Ensembl_Gene_ID"] = "X"
    b = qc_feature_table()
    assert (b["Ensembl_Gene_ID"] != "X").all()
