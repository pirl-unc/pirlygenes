"""Contract tests for the curated gene-family panels.

These tables are the source of truth that
:func:`trufflepig.expression_qc.classify_gene_qc` reads when falling
back from regex to a stable ENSG lookup. The shapes pinned here are
part of the public cross-package API — changing them without
updating trufflepig will silently break the downstream lookup path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pirlygenes.gene_families import (
    GENE_FAMILIES,
    gene_family_for_ensembl_id,
    gene_family_for_symbol,
    gene_family_ids,
    gene_family_names,
    gene_family_table,
    gene_family_symbols,
    hemoglobin_gene_ids,
    histone_gene_ids,
    hpa_housekeeping_candidates,
    immune_receptor_segment_ids,
    nuclear_retained_lncrna_ids,
    nuclear_retained_lncrna_symbols,
    numt_pseudogene_ids,
    recommended_hpa_housekeeping_panel,
    ribosomal_protein_ids,
    ribosomal_protein_pseudogene_ids,
    rrna_and_pseudogene_ids,
    small_noncoding_rna_ids,
)
from pirlygenes.gene_sets_cancer import housekeeping_gene_ids, housekeeping_gene_names


# ---------- shape / schema ----------


def test_gene_family_table_has_required_columns():
    df = gene_family_table()
    required = {"Symbol", "Ensembl_Gene_ID", "family"}
    missing = required - set(df.columns)
    assert not missing, f"gene_family_table missing columns: {missing}"


def test_gene_family_names_covers_every_shipped_csv():
    names = set(gene_family_names())
    expected = {f.name for f in GENE_FAMILIES}
    assert names == expected


# ---------- typed per-family accessors ----------


def test_nuclear_retained_lncrnas_include_malat1_and_neat1():
    syms = nuclear_retained_lncrna_symbols()
    ids = nuclear_retained_lncrna_ids()
    assert "MALAT1" in syms
    assert "NEAT1" in syms
    assert "ENSG00000251562" in ids  # canonical MALAT1
    assert "ENSG00000245532" in ids  # NEAT1


def test_numt_pseudogenes_have_expected_size_and_shape():
    ids = numt_pseudogene_ids()
    assert 100 < len(ids) < 1000
    # Every NUMT pseudogene ID is a stable, unversioned ENSG.
    assert all(i.startswith("ENSG") and "." not in i for i in ids)


def test_ribosomal_protein_and_pseudogene_sets_both_nonempty():
    """Both subsets should ship and stay well over a triple-digit count
    each. They share a few IDs from HGNC renames where the same ENSG
    appears as ``RPL5`` in one release and ``RPL5P1`` in another —
    ``gene_family_for_ensembl_id`` resolves the overlap by priority."""
    rp = ribosomal_protein_ids()
    rpp = ribosomal_protein_pseudogene_ids()
    assert len(rp) > 50
    assert len(rpp) > 100


def test_rrna_and_pseudogene_set_non_empty():
    ids = rrna_and_pseudogene_ids()
    assert len(ids) >= 100


def test_small_noncoding_rna_set_is_large():
    ids = small_noncoding_rna_ids()
    assert len(ids) >= 1000


def test_histone_and_hemoglobin_sets_have_expected_size():
    assert 50 <= len(histone_gene_ids()) <= 300
    assert 5 <= len(hemoglobin_gene_ids()) <= 30


def test_immune_receptor_set_includes_ig_and_tr_segments():
    ids = immune_receptor_segment_ids()
    syms = gene_family_symbols("immune_receptor_segment")
    assert ids
    assert any(s.startswith("IGH") for s in syms)
    assert any(s.startswith("TR") for s in syms)


# ---------- ENSG / Symbol → family lookup ----------


def test_gene_family_for_ensembl_id_canonical_lookups():
    assert gene_family_for_ensembl_id("ENSG00000251562") == "nuclear_retained_lncrna"
    assert gene_family_for_ensembl_id("ENSG00000245532") == "nuclear_retained_lncrna"
    # MT-CO1 — sourced from the curated mitochondrial-genes.csv
    assert gene_family_for_ensembl_id("ENSG00000198804") == "mitochondrial"


def test_versioned_ensembl_id_is_stripped_to_unversioned():
    plain = gene_family_for_ensembl_id("ENSG00000251562")
    versioned = gene_family_for_ensembl_id("ENSG00000251562.5")
    assert plain == versioned == "nuclear_retained_lncrna"


def test_historic_ensembl_id_still_resolves():
    """MALAT1 had ENSG00000278217 in releases 77-87 (biotype misc_RNA).
    The multi-release walk preserved it so a sample quantified against
    an old annotation still classifies."""
    assert gene_family_for_ensembl_id("ENSG00000278217") == "nuclear_retained_lncrna"


def test_symbol_lookup_is_case_insensitive():
    upper = gene_family_for_symbol("MALAT1")
    lower = gene_family_for_symbol("malat1")
    mixed = gene_family_for_symbol("Malat1")
    assert upper == lower == mixed == "nuclear_retained_lncrna"


def test_unknown_input_returns_none():
    assert gene_family_for_ensembl_id("ENSG99999999999") is None
    assert gene_family_for_ensembl_id("") is None
    assert gene_family_for_ensembl_id(None) is None
    assert gene_family_for_symbol("MYC") is None  # protein-coding/other
    assert gene_family_for_symbol("") is None


# ---------- internal-consistency checks ----------


def test_gene_family_ids_returns_unversioned_only():
    for name in gene_family_names():
        ids = gene_family_ids(name)
        assert all("." not in i for i in ids), (
            f"family {name} contains versioned IDs"
        )


def test_gene_family_symbols_returns_uppercase_only():
    for name in gene_family_names():
        syms = gene_family_symbols(name)
        assert syms == {s.upper() for s in syms}


def test_gene_family_table_idempotent_under_mutation():
    """Internal lru_cache shouldn't leak mutation back to callers."""
    a = gene_family_table()
    a["Ensembl_Gene_ID"] = "X"
    b = gene_family_table()
    assert (b["Ensembl_Gene_ID"] != "X").all()


def test_hpa_housekeeping_derivation_delegates_to_oncoref(monkeypatch):
    import oncoref.gene_families as ogf

    calls = []

    def fake_candidates(*args, **kwargs):
        calls.append(("candidates", args, kwargs))
        return pd.DataFrame({"Symbol": ["A"]})

    def fake_panel(*args, **kwargs):
        calls.append(("panel", args, kwargs))
        return pd.DataFrame({"Symbol": ["B"]})

    monkeypatch.setattr(ogf, "hpa_housekeeping_candidates", fake_candidates)
    monkeypatch.setattr(ogf, "recommended_hpa_housekeeping_panel", fake_panel)

    assert hpa_housekeeping_candidates(min_ntpm=123)["Symbol"].tolist() == ["A"]
    assert recommended_hpa_housekeeping_panel(target_size=3)["Symbol"].tolist() == ["B"]
    assert calls == [
        ("candidates", (), {"min_ntpm": 123}),
        ("panel", (), {"target_size": 3}),
    ]
    assert pytest.importorskip("pirlygenes").recommended_hpa_housekeeping_panel is (
        recommended_hpa_housekeeping_panel
    )


def test_active_housekeeping_panel_is_hpa_derived_core_panel():
    symbols = housekeeping_gene_names()
    ids = housekeeping_gene_ids()
    assert len(symbols) == len(ids) == 30
    assert {"SUMO2", "HSP90AB1", "EEF1A1"} <= symbols
    assert symbols.isdisjoint({"ACTB", "GAPDH", "PPIA", "HSP90AA1", "RPLP0", "RPS18"})
    assert housekeeping_gene_names(core_only=True) == symbols
    assert housekeeping_gene_ids(core_only=True) == ids
