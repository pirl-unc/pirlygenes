"""Unit tests for the shared gene-identifier mapping primitives.

Everything here runs against injected fake tables and a fake pyensembl
genome — no network, no pyensembl, no NCBI download. That is the point
of centralizing the logic: each primitive is exercised in isolation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pirlygenes.builders import gene_mapping as gm
from pirlygenes.builders.ncbi_gene_info import (
    GENE_INFO_SYNONYM_CONFIDENCE,
    SymbolAliasCandidate,
    SymbolAliasIndex,
)


# ─── fake pyensembl ──────────────────────────────────────────────────


class FakeGene:
    def __init__(self, gene_id, gene_name):
        self.gene_id = gene_id
        self.gene_name = gene_name


class FakeGenome:
    """Minimal genome: by_name returns a list, by_id raises on miss."""

    def __init__(self, by_name=None, by_id=None):
        self._by_name = by_name or {}
        self._by_id = by_id or {}

    def genes_by_name(self, symbol):
        return self._by_name.get(symbol, [])

    def gene_by_id(self, gene_id):
        if gene_id in self._by_id:
            return self._by_id[gene_id]
        raise ValueError(f"no gene {gene_id}")


# ─── version + id-type primitives ────────────────────────────────────


def test_strip_version():
    assert gm.strip_version("ENSG00000251562.5") == "ENSG00000251562"
    assert gm.strip_version("ENSG00000251562") == "ENSG00000251562"
    assert gm.strip_version(" ENSG00000251562.12 ") == "ENSG00000251562"


def test_detect_id_type():
    assert gm.detect_id_type(["ENSG00000251562.5", "ENSG00000000003"]) == "ensembl"
    assert gm.detect_id_type(["780", "5982", "99999"]) == "entrez"
    assert gm.detect_id_type(["TP53", "EGFR", "MYC"]) == "hugo"
    assert gm.detect_id_type([]) == "hugo"


# ─── single-gene resolution ──────────────────────────────────────────


def test_gene_from_ensembl_id_strips_and_resolves():
    genome = FakeGenome(by_id={"ENSG00000141510": FakeGene("ENSG00000141510.1", "TP53")})
    assert gm.gene_from_ensembl_id(genome, "ENSG00000141510.17") == (
        "ENSG00000141510",
        "TP53",
    )
    assert gm.gene_from_ensembl_id(genome, "ENSG00000000000") is None
    assert gm.gene_from_ensembl_id(genome, "") is None


def test_gene_from_symbol_unique_only():
    genome = FakeGenome(
        by_name={
            "TP53": [FakeGene("ENSG00000141510.1", "TP53")],
            "AMBIG": [
                FakeGene("ENSG00000000001.1", "AMBIG"),
                FakeGene("ENSG00000000002.1", "AMBIG"),
            ],
        }
    )
    assert gm.gene_from_symbol(genome, "TP53") == ("ENSG00000141510", "TP53")
    assert gm.gene_from_symbol(genome, "AMBIG") is None  # ambiguous → None
    assert gm.gene_from_symbol(genome, "NOPE") is None
    assert gm.gene_from_symbol(genome, "  ") is None


# ─── Entrez resolution tiers ─────────────────────────────────────────


def _patch_entrez(monkeypatch, *, to_ensembl=None, to_symbol=None, history=None):
    monkeypatch.setattr(gm, "cached_entrez_to_ensembl", lambda: to_ensembl or {})
    monkeypatch.setattr(gm, "cached_entrez_to_symbol", lambda: to_symbol or {})
    monkeypatch.setattr(gm, "cached_entrez_history", lambda: history or {})


def test_entrez_to_gene_tier1_dbxrefs(monkeypatch):
    _patch_entrez(
        monkeypatch,
        to_ensembl={"3122": "ENSG00000204287"},
        to_symbol={"3122": "HLA-DRA"},
    )
    genome = FakeGenome()  # dbXrefs path needs no pyensembl
    assert gm.entrez_to_gene(genome, "3122") == (
        "ENSG00000204287",
        "HLA-DRA",
        gm.METHOD_ENTREZ_DBXREFS,
    )


def test_entrez_to_gene_tier2_current_symbol(monkeypatch):
    # No dbXrefs ENSG, but the current symbol resolves uniquely in pyensembl.
    _patch_entrez(monkeypatch, to_symbol={"100": "SEPTIN2"})
    genome = FakeGenome(by_name={"SEPTIN2": [FakeGene("ENSG00000168385.1", "SEPTIN2")]})
    assert gm.entrez_to_gene(genome, "100", legacy_symbol="SEPT2") == (
        "ENSG00000168385",
        "SEPTIN2",
        gm.METHOD_ENTREZ_CURRENT_SYMBOL,
    )


def test_entrez_to_gene_tier3_gene_history(monkeypatch):
    # Original ID is dead; history redirects to a live ID with dbXrefs.
    _patch_entrez(
        monkeypatch,
        to_ensembl={"999": "ENSG00000123456"},
        to_symbol={"999": "GENEX"},
        history={"111": "999"},
    )
    genome = FakeGenome()
    assert gm.entrez_to_gene(genome, "111") == (
        "ENSG00000123456",
        "GENEX",
        gm.METHOD_GENE_HISTORY,
    )


def test_entrez_to_gene_unresolved(monkeypatch):
    _patch_entrez(monkeypatch)
    assert gm.entrez_to_gene(FakeGenome(), "404") is None
    assert gm.entrez_to_gene(FakeGenome(), "") is None


# ─── combined alias pool (NCBI + curated display names) ──────────────


def test_combined_alias_index_includes_curated_display(monkeypatch):
    base = SymbolAliasIndex(
        official_symbols=frozenset({"CTAG1B", "CD274"}),
        alias_candidates={
            "OLD": (SymbolAliasCandidate("CTAG1B", "ncbi", GENE_INFO_SYNONYM_CONFIDENCE),),
        },
    )
    monkeypatch.setattr(gm, "cached_symbol_alias_index", lambda: base)
    gm.cached_combined_alias_index.cache_clear()
    try:
        index = gm.cached_combined_alias_index()
        # Curated display label resolves back to its official symbol.
        targets = {c.official_symbol for c in index.alias_candidates["NY-ESO-1"]}
        assert "CTAG1B" in targets
        assert {c.official_symbol for c in index.alias_candidates["PD-L1"]} == {"CD274"}
        # Pre-existing NCBI candidates are preserved.
        assert index.alias_candidates["OLD"][0].official_symbol == "CTAG1B"
    finally:
        gm.cached_combined_alias_index.cache_clear()


def test_synonym_to_official_guards():
    index = SymbolAliasIndex(
        official_symbols=frozenset({"TTTY21", "GENE1", "GENE2"}),
        alias_candidates={
            "TTTY21": (SymbolAliasCandidate("TTTY7B", "t", GENE_INFO_SYNONYM_CONFIDENCE),),
            "OLD": (SymbolAliasCandidate("GENE1", "t", GENE_INFO_SYNONYM_CONFIDENCE),),
            "AMBIG": (
                SymbolAliasCandidate("GENE1", "t", GENE_INFO_SYNONYM_CONFIDENCE),
                SymbolAliasCandidate("GENE2", "t", GENE_INFO_SYNONYM_CONFIDENCE),
            ),
            "WEAK": (SymbolAliasCandidate("GENE1", "t", 10),),
        },
    )
    assert gm.synonym_to_official("TTTY21", index) is None  # live official owns it
    assert gm.synonym_to_official("OLD", index) == "GENE1"
    assert gm.synonym_to_official("AMBIG", index) is None
    assert gm.synonym_to_official("WEAK", index) is None


# ─── rescue_symbol composition ───────────────────────────────────────


def test_rescue_symbol_via_entrez(monkeypatch):
    _patch_entrez(monkeypatch, to_ensembl={"7": "ENSG00000000007"}, to_symbol={"7": "GENE7"})
    genome = FakeGenome()
    assert gm.rescue_symbol(genome, "LEGACY7", entrez_id="7") == (
        "ENSG00000000007",
        "GENE7",
        gm.METHOD_ENTREZ_DBXREFS,
    )


def test_rescue_symbol_via_curated_display_name(monkeypatch):
    # "NY-ESO-1" is not a live official symbol but maps to CTAG1B via curated.
    base = SymbolAliasIndex(
        official_symbols=frozenset({"CTAG1B"}),
        alias_candidates={},
    )
    monkeypatch.setattr(gm, "cached_symbol_alias_index", lambda: base)
    gm.cached_combined_alias_index.cache_clear()
    genome = FakeGenome(by_name={"CTAG1B": [FakeGene("ENSG00000184033.1", "CTAG1B")]})
    try:
        assert gm.rescue_symbol(genome, "NY-ESO-1") == (
            "ENSG00000184033",
            "CTAG1B",
            gm.METHOD_SYNONYM,
        )
    finally:
        gm.cached_combined_alias_index.cache_clear()


# ─── resolve_symbol: the one entry point every builder uses ──────────


def test_resolve_symbol_direct_hit_returns_symbol_method(monkeypatch):
    """A direct pyensembl hit short-circuits — no Entrez cache load."""
    def fail():
        raise AssertionError("Entrez cache must not load on a direct hit")
    monkeypatch.setattr(gm, "cached_entrez_to_symbol", fail)
    monkeypatch.setattr(gm, "cached_entrez_to_ensembl", fail)
    genome = FakeGenome(by_name={"TP53": [FakeGene("ENSG00000141510.1", "TP53")]})
    assert gm.resolve_symbol(genome, "TP53") == (
        "ENSG00000141510", "TP53", gm.METHOD_SYMBOL,
    )


def test_resolve_symbol_synonym_fallback_without_entrez(monkeypatch):
    """A renamed symbol with no Entrez ID resolves via the synonym pool
    and never touches the Entrez caches — the uniform path that makes
    HIST1H1T→H1-6 resolve the same in every builder."""
    base = SymbolAliasIndex(
        official_symbols=frozenset({"H1-6"}),
        alias_candidates={
            "HIST1H1T": (
                SymbolAliasCandidate("H1-6", "ncbi", GENE_INFO_SYNONYM_CONFIDENCE),
            ),
        },
    )
    monkeypatch.setattr(gm, "cached_symbol_alias_index", lambda: base)

    def fail():
        raise AssertionError("Entrez cache must not load for symbol-only resolve")
    monkeypatch.setattr(gm, "cached_entrez_to_symbol", fail)
    monkeypatch.setattr(gm, "cached_entrez_to_ensembl", fail)
    monkeypatch.setattr(gm, "cached_entrez_history", fail)

    gm.cached_combined_alias_index.cache_clear()
    genome = FakeGenome(by_name={"H1-6": [FakeGene("ENSG00000187475.1", "H1-6")]})
    try:
        assert gm.resolve_symbol(genome, "HIST1H1T") == (
            "ENSG00000187475", "H1-6", gm.METHOD_SYNONYM,
        )
    finally:
        gm.cached_combined_alias_index.cache_clear()


def test_resolve_symbol_uses_entrez_when_id_present(monkeypatch):
    """When a direct hit fails but an Entrez ID is supplied, the chain runs."""
    _patch_entrez(monkeypatch, to_ensembl={"9": "ENSG00000000009"}, to_symbol={"9": "GENE9"})
    genome = FakeGenome()  # direct lookup misses
    assert gm.resolve_symbol(genome, "LEGACY9", entrez_id="9") == (
        "ENSG00000000009", "GENE9", gm.METHOD_ENTREZ_DBXREFS,
    )


# ─── matrix aggregation ──────────────────────────────────────────────


def test_aggregate_matrix_by_mapping_sums_collisions():
    matrix = pd.DataFrame(
        {"s1": [1.0, 2.0, 4.0], "s2": [10.0, 20.0, 40.0]},
        index=pd.Index(["A", "B", "C"], name="source_id"),
    )
    mapping = pd.DataFrame(
        {
            "source_id": ["A", "B", "C"],
            "Ensembl_Gene_ID": ["ENSG1", "ENSG1", "ENSG2"],  # A+B collide
            "Symbol": ["G1", "G1", "G2"],
        }
    )
    agg = gm.aggregate_matrix_by_mapping(matrix, mapping)
    assert agg.loc["ENSG1", "s1"] == 3.0
    assert agg.loc["ENSG1", "s2"] == 30.0
    assert agg.loc["ENSG2", "s1"] == 4.0


def test_aggregate_matrix_by_mapping_empty():
    matrix = pd.DataFrame({"s1": [1.0]}, index=["A"])
    agg = gm.aggregate_matrix_by_mapping(matrix, pd.DataFrame())
    assert list(agg.columns) == ["s1"]
    assert agg.empty
