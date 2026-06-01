from types import SimpleNamespace

import pytest

import pirlygenes.gene_ids as gi


class FakeTx:
    def __init__(self, is_protein_coding=True):
        self.is_protein_coding = is_protein_coding


class FakeGene:
    def __init__(self, gene_id, name, gene_name=None, num_pc=1):
        self.id = gene_id
        self.name = name
        self.gene_name = gene_name or name
        self.transcripts = [FakeTx(True) for _ in range(num_pc)]

    def __repr__(self):
        return f"FakeGene({self.id},{self.name})"


class FakeGenome:
    def __init__(self, release, gene_by_id_map=None, tx_by_id_map=None, by_name=None):
        self.release = release
        self._gene_by_id_map = gene_by_id_map or {}
        self._tx_by_id_map = tx_by_id_map or {}
        self._by_name = by_name or {}

    def gene_by_id(self, gene_id):
        if gene_id not in self._gene_by_id_map:
            raise KeyError(gene_id)
        return self._gene_by_id_map[gene_id]

    def transcript_by_id(self, tx_id):
        if tx_id not in self._tx_by_id_map:
            raise KeyError(tx_id)
        return self._tx_by_id_map[tx_id]

    def genes(self):
        return list(self._gene_by_id_map.values())

    def transcripts(self):
        return [
            SimpleNamespace(id=tid, gene_name=tx.gene_name)
            for tid, tx in self._tx_by_id_map.items()
        ]

    def genes_by_name(self, name):
        return self._by_name.get(name, [])


def test_strip_version():
    assert gi.strip_version("ENSG00000141510.17") == "ENSG00000141510"
    assert gi.strip_version("ENSG00000141510") == "ENSG00000141510"
    assert gi.strip_version(" ENSG00000141510.3 ") == "ENSG00000141510"


def test_gene_for_ensembl_id_version_tolerant_and_safe():
    gene = FakeGene("ENSG00000141510", "TP53")
    genome = FakeGenome(release=112, gene_by_id_map={"ENSG00000141510": gene})
    # Version suffix is stripped before lookup.
    assert gi.gene_for_ensembl_id(genome, "ENSG00000141510.9") is gene
    # A miss (KeyError from gene_by_id) is swallowed → None, not raised.
    assert gi.gene_for_ensembl_id(genome, "ENSG00000000000") is None


def test_pick_best_gene_prefers_more_protein_coding():
    g1 = FakeGene("ENSG1", "TP53", num_pc=1)
    g2 = FakeGene("ENSG2", "TP53", num_pc=3)
    assert gi.pick_best_gene([g1, g2]).id == "ENSG2"


def test_lookup_functions_with_fake_genomes(monkeypatch, tmp_path):
    gene_a = FakeGene("ENSGA", "GENEA")
    tx_a = SimpleNamespace(gene_name="GENEA")
    by_name_gene = FakeGene("ENSGX", "CD276", gene_name="CD276")
    genomes = [
        FakeGenome(
            release=112,
            gene_by_id_map={"ENSGA": gene_a},
            tx_by_id_map={"ENST1": tx_a},
            by_name={
                "CD276": [by_name_gene],
                "B7-H3": [by_name_gene],
                "cd276": [by_name_gene],
            },
        )
    ]
    monkeypatch.setattr(gi, "genomes", genomes)
    monkeypatch.setattr(gi, "_indexes_built", False)
    monkeypatch.setattr(gi, "_gene_id_to_name", {})
    monkeypatch.setattr(gi, "_transcript_id_to_gene_name", {})
    monkeypatch.setattr(gi, "_gene_id_miss_cache", set())
    monkeypatch.setattr(gi, "_transcript_id_miss_cache", set())
    # Redirect the on-disk pickle cache so the fake-genome's fixture
    # data ({ENSGA: GENEA}) doesn't leak into the user's real
    # ~/.cache/pirlygenes/ensembl-<release>-id-index.pkl. Bug fixed
    # in 5.4.1 after a stale fixture cache was found in the
    # developer's home cache during release prep.
    monkeypatch.setattr(
        gi, "_index_cache_path",
        lambda release: tmp_path / f"fake-ensembl-{release}-id-index.pkl",
    )

    assert gi.find_gene_name_from_ensembl_gene_id("ENSGA", verbose=False) == "GENEA"
    assert (
        gi.find_gene_name_from_ensembl_transcript_id("ENST1", verbose=False) == "GENEA"
    )

    genome, gene = gi.find_gene_and_ensembl_release_by_name("CD276", verbose=False)
    assert genome.release == 112
    assert gene.id == "ENSGX"

    assert gi.find_gene_by_name_from_ensembl("CD276", verbose=False).id == "ENSGX"
    assert gi.find_gene_id_by_name_from_ensembl("CD276", verbose=False) == "ENSGX"
    assert gi.find_canonical_gene_id_and_name("CD276") == ("ENSGX", "CD276")


def test_lookup_functions_fall_back_to_older_release_on_latest_miss(
    monkeypatch, tmp_path,
):
    older_gene = FakeGene("ENSGOLD", "OLDER")
    older_tx = SimpleNamespace(gene_name="OLDER")
    genomes = [
        FakeGenome(release=112),
        FakeGenome(
            release=111,
            gene_by_id_map={"ENSGOLD": older_gene},
            tx_by_id_map={"ENSTOLD": older_tx},
        ),
    ]
    monkeypatch.setattr(gi, "genomes", genomes)
    monkeypatch.setattr(gi, "_indexes_built", False)
    monkeypatch.setattr(gi, "_gene_id_to_name", {})
    monkeypatch.setattr(gi, "_transcript_id_to_gene_name", {})
    monkeypatch.setattr(gi, "_gene_id_miss_cache", set())
    monkeypatch.setattr(gi, "_transcript_id_miss_cache", set())
    # See test_lookup_functions_with_fake_genomes for context.
    monkeypatch.setattr(
        gi, "_index_cache_path",
        lambda release: tmp_path / f"fake-ensembl-{release}-id-index.pkl",
    )

    assert gi.find_gene_name_from_ensembl_gene_id("ENSGOLD", verbose=False) == "OLDER"
    assert (
        gi.find_gene_name_from_ensembl_transcript_id("ENSTOLD", verbose=False)
        == "OLDER"
    )


def test_ncbi_synonym_official_symbol_casing(monkeypatch):
    monkeypatch.setattr(gi, "_ncbi_symbol_synonyms", lambda: {"GNB2L1": "RACK1"})
    assert gi.ncbi_synonym_official_symbol("gnb2l1") == "RACK1"
    assert gi.ncbi_synonym_official_symbol("GNB2L1") == "RACK1"
    assert gi.ncbi_synonym_official_symbol("NOPE") is None


def test_symbol_lookup_lowest_tier_resolves_via_ncbi_synonym(monkeypatch):
    """A legacy symbol unknown to every release resolves via the bundled
    NCBI synonym snapshot (the lowest tier)."""
    rack1 = FakeGene("ENSGRACK1", "RACK1")
    genomes = [FakeGenome(release=112, by_name={"RACK1": [rack1]})]
    monkeypatch.setattr(gi, "genomes", genomes)
    monkeypatch.setattr(gi, "_ncbi_symbol_synonyms", lambda: {"GNB2L1": "RACK1"})

    genome, gene = gi.find_gene_and_ensembl_release_by_name("GNB2L1")
    assert gene.id == "ENSGRACK1"


def test_symbol_lookup_direct_hit_beats_synonym(monkeypatch):
    """The lowest tier fires only after the release loop fails — a direct
    pyensembl match is never overridden by a synonym."""
    direct = FakeGene("ENSGDIRECT", "FOO")
    other = FakeGene("ENSGOTHER", "BAR")
    genomes = [FakeGenome(release=112, by_name={"FOO": [direct], "BAR": [other]})]
    monkeypatch.setattr(gi, "genomes", genomes)
    monkeypatch.setattr(gi, "_ncbi_symbol_synonyms", lambda: {"FOO": "BAR"})

    genome, gene = gi.find_gene_and_ensembl_release_by_name("FOO")
    assert gene.id == "ENSGDIRECT"


def test_bundled_ncbi_synonyms_present_and_sane():
    """The bundled snapshot ships, loads offline, and carries known renames."""
    table = gi._ncbi_symbol_synonyms()
    assert len(table) > 10_000
    assert table.get("GNB2L1") == "RACK1"
    assert table.get("TCEB2") == "ELOB"
    assert table.get("NARS") == "NARS1"


# Real HGNC renames / classic aliases → current official symbol, straight
# from the bundled snapshot (offline, deterministic, no pyensembl). A
# deliberately gnarly spread: elongin renames, the KMT2/MLL family, the
# ATP-synthase and septin mass-renames, KIAA/ORF/FLJ/ODZ names, and the
# myeloma-relevant WHSC1/MMSET→NSD2 and FAM46C→TENT5C.
_SNAPSHOT_RENAMES = [
    ("GNB2L1", "RACK1"), ("TCEB1", "ELOC"), ("TCEB2", "ELOB"),
    ("TCEB3", "ELOA"), ("NARS", "NARS1"), ("HARS", "HARS1"),
    ("CARS", "CARS1"), ("MLL", "KMT2A"), ("MLL3", "KMT2C"),
    ("WHSC1", "NSD2"), ("MMSET", "NSD2"), ("FAM46C", "TENT5C"),
    ("C11orf30", "EMSY"), ("PARK2", "PRKN"), ("ADCK3", "COQ8A"),
    ("MRE11A", "MRE11"), ("CASC5", "KNL1"), ("FIGF", "VEGFD"),
    ("ODZ1", "TENM1"), ("KIAA1524", "CIP2A"), ("ATP5B", "ATP5F1B"),
    ("ATP5A1", "ATP5F1A"), ("SEPT9", "SEPTIN9"), ("MLLT4", "AFDN"),
    ("NHP2L1", "SNU13"), ("HER2", "ERBB2"),
]


@pytest.mark.parametrize("old,official", _SNAPSHOT_RENAMES)
def test_ncbi_synonym_snapshot_renames(old, official):
    assert gi.ncbi_synonym_official_symbol(old) == official


# Genuinely ambiguous historical aliases (pointed at >1 gene) — or
# themselves a live official symbol — must NOT resolve. MLL2 historically
# meant both KMT2B and KMT2D; B3GNT1 split into B4GAT1/B3GNT2.
@pytest.mark.parametrize("ambiguous", ["MLL2", "B3GNT1", "GARS", "PTC"])
def test_ncbi_synonym_snapshot_drops_ambiguous(ambiguous):
    assert gi.ncbi_synonym_official_symbol(ambiguous) is None


def test_find_canonical_gene_ids_and_names(monkeypatch):
    mapping = {
        "A": ("ENSGA", "A"),
        "B": (None, None),
    }
    monkeypatch.setattr(
        gi,
        "find_canonical_gene_id_and_name",
        lambda name: mapping.get(name, (None, None)),
    )
    ids, names = gi.find_canonical_gene_ids_and_names(["A", "B"])
    assert ids == ["ENSGA", None]
    assert names == ["A", None]
