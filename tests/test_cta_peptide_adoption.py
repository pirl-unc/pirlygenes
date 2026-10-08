"""Exercise analysis consumers against the released owner identity/cache logic."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from oncoref import cta_peptides, gene_identity, peptides

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))
import cta_patient_counts as counts  # noqa: E402

PRAME = "ENSG00000185686"
ALTERNATE_PRAME = "ENSG00000275013"
TP53 = "ENSG00000141510"
UNKNOWN = "ENSG99999999999"


class Annotation:
    """Small synthetic sequence fixture using real, shipped PRAME alias evidence."""

    release = 112
    reference_name = "GRCh38"
    species = SimpleNamespace(latin_name="homo_sapiens")

    def __init__(self, *, alternate_contig="HSCHR22_1_CTG1", unknown=False,
                 identical_non_cta=False):
        self.genes = {
            PRAME: SimpleNamespace(gene_name="PRAME", contig="22"),
            ALTERNATE_PRAME: SimpleNamespace(gene_name="PRAME", contig=alternate_contig),
            TP53: SimpleNamespace(gene_name="TP53", contig="17"),
        }
        self.sequences = {
            PRAME: "ACDEFGHIKLMN", ALTERNATE_PRAME: "ACDEFGHIKLMN",
            TP53: "EFGHIKLMNPQR",  # retains one genuine shared 9-mer
        }
        if unknown:
            self.genes[UNKNOWN] = SimpleNamespace(gene_name="PRAME", contig="ALT_UNKNOWN")
            self.sequences[UNKNOWN] = self.sequences[PRAME]
        if identical_non_cta:
            self.sequences[TP53] = self.sequences[PRAME]

    def gene_ids(self):
        return list(self.genes)

    def gene_by_id(self, gene_id):
        if gene_id not in self.genes:
            raise ValueError(gene_id)
        return self.genes[gene_id]

    def transcripts(self):
        return [SimpleNamespace(gene_id=gene, biotype="protein_coding", protein_sequence=seq)
                for gene, seq in self.sequences.items()]


@pytest.fixture
def owner_cache(tmp_path, monkeypatch):
    # Only shrink the proteome completeness gate for this synthetic fixture;
    # membership, alias validation, fingerprinting and counting are real owner code.
    monkeypatch.setattr(peptides, "_MIN_PROTEOME_GENES", 1)
    monkeypatch.setattr(peptides, "_COUNTS_CACHE", {})
    cache = tmp_path / "owner"
    cache.mkdir()
    monkeypatch.setattr(peptides, "_derived_cache_dir", lambda: cache)
    return cache


def test_analysis_uses_verified_aliases_and_owner_disk_cache(owner_cache, tmp_path, monkeypatch):
    genome = Annotation()
    monkeypatch.setattr(peptides, "_usable_genome", lambda: genome)
    # Old canonical-only analysis counts must survive untouched, but never win.
    monkeypatch.setattr(counts, "CACHE", tmp_path)
    legacy = tmp_path / "cta_specific_9mers.csv"
    legacy.write_text(f"Ensembl_Gene_ID,Symbol,n_9mers,n_specific_9mers\n{PRAME},PRAME,4,0\n")
    before = legacy.read_bytes()
    result = counts.cta_specific_9mer_counts()
    row = result.set_index("Ensembl_Gene_ID").loc[PRAME]
    assert (row.n_9mers, row.n_specific_9mers) == (4, 3)
    assert ALTERNATE_PRAME not in set(result.Ensembl_Gene_ID)
    assert len(result) == 624
    assert legacy.read_bytes() == before
    assert len(list(owner_cache.glob("*.csv"))) == 1

    identity = gene_identity.resolve_gene_identity(ALTERNATE_PRAME + ".7", genome=genome)
    assert identity.verified and identity.status == "alias"
    assert identity.input_gene_id == ALTERNATE_PRAME + ".7"
    assert identity.source_gene_id == ALTERNATE_PRAME
    assert identity.canonical_gene_id == PRAME
    assert identity.identity_contract_version == 1
    assert identity.annotation_release == 112
    assert identity.aliases_sha256 and identity.canonical_gene_space_sha256

    peptides._COUNTS_CACHE.clear()
    monkeypatch.setattr(genome, "transcripts", lambda: pytest.fail("should reuse owner disk cache"))
    pd.testing.assert_frame_equal(result, counts.cta_specific_9mer_counts())


@pytest.mark.parametrize("kwargs", [
    {"alternate_contig": "22"},  # mapping conflicts with the selected annotation
    {"unknown": True},  # same symbol and sequence do not establish an alias
    {"identical_non_cta": True},  # identical protein does not establish CTA membership
])
def test_unverified_and_non_cta_sources_remain_in_background(owner_cache, monkeypatch, kwargs):
    genome = Annotation(**kwargs)
    monkeypatch.setattr(peptides, "_usable_genome", lambda: genome)
    result = counts.cta_specific_9mer_counts().set_index("Ensembl_Gene_ID")
    assert result.loc[PRAME, "n_specific_9mers"] == 0


def test_annotation_evidence_invalidates_cached_counts(owner_cache, monkeypatch):
    genome = Annotation()
    monkeypatch.setattr(peptides, "_usable_genome", lambda: genome)
    before = counts.cta_specific_9mer_counts().set_index("Ensembl_Gene_ID")
    assert before.loc[PRAME, "n_specific_9mers"] == 3
    genome.genes[ALTERNATE_PRAME].contig = "22"
    after = counts.cta_specific_9mer_counts().set_index("Ensembl_Gene_ID")
    assert after.loc[PRAME, "n_specific_9mers"] == 0
    assert len(list(owner_cache.glob("*.csv"))) == 2


def test_analysis_forwards_k_and_refresh(monkeypatch):
    calls = []

    def owner(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"n_specific_9mers": [1, 3]})

    monkeypatch.setattr(cta_peptides, "cta_specific_9mer_counts", owner)
    assert counts.cta_specific_9mer_counts(k=10, refresh=True).n_specific_9mers.tolist() == [3, 1]
    assert calls == [{"k": 10, "refresh": True}]
