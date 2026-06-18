"""Generate the gene-level sequence-identity equivalence table (#465).

Multiple Ensembl gene IDs can be *representations of the same underlying
sequence* — alt-haplotype / MHC alt-contig copies of one gene, and the
hundreds of byte-identical copies of small non-coding RNA families
(``U6``, ``Y_RNA``, ``Metazoa_SRP``, ``5S_rRNA`` …).  Joining cohorts on
the raw ENSG keeps every copy as its own sparse row.  This script groups
genes whose spliced cDNA sequence is **byte-identical** and emits a
``member -> canonical`` map so ``pirlygenes.gene_canonicalization`` can
collapse them onto one id and *sum* their TPM (the existing
``canonicalize_gene_table`` sum-collapse does the arithmetic).

This is the cDNA analogue of ``generate_protein_identical_groups.py`` but
across **all biotypes** and **all installed Ensembl releases**, so it also
covers the non-coding and retired-release genes the protein-only table
misses.

Universe = every distinct gene id in ``cancer-reference-expression``.  Each
gene's sequence is read from the newest installed release that carries it
(releases whose GTF/sequence data is not installed are skipped).  Canonical
representative per identical-sequence group: prefer a **primary-contig** id
(so an alt-contig copy folds onto the canonical-contig gene), then the
lexicographically smallest ENSG (deterministic).

    python scripts/generate_sequence_identical_gene_groups.py
"""

from __future__ import annotations

import collections
import hashlib

import pandas as pd

from pirlygenes.gene_ids import (
    collect_all_installed_ensembl_releases,
    strip_version,
)
from pirlygenes.load_dataset import get_data

OUTPUT = "pirlygenes/data/sequence-identical-gene-groups.csv"
PRIMARY_CONTIGS = {str(c) for c in list(range(1, 23)) + ["X", "Y", "MT"]}


def _release_gene_ids(genome) -> set[str]:
    """Unversioned gene ids in one release, or empty if its db is unavailable."""
    try:
        return {
            strip_version(gid)
            for (gid,) in genome.db.connection.execute("SELECT gene_id FROM gene")
        }
    except Exception:
        return set()


def _longest_cdna_hash(gene) -> str | None:
    """sha1 of the gene's longest spliced-transcript cDNA, or None."""
    best = ""
    for transcript in gene.transcripts:
        try:
            seq = transcript.sequence or ""
        except Exception:
            seq = ""
        if len(seq) > len(best):
            best = seq
    return hashlib.sha1(best.encode()).hexdigest() if best else None


def build() -> pd.DataFrame:
    genomes = {g.release: g for g in collect_all_installed_ensembl_releases()}
    order = sorted(genomes, reverse=True)  # newest release wins the sequence

    universe = {
        strip_version(str(x))
        for x in get_data("cancer-reference-expression")["Ensembl_Gene_ID"]
    }
    print(f"universe: {len(universe)} bundle genes")

    seq_hash: dict[str, str] = {}
    contig: dict[str, str] = {}
    symbol: dict[str, str] = {}
    for rel in order:
        genome = genomes[rel]
        present = _release_gene_ids(genome)
        if not present:
            continue
        todo = (universe & present) - seq_hash.keys()
        for ensg in todo:
            try:
                gene = genome.gene_by_id(ensg)
            except Exception:
                continue
            h = _longest_cdna_hash(gene)
            if not h:
                continue
            seq_hash[ensg] = h
            contig[ensg] = str(gene.contig)
            symbol[ensg] = gene.gene_name or ""
        print(f"  r{rel}: sequenced {len(seq_hash)}/{len(universe)}")
        if len(seq_hash) == len(universe):
            break

    groups: dict[str, set[str]] = collections.defaultdict(set)
    for ensg, h in seq_hash.items():
        groups[h].add(ensg)

    def representative(members: set[str]) -> str:
        # primary-contig first (fold alt copies onto the canonical contig),
        # then the smallest stable id for determinism.
        return sorted(
            members,
            key=lambda e: (contig.get(e, "~") not in PRIMARY_CONTIGS, e),
        )[0]

    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        canon = representative(members)
        for member in sorted(members):
            rows.append(
                {
                    "member_ensembl_gene_id": member,
                    "canonical_ensembl_gene_id": canon,
                    "canonical_symbol": symbol.get(canon, ""),
                    "group_size": len(members),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "member_ensembl_gene_id",
            "canonical_ensembl_gene_id",
            "canonical_symbol",
            "group_size",
        ],
    ).sort_values(["canonical_ensembl_gene_id", "member_ensembl_gene_id"])


def main() -> None:
    out = build()
    out.to_csv(OUTPUT, index=False)
    n_groups = out["canonical_ensembl_gene_id"].nunique()
    n_members = len(out)
    print(
        f"\nwrote {OUTPUT}: {n_groups} groups, {n_members} member rows "
        f"({n_members - n_groups} loci fold onto a representative)"
    )
    top = (
        out.drop_duplicates("canonical_ensembl_gene_id")
        .nlargest(12, "group_size")[
            ["canonical_ensembl_gene_id", "canonical_symbol", "group_size"]
        ]
    )
    print("largest sequence-identity groups:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
