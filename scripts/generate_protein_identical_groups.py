"""Generate the protein-identical gene-group table (derived data).

Two distinct Ensembl gene loci sometimes encode the **identical protein** —
segmental-duplication paralogs (NPIPA9, USP17L*), tandem gene clusters that
produce one protein (histones H4C*/H3C*/H2*, the CT47A cancer-testis cluster),
and reassigned/duplicate gene models (SOD2-class). When a quantifier assigns
reads per-locus, each copy gets a *fraction* of the true signal, so any
expression read as a **protein-abundance proxy** is under-counted unless the
protein-identical loci are summed back together.

This script walks the newest installed GRCh38 Ensembl release and groups
protein-coding genes whose **canonical (longest) protein sequence is byte
-identical** (length >= ``MIN_AA`` to avoid micro-peptide / shared-fragment
artefacts that would otherwise bridge unrelated genes). It writes
``pirlygenes/data/protein-identical-gene-groups.csv`` — the curated/derived
input to :func:`pirlygenes.expression.protein_groups.collapse_protein_identical_loci`.

Re-run after an Ensembl release bump:

    python scripts/generate_protein_identical_groups.py

Note this groups by PROTEIN identity only — CDS / UTR differences are allowed
(synonymous substitutions still encode the same protein), which is the right
grain for a protein-abundance proxy. Paralog families that differ at the protein
level (PSG*, CSH*, most MAGE/GAGE members) are deliberately NOT grouped.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

MIN_AA = 30
OUT = Path(__file__).resolve().parent.parent / "pirlygenes" / "data" / \
    "protein-identical-gene-groups.csv"


def _newest_installed_grch38_release() -> int | None:
    """Highest GRCh38 release whose gene annotation actually loads."""
    for release in range(199, 75, -1):
        try:
            rel = EnsemblRelease(release)
            rel.gene_ids()  # forces the DB to load; raises if not downloaded
            return release
        except Exception:
            continue
    return None


def _canonical_protein_by_gene(data: EnsemblRelease) -> dict[str, tuple[str, str]]:
    """``{gene_id: (longest_protein_sequence, symbol)}`` for protein-coding
    genes whose longest protein is at least ``MIN_AA`` residues."""
    out: dict[str, tuple[str, str]] = {}
    for gene in data.genes():
        if gene.biotype != "protein_coding":
            continue
        best = None
        for tx in gene.transcripts:
            try:
                prot = tx.protein_sequence
            except Exception:
                prot = None
            if prot and (best is None or len(prot) > len(best)):
                best = prot
        if best and len(best) >= MIN_AA:
            out[gene.gene_id] = (best, gene.gene_name or "")
    return out


def build_groups(release: int) -> pd.DataFrame:
    data = EnsemblRelease(release)
    canon = _canonical_protein_by_gene(data)
    by_protein: dict[str, set[str]] = defaultdict(set)
    for gene_id, (prot, _sym) in canon.items():
        by_protein[prot].add(gene_id)

    rows = []
    for prot, gene_ids in by_protein.items():
        if len(gene_ids) < 2:
            continue
        members = sorted(gene_ids)              # deterministic
        canonical = members[0]                  # smallest accession = representative
        canonical_symbol = canon[canonical][1]
        for member in members:
            rows.append({
                "group_canonical_ensembl_gene_id": canonical,
                "group_canonical_symbol": canonical_symbol,
                "ensembl_gene_id": member,
                "symbol": canon[member][1],
                "protein_aa": len(prot),
                "n_members": len(members),
            })
    df = pd.DataFrame(rows).sort_values(
        ["protein_aa", "group_canonical_ensembl_gene_id", "ensembl_gene_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return df


def main() -> int:
    release = _newest_installed_grch38_release()
    if release is None:
        raise SystemExit("no installed GRCh38 Ensembl release found")
    df = build_groups(release)
    df.to_csv(OUT, index=False)
    n_groups = df["group_canonical_ensembl_gene_id"].nunique()
    print(f"release {release}: {n_groups} protein-identical groups, "
          f"{len(df)} member genes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
