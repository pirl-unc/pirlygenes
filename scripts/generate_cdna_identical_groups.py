"""Generate the cDNA-identical gene-group table (derived data).

The principled, *universal* proteoform-collapse criterion for a read-recovery
fix is **identical coding sequence** (cDNA), not merely identical protein. When
two loci share a byte-identical canonical CDS a short-read quantifier genuinely
cannot assign reads between them — they multi-map and each locus is split /
under-counted, so only the SUM is reliable. (Protein-identical-but-cDNA-distinct
loci — histone clusters, tubulins — have synonymous differences, so reads ARE
assignable and they must NOT be collapsed here; those are handled, if wanted,
via a small curated override.)

This walks the newest installed GRCh38 Ensembl release and groups protein-coding
genes whose **canonical (longest) coding sequence is byte-identical** (length
>= ``MIN_CDS`` nt) into ``pirlygenes/data/cdna-identical-gene-groups.csv`` — the
transcriptome-wide input to the read-recovery collapse
(:func:`pirlygenes.expression.protein_groups.collapse_cdna_identical_loci_long`).

Re-run after an Ensembl release bump:

    python scripts/generate_cdna_identical_groups.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

MIN_CDS = 90  # nt (>= 30 codons), mirrors the protein generator's 30-aa floor
OUT = Path(__file__).resolve().parent.parent / "pirlygenes" / "data" / \
    "cdna-identical-gene-groups.csv"


def _newest_installed_grch38_release() -> int | None:
    for release in range(199, 75, -1):
        try:
            rel = EnsemblRelease(release)
            rel.gene_ids()
            return release
        except Exception:
            continue
    return None


def _canonical_cds_by_gene(data: EnsemblRelease) -> dict[str, tuple[str, str]]:
    """``{gene_id: (longest_coding_sequence, symbol)}`` for protein-coding genes
    whose longest CDS is at least ``MIN_CDS`` nt."""
    out: dict[str, tuple[str, str]] = {}
    for gene in data.genes():
        if gene.biotype != "protein_coding":
            continue
        best = None
        for tx in gene.transcripts:
            try:
                cds = tx.coding_sequence
            except Exception:
                cds = None
            if cds and (best is None or len(cds) > len(best)):
                best = cds
        if best and len(best) >= MIN_CDS:
            out[gene.gene_id] = (best, gene.gene_name or "")
    return out


def build_groups(release: int) -> pd.DataFrame:
    data = EnsemblRelease(release)
    canon = _canonical_cds_by_gene(data)
    by_cds: dict[str, set[str]] = defaultdict(set)
    for gene_id, (cds, _sym) in canon.items():
        by_cds[cds].add(gene_id)

    rows = []
    for cds, gene_ids in by_cds.items():
        if len(gene_ids) < 2:
            continue
        members = sorted(gene_ids)
        canonical = members[0]
        canonical_symbol = canon[canonical][1]
        for member in members:
            rows.append({
                "group_canonical_ensembl_gene_id": canonical,
                "group_canonical_symbol": canonical_symbol,
                "ensembl_gene_id": member,
                "symbol": canon[member][1],
                "cds_nt": len(cds),
                "n_members": len(members),
            })
    return pd.DataFrame(rows).sort_values(
        ["cds_nt", "group_canonical_ensembl_gene_id", "ensembl_gene_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def main() -> int:
    release = _newest_installed_grch38_release()
    if release is None:
        raise SystemExit("no installed GRCh38 Ensembl release found")
    df = build_groups(release)
    df.to_csv(OUT, index=False)
    n = df["group_canonical_ensembl_gene_id"].nunique()
    print(f"release {release}: {n} cDNA-identical groups, "
          f"{len(df)} member genes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
