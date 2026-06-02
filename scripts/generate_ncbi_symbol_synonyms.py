"""Regenerate ``pirlygenes/data/ncbi-symbol-synonyms.csv.gz``.

This is the bundled, static snapshot of NCBI ``gene_info`` ``Synonyms``
that backs the *lowest tier* of symbol resolution in
:func:`pirlygenes.gene_ids.find_gene_and_ensembl_release_by_name` — the
last resort when a name matches no installed Ensembl release and is not
a curated display alias (``GNB2L1`` → ``RACK1``, ``TCEB2`` → ``ELOB``,
``NARS`` → ``NARS1``, and the rest of the HGNC-renamed long tail).

It is a *derived* file, like the gene-family CSVs. Re-run after a major
NCBI ``gene_info`` refresh:

    python scripts/generate_ncbi_symbol_synonyms.py

Only **unambiguous, safe** aliases are kept: the single official-symbol
target an alias resolves to, excluding aliases that are themselves a
live official symbol or that point at more than one gene. That pruning
reuses :func:`pirlygenes.builders.gene_mapping.synonym_to_official`, so
the bundled snapshot and the builders' live lookup apply *identical*
guard logic — there is one definition of "safe alias → official".
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

from pirlygenes.builders.gene_mapping import synonym_to_official
from pirlygenes.builders.ncbi_gene_info import cached_symbol_alias_index

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "pirlygenes" / "data" / "ncbi-symbol-synonyms.csv.gz"
)


def build_pairs() -> dict[str, str]:
    index = cached_symbol_alias_index()
    pairs: dict[str, str] = {}
    for alias in index.alias_candidates:
        official = synonym_to_official(alias, index)
        if official:
            pairs[alias] = official
    return pairs


def main() -> None:
    pairs = build_pairs()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "official_symbol"])
        for alias in sorted(pairs):
            writer.writerow([alias, pairs[alias]])
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {len(pairs):,} alias→official pairs to {OUT_PATH} "
          f"({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
