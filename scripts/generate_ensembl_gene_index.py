"""Bundle a static cross-release ENSG -> symbol index (#465).

Canonicalization's name lookups (`find_gene_name_from_ensembl_gene_id`,
used to rescue a retired id by its old symbol) otherwise depend on the
`pirlygenes.gene_ids` union index, which is built from whatever pyensembl
releases happen to be installed locally (the `[index] Loaded … union of N
releases` log line, and a heavy multi-release install requirement).

This snapshots that ENSG -> symbol map once, at build time, into a small
bundled CSV so the runtime canonicalizer needs no live pyensembl install.
Newest installed release wins per id; releases whose GTF db is missing are
skipped.

    python scripts/generate_ensembl_gene_index.py
"""

from __future__ import annotations

import pandas as pd

from pirlygenes.gene_ids import (
    collect_all_installed_ensembl_releases,
    strip_version,
)

OUTPUT = "pirlygenes/data/ensembl-gene-index.csv.gz"


def build() -> pd.DataFrame:
    genomes = sorted(
        collect_all_installed_ensembl_releases(),
        key=lambda g: g.release,
        reverse=True,
    )
    id_to_name: dict[str, str] = {}
    for genome in genomes:
        try:
            rows = genome.db.connection.execute(
                "SELECT gene_id, gene_name FROM gene"
            ).fetchall()
        except Exception:
            continue  # release without a usable GTF db
        for gid, name in rows:
            if not gid:
                continue
            ensg = strip_version(str(gid))
            clean = str(name).strip() if name else ""
            if ensg not in id_to_name and clean:
                id_to_name[ensg] = clean
    return pd.DataFrame(
        sorted(id_to_name.items()),
        columns=["ensembl_gene_id", "symbol"],
    )


def main() -> None:
    out = build()
    out.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}: {len(out)} ENSG->symbol across installed releases")


if __name__ == "__main__":
    main()
