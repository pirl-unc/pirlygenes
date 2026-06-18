"""Generate the bundled canonical-gene authority reference (#465).

Canonicalization must not depend on which pyensembl releases happen to be
installed at *runtime* (installs are patchy: missing GTF dbs, duplicate
genome objects, partial downloads).  This script snapshots the pinned
authority release's gene table once, at build time, into a small bundled
CSV so the runtime authority is offline, versioned, and reproducible —
the same pattern as ``ncbi-symbol-synonyms.csv.gz`` and
``ensembl-id-aliases.csv``.

Output ``pirlygenes/data/canonical-gene-reference.csv.gz`` has one row per
unversioned authority-release gene:

    ensembl_gene_id, symbol, contig, biotype

``pirlygenes.gene_canonicalization`` loads this for the authority id set,
the unique-symbol→id rescue map, primary-vs-alt-contig decisions, and the
biotype filter — falling back to live pyensembl only if the CSV is absent.

Re-run after bumping ``CANONICAL_ENSEMBL_RELEASE``:

    python scripts/generate_canonical_gene_reference.py
"""

from __future__ import annotations

import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.gene_canonicalization import CANONICAL_ENSEMBL_RELEASE
from pirlygenes.gene_ids import strip_version

OUTPUT = "pirlygenes/data/canonical-gene-reference.csv.gz"
MIN_GENES = 50_000  # guard against a partial/corrupt install


def build() -> pd.DataFrame:
    genome = EnsemblRelease(CANONICAL_ENSEMBL_RELEASE)
    rows = genome.db.connection.execute(
        "SELECT gene_id, gene_name, seqname, gene_biotype FROM gene"
    ).fetchall()
    records = {}
    for gid, name, seqname, biotype in rows:
        if not gid:
            continue
        ensg = strip_version(str(gid))
        # one row per unversioned id; first occurrence wins (stable)
        records.setdefault(
            ensg,
            {
                "ensembl_gene_id": ensg,
                "symbol": (str(name).strip() if name else ""),
                "contig": (str(seqname).strip() if seqname else ""),
                "biotype": (str(biotype).strip() if biotype else ""),
            },
        )
    if len(records) < MIN_GENES:
        raise SystemExit(
            f"authority release {CANONICAL_ENSEMBL_RELEASE} returned only "
            f"{len(records)} genes (<{MIN_GENES}); install/repair it with "
            f"`pyensembl install --release {CANONICAL_ENSEMBL_RELEASE}` before "
            f"regenerating the bundled reference."
        )
    return pd.DataFrame(
        sorted(records.values(), key=lambda r: r["ensembl_gene_id"])
    )


def main() -> None:
    out = build()
    out.to_csv(OUTPUT, index=False)
    print(
        f"wrote {OUTPUT}: {len(out)} genes from Ensembl release "
        f"{CANONICAL_ENSEMBL_RELEASE}"
    )
    print(out["biotype"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
