"""Generate the gene-family CSVs in ``pirlygenes/data/``.

For each installed GRCh38 Ensembl release, this script walks every
annotated gene, applies the symbol-level regex from
:func:`pirlygenes.expression.qc.classify_gene_qc`, and emits one CSV
per gene family with rows:

    Symbol, Ensembl_Gene_ID

Multi-release ENSG history is preserved by emitting one row per
unique ``(Symbol, ENSG)`` pair across all installed releases — e.g.
``MALAT1`` has 3 rows because three different ENSG IDs have been
assigned to it across releases 77–114.

The mitochondrial-DNA family is NOT generated here: ``MT-*`` genes
are curated by hand in ``pirlygenes/data/mitochondrial-genes.csv``
(with a ``Role`` column distinguishing protein-coding, rRNA, and tRNA
entries).

ENSG IDs are stored **unversioned**; a sample input carrying a
versioned ID (``ENSG00000251562.5``) should strip the suffix before
lookup.

Run from the pirlygenes repo root:

    python scripts/generate_gene_family_sets.py [--releases 100-114]

Re-run after the regex panel in ``pirlygenes/expression/qc.py``
changes — these CSVs are derived data, not curated.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

REPO_ROOT = Path(__file__).resolve().parent.parent

from pirlygenes.expression.qc import classify_gene_qc


# QC-classifier group → on-disk slug. ``mt_dna`` is intentionally
# absent: it's curated by hand in mitochondrial-genes.csv with a
# semantic ``Role`` column the regex can't reproduce.
GROUP_TO_SLUG = {
    "mt_like_pseudogene": "numt-pseudogenes",
    "rrna_like": "rrna-and-pseudogenes",
    "ribosomal_protein": "ribosomal-protein-genes",
    "ribosomal_protein_pseudogene": "ribosomal-protein-pseudogenes",
    "small_ncrna": "small-noncoding-rnas",
    "histone": "histone-genes",
    "hemoglobin": "hemoglobin-genes",
    "immune_receptor": "immune-receptor-segments",
    "polyadenylation_bias_lncrna": "nuclear-retained-lncrnas",
}


def _installed_grch38_releases() -> list[int]:
    """All GRCh38 Ensembl releases with a BUILT GTF database on disk — i.e.
    releases we can read WITHOUT triggering a (multi-GB) FTP download.

    We look for pyensembl's parsed ``*.gtf.db`` files in its own cache
    directory. The cache root is obtained from pyensembl itself
    (``download_cache.cache_directory_path``), so this honours
    ``PYENSEMBL_CACHE_DIR`` and platform cache roots. We deliberately do NOT
    use ``EnsemblRelease.gtf_path`` (it returns ``None`` until a download is
    attempted in recent pyensembl, which made this function silently report
    *zero* releases — a no-op regeneration), and we do NOT probe via
    ``genes()`` (that auto-downloads any release merely listed). Only releases
    whose GTF DB is already present are returned.
    """
    try:
        per_release_dir = Path(
            EnsemblRelease(111).download_cache.cache_directory_path)
        grch38_root = per_release_dir.parent
    except Exception:
        return []
    rels: set[int] = set()
    for db in grch38_root.glob("ensembl*/Homo_sapiens.GRCh38.*.gtf.db"):
        m = re.search(r"GRCh38\.(\d+)\.gtf\.db$", db.name)
        if m:
            rels.add(int(m.group(1)))
    return sorted(rels)


def _parse_releases(spec: str | None) -> list[int]:
    available = _installed_grch38_releases()
    if not spec:
        return available
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            for r in range(int(lo), int(hi) + 1):
                if r in available:
                    out.add(r)
        else:
            r = int(chunk)
            if r in available:
                out.add(r)
    return sorted(out)


def _walk_release(release: int) -> list[tuple[str, str]] | None:
    """Yield (unversioned_ensg, symbol) for every gene in the release.

    Returns ``None`` if the release's pyensembl GTF database isn't
    actually populated (so the caller can skip cleanly).
    """
    rel = EnsemblRelease(release)
    try:
        genes = list(rel.genes())
    except ValueError as exc:
        if "needs to be created" in str(exc):
            return None
        raise
    rows: list[tuple[str, str]] = []
    for g in genes:
        ensg = (g.gene_id or "").split(".")[0]
        symbol = (g.gene_name or "").strip()
        if not ensg or not symbol:
            continue
        rows.append((ensg, symbol))
    return rows


def build_family_tables(releases: list[int]) -> dict[str, pd.DataFrame]:
    """Return ``{slug: DataFrame}`` covering all genes in any release."""
    # key = (ensg, symbol); value = (group, set_of_releases)
    bucket: dict[tuple[str, str], list[str]] = defaultdict(list)
    used_releases: list[int] = []
    for release in releases:
        print(f"  scanning Ensembl release {release}...", flush=True)
        rows = _walk_release(release)
        if rows is None:
            print(
                f"    (release {release} GTF database not installed — skipping)",
                flush=True,
            )
            continue
        used_releases.append(release)
        for ensg, symbol in rows:
            qc = classify_gene_qc(symbol)
            if qc.group not in GROUP_TO_SLUG:
                continue
            bucket[(ensg, symbol.upper())].append(qc.group)
    if not used_releases:
        raise RuntimeError("No Ensembl release had a usable GTF database.")
    print(f"  used releases: {used_releases}", flush=True)

    # First-match-wins resolution: if a single (ENSG, Symbol) pair
    # mapped to multiple QC groups across releases (HGNC rename of
    # one ID across families — e.g. ENSG00000237973 was MTCO1P12 and
    # MIR6723 in different releases), pick the first group in the
    # canonical priority order below.
    PRIORITY = list(GROUP_TO_SLUG.keys())
    grouped: dict[str, list[dict]] = defaultdict(list)
    for (ensg, symbol), groups in bucket.items():
        # symbol's group across the releases where it appeared is
        # already deduped via the GROUP_TO_SLUG filter; pick the
        # highest-priority one for that (ENSG, Symbol).
        seen = set(groups)
        for g in PRIORITY:
            if g in seen:
                chosen = g
                break
        else:
            continue
        grouped[GROUP_TO_SLUG[chosen]].append(
            {"Symbol": symbol, "Ensembl_Gene_ID": ensg}
        )

    out: dict[str, pd.DataFrame] = {}
    for slug, rows in grouped.items():
        df = pd.DataFrame(rows).sort_values(["Symbol", "Ensembl_Gene_ID"]).reset_index(drop=True)
        out[slug] = df
    return out


def _existing_row_count(path: Path) -> int | None:
    """Data-row count (excluding header) of an existing CSV, or ``None``."""
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def shrinking_families(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[tuple[str, int, int]]:
    """``[(slug, existing_rows, new_rows)]`` for families whose regenerated table
    has FEWER rows than the committed file. These CSVs are a cross-release UNION
    of ``(Symbol, ENSG)`` pairs, so a shrink almost always means the run saw
    fewer installed releases than the committed data was built from — i.e. it
    would silently DROP historical ENSG mappings. The caller refuses to write in
    that case (override with ``--allow-shrink``)."""
    shrunk = []
    for slug, df in tables.items():
        existing = _existing_row_count(out_dir / f"{slug}.csv")
        if existing is not None and len(df) < existing:
            shrunk.append((slug, existing, len(df)))
    return sorted(shrunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--releases",
        help="Comma-separated list / range of Ensembl releases (e.g. '100-114'). Default: all installed.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "pirlygenes" / "data"),
        help="Where to write the gene-family CSVs (default: pirlygenes/data/).",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="Permit overwriting a family CSV with FEWER rows than the committed "
             "file. Off by default: these CSVs are a cross-release union, so a "
             "shrink usually means too few Ensembl releases are installed and the "
             "run would drop historical ENSG mappings.",
    )
    args = parser.parse_args(argv)

    releases = _parse_releases(args.releases)
    if not releases:
        print("No Ensembl releases with a built GTF database were found. Install "
              "them first, e.g. `pyensembl install --release 77 100 ... --species "
              "homo_sapiens` — listing a release alone is not enough; its GTF DB "
              "must be built.", file=sys.stderr)
        return 2
    print(f"Using Ensembl releases: {releases}", flush=True)

    tables = build_family_tables(releases)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shrunk = shrinking_families(tables, out_dir)
    if shrunk and not args.allow_shrink:
        print("\nREFUSING to write — regeneration would SHRINK the cross-release "
              "union (fewer installed releases than the committed data was built "
              "from). Install the missing GTF databases and re-run, or pass "
              "--allow-shrink if this is genuinely intended:", file=sys.stderr)
        for slug, old, new in shrunk:
            print(f"  {slug}: {old} -> {new} rows", file=sys.stderr)
        return 3

    for slug, df in tables.items():
        path = out_dir / f"{slug}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path.name}: {len(df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
