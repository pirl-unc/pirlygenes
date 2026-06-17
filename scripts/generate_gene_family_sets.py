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


def _grch38_cache_root() -> Path | None:
    """pyensembl's GRCh38 cache directory (the parent of the per-release
    ``ensemblN`` dirs), obtained **release-agnostically**.

    Every ``EnsemblRelease(n)`` computes the same ``<root>/GRCh38/ensemblN``
    path whether or not ``n`` is installed, so we take the newest release number
    pyensembl still *knows* (scanning down from a high bound) and return its
    parent — honouring ``PYENSEMBL_CACHE_DIR`` / platform cache roots. Scanning
    down rather than hardcoding one release means a future pyensembl dropping any
    particular release from its known list can't make this silently fail.
    Returns ``None`` only if pyensembl knows no GRCh38 release at all.
    """
    for n in range(199, 75, -1):
        try:
            return Path(EnsemblRelease(n).download_cache.cache_directory_path).parent
        except Exception:
            continue  # n not a release pyensembl knows; try the next one down
    return None


def _installed_grch38_releases() -> list[int]:
    """All GRCh38 Ensembl releases with a BUILT GTF database on disk — i.e.
    releases we can read WITHOUT triggering a (multi-GB) FTP download.

    Globs pyensembl's own cache (see :func:`_grch38_cache_root`) for parsed
    ``*.gtf.db`` files. We deliberately do NOT use ``EnsemblRelease.gtf_path``
    (it returns ``None`` until a download is attempted in recent pyensembl, which
    made this silently report *zero* releases — a no-op regeneration), and do NOT
    probe via ``genes()`` (that auto-downloads any release merely listed). Only
    releases whose GTF DB is already present are returned.
    """
    root = _grch38_cache_root()
    if root is None:
        print(
            "WARNING: could not locate pyensembl's GRCh38 cache directory — "
            "pyensembl's cache API may have changed; reporting no releases.",
            file=sys.stderr,
        )
        return []
    rels: set[int] = set()
    for db in root.glob("ensembl*/Homo_sapiens.GRCh38.*.gtf.db"):
        m = re.search(r"GRCh38\.(\d+)\.gtf\.db$", db.name)
        if m:
            rels.add(int(m.group(1)))
    return sorted(rels)


def _most_recent_installed_release() -> int | None:
    """Newest GRCh38 Ensembl release with a built GTF database on disk, or
    ``None`` if none are installed."""
    rels = _installed_grch38_releases()
    return rels[-1] if rels else None


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


def shrinking_families(
    tables: dict[str, pd.DataFrame], out_dir: Path, *, known_slugs=None,
) -> list[tuple[str, int, int]]:
    """``[(slug, existing_rows, new_rows)]`` for families whose regenerated table
    has FEWER rows than the committed file — **including a family that
    regenerated to zero rows** (its slug absent from ``tables`` but a committed
    CSV still on disk; ``build_family_tables`` only emits slugs that produced
    rows, so a vanished family would otherwise slip past this guard and leave a
    stale file). These CSVs are a cross-release UNION of ``(Symbol, ENSG)``
    pairs, so a shrink almost always means the run saw fewer installed releases
    than the committed data was built from — i.e. it would silently DROP
    historical ENSG mappings. The caller refuses to write in that case (override
    with ``--allow-shrink``). ``known_slugs`` defaults to every family slug the
    generator emits, so the on-disk set is checked even when ``tables`` omits a
    family entirely."""
    slugs = set(known_slugs if known_slugs is not None else GROUP_TO_SLUG.values())
    slugs |= set(tables)
    shrunk = []
    for slug in slugs:
        existing = _existing_row_count(out_dir / f"{slug}.csv")
        if existing is None:
            continue
        new = len(tables.get(slug, ()))  # 0 if the family produced no rows
        if new < existing:
            shrunk.append((slug, existing, new))
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
