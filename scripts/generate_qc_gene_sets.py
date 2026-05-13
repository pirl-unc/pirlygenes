"""Generate the QC feature-classification CSVs in pirlygenes/data/.

For each installed GRCh38 Ensembl release, this script walks every
annotated gene, applies the symbol-level QC regex from
:func:`trufflepig.expression_qc.classify_gene_qc`, and emits one CSV
per QC group with rows:

    Ensembl_Gene_ID, Symbol, qc_group, qc_label, ensembl_releases, biotypes

Where ``ensembl_releases`` is a ``;``-joined list of release numbers
that mapped this (ENSG, Symbol, qc_group) tuple, and ``biotypes`` is
the matching ``;``-joined biotype values from those releases.

ENSG IDs are stored **unversioned** (e.g. ``ENSG00000251562`` not
``ENSG00000251562.5``); a sample input carrying a versioned ID should
strip the suffix before lookup.

Run from the pirlygenes repo root:

    python scripts/generate_qc_gene_sets.py [--releases 100-114]

Re-run after the regex panel in ``trufflepig/expression_qc.py``
changes — these CSVs are derived data, not curated.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Locate trufflepig.expression_qc — needed to pin the classifier to
# the same regex used at runtime in normalize_expression.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUFFLEPIG_ROOT = REPO_ROOT.parent / "trufflepig"
sys.path.insert(0, str(TRUFFLEPIG_ROOT))

from trufflepig.expression_qc import classify_gene_qc  # noqa: E402

import pandas as pd  # noqa: E402
from pyensembl import EnsemblRelease  # noqa: E402


def _installed_grch38_releases() -> list[int]:
    """All GRCh38 Ensembl releases present in the pyensembl cache."""
    import os

    candidates: set[int] = set()
    home_cache = Path.home() / "Library" / "Caches" / "pyensembl" / "GRCh38"
    xdg_cache = Path.home() / ".cache" / "pyensembl" / "GRCh38"
    for cache in (home_cache, xdg_cache):
        if not cache.is_dir():
            continue
        for entry in os.listdir(cache):
            if entry.startswith("ensembl") and entry[len("ensembl"):].isdigit():
                candidates.add(int(entry[len("ensembl"):]))
    return sorted(candidates)


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


def _walk_release(release: int) -> list[tuple[str, str, str]] | None:
    """Yield (unversioned_ensg, symbol, biotype) for every gene in the release.

    Returns ``None`` if the release's pyensembl GTF database isn't
    actually populated (so the caller can skip cleanly).
    """
    rel = EnsemblRelease(release)
    try:
        genes = list(rel.genes())
    except ValueError as exc:
        # "GTF database needs to be created, run: pyensembl install --release N"
        if "needs to be created" in str(exc):
            return None
        raise
    rows: list[tuple[str, str, str]] = []
    for g in genes:
        ensg = (g.gene_id or "").split(".")[0]
        symbol = (g.gene_name or "").strip()
        biotype = (g.biotype or "").strip()
        if not ensg or not symbol:
            continue
        rows.append((ensg, symbol, biotype))
    return rows


def build_qc_tables(releases: list[int]) -> dict[str, pd.DataFrame]:
    """Return ``{qc_group: DataFrame}`` covering all genes in any release."""
    # key = (ensg, symbol, qc_group); value = {"qc_label", "releases": set, "biotypes": set}
    bucket: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"qc_label": None, "releases": set(), "biotypes": set()}
    )
    used_releases: list[int] = []
    for release in releases:
        print(f"  scanning Ensembl release {release}...", flush=True)
        rows = _walk_release(release)
        if rows is None:
            print(f"    (release {release} GTF database not installed — skipping)", flush=True)
            continue
        used_releases.append(release)
        for ensg, symbol, biotype in rows:
            qc = classify_gene_qc(symbol)
            if qc.group == "other":
                continue
            key = (ensg, symbol.upper(), qc.group)
            entry = bucket[key]
            entry["qc_label"] = qc.label
            entry["releases"].add(release)
            if biotype:
                entry["biotypes"].add(biotype)
    if not used_releases:
        raise RuntimeError("No Ensembl release had a usable GTF database.")
    print(f"  used releases: {used_releases}", flush=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for (ensg, symbol_upper, qc_group), entry in bucket.items():
        grouped[qc_group].append(
            {
                "Ensembl_Gene_ID": ensg,
                "Symbol": symbol_upper,
                "qc_group": qc_group,
                "qc_label": entry["qc_label"],
                "ensembl_releases": ";".join(
                    str(r) for r in sorted(entry["releases"])
                ),
                "biotypes": ";".join(sorted(entry["biotypes"])),
            }
        )

    out: dict[str, pd.DataFrame] = {}
    for qc_group, rows in grouped.items():
        df = pd.DataFrame(rows).sort_values(["Symbol", "Ensembl_Gene_ID"]).reset_index(drop=True)
        out[qc_group] = df
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--releases",
        help="Comma-separated list / range of Ensembl releases (e.g. '100-114'). Default: all installed.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "pirlygenes" / "data"),
        help="Where to write the qc-*.csv files (default: pirlygenes/data/).",
    )
    args = parser.parse_args(argv)

    releases = _parse_releases(args.releases)
    if not releases:
        print("No Ensembl releases available — install via pyensembl first.", file=sys.stderr)
        return 2
    print(f"Using Ensembl releases: {releases}", flush=True)

    tables = build_qc_tables(releases)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # qc_group → on-disk file slug
    slug_map = {
        "mt_dna": "qc-mt-dna",
        "mt_like_pseudogene": "qc-mt-like-pseudogene",
        "rrna_like": "qc-rrna-like",
        "ribosomal_protein": "qc-ribosomal-protein",
        "ribosomal_protein_pseudogene": "qc-ribosomal-protein-pseudogene",
        "small_ncrna": "qc-small-ncrna",
        "histone": "qc-histone",
        "hemoglobin": "qc-hemoglobin",
        "immune_receptor": "qc-immune-receptor",
        "polyadenylation_bias_lncrna": "qc-polya-bias-lncrna",
    }
    for qc_group, df in tables.items():
        slug = slug_map.get(qc_group, f"qc-{qc_group.replace('_', '-')}")
        path = out_dir / f"{slug}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path.name}: {len(df)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
