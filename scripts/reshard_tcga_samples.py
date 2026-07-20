#!/usr/bin/env python
"""One-shot: split TREEHOUSE_POLYA_25_01_TCGA_SAMPLES into per-code shards.

The combined shard hit 99.47 MiB after the v5.4 schema addition
(tumor_origin + metastasis_site) — within touching distance of
GitHub's 100 MiB hard limit. This script reads it once, groups by
cancer_code, and writes ``TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__<CODE>.csv.gz``
per code via :func:`pirlygenes.expression.stats.write_reference_rows` in
its new ``per_cancer_code_shards=True`` mode. The original combined
file is then deleted.

The reader (:func:`pirlygenes.load_dataset._load_shard_directory`) globs
``*.csv.gz`` and concatenates everything in the directory so no
consumer needs to know about the split.

The Treehouse TCGA sweep (``scripts/sweep_treehouse_tcga_cohorts.py``)
already sets ``per_cancer_code_shards=True`` on its release config
so re-runs preserve the split rather than rebuilding a giant
combined file.

Safe to re-run: idempotent, but if the per-code shards already exist
they get re-written from the combined file (or skipped if combined is
already gone).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.expression.stats import REFERENCE_COLUMNS, write_reference_rows

SHARD_NAME = "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--source-cohort", default=SHARD_NAME)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without writing or deleting.",
    )
    args = parser.parse_args()

    combined_path = args.shard_dir / f"{args.source_cohort}.csv.gz"
    if not combined_path.exists():
        print(f"no combined shard at {combined_path} — already migrated?")
        existing = sorted(
            args.shard_dir.glob(f"{args.source_cohort}__*.csv.gz")
        )
        if existing:
            print(f"  found {len(existing)} existing per-code shards:")
            for p in existing[:5]:
                print(f"    {p.name}")
            if len(existing) > 5:
                print(f"    ... and {len(existing) - 5} more")
        return 0

    print(f"reading {combined_path}...")
    df = pd.read_csv(combined_path, low_memory=False)
    print(f"  rows: {len(df):,}")
    df = df.reindex(columns=list(REFERENCE_COLUMNS))

    codes = sorted(df["cancer_code"].dropna().astype(str).unique())
    print(f"  cancer codes: {len(codes)} ({', '.join(codes[:6])}…)")

    if args.dry_run:
        print("dry-run: would split into per-code shards and delete combined.")
        for code in codes:
            n_rows = int((df["cancer_code"].astype(str) == code).sum())
            print(f"    {args.source_cohort}__{code}.csv.gz  ({n_rows:,} rows)")
        return 0

    print("writing per-code shards via write_reference_rows(per_cancer_code_shards=True)...")
    write_reference_rows(
        args.shard_dir,
        df,
        source_cohort=args.source_cohort,
        cancer_codes=codes,
        per_cancer_code_shards=True,
    )

    # Verify all per-code shards landed before deleting the combined file.
    written = sorted(args.shard_dir.glob(f"{args.source_cohort}__*.csv.gz"))
    if len(written) != len(codes):
        print(
            f"REFUSING to delete combined shard: wrote {len(written)} "
            f"per-code shards but expected {len(codes)}"
        )
        return 1
    total_written = sum(p.stat().st_size for p in written)
    print(
        f"  {len(written)} per-code shards written "
        f"(total {total_written / 1e6:.1f} MB, vs combined "
        f"{combined_path.stat().st_size / 1e6:.1f} MB)"
    )

    print(f"deleting combined shard {combined_path.name}...")
    combined_path.unlink()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
