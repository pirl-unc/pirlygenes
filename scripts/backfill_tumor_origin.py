#!/usr/bin/env python
"""Backfill ``tumor_origin`` / ``metastasis_site`` on existing shards.

The v5.4 schema adds primary-vs-metastasis annotation. Legacy shards
have NaN for these columns; this script walks each shard, looks up
its ``source_cohort`` via
:func:`pirlygenes.expression.source_cohort_origin.classify_source_cohort`,
and writes the right values.

Going forward this script is only needed for one-time schema-migration
passes against legacy data — new builders set ``tumor_origin`` per-row
and ``upsert_to_shard`` rejects unset / unrecognised values at write
time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.expression.source_cohort_origin import (
    SELF_ANNOTATED_SOURCES,
    classify_source_cohort,
)
from pirlygenes.expression.stats import REFERENCE_COLUMNS


def backfill_shard(path: Path, *, dry_run: bool = False) -> dict[str, int]:
    df = pd.read_csv(path, low_memory=False)
    if "source_cohort" not in df.columns:
        return {"skipped_no_source_cohort": 1}
    # Each shard holds one source_cohort by construction.
    sc_vals = df["source_cohort"].dropna().unique()
    if len(sc_vals) != 1:
        return {"skipped_multi_source_cohort": 1}
    source_cohort = str(sc_vals[0])
    new_origin, new_metsite = classify_source_cohort(source_cohort)

    if new_origin is None:
        if source_cohort in SELF_ANNOTATED_SOURCES:
            return {
                "preserved_self_annotated": 1,
                "source_cohort": source_cohort,
            }
        return {"skipped_unclassified": 1, "source_cohort": source_cohort}

    # Don't overwrite values that the builder already set explicitly.
    has_origin = (
        "tumor_origin" in df.columns
        and df["tumor_origin"].notna().any()
    )
    if has_origin:
        return {
            "skipped_already_set": 1,
            "source_cohort": source_cohort,
            "existing_origin": str(df["tumor_origin"].dropna().iloc[0]),
        }

    df["tumor_origin"] = new_origin
    df["metastasis_site"] = new_metsite if new_metsite is not None else pd.NA
    df = df.reindex(columns=list(REFERENCE_COLUMNS))

    if not dry_run:
        df.to_csv(path, index=False, compression="gzip")
    return {
        "updated": len(df),
        "source_cohort": source_cohort,
        "tumor_origin": new_origin,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shards = sorted(args.shard_dir.glob("*.csv.gz"))
    print(f"backfilling {len(shards)} shard(s) in {args.shard_dir}...")
    updated = preserved = skipped = unclassified = 0
    for shard in shards:
        result = backfill_shard(shard, dry_run=args.dry_run)
        tag = ", ".join(f"{k}={v}" for k, v in result.items())
        print(f"  {shard.name}: {tag}")
        if "updated" in result:
            updated += 1
        elif "preserved_self_annotated" in result:
            preserved += 1
        elif "skipped_unclassified" in result:
            unclassified += 1
        else:
            skipped += 1
    print(
        f"done. updated={updated}, preserved={preserved}, "
        f"skipped={skipped}, unclassified={unclassified} "
        f"(dry_run={args.dry_run})"
    )
    if unclassified:
        print(
            "  unclassified shards need a manual entry in "
            "PRIMARY_SOURCES / MIXED_SOURCES / SELF_ANNOTATED_SOURCES."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
