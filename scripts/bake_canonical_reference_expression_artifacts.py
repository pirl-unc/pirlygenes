"""Bake canonical gene IDs into reference-expression bundle artifacts.

This is the DATA_VERSION migration companion for the offline canonicalization
work: convert bundled artifacts from "raw source gene IDs plus read-time
canonicalization" to "canonical IDs on disk".

The summary shards and representative vectors can be collapsed safely by
summing linear TPM-like columns per canonical ENSG. Percentile vectors are
validated and rewritten only when canonicalization is one-to-one within a
cohort; if a percentile shard would need a merge, regenerate it from per-sample
source matrices instead of summing percentiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pirlygenes.expression.stats import REFERENCE_COLUMNS
from pirlygenes.gene_canonicalization import canonicalize_gene_table
from pirlygenes.version import DATA_VERSION


DATA_DIR = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"
SUMMARY_DIR = DATA_DIR / "cancer-reference-expression"
REPRESENTATIVES_DIR = DATA_DIR / "cancer-reference-expression-representatives"
PERCENTILES_DIR = DATA_DIR / "cancer-reference-expression-percentiles"
MANIFEST = "_manifest.json"


def _write_manifest(root: Path, artifact: str, extra: dict | None = None) -> None:
    payload = {
        "artifact": artifact,
        "data_version": DATA_VERSION,
        "canonical_gene_ids": True,
        "format": 1,
    }
    if extra:
        payload.update(extra)
    (root / MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def bake_summary_shards(root: Path = SUMMARY_DIR) -> None:
    value_cols = [c for c in REFERENCE_COLUMNS if c.startswith("TPM_")]
    n_files = 0
    before_rows = 0
    after_rows = 0
    for path in sorted(root.glob("*.csv.gz")):
        df = pd.read_csv(path, low_memory=False)
        baked = canonicalize_gene_table(
            df,
            group_keys=["cancer_code", "source_cohort"],
            value_cols=value_cols,
            max_cols=["n_detected"],
        )
        baked = baked.reindex(columns=list(REFERENCE_COLUMNS))
        baked.to_csv(path, index=False, compression="gzip")
        n_files += 1
        before_rows += len(df)
        after_rows += len(baked)
        print(f"  summary {path.name}: {len(df)} -> {len(baked)} rows", flush=True)
    _write_manifest(
        root,
        "cancer-reference-expression",
        {"files": n_files, "rows_before": before_rows, "rows_after": after_rows},
    )


def bake_representatives(root: Path = REPRESENTATIVES_DIR) -> None:
    n_files = 0
    before_rows = 0
    after_rows = 0
    for path in sorted(root.glob("*.parquet")):
        if path.name.startswith("_"):
            continue
        df = pd.read_parquet(path)
        value_cols = [c for c in df.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
        baked = canonicalize_gene_table(
            df,
            value_cols=value_cols,
            source_version_col=None,
        )
        baked.to_parquet(path, index=False, compression="zstd")
        n_files += 1
        before_rows += len(df)
        after_rows += len(baked)
        print(
            f"  representatives {path.name}: {len(df)} -> {len(baked)} rows",
            flush=True,
        )
    _write_manifest(
        root,
        "cancer-reference-expression-representatives",
        {"files": n_files, "rows_before": before_rows, "rows_after": after_rows},
    )


def bake_percentiles_if_one_to_one(root: Path = PERCENTILES_DIR) -> None:
    n_files = 0
    for path in sorted(root.glob("*.parquet")):
        if path.name.startswith("_"):
            continue
        df = pd.read_parquet(path)
        value_cols = [c for c in df.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
        baked = canonicalize_gene_table(
            df,
            value_cols=value_cols,
            source_version_col=None,
        )
        if len(baked) != len(df):
            raise ValueError(
                f"{path} would collapse {len(df)} percentile rows to "
                f"{len(baked)}; regenerate percentiles from per-sample source "
                "matrices instead of summing percentile breakpoints."
            )
        baked.to_parquet(path, index=False, compression="zstd")
        n_files += 1
        print(f"  percentiles {path.name}: {len(df)} rows", flush=True)
    _write_manifest(
        root,
        "cancer-reference-expression-percentiles",
        {"files": n_files, "percentile_canonicalization": "one_to_one"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-percentiles",
        action="store_true",
        help=(
            "Rewrite percentile parquets only if canonicalization is one-to-one. "
            "If any shard would need a row merge, regenerate percentiles from "
            "per-sample source matrices instead."
        ),
    )
    args = parser.parse_args()

    bake_summary_shards()
    bake_representatives()
    if args.include_percentiles:
        bake_percentiles_if_one_to_one()
    else:
        print(
            "skipped percentiles; run generate_cohort_gene_percentiles.py from "
            "per-sample source matrices, or rerun this script with "
            "--include-percentiles to validate one-to-one-only rewrites",
            flush=True,
        )


if __name__ == "__main__":
    main()
