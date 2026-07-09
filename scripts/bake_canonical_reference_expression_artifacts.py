"""Bake canonical gene IDs into reference-expression bundle artifacts.

This is the DATA_VERSION migration companion for the offline canonicalization
work: convert bundled artifacts from "raw source gene IDs plus read-time
canonicalization" to "canonical IDs on disk".

The summary shards are collapsed safely by summing linear TPM-like columns per
canonical ENSG. (Per-sample representative medoids and per-gene percentile
vectors moved to oncoref in pirlygenes#208 and are baked there, not here.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pirlygenes.expression.stats import REFERENCE_COLUMNS
from pirlygenes.gene_canonicalization import canonicalize_gene_table
from pirlygenes.version import DATA_VERSION


DATA_DIR = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"
SUMMARY_DIR = DATA_DIR / "cancer-reference-expression"
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


def main() -> None:
    bake_summary_shards()


if __name__ == "__main__":
    main()
