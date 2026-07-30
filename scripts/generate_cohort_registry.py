#!/usr/bin/env python
"""Snapshot oncoref's first-class cohort vocabulary for pirlygenes.

Oncoref owns both the cohort registry and the empirical reference manifest.
Pirlygenes packages a small compatibility snapshot so existing
``get_data("cohort-registry")`` callers keep working without loading expression
values. The snapshot is copied from the owner and checked against its compact
all-source manifest before it is written.

The manifest reconciliation is intentionally general: when a registry row
advertises fewer cancer codes than the same physical cohort actually exposes,
the manifest's routed code/sample counts replace the stale registry counts.
This currently repairs GSE294016 (oncoref#448), whose owner registry says one
code even though its released summary contains distinct ADCC and ACINIC rows.

Run after changing the pinned oncoref release:

    python scripts/generate_cohort_registry.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA = Path("pirlygenes/data")
OUTPUT = DATA / "cohort-registry.csv"


def _owner_registry_snapshot() -> pd.DataFrame:
    import oncoref

    registry = oncoref.cohort_registry_df().copy()
    manifest = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="all",
        reference_source="summary_rows_all",
        all_sources=True,
    )
    manifest = manifest.loc[
        manifest["available"] & manifest["source_cohort"].notna()
    ].drop_duplicates(["source_cohort", "cancer_code"])
    routed = (
        manifest.groupby("source_cohort", sort=False)
        .agg(
            routed_samples=("n_reference_samples", "sum"),
            routed_codes=("cancer_code", "nunique"),
            source_version=(
                "source_version",
                lambda values: "; ".join(
                    dict.fromkeys(
                        str(value)
                        for value in values.dropna()
                        if str(value).strip()
                    )
                ),
            ),
        )
    )

    joined = registry[["cohort_id", "n_samples", "n_codes"]].merge(
        routed,
        left_on="cohort_id",
        right_index=True,
        how="left",
    )
    stale = joined["routed_codes"].gt(joined["n_codes"])
    for row in joined.loc[stale].itertuples(index=False):
        mask = registry["cohort_id"].astype(str).eq(str(row.cohort_id))
        registry.loc[mask, "n_samples"] = int(row.routed_samples)
        registry.loc[mask, "n_codes"] = int(row.routed_codes)
        if str(row.source_version).strip():
            registry.loc[mask, "provenance"] = str(row.source_version)

    return registry.sort_values(["kind", "cohort_id"]).reset_index(drop=True)


def main() -> None:
    out = _owner_registry_snapshot()
    out.to_csv(OUTPUT, index=False)
    print(
        f"wrote {OUTPUT.name}: {len(out)} cohorts "
        f"({out['kind'].value_counts().to_dict()})"
    )


if __name__ == "__main__":
    main()
