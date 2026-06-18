"""Generate precomputed canonical cohort-expression views.

The public ``cohort_expression_views()`` API needs the all-cohort TPM and
clean-TPM matrices often enough that rebuilding them from the 9M-row long form
at read time is waste. This script materializes the canonical wide matrices as
a data-bundle artifact:

    pirlygenes/data/cancer-reference-expression-views/tpm.parquet
    pirlygenes/data/cancer-reference-expression-views/clean_tpm.parquet
    pirlygenes/data/cancer-reference-expression-views/provenance.parquet
    pirlygenes/data/cancer-reference-expression-views/_manifest.json

Run after baking ``cancer-reference-expression/`` shards and before creating the
DATA_VERSION tarball.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pirlygenes.expression import accessors
from pirlygenes.version import DATA_VERSION


OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "pirlygenes"
    / "data"
    / "cancer-reference-expression-views"
)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    views = accessors._cohort_expression_views_from_reference(  # noqa: SLF001
        canonicalize_genes=True,
    )
    views.tpm.to_parquet(OUT_DIR / "tpm.parquet", index=False, compression="zstd")
    views.clean_tpm.to_parquet(
        OUT_DIR / "clean_tpm.parquet",
        index=False,
        compression="zstd",
    )

    ref = accessors._load_cancer_reference_expression()  # noqa: SLF001
    provenance_cols = [
        "cancer_code",
        "source_cohort",
        "processing_pipeline",
        "n_samples",
    ]
    provenance = (
        ref[[c for c in provenance_cols if c in ref.columns]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    provenance.to_parquet(
        OUT_DIR / "provenance.parquet",
        index=False,
        compression="zstd",
    )

    manifest = {
        "artifact": "cancer-reference-expression-views",
        "data_version": DATA_VERSION,
        "canonical_gene_ids": True,
        "format": 1,
        "rows": {
            "tpm": int(len(views.tpm)),
            "clean_tpm": int(len(views.clean_tpm)),
            "provenance": int(len(provenance)),
        },
    }
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*")) / 1e6
    print(
        f"done: {len(views.tpm)} genes, "
        f"{len(accessors._cohort_value_cols(views.tpm))} cohorts, "  # noqa: SLF001
        f"{total_mb:.1f} MB -> {OUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    build()
