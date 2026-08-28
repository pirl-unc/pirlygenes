"""Generate canonical cohort-expression matrices.

The public ``cohort_expression_matrices()`` API needs the all-cohort TPM and
clean-TPM matrices often enough that rebuilding them from the 9M-row long form
at read time is waste. This script materializes the canonical wide matrices as
a data-bundle artifact:

    pirlygenes/data/cohort-expression-matrices/tpm.parquet
    pirlygenes/data/cohort-expression-matrices/clean_tpm.parquet
    pirlygenes/data/cohort-expression-matrices/provenance.parquet
    pirlygenes/data/cohort-expression-matrices/metadata.json

Run against the pinned delegated oncoref summary before creating the
``DATA_VERSION`` tarball. This is a pirlygenes-specific wide compatibility
cache, not an empirical source builder.
"""

from __future__ import annotations

import json
from pathlib import Path

import oncoref
from oncoref import data_bundle as oncoref_data_bundle

from pirlygenes.expression import (
    COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
    COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
    build_canonical_cohort_expression_matrices,
)
from pirlygenes.version import DATA_VERSION


OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "pirlygenes"
    / "data"
    / "cohort-expression-matrices"
)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The artifact is, by construction, the serialized output of the same
    # function the read path falls back to when the artifact is absent
    # (build_canonical_cohort_expression_matrices). Generating it any other way
    # would let the cache drift from the fallback, so call that function
    # directly — tpm/clean_tpm/provenance here are exactly what a cache miss
    # would recompute.
    tpm, clean_tpm, provenance = build_canonical_cohort_expression_matrices()
    tpm.to_parquet(OUT_DIR / "tpm.parquet", index=False, compression="zstd")
    clean_tpm.to_parquet(
        OUT_DIR / "clean_tpm.parquet",
        index=False,
        compression="zstd",
    )
    provenance.to_parquet(
        OUT_DIR / "provenance.parquet",
        index=False,
        compression="zstd",
    )

    metadata = {
        "artifact_type": COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
        "schema_version": COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
        "pirlygenes_data_version": DATA_VERSION,
        "canonical_gene_ids": True,
        "built_from": {
            "package": "oncoref",
            "package_version": oncoref.__version__,
            "data_version": oncoref_data_bundle.DATA_VERSION,
        },
        "tables": {
            "tpm": {"file": "tpm.parquet", "rows": int(len(tpm))},
            "clean_tpm": {
                "file": "clean_tpm.parquet",
                "rows": int(len(clean_tpm)),
            },
            "provenance": {
                "file": "provenance.parquet",
                "rows": int(len(provenance)),
            },
        },
    }
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*")) / 1e6
    print(
        f"done: {len(tpm)} genes, "
        f"{provenance['cancer_code'].astype(str).nunique()} cohorts, "
        f"{total_mb:.1f} MB -> {OUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    build()
