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
    # The artifact is, by construction, the serialized output of the same
    # function the read path falls back to when the artifact is absent
    # (accessors._rebuild_full_canonical_views). Generating it any other way
    # would let the cache drift from the fallback, so call that function
    # directly — tpm/clean_tpm/provenance here are exactly what a cache miss
    # would recompute.
    tpm, clean_tpm, provenance = (
        accessors._rebuild_full_canonical_views()  # noqa: SLF001
    )
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

    manifest = {
        "artifact": "cancer-reference-expression-views",
        "data_version": DATA_VERSION,
        "canonical_gene_ids": True,
        "format": 1,
        "rows": {
            "tpm": int(len(tpm)),
            "clean_tpm": int(len(clean_tpm)),
            "provenance": int(len(provenance)),
        },
    }
    (OUT_DIR / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    total_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*")) / 1e6
    print(
        f"done: {len(tpm)} genes, "
        f"{len(accessors._cohort_value_cols(tpm))} cohorts, "  # noqa: SLF001
        f"{total_mb:.1f} MB -> {OUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    build()
