"""Bake exact selected-source rollups for ``pan_cancer_expression``.

The generic canonical cohort-view artifact is an all-source pivot. For a code
with more than one source (currently NET_PANCREAS), ``aggfunc="first"`` can
fall back to a later source for genes absent from the first source. Pan-cancer
rollups instead have a whole-cohort selected-source contract. This generator
reads those exact source shards, canonicalizes aliases before pooling, and
writes the small wheel-shipped compatibility artifact used by the accessor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
from oncoref import cancer_reference_expression_source_metadata
from oncoref import data_bundle as oncoref_data_bundle

from pirlygenes.expression import (
    PAN_CANCER_ROLLUP_MEMBERS,
    canonicalize_expression_gene_rows,
)
from pirlygenes.load_dataset import get_data


OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "pirlygenes"
    / "data"
    / "pan-cancer-expression-rollups.csv.gz"
)

# These are the sources selected by oncoref's reference-summary ranking for the
# current version-pinned data. Keep this table explicit: a source change should
# produce a reviewable artifact diff rather than a silent gene-wise fallback.
SELECTED_SOURCE_SHARDS = {
    "CHOL": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES.csv.gz",
    "GBC": "GSE139682_GBC.csv.gz",
    "COAD": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES.csv.gz",
    "READ": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES.csv.gz",
    "NET_PANCREAS": "GSE118014_ALVAREZ_2018.csv.gz",
    "NET_MIDGUT": "GSE98894_ALVAREZ_2018_NET.csv.gz",
    "NET_RECTAL": "GSE98894_ALVAREZ_2018_NET.csv.gz",
    "NET_LUNG": "DRMETRICS_ALCALA_2019_LNEN.csv.gz",
    "LUAD": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES.csv.gz",
    "LUSC": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES.csv.gz",
    "ADCC": "GSE294016_BARTL_2025_SGC.csv.gz",
    "ACINIC": "GSE294016_BARTL_2025_SGC.csv.gz",
}

_SOURCE_COLUMNS = [
    "Ensembl_Gene_ID",
    "Symbol",
    "cancer_code",
    "TPM_median",
    "n_samples",
]


def validate_selected_source_shards(
    selected_source_shards: Mapping[str, str] | None = None,
    source_metadata: Callable[[str], Mapping[str, object]] | None = None,
) -> None:
    """Fail when configured rollup shards differ from owner selections."""
    selected_source_shards = (
        SELECTED_SOURCE_SHARDS
        if selected_source_shards is None
        else selected_source_shards
    )
    source_metadata = (
        cancer_reference_expression_source_metadata
        if source_metadata is None
        else source_metadata
    )
    expected_codes = {
        code
        for members in PAN_CANCER_ROLLUP_MEMBERS.values()
        for code in members
    }
    configured_codes = set(selected_source_shards)
    if configured_codes != expected_codes:
        raise ValueError(
            "selected-source table does not match the rollup members: "
            f"missing={sorted(expected_codes - configured_codes)!r}, "
            f"extra={sorted(configured_codes - expected_codes)!r}"
        )

    drift = []
    suffix = ".csv.gz"
    for code in sorted(expected_codes):
        filename = selected_source_shards[code]
        if not filename.endswith(suffix):
            raise ValueError(
                f"selected rollup shard for {code} must end in {suffix!r}: "
                f"{filename!r}"
            )
        configured_cohort = filename[: -len(suffix)]
        owner_cohort = source_metadata(code).get("source_cohort")
        if configured_cohort != owner_cohort:
            drift.append(
                f"{code}: configured={configured_cohort!r}, "
                f"owner={owner_cohort!r}"
            )
    if drift:
        raise RuntimeError(
            "oncoref selected rollup sources changed; update "
            "SELECTED_SOURCE_SHARDS and review the regenerated artifact: "
            + "; ".join(drift)
        )


def _read_selected_source_rows(
    path: Path,
    cancer_codes: set[str],
) -> pd.DataFrame:
    """Read only requested codes from a possibly consolidated owner shard."""
    parts = []
    for chunk in pd.read_csv(path, usecols=_SOURCE_COLUMNS, chunksize=250_000):
        selected = chunk.loc[
            chunk["cancer_code"].astype(str).isin(cancer_codes)
        ]
        if not selected.empty:
            parts.append(selected.copy())
    if not parts:
        return pd.DataFrame(columns=_SOURCE_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def build() -> pd.DataFrame:
    validate_selected_source_shards()
    root = oncoref_data_bundle.find("cancer-reference-expression")
    if root is None:
        oncoref_data_bundle.ensure_local()
        root = oncoref_data_bundle.find("cancer-reference-expression")
    if root is None:
        raise FileNotFoundError(
            "oncoref's cancer-reference-expression bundle is unavailable"
        )
    root = Path(root)
    member_values: dict[str, pd.Series] = {}
    sample_counts: dict[str, float] = {}
    shard_cache: dict[str, pd.DataFrame] = {}
    codes_by_shard: dict[str, set[str]] = {}
    for code, filename in SELECTED_SOURCE_SHARDS.items():
        codes_by_shard.setdefault(filename, set()).add(code)

    for code, filename in SELECTED_SOURCE_SHARDS.items():
        if filename not in shard_cache:
            shard_cache[filename] = _read_selected_source_rows(
                root / filename,
                codes_by_shard[filename],
            )
        source = shard_cache[filename]
        source = source[source["cancer_code"].astype(str) == code].copy()
        if source.empty:
            raise ValueError(f"{filename} does not contain {code}")
        source = canonicalize_expression_gene_rows(
            source,
            value_cols=["TPM_median"],
        )
        counts = pd.to_numeric(source["n_samples"], errors="coerce").dropna()
        if counts.empty or counts.nunique() != 1:
            raise ValueError(f"{filename}/{code} has no unique n_samples")
        sample_counts[code] = float(counts.iloc[0])
        member_values[code] = (
            source.set_index("Ensembl_Gene_ID")["TPM_median"]
            .pipe(pd.to_numeric, errors="coerce")
            .rename(code)
        )

    members = pd.concat(member_values.values(), axis=1)
    aggregate_values: dict[str, pd.Series] = {}
    for aggregate, codes in PAN_CANCER_ROLLUP_MEMBERS.items():
        values = members[list(codes)]
        weights = pd.Series({code: sample_counts[code] for code in codes})
        numerator = values.mul(weights, axis="columns").sum(axis=1, min_count=1)
        denominator = values.notna().mul(weights, axis="columns").sum(axis=1)
        aggregate_values[f"TPM_{aggregate}"] = numerator.div(
            denominator.where(denominator > 0)
        )

    base = get_data("pan-cancer-expression", copy=False)[
        ["Ensembl_Gene_ID", "Symbol"]
    ]
    base = canonicalize_expression_gene_rows(base, value_cols=[])
    out = base[["Ensembl_Gene_ID"]].copy()
    for column, values in aggregate_values.items():
        out[column] = out["Ensembl_Gene_ID"].map(values)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        OUT_PATH,
        index=False,
        float_format="%.15g",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    print(f"wrote {len(out):,} rows ({OUT_PATH.stat().st_size / 1e6:.2f} MB) to {OUT_PATH}")
    return out


if __name__ == "__main__":
    build()
