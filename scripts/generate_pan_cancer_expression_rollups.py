"""Bake exact selected-source rollups for ``pan_cancer_expression``.

The generic canonical cohort-view artifact is an all-source pivot. For a code
with more than one source (currently NET_PANCREAS), ``aggfunc="first"`` can
fall back to a later source for genes absent from the first source. Pan-cancer
rollups instead have a whole-cohort selected-source contract. This generator
reads those exact source shards, canonicalizes aliases before pooling, and
writes the small wheel-shipped compatibility artifact used by the accessor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pirlygenes.expression import accessors
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
    "CHOL": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__CHOL.csv.gz",
    "COAD": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__COAD.csv.gz",
    "READ": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__READ.csv.gz",
    "NET_PANCREAS": "GSE118014_ALVAREZ_2018.csv.gz",
    "NET_MIDGUT": "GSE98894_ALVAREZ_2018_NET.csv.gz",
    "NET_RECTAL": "GSE98894_ALVAREZ_2018_NET.csv.gz",
    "NET_LUNG": "DRMETRICS_ALCALA_2019_LNEN.csv.gz",
    "LUAD": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__LUAD.csv.gz",
    "LUSC": "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__LUSC.csv.gz",
    "ADCC": "GSE294016_BARTL_2025_SGC.csv.gz",
}


def build() -> pd.DataFrame:
    expected_codes = set(accessors._PAN_ROLLUP_MEMBER_CODES)
    if set(SELECTED_SOURCE_SHARDS) != expected_codes:
        raise ValueError(
            "selected-source table does not match the rollup members: "
            f"missing={sorted(expected_codes - set(SELECTED_SOURCE_SHARDS))!r}, "
            f"extra={sorted(set(SELECTED_SOURCE_SHARDS) - expected_codes)!r}"
        )
    root = Path(accessors._bundle_subdir("cancer-reference-expression"))
    member_values: dict[str, pd.Series] = {}
    sample_counts: dict[str, float] = {}
    shard_cache: dict[str, pd.DataFrame] = {}

    for code, filename in SELECTED_SOURCE_SHARDS.items():
        if filename not in shard_cache:
            shard_cache[filename] = pd.read_csv(
                root / filename,
                usecols=[
                    "Ensembl_Gene_ID",
                    "Symbol",
                    "cancer_code",
                    "TPM_median",
                    "n_samples",
                ],
            )
        source = shard_cache[filename]
        source = source[source["cancer_code"].astype(str) == code].copy()
        if source.empty:
            raise ValueError(f"{filename} does not contain {code}")
        source = accessors._oncoref_canonicalize_gene_rows(
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
    for aggregate, codes in accessors._PAN_COMPUTED_ROLLUP_MEMBERS.items():
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
    base = accessors._oncoref_canonicalize_gene_rows(base, value_cols=[])
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
