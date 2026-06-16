#!/usr/bin/env python
"""LUNG_NET_LC + LUNG_NET_LCNEC builder from IARCbioinfo/DRMetrics.

Source: Alcala 2019 (PMID 31431620) / Gabriel 2020 (PMID 33203751)
pan-LNEN cohort. 238 samples = 118 carcinoid + 69 LCNEC + 51 SCLC,
processed uniformly (STAR + featureCounts, GENCODE v33, GRCh38).
Histology labels live in a separate ``Attributes.txt`` file in the
same DRMetrics GitHub release.

This is the only public single-file bulk RNA-seq matrix for lung
carcinoid + LCNEC at this scale; SCLC isn't built here because we
already have the UCologne 2015 cohort with similar n.

Source-id ``drmetrics-lnen-2020`` in ``expression_sources.yaml``.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from pirlygenes.builders.geo_matrix import (
    _clean_tpm,
    _technical_mask,
    harmonize_gene_ids,
    normalize_to_tpm,
    read_matrix,
    _gene_lengths_kb_for_index,
)
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)


COUNTS_URL = (
    "https://raw.githubusercontent.com/IARCbioinfo/DRMetrics/NextJournalH/"
    "data/read_counts_all.txt.zip"
)
ATTRS_URL = (
    "https://raw.githubusercontent.com/IARCbioinfo/DRMetrics/NextJournalH/"
    "data/Attributes.txt.zip"
)
SOURCE_COHORT = "DRMETRICS_ALCALA_2019_LNEN"
SOURCE_PROJECT = "IARC pan-LNEN (Alcala 2019 / Gabriel 2020)"
CITATION = "PMID 31431620 (Alcala 2019); PMID 33203751 (Gabriel 2020)"

# Histopathology_simplified → pirlygenes cancer_code
HISTOLOGY_TO_CODE = {
    "Typical": "LUNG_NET_LC",
    "Atypical": "LUNG_NET_LC",
    "Carcinoid": "LUNG_NET_LC",
    "Supra_carcinoid": "LUNG_NET_LC",
    "LCNEC": "LUNG_NET_LCNEC",
    # SCLC samples (n=51) skipped — we have SCLC UCologne 2015 separately
}


def _download_and_unzip(url: str, dest_dir: Path, zip_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / zip_name
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
            shutil.copyfileobj(r, h)
        tmp.replace(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        extracted = zf.namelist()[0]
        out_path = dest_dir / extracted
        if not out_path.exists() or out_path.stat().st_size == 0:
            zf.extractall(dest_dir)
        return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression"
                / "drmetrics-lnen-2020",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--samples-output", type=Path, default=None,
        help="(Accepted for dispatcher compatibility; not used here.)",
    )
    args = parser.parse_args()

    print("downloading DRMetrics counts + attributes...")
    counts_path = _download_and_unzip(
        COUNTS_URL, args.cache_dir, "read_counts_all.txt.zip",
    )
    attrs_path = _download_and_unzip(
        ATTRS_URL, args.cache_dir, "Attributes.txt.zip",
    )

    print("reading attribute table...")
    attrs = pd.read_csv(attrs_path, sep="\t", usecols=[
        "Sample_ID", "Histopathology_simplified",
    ])
    print(f"  {len(attrs)} samples; histology counts:")
    print(attrs["Histopathology_simplified"].value_counts().to_string())

    print("reading counts matrix (space-separated)...")
    matrix = read_matrix(
        counts_path, sep=r"\s+", gene_id_col="gene_id", drop_cols=(),
    )
    print(f"  shape: {matrix.shape}")

    print("length-normalize raw counts → TPM (Ensembl 112 gene lengths)...")
    lengths_kb = _gene_lengths_kb_for_index(
        matrix.index, gene_id_type="ensembl",
        ensembl_release=args.ensembl_release,
    )
    print(f"  gene lengths resolved for {len(lengths_kb)} of {len(matrix.index)} rows")
    tpm = normalize_to_tpm(matrix, unit="raw_counts", gene_lengths_kb=lengths_kb)

    print(f"harmonizing → Ensembl release {args.ensembl_release}...")
    mapping, values = harmonize_gene_ids(
        tpm, gene_id_type="ensembl", ensembl_release=args.ensembl_release,
    )
    gene_table = (
        mapping.drop_duplicates("Ensembl_Gene_ID")[["Ensembl_Gene_ID", "Symbol"]]
        .reset_index(drop=True)
    )
    values = values.reindex(gene_table["Ensembl_Gene_ID"]).fillna(0.0)
    print(f"  canonical genes: {len(gene_table)}")

    # Group samples by histology → cancer_code
    sample_to_code = {
        row.Sample_ID: HISTOLOGY_TO_CODE.get(row.Histopathology_simplified)
        for row in attrs.itertuples(index=False)
    }
    by_code: dict[str, list[str]] = {}
    for col in values.columns:
        code = sample_to_code.get(col)
        if code is None:
            continue
        by_code.setdefault(code, []).append(col)

    summaries = []
    for code, cols in by_code.items():
        sub_values = values[cols]
        clean = _clean_tpm(sub_values, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = SOURCE_COHORT
        out["source_project"] = SOURCE_PROJECT
        out["source_version"] = (
            f"DRMetrics LNEN (Alcala 2019 / Gabriel 2020); raw counts "
            f"length-normalized to TPM with Ensembl release "
            f"{args.ensembl_release} gene lengths; histology routed "
            "via Attributes.txt Histopathology_simplified."
        )
        assign_stats(out, sub_values, clean)
        out["processing_pipeline"] = (
            f"drmetrics_alcala_2019_lnen_raw_counts_to_tpm_ensembl"
            f"{args.ensembl_release}_clean_tpm_16_9_75"
        )
        histos = (
            attrs.loc[attrs["Sample_ID"].isin(cols), "Histopathology_simplified"]
            .value_counts().to_dict()
        )
        out["notes"] = (
            f"{code} from IARC DRMetrics pan-LNEN cohort (n={len(cols)}). "
            f"Histopathology_simplified breakdown: {histos}. "
            f"Raw counts → length-norm TPM; tech-RNA zero; v5.3 stats. "
            f"{CITATION}."
        )
        out["tumor_origin"] = "primary"  # IARC LNEN is a primary-tumor cohort
        # reindex (not strict select): tumor_origin / metastasis_site aren't set
        # by assign_stats; reindex backfills metastasis_site as NaN.
        out = round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))
        summaries.append(out)
        print(f"  {code}: n={len(cols)} → {len(out)} gene rows")

    combined = pd.concat(summaries, ignore_index=True)
    upsert_to_shard(
        args.summary_output, combined,
        source_cohort=SOURCE_COHORT, cancer_codes=list(by_code.keys()),
    )
    print(f"upserted {len(combined)} rows into shard {SOURCE_COHORT}.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
