#!/usr/bin/env python
"""Per-sample GSE75885 (Delespaul 2017) soft-tissue sarcoma builder.

Replaces the summary-only import (whose symbol-level median/Q1/Q3 source CSV was
lost — see #305) with a real per-sample build straight from the GEO processed
matrix ``GSE75885_Expression_117_sarcomas.tsv.gz`` (RNA-seq, Illumina HiSeq
2000; 117 sarcomas). Values are RPKM-like (per-sample totals ~1.6-3.4e6×10,
not pinned at 1e6), so each sample is renormalized to 1e6 (RPKM→TPM), HUGO
symbols are harmonized to Ensembl, and the three-compartment fixed-fraction
clean-TPM is applied — matching every other per-sample cohort.

Per-sample subtype labels come from the series-matrix ``!Sample_title`` (which
prefixes each author sample ID, e.g. ``"M969 - Liposarcoma - dedifferentiated"``).
Builds the three registry codes this cohort backs; the matrix also contains
LMS/UPS/MFS/pleomorphic-RMS samples that other sources already cover.
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from pirlygenes.builders.treehouse import (
    _aggregate_by_ensembl,
    _build_or_load_symbol_mapping,
    _clean_tpm,
)
from pirlygenes.expression.stats import (
    assign_stats,
    finalize_reference_rows,
    upsert_to_shard,
)

EXPR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75885/suppl/"
    "GSE75885_Expression_117_sarcomas.tsv.gz"
)
SERIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75885/matrix/"
    "GSE75885_series_matrix.txt.gz"
)
SOURCE_COHORT = "GSE75885_DELESPAUL_2017"
SOURCE_PROJECT = "GEO"
PIPELINE = "gse75885_rpkm_to_tpm_ensembl{ensembl}_clean_tpm_16_9_75"

# series-matrix tumor-type label -> registry code (only the codes this cohort
# is registered to back; the matrix has other subtypes too).
SUBTYPE_TO_CODE = {
    "Liposarcoma - dedifferentiated": "SARC_DDLPS",
    "Liposarcoma - pleomorphic": "SARC_PLEOLPS",
    "Low grade fibromyxoid sarcoma": "SARC_LGFMS",
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _sample_subtypes(series_matrix: Path) -> dict[str, str]:
    """{author_sample_id: tumor_type} from the series-matrix !Sample_title
    (e.g. '"M969 - Liposarcoma - dedifferentiated"' -> M969: Liposarcoma ...)."""
    line = None
    with gzip.open(series_matrix, "rt") as h:
        for raw in h:
            if raw.startswith("!Sample_title"):
                line = raw
                break
    out = {}
    for title in re.findall(r'"([^"]+)"', line or ""):
        if " - " in title:
            sid, sub = title.split(" - ", 1)
            out[sid.strip()] = sub.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression" / "gse75885-sarc",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    expr_path = _download(EXPR_URL, args.cache_dir / "GSE75885_Expression_117_sarcomas.tsv.gz")
    series_path = _download(SERIES_URL, args.cache_dir / "GSE75885_series_matrix.txt.gz")

    print("reading processed expression matrix (symbol × sample)...")
    with gzip.open(expr_path, "rt") as h:
        expr = pd.read_csv(h, sep="\t", index_col=0)
    expr.index.name = "source_symbol"
    subtypes = _sample_subtypes(series_path)

    print(f"harmonizing HUGO symbols → Ensembl {args.ensembl_release}...")
    mapping = _build_or_load_symbol_mapping(
        expr.index,
        ensembl_release=args.ensembl_release,
        cache_path=args.cache_dir / f"symbol_to_ensembl_{args.ensembl_release}.parquet",
        refresh=False,
    )
    gene_table, values = _aggregate_by_ensembl(expr, mapping)
    print(f"  canonical genes: {len(gene_table)}")

    summaries, counts = [], {}
    for label, code in SUBTYPE_TO_CODE.items():
        cols = [s for s in values.columns if subtypes.get(s) == label]
        if not cols:
            print(f"  WARN no samples for {label!r} -> {code}; skipping")
            continue
        # RPKM-like -> TPM: renormalize each sample to 1e6, then clean-TPM.
        sub = values[cols]
        sub = sub.div(sub.sum(axis=0), axis=1) * 1_000_000.0
        from pirlygenes import cohorts as _cohorts
        _cohorts.write_per_sample(gene_table, sub, args.cache_dir.name, code)
        clean = _clean_tpm(sub, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = SOURCE_COHORT
        out["source_project"] = SOURCE_PROJECT
        out["source_version"] = (
            "GSE75885 (Delespaul 2017) RNA-seq processed expression "
            "(GSE75885_Expression_117_sarcomas.tsv.gz; Illumina HiSeq 2000); "
            "RPKM-like per-sample renormalized to 1e6 (TPM); HUGO symbols "
            f"harmonized to Ensembl release {args.ensembl_release}"
        )
        assign_stats(out, sub, clean)
        out["processing_pipeline"] = PIPELINE.format(ensembl=args.ensembl_release)
        out["notes"] = (
            f"Per-sample TPM from GSE75885 (Delespaul 2017, n={len(cols)} {label}). "
            "Processed RNA-seq matrix from GEO supplementary; RPKM-like values "
            "renormalized per sample to 1e6. TPM_clean computed per-sample by "
            "three-compartment fixed-fraction clean-TPM (ribosomal-protein 16% / "
            "other-technical 9% / biological 75%, each renormalized within its "
            "compartment). Supersedes the lost "
            "summary-only import (#305)."
        )
        # Mixed primary/metastasis cohort (series-matrix 'metastasis' field).
        out = finalize_reference_rows(out, tumor_origin="mixed")
        summaries.append(out)
        counts[code] = len(cols)

    combined = pd.concat(summaries, ignore_index=True)
    upsert_to_shard(
        args.summary_output, combined,
        source_cohort=SOURCE_COHORT, cancer_codes=list(counts),
    )
    print(f"upserted {len(combined)} rows into shard {SOURCE_COHORT}.csv.gz; "
          f"samples per code: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
