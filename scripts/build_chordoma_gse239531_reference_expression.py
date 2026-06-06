#!/usr/bin/env python
"""Per-sample chordoma builder from GSE239531 (van Oost 2024).

Augments the thin Treehouse ribo-depleted CHOR cohort (3 samples) with 20
primary chordoma tumors from van Oost et al. 2024 (J Immunother Cancer,
PMID 38272563). The GEO supplementary `GSE239531_raw_counts.tsv.gz` is an
Ensembl-keyed raw-counts matrix with the exact shape the CHON builder
already handles, so this script reuses that builder's download / harmonize /
counts->length-normalized-TPM core verbatim and only swaps the source
constants and provenance strings. Lands as its own source_cohort shard
(`GSE239531_VANOOST_2024`) alongside the Treehouse ribo CHOR rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_chon_reference_expression import (  # noqa: E402  reuse the CHON core
    _download,
    _read_counts,
    _harmonize_and_lengths,
    _counts_to_tpm,
)
from pirlygenes.expression.stats import (  # noqa: E402
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
from pirlygenes.expression.normalize import (  # noqa: E402
    clean_tpm_matrix as _clean_tpm,
    technical_rna_mask as _technical_mask,
)

SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE239nnn/GSE239531/suppl/"
    "GSE239531_raw_counts.tsv.gz"
)
CANCER_CODE = "SARC_CHOR"
SOURCE_COHORT = "GSE239531_VANOOST_2024"
SOURCE_PROJECT = "GEO"
PIPELINE = "gse239531_chordoma_raw_counts_to_tpm_ensembl{ensembl}_clean_tpm_v4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression" / "gse239531-chordoma",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = args.cache_dir / "GSE239531_raw_counts.tsv.gz"
    print(f"downloading {file_path.name}...")
    _download(SOURCE_URL, file_path)

    print("reading raw counts...")
    counts_df = _read_counts(file_path)
    print(f"  shape: {counts_df.shape} (genes × samples)")

    print(f"harmonizing Ensembl IDs (release {args.ensembl_release}) + computing gene lengths...")
    mapping, harm_counts = _harmonize_and_lengths(counts_df.index, args.ensembl_release)
    print(f"  resolved={harm_counts['resolved']}, dropped={harm_counts['dropped']}")

    print("counts → TPM (length-normalized + per-sample renormalize to 1e6)...")
    gene_table, tpm = _counts_to_tpm(counts_df, mapping)
    print(f"  canonical genes: {len(gene_table)}")

    print("computing stats...")
    clean = _clean_tpm(tpm, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = CANCER_CODE
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = (
        f"GSE239531 (van Oost 2024, chordoma; PMID 38272563) raw counts; "
        f"length-normalized to TPM with Ensembl release "
        f"{args.ensembl_release} gene lengths; downloaded from GEO supplementary"
    )
    out["tumor_origin"] = "primary"  # van Oost 2024 is a primary-tumor cohort
    assign_stats(out, tpm, clean)
    out["processing_pipeline"] = PIPELINE.format(ensembl=args.ensembl_release)
    out["notes"] = (
        f"Per-sample TPMs from GSE239531 (van Oost 2024, "
        f"n={tpm.shape[1]} primary chordoma; PMID 38272563). Source data is "
        "raw counts keyed by Ensembl ID; length-normalized to TPM with "
        f"Ensembl release {args.ensembl_release} gene lengths. TPM_clean "
        "computed per-sample by two-compartment fixed-fraction clean-TPM (technical 25% / biological 75%, each renormalized within its group)."
    )
    # reindex (not strict select): tumor_origin / metastasis_site aren't set
    # by this source and should land as empty rather than KeyError.
    out = round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))
    print(f"  built {len(out)} gene rows for {CANCER_CODE}")

    upsert_to_shard(
        args.summary_output, out,
        source_cohort=SOURCE_COHORT, cancer_codes=[CANCER_CODE],
    )
    print(f"upserted {len(out)} rows into shard {SOURCE_COHORT}.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
