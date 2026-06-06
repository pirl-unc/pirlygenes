#!/usr/bin/env python
"""Build per-cohort reference rows from the Treehouse Tumor Compendium.

Parameterized over (release, disease_label, output_cancer_code).
A single Treehouse release file (e.g. the 25.01 PolyA TPM matrix —
13,359 samples, 58,581 genes, ~6 GB TSV) can be re-used to build
multiple cancer codes by re-invoking this script with different
``--disease-label`` / ``--output-cancer-code`` pairs.

Input format
------------
The TPM matrix is RSEM ``log2(TPM + 1)`` keyed by HUGO symbol.
This script:

1. Reads the clinical metadata file and selects samples whose
   ``disease`` column matches ``--disease-label``.
2. Reads only those sample columns from the TPM matrix
   (pandas ``usecols=``) so peak memory is bounded by
   ``n_genes * n_selected_samples`` rather than the full matrix.
3. Inverse-transforms ``TPM = 2 ** log2tpm - 1`` (clamped at 0 for
   any tiny negative residuals from float arithmetic).
4. Harmonizes HUGO symbols to current Ensembl release IDs via the
   active pyensembl release (default 112); rows whose symbols can't
   be uniquely mapped are dropped.
5. Computes the standard stat suite
   (:func:`pirlygenes.expression.stats.assign_stats`) over the raw
   and technical-RNA-zeroed-renormalized matrices.
6. Upserts the resulting per-gene rows into
   ``pirlygenes/data/cancer-reference-expression``,
   replacing any prior rows for the same
   ``(cancer_code, source_cohort)`` pair.

Cache layout
------------
By default this script reads its inputs from
``~/.cache/pirlygenes/expression/<source_id>/`` (matching the
``pirlygenes downloads`` convention), and the input filenames default
to the 25.01 PolyA release. Override with ``--cache-dir`` /
``--tpm-file`` / ``--clinical-file`` for other releases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
)
from pirlygenes.expression.normalize import clean_tpm_matrix as _clean_tpm, technical_rna_mask as _technical_mask


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"
DEFAULT_SOURCE_ID = "treehouse-polya-25-01"
DEFAULT_TPM_FILE = "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
DEFAULT_CLINICAL_FILE = (
    "clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv"
)
DEFAULT_SOURCE_COHORT = "TREEHOUSE_POLYA_25_01"
DEFAULT_SOURCE_PROJECT = "Treehouse"
DEFAULT_SOURCE_VERSION = (
    "Treehouse Tumor Compendium 25.01 PolyA "
    "(hugo_log2tpm_58581genes_2025-02-27); "
    "downloaded from public.gi.ucsc.edu/~ekephart/public-data/; "
    "log2(TPM+1) inverse-transformed; "
    "HUGO symbols harmonized to Ensembl release {ensembl_release}"
)
PIPELINE = "treehouse_polya_25_01_log2tpm_to_tpm_ensembl{ensembl}_clean_tpm_v4"
DEFAULT_NOTES = (
    "Per-sample TPMs from the Treehouse Tumor Compendium 25.01 PolyA "
    "(hugo_log2tpm matrix, inverse-log2 transformed). Sample selection: "
    "clinical.disease == '{disease_label}'. HUGO symbols mapped to "
    "Ensembl release {ensembl_release}; duplicate symbol mappings dropped. "
    "TPM_clean is computed per-sample by two-compartment fixed-fraction clean-TPM (technical 25% / biological 75%, each renormalized within its group) + "
    "denominator rescaling."
)


def _read_clinical_disease_samples(
    clinical_path: Path,
    disease_label: str,
) -> list[str]:
    clin = pd.read_csv(clinical_path, sep="\t")
    if "th_dataset_id" not in clin.columns or "disease" not in clin.columns:
        raise RuntimeError(
            f"Clinical file {clinical_path} is missing th_dataset_id or "
            f"disease column; got {clin.columns.tolist()}"
        )
    mask = clin["disease"].astype(str).str.strip().str.lower().eq(
        disease_label.strip().lower()
    )
    selected = clin.loc[mask, "th_dataset_id"].astype(str).tolist()
    if not selected:
        raise RuntimeError(
            f"No samples matched disease={disease_label!r} in {clinical_path}"
        )
    return selected


def _read_tpm_for_samples(
    tpm_path: Path,
    sample_ids: list[str],
    *,
    gene_col: str = "Gene",
) -> pd.DataFrame:
    with tpm_path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
    available = set(header)
    missing = [sid for sid in sample_ids if sid not in available]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(sample_ids)} selected samples not in TPM "
            f"matrix header. First few missing: {missing[:5]}"
        )
    keep_cols = [gene_col] + sample_ids
    df = pd.read_csv(tpm_path, sep="\t", usecols=keep_cols, low_memory=False)
    return df.set_index(gene_col)


def _inverse_log2(values: pd.DataFrame) -> pd.DataFrame:
    """Recover TPM from log2(TPM+1). Clamp tiny float negatives at 0."""
    tpm = np.power(2.0, values.to_numpy()) - 1.0
    tpm[tpm < 0] = 0.0
    return pd.DataFrame(tpm, index=values.index, columns=values.columns)


def _harmonize_symbols(
    symbols: pd.Index,
    *,
    ensembl_release: int,
) -> pd.DataFrame:
    """Map HUGO symbols to (Ensembl_Gene_ID, Symbol) rows.

    Dropped: symbols that pyensembl returns 0 or >1 distinct ensembl
    IDs for. The dropped fraction is reported to stdout for sanity.
    """
    genome = EnsemblRelease(ensembl_release)
    rows: list[dict[str, str]] = []
    counts = {"resolved": 0, "unresolved": 0, "ambiguous": 0}
    for sym in symbols:
        s = str(sym).strip()
        if not s:
            counts["unresolved"] += 1
            continue
        try:
            genes = genome.genes_by_name(s)
        except Exception:
            genes = []
        ids = {gene.gene_id.split(".", 1)[0] for gene in genes}
        if len(ids) == 1:
            gene = genes[0]
            rows.append(
                {
                    "source_symbol": s,
                    "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
                    "Symbol": gene.gene_name or s,
                }
            )
            counts["resolved"] += 1
        elif not ids:
            counts["unresolved"] += 1
        else:
            counts["ambiguous"] += 1
    return pd.DataFrame(rows), counts


def _summarize(
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
    *,
    cancer_code: str,
    source_cohort: str,
    source_project: str,
    source_version: str,
    pipeline: str,
    notes: str,
) -> pd.DataFrame:
    clean = _clean_tpm(values, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cancer_code
    out["source_cohort"] = source_cohort
    out["source_project"] = source_project
    out["source_version"] = source_version
    assign_stats(out, values, clean)
    out["processing_pipeline"] = pipeline
    out["notes"] = notes
    return round_stat_columns(out)[list(REFERENCE_COLUMNS)]


def _upsert(
    summary_output: Path,
    new_rows: pd.DataFrame,
    *,
    cancer_code: str,
    source_cohort: str,
) -> pd.DataFrame:
    """Write/update a single cancer_code's rows in the per-source shard.

    ``summary_output`` is the
    ``pirlygenes/data/cancer-reference-expression/`` directory. The
    per-source shard ``<dir>/<source_cohort>.csv.gz`` holds every row
    for that source; this function replaces the rows for
    ``cancer_code`` and preserves rows for every other cancer code in
    that source.
    """
    summary_output = Path(summary_output)
    if summary_output.suffix == ".gz" or summary_output.is_file():
        shard_dir = summary_output.parent / "cancer-reference-expression"
    else:
        shard_dir = summary_output
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{source_cohort}.csv.gz"

    if shard_path.exists():
        existing = pd.read_csv(shard_path, low_memory=False)
        keep = ~existing["cancer_code"].astype(str).eq(cancer_code)
        merged = pd.concat(
            [existing.loc[keep].reindex(columns=list(REFERENCE_COLUMNS)), new_rows],
            ignore_index=True,
        )
    else:
        merged = new_rows.copy()

    merged = merged.reindex(columns=list(REFERENCE_COLUMNS)).sort_values(
        ["cancer_code", "Ensembl_Gene_ID"], na_position="last",
    )
    merged.to_csv(shard_path, index=False, compression="gzip")
    return merged


def build(args: argparse.Namespace) -> int:
    cache_dir = args.cache_dir or (DEFAULT_CACHE_ROOT / args.source_id)
    tpm_path = args.tpm_file or (cache_dir / DEFAULT_TPM_FILE)
    clinical_path = args.clinical_file or (cache_dir / DEFAULT_CLINICAL_FILE)
    for p in (tpm_path, clinical_path):
        if not Path(p).exists():
            raise SystemExit(f"missing input file: {p}")

    samples = _read_clinical_disease_samples(Path(clinical_path), args.disease_label)
    print(f"clinical: {len(samples)} samples match disease={args.disease_label!r}")

    print(f"reading TPM matrix for {len(samples)} samples from {tpm_path}...")
    log2_values = _read_tpm_for_samples(Path(tpm_path), samples)
    print(
        f"  raw shape (genes, samples): {log2_values.shape}; "
        f"inverse-transforming log2(TPM+1)..."
    )
    values_by_symbol = _inverse_log2(log2_values)

    print("harmonizing HUGO symbols to Ensembl release "
          f"{args.ensembl_release}...")
    mapping, counts = _harmonize_symbols(
        values_by_symbol.index,
        ensembl_release=args.ensembl_release,
    )
    print(
        f"  symbol harmonization: {counts['resolved']} resolved, "
        f"{counts['ambiguous']} ambiguous (dropped), "
        f"{counts['unresolved']} unresolved (dropped)"
    )

    # Aggregate TPM by Ensembl_Gene_ID; multiple symbols can map to
    # the same gene (synonyms collapsed by pyensembl), so sum-aggregate
    # to be safe.
    symbol_col = values_by_symbol.index.name or "Gene"
    values_flat = values_by_symbol.reset_index().rename(
        columns={symbol_col: "source_symbol"}
    )
    merged = mapping.merge(values_flat, on="source_symbol", how="inner")
    sample_set = set(samples)
    sample_cols = [c for c in merged.columns if c in sample_set]
    by_gene = merged.groupby("Ensembl_Gene_ID", as_index=False, sort=False).agg(
        {"Symbol": "first", **{c: "sum" for c in sample_cols}}
    )
    gene_table = by_gene[["Ensembl_Gene_ID", "Symbol"]].copy()
    values = by_gene.set_index("Ensembl_Gene_ID")[sample_cols]
    print(f"  unique genes after aggregation: {values.shape[0]}")

    source_version = DEFAULT_SOURCE_VERSION.format(
        ensembl_release=args.ensembl_release,
    )
    pipeline = PIPELINE.format(ensembl=args.ensembl_release)
    notes = DEFAULT_NOTES.format(
        disease_label=args.disease_label,
        ensembl_release=args.ensembl_release,
    )

    print(f"computing stats for cancer_code={args.output_cancer_code}...")
    summary = _summarize(
        gene_table,
        values,
        cancer_code=args.output_cancer_code,
        source_cohort=args.source_cohort,
        source_project=args.source_project,
        source_version=source_version,
        pipeline=pipeline,
        notes=notes,
    )
    print(f"  wrote {len(summary)} gene rows for {args.output_cancer_code}")

    combined = _upsert(
        Path(args.summary_output),
        summary,
        cancer_code=args.output_cancer_code,
        source_cohort=args.source_cohort,
    )
    print(
        f"upsert complete: {len(combined)} total rows in "
        f"{args.summary_output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build cancer-reference-expression rows from a "
                    "Treehouse Tumor Compendium release."
    )
    parser.add_argument(
        "--source-id",
        default=DEFAULT_SOURCE_ID,
        help="Cache subdir name + source identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override the cache directory (defaults to "
             "~/.cache/pirlygenes/expression/<source-id>/).",
    )
    parser.add_argument(
        "--tpm-file",
        type=Path,
        default=None,
        help="Path to the log2(TPM+1) TSV (default: matches the 25.01 "
             "PolyA release filename under --cache-dir).",
    )
    parser.add_argument(
        "--clinical-file",
        type=Path,
        default=None,
        help="Path to the clinical metadata TSV (default: matches the "
             "25.01 PolyA release filename under --cache-dir).",
    )
    parser.add_argument(
        "--disease-label",
        required=True,
        help="Treehouse clinical 'disease' label to select on, e.g. "
             "'Ewing sarcoma'.",
    )
    parser.add_argument(
        "--output-cancer-code",
        required=True,
        help="pirlygenes cancer_code to use in the output rows, e.g. EWS.",
    )
    parser.add_argument(
        "--source-cohort",
        default=DEFAULT_SOURCE_COHORT,
        help="cancer-reference-expression source_cohort value "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--source-project",
        default=DEFAULT_SOURCE_PROJECT,
        help="cancer-reference-expression source_project value "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args = parser.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
