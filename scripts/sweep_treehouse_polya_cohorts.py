#!/usr/bin/env python
"""Single-pass sweep: build every Treehouse 25.01 PolyA cohort at once.

The single-cohort builder (build_treehouse_reference_expression.py)
re-pays its setup cost on every invocation — re-scanning the 6 GB
TPM matrix, re-walking pyensembl for the 58,581 HUGO symbols, etc.
This sweep loads the matrix and the symbol mapping exactly once,
splits by disease, and writes all cohorts in a single CSV upsert.

Cached artifacts under
``~/.cache/pirlygenes/expression/treehouse-polya-25-01/derived/``:

- ``symbol_to_ensembl_<release>.parquet`` — mapping from HUGO symbol
  → (Ensembl_Gene_ID, Symbol). One-time pyensembl pass; reused on
  every subsequent run (any release, any disease, any subset).
- ``<cancer_code>_per_sample_tpm.parquet`` — per-cohort, per-sample
  TPM matrix (post inverse-log2, post symbol harmonization,
  Ensembl-keyed). Skips the 6 GB matrix scan on re-runs.

Cache invalidation: ``--refresh-cache`` blows away both layers and
re-derives.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
)


SOURCE_ID = "treehouse-polya-25-01"
SOURCE_COHORT = "TREEHOUSE_POLYA_25_01"
SOURCE_PROJECT = "Treehouse"
CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression" / SOURCE_ID
TPM_FILE = CACHE_ROOT / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
CLINICAL_FILE = (
    CACHE_ROOT
    / "clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv"
)
DERIVED_DIR = CACHE_ROOT / "derived"


# (cancer_code, Treehouse disease label) — pulled from
# docs/expression-data-audit.md Status B Treehouse PolyA table.
COHORTS = [
    ("ATRT", "atypical teratoid/rhabdoid tumor"),
    ("EWS", "Ewing sarcoma"),
    ("HEPB", "hepatoblastoma"),
    ("MBL", "medulloblastoma"),
    ("NUTM", "NUT midline carcinoma"),
    ("OS", "osteosarcoma"),
    ("RMS_ARMS", "alveolar rhabdomyosarcoma"),
    ("RMS_ERMS", "embryonal rhabdomyosarcoma"),
    ("RMS_PRMS", "pleomorphic rhabdomyosarcoma"),
    ("RMS_SSRMS", "spindle cell/sclerosing rhabdomyosarcoma"),
    ("SARC_LMS", "leiomyosarcoma"),
    ("SARC_LPS_UNSPEC", "liposarcoma"),
    ("SARC_MYXFIB", "myxofibrosarcoma"),
    ("SARC_SYN", "synovial sarcoma"),
    ("SARC_UPS", "undifferentiated pleomorphic sarcoma"),
]


def _log(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_clinical_buckets(
    clinical_path: Path, cohorts: list[tuple[str, str]],
) -> dict[str, list[str]]:
    clin = pd.read_csv(clinical_path, sep="\t")
    buckets: dict[str, list[str]] = {}
    norm_disease = clin["disease"].astype(str).str.strip().str.lower()
    for cancer_code, label in cohorts:
        mask = norm_disease.eq(label.strip().lower())
        ids = clin.loc[mask, "th_dataset_id"].astype(str).tolist()
        if not ids:
            raise RuntimeError(
                f"no samples matched disease={label!r} for code={cancer_code}"
            )
        buckets[cancer_code] = ids
        _log(f"  {cancer_code}: {len(ids):>4} samples ({label})")
    return buckets


def _build_or_load_symbol_mapping(
    all_symbols: pd.Index,
    *,
    ensembl_release: int,
    cache_path: Path,
    refresh: bool,
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        _log(f"loading cached symbol mapping from {cache_path}")
        return pd.read_parquet(cache_path)
    _log(
        f"building symbol → Ensembl mapping for {len(all_symbols):,} HUGO "
        f"symbols (release {ensembl_release})..."
    )
    genome = EnsemblRelease(ensembl_release)
    rows: list[dict[str, str]] = []
    resolved = ambiguous = unresolved = 0
    for sym in all_symbols:
        s = str(sym).strip()
        if not s:
            unresolved += 1
            continue
        try:
            genes = genome.genes_by_name(s)
        except Exception:
            genes = []
        ids = {g.gene_id.split(".", 1)[0] for g in genes}
        if len(ids) == 1:
            gene = genes[0]
            rows.append(
                {
                    "source_symbol": s,
                    "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
                    "Symbol": gene.gene_name or s,
                }
            )
            resolved += 1
        elif not ids:
            unresolved += 1
        else:
            ambiguous += 1
    mapping = pd.DataFrame(rows)
    _log(
        f"  resolved={resolved}, ambiguous={ambiguous} (dropped), "
        f"unresolved={unresolved} (dropped)"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(cache_path, index=False)
    _log(f"  wrote symbol mapping cache to {cache_path}")
    return mapping


def _read_tpm_columns(
    tpm_path: Path, sample_cols: list[str],
) -> pd.DataFrame:
    with tpm_path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
    available = set(header)
    missing = [s for s in sample_cols if s not in available]
    if missing:
        raise RuntimeError(
            f"{len(missing)} samples not in TPM header; "
            f"first few: {missing[:5]}"
        )
    gene_col = header[0]
    keep = [gene_col] + sample_cols
    _log(
        f"reading TPM matrix for {len(sample_cols):,} samples (one disk "
        f"scan)..."
    )
    df = pd.read_csv(tpm_path, sep="\t", usecols=keep, low_memory=False)
    return df.set_index(gene_col)


def _inverse_log2(log2_df: pd.DataFrame) -> pd.DataFrame:
    tpm = np.power(2.0, log2_df.to_numpy()) - 1.0
    tpm[tpm < 0] = 0.0
    return pd.DataFrame(tpm, index=log2_df.index, columns=log2_df.columns)


def _aggregate_by_ensembl(
    values_by_symbol: pd.DataFrame, mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_col = values_by_symbol.index.name or "Gene"
    values_flat = values_by_symbol.reset_index().rename(
        columns={symbol_col: "source_symbol"}
    )
    sample_cols = [c for c in values_flat.columns if c != "source_symbol"]
    merged = mapping.merge(values_flat, on="source_symbol", how="inner")
    by_gene = merged.groupby(
        "Ensembl_Gene_ID", as_index=False, sort=False,
    ).agg({"Symbol": "first", **{c: "sum" for c in sample_cols}})
    gene_table = by_gene[["Ensembl_Gene_ID", "Symbol"]].copy()
    values = by_gene.set_index("Ensembl_Gene_ID")[sample_cols]
    return gene_table, values


def _technical_mask(gene_table: pd.DataFrame) -> pd.Series:
    remove_groups = {str(group) for group in _TECHNICAL_RNA_GROUPS}
    qc = [
        classify_gene_qc(symbol, ensembl_id=ensg)
        for symbol, ensg in zip(gene_table["Symbol"], gene_table["Ensembl_Gene_ID"])
    ]
    return pd.Series(
        [klass.group in remove_groups for klass in qc],
        index=gene_table.index,
    )


def _clean_tpm(values: pd.DataFrame, removable: pd.Series) -> pd.DataFrame:
    clean = values.copy()
    clean.loc[removable.to_numpy(), :] = 0.0
    remaining = clean.sum(axis=0)
    scale = pd.Series(np.nan, index=remaining.index, dtype=float)
    positive = remaining > 0
    scale.loc[positive] = 1_000_000.0 / remaining.loc[positive]
    return clean.mul(scale, axis=1).fillna(0.0)


def _build_or_load_per_cohort_tpm(
    cancer_code: str,
    sample_ids: list[str],
    values_full: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    cache_path: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (gene_table, ensembl-keyed per-sample TPM matrix)."""
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        gene_table = cached[["Ensembl_Gene_ID", "Symbol"]].copy()
        sample_cols = [c for c in cached.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
        values = cached.set_index("Ensembl_Gene_ID")[sample_cols]
        return gene_table, values
    cohort_log2 = values_full[sample_ids]
    cohort_tpm = _inverse_log2(cohort_log2)
    gene_table, values = _aggregate_by_ensembl(cohort_tpm, mapping)
    # Persist as a single parquet: gene_table cols + sample columns.
    # Concatenate in one pass to avoid pandas insert-fragmentation.
    cached = pd.concat(
        [gene_table.reset_index(drop=True), values.reset_index(drop=True)],
        axis=1,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path, index=False)
    return gene_table, values


def _summarize_cohort(
    cancer_code: str,
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
    *,
    source_version: str,
    pipeline: str,
    notes: str,
) -> pd.DataFrame:
    clean = _clean_tpm(values, _technical_mask(gene_table))
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cancer_code
    out["source_cohort"] = SOURCE_COHORT
    out["source_project"] = SOURCE_PROJECT
    out["source_version"] = source_version
    assign_stats(out, values, clean)
    out["processing_pipeline"] = pipeline
    out["notes"] = notes
    return round_stat_columns(out)[list(REFERENCE_COLUMNS)]


def _upsert_many(
    summary_path: Path,
    new_rows: pd.DataFrame,
    *,
    source_cohort: str,
    cancer_codes: list[str],
) -> pd.DataFrame:
    if summary_path.exists():
        existing = pd.read_csv(summary_path, low_memory=False)
        keep = ~(
            existing["cancer_code"].astype(str).isin(cancer_codes)
            & existing["source_cohort"].astype(str).eq(source_cohort)
        )
        out = pd.concat(
            [existing.loc[keep].reindex(columns=list(REFERENCE_COLUMNS)), new_rows],
            ignore_index=True,
        )
    else:
        out = new_rows.copy()
    out = out.reindex(columns=list(REFERENCE_COLUMNS)).sort_values(
        ["cancer_code", "source_cohort", "Ensembl_Gene_ID"],
        na_position="last",
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(summary_path, index=False, compression="gzip")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ensembl-release", default=112, type=int,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression.csv.gz"),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached symbol mapping + per-cohort TPM matrices "
             "and rebuild from the raw 6 GB matrix.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated cancer_codes; restrict the sweep to these.",
    )
    args = parser.parse_args()

    if not TPM_FILE.exists() or not CLINICAL_FILE.exists():
        raise SystemExit(
            f"Missing Treehouse files under {CACHE_ROOT}. Fetch first."
        )

    cohorts = COHORTS
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        cohorts = [(c, lbl) for c, lbl in COHORTS if c in wanted]
        if not cohorts:
            raise SystemExit(f"--only={args.only!r} matched no cohorts")
    _log(f"sweep targets: {[c for c, _ in cohorts]}")

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    _log(f"reading clinical from {CLINICAL_FILE}...")
    buckets = _load_clinical_buckets(CLINICAL_FILE, cohorts)
    all_samples = sorted({s for ids in buckets.values() for s in ids})
    _log(
        f"  union: {len(all_samples):,} distinct samples across "
        f"{len(buckets)} cohorts"
    )

    log2_values = _read_tpm_columns(TPM_FILE, all_samples)
    _log(f"  TPM frame shape: {log2_values.shape}")

    symbol_cache = DERIVED_DIR / f"symbol_to_ensembl_{args.ensembl_release}.parquet"
    mapping = _build_or_load_symbol_mapping(
        log2_values.index,
        ensembl_release=args.ensembl_release,
        cache_path=symbol_cache,
        refresh=args.refresh_cache,
    )

    source_version = (
        f"Treehouse Tumor Compendium 25.01 PolyA "
        f"(hugo_log2tpm_58581genes_2025-02-27); "
        f"downloaded from public.gi.ucsc.edu/~ekephart/public-data/; "
        f"log2(TPM+1) inverse-transformed; "
        f"HUGO symbols harmonized to Ensembl release {args.ensembl_release}"
    )
    pipeline = (
        f"treehouse_polya_25_01_log2tpm_to_tpm_"
        f"ensembl{args.ensembl_release}_clean_tpm_v1"
    )

    per_cohort_summaries: list[pd.DataFrame] = []
    for cancer_code, label in cohorts:
        _log(f"== {cancer_code} ({label}) ==")
        sample_ids = buckets[cancer_code]
        cache_path = DERIVED_DIR / f"{cancer_code}_per_sample_tpm.parquet"
        gene_table, values = _build_or_load_per_cohort_tpm(
            cancer_code,
            sample_ids,
            log2_values,
            mapping,
            cache_path=cache_path,
            refresh=args.refresh_cache,
        )
        notes = (
            "Per-sample TPMs from the Treehouse Tumor Compendium 25.01 "
            f"PolyA (hugo_log2tpm matrix, inverse-log2 transformed). "
            f"Sample selection: clinical.disease == '{label}'. HUGO "
            f"symbols mapped to Ensembl release {args.ensembl_release}; "
            "duplicate symbol mappings dropped. TPM_clean is computed "
            "per-sample by technical-RNA zeroing + denominator rescaling."
        )
        summary = _summarize_cohort(
            cancer_code,
            gene_table,
            values,
            source_version=source_version,
            pipeline=pipeline,
            notes=notes,
        )
        _log(
            f"  {cancer_code}: {len(summary):,} gene rows ready; cached "
            f"per-sample matrix at {cache_path.name}"
        )
        per_cohort_summaries.append(summary)

    combined_new = pd.concat(per_cohort_summaries, ignore_index=True)
    _log(
        f"writing {len(combined_new):,} new rows across "
        f"{len(per_cohort_summaries)} cohorts into {args.summary_output}..."
    )
    combined = _upsert_many(
        Path(args.summary_output),
        combined_new,
        source_cohort=SOURCE_COHORT,
        cancer_codes=[c for c, _ in cohorts],
    )
    _log(f"  {len(combined):,} total rows in {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
