"""Emit per-sample TPM parquets for the neuroendocrine cohorts (#318).

The packaged NE references are summary-only (per-cohort medians), so the
representative-samples artifact (#312) had no NE coverage and trufflepig's
sample-level battery left the NE axis unscored. This produces the per-sample
joint matrices the representatives generator needs, **reusing the existing NE
builders' parsing** (so the per-sample TPM is identical to what the summaries
were aggregated from) and writing them in the standard
``<source>/derived/<CODE>_per_sample_tpm.parquet`` location (linear raw TPM;
the representatives generator applies clean_tpm_v4 + medoid selection).

Covers the two cleanly TPM-scale NE sources (the canonical NE poles): pancreatic
NET (GSE118014 log2TPM) and small-cell lung cancer (UCologne FPKM). The
counts-based NE sources (GSE98894 midgut NET; IARC LNEN lung carcinoid / LCNEC)
need Entrez + gene-length TPM conversion and are a follow-up.

These parquets live in the local cache only (build artifacts) — they are NOT
shipped; only the resulting representative medoids are bundled.

Run:  python scripts/build_ne_per_sample_parquets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pirlygenes.builders.treehouse import (  # noqa: E402
    _aggregate_by_ensembl,
    _build_or_load_symbol_mapping,
    _inverse_log2,
)

import build_pannet_reference_expression as pannet  # noqa: E402
import build_sclc_reference_expression as sclc  # noqa: E402

CACHE = Path.home() / ".cache" / "pirlygenes" / "expression"
ENSEMBL = 112


def _write(gene_table: pd.DataFrame, values: pd.DataFrame, source_dir: Path,
           code: str) -> int:
    derived = source_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    out = pd.concat(
        [gene_table[["Ensembl_Gene_ID", "Symbol"]].reset_index(drop=True),
         values.reset_index(drop=True)],
        axis=1,
    )
    out.to_parquet(derived / f"{code}_per_sample_tpm.parquet", index=False)
    n = values.shape[1]
    print(f"  {code}: {len(out)} genes × {n} samples -> "
          f"{derived / f'{code}_per_sample_tpm.parquet'}", flush=True)
    return n


def _mapping(symbols, source_dir: Path):
    return _build_or_load_symbol_mapping(
        symbols, ensembl_release=ENSEMBL,
        cache_path=source_dir / f"symbol_to_ensembl_{ENSEMBL}.parquet",
        refresh=False)


def build_net_pancreas() -> None:
    d = CACHE / "gse118014-pannet"
    log2v = pannet._read_log2tpm(d / "GSE118014_PanNETs_log2TPM.txt.gz")
    tpm = _inverse_log2(log2v)
    gene_table, values = _aggregate_by_ensembl(tpm, _mapping(log2v.index, d))
    _write(gene_table, values, d, "NET_PANCREAS")


def build_sclc() -> None:
    d = CACHE / "sclc-ucologne-2015"
    fpkm = sclc._read_fpkm(d / "data_mrna_seq_fpkm.txt")
    tpm = sclc._fpkm_to_tpm(fpkm)
    gene_table, values = _aggregate_by_ensembl(tpm, _mapping(fpkm.index, d))
    _write(gene_table, values, d, "SCLC")


def build_lung_ne() -> None:
    """Lung NE (NET_LUNG carcinoid + NEC_LUNG_LARGECELL) from the IARC DRMetrics
    pan-LNEN counts, reusing the LNEN builder's counts→TPM + histology split."""
    from pirlygenes.builders.geo_matrix import (
        _gene_lengths_kb_for_index,
        harmonize_gene_ids,
        normalize_to_tpm,
        read_matrix,
    )
    from pirlygenes.gene_sets_cancer import canonical_cancer_code

    import build_lnen_drmetrics_reference_expression as lnen

    d = CACHE / "drmetrics-lnen-2020"
    attrs = pd.read_csv(d / "Attributes.txt", sep="\t",
                        usecols=["Sample_ID", "Histopathology_simplified"])
    matrix = read_matrix(d / "read_counts_all.txt", sep=r"\s+",
                         gene_id_col="gene_id", drop_cols=())
    lengths = _gene_lengths_kb_for_index(matrix.index, gene_id_type="ensembl",
                                         ensembl_release=ENSEMBL)
    tpm = normalize_to_tpm(matrix, unit="raw_counts", gene_lengths_kb=lengths)
    mapping, values = harmonize_gene_ids(tpm, gene_id_type="ensembl",
                                         ensembl_release=ENSEMBL)
    gene_table = (mapping.drop_duplicates("Ensembl_Gene_ID")
                  [["Ensembl_Gene_ID", "Symbol"]].reset_index(drop=True))
    values = values.reindex(gene_table["Ensembl_Gene_ID"]).fillna(0.0)
    sample_to_code = {r.Sample_ID: lnen.HISTOLOGY_TO_CODE.get(
        r.Histopathology_simplified) for r in attrs.itertuples(index=False)}
    by_code: dict[str, list[str]] = {}
    for col in values.columns:
        raw_code = sample_to_code.get(col)
        if raw_code:
            by_code.setdefault(canonical_cancer_code(raw_code), []).append(col)
    for code, cols in by_code.items():
        _write(gene_table, values[cols], d, code)


def main() -> int:
    print("building NE per-sample TPM parquets...", flush=True)
    build_net_pancreas()
    build_sclc()
    build_lung_ne()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
