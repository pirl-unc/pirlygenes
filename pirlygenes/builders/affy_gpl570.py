"""GPL570 (Affymetrix HG-U133 Plus 2.0) microarray → TPM-proxy builder.

Microarray expression is not the same scale as RNA-seq TPM. GPL570
arrays measure 54,675 probes; each probe maps to 0..1 genes. Multi-
probe genes are usually summarized here by *max* (most expressed
probe set wins).

GEO's series_matrix files typically ship intensities that have been
RMA- or GC-RMA-normalized, which means they're already on a **log2**
scale. A handful of older studies ship MAS5 instead, which is on a
*linear* scale. The builder sniffs which one a given matrix uses
(values mostly < 50 → assume log2; otherwise leave linear), and
exponentiates only when needed. Either way the post-exponentiation
matrix is then per-sample sum-to-1e6 to produce a TPM PROXY.

**Caveats baked into the output:**
- This is *not* directly comparable to RNA-seq TPM in absolute
  magnitude — microarray dynamic range, probe-design saturation, and
  whole-transcriptome coverage all differ.
- Within-sample gene *rank* is preserved (useful for "which genes are
  expressed at all in this sample?" questions) but absolute TPM
  values should not be cross-compared to RNA-seq cohorts.
- Sniffing log2 vs linear is heuristic; if you know the input is
  MAS5 (or some other linear-scale variant), this builder will leave
  it linear and the sum-to-1e6 result is still a valid TPM proxy.

The caveat is encoded in two places: the ``notes`` column carries a
human-readable warning, and ``processing_pipeline`` is tagged
``gpl570_microarray_tpm_proxy_...`` so programmatic consumers can
detect-and-filter.

Probe→gene mapping uses GPL570's standard GEO platform-table
(downloaded via GEO's ``acc.cgi`` text view) — the ``Gene symbol``
column gives HUGO symbols.
"""

from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .geo_matrix import _clean_tpm, _technical_mask
from .treehouse import _build_or_load_symbol_mapping
from ..expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)

GPL570_ANNOT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?targ=self&form=text&view=full&acc=GPL570"
)
GPL570_ANNOT_LOCAL = "GPL570.platform_table.txt"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _open_maybe_gzip(path: Path):
    """Open path as text; transparently gunzip if .gz suffix."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _parse_gpl570_annot(annot_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: probe_id, gene_symbol."""
    with _open_maybe_gzip(annot_path) as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                break
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            if line.startswith("!platform_table_end"):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))
            rows.append(parts[: len(header)])
    df = pd.DataFrame(rows, columns=header)
    probe_col = next(
        (c for c in ("ID", "ID_REF", "Probe Set ID") if c in df.columns), None
    )
    symbol_col = next(
        (c for c in ("Gene symbol", "Gene Symbol", "Symbol") if c in df.columns),
        None,
    )
    if probe_col is None or symbol_col is None:
        raise RuntimeError(
            f"GPL570 annot missing probe/symbol cols: {list(df.columns)[:15]}"
        )
    out = df[[probe_col, symbol_col]].rename(
        columns={probe_col: "probe_id", symbol_col: "gene_symbol"}
    )
    out["gene_symbol"] = (
        out["gene_symbol"].astype(str).str.split("///").str[0].str.strip()
    )
    out = out[
        out["gene_symbol"].ne("")
        & out["gene_symbol"].ne("---")
        & out["probe_id"].ne("")
    ].copy()
    return out


def parse_series_matrix(
    series_matrix_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Parse a GEO series_matrix.txt.gz file.

    Returns ``(probe_intensity_matrix, {sample_id: {field: value}})``.
    """
    metadata: dict[str, list[str]] = {}
    char_rows: list[list[str]] = []
    with gzip.open(series_matrix_path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                break
            if not line.startswith("!Sample_"):
                continue
            key, *values = line.split("\t")
            key = key.lstrip("!").rstrip()
            cleaned = [v.strip().strip('"') for v in values]
            if key == "Sample_characteristics_ch1":
                char_rows.append(cleaned)
            else:
                metadata.setdefault(key, []).extend(cleaned)

        header = next(f).rstrip("\n").split("\t")
        header = [h.strip().strip('"') for h in header]
        sample_ids = header[1:]
        rows = []
        probe_ids = []
        for line in f:
            if line.startswith("!series_matrix_table_end"):
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            probe_ids.append(parts[0].strip().strip('"'))
            rows.append(
                [
                    pd.to_numeric(v.strip().strip('"'), errors="coerce")
                    for v in parts[1:]
                ]
            )
    matrix = pd.DataFrame(rows, index=probe_ids, columns=sample_ids).astype(float)
    matrix.index.name = "probe_id"

    sample_meta: dict[str, dict[str, str]] = {sid: {} for sid in sample_ids}
    for key, values in metadata.items():
        if len(values) == len(sample_ids):
            for sid, val in zip(sample_ids, values):
                sample_meta[sid][key] = val
    for row in char_rows:
        if len(row) != len(sample_ids):
            continue
        for sid, val in zip(sample_ids, row):
            if not val:
                continue
            if ":" in val:
                ck, cv = val.split(":", 1)
                cs_key = "char_" + ck.strip().lower().replace(" ", "_")
                sample_meta[sid].setdefault(cs_key, cv.strip())
            else:
                sample_meta[sid].setdefault("char", val.strip())
    return matrix, sample_meta


def build_gpl570_source(
    *,
    series_matrix_url: str,
    series_matrix_filename: str,
    cache_dir: Path,
    cancer_code: str,
    source_cohort: str,
    source_project: str,
    citation: str,
    summary_output: Path,
    ensembl_release: int = 112,
    sample_include_regex: str | None = None,
    sample_exclude_regex: str | None = None,
    extra_notes: str = "",
    tumor_origin: str = "primary",
    metastasis_site: str | None = None,
) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    series_path = cache_dir / series_matrix_filename
    annot_path = cache_dir / GPL570_ANNOT_LOCAL

    print(f"downloading {series_matrix_filename}...")
    _download(series_matrix_url, series_path)
    print("downloading GPL570 annotation (via GEO acc.cgi)...")
    _download(GPL570_ANNOT_URL, annot_path)

    print("parsing series_matrix...")
    intensities, sample_meta = parse_series_matrix(series_path)
    print(f"  probes={intensities.shape[0]} samples={intensities.shape[1]}")

    if sample_include_regex:
        rgx = re.compile(sample_include_regex)
        keep = [
            sid
            for sid in intensities.columns
            if any(rgx.search(str(v)) for v in sample_meta[sid].values())
        ]
        print(
            f"  include filter: {len(keep)} of {intensities.shape[1]} samples"
        )
        intensities = intensities[keep]
    if sample_exclude_regex:
        rgx = re.compile(sample_exclude_regex)
        drop_ids = {
            sid
            for sid in intensities.columns
            if any(rgx.search(str(v)) for v in sample_meta[sid].values())
        }
        keep = [sid for sid in intensities.columns if sid not in drop_ids]
        print(
            f"  exclude filter: {len(keep)} of {intensities.shape[1]} samples"
        )
        intensities = intensities[keep]

    if intensities.shape[1] == 0:
        raise RuntimeError("no samples after filter")

    print("parsing GPL570 probe→gene annotation...")
    annot = _parse_gpl570_annot(annot_path)
    print(f"  {len(annot)} probes with HUGO assignment")

    print("aggregating probe → gene (max) and converting to TPM-proxy...")
    intensities.index = intensities.index.astype(str)
    annot_indexed = annot.set_index("probe_id")["gene_symbol"]
    joined = intensities.join(annot_indexed, how="inner")
    by_gene = joined.groupby("gene_symbol").max()
    looks_log2 = float(np.nanmax(by_gene.values)) < 50.0
    linear = np.power(2.0, by_gene) if looks_log2 else by_gene.copy()
    sums = linear.sum(axis=0)
    tpm_proxy = (
        linear.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0
    )

    print(f"harmonizing HUGO → Ensembl release {ensembl_release}...")
    mapping = _build_or_load_symbol_mapping(
        tpm_proxy.index,
        ensembl_release=ensembl_release,
        cache_path=cache_dir / f"symbol_to_ensembl_{ensembl_release}.parquet",
        refresh=False,
    )
    flat = tpm_proxy.reset_index().rename(
        columns={"gene_symbol": "source_symbol"}
    )
    merged = mapping.merge(flat, on="source_symbol", how="inner")
    sample_cols = [
        c
        for c in merged.columns
        if c not in {"source_symbol", "Ensembl_Gene_ID", "Symbol"}
    ]
    by_ensg = merged.groupby("Ensembl_Gene_ID", as_index=True, sort=False)[
        sample_cols
    ].sum()
    gene_table = (
        mapping.drop_duplicates("Ensembl_Gene_ID")[["Ensembl_Gene_ID", "Symbol"]]
        .reset_index(drop=True)
    )
    by_ensg = by_ensg.reindex(gene_table["Ensembl_Gene_ID"]).fillna(0.0)
    print(f"  canonical genes: {len(gene_table)}")

    clean = _clean_tpm(by_ensg, _technical_mask(gene_table))
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cancer_code
    out["source_cohort"] = source_cohort
    out["source_project"] = source_project
    out["source_version"] = (
        f"GPL570 microarray; series_matrix log2 intensity → probe-max → "
        f"anti-log2 → per-sample sum-to-1e6 (TPM proxy); HUGO "
        f"harmonized to Ensembl release {ensembl_release}"
    )
    assign_stats(out, by_ensg, clean)
    out["processing_pipeline"] = (
        f"gpl570_microarray_tpm_proxy_ensembl{ensembl_release}_clean_tpm_v1"
    )
    out["notes"] = (
        f"GPL570 (Affy HG-U133 Plus 2.0) microarray-derived TPM-proxy "
        f"(n={by_ensg.shape[1]}). Not directly comparable to RNA-seq TPM "
        f"in absolute magnitude — preserves within-sample gene rank only. "
        f"Citation: {citation}. {extra_notes}"
    ).strip()
    out["tumor_origin"] = tumor_origin
    out["metastasis_site"] = metastasis_site if metastasis_site else pd.NA
    out = round_stat_columns(out)[list(REFERENCE_COLUMNS)]

    upsert_to_shard(
        summary_output,
        out,
        source_cohort=source_cohort,
        cancer_codes=[cancer_code],
    )
    print(f"upserted {len(out)} rows into {source_cohort}.csv.gz")
    return by_ensg.shape[1]


__all__ = ["build_gpl570_source", "parse_series_matrix"]
