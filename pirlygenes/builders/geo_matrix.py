"""Generic GEO / GSA / cBioPortal expression-matrix builder.

Reads a single supplementary-file expression matrix (TPM / FPKM /
RPKM / log2(TPM+1) / raw counts) keyed by gene IDs (Ensembl / HUGO
symbol / Entrez), normalizes everything to per-sample TPM, applies
the technical-RNA filter + renormalize, and computes the v5.3 stat
suite. Output goes to the standard per-source shard via
``pirlygenes.expression.stats.upsert_to_shard``.

Used by ``scripts/build_geo_matrix.py``, which looks up source
config from ``pirlygenes/data/expression_sources.yaml`` by source-id
and dispatches here. All builders that follow the
single-supplementary-file pattern go through this module so the
``pirlygenes build <source-id>`` CLI sees them uniformly.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from ..expression.normalize import (
    clean_tpm_matrix as _clean_tpm,
    technical_rna_mask as _technical_mask,
)
from ..expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
from .gene_mapping import (
    GeneIdType,
    aggregate_matrix_by_mapping,
    detect_id_type,
    gene_from_ensembl_id,
    harmonize_entrez_matrix,
    resolve_symbol,
    strip_version,
)

Unit = Literal["TPM", "FPKM", "RPKM", "log2(TPM+1)", "raw_counts"]


@dataclass(frozen=True)
class GeoMatrixSource:
    """One GEO-style supplementary-matrix expression source.

    The file is expected to be a single TSV/CSV (optionally gzipped)
    with one row per gene and one column per sample, plus a column
    (named or unnamed) of gene IDs. The file may also have extra
    annotation columns (e.g. ``Entrez_Gene_Id``) which should be
    listed in ``drop_cols``.
    """

    cancer_code: str | list[str]
    source_cohort: str
    source_project: str
    citation: str
    file_url: str
    file_name: str           # local filename in cache_dir
    unit: Unit
    gene_id_col: str = ""    # "" = first column (index_col=0)
    gene_id_type: GeneIdType = "auto"
    drop_cols: tuple[str, ...] = ()  # extra non-sample columns to drop
    sep: str = "\t"
    transposed: bool = False  # True if matrix is samples-as-rows, genes-as-cols
    sample_filter: Callable[[list[str]], list[str]] | None = None
    sample_to_cancer_code: Callable[[str], str | None] | None = None
    notes: str = ""
    pipeline_stem: str = ""  # for processing_pipeline; defaults to source_cohort.lower()
    # v5.4 schema: cohort-level provenance. Defaults are correct for
    # the Tier-A GEO-matrix sources currently registered (all primary
    # tumor cohorts). Sources covering metastatic or mixed material
    # should override.
    tumor_origin: str = "primary"
    metastasis_site: str | None = None


# ─── ID type detection ──────────────────────────────────────────────────────

# Canonical implementation lives in gene_mapping; kept here under the
# historical name so existing callers (and __all__) keep working.
detect_gene_id_type = detect_id_type


# ─── Reading + cleanup ──────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _open_text(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("r")


def read_matrix(path: Path, *, sep: str, gene_id_col: str, drop_cols: tuple[str, ...]) -> pd.DataFrame:
    """Return a DataFrame indexed by gene ID, columns = sample IDs.

    ``gene_id_col=""`` treats the first column as the gene-ID index
    even if its header is blank (common in GEO supplementaries).
    Supports any pandas-accepted ``sep`` including regex like ``r"\\s+"``
    for whitespace-separated files (pandas needs engine='python' for
    multi-char or regex separators).
    """
    # Multi-char or regex separators need the python engine.
    engine = "python" if (len(sep) > 1 or "\\" in sep) else "c"
    read_kwargs = {"sep": sep, "engine": engine}
    if engine == "python":
        read_kwargs.pop("low_memory", None)
    if gene_id_col == "":
        df = pd.read_csv(path, index_col=0, **read_kwargs)
    else:
        df = pd.read_csv(path, **read_kwargs)
        if gene_id_col not in df.columns:
            raise RuntimeError(
                f"gene_id_col={gene_id_col!r} not in columns: {list(df.columns)[:10]}"
            )
        df = df.dropna(subset=[gene_id_col]).set_index(gene_id_col)
    df.index = df.index.astype(str).str.strip()
    df = df[df.index != ""]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)
    # Coerce sample columns to numeric; drop any that don't parse at
    # all (these are typically gene-annotation columns like Symbol /
    # locus / Description that survived index_col=0 but aren't actual
    # samples). Distinguish them from real samples that happen to
    # contain a single NaN: a non-sample column will parse to all NaN.
    for col in list(df.columns):
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            df = df.drop(columns=col)
        else:
            df[col] = numeric.fillna(0.0)
    return df


# ─── Unit normalization → TPM ───────────────────────────────────────────────

def _per_sample_to_tpm(matrix: pd.DataFrame) -> pd.DataFrame:
    """Renormalize each column to sum to 1e6 (the TPM identity)."""
    sums = matrix.sum(axis=0)
    return matrix.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0


def _inverse_log2(matrix: pd.DataFrame) -> pd.DataFrame:
    arr = np.power(2.0, matrix.to_numpy()) - 1.0
    arr[arr < 0] = 0.0
    return pd.DataFrame(arr, index=matrix.index, columns=matrix.columns)


def _counts_to_tpm_via_gene_lengths(
    counts: pd.DataFrame, gene_lengths_kb: pd.Series,
) -> pd.DataFrame:
    """Length-normalized TPM from raw counts: counts/length_kb → per-sample 1e6 renorm."""
    common = counts.index.intersection(gene_lengths_kb.index)
    counts = counts.loc[common]
    lengths_kb = gene_lengths_kb.loc[common].replace(0, np.nan)
    rpk = counts.div(lengths_kb, axis=0).fillna(0.0)
    sums = rpk.sum(axis=0)
    return rpk.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0


def normalize_to_tpm(
    matrix: pd.DataFrame, *, unit: Unit, gene_lengths_kb: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert any supported unit to per-sample TPM (sums to 1e6).

    Even ``unit='TPM'`` inputs are re-renormalized — some published
    "TPM" matrices don't strictly sum to 1e6 per sample (the authors
    may have filtered or transformed without restoring the sum), so
    we enforce the per-sample sum-to-1e6 invariant here to keep raw
    stats comparable across cohorts.
    """
    if unit == "TPM":
        return _per_sample_to_tpm(matrix)
    if unit in {"FPKM", "RPKM"}:
        return _per_sample_to_tpm(matrix)
    if unit == "log2(TPM+1)":
        return _inverse_log2(matrix)
    if unit == "raw_counts":
        if gene_lengths_kb is None:
            raise RuntimeError(
                "raw_counts requires gene_lengths_kb (use _gene_lengths_kb_for_index)"
            )
        return _counts_to_tpm_via_gene_lengths(matrix, gene_lengths_kb)
    raise RuntimeError(f"unsupported unit: {unit!r}")


# ─── Gene ID harmonization (Ensembl / HUGO / Entrez → Ensembl 112) ──────────

def _harmonize_by_ensembl_id(
    matrix: pd.DataFrame, ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strip version suffix; map to (Ensembl_Gene_ID, Symbol) via pyensembl."""
    genome = EnsemblRelease(ensembl_release)
    rows = []
    for raw in matrix.index:
        result = gene_from_ensembl_id(genome, raw)
        if result is None:
            continue
        ensembl_id, name = result
        rows.append({
            "source_id": str(raw),
            "Ensembl_Gene_ID": ensembl_id,
            "Symbol": name,
        })
    mapping = pd.DataFrame(rows)
    return mapping, aggregate_matrix_by_mapping(matrix, mapping)


def _harmonize_by_symbol(
    matrix: pd.DataFrame, ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve each HUGO symbol via the shared resolver (direct → synonym
    rescue); drop ambiguous (>1 gene) and unresolved."""
    genome = EnsemblRelease(ensembl_release)
    rows = []
    for raw in matrix.index:
        result = resolve_symbol(genome, str(raw).strip())
        if result is None:
            continue
        ensembl_id, name, _method = result
        rows.append({
            "source_id": str(raw).strip(),
            "Ensembl_Gene_ID": ensembl_id,
            "Symbol": name,
        })
    mapping = pd.DataFrame(rows)
    return mapping, aggregate_matrix_by_mapping(matrix, mapping)


def harmonize_gene_ids(
    matrix: pd.DataFrame, *, gene_id_type: GeneIdType, ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect ID type if needed, then dispatch to the right harmonizer.

    Returns (mapping_table, gene_indexed_tpm_matrix). The mapping has
    columns (source_id, Ensembl_Gene_ID, Symbol). Entrez input goes
    through the shared NCBI-backed chain (dbXrefs → current-symbol →
    gene_history); Ensembl/HUGO go through pyensembl.
    """
    actual = gene_id_type
    if gene_id_type == "auto":
        actual = detect_gene_id_type(matrix.index)
    if actual == "ensembl":
        return _harmonize_by_ensembl_id(matrix, ensembl_release)
    if actual == "entrez":
        return harmonize_entrez_matrix(matrix, ensembl_release=ensembl_release)
    return _harmonize_by_symbol(matrix, ensembl_release)


def _gene_lengths_kb_for_index(
    ensembl_ids_or_symbols: Iterable[str], *,
    gene_id_type: GeneIdType, ensembl_release: int,
) -> pd.Series:
    """Map a gene index to gene lengths (kb) via pyensembl.

    Used for raw_counts → TPM length-normalization. Returns a Series
    indexed by the SAME identifier the matrix is indexed by.
    """
    genome = EnsemblRelease(ensembl_release)
    out: dict[str, float] = {}
    for raw in ensembl_ids_or_symbols:
        rid = str(raw).strip()
        if not rid:
            continue
        gene = None
        if gene_id_type == "ensembl":
            try:
                gene = genome.gene_by_id(strip_version(rid))
            except Exception:
                pass
        else:
            try:
                gs = genome.genes_by_name(rid)
            except Exception:
                gs = []
            ids = {strip_version(g.gene_id) for g in gs}
            if len(ids) == 1:
                gene = gs[0]
        if gene is None:
            continue
        try:
            length = float(gene.length)
        except Exception:
            length = float("nan")
        out[rid] = length / 1000.0
    return pd.Series(out, name="gene_length_kb")


# ─── Tech-RNA filter + clean TPM: imported from expression.normalize ────────
# (_clean_tpm / _technical_mask are aliases of the shared helpers, re-exported
# at the top of this module so builders can pull both from one place.)


# ─── End-to-end build ───────────────────────────────────────────────────────

def build_source(
    source: GeoMatrixSource,
    *,
    cache_dir: Path,
    summary_output: Path,
    ensembl_release: int = 112,
) -> dict[str, int]:
    """Download, parse, normalize, harmonize, summarize, and upsert."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / source.file_name
    print(f"downloading {source.file_name}...")
    _download(source.file_url, file_path)

    print(f"reading matrix (unit={source.unit}, sep={source.sep!r})...")
    matrix = read_matrix(
        file_path,
        sep=source.sep,
        gene_id_col=source.gene_id_col,
        drop_cols=source.drop_cols,
    )
    if source.transposed:
        # File is samples-as-rows × genes-as-cols; transpose so the
        # rest of the pipeline (genes-as-rows, samples-as-cols) works.
        matrix = matrix.T
        matrix.index.name = source.gene_id_col or "gene_id"
        print(f"  transposed: now shape {matrix.shape} (genes × samples)")
    if source.sample_filter is not None:
        keep = source.sample_filter(list(matrix.columns))
        matrix = matrix[keep]
    print(f"  shape: {matrix.shape} (genes × samples)")
    if matrix.shape[1] < 1:
        raise RuntimeError("matrix has 0 samples after filter")

    # Detect ID type
    gene_id_type = source.gene_id_type
    if gene_id_type == "auto":
        gene_id_type = detect_gene_id_type(matrix.index)
    print(f"  detected gene_id_type: {gene_id_type}")

    # Unit normalization → TPM
    if source.unit == "raw_counts":
        print("  computing gene lengths (kb) for length-normalization...")
        lengths_kb = _gene_lengths_kb_for_index(
            matrix.index, gene_id_type=gene_id_type, ensembl_release=ensembl_release,
        )
        print(f"  got lengths for {len(lengths_kb)} / {len(matrix.index)} rows")
        tpm = normalize_to_tpm(matrix, unit=source.unit, gene_lengths_kb=lengths_kb)
    else:
        tpm = normalize_to_tpm(matrix, unit=source.unit)
    print(f"  per-sample sums (after TPM normalize, first 3): "
          f"{tpm.sum(axis=0).head(3).to_dict()}")

    # Harmonize → Ensembl
    print(f"  harmonizing → Ensembl release {ensembl_release}...")
    mapping, values = harmonize_gene_ids(
        tpm, gene_id_type=gene_id_type, ensembl_release=ensembl_release,
    )
    print(f"  resolved {len(mapping)}/{len(tpm.index)} rows → "
          f"{len(values)} canonical genes")

    # Build per-gene-per-cohort rows. The cancer_code may be a list when
    # the same matrix splits across multiple codes (e.g. NET_LUNG vs
    # NEC_LUNG_LARGECELL) via sample_to_cancer_code.
    gene_table = (
        mapping.drop_duplicates("Ensembl_Gene_ID")[["Ensembl_Gene_ID", "Symbol"]]
        .reset_index(drop=True)
    )
    values = values.reindex(gene_table["Ensembl_Gene_ID"]).fillna(0.0)

    if source.sample_to_cancer_code is not None:
        cohort_to_cols: dict[str, list[str]] = {}
        for col in values.columns:
            code = source.sample_to_cancer_code(col)
            if code is None:
                continue
            cohort_to_cols.setdefault(code, []).append(col)
    else:
        if isinstance(source.cancer_code, list):
            raise RuntimeError(
                "cancer_code is a list but no sample_to_cancer_code provided"
            )
        cohort_to_cols = {source.cancer_code: list(values.columns)}

    summaries: list[pd.DataFrame] = []
    counts_by_code: dict[str, int] = {}
    for code, cols in cohort_to_cols.items():
        if not cols:
            continue
        sub_values = values[cols]
        clean = _clean_tpm(sub_values, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = source.source_cohort
        out["source_project"] = source.source_project
        out["source_version"] = (
            f"{source.citation}; gene-id-type={gene_id_type}; "
            f"unit={source.unit}; harmonized to Ensembl release "
            f"{ensembl_release}"
        )
        assign_stats(out, sub_values, clean)
        pipeline_stem = source.pipeline_stem or source.source_cohort.lower()
        out["processing_pipeline"] = (
            f"{pipeline_stem}_{source.unit.lower().replace('(','').replace(')','').replace('+','plus')}"
            f"_to_tpm_ensembl{ensembl_release}_clean_tpm_v4"
        )
        out["notes"] = source.notes or (
            f"Per-sample expression from {source.source_cohort} (n={len(cols)}). "
            f"Unit-normalized to TPM; tech-RNA-zeroed; v5.3 stats."
        )
        out["tumor_origin"] = source.tumor_origin
        out["metastasis_site"] = source.metastasis_site if source.metastasis_site else pd.NA
        out = round_stat_columns(out)[list(REFERENCE_COLUMNS)]
        summaries.append(out)
        counts_by_code[code] = len(cols)
        print(f"    {code}: n={len(cols)} → {len(out)} gene rows")

    if not summaries:
        raise RuntimeError("no cohort had samples after filtering")
    combined = pd.concat(summaries, ignore_index=True)
    upsert_to_shard(
        summary_output, combined,
        source_cohort=source.source_cohort,
        cancer_codes=list(counts_by_code.keys()),
    )
    print(f"  upserted {len(combined)} rows into shard "
          f"{source.source_cohort}.csv.gz")
    return counts_by_code


__all__ = [
    "GeoMatrixSource",
    "GeneIdType",
    "Unit",
    "detect_gene_id_type",
    "read_matrix",
    "normalize_to_tpm",
    "harmonize_gene_ids",
    "build_source",
    # Re-exported shared clean-TPM helpers (builders import them from here).
    "_clean_tpm",
    "_technical_mask",
]
