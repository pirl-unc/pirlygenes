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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from ..expression.normalize import (
    clean_tpm_removal_mask,
    clean_tpm_matrix as _clean_tpm,
    technical_rna_mask as _technical_mask,
)
from ..expression.stats import (
    assign_stats,
    finalize_reference_rows,
    upsert_to_shard,
)
from .gene_mapping import (
    GeneIdType,
    aggregate_matrix_by_mapping,
    detect_id_type,
    entrez_to_gene,
    gene_from_ensembl_id,
    resolve_symbol,
    strip_version,
)

Unit = Literal["TPM", "FPKM", "RPKM", "log2(TPM+1)", "raw_counts"]


@dataclass(frozen=True)
class MatrixReadDiagnostics:
    """Diagnostics from parsing a source matrix before numeric NaNs are filled."""

    numeric_value_count: pd.Series
    numeric_missing_count: pd.Series
    dropped_non_sample_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleQcConfig:
    """Generic sparse-expression sample QC for full-transcriptome RNA-seq sources."""

    enabled: bool = True
    min_detected_genes: int = 5_000
    min_universal_nonzero_genes: int = 5
    min_universal_nonzero_fraction: float = 0.5
    universal_floor_tpm: float = 1.0
    max_top_gene_fraction: float | None = None
    max_top10_fraction: float | None = None
    source_scale_class: str = ""
    # ``None`` (the default) derives comparability from ``source_scale_class`` so
    # the two never desync; set an explicit bool only to override that. Genuine
    # non-linear proxies (microarray / rank / percentile scales) resolve False.
    linear_tpm_comparable: bool | None = None
    special_source_warning: str = ""


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
    sample_qc: SampleQcConfig = field(default_factory=SampleQcConfig)


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


def read_matrix_with_diagnostics(
    path: Path, *, sep: str, gene_id_col: str, drop_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, MatrixReadDiagnostics]:
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
    numeric_value_counts: dict[str, int] = {}
    numeric_missing_counts: dict[str, int] = {}
    dropped_non_sample_columns: list[str] = []
    for col in list(df.columns):
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            df = df.drop(columns=col)
            dropped_non_sample_columns.append(str(col))
        else:
            numeric_value_counts[str(col)] = int(numeric.notna().sum())
            numeric_missing_counts[str(col)] = int(numeric.isna().sum())
            df[col] = numeric.fillna(0.0)
    return df, MatrixReadDiagnostics(
        numeric_value_count=pd.Series(numeric_value_counts, dtype="int64"),
        numeric_missing_count=pd.Series(numeric_missing_counts, dtype="int64"),
        dropped_non_sample_columns=tuple(dropped_non_sample_columns),
    )


def read_matrix(path: Path, *, sep: str, gene_id_col: str, drop_cols: tuple[str, ...]) -> pd.DataFrame:
    """Return a numeric source matrix, filling parse failures with ``0.0``.

    Use :func:`read_matrix_with_diagnostics` when a builder needs to audit those
    parse failures separately from literal source zeros.
    """
    matrix, _diagnostics = read_matrix_with_diagnostics(
        path, sep=sep, gene_id_col=gene_id_col, drop_cols=drop_cols,
    )
    return matrix


def _series_for_samples(series: pd.Series, samples: Iterable[str]) -> pd.Series:
    return (
        series.reindex([str(s) for s in samples])
        .fillna(0)
        .astype(int)
    )


def _default_source_scale_class(unit: Unit) -> str:
    if unit == "raw_counts":
        return "count_derived_tpm"
    if unit == "log2(TPM+1)":
        return "log2_tpm_inverse"
    return "linear_rnaseq_tpm"


# Substrings that mark a source-scale class as NOT linear-TPM comparable. The
# RNA-seq / count-derived / log2-inverse classes all resolve to linear TPM after
# ``normalize_to_tpm``; only genuine proxy scales (microarray, rank/percentile
# surrogates) are non-comparable.
_NONLINEAR_SCALE_HINTS = ("proxy", "microarray", "rank", "percentile")


def _scale_class_is_linear(source_scale_class: str) -> bool:
    lowered = source_scale_class.lower()
    return not any(hint in lowered for hint in _NONLINEAR_SCALE_HINTS)


def _resolve_linear_comparable(config: SampleQcConfig) -> bool:
    """Effective ``linear_tpm_comparable``: explicit override, else derived."""
    if config.linear_tpm_comparable is not None:
        return config.linear_tpm_comparable
    return _scale_class_is_linear(config.source_scale_class)


def _effective_sample_qc_config(config: SampleQcConfig, unit: Unit) -> SampleQcConfig:
    if config.source_scale_class:
        return config
    return replace(config, source_scale_class=_default_source_scale_class(unit))


def _top_n_sum(values: pd.DataFrame, n: int) -> pd.Series:
    if values.empty:
        return pd.Series(0.0, index=values.columns, dtype=float)
    return values.apply(lambda col: col.nlargest(min(n, len(col))).sum(), axis=0)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise ``numerator / denominator``, 0.0 where ``denominator <= 0``."""
    return (numerator / denominator.where(denominator > 0)).fillna(0.0)


def _mapping_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["source_id", "Ensembl_Gene_ID", "Symbol", "mapping_method"],
    )


def _diagnostics_for_uniquified_columns(
    diagnostics: MatrixReadDiagnostics,
    origin: dict[str, str],
) -> MatrixReadDiagnostics:
    """Re-key parse diagnostics onto uniquified sample columns.

    ``build_source`` renames duplicated sample columns (e.g. ``P-58`` ×15 →
    ``P-58``, ``P-58.1``, …). The parser diagnostics are keyed by the ORIGINAL
    column names, so without this remap every renamed replicate misses the
    lookup and reports ``source_numeric_*`` / ``source_parse_missing_fraction``
    as 0; here each uniquified name inherits its source column's counts.
    """
    def _remap(series: pd.Series) -> pd.Series:
        return pd.Series(
            {new: int(series.get(src, 0)) for new, src in origin.items()},
            dtype="int64",
        )

    return MatrixReadDiagnostics(
        numeric_value_count=_remap(diagnostics.numeric_value_count),
        numeric_missing_count=_remap(diagnostics.numeric_missing_count),
        dropped_non_sample_columns=diagnostics.dropped_non_sample_columns,
    )


def sample_qc_table(
    raw_values: pd.DataFrame,
    clean_values: pd.DataFrame,
    gene_table: pd.DataFrame,
    *,
    config: SampleQcConfig | None = None,
    read_diagnostics: MatrixReadDiagnostics | None = None,
    missing_after_harmonization: pd.Series | None = None,
) -> pd.DataFrame:
    """Return generic per-sample expression QC metrics and inclusion decisions.

    The rules are intentionally sample-ID agnostic. Sparse samples are excluded
    only when they fail documented numeric thresholds, not because a source
    sample name appears on a curated denylist.
    """
    cfg = config or SampleQcConfig()
    sample_ids = [str(c) for c in raw_values.columns]
    raw_values = raw_values.copy()
    clean_values = clean_values.reindex(
        index=raw_values.index,
        columns=raw_values.columns,
    )
    raw_values.columns = sample_ids
    clean_values.columns = sample_ids

    raw_totals = raw_values.sum(axis=0)
    clean_totals = clean_values.sum(axis=0)
    top_gene_raw = raw_values.max(axis=0)
    top_gene_clean = clean_values.max(axis=0)
    top10_raw = _top_n_sum(raw_values, 10)
    top10_clean = _top_n_sum(clean_values, 10)
    detected_raw = (raw_values > 0).sum(axis=0)
    detected_clean = (clean_values > 0).sum(axis=0)
    n_genes = int(raw_values.shape[0])
    source_zero = (raw_values == 0).sum(axis=0)

    linear_comparable = _resolve_linear_comparable(cfg)

    # Align gene-level masks to the value-matrix rows BY LABEL: a caller whose
    # gene_table is reordered or a subset of the value rows still selects the
    # right genes (positional boolean masking silently mis-selected before).
    gene_ids = gene_table["Ensembl_Gene_ID"].map(strip_version)
    row_ids = pd.Index(raw_values.index).map(strip_version)

    removal_by_id = pd.Series(
        clean_tpm_removal_mask(gene_table).to_numpy(), index=gene_ids.to_numpy()
    )
    removal_mask = removal_by_id.reindex(row_ids, fill_value=False).to_numpy()
    biological_clean = clean_values.loc[~removal_mask]
    detected_clean_biological = (biological_clean > 0).sum(axis=0)

    from ..gene_sets_cancer import housekeeping_gene_ids

    universal_ids = {
        strip_version(x) for x in housekeeping_gene_ids() if str(x).strip()
    }
    universal_by_id = pd.Series(
        gene_ids.isin(universal_ids).to_numpy(), index=gene_ids.to_numpy()
    )
    universal_mask = universal_by_id.reindex(row_ids, fill_value=False).to_numpy()
    n_universal_expected = len(universal_ids)
    n_universal_present = int(universal_mask.sum())
    if n_universal_present:
        universal_values = raw_values.loc[universal_mask]
        universal_detected = (universal_values > 0).sum(axis=0)
        universal_floor = (universal_values >= cfg.universal_floor_tpm).sum(axis=0)
        universal_fraction = universal_detected / n_universal_present
        universal_floor_fraction = universal_floor / n_universal_present
    else:
        universal_detected = pd.Series(0, index=sample_ids, dtype=int)
        universal_floor = pd.Series(0, index=sample_ids, dtype=int)
        universal_fraction = pd.Series(0.0, index=sample_ids)
        universal_floor_fraction = pd.Series(0.0, index=sample_ids)

    numeric_missing = (
        _series_for_samples(read_diagnostics.numeric_missing_count, sample_ids)
        if read_diagnostics is not None
        else pd.Series(0, index=sample_ids, dtype=int)
    )
    numeric_values = (
        _series_for_samples(read_diagnostics.numeric_value_count, sample_ids)
        if read_diagnostics is not None
        else pd.Series(0, index=sample_ids, dtype=int)
    )
    missing_after = (
        _series_for_samples(missing_after_harmonization, sample_ids)
        if missing_after_harmonization is not None
        else pd.Series(0, index=sample_ids, dtype=int)
    )
    parse_total = numeric_values + numeric_missing

    # Every per-sample metric is already a columnar Series indexed by sample_id;
    # assemble them directly rather than re-deriving each via scalar lookups.
    if n_genes:
        detected_raw_fraction = detected_raw / n_genes
        source_zero_fraction = source_zero / n_genes
    else:
        detected_raw_fraction = pd.Series(0.0, index=sample_ids)
        source_zero_fraction = pd.Series(0.0, index=sample_ids)

    # Fail/warn flags as boolean columns; column insertion order fixes the order
    # reasons appear in ``sample_qc_reasons`` (fails first, then warns).
    fail_flags = pd.DataFrame(index=pd.Index(sample_ids))
    warn_flags = pd.DataFrame(index=pd.Index(sample_ids))
    if not cfg.enabled:
        warn_flags["sample_qc_disabled"] = True
    elif not linear_comparable:
        warn_flags["nonlinear_or_proxy_source_scale"] = True
    else:
        fail_flags["detected_genes_below_min"] = detected_raw < cfg.min_detected_genes
        if n_universal_present:
            fail_flags["universal_nonzero_below_min"] = (
                universal_detected < cfg.min_universal_nonzero_genes
            )
            fail_flags["universal_nonzero_fraction_below_min"] = (
                universal_fraction < cfg.min_universal_nonzero_fraction
            )
        if cfg.max_top_gene_fraction is not None:
            fail_flags["top_gene_fraction_above_max"] = (raw_totals > 0) & (
                _safe_ratio(top_gene_raw, raw_totals) > cfg.max_top_gene_fraction
            )
        if cfg.max_top10_fraction is not None:
            fail_flags["top10_fraction_above_max"] = (raw_totals > 0) & (
                _safe_ratio(top10_raw, raw_totals) > cfg.max_top10_fraction
            )
    if cfg.special_source_warning:
        warn_flags[cfg.special_source_warning] = True

    fail_cols = list(fail_flags.columns)
    warn_cols = list(warn_flags.columns)
    statuses: list[str] = []
    reason_strs: list[str] = []
    included: list[bool] = []
    for sample in sample_ids:
        frs = [c for c in fail_cols if bool(fail_flags.at[sample, c])]
        wrs = [c for c in warn_cols if bool(warn_flags.at[sample, c])]
        status = "fail" if frs else ("warn" if wrs else "pass")
        statuses.append(status)
        reason_strs.append(";".join(frs + wrs))
        included.append(status != "fail")

    def _col(series: pd.Series):
        return series.reindex(sample_ids).to_numpy()

    return pd.DataFrame({
        "sample_id": sample_ids,
        "included": included,
        "sample_qc_status": statuses,
        "sample_qc_reasons": reason_strs,
        "n_genes_harmonized": n_genes,
        "n_detected_raw": _col(detected_raw),
        "n_detected_clean": _col(detected_clean),
        "n_detected_clean_biological": _col(detected_clean_biological),
        "detected_raw_fraction": _col(detected_raw_fraction),
        "source_zero_fraction": _col(source_zero_fraction),
        "source_total_tpm_raw": _col(raw_totals),
        "source_total_tpm_clean": _col(clean_totals),
        "top1_tpm_raw": _col(top_gene_raw),
        "top1_fraction_raw": _col(_safe_ratio(top_gene_raw, raw_totals)),
        "top10_tpm_raw": _col(top10_raw),
        "top10_fraction_raw": _col(_safe_ratio(top10_raw, raw_totals)),
        "top1_tpm_clean": _col(top_gene_clean),
        "top1_fraction_clean": _col(_safe_ratio(top_gene_clean, clean_totals)),
        "top10_tpm_clean": _col(top10_clean),
        "top10_fraction_clean": _col(_safe_ratio(top10_clean, clean_totals)),
        "n_universal_genes_expected": n_universal_expected,
        "n_universal_genes_present": n_universal_present,
        "n_universal_genes_missing": n_universal_expected - n_universal_present,
        "n_universal_nonzero_genes": _col(universal_detected),
        "universal_nonzero_fraction": _col(universal_fraction),
        "universal_floor_tpm": cfg.universal_floor_tpm,
        "n_universal_ge_floor_genes": _col(universal_floor),
        "universal_ge_floor_fraction": _col(universal_floor_fraction),
        "source_scale_class": cfg.source_scale_class,
        "linear_tpm_comparable": linear_comparable,
        "source_numeric_values": _col(numeric_values),
        "source_numeric_missing_values": _col(numeric_missing),
        "source_parse_missing_fraction": _col(_safe_ratio(numeric_missing, parse_total)),
        "missing_after_harmonization": _col(missing_after),
    })


_HIGH_EXPRESSION_UNRESOLVED_MAX_TPM = 1.0


def mapping_audit_table(
    matrix: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    detected_gene_id_type: GeneIdType,
) -> pd.DataFrame:
    """Return a per-source-row mapping audit for a normalized source matrix."""
    sample_cols = [str(c) for c in matrix.columns]
    row_numbers = np.arange(len(matrix.index), dtype=int)
    source_ids = pd.Series(matrix.index.astype(str), name="source_id")
    if sample_cols:
        expression_total = matrix.sum(axis=1).to_numpy(dtype=float)
        expression_max = matrix.max(axis=1).to_numpy(dtype=float)
        sample_with_max = matrix.idxmax(axis=1).astype(str).to_numpy()
        nonzero_samples = (matrix > 0).sum(axis=1).to_numpy(dtype=int)
    else:
        expression_total = np.zeros(len(matrix.index), dtype=float)
        expression_max = np.zeros(len(matrix.index), dtype=float)
        sample_with_max = np.array([""] * len(matrix.index), dtype=object)
        nonzero_samples = np.zeros(len(matrix.index), dtype=int)
    audit = pd.DataFrame({
        "source_row_number": row_numbers,
        "source_id": source_ids,
        "detected_gene_id_type": detected_gene_id_type,
        "source_expression_total": expression_total,
        "source_expression_max": expression_max,
        "source_expression_sample_with_max": sample_with_max,
        "source_expression_nonzero_samples": nonzero_samples,
    })
    if mapping.empty:
        grouped = pd.DataFrame(
            columns=[
                "source_id",
                "Ensembl_Gene_ID",
                "Symbol",
                "mapping_method",
                "mapping_candidate_count",
                "n_canonical_ids",
            ]
        )
    else:
        mapping = mapping.copy()
        if "mapping_method" not in mapping.columns:
            mapping["mapping_method"] = ""
        grouped = (
            mapping.assign(source_id=mapping["source_id"].astype(str))
            .groupby("source_id", sort=False)
            .agg(
                Ensembl_Gene_ID=(
                    "Ensembl_Gene_ID",
                    lambda s: ";".join(sorted({str(x) for x in s if str(x)})),
                ),
                Symbol=(
                    "Symbol",
                    lambda s: ";".join(sorted({str(x) for x in s if str(x)})),
                ),
                mapping_method=(
                    "mapping_method",
                    lambda s: ";".join(sorted({str(x) for x in s if str(x)})),
                ),
                mapping_candidate_count=("Ensembl_Gene_ID", "size"),
                n_canonical_ids=(
                    "Ensembl_Gene_ID",
                    lambda s: len({str(x) for x in s if str(x)}),
                ),
            )
            .reset_index()
        )
    audit = audit.merge(grouped, on="source_id", how="left")
    audit["mapping_candidate_count"] = (
        audit["mapping_candidate_count"].fillna(0).astype(int)
    )
    audit["n_canonical_ids"] = audit["n_canonical_ids"].fillna(0).astype(int)
    audit["mapping_status"] = np.select(
        [
            audit["n_canonical_ids"].eq(0),
            audit["n_canonical_ids"].eq(1),
        ],
        ["unresolved", "resolved"],
        default="ambiguous",
    )
    audit["high_expression_unresolved"] = (
        audit["mapping_status"].eq("unresolved")
        & audit["source_expression_max"].ge(_HIGH_EXPRESSION_UNRESOLVED_MAX_TPM)
    )
    ordered = [
        "source_row_number",
        "source_id",
        "detected_gene_id_type",
        "mapping_status",
        "mapping_method",
        "Ensembl_Gene_ID",
        "Symbol",
        "mapping_candidate_count",
        "n_canonical_ids",
        "source_expression_total",
        "source_expression_max",
        "source_expression_sample_with_max",
        "source_expression_nonzero_samples",
        "high_expression_unresolved",
    ]
    return audit[ordered]


def mapping_audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize a mapping audit table in the fields tracked by issue #515."""
    return pd.DataFrame([{
        "detected_gene_id_type": (
            audit["detected_gene_id_type"].iloc[0] if not audit.empty else ""
        ),
        "source_row_count": int(len(audit)),
        "resolved_row_count": int(audit["mapping_status"].eq("resolved").sum()),
        "unresolved_row_count": int(audit["mapping_status"].eq("unresolved").sum()),
        "ambiguous_row_count": int(audit["mapping_status"].eq("ambiguous").sum()),
        "high_expression_unresolved_row_count": int(
            audit["high_expression_unresolved"].sum()
        ),
        "high_expression_unresolved_max_tpm_threshold": (
            _HIGH_EXPRESSION_UNRESOLVED_MAX_TPM
        ),
    }])


def _artifact_stem(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _write_mapping_audit(
    cache_dir: Path,
    source_cohort: str,
    audit: pd.DataFrame,
) -> tuple[Path, Path]:
    derived = cache_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem(source_cohort)
    audit_path = derived / f"{stem}_mapping_audit.csv"
    summary_path = derived / f"{stem}_mapping_audit_summary.csv"
    audit.to_csv(audit_path, index=False)
    mapping_audit_summary(audit).to_csv(summary_path, index=False)
    return audit_path, summary_path


def _write_sample_qc(cache_dir: Path, code: str, qc: pd.DataFrame) -> Path:
    derived = cache_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / f"{code}_sample_qc.csv"
    qc.to_csv(path, index=False)
    return path


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
            "mapping_method": "ensembl_id",
        })
    mapping = _mapping_frame(rows)
    if mapping.empty:
        return mapping, pd.DataFrame(columns=matrix.columns)
    return mapping, aggregate_matrix_by_mapping(
        matrix, mapping[["source_id", "Ensembl_Gene_ID", "Symbol"]],
    )


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
            "mapping_method": _method,
        })
    mapping = _mapping_frame(rows)
    if mapping.empty:
        return mapping, pd.DataFrame(columns=matrix.columns)
    return mapping, aggregate_matrix_by_mapping(
        matrix, mapping[["source_id", "Ensembl_Gene_ID", "Symbol"]],
    )


def _harmonize_by_entrez_id(
    matrix: pd.DataFrame, ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve Entrez IDs through the shared NCBI chain, keeping method labels."""
    genome = EnsemblRelease(ensembl_release)
    rows = []
    for raw in matrix.index:
        entrez_id = str(raw).strip()
        if not entrez_id.isdigit():
            continue
        result = entrez_to_gene(genome, entrez_id)
        if result is None:
            continue
        ensembl_id, name, method = result
        rows.append({
            "source_id": entrez_id,
            "Ensembl_Gene_ID": ensembl_id,
            "Symbol": name,
            "mapping_method": method,
        })
    mapping = _mapping_frame(rows)
    if mapping.empty:
        return mapping, pd.DataFrame(columns=matrix.columns)
    return mapping, aggregate_matrix_by_mapping(
        matrix, mapping[["source_id", "Ensembl_Gene_ID", "Symbol"]],
    )


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
        return _harmonize_by_entrez_id(matrix, ensembl_release)
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
    matrix, read_diagnostics = read_matrix_with_diagnostics(
        file_path,
        sep=source.sep,
        gene_id_col=source.gene_id_col,
        drop_cols=source.drop_cols,
    )
    matrix.columns = matrix.columns.astype(str)
    if source.transposed:
        # File is samples-as-rows × genes-as-cols; transpose so the
        # rest of the pipeline (genes-as-rows, samples-as-cols) works.
        dropped_non_sample_columns = read_diagnostics.dropped_non_sample_columns
        matrix = matrix.T
        matrix.index.name = source.gene_id_col or "gene_id"
        matrix.columns = matrix.columns.astype(str)
        # The parser diagnostics were column-oriented before transpose (genes,
        # not samples), so do not report them as per-sample parse missingness.
        read_diagnostics = MatrixReadDiagnostics(
            numeric_value_count=pd.Series(dtype="int64"),
            numeric_missing_count=pd.Series(dtype="int64"),
            dropped_non_sample_columns=dropped_non_sample_columns,
        )
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
    audit = mapping_audit_table(
        tpm, mapping, detected_gene_id_type=gene_id_type,
    )
    audit_path, audit_summary_path = _write_mapping_audit(
        cache_dir, source.source_cohort, audit,
    )
    print(
        f"  mapping audit: {audit_path} "
        f"(summary: {audit_summary_path})"
    )

    # Build per-gene-per-cohort rows. The cancer_code may be a list when
    # the same matrix splits across multiple codes (e.g. NET_LUNG vs
    # NEC_LUNG_LARGECELL) via sample_to_cancer_code.
    gene_table = (
        mapping.drop_duplicates("Ensembl_Gene_ID")[["Ensembl_Gene_ID", "Symbol"]]
        .reset_index(drop=True)
    )
    values = values.reindex(gene_table["Ensembl_Gene_ID"])
    missing_after_harmonization = values.isna().sum(axis=0)
    values = values.fillna(0.0)

    # Some source matrices repeat a sample id across columns (e.g. GSE294016
    # ADCC has P-58/P-77 ×15). Uniquify so each is a distinct per-sample column
    # — required for the per-sample parquet (and harmless for the summary).
    if values.columns.duplicated().any():
        seen: dict[str, int] = {}
        uniq = []
        origin: dict[str, str] = {}
        for c in values.columns:
            if c in seen:
                seen[c] += 1
                new = f"{c}.{seen[c]}"
            else:
                seen[c] = 0
                new = c
            uniq.append(new)
            origin[new] = c
        values.columns = uniq
        missing_after_harmonization.index = uniq
        # Carry parse diagnostics onto the renamed replicates so each reports its
        # source column's parse counts instead of a misleading 0.
        read_diagnostics = _diagnostics_for_uniquified_columns(
            read_diagnostics, origin,
        )

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
    sample_qc_config = _effective_sample_qc_config(source.sample_qc, source.unit)
    for code, cols in cohort_to_cols.items():
        if not cols:
            continue
        sub_values = values[cols]
        missing_after_sub = missing_after_harmonization.reindex(cols).fillna(0).astype(int)
        clean = _clean_tpm(sub_values, gene_table=gene_table)
        qc = sample_qc_table(
            sub_values,
            clean,
            gene_table,
            config=sample_qc_config,
            read_diagnostics=read_diagnostics,
            missing_after_harmonization=missing_after_sub,
        )
        qc_path = _write_sample_qc(cache_dir, code, qc)
        if sample_qc_config.enabled:
            keep = qc.loc[qc["included"], "sample_id"].tolist()
            excluded = qc.loc[~qc["included"], ["sample_id", "sample_qc_reasons"]]
            if not excluded.empty:
                print(
                    f"    {code}: excluding {len(excluded)} sparse/QC-failed samples; "
                    f"QC details: {qc_path}"
                )
            sub_values = sub_values[keep]
            clean = clean[keep]
            if sub_values.shape[1] == 0:
                print(f"    {code}: all samples failed generic expression QC")
                continue
        # Persist the per-code per-sample matrix for medoids + percentiles
        # (uniform with every other per-sample cohort; the read path discovers
        # it by code==stem under this source's cache). One hook covers every
        # geo_matrix source.
        from ..cohorts import write_per_sample as _write_per_sample
        _write_per_sample(gene_table, sub_values, cache_dir.name, code)
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
            f"_to_tpm_ensembl{ensembl_release}_clean_tpm_16_9_75"
        )
        out["notes"] = source.notes or (
            f"Per-sample expression from {source.source_cohort} "
            f"(n={sub_values.shape[1]}). "
            f"Unit-normalized to TPM; tech-RNA-zeroed; v5.3 stats."
        )
        out["metastasis_site"] = source.metastasis_site if source.metastasis_site else pd.NA
        out = finalize_reference_rows(out, tumor_origin=source.tumor_origin)
        summaries.append(out)
        counts_by_code[code] = sub_values.shape[1]
        print(f"    {code}: n={sub_values.shape[1]} → {len(out)} gene rows")

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
    "MatrixReadDiagnostics",
    "SampleQcConfig",
    "Unit",
    "detect_gene_id_type",
    "read_matrix",
    "read_matrix_with_diagnostics",
    "sample_qc_table",
    "mapping_audit_table",
    "mapping_audit_summary",
    "normalize_to_tpm",
    "harmonize_gene_ids",
    "build_source",
    # Re-exported shared clean-TPM helpers (builders import them from here).
    "_clean_tpm",
    "_technical_mask",
]
