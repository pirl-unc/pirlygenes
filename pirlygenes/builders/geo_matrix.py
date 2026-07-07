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
    finalize_reference_rows,
    upsert_to_shard,
)
from . import oncoref_source as _osrc
from .oncoref_source import SampleQcMode
from .gene_mapping import (
    GeneIdType,
    detect_id_type,
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
    # Source-scale class for the oncoref sample-QC contract. Empty → derived
    # from ``unit`` (RNA-seq TPM). A non-linear / proxy source (microarray, rank,
    # percentile) sets its own class + ``linear_tpm_comparable=False`` so the
    # RNA-seq fail gates are skipped (warn-only). ``sample_qc_mode`` picks which
    # samples feed the summary stats + per-sample parquet; the full QC manifest
    # is persisted regardless so a consumer can re-filter at read time.
    source_scale_class: str = ""
    linear_tpm_comparable: bool | None = None
    sample_qc_mode: SampleQcMode = "pass_or_warn"


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


# ─── Gene-length lookup (raw counts → TPM) ──────────────────────────────────
# Gene-id *harmonization* is delegated to oncoref (see oncoref_source); this
# module keeps only the pyensembl gene-length lookup that unit normalization
# needs, since oncoref does not own unit conversion.

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


# ─── Raw read (source strings, for oncoref parse diagnostics) ───────────────

def _read_raw_matrix(
    path: Path, *, sep: str, gene_id_col: str, drop_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Like :func:`read_matrix` but keep sample values as RAW strings.

    oncoref's ``coerce_source_expression_values`` measures missing-vs-parse-fail-
    vs-literal-zero on the source strings, so the numeric coercion is deferred to
    it rather than done here. Gene-annotation columns (entirely non-numeric) are
    still dropped, and empty gene-id rows removed, exactly as ``read_matrix``.
    """
    engine = "python" if (len(sep) > 1 or "\\" in sep) else "c"
    read_kwargs = {"sep": sep, "engine": engine, "dtype": str}
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
    for col in list(df.columns):
        # Drop a genuine text-annotation column (has non-null values, none of them
        # numeric). An entirely-blank/all-NaN column is NOT annotation — keep it so
        # it reaches oncoref's parse-diagnostics + per-sample QC (which cover every
        # sample) instead of vanishing silently before QC ever sees it.
        non_null = int(df[col].notna().sum())
        numeric_ok = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        if non_null > 0 and numeric_ok == 0:
            df = df.drop(columns=col)
    return df


def _artifact_stem(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _write_derived_csv(cache_dir: Path, stem: str, frame: pd.DataFrame) -> Path:
    derived = cache_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / f"{_artifact_stem(stem)}.csv"
    frame.to_csv(path, index=False)
    return path


# ─── End-to-end build ───────────────────────────────────────────────────────

def build_source(
    source: GeoMatrixSource,
    *,
    cache_dir: Path,
    summary_output: Path,
    ensembl_release: int = 112,
) -> dict[str, int]:
    """Download, parse, normalize, canonicalize (oncoref), QC, summarize, upsert.

    Gene mapping (Ensembl / HUGO / Entrez / transcript / synonym → canonical
    ENSG), parse diagnostics, the per-source-row mapping audit, and the
    per-sample QC manifest are all delegated to oncoref via
    :mod:`pirlygenes.builders.oncoref_source`. pirlygenes keeps unit → TPM
    normalization, clean-TPM stats, the per-sample parquet, and the shard upsert.
    """
    import oncoref

    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / source.file_name
    print(f"downloading {source.file_name}...")
    _download(source.file_url, file_path)

    print(f"reading matrix (unit={source.unit}, sep={source.sep!r})...")
    raw = _read_raw_matrix(
        file_path,
        sep=source.sep,
        gene_id_col=source.gene_id_col,
        drop_cols=source.drop_cols,
    )
    if source.transposed:
        # File is samples-as-rows × genes-as-cols; transpose so the
        # rest of the pipeline (genes-as-rows, samples-as-cols) works.
        raw = raw.T
        raw.index.name = source.gene_id_col or "gene_id"
        print(f"  transposed: now shape {raw.shape} (genes × samples)")
    if source.sample_filter is not None:
        keep = source.sample_filter(list(raw.columns))
        raw = raw[keep]
    print(f"  shape: {raw.shape} (genes × samples)")
    if raw.shape[1] < 1:
        raise RuntimeError("matrix has 0 samples after filter")

    # Numeric view for unit normalization: missing / parse-fail cells → 0 for the
    # per-sample TPM sum. The missing-vs-zero distinction is preserved separately
    # by oncoref's parse diagnostics (computed from ``raw``), so no source
    # missingness is silently collapsed into a measured zero before QC.
    numeric = raw.apply(lambda c: pd.to_numeric(c, errors="coerce")).fillna(0.0)

    gene_id_type = source.gene_id_type
    if gene_id_type == "auto":
        gene_id_type = detect_gene_id_type(numeric.index)
    print(f"  detected gene_id_type: {gene_id_type}")

    # Unit normalization → TPM (pirlygenes owns unit conversion; oncoref does not).
    if source.unit == "raw_counts":
        print("  computing gene lengths (kb) for length-normalization...")
        lengths_kb = _gene_lengths_kb_for_index(
            numeric.index, gene_id_type=gene_id_type, ensembl_release=ensembl_release,
        )
        print(f"  got lengths for {len(lengths_kb)} / {len(numeric.index)} rows")
        tpm = normalize_to_tpm(numeric, unit=source.unit, gene_lengths_kb=lengths_kb)
    else:
        tpm = normalize_to_tpm(numeric, unit=source.unit)
    print(f"  per-sample sums (after TPM normalize, first 3): "
          f"{tpm.sum(axis=0).head(3).to_dict()}")

    # Canonicalize gene ids (release-independent, incl. Entrez) via oncoref. For a
    # symbol-keyed source, pass the ids as HUGO symbol candidates so oncoref rescues
    # tokens its row-type sniff would otherwise skip (leading-digit ncRNA symbols
    # like 7SL/45S), matching the pre-delegation unconditional symbol resolver.
    print("  canonicalizing gene ids via oncoref...")
    symbols = list(tpm.index) if gene_id_type == "hugo" else None
    canon = _osrc.canonicalize_source(
        tpm, row_id_name=source.gene_id_col or "gene_id", raw_matrix=raw,
        symbols=symbols,
    )
    gene_table = canon.gene_table
    values = canon.values
    stats = canon.mapping_stats
    print(f"  resolved {stats['n_resolved_rows']}/{stats['n_source_rows']} rows → "
          f"{len(gene_table)} canonical genes "
          f"({stats['n_ambiguous_rows']} ambiguous, "
          f"{stats['n_high_expression_unresolved_rows']} high-expression unresolved)")
    stem = _artifact_stem(source.source_cohort)
    audit_path = _write_derived_csv(cache_dir, f"{stem}_mapping_audit", canon.audit)
    _write_derived_csv(cache_dir, f"{stem}_parse_diagnostics", canon.parse_diagnostics)
    print(f"  mapping audit + parse diagnostics: {audit_path.parent}")

    metadata = _osrc.source_metadata(
        unit=source.unit,
        source_scale_class=source.source_scale_class,
        linear_tpm_comparable=source.linear_tpm_comparable,
        source_cohort=source.source_cohort,
        source_type=source.source_project,
    )

    # Split by cancer_code. The cancer_code may be a list when the same matrix
    # splits across multiple codes (e.g. NET_LUNG vs NEC_LUNG_LARGECELL).
    if source.sample_to_cancer_code is not None:
        cohort_to_cols: dict[str, list[str]] = {}
        for col in canon.sample_cols:
            code = source.sample_to_cancer_code(col)
            if code is None:
                continue
            cohort_to_cols.setdefault(code, []).append(col)
    else:
        if isinstance(source.cancer_code, list):
            raise RuntimeError(
                "cancer_code is a list but no sample_to_cancer_code provided"
            )
        cohort_to_cols = {source.cancer_code: list(canon.sample_cols)}

    # Canonicalize cohort codes once so every downstream artifact — the QC
    # manifest CSV, the per-sample parquet, and the shard rows — shares one stem.
    # write_per_sample / remove_per_sample canonicalize internally; without this
    # the QC-CSV stem (written from the raw code) would diverge from the parquet
    # when a sample_to_cancer_code rule emits a pre-rename alias. Merge collisions.
    from ..gene_sets_cancer import canonical_cancer_code as _canonical_code
    _canonicalized: dict[str, list[str]] = {}
    for _code, _cols in cohort_to_cols.items():
        _canonicalized.setdefault(_canonical_code(_code), []).extend(_cols)
    cohort_to_cols = _canonicalized

    summaries: list[pd.DataFrame] = []
    counts_by_code: dict[str, int] = {}
    # Every code we evaluated (had ≥1 column), incl. any whose samples all fail
    # QC. Passed to upsert_to_shard so a fully-failed code's stale shard rows are
    # REMOVED on rebuild rather than left behind under the preserve-others rule.
    processed_codes: list[str] = []
    for code, cols in cohort_to_cols.items():
        if not cols:
            continue
        processed_codes.append(code)
        sub_matrix = canon.matrix[["Ensembl_Gene_ID", "Symbol", *cols]]
        # Per-sample QC (oncoref). The manifest covers EVERY sample; ``kept`` is
        # the subset that feeds the summary stats + parquet, per sample_qc_mode.
        qc, kept = _osrc.sample_qc(
            sub_matrix, cols, metadata=metadata, cancer_type=code,
            mode=source.sample_qc_mode,
        )
        qc_path = _write_derived_csv(cache_dir, f"{code}_sample_qc", qc)
        n_excluded = len(cols) - len(kept)
        if n_excluded:
            print(f"    {code}: excluding {n_excluded} QC-filtered sample(s) "
                  f"(mode={source.sample_qc_mode}); manifest: {qc_path}")
        if not kept:
            print(f"    {code}: all samples failed generic expression QC")
            # Drop any stale per-sample parquet from a previous build so the
            # read path can't serve a code that now contributes no samples.
            from ..cohorts import remove_per_sample as _remove_per_sample
            _remove_per_sample(cache_dir.name, code)
            continue
        sub_values = values[kept]
        # Persist the per-code per-sample matrix for medoids + percentiles.
        from ..cohorts import write_per_sample as _write_per_sample
        _write_per_sample(gene_table, sub_values, cache_dir.name, code)
        clean = _clean_tpm(sub_values, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = source.source_cohort
        out["source_project"] = source.source_project
        # Record any build-time QC exclusion in the durable shard provenance — the
        # per-sample manifest lives only in the cache, so without this the shard's
        # shifted n has no on-record explanation.
        qc_note = (f"; {n_excluded} sample(s) QC-excluded (mode={source.sample_qc_mode})"
                   if n_excluded else "")
        out["source_version"] = (
            f"{source.citation}; gene-id-type={gene_id_type}; "
            f"unit={source.unit}; oncoref-canonicalized (oncoref {oncoref.__version__})"
            f"{qc_note}"
        )
        assign_stats(out, sub_values, clean)
        pipeline_stem = source.pipeline_stem or source.source_cohort.lower()
        unit_slug = source.unit.lower().replace('(', '').replace(')', '').replace('+', 'plus')
        out["processing_pipeline"] = (
            f"{pipeline_stem}_{unit_slug}_to_tpm_oncoref_canonical_clean_tpm_16_9_75"
        )
        out["notes"] = source.notes or (
            f"Per-sample expression from {source.source_cohort} (n={len(kept)}). "
            f"Unit-normalized to TPM; oncoref-canonicalized gene ids; "
            f"tech-RNA-zeroed; v5.3 stats."
        )
        out["metastasis_site"] = source.metastasis_site if source.metastasis_site else pd.NA
        out = finalize_reference_rows(out, tumor_origin=source.tumor_origin)
        summaries.append(out)
        counts_by_code[code] = len(kept)
        print(f"    {code}: n={len(kept)} → {len(out)} gene rows")

    if not processed_codes:
        # Nothing routed to any cancer code — an empty / mis-filtered matrix (a
        # config error), distinct from "samples routed but every one failed QC".
        raise RuntimeError("no cohort had samples after filtering")
    combined = (pd.concat(summaries, ignore_index=True) if summaries
                else pd.DataFrame(columns=list(REFERENCE_COLUMNS)))
    # Always upsert — even when every sample failed QC and `combined` is empty — so
    # a fully-failed code's stale shard rows are purged in lockstep with its removed
    # per-sample parquet. Otherwise the shard keeps advertising a cohort the read
    # path (medoids/percentiles) can no longer serve: an inconsistent half-state.
    upsert_to_shard(
        summary_output, combined,
        source_cohort=source.source_cohort,
        cancer_codes=processed_codes,
    )
    if summaries:
        print(f"  upserted {len(combined)} rows into shard "
              f"{source.source_cohort}.csv.gz")
    else:
        print(f"  WARNING: every sample in all {len(processed_codes)} cohort(s) "
              f"{processed_codes} failed QC (mode={source.sample_qc_mode}); purged "
              f"their stale shard rows, wrote no summary rows")
    return counts_by_code


__all__ = [
    "GeoMatrixSource",
    "GeneIdType",
    "Unit",
    "detect_gene_id_type",
    "read_matrix",
    "normalize_to_tpm",
    "build_source",
    # Re-exported shared clean-TPM helpers (builders import them from here).
    "_clean_tpm",
    "_technical_mask",
]
