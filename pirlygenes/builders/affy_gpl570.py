"""Generic GEO microarray → TPM-proxy builder.

Originally written for Affymetrix GPL570 (HG-U133 Plus 2.0); v5.5.0
generalized to any GEO microarray platform whose ``acc.cgi`` text
view exposes a probe→gene-symbol table (Affymetrix, Agilent, Illumina
BeadChip, etc.). The pipeline is unchanged across platforms:

  1. Parse the series_matrix.txt.gz → per-sample probe-intensity table
  2. Sniff log2 vs linear (values mostly < 50 → log2; else linear)
  3. Probe → gene rollup via *max* (most-expressed probe wins)
  4. Anti-log2 (if needed) → per-sample sum-to-1e6 = TPM PROXY
  5. HUGO → Ensembl harmonization

**Caveats baked into the output:**
- This is *not* directly comparable to RNA-seq TPM in absolute
  magnitude — microarray dynamic range, probe-design saturation, and
  whole-transcriptome coverage all differ.
- Within-sample gene *rank* is preserved (useful for "which genes are
  expressed at all in this sample?" questions) but absolute TPM
  values should not be cross-compared to RNA-seq cohorts.
- Sniffing log2 vs linear is heuristic; if you know the input is
  MAS5 (Affy) / linear-scale Agilent / etc., this builder will leave
  it linear and the sum-to-1e6 result is still a valid TPM proxy.

The caveat is encoded in two places: the ``notes`` column carries a
human-readable warning, and ``processing_pipeline`` is tagged
``<platform_lower>_microarray_tpm_proxy_...`` so programmatic consumers
can detect-and-filter.

Symbol-column detection covers the common platform conventions:
``Gene symbol`` (Affymetrix), ``GeneSymbol`` (Agilent), ``Symbol``,
``GENE_SYMBOL``, ``ILMN_Gene`` (Illumina).

Tested platforms:
- GPL570  — Affymetrix HG-U133 Plus 2.0
- GPL22303 — Agilent SurePrint G3 Human GE v3
- GPL6480  — Agilent Human GE 4x44K
"""

from __future__ import annotations

import gzip
import re
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .geo_matrix import _clean_tpm
from .treehouse import _build_or_load_symbol_mapping
from ..expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)


def annot_url_for(platform_id: str) -> str:
    """Build the GEO ``acc.cgi`` text-view URL for a platform table."""
    return (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?targ=self&form=text&view=full&acc={platform_id}"
    )


def annot_local_filename(platform_id: str) -> str:
    return f"{platform_id}.platform_table.txt"


# Back-compat aliases for the GPL570-only entry points used pre-v5.5.0
GPL570_ANNOT_URL = annot_url_for("GPL570")
GPL570_ANNOT_LOCAL = annot_local_filename("GPL570")


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


def parse_geo_platform_table(annot_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns ``probe_id``, ``gene_symbol`` from
    a GEO platform-table text dump.

    Covers the common platform conventions:
      - probe column:  ``ID`` / ``ID_REF`` / ``Probe Set ID`` /
                       ``ProbeName`` (Agilent) / ``ProbeID``
      - symbol column: ``Gene symbol`` / ``Gene Symbol`` (Affymetrix) /
                       ``GeneSymbol`` (Agilent) / ``Symbol`` /
                       ``GENE_SYMBOL`` / ``ILMN_Gene`` (Illumina)

    Multi-symbol cells (e.g. Affymetrix's ``DDR1 /// MIR4640``) are
    split on ``///`` and the first symbol is kept (most-expressed
    paralog wins after the per-gene max rollup downstream).
    """
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
        (c for c in (
            "ID", "ID_REF", "Probe Set ID", "ProbeName", "ProbeID",
        ) if c in df.columns),
        None,
    )
    symbol_col = next(
        (c for c in (
            "Gene symbol", "Gene Symbol", "GeneSymbol",
            "Symbol", "GENE_SYMBOL", "ILMN_Gene",
        ) if c in df.columns),
        None,
    )
    if probe_col is None:
        raise RuntimeError(
            f"GEO platform table missing probe id column: "
            f"{list(df.columns)[:15]}"
        )

    # Some Agilent "SystematicName Version" platforms (e.g. GPL22303)
    # ship NO separate symbol column — the ``ID`` column itself holds
    # the SystematicName, which is HUGO for known genes and a GenBank
    # accession or systematic name for unannotated probes. Use the
    # probe id AS the symbol; downstream HUGO→ENSG harmonization will
    # filter out non-HUGO entries naturally (pyensembl returns None
    # for accession IDs).
    # Entrez ID column is optional but very useful: when a HUGO symbol
    # has been renamed by HGNC since the platform was annotated (e.g.
    # SEPT2 → SEPTIN2, NARS → NARS1, H3F3A → H3-3A, TCEB2 → ELOB),
    # pyensembl's name lookup fails on the legacy symbol but the Entrez
    # ID is stable. Carrying it through lets the builder fall back to
    # Entrez → current HUGO → ENSG when the direct lookup misses.
    entrez_col = next(
        (c for c in (
            "ENTREZ_GENE_ID", "Entrez Gene ID", "Entrez_Gene_ID",
            "EntrezGeneID", "ENTREZID",
            # Agilent convention: plain "GENE" holds the Entrez ID
            # (not the symbol — that's "GENE_SYMBOL"). Verified on
            # GPL6480 (Human 4x44K v1), GPL13497 (v2).
            "GENE",
        ) if c in df.columns),
        None,
    )

    if symbol_col is None:
        out = df[[probe_col]].rename(columns={probe_col: "probe_id"})
        out["gene_symbol"] = out["probe_id"]
    else:
        out = df[[probe_col, symbol_col]].rename(
            columns={probe_col: "probe_id", symbol_col: "gene_symbol"}
        )
    if entrez_col is not None:
        out["entrez_id"] = df[entrez_col].values
    else:
        out["entrez_id"] = ""

    # Multi-gene probes (annotated as ``GeneA /// GeneB`` in Affymetrix
    # convention) target sequence shared between paralogs and genuinely
    # measure both — so we attribute the intensity to BOTH genes via
    # explode rather than dropping one. Before this fix (pre-5.6.1)
    # we kept only the first symbol, silently losing ~970 paralog
    # symbols per platform — including CTAG1B (NY-ESO-1B), MAGEA2B,
    # and several GAGE/PAGE/XAGE paralogs that share probes with their
    # canonical-name siblings.
    #
    # The gene_symbol and entrez_id columns must explode in parallel
    # so each (probe, symbol, entrez) triple stays paired. We zip
    # them per-row before exploding.
    def _zip_lists(row):
        syms = [s.strip() for s in str(row["gene_symbol"]).split("///")]
        eids = [e.strip() for e in str(row["entrez_id"]).split("///")]
        # When Entrez list is shorter than symbol list (or empty), pad
        # with empty strings — better to lose the Entrez fallback for
        # the trailing paralogs than crash on misalignment.
        if len(eids) < len(syms):
            eids = eids + [""] * (len(syms) - len(eids))
        return list(zip(syms, eids[: len(syms)]))

    out["pairs"] = out.apply(_zip_lists, axis=1)
    out = out.explode("pairs")
    out[["gene_symbol", "entrez_id"]] = pd.DataFrame(
        out["pairs"].tolist(), index=out.index
    )
    out = out.drop(columns="pairs")
    out = out[
        out["gene_symbol"].ne("")
        & out["gene_symbol"].ne("---")
        & out["probe_id"].ne("")
    ].copy()
    return out


# Back-compat alias for the GPL570-only name from the original builder.
_parse_gpl570_annot = parse_geo_platform_table


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


def build_microarray_source(
    *,
    series_matrix_url: str,
    series_matrix_filename: str,
    cache_dir: Path,
    cancer_code: str,
    source_cohort: str,
    source_project: str,
    citation: str,
    summary_output: Path,
    platform_id: str = "GPL570",
    platform_name: str | None = None,
    ensembl_release: int = 112,
    sample_include_regex: str | None = None,
    sample_exclude_regex: str | None = None,
    extra_notes: str = "",
    tumor_origin: str = "primary",
    metastasis_site: str | None = None,
) -> int:
    """Build a single-cohort microarray TPM-proxy shard from a GEO
    ``series_matrix.txt.gz`` for any platform whose ``acc.cgi`` text
    view exposes a probe→gene-symbol table.

    ``platform_id`` (e.g. ``"GPL570"``, ``"GPL22303"``, ``"GPL6480"``)
    drives the annot URL and the on-disk cache key. ``platform_name``
    is a human-readable label that lands in the ``notes`` column —
    defaults to ``platform_id`` if not given.

    For multi-cohort series (e.g. GSE85383 = LG-ESS + HG-ESS + UUS +
    LMS in one platform table), call this once per cancer_code with
    a different ``sample_include_regex`` that matches the relevant
    rows' Sample_characteristics_ch1 field (e.g. ``"(?i)low.?grade"``).
    """
    platform_name = platform_name or platform_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    series_path = cache_dir / series_matrix_filename
    annot_path = cache_dir / annot_local_filename(platform_id)

    print(f"downloading {series_matrix_filename}...")
    _download(series_matrix_url, series_path)
    print(f"downloading {platform_id} annotation (via GEO acc.cgi)...")
    _download(annot_url_for(platform_id), annot_path)

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

    print(f"parsing {platform_id} probe→gene annotation...")
    annot = parse_geo_platform_table(annot_path)
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

    # Build a per-symbol Entrez ID lookup so symbols that fail the
    # pyensembl name-lookup (typically HGNC-renamed since the platform
    # was annotated) can be re-resolved via Entrez → current HUGO → ENSG.
    # Worth ~1,000 extra genes per pre-2010 platform (SEPT2→SEPTIN2,
    # NARS→NARS1, H3F3A→H3-3A, TCEB2→ELOB, GNB2L1→RACK1, PHB→PHB1, ...).
    symbol_to_entrez = (
        annot[annot["entrez_id"].astype(str).ne("")]
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")["entrez_id"]
        .astype(str)
        .to_dict()
    )

    # One pass through the shared resolver: direct pyensembl lookup →
    # Entrez chain (dbXrefs → current-symbol → gene_history, using the
    # probe's Entrez ID) → synonym/curated-display rescue. Identical to
    # the path every other builder uses; the Entrez map is the only
    # microarray-specific input. The "_rescued" cache suffix invalidates
    # the older direct-only parquet.
    print(f"harmonizing HUGO → Ensembl release {ensembl_release}...")
    mapping = _build_or_load_symbol_mapping(
        tpm_proxy.index,
        ensembl_release=ensembl_release,
        cache_path=cache_dir / f"symbol_to_ensembl_{ensembl_release}_rescued.parquet",
        refresh=False,
        symbol_to_entrez=symbol_to_entrez,
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

    clean = _clean_tpm(by_ensg, gene_table=gene_table)
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = cancer_code
    out["source_cohort"] = source_cohort
    out["source_project"] = source_project
    out["source_version"] = (
        f"{platform_id} microarray; series_matrix log2 intensity → "
        f"probe-max → anti-log2 → per-sample sum-to-1e6 (TPM proxy); "
        f"HUGO harmonized to Ensembl release {ensembl_release}"
    )
    assign_stats(out, by_ensg, clean)
    out["processing_pipeline"] = (
        f"{platform_id.lower()}_microarray_tpm_proxy_ensembl"
        f"{ensembl_release}_clean_tpm_v4"
    )
    out["notes"] = (
        f"{platform_name} microarray-derived TPM-proxy "
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


def build_gpl570_source(**kwargs) -> int:
    """Back-compat thin wrapper — pins ``platform_id="GPL570"`` and
    ``platform_name="Affymetrix HG-U133 Plus 2.0"`` for the original
    pre-v5.5.0 callers. New cohorts should call
    :func:`build_microarray_source` directly with the right platform."""
    kwargs.setdefault("platform_id", "GPL570")
    kwargs.setdefault("platform_name", "Affymetrix HG-U133 Plus 2.0 (GPL570)")
    return build_microarray_source(**kwargs)


__all__ = [
    "build_microarray_source",
    "build_gpl570_source",
    "parse_series_matrix",
    "parse_geo_platform_table",
    "annot_url_for",
    "annot_local_filename",
]
