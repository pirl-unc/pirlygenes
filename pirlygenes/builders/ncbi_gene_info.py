"""Entrez Gene ID ↔ HUGO symbol / Ensembl mapping from NCBI gene_info
and gene_history.

Pyensembl doesn't expose Entrez IDs on its Gene objects, so we
download the canonical NCBI mappings once and cache them. Used by any
GEO builder whose source matrix is keyed by Entrez Gene IDs (HTSeq +
GENCODE Entrez output is common; e.g. GSE98894) AND by the
microarray builder's symbol-rescue chain for pre-2010 platforms
whose HUGO column has since been renamed by HGNC.

Source files:
  https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
  https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz

``gene_info`` (one row per *live* Entrez ID) carries the current
canonical HUGO symbol plus a ``dbXrefs`` column with cross-references
including ``Ensembl:ENSGxxxxx`` for the ~38k well-curated genes.
That gives us TWO independent Entrez→ENSG paths: name-lookup via
pyensembl, OR direct ENSG extraction from dbXrefs. The latter wins
when pyensembl's release doesn't know the current symbol (HLA region,
copy-number-variable loci, intronic transcripts).

``gene_history`` (one row per *discontinued* Entrez ID) maps every
withdrawn or merged Entrez ID to its replacement (or "-" for
permanently withdrawn). This rescues pre-2010 platforms that ship
Entrez IDs which have since been merged into other records.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

NCBI_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)
NCBI_GENE_HISTORY_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz"

HUMAN_TAX_ID = "9606"
GENE_INFO_SYNONYM_CONFIDENCE = 80


@dataclass(frozen=True)
class SymbolAliasCandidate:
    """One possible alias → official-symbol mapping from an external source."""

    official_symbol: str
    source: str
    confidence: int


@dataclass(frozen=True)
class SymbolAliasIndex:
    """Current official symbols plus all alias candidates keyed by alias."""

    official_symbols: frozenset[str]
    alias_candidates: dict[str, tuple[SymbolAliasCandidate, ...]]


def _default_cache_path() -> Path:
    return (
        Path.home() / ".cache" / "pirlygenes" / "ncbi_gene_info"
        / "Homo_sapiens.gene_info.gz"
    )


def _default_history_cache_path() -> Path:
    return (
        Path.home() / ".cache" / "pirlygenes" / "ncbi_gene_info"
        / "gene_history.gz"
    )


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def load_entrez_to_symbol(
    cache_path: Path | None = None, *, refresh: bool = False,
) -> dict[str, str]:
    """Return {entrez_id (str): hugo_symbol} from NCBI gene_info."""
    path = cache_path or _default_cache_path()
    if refresh and path.exists():
        path.unlink()
    _download(NCBI_GENE_INFO_URL, path)
    print(f"loading NCBI Entrez→Symbol map from {path.name}...")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(
            f, sep="\t", usecols=["GeneID", "Symbol"],
            dtype={"GeneID": "string", "Symbol": "string"},
            low_memory=False,
        )
    df = df.dropna(subset=["GeneID", "Symbol"])
    df = df[df["Symbol"].ne("") & df["Symbol"].ne("NEWENTRY")]
    out = dict(zip(df["GeneID"].tolist(), df["Symbol"].tolist()))
    print(f"  {len(out):,} Entrez → Symbol entries")
    return out


@lru_cache(maxsize=1)
def cached_entrez_to_symbol() -> dict[str, str]:
    """Module-level cached singleton (in-process)."""
    return load_entrez_to_symbol()


def load_entrez_to_ensembl(
    cache_path: Path | None = None, *, refresh: bool = False,
) -> dict[str, str]:
    """Return {entrez_id (str): ENSG (unversioned)} from gene_info dbXrefs.

    dbXrefs is pipe-separated; the Ensembl reference looks like
    ``Ensembl:ENSG00000121410``. About 38k of NCBI's 194k human genes
    have one, covering the well-curated protein-coding + lncRNA set.

    Lets the microarray builder skip pyensembl's name lookup for
    Entrez IDs whose current symbol pyensembl doesn't know
    (HLA-DRB4, CCL3L1, intronic ``*-IT1`` transcripts, etc.).
    """
    path = cache_path or _default_cache_path()
    if refresh and path.exists():
        path.unlink()
    _download(NCBI_GENE_INFO_URL, path)
    print(f"loading NCBI Entrez→Ensembl map from {path.name}...")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(
            f, sep="\t", usecols=["GeneID", "dbXrefs"],
            dtype={"GeneID": "string", "dbXrefs": "string"},
            low_memory=False,
        )
    df = df.dropna(subset=["GeneID", "dbXrefs"])
    df = df[df["dbXrefs"].str.contains("Ensembl:ENSG", na=False, regex=False)]
    ensg = df["dbXrefs"].str.extract(r"Ensembl:(ENSG\d+)", expand=False)
    out = dict(zip(df["GeneID"].tolist(), ensg.tolist()))
    out = {k: v for k, v in out.items() if isinstance(v, str)}
    print(f"  {len(out):,} Entrez → Ensembl entries")
    return out


@lru_cache(maxsize=1)
def cached_entrez_to_ensembl() -> dict[str, str]:
    return load_entrez_to_ensembl()


def load_entrez_history(
    cache_path: Path | None = None, *, refresh: bool = False,
) -> dict[str, str]:
    """Return {discontinued_entrez_id (str): live_entrez_id (str)}.

    Skips permanently-withdrawn IDs (where the live column is "-").
    Of NCBI's 166k human discontinued IDs, ~28k have a live
    replacement; the rest were deleted outright (LOC* placeholders
    that never got promoted to a real symbol).

    Used by the microarray-builder's symbol-rescue chain: when an
    Entrez ID from a 2003-era platform table isn't in the current
    gene_info, we look it up here and retry the lookup with the
    replacement ID.
    """
    path = cache_path or _default_history_cache_path()
    if refresh and path.exists():
        path.unlink()
    _download(NCBI_GENE_HISTORY_URL, path)
    print(f"loading NCBI Entrez history from {path.name}...")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(
            f, sep="\t",
            usecols=["#tax_id", "GeneID", "Discontinued_GeneID"],
            dtype={
                "#tax_id": "string",
                "GeneID": "string",
                "Discontinued_GeneID": "string",
            },
            low_memory=False,
        )
    df = df[df["#tax_id"] == HUMAN_TAX_ID]
    df = df[df["GeneID"].ne("-") & df["Discontinued_GeneID"].ne("-")]
    out = dict(zip(
        df["Discontinued_GeneID"].tolist(), df["GeneID"].tolist(),
    ))
    print(f"  {len(out):,} discontinued → live Entrez IDs")
    return out


@lru_cache(maxsize=1)
def cached_entrez_history() -> dict[str, str]:
    return load_entrez_history()


def load_symbol_alias_index(
    cache_path: Path | None = None, *, refresh: bool = False,
) -> SymbolAliasIndex:
    """Return broad NCBI symbol-alias candidates plus live official symbols.

    ``gene_info`` carries a live official ``Symbol`` column and a
    pipe-separated ``Synonyms`` column. We preserve every synonym
    candidate instead of doing a first-wins inversion, so downstream
    callers can choose confidence thresholds and reject ambiguous
    aliases. The official-symbol set lets callers block cases where an
    unresolved platform symbol is already owned by a current Entrez
    record, even if that current record has no Ensembl cross-reference.
    """
    path = cache_path or _default_cache_path()
    if refresh and path.exists():
        path.unlink()
    _download(NCBI_GENE_INFO_URL, path)
    print(f"loading NCBI symbol alias index from {path.name}...")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(
            f, sep="\t", usecols=["Symbol", "Synonyms"],
            dtype={"Symbol": "string", "Synonyms": "string"},
            low_memory=False,
        )
    df = df[df["Symbol"].notna() & df["Symbol"].ne("")]
    df = df[df["Symbol"].ne("NEWENTRY")]
    official_symbols = frozenset(df["Symbol"].tolist())

    rows = df[df["Synonyms"].notna() & df["Synonyms"].ne("-")]
    tmp: dict[str, list[SymbolAliasCandidate]] = {}
    seen: set[tuple[str, str, str]] = set()
    source = "ncbi_gene_info.Synonyms"
    for sym, syn_str in zip(rows["Symbol"].tolist(), rows["Synonyms"].tolist()):
        for alias in syn_str.split("|"):
            alias = alias.strip()
            if not alias or alias == sym:
                continue
            key = (alias, sym, source)
            if key in seen:
                continue
            seen.add(key)
            tmp.setdefault(alias, []).append(
                SymbolAliasCandidate(
                    official_symbol=sym,
                    source=source,
                    confidence=GENE_INFO_SYNONYM_CONFIDENCE,
                )
            )

    alias_candidates = {alias: tuple(cands) for alias, cands in tmp.items()}
    print(
        f"  {len(official_symbols):,} official symbols; "
        f"{len(alias_candidates):,} alias candidate keys"
    )
    return SymbolAliasIndex(
        official_symbols=official_symbols,
        alias_candidates=alias_candidates,
    )


@lru_cache(maxsize=1)
def cached_symbol_alias_index() -> SymbolAliasIndex:
    return load_symbol_alias_index()


def harmonize_entrez_via_ncbi(
    matrix: pd.DataFrame,
    *,
    ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrez-indexed matrix → (mapping, ENSG-indexed matrix).

    Backward-compatible name for the shared
    :func:`pirlygenes.builders.gene_mapping.harmonize_entrez_matrix`.
    Returns the same shape as ``geo_matrix.harmonize_gene_ids``: a
    mapping table with columns ``(source_id, Ensembl_Gene_ID, Symbol)``
    and the matrix re-aggregated (sum) per Ensembl_Gene_ID. The
    import is function-level because ``gene_mapping`` imports this
    module's table loaders.
    """
    from .gene_mapping import harmonize_entrez_matrix

    return harmonize_entrez_matrix(matrix, ensembl_release=ensembl_release)


__all__ = [
    "NCBI_GENE_INFO_URL",
    "NCBI_GENE_HISTORY_URL",
    "load_entrez_to_symbol",
    "cached_entrez_to_symbol",
    "load_entrez_to_ensembl",
    "cached_entrez_to_ensembl",
    "load_entrez_history",
    "cached_entrez_history",
    "GENE_INFO_SYNONYM_CONFIDENCE",
    "SymbolAliasCandidate",
    "SymbolAliasIndex",
    "load_symbol_alias_index",
    "cached_symbol_alias_index",
    "harmonize_entrez_via_ncbi",
]
