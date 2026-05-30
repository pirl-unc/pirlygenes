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
from functools import lru_cache
from pathlib import Path

import pandas as pd

NCBI_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
    "Homo_sapiens.gene_info.gz"
)
NCBI_GENE_HISTORY_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz"

HUMAN_TAX_ID = "9606"


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


def harmonize_entrez_via_ncbi(
    matrix: pd.DataFrame,
    *,
    ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrez-indexed matrix → (mapping, ENSG-indexed matrix).

    Returns the same shape as ``geo_matrix.harmonize_gene_ids``:
    a mapping table with columns ``(source_id, Ensembl_Gene_ID, Symbol)``
    and the matrix re-aggregated (sum) per Ensembl_Gene_ID. Multi-Entrez
    → single-symbol collisions are summed; ambiguous symbols (pyensembl
    returns multiple ENSG for one symbol) are dropped.
    """
    from pyensembl import EnsemblRelease

    entrez_to_sym = cached_entrez_to_symbol()
    genome = EnsemblRelease(ensembl_release)

    rows: list[dict[str, str]] = []
    resolved = unresolved_entrez = ambiguous_symbol = 0
    for raw in matrix.index:
        eid = str(raw).strip()
        if not eid.isdigit():
            unresolved_entrez += 1
            continue
        sym = entrez_to_sym.get(eid)
        if not sym:
            unresolved_entrez += 1
            continue
        try:
            genes = genome.genes_by_name(sym)
        except Exception:
            genes = []
        ids = {g.gene_id.split(".", 1)[0] for g in genes}
        if len(ids) == 1:
            g = genes[0]
            rows.append({
                "source_id": eid,
                "Ensembl_Gene_ID": g.gene_id.split(".", 1)[0],
                "Symbol": g.gene_name or sym,
            })
            resolved += 1
        elif not ids:
            unresolved_entrez += 1
        else:
            ambiguous_symbol += 1
    print(
        f"  Entrez→ENSG: resolved={resolved:,}, "
        f"unresolved={unresolved_entrez:,}, "
        f"ambiguous_symbol={ambiguous_symbol:,}"
    )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        return mapping, pd.DataFrame(columns=matrix.columns)

    flat = matrix.reset_index().rename(
        columns={matrix.index.name or "index": "source_id"}
    )
    flat["source_id"] = flat["source_id"].astype(str)
    merged = mapping.merge(flat, on="source_id", how="inner")
    sample_cols = [
        c for c in merged.columns
        if c not in {"source_id", "Ensembl_Gene_ID", "Symbol"}
    ]
    by_gene = merged.groupby(
        "Ensembl_Gene_ID", as_index=True, sort=False,
    )[sample_cols].sum()
    return mapping, by_gene


__all__ = [
    "NCBI_GENE_INFO_URL",
    "NCBI_GENE_HISTORY_URL",
    "load_entrez_to_symbol",
    "cached_entrez_to_symbol",
    "load_entrez_to_ensembl",
    "cached_entrez_to_ensembl",
    "load_entrez_history",
    "cached_entrez_history",
    "harmonize_entrez_via_ncbi",
]
