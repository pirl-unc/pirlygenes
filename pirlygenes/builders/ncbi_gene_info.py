"""Entrez Gene ID ↔ HUGO symbol mapping from NCBI's gene_info table.

Pyensembl doesn't expose Entrez IDs on its Gene objects, so we
download the canonical NCBI mapping once and cache it. Used by any
GEO builder whose source matrix is keyed by Entrez Gene IDs (HTSeq +
GENCODE Entrez output is common; e.g. GSE98894).

Source file:
  https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz

Columns (NCBI standard, tab-separated):
  #tax_id  GeneID  Symbol  LocusTag  Synonyms  dbXrefs  ...

This file is the canonical NCBI Gene → HUGO symbol map. Combined with
pyensembl's ``genes_by_name``, that gives Entrez → ENSG.
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


def _default_cache_path() -> Path:
    return (
        Path.home() / ".cache" / "pirlygenes" / "ncbi_gene_info"
        / "Homo_sapiens.gene_info.gz"
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
    "load_entrez_to_symbol",
    "cached_entrez_to_symbol",
    "harmonize_entrez_via_ncbi",
]
