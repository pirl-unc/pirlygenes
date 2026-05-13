"""Curated functional gene families — symbol/ENSG sets shipped as CSVs.

Each family is a set of HGNC symbols and Ensembl gene IDs sharing a
biological or structural origin (NUMTs, nuclear rRNA pseudogenes,
ribosomal proteins, histones, hemoglobins, etc.). These are useful
panels for QC, normalization, attribution, and downstream analysis —
:mod:`trufflepig.expression_qc` reads them as the source of truth for
ENSG-stable feature classification, falling back to symbol regex when
an ID isn't present in any family.

Files in ``pirlygenes/data/``:

    mitochondrial-genes.csv          (curated; ``Symbol, Ensembl_Gene_ID, Role``)
    numt-pseudogenes.csv             (derived)
    nuclear-retained-lncrnas.csv     (derived; MALAT1, NEAT1)
    rrna-and-pseudogenes.csv         (derived)
    ribosomal-protein-genes.csv      (derived)
    ribosomal-protein-pseudogenes.csv (derived)
    small-noncoding-rnas.csv         (derived; snoRNAs, snRNAs, miRNAs, Y RNAs, vault RNAs, ...)
    histone-genes.csv                (derived)
    hemoglobin-genes.csv             (derived)
    immune-receptor-segments.csv     (derived; IG/TR V/D/J/C)

Derived CSVs are produced by ``scripts/generate_gene_family_sets.py``
walking every installed Ensembl release and applying the regex panel
in :func:`trufflepig.expression_qc.classify_gene_qc`. Re-run after the
regex changes.

ENSG IDs are stored **unversioned** (``ENSG00000251562``); a sample
carrying ``ENSG00000251562.5`` is stripped at lookup. Multi-release
history is preserved by emitting one row per ``(Symbol, ENSG)`` pair,
so ``MALAT1`` has 3 rows (one canonical + two historic IDs that
deprecated annotation no longer assigns).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------- public family registry ----------


@dataclass(frozen=True)
class GeneFamily:
    """One curated gene family."""

    name: str  # canonical family name, e.g. "numt_pseudogene"
    filename: str  # CSV filename under pirlygenes/data/


# Ordered: when an ENSG/Symbol appears in multiple families (rare HGNC
# rename across families), the first match wins at lookup time. Order
# reflects the QC-relevance priority used by trufflepig.expression_qc.
GENE_FAMILIES: tuple[GeneFamily, ...] = (
    GeneFamily("mitochondrial", "mitochondrial-genes.csv"),
    GeneFamily("numt_pseudogene", "numt-pseudogenes.csv"),
    GeneFamily("nuclear_retained_lncrna", "nuclear-retained-lncrnas.csv"),
    GeneFamily("rrna_and_pseudogene", "rrna-and-pseudogenes.csv"),
    GeneFamily("ribosomal_protein_pseudogene", "ribosomal-protein-pseudogenes.csv"),
    GeneFamily("ribosomal_protein", "ribosomal-protein-genes.csv"),
    GeneFamily("small_noncoding_rna", "small-noncoding-rnas.csv"),
    GeneFamily("histone", "histone-genes.csv"),
    GeneFamily("hemoglobin", "hemoglobin-genes.csv"),
    GeneFamily("immune_receptor_segment", "immune-receptor-segments.csv"),
)


# ---------- helpers ----------


def _strip_version(ensg: str) -> str:
    """``ENSG00000251562.5`` → ``ENSG00000251562``."""
    return str(ensg or "").split(".", 1)[0].strip()


@lru_cache(maxsize=1)
def _load_table() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in GENE_FAMILIES:
        path = _DATA_DIR / family.filename
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df["family"] = family.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Ensembl_Gene_ID", "family"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["Ensembl_Gene_ID"] = out["Ensembl_Gene_ID"].astype(str).map(_strip_version)
    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()
    return out


def _family_df(name: str) -> pd.DataFrame:
    return _load_table().query("family == @name").reset_index(drop=True)


def _family_ids(name: str) -> set[str]:
    return set(_family_df(name)["Ensembl_Gene_ID"].astype(str))


def _family_symbols(name: str) -> set[str]:
    return set(_family_df(name)["Symbol"].astype(str).str.upper())


# ---------- typed per-family accessors ----------


def numt_pseudogene_ids() -> set[str]:
    """NUMT-like nuclear mitochondrial pseudogene Ensembl IDs."""
    return _family_ids("numt_pseudogene")


def numt_pseudogene_symbols() -> set[str]:
    """NUMT-like nuclear mitochondrial pseudogene symbols."""
    return _family_symbols("numt_pseudogene")


def nuclear_retained_lncrna_ids() -> set[str]:
    """Nuclear-retained ENE-stabilized lncRNA Ensembl IDs (MALAT1, NEAT1)."""
    return _family_ids("nuclear_retained_lncrna")


def nuclear_retained_lncrna_symbols() -> set[str]:
    """Nuclear-retained ENE-stabilized lncRNA symbols."""
    return _family_symbols("nuclear_retained_lncrna")


def rrna_and_pseudogene_ids() -> set[str]:
    """Nuclear rRNA + rRNA-pseudogene Ensembl IDs."""
    return _family_ids("rrna_and_pseudogene")


def rrna_and_pseudogene_symbols() -> set[str]:
    """Nuclear rRNA + rRNA-pseudogene symbols."""
    return _family_symbols("rrna_and_pseudogene")


def ribosomal_protein_ids() -> set[str]:
    """Functional ribosomal protein Ensembl IDs (RPL*, RPS*, RPLP*)."""
    return _family_ids("ribosomal_protein")


def ribosomal_protein_symbols() -> set[str]:
    """Functional ribosomal protein symbols."""
    return _family_symbols("ribosomal_protein")


def ribosomal_protein_pseudogene_ids() -> set[str]:
    """Ribosomal protein pseudogene Ensembl IDs (RPL*P\\d+, RPS*P\\d+)."""
    return _family_ids("ribosomal_protein_pseudogene")


def ribosomal_protein_pseudogene_symbols() -> set[str]:
    """Ribosomal protein pseudogene symbols."""
    return _family_symbols("ribosomal_protein_pseudogene")


def small_noncoding_rna_ids() -> set[str]:
    """Small non-coding RNA Ensembl IDs (snoRNA, snRNA, miRNA, Y RNA, ...)."""
    return _family_ids("small_noncoding_rna")


def small_noncoding_rna_symbols() -> set[str]:
    """Small non-coding RNA symbols."""
    return _family_symbols("small_noncoding_rna")


def histone_gene_ids() -> set[str]:
    """Replication-dependent histone Ensembl IDs (H1, H2A, H2B, H3, H4)."""
    return _family_ids("histone")


def histone_gene_symbols() -> set[str]:
    """Replication-dependent histone symbols."""
    return _family_symbols("histone")


def hemoglobin_gene_ids() -> set[str]:
    """Hemoglobin Ensembl IDs (HBA, HBB, HBD, HBE, HBG, HBM, HBQ, HBZ + pseudogenes)."""
    return _family_ids("hemoglobin")


def hemoglobin_gene_symbols() -> set[str]:
    """Hemoglobin symbols."""
    return _family_symbols("hemoglobin")


def immune_receptor_segment_ids() -> set[str]:
    """Immunoglobulin and T-cell receptor V/D/J/C segment Ensembl IDs."""
    return _family_ids("immune_receptor_segment")


def immune_receptor_segment_symbols() -> set[str]:
    """Immunoglobulin and T-cell receptor V/D/J/C segment symbols."""
    return _family_symbols("immune_receptor_segment")


# ---------- generic accessors ----------


def gene_family_names() -> list[str]:
    """Canonical names of every shipped family."""
    return [f.name for f in GENE_FAMILIES]


def gene_family_table() -> pd.DataFrame:
    """Long-form table across every family — one row per (Symbol, ENSG, family)."""
    return _load_table().copy()


def gene_family_ids(name: str) -> set[str]:
    """Unversioned ENSG set for one named family."""
    return _family_ids(name)


def gene_family_symbols(name: str) -> set[str]:
    """Uppercase symbol set for one named family."""
    return _family_symbols(name)


# ---------- ENSG / Symbol → family lookup ----------


@lru_cache(maxsize=1)
def _ensembl_id_to_family() -> dict[str, str]:
    """Build a reverse lookup ``{ENSG: family_name}`` with deterministic
    first-match-wins ordering when an ID appears in multiple families
    (HGNC rename across families)."""
    df = _load_table()
    priority = {f.name: i for i, f in enumerate(GENE_FAMILIES)}
    df = df.assign(_priority=df["family"].map(priority).fillna(len(priority)))
    df = df.sort_values("_priority")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        ensg = row["Ensembl_Gene_ID"]
        if ensg in out:
            continue
        out[ensg] = row["family"]
    return out


@lru_cache(maxsize=1)
def _symbol_to_family() -> dict[str, str]:
    df = _load_table()
    priority = {f.name: i for i, f in enumerate(GENE_FAMILIES)}
    df = df.assign(_priority=df["family"].map(priority).fillna(len(priority)))
    df = df.sort_values("_priority")
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        if sym in out:
            continue
        out[sym] = row["family"]
    return out


def gene_family_for_ensembl_id(ensembl_id: str | None) -> str | None:
    """ENSG → family name; ``None`` if the ID isn't in any family."""
    if not ensembl_id:
        return None
    return _ensembl_id_to_family().get(_strip_version(ensembl_id))


def gene_family_for_symbol(symbol: str | None) -> str | None:
    """Symbol → family name; ``None`` if the symbol isn't in any family."""
    if not symbol:
        return None
    return _symbol_to_family().get(str(symbol).strip().upper())


__all__ = [
    "GeneFamily",
    "GENE_FAMILIES",
    # generic
    "gene_family_names",
    "gene_family_table",
    "gene_family_ids",
    "gene_family_symbols",
    "gene_family_for_ensembl_id",
    "gene_family_for_symbol",
    # typed per family
    "numt_pseudogene_ids",
    "numt_pseudogene_symbols",
    "nuclear_retained_lncrna_ids",
    "nuclear_retained_lncrna_symbols",
    "rrna_and_pseudogene_ids",
    "rrna_and_pseudogene_symbols",
    "ribosomal_protein_ids",
    "ribosomal_protein_symbols",
    "ribosomal_protein_pseudogene_ids",
    "ribosomal_protein_pseudogene_symbols",
    "small_noncoding_rna_ids",
    "small_noncoding_rna_symbols",
    "histone_gene_ids",
    "histone_gene_symbols",
    "hemoglobin_gene_ids",
    "hemoglobin_gene_symbols",
    "immune_receptor_segment_ids",
    "immune_receptor_segment_symbols",
]
