"""Shared gene-identifier mapping primitives.

Every builder turns some source identifier — an Ensembl gene ID, an
Entrez Gene ID, or a (possibly legacy) HUGO symbol — into the same
canonical ``(Ensembl_Gene_ID, Symbol)`` pair. Before this module each
builder carried its own copy of that logic (``geo_matrix`` had three,
``treehouse`` one, ``affy_gpl570`` one, ``ncbi_gene_info`` one), which
drifted apart and were individually hard to test.

This module is the single home for those primitives. They are plain
functions, not a framework: each builder still owns its file reading
and stat computation and simply *composes* these. The only thing that
differs between datasets is the starting identifier type — so the
front doors (`gene_from_ensembl_id`, `entrez_to_gene`, `gene_from_symbol`,
`rescue_symbol`) are split by input type and converge on the same
canonical output.

Raw lookup tables (NCBI gene_info / gene_history) live in
:mod:`pirlygenes.builders.ncbi_gene_info`; the curated literature
display names live in :mod:`pirlygenes.gene_names`. Both feed
:func:`cached_combined_alias_index`, so a synonym like ``NY-ESO-1`` or
``PD-L1`` resolves the same way an NCBI ``Synonyms`` entry does.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Literal

import pandas as pd

from ..gene_ids import gene_for_ensembl_id, strip_version
from ..gene_names import aliases as _CURATED_DISPLAY_ALIASES
from .ncbi_gene_info import (
    GENE_INFO_SYNONYM_CONFIDENCE,
    SymbolAliasCandidate,
    SymbolAliasIndex,
    cached_entrez_history,
    cached_entrez_to_ensembl,
    cached_entrez_to_symbol,
    cached_symbol_alias_index,
)

GeneIdType = Literal["ensembl", "hugo", "entrez", "auto"]

# Hand-curated display names are trusted slightly above NCBI synonyms.
CURATED_DISPLAY_CONFIDENCE = 90
_CURATED_DISPLAY_SOURCE = "gene_names.display_aliases"

# How a symbol/id was resolved — short strings, used only for build-log
# breakdowns. No enum: a plain label keeps the API weightless.
METHOD_ENSEMBL_ID = "ensembl_id"
METHOD_SYMBOL = "symbol"
METHOD_ENTREZ_DBXREFS = "entrez_dbxrefs"
METHOD_ENTREZ_CURRENT_SYMBOL = "entrez_current_symbol"
METHOD_GENE_HISTORY = "gene_history"
METHOD_SYNONYM = "synonym"


# ─── id-type detection ──────────────────────────────────────────────────────
# ``strip_version`` and ``gene_for_ensembl_id`` are imported from
# ``pirlygenes.gene_ids`` — the single home for those primitives.

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$")
_ENTREZ_RE = re.compile(r"^\d+$")


def detect_id_type(ids: Iterable[str], sample_size: int = 200) -> GeneIdType:
    """Sniff whether an index is Ensembl IDs, Entrez IDs, or HUGO symbols."""
    sampled = list(ids)[:sample_size]
    if not sampled:
        return "hugo"
    ensembl = sum(1 for x in sampled if _ENSEMBL_RE.match(str(x)))
    entrez = sum(1 for x in sampled if _ENTREZ_RE.match(str(x)))
    if ensembl / len(sampled) > 0.5:
        return "ensembl"
    if entrez / len(sampled) > 0.5:
        return "entrez"
    return "hugo"


# ─── pyensembl single-gene resolution ───────────────────────────────────────

def gene_from_ensembl_id(genome, raw: str) -> tuple[str, str] | None:
    """Canonicalize an Ensembl gene ID → (unversioned ENSG, symbol) or None.

    Builder-side wrapper that applies the symbol-naming policy (fall back
    to the bare ID) on top of the shared ``gene_for_ensembl_id`` resolver.
    """
    sid = strip_version(raw)
    if not sid:
        return None
    gene = gene_for_ensembl_id(genome, sid)
    if gene is None:
        return None
    return strip_version(gene.gene_id), gene.gene_name or sid


def unique_ensembl_id_for_symbol(genome, symbol: str) -> str | None:
    """Return the ENSG for a symbol only if pyensembl maps it unambiguously."""
    try:
        genes = genome.genes_by_name(symbol)
    except Exception:
        genes = []
    ids = {strip_version(g.gene_id) for g in genes}
    if len(ids) != 1:
        return None
    return next(iter(ids))


def gene_from_symbol(genome, symbol: str) -> tuple[str, str] | None:
    """Resolve a HUGO symbol → (ENSG, canonical symbol) if unambiguous.

    Returns pyensembl's canonical ``gene_name`` (falling back to the
    input) so callers store the current symbol, not whatever spelling
    the source matrix used. Ambiguous (>1 gene) or unknown → None.
    """
    sym = str(symbol).strip()
    if not sym:
        return None
    try:
        genes = genome.genes_by_name(sym)
    except Exception:
        genes = []
    ids = {strip_version(g.gene_id) for g in genes}
    if len(ids) != 1:
        return None
    gene = genes[0]
    return strip_version(gene.gene_id), gene.gene_name or sym


# ─── Entrez resolution (NCBI tables → pyensembl) ─────────────────────────────

def entrez_to_gene(
    genome,
    entrez_id: str,
    *,
    legacy_symbol: str | None = None,
    use_history: bool = True,
) -> tuple[str, str, str] | None:
    """Resolve an Entrez Gene ID → (ENSG, symbol, method) or None.

    Walks the same chain every Entrez-keyed dataset needs:

    1. ``entrez_dbxrefs`` — direct ENSG from gene_info dbXrefs (wins when
       pyensembl's release doesn't know the current symbol).
    2. ``entrez_current_symbol`` — current NCBI symbol → unique pyensembl ENSG.
    3. ``gene_history`` — a discontinued ID is redirected to its live
       replacement and steps 1–2 retried (counted as gene_history).

    ``legacy_symbol`` is only used as a final naming fall-back.
    """
    entrez_to_ensembl = cached_entrez_to_ensembl()
    entrez_to_symbol = cached_entrez_to_symbol()

    def _direct(eid: str) -> tuple[str, str] | None:
        ensembl_id = entrez_to_ensembl.get(eid)
        current_symbol = entrez_to_symbol.get(eid)
        # Tier 1: dbXrefs ENSG.
        if ensembl_id:
            name = current_symbol or legacy_symbol
            if name is None:
                resolved = gene_from_ensembl_id(genome, ensembl_id)
                name = resolved[1] if resolved else ensembl_id
            return ensembl_id, name
        # Tier 2: current symbol → pyensembl.
        if current_symbol and current_symbol != legacy_symbol:
            ensembl_id = unique_ensembl_id_for_symbol(genome, current_symbol)
            if ensembl_id:
                return ensembl_id, current_symbol
        return None

    eid = str(entrez_id).strip()
    if not eid:
        return None

    direct = _direct(eid)
    if direct:
        ensembl_id, name = direct
        method = (
            METHOD_ENTREZ_DBXREFS
            if entrez_to_ensembl.get(eid)
            else METHOD_ENTREZ_CURRENT_SYMBOL
        )
        return ensembl_id, name, method

    if use_history:
        live_eid = cached_entrez_history().get(eid)
        if live_eid:
            retried = _direct(live_eid)
            if retried:
                ensembl_id, name = retried
                return ensembl_id, name, METHOD_GENE_HISTORY
    return None


# ─── combined alias pool (NCBI Synonyms + curated display names) ─────────────

def _curated_display_candidates() -> dict[str, list[SymbolAliasCandidate]]:
    """Curated literature display names as bidirectional alias candidates.

    ``gene_names.aliases`` maps a symbol to its preferred display label
    (``CTAG1B`` → ``NY-ESO-1``). For *mapping* we want either string to
    resolve to the other, so both directions are registered. The
    unique-ENSG requirement and live-symbol-ownership guard in
    :func:`rescue_symbol` keep that safe.
    """
    out: dict[str, list[SymbolAliasCandidate]] = {}
    for symbol, display in _CURATED_DISPLAY_ALIASES.items():
        for alias, official in ((display, symbol), (symbol, display)):
            if not alias or alias == official:
                continue
            out.setdefault(alias, []).append(
                SymbolAliasCandidate(
                    official_symbol=official,
                    source=_CURATED_DISPLAY_SOURCE,
                    confidence=CURATED_DISPLAY_CONFIDENCE,
                )
            )
    return out


@lru_cache(maxsize=1)
def cached_combined_alias_index() -> SymbolAliasIndex:
    """NCBI ``Synonyms`` alias index, augmented with curated display names."""
    base = cached_symbol_alias_index()
    merged: dict[str, tuple[SymbolAliasCandidate, ...]] = dict(
        base.alias_candidates
    )
    for alias, extra in _curated_display_candidates().items():
        existing = merged.get(alias, ())
        seen = {(c.official_symbol, c.source) for c in existing}
        fresh = tuple(
            c for c in extra if (c.official_symbol, c.source) not in seen
        )
        if fresh:
            merged[alias] = existing + fresh
    return SymbolAliasIndex(
        official_symbols=base.official_symbols,
        alias_candidates=merged,
    )


def synonym_to_official(
    symbol: str,
    alias_index: SymbolAliasIndex,
    *,
    min_confidence: int = GENE_INFO_SYNONYM_CONFIDENCE,
) -> str | None:
    """Return the unique official symbol a synonym points to, if it is safe.

    A live official NCBI symbol owns its row even when the current
    record lacks an Ensembl ID, so it is never reassigned through a
    synonym claim on a different gene. Ambiguous (>1 target) or
    low-confidence aliases return None.
    """
    if symbol in alias_index.official_symbols:
        return None
    candidates = [
        c for c in alias_index.alias_candidates.get(symbol, ())
        if c.confidence >= min_confidence and c.official_symbol != symbol
    ]
    targets = {c.official_symbol for c in candidates}
    if len(targets) != 1:
        return None
    return next(iter(targets))


def rescue_symbol(
    genome,
    symbol: str,
    *,
    entrez_id: str | None = None,
    alias_index: SymbolAliasIndex | None = None,
) -> tuple[str, str, str] | None:
    """Best-effort resolve a symbol pyensembl couldn't map directly.

    Tries, in order: the Entrez chain (if an ID is supplied), then the
    combined synonym pool (NCBI ``Synonyms`` + curated display names).
    Returns (ENSG, symbol, method) or None.
    """
    if entrez_id and str(entrez_id).strip():
        via_entrez = entrez_to_gene(genome, entrez_id, legacy_symbol=symbol)
        if via_entrez:
            return via_entrez

    if alias_index is None:
        alias_index = cached_combined_alias_index()
    official = synonym_to_official(symbol, alias_index)
    if official:
        ensembl_id = unique_ensembl_id_for_symbol(genome, official)
        if ensembl_id:
            return ensembl_id, official, METHOD_SYNONYM
    return None


def resolve_symbol(
    genome,
    symbol: str,
    *,
    entrez_id: str | None = None,
    alias_index: SymbolAliasIndex | None = None,
) -> tuple[str, str, str] | None:
    """THE symbol → (ENSG, symbol, method) resolver every builder routes through.

    Direct pyensembl lookup → Entrez chain (only if an ID is available) →
    synonym rescue (NCBI ``Synonyms`` + curated display aliases). Because
    every symbol-keyed dataset calls this one function, a symbol resolves
    the *same way* regardless of which builder sees it — the only
    per-dataset difference is whether an Entrez ID is on hand. Returns
    (ENSG, symbol, method) or None.
    """
    direct = gene_from_symbol(genome, symbol)
    if direct is not None:
        ensembl_id, name = direct
        return ensembl_id, name, METHOD_SYMBOL
    return rescue_symbol(
        genome, symbol, entrez_id=entrez_id, alias_index=alias_index,
    )


# ─── matrix aggregation ──────────────────────────────────────────────────────

def aggregate_matrix_by_mapping(
    matrix: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    source_col: str = "source_id",
) -> pd.DataFrame:
    """Sum-aggregate a gene×sample matrix to one row per Ensembl_Gene_ID.

    ``mapping`` has columns ``(source_col, Ensembl_Gene_ID, Symbol)``;
    the matrix is indexed by the source identifier. Many-source →
    one-gene collisions are summed.
    """
    if mapping.empty:
        return pd.DataFrame(columns=matrix.columns)
    flat = matrix.reset_index().rename(
        columns={matrix.index.name or "index": source_col}
    )
    flat[source_col] = flat[source_col].astype(str)
    merged = mapping.merge(flat, on=source_col, how="inner")
    sample_cols = [
        c for c in merged.columns
        if c not in {source_col, "Ensembl_Gene_ID", "Symbol"}
    ]
    return merged.groupby(
        "Ensembl_Gene_ID", as_index=True, sort=False,
    )[sample_cols].sum()


def harmonize_entrez_matrix(
    matrix: pd.DataFrame, *, ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrez-indexed matrix → (mapping, ENSG-indexed sum-aggregated matrix).

    Mapping columns are ``(source_id, Ensembl_Gene_ID, Symbol,
    mapping_method)``. Each Entrez ID goes through the shared
    :func:`entrez_to_gene` chain (dbXrefs → current-symbol → gene_history),
    so this benefits from the same recovery every other dataset gets. The
    ``mapping_method`` column records which tier resolved each row (for the
    builder-side mapping audit); it is dropped before aggregation so it is
    never mistaken for a sample column. The columns are fixed even when the
    mapping is empty so downstream ``drop_duplicates("Ensembl_Gene_ID")`` /
    audit callers never hit a ``KeyError`` on a bare empty frame.
    """
    from pyensembl import EnsemblRelease

    genome = EnsemblRelease(ensembl_release)
    rows: list[dict[str, str]] = []
    resolved = unresolved = 0
    for raw in matrix.index:
        entrez_id = str(raw).strip()
        if not entrez_id.isdigit():
            unresolved += 1
            continue
        result = entrez_to_gene(genome, entrez_id)
        if result is None:
            unresolved += 1
            continue
        ensembl_id, name, method = result
        rows.append({
            "source_id": entrez_id,
            "Ensembl_Gene_ID": ensembl_id,
            "Symbol": name,
            "mapping_method": method,
        })
        resolved += 1
    print(f"  Entrez→ENSG: resolved={resolved:,}, unresolved={unresolved:,}")
    mapping = pd.DataFrame(
        rows,
        columns=["source_id", "Ensembl_Gene_ID", "Symbol", "mapping_method"],
    )
    if mapping.empty:
        return mapping, pd.DataFrame(columns=matrix.columns)
    return mapping, aggregate_matrix_by_mapping(
        matrix, mapping[["source_id", "Ensembl_Gene_ID", "Symbol"]],
    )


__all__ = [
    "GeneIdType",
    "CURATED_DISPLAY_CONFIDENCE",
    "METHOD_ENSEMBL_ID",
    "METHOD_SYMBOL",
    "METHOD_ENTREZ_DBXREFS",
    "METHOD_ENTREZ_CURRENT_SYMBOL",
    "METHOD_GENE_HISTORY",
    "METHOD_SYNONYM",
    "strip_version",
    "detect_id_type",
    "gene_from_ensembl_id",
    "unique_ensembl_id_for_symbol",
    "gene_from_symbol",
    "entrez_to_gene",
    "cached_combined_alias_index",
    "synonym_to_official",
    "rescue_symbol",
    "resolve_symbol",
    "aggregate_matrix_by_mapping",
    "harmonize_entrez_matrix",
]
