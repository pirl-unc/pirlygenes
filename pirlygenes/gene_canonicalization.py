"""Canonical gene and proteoform identifier helpers.

This module is the runtime boundary for expression-table identity:
source identifiers enter here, are mapped onto one canonical gene key,
and only then are rows joined or compared.  The current packaged map is
offline and versioned: it pins the canonical gene space to Ensembl release
112 when that authority release is installed, then combines the bundled
Ensembl ID alias table with the existing pyensembl/NCBI-symbol resolver.
The API is deliberately separate from builder code so consumers can
canonicalize their own inputs without importing a particular ingest path.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence, cast

import pandas as pd

from .gene_ids import (
    find_gene_and_ensembl_release_by_name,
    find_gene_name_from_ensembl_gene_id,
    ncbi_synonym_official_symbol,
    strip_version,
)
from .load_dataset import get_data

CANONICAL_GENE_MAP_VERSION = "pirlygenes-gene-canonicalization-v1"
CANONICAL_PROTEOFORM_MAP_VERSION = "pirlygenes-proteoform-canonicalization-v1"
CANONICAL_ENSEMBL_RELEASE = 112

_ENSEMBL_GENE_RE = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$")
_CANONICAL_ENSG_RE = re.compile(r"^ENSG\d{11}$")
_ENTREZ_RE = re.compile(r"^\d+$")


class GeneIdentitySpaceViolation(ValueError):
    """Raised when a table claims a canonical identity space but violates it."""


@dataclass(frozen=True)
class GeneIdentitySpaceReport:
    """Structural audit of a table's gene identity semantics.

    Semantics enforced by the canonical *gene* space:

    - ``Ensembl_Gene_ID`` is the identity key and must contain only unversioned
      human Ensembl gene stable IDs from the pinned authority release (``ENSG``),
      unless a caller explicitly opts into a proteoform space.
    - ``Symbol`` is display metadata only.  It must never participate in joins,
      and one canonical gene ID should have one display symbol in an output.
    - A canonicalized table has at most one row per canonical gene ID within a
      caller-declared context (for example cohort + normalization).
    - Rows that cannot be mapped into the authority release are quarantined
      before joins; they should not survive as raw source identifiers.

    Semantics enforced by the reduced *proteoform* space:

    - Unique gene-to-protein mappings keep the canonical ENSG as their key.
    - Byte-identical multi-gene proteins use a synthetic proteoform key such as
      ``CTAG1A/B`` that means "sum these member genes in linear expression
      space."  Such keys are valid only when a caller declares
      ``allow_proteoform_ids=True``.
    """

    n_rows: int
    n_gene_ids: int
    n_duplicate_key_rows: int
    n_invalid_ids: int
    n_versioned_ids: int
    n_symbol_fallback_ids: int
    n_multi_symbol_gene_ids: int
    duplicate_examples: tuple[tuple[str, ...], ...] = ()
    invalid_id_examples: tuple[str, ...] = ()
    versioned_id_examples: tuple[str, ...] = ()
    symbol_fallback_examples: tuple[str, ...] = ()
    multi_symbol_examples: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            self.n_duplicate_key_rows == 0
            and self.n_invalid_ids == 0
            and self.n_versioned_ids == 0
            and self.n_multi_symbol_gene_ids == 0
        )

    def error_messages(self, *, forbid_symbol_fallback_ids: bool = False) -> list[str]:
        messages: list[str] = []
        if self.n_duplicate_key_rows:
            messages.append(
                f"{self.n_duplicate_key_rows} duplicate canonical gene/context rows "
                f"(examples: {list(self.duplicate_examples)})"
            )
        if self.n_invalid_ids:
            messages.append(
                f"{self.n_invalid_ids} IDs outside the declared space "
                f"(examples: {list(self.invalid_id_examples)})"
            )
        if self.n_versioned_ids:
            messages.append(
                f"{self.n_versioned_ids} versioned Ensembl IDs survived "
                f"(examples: {list(self.versioned_id_examples)})"
            )
        if self.n_multi_symbol_gene_ids:
            messages.append(
                f"{self.n_multi_symbol_gene_ids} canonical IDs carry multiple symbols "
                f"(examples: {list(self.multi_symbol_examples)})"
            )
        if forbid_symbol_fallback_ids and self.n_symbol_fallback_ids:
            messages.append(
                f"{self.n_symbol_fallback_ids} rows use the raw ENSG as Symbol "
                f"(examples: {list(self.symbol_fallback_examples)})"
            )
        return messages


def _clean_identifier(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def _gene_id_from_pyensembl(gene) -> str | None:
    raw = getattr(gene, "gene_id", None) or getattr(gene, "id", None)
    return strip_version(raw) if raw else None


@lru_cache(maxsize=1)
def _canonical_release_maps() -> dict[str, object]:
    """Return ID/symbol maps for the pinned Ensembl authority release.

    Release 112 is the authority because the packaged reference sources claim
    to have been harmonized there.  If that release is unavailable in a local
    environment, fall back to the newest usable installed human release so the
    API remains usable rather than failing at import time.
    """
    try:
        from .gene_ids import genomes
    except Exception:
        return {
            "release": None,
            "ids": frozenset(),
            "unique_symbol_to_id": {},
        }

    preferred = [
        g for g in genomes if getattr(g, "release", None) == CANONICAL_ENSEMBL_RELEASE
    ]
    candidates = preferred + [
        g for g in genomes if getattr(g, "release", None) != CANONICAL_ENSEMBL_RELEASE
    ]
    for genome in candidates:
        try:
            rows = genome.db.connection.execute(
                "SELECT gene_id, gene_name FROM gene"
            ).fetchall()
        except Exception:
            continue
        ids = {
            strip_version(gid)
            for gid, _name in rows
            if gid
        }
        if len(ids) < 50_000:
            continue
        symbol_to_ids: dict[str, set[str]] = defaultdict(set)
        for gid, name in rows:
            if gid and name:
                symbol_to_ids[str(name).strip().upper()].add(strip_version(gid))
        unique_symbol_to_id = {
            symbol: next(iter(symbol_ids))
            for symbol, symbol_ids in symbol_to_ids.items()
            if len(symbol_ids) == 1
        }
        return {
            "release": getattr(genome, "release", None),
            "ids": frozenset(ids),
            "unique_symbol_to_id": unique_symbol_to_id,
        }
    return {
        "release": None,
        "ids": frozenset(),
        "unique_symbol_to_id": {},
    }


def _authority_gene_ids() -> frozenset[str]:
    return cast(frozenset[str], _canonical_release_maps()["ids"])


def _authority_gene_id_for_symbol(symbol: str) -> str | None:
    text = _clean_identifier(symbol)
    if not text or _ENSEMBL_GENE_RE.match(text):
        return None
    unique_symbol_to_id = cast(
        dict[str, str], _canonical_release_maps()["unique_symbol_to_id"]
    )
    target = unique_symbol_to_id.get(text.upper())
    if target:
        return target
    official = ncbi_synonym_official_symbol(text)
    if official and official != text:
        return unique_symbol_to_id.get(str(official).strip().upper())
    return None


@lru_cache(maxsize=1)
def _ensembl_alias_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(alias->canonical, canonical->symbol_hint)``.

    ``ensembl-id-aliases`` currently carries retired / alternate-haplotype
    Ensembl IDs that have a documented primary-contig or successor ID.  It is
    small enough to load eagerly and safe to use at runtime.
    """
    try:
        df = get_data("ensembl-id-aliases")
    except Exception:
        return {}, {}

    alias_to_canonical: dict[str, str] = {}
    canonical_to_symbol: dict[str, str] = {}
    for _, row in df.iterrows():
        src = _clean_identifier(row.get("alt_haplotype_id"))
        dst = _clean_identifier(row.get("primary_contig_id"))
        if not src or not dst:
            continue
        src = strip_version(src)
        dst = strip_version(dst)
        alias_to_canonical[src] = dst
        sym = _clean_identifier(row.get("symbol"))
        if sym:
            canonical_to_symbol.setdefault(dst, sym)
    return alias_to_canonical, canonical_to_symbol


def _canonicalize_ensembl_gene_id(
    gene_id: str,
    *,
    symbol_hint: str | None = None,
) -> str | None:
    sid = strip_version(gene_id)
    alias_to_canonical, symbol_hints = _ensembl_alias_maps()
    mapped = alias_to_canonical.get(sid, sid)
    authority_ids = _authority_gene_ids()
    if not authority_ids:
        return mapped
    if mapped in authority_ids:
        return mapped

    candidate_symbols = [
        symbol_hint,
        symbol_hints.get(mapped),
        symbol_hints.get(sid),
        find_gene_name_from_ensembl_gene_id(mapped),
        find_gene_name_from_ensembl_gene_id(sid),
    ]
    seen: set[str] = set()
    for symbol in candidate_symbols:
        clean = _clean_identifier(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        target = _authority_gene_id_for_symbol(clean)
        if target:
            return target
    return None


@lru_cache(maxsize=1)
def _known_proteoform_ids() -> frozenset[str]:
    """Bundled synthetic proteoform keys across both reduced spaces."""
    try:
        from .expression.protein_groups import canonical_to_symbol

        return frozenset(
            set(canonical_to_symbol("protein").values())
            | set(canonical_to_symbol("cdna").values())
        )
    except Exception:
        return frozenset()


def _id_is_in_declared_space(gene_id: str) -> bool:
    authority_ids = _authority_gene_ids()
    if authority_ids:
        return gene_id in authority_ids
    return bool(_CANONICAL_ENSG_RE.match(gene_id))


@lru_cache(maxsize=None)
def _canonical_gene_id_cached(
    identifier: str,
    source_version: str | None,
    symbol_hint: str | None,
) -> str | None:
    text = _clean_identifier(identifier)
    if not text:
        return None

    if _ENSEMBL_GENE_RE.match(text):
        return _canonicalize_ensembl_gene_id(text, symbol_hint=symbol_hint)

    if _ENTREZ_RE.match(text):
        # Numeric source IDs are Entrez/NCBI GeneIDs in this codebase.  These
        # tables live under builders because ingest paths are their main user;
        # import lazily so symbol/ENSG callers never trigger the cache or any
        # mirror fetch.
        try:
            from .builders.ncbi_gene_info import (
                cached_entrez_history,
                cached_entrez_to_ensembl,
                cached_entrez_to_symbol,
            )
        except Exception:
            return None

        def _via_entrez(eid: str) -> str | None:
            ensg = cached_entrez_to_ensembl().get(eid)
            if ensg:
                return _canonicalize_ensembl_gene_id(ensg, symbol_hint=symbol_hint)
            symbol = cached_entrez_to_symbol().get(eid)
            if symbol:
                return _canonical_gene_id_cached(
                    symbol, source_version, symbol_hint
                )
            return None

        found = _via_entrez(text)
        if found:
            return found
        live = cached_entrez_history().get(text)
        return _via_entrez(live) if live else None

    authority_id = _authority_gene_id_for_symbol(text)
    if authority_id:
        return authority_id

    resolved = find_gene_and_ensembl_release_by_name(text)
    if resolved is None:
        return None
    _genome, gene = resolved
    gene_id = _gene_id_from_pyensembl(gene)
    return _canonicalize_ensembl_gene_id(gene_id, symbol_hint=text) if gene_id else None


def canonical_gene_id(
    identifier,
    *,
    source_version: str | None = None,
    symbol_hint: str | None = None,
    strict: bool = False,
) -> str | None:
    """Map a source gene identifier to the canonical unversioned ENSG.

    ``identifier`` may be an Ensembl gene ID, HUGO/HGNC symbol, curated display
    alias, NCBI synonym, or numeric Entrez/NCBI GeneID.  The canonical space is
    pinned to Ensembl release 112 when that authority release is installed.
    ``symbol_hint`` lets table canonicalization rescue retired Ensembl IDs whose
    stable-ID history is absent but whose row symbol maps uniquely into release
    112.

    ``source_version`` is accepted as the per-source provenance hook; the
    current packaged map is global, but callers should pass it now so the call
    sites are ready for a release-specific stable-ID-history table.
    """
    source_key = _clean_identifier(source_version) or None
    hint_key = _clean_identifier(symbol_hint) or None
    found = _canonical_gene_id_cached(
        _clean_identifier(identifier), source_key, hint_key
    )
    if not strict or found is None:
        return found
    if _id_is_in_declared_space(found):
        return found
    return None


@lru_cache(maxsize=None)
def canonical_gene_symbol(ensembl_gene_id: str, fallback: str | None = None) -> str:
    """Return one display symbol for a canonical ENSG."""
    gene_id = canonical_gene_id(ensembl_gene_id) or _clean_identifier(ensembl_gene_id)
    if not gene_id:
        return _clean_identifier(fallback)
    name = find_gene_name_from_ensembl_gene_id(gene_id)
    if name:
        return name
    _, symbol_hints = _ensembl_alias_maps()
    if gene_id in symbol_hints:
        return symbol_hints[gene_id]
    clean_fallback = _clean_identifier(fallback)
    if clean_fallback and not _ENSEMBL_GENE_RE.match(clean_fallback):
        return clean_fallback
    return gene_id


def canonical_gene_id_map() -> pd.DataFrame:
    """Versioned table of bundled source-ID aliases to canonical ENSGs."""
    try:
        df = get_data("ensembl-id-aliases")
    except Exception:
        return pd.DataFrame(
            columns=[
                "map_version",
                "source_identifier",
                "source_identifier_type",
                "source_version",
                "canonical_gene_id",
                "canonical_symbol",
                "mapping_source",
            ]
        )
    rows = []
    for _, row in df.iterrows():
        src = _clean_identifier(row.get("alt_haplotype_id"))
        dst = _clean_identifier(row.get("primary_contig_id"))
        if not src or not dst:
            continue
        fallback_symbol = _clean_identifier(row.get("symbol"))
        canon = _canonicalize_ensembl_gene_id(dst, symbol_hint=fallback_symbol)
        if canon is None:
            continue
        rows.append(
            {
                "map_version": CANONICAL_GENE_MAP_VERSION,
                "source_identifier": strip_version(src),
                "source_identifier_type": "ensembl_gene_id",
                "source_version": "",
                "canonical_gene_id": canon,
                "canonical_symbol": canonical_gene_symbol(canon, fallback=fallback_symbol),
                "mapping_source": _clean_identifier(row.get("source")),
            }
        )
    return pd.DataFrame(rows)


def canonical_gene_space_report(
    df: pd.DataFrame,
    *,
    id_col: str = "Ensembl_Gene_ID",
    symbol_col: str | None = "Symbol",
    context_cols: Sequence[str] = (),
    allow_proteoform_ids: bool = False,
) -> GeneIdentitySpaceReport:
    """Audit whether ``df`` obeys the declared identity-space contract.

    Use this on any table before a cross-source join.  For gene-space tables,
    leave ``allow_proteoform_ids=False`` so synthetic proteoform keys are caught
    if they leak into ``Ensembl_Gene_ID``.  For a deliberately collapsed
    proteoform frame, pass ``allow_proteoform_ids=True``.
    """
    if id_col not in df.columns:
        raise ValueError(f"canonical_gene_space_report needs an {id_col!r} column")
    missing_context = [c for c in context_cols if c not in df.columns]
    if missing_context:
        raise ValueError(f"context columns not present: {missing_context!r}")

    ids = df[id_col].map(_clean_identifier)
    nonempty = ids.ne("")
    versioned = ids.str.contains(r"\.\d+$", regex=True, na=False)
    authority_ids = _authority_gene_ids()
    if authority_ids:
        valid_gene_ids = ids.isin(authority_ids)
    else:
        valid_gene_ids = ids.str.match(_CANONICAL_ENSG_RE, na=False)
    if allow_proteoform_ids:
        known_proteoforms = _known_proteoform_ids()
        invalid = ~(
            nonempty
            & (
                valid_gene_ids
                | ids.isin(known_proteoforms)
            )
        )
    else:
        invalid = ~(nonempty & valid_gene_ids)
    invalid |= versioned

    work = df.copy()
    work["_canonical_report_id"] = ids
    key_cols = ["_canonical_report_id", *context_cols]
    dup_sizes = work.groupby(key_cols, dropna=False).size()
    dup_sizes = dup_sizes[dup_sizes > 1]
    n_duplicate_key_rows = int((dup_sizes - 1).sum())

    duplicate_examples: list[tuple[str, ...]] = []
    for key in dup_sizes.head(5).index:
        if not isinstance(key, tuple):
            key = (key,)
        duplicate_examples.append(tuple(str(x) for x in key))

    n_symbol_fallback_ids = 0
    symbol_fallback_examples: tuple[str, ...] = ()
    n_multi_symbol_gene_ids = 0
    multi_symbol_examples: tuple[str, ...] = ()
    if symbol_col is not None and symbol_col in df.columns:
        symbols = df[symbol_col].map(_clean_identifier)
        symbol_fallback = symbols.eq(ids) & ids.str.match(_CANONICAL_ENSG_RE, na=False)
        n_symbol_fallback_ids = int(symbol_fallback.sum())
        symbol_fallback_examples = tuple(ids[symbol_fallback].drop_duplicates().head(5))

        sym_work = pd.DataFrame({"id": ids, "symbol": symbols})
        sym_work = sym_work[
            sym_work["id"].ne("")
            & sym_work["symbol"].ne("")
            & ~sym_work["symbol"].str.match(_ENSEMBL_GENE_RE, na=False)
        ]
        multi = sym_work.groupby("id")["symbol"].nunique()
        multi = multi[multi > 1]
        n_multi_symbol_gene_ids = int(len(multi))
        multi_symbol_examples = tuple(multi.head(5).index.astype(str))

    return GeneIdentitySpaceReport(
        n_rows=int(len(df)),
        n_gene_ids=int(ids[nonempty].nunique()),
        n_duplicate_key_rows=n_duplicate_key_rows,
        n_invalid_ids=int(invalid.sum()),
        n_versioned_ids=int(versioned.sum()),
        n_symbol_fallback_ids=n_symbol_fallback_ids,
        n_multi_symbol_gene_ids=n_multi_symbol_gene_ids,
        duplicate_examples=tuple(duplicate_examples),
        invalid_id_examples=tuple(ids[invalid].drop_duplicates().head(5)),
        versioned_id_examples=tuple(ids[versioned].drop_duplicates().head(5)),
        symbol_fallback_examples=symbol_fallback_examples,
        multi_symbol_examples=multi_symbol_examples,
    )


def validate_canonical_gene_table(
    df: pd.DataFrame,
    *,
    id_col: str = "Ensembl_Gene_ID",
    symbol_col: str | None = "Symbol",
    context_cols: Sequence[str] = (),
    allow_proteoform_ids: bool = False,
    forbid_symbol_fallback_ids: bool = False,
) -> GeneIdentitySpaceReport:
    """Return a report or raise if ``df`` violates canonical-space semantics."""
    report = canonical_gene_space_report(
        df,
        id_col=id_col,
        symbol_col=symbol_col,
        context_cols=context_cols,
        allow_proteoform_ids=allow_proteoform_ids,
    )
    messages = report.error_messages(
        forbid_symbol_fallback_ids=forbid_symbol_fallback_ids
    )
    if messages:
        raise GeneIdentitySpaceViolation(
            "canonical gene identity contract violated: " + "; ".join(messages)
        )
    return report


def _existing_symbol_by_gene(
    df: pd.DataFrame,
    *,
    id_col: str,
    symbol_col: str | None,
) -> dict[str, str]:
    if symbol_col is None or symbol_col not in df.columns:
        return {}
    out: dict[str, str] = {}
    for gid, sym in zip(df[id_col].astype(str), df[symbol_col]):
        clean = _clean_identifier(sym)
        if not clean or _ENSEMBL_GENE_RE.match(clean):
            continue
        out.setdefault(gid, clean)
    return out


def canonicalize_gene_table(
    df: pd.DataFrame,
    *,
    id_col: str = "Ensembl_Gene_ID",
    symbol_col: str | None = "Symbol",
    source_version_col: str | None = "source_version",
    group_keys: Sequence[str] = (),
    value_cols: Sequence[str] | None = None,
    max_cols: Sequence[str] = (),
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """Canonicalize and collapse an expression-like table by gene identity.

    Rows are keyed by canonical ENSG, not by ``(ENSG, Symbol)`` pairs.  When
    several source rows map to the same canonical gene within the same
    ``group_keys`` context, numeric ``value_cols`` are summed in linear space
    with ``min_count=1`` so all-missing groups stay missing.  Other columns are
    taken from the first source row in original order.
    """
    if df.empty:
        return df.copy()
    if id_col not in df.columns:
        raise ValueError(f"canonicalize_gene_table needs an {id_col!r} column")

    work = df.reset_index(drop=True).copy()
    work["_canonical_ord"] = range(len(work))
    source_versions = (
        work[source_version_col].tolist()
        if source_version_col and source_version_col in work.columns
        else [None] * len(work)
    )
    symbols = (
        work[symbol_col].tolist()
        if symbol_col is not None and symbol_col in work.columns
        else [None] * len(work)
    )

    # Resolve each DISTINCT (id, symbol, source_version) combination once instead
    # of once per row.  The cohort long form is ~9.4M rows, and a per-row
    # canonical_gene_id() call made cohort_expression_views ~34x slower (#465);
    # memoizing on the exact key keeps behaviour identical while collapsing the
    # work to the ~10^5 distinct combinations actually present.
    unresolved = object()
    resolution_cache: dict[tuple, "str | None"] = {}
    canonical_ids = []
    for raw, sym, source_version in zip(
        work[id_col].tolist(), symbols, source_versions
    ):
        key = (raw, sym, source_version)
        gene_id = resolution_cache.get(key, unresolved)
        if gene_id is unresolved:
            gene_id = canonical_gene_id(
                raw, source_version=source_version, symbol_hint=sym
            )
            if gene_id is None and sym is not None:
                gene_id = canonical_gene_id(sym, source_version=source_version)
            resolution_cache[key] = gene_id
        canonical_ids.append(gene_id)
    work[id_col] = canonical_ids
    if drop_unmapped:
        work = work[work[id_col].notna()].copy()
    else:
        missing = work[id_col].isna()
        fallback_ids = df.reset_index(drop=True).loc[missing, id_col].map(
            _clean_identifier
        )
        work.loc[missing, id_col] = fallback_ids
    if work.empty:
        return df.iloc[0:0].copy()

    if value_cols is None:
        excluded = {id_col, "_canonical_ord", *group_keys}
        if symbol_col:
            excluded.add(symbol_col)
        value_cols = [
            c for c in work.select_dtypes(include="number").columns
            if c not in excluded
        ]
    present_sum = [c for c in value_cols if c in work.columns]
    present_max = [c for c in max_cols if c in work.columns]

    full_keys = [id_col, *group_keys]
    existing_symbols = _existing_symbol_by_gene(
        work, id_col=id_col, symbol_col=symbol_col,
    )
    rep = (
        work.sort_values(full_keys + ["_canonical_ord"])
        .drop_duplicates(full_keys, keep="first")
        .drop(columns=present_sum + present_max, errors="ignore")
    )
    out = rep
    if present_sum:
        out = out.merge(
            work.groupby(full_keys, as_index=False)[present_sum].sum(min_count=1),
            on=full_keys,
            how="left",
        )
    if present_max:
        out = out.merge(
            work.groupby(full_keys, as_index=False)[present_max].max(),
            on=full_keys,
            how="left",
        )

    if symbol_col is not None and symbol_col in out.columns:
        out[symbol_col] = [
            canonical_gene_symbol(gid, fallback=existing_symbols.get(str(gid)))
            for gid in out[id_col].astype(str)
        ]
    keep_cols = list(df.columns)
    result = (
        out.sort_values("_canonical_ord")
        .reset_index(drop=True)[keep_cols]
    )
    validate_canonical_gene_table(
        result,
        id_col=id_col,
        symbol_col=symbol_col,
        context_cols=group_keys,
    )
    return result


def canonical_proteoform_id(
    ensembl_gene_id: str,
    *,
    kind: str = "protein",
) -> str | None:
    """Map a canonical gene ENSG to the selected proteoform key.

    ``kind='protein'`` collapses byte-identical proteins; ``kind='cdna'`` uses
    the read-recovery cDNA-identical space.  Ungrouped genes return their ENSG.
    """
    gene_id = canonical_gene_id(ensembl_gene_id)
    if gene_id is None:
        return None
    from .expression.protein_groups import fold_ids

    return fold_ids([gene_id], kind=kind)[0]


def canonical_proteoform_id_map(kind: str = "protein") -> pd.DataFrame:
    """Versioned source-gene to proteoform map for a proteoform space."""
    from .expression.protein_groups import canonical_to_symbol, member_to_canonical

    if kind not in {"protein", "cdna"}:
        raise ValueError("kind must be 'protein' or 'cdna'")
    m2c = member_to_canonical(kind)
    c2s = canonical_to_symbol(kind)
    rows = []
    for member, canon in sorted(m2c.items()):
        rows.append(
            {
                "map_version": CANONICAL_PROTEOFORM_MAP_VERSION,
                "proteoform_kind": kind,
                "canonical_gene_id": member,
                "proteoform_id": c2s.get(canon, canon),
                "proteoform_canonical_gene_id": canon,
            }
        )
    return pd.DataFrame(rows)


def canonicalize_gene_ids(
    identifiers: Iterable,
    *,
    source_version: str | None = None,
    strict: bool = False,
) -> list[str | None]:
    """Vector-friendly wrapper around :func:`canonical_gene_id`."""
    return [
        canonical_gene_id(x, source_version=source_version, strict=strict)
        for x in identifiers
    ]


__all__ = [
    "CANONICAL_GENE_MAP_VERSION",
    "CANONICAL_ENSEMBL_RELEASE",
    "CANONICAL_PROTEOFORM_MAP_VERSION",
    "GeneIdentitySpaceReport",
    "GeneIdentitySpaceViolation",
    "canonical_gene_id",
    "canonical_gene_symbol",
    "canonical_gene_id_map",
    "canonical_gene_space_report",
    "canonicalize_gene_table",
    "canonicalize_gene_ids",
    "canonical_proteoform_id",
    "canonical_proteoform_id_map",
    "validate_canonical_gene_table",
]
