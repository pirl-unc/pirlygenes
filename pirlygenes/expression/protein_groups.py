"""Collapse protein-identical gene loci for protein-abundance proxies.

Two distinct Ensembl loci that encode the **identical protein** (segmental
-duplication paralogs, histone clusters, the CT47A cancer-testis cluster,
reassigned/duplicate gene models) each receive a *fraction* of the reads a
quantifier would otherwise assign to a single gene. When expression is read as
a **protein-abundance proxy** those fractions must be summed back together, or
the protein looks under-expressed and a per-gene threshold (e.g. CTA "ON"
counting) is distorted.

The protein-identical groups are derived genome-wide by
``scripts/generate_protein_identical_groups.py`` (grouping protein-coding genes
whose canonical/longest protein is byte-identical) into
``data/protein-identical-gene-groups.csv``. This module applies them:

- summing happens in **linear TPM space** (you must collapse before any
  log transform — sums of logs are meaningless);
- a member a source did **not** measure (``NaN``) is **ignored**, not treated
  as zero — the group total is the sum of the *present* members, and is ``NaN``
  only when every member is absent (``min_count=1``);
- the whole group collapses **consistently** to one row keyed by the group's
  canonical Ensembl id.

Groups are protein-identity only: paralog families that differ at the protein
level (PSG*, CSH*, most MAGE/GAGE members) are NOT collapsed.

Identifier contract (used consistently everywhere a collapse runs)
------------------------------------------------------------------
A row in a collapsed expression table is keyed by ``Ensembl_Gene_ID`` and named
by ``Symbol``:

- **Unique gene → protein (not folded):** the key is the gene's real **ENSG**
  and the name is its HGNC symbol. Untouched.
- **Folded proteoform (>=2 loci summed):** the member ENSGs LEAVE the key space
  and the merged row is keyed by a **proteoform ID** — the member symbols
  combined so the id shows exactly what was summed and is unique by construction
  (:func:`proteoform_id`: ``XAGE1A``+``XAGE1B`` -> ``XAGE1A/B``; ``CT47A1..12``
  -> ``CT47A1/2/.../12``). Both ``Ensembl_Gene_ID`` and ``Symbol`` carry this id.

So ``Ensembl_Gene_ID`` holds an ENSG xor a proteoform id; never a member locus
standing in for the group. The single id→display-name authority is
:func:`pirlygenes.gene_names.display_name`, which maps a gene symbol OR a
proteoform id to its label (``CTAG1A/B`` -> ``NY-ESO-1``; ``XAGE1A/B`` -> itself).

ONE canonical space, mapping up front
-------------------------------------
For a given collapse there is **one** canonical space (the proteoform space —
cDNA-identical for the read-recovery ``collapse_cdna_identical`` matrix, or
protein-identical for the protein-abundance fold; pick one per analysis). The
rule that prevents the recurring "panel in member space vs data in proteoform
space" bug: **map every identifier into that one space UP FRONT, never compare
raw symbols to collapsed data.** The fold functions are that boundary mapper and
resolve synonyms up front:

- :func:`fold_symbols` (``Symbol``) and :func:`fold_ids` (``Ensembl_Gene_ID``)
  fold a panel onto a space's collapsed key, selected by ``kind='cdna'`` (default)
  or ``'protein'``. The symbol fold also resolves a grouped member's curated
  **display alias** (``NY-ESO-1`` -> ``CTAG1A/B``) so a display-named panel lands
  in the same space.
- the ENSG fold is robust to symbol renames (stable accessions); use it when a
  panel carries old/renamed symbols.
- arbitrary NCBI synonyms normalise to an official symbol via the single resolver
  :func:`pirlygenes.gene_ids.find_gene_and_ensembl_release_by_name` first.

Curated panels ship pre-folded — e.g.
:func:`pirlygenes.gene_sets_cancer.CTA_proteoform_symbols` /
:func:`~pirlygenes.gene_sets_cancer.CTA_proteoform_ids` — so a consumer never has
to fold at the selection site.

ONE parameterized core
----------------------
Both proteoform spaces flow through ONE ``kind``-parameterized core, so there is a
single implementation of each concept rather than a per-space copy:

- maps: :func:`member_to_canonical`, :func:`canonical_to_symbol`,
  :func:`symbol_to_canonical` (each ``kind='cdna'|'protein'``);
- folds: :func:`fold_symbols`, :func:`fold_ids` (same ``kind``).

The older per-space names (``fold_to_cdna_canonical_symbol``,
``fold_symbols_to_canonical``, ``cdna_member_to_canonical``,
``protein_member_to_canonical``, …) are kept as one-line aliases over this core
for back-compat (incl. external consumers like trufflepig); prefer the ``kind=``
functions in new code.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import lru_cache

import pandas as pd

from pirlygenes.load_dataset import get_data

def _natural_key(s):
    """Sort key that orders 'CT47A2' before 'CT47A10' (numeric chunks compared
    as ints), so a folded proteoform ID lists its members in human order."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def proteoform_id(member_symbols):
    """The identifier for a folded proteoform group — one row of expression
    summed across **>=2 distinct ENSG loci** that the collapse removes from the
    key space. It is built from the member gene symbols so it shows exactly what
    was merged and is unique by construction (no two groups share a member set):

    - all members carry the **same** symbol -> that symbol (e.g. two ``FOO``
      loci -> ``FOO``);
    - members share a common prefix -> factor it out and slash the distinct
      suffixes (``CTAG1A``+``CTAG1B`` -> ``CTAG1A/B``; ``XAGE1A``+``XAGE1B`` ->
      ``XAGE1A/B``; ``CT47A1..CT47A12`` -> ``CT47A1/2/.../12``);
    - no shared prefix -> slash the full symbols (``GENEA`` + ``GENEB`` ->
      ``GENEA/GENEB``).

    A single-locus gene is NOT a proteoform group: it keeps its own ENSG as the
    key and its symbol as the name (this function is only called for >=2 loci).
    Returns ``None`` if no member carries a symbol. The human-readable label for
    the result comes from :func:`pirlygenes.gene_names.display_name` (e.g.
    ``CTAG1A/B`` -> ``NY-ESO-1``).
    """
    syms = []
    for s in member_symbols:
        s = str(s).strip()
        if s and s.lower() != "nan" and s not in syms:
            syms.append(s)
    if not syms:
        return None
    syms.sort(key=_natural_key)
    if len(syms) == 1:
        return syms[0]
    stem = os.path.commonprefix(syms)
    # Don't split inside a multi-digit number: if the common prefix ends mid-run
    # (its last char is a digit and a member continues with another digit, e.g.
    # PRAMEF25/PRAMEF26 -> prefix 'PRAMEF2'), back it up to the start of that
    # trailing digit run so the suffixes stay whole numbers (-> 'PRAMEF25/26').
    if stem and stem[-1].isdigit() and any(
            len(s) > len(stem) and s[len(stem)].isdigit() for s in syms):
        stem = stem.rstrip("0123456789")
    suffixes = [s[len(stem):] for s in syms]
    if stem and all(suffixes):
        return stem + "/".join(suffixes)
    return "/".join(syms)


@lru_cache(maxsize=1)
def protein_identical_groups() -> pd.DataFrame:
    """The derived protein-identical gene-group table (one row per member)."""
    return get_data("protein-identical-gene-groups")


def _with_display_aliases(member_map: dict) -> dict:
    """Augment a ``{member_symbol_upper: canonical}`` fold map with the display
    alias of each **grouped member**, so a panel named in display space lands in
    the SAME proteoform space up front: ``NY-ESO-1`` -> ``CTAG1A/B``.

    Only grouped members are aliased. Single-locus display nicknames are NOT
    synthesised: ``gene_names.aliases`` is a display map whose direction isn't
    guaranteed — a stray ``old_symbol -> current_symbol`` entry (like the removed
    ``PVRL4 -> NECTIN4``) would, under a blind ``display -> official`` map, remap a
    real current symbol onto a dead one. A grouped member's symbol is always the
    current Ensembl symbol, so restricting to grouped members keeps the alias
    direction unambiguous and is robust even if such an entry were re-added.
    Single-locus official symbols fold via passthrough, and arbitrary synonyms go
    through :func:`pirlygenes.gene_ids.find_gene_and_ensembl_release_by_name`."""
    from ..gene_names import aliases as _display_aliases
    out = dict(member_map)
    for official, display in _display_aliases.items():
        off_u = str(official).strip().upper()
        if off_u in member_map:                       # grouped -> real proteoform ID
            out.setdefault(str(display).strip().upper(), member_map[off_u])
    return out


# ============================================================================
# Proteoform spaces — ONE parameterized core
# ============================================================================
# A proteoform "space" is one identity criterion for collapsing loci:
#   'cdna'    byte-identical canonical CDS — a quantifier can't assign reads
#             between members, so only the SUM is reliable (the read-recovery
#             collapse; the matrix's collapse_cdna_identical). A small curated
#             override (proteoform-collapse-overrides) force-collapses a few
#             protein-identical / cDNA-distinct antigens (the CT47A cluster).
#   'protein' byte-identical canonical protein — the protein-abundance collapse
#             (collapse_protein_identical). No override.
# Each space provides three maps; fold_symbols / fold_ids and the collapse
# helpers are thin layers on top. Pick ONE space per analysis and map both panels
# and data into it (see the "ONE canonical space" note in the module docstring).
# The per-space named functions further down are one-line aliases over this core,
# kept for back-compat (incl. external consumers like trufflepig).


def _space_groups(kind: str) -> pd.DataFrame:
    return cdna_identical_groups() if kind == "cdna" else protein_identical_groups()


@lru_cache(maxsize=None)
def member_to_canonical(kind: str = "cdna") -> dict[str, str]:
    """``{member_ensg: canonical_ensg}`` for a proteoform space. The 'cdna' space
    applies the curated overrides last (they force-collapse a whole protein group,
    superseding any cDNA-subgroup split)."""
    df = _space_groups(kind)
    m = dict(zip(df["ensembl_gene_id"].astype(str),
                 df["group_canonical_ensembl_gene_id"].astype(str)))
    if kind == "cdna":
        overrides = set(get_data("proteoform-collapse-overrides")
                        ["group_canonical_ensembl_gene_id"].astype(str))
        if overrides:
            pg = protein_identical_groups()
            for canon, sub in pg.groupby("group_canonical_ensembl_gene_id"):
                if str(canon) in overrides:
                    for member in sub["ensembl_gene_id"].astype(str):
                        m[member] = str(canon)
    return m


@lru_cache(maxsize=None)
def canonical_to_symbol(kind: str = "cdna") -> dict[str, str]:
    """``{canonical_ensg: proteoform_symbol}`` for a space (the 'cdna' space
    applies the override's symbol)."""
    df = _space_groups(kind)
    out = dict(zip(df["group_canonical_ensembl_gene_id"].astype(str),
                   df["group_canonical_symbol"].astype(str)))
    if kind == "cdna":
        ov = get_data("proteoform-collapse-overrides")
        out.update(dict(zip(ov["group_canonical_ensembl_gene_id"].astype(str),
                            ov["group_symbol"].astype(str))))
    return out


@lru_cache(maxsize=None)
def symbol_to_canonical(kind: str = "cdna") -> dict[str, str]:
    """``{member_symbol_upper: proteoform_symbol}`` (+ display aliases) for a
    space — fold a symbol-keyed panel/matrix onto its proteoform symbols. The
    'cdna' space scans both group tables so override members (which live in the
    protein table) are caught."""
    m2c, c2s = member_to_canonical(kind), canonical_to_symbol(kind)
    tables = ((cdna_identical_groups(), protein_identical_groups())
              if kind == "cdna" else (protein_identical_groups(),))
    out = {}
    for df in tables:
        for sym, ensg in zip(df["symbol"], df["ensembl_gene_id"].astype(str)):
            s = str(sym).strip().upper()
            if s and ensg in m2c:
                out[s] = c2s.get(m2c[ensg], sym)
    return _with_display_aliases(out)


def _dedup(items):
    """Order-preserving de-duplication."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def fold_symbols(symbols, *, kind: str = "cdna") -> list[str]:
    """Fold a symbol panel onto a space's proteoform symbols (identity if
    ungrouped), de-duplicated + order-preserving. Match against a collapsed
    frame's ``Symbol`` column."""
    m = symbol_to_canonical(kind)
    return _dedup(m.get(str(s).strip().upper(), s) for s in symbols)


def fold_ids(ensembl_ids, *, kind: str = "cdna") -> list[str]:
    """Fold an ENSG panel onto a space's collapsed key — the group's proteoform ID
    if grouped, else its own version-stripped ENSG. Match against the
    ``Ensembl_Gene_ID`` column of the matching collapsed frame."""
    return _fold_ids(ensembl_ids, member_to_canonical(kind),
                     canonical_to_symbol(kind))


# ---- back-compat aliases over the core (prefer the kind= API above) ----------
def _member_to_canonical() -> dict[str, str]:
    return member_to_canonical("protein")


def _canonical_id_to_symbol() -> dict[str, str]:
    return canonical_to_symbol("protein")


def canonical_symbol_map() -> dict[str, str]:
    """Alias: protein-space :func:`symbol_to_canonical` (``kind='protein'``)."""
    return symbol_to_canonical("protein")


def fold_symbols_to_canonical(symbols) -> list[str]:
    """Alias: :func:`fold_symbols` with ``kind='protein'``."""
    return fold_symbols(symbols, kind="protein")


def _strip_version(ensg: str) -> str:
    return str(ensg).split(".")[0]


def collapse_protein_identical_loci(
    df: pd.DataFrame,
    *,
    id_col: str = "Ensembl_Gene_ID",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Sum protein-identical loci in a **linear-space** (TPM/nTPM) wide matrix.

    ``df`` is a gene table with an Ensembl-id column ``id_col`` and one column
    per sample/tissue/cohort of linear expression. Rows whose (version-stripped)
    Ensembl id falls in the same protein-identical group — and where **>= 2**
    members are actually present in ``df`` — are summed into a single row keyed
    by the group's canonical id (its other identifier columns are taken from the
    canonical member if present, else the lowest-accession present member).

    Summation is ``min_count=1``: a member that is ``NaN`` in a column is
    ignored, and the result is ``NaN`` only if every present member is ``NaN``
    there. Genes in no multi-member group pass through unchanged. Row order
    follows the first original appearance of each group's representative.

    Returns a new collapsed DataFrame with the same columns as ``df``.
    """
    if id_col not in df.columns:
        raise ValueError(f"collapse_protein_identical_loci needs an {id_col!r} column")
    m2c = _member_to_canonical()
    work = df.reset_index(drop=True).copy()
    work["_ord"] = range(len(work))
    sid = work[id_col].map(_strip_version)
    # group key = canonical id for grouped genes, else the gene's own id
    work["_gkey"] = sid.map(m2c).fillna(sid)
    work["_is_canon"] = sid == work["_gkey"]

    if value_cols is None:
        value_cols = work.select_dtypes(include="number").columns.tolist()
        for helper in ("_ord",):
            if helper in value_cols:
                value_cols.remove(helper)

    # representative row per group: prefer the canonical member, then lowest id
    rep = (work.sort_values(["_gkey", "_is_canon", id_col],
                            ascending=[True, False, True])
               .drop_duplicates("_gkey", keep="first")
               .set_index("_gkey"))
    summed = work.groupby("_gkey")[value_cols].sum(min_count=1)
    rep[value_cols] = summed
    out = (rep.sort_values("_ord")
              .reset_index(drop=True)[list(df.columns)])
    return out


def collapse_protein_identical_loci_long(
    df: pd.DataFrame,
    *,
    id_col: str = "Ensembl_Gene_ID",
    symbol_col: str = "Symbol",
    group_keys: list[str],
    sum_cols: list[str],
    max_cols: tuple[str, ...] = (),
    member_to_canonical: dict[str, str] | None = None,
    canonical_to_symbol: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Collapse identical loci in a **long** table (one row per gene per context).

    ``group_keys`` are the columns that, together with the gene, identify a row
    (e.g. ``["cancer_code", "source_cohort", "normalization"]``). Within each
    (group canonical id, ``*group_keys``) the ``sum_cols`` are summed in **linear
    space** (``min_count=1``: ``NaN`` members ignored, all-``NaN`` stays ``NaN``);
    ``max_cols`` take the max (e.g. a per-gene detection count); every other
    column is taken from the canonical member's row. The merged row is keyed by
    the group's canonical Ensembl id + symbol. Genes in no multi-member group are
    unchanged. Returns a new long DataFrame with the same columns.

    ``member_to_canonical`` / ``canonical_to_symbol`` override the default
    genome-wide **protein**-identical grouping — pass the cDNA-identical maps for
    the read-recovery collapse (:func:`collapse_cdna_identical_loci_long`).
    """
    if df.empty:
        return df
    m2c = member_to_canonical if member_to_canonical is not None \
        else _member_to_canonical()
    csym = canonical_to_symbol if canonical_to_symbol is not None \
        else _canonical_id_to_symbol()
    work = df.reset_index(drop=True).copy()
    work["_ord"] = range(len(work))
    sid = work[id_col].map(_strip_version)
    work["_canon"] = sid.map(m2c).fillna(sid)
    work["_is_canon"] = sid == work["_canon"]
    full_keys = ["_canon", *group_keys]

    present_sum = [c for c in sum_cols if c in work.columns]
    present_max = [c for c in max_cols if c in work.columns]

    rep = (work.sort_values(full_keys + ["_is_canon", id_col],
                            ascending=[True] * len(full_keys) + [False, True])
               .drop_duplicates(full_keys, keep="first")
               .drop(columns=present_sum + present_max, errors="ignore"))
    out = rep
    if present_sum:
        out = out.merge(
            work.groupby(full_keys, as_index=False)[present_sum].sum(min_count=1),
            on=full_keys, how="left")
    if present_max:
        out = out.merge(
            work.groupby(full_keys, as_index=False)[present_max].max(),
            on=full_keys, how="left")
    # Every member of a fold group leaves the ENSG key space and is keyed by the
    # group's proteoform ID (``csym[canon]`` — e.g. ``CTAG1A/B``) in both the id
    # and symbol columns, CONSISTENTLY across cohorts (whether 1 or N members were
    # present in a given context — the reads can't be told apart). Single-locus
    # genes (``_canon`` is their own ENSG, not a group key) keep their real ENSG.
    pid = out["_canon"].map(csym)
    out[id_col] = out[id_col].mask(pid.notna(), pid)
    out[symbol_col] = out[symbol_col].mask(pid.notna(), pid)
    # Dual-identifier contract: the proteoform ID is the key, but the real
    # constituent ENSGs stay reachable for ENSG-keyed consumers via the
    # ``Member_Ensembl_Gene_IDs`` column (";"-joined; the gene's own ENSG for a
    # single locus). Members come from the (canonical -> members) inverse of
    # ``m2c``, so it's the full group membership regardless of per-context presence.
    members = defaultdict(set)
    for member, canon in m2c.items():
        members[canon].add(member)
    members_str = {c: ";".join(sorted(ms)) for c, ms in members.items()}
    out["Member_Ensembl_Gene_IDs"] = [members_str.get(c, c) for c in out["_canon"]]
    keep_cols = list(df.columns)
    if "Member_Ensembl_Gene_IDs" not in keep_cols:
        keep_cols = keep_cols + ["Member_Ensembl_Gene_IDs"]
    return out.sort_values("_ord").reset_index(drop=True)[keep_cols]


# ---- cDNA-identical (read-recovery) collapse: the universal, principled one ----
#
# Two loci with byte-identical canonical CDS multi-map (a quantifier can't assign
# reads between them), so each is split / under-counted and only the SUM is
# reliable. This is the universal, transcriptome-wide collapse. A small curated
# override (``proteoform-collapse-overrides``) force-collapses a few 100%-protein
# /cDNA-distinct groups whose members should still be one entity (e.g. the CT47A
# antigen). NOT a >=90% near-identical grouping — distinct proteins (MAGEA3 vs
# MAGEA6) stay separate.


@lru_cache(maxsize=1)
def cdna_identical_groups() -> pd.DataFrame:
    """The derived cDNA-identical gene-group table (one row per member)."""
    return get_data("cdna-identical-gene-groups")


def _fold_ids(ensembl_ids, m2c, c2s) -> list[str]:
    """Inner ENSG fold under explicit (member->canonical) ``m2c`` + (canonical->
    proteoform symbol) ``c2s`` maps: the group's proteoform ID if grouped, else
    the version-stripped ENSG. Used by :func:`fold_ids` and the collapse path."""
    return _dedup(c2s.get(m2c.get(s, s), s)
                  for s in (_strip_version(str(e).strip()) for e in ensembl_ids))


# ---- back-compat aliases over the core (prefer the kind= API above) ----------
def _cdna_member_to_canonical() -> dict[str, str]:
    return member_to_canonical("cdna")


def _cdna_canonical_to_symbol() -> dict[str, str]:
    return canonical_to_symbol("cdna")


def _cdna_symbol_to_canonical_symbol() -> dict[str, str]:
    return symbol_to_canonical("cdna")


def fold_to_cdna_canonical_symbol(symbols) -> list[str]:
    """Alias: :func:`fold_symbols` with ``kind='cdna'`` — fold a panel the way the
    matrix's ``collapse_cdna_identical`` did (match the ``Symbol`` column)."""
    return fold_symbols(symbols, kind="cdna")


def fold_to_cdna_canonical_id(ensembl_ids) -> list[str]:
    """Alias: :func:`fold_ids` with ``kind='cdna'`` (match ``Ensembl_Gene_ID``)."""
    return fold_ids(ensembl_ids, kind="cdna")


def fold_to_protein_canonical_id(ensembl_ids) -> list[str]:
    """Alias: :func:`fold_ids` with ``kind='protein'``."""
    return fold_ids(ensembl_ids, kind="protein")


def cdna_member_to_canonical() -> dict[str, str]:
    """Alias: ``dict(member_to_canonical('cdna'))`` — a fresh copy for consumers
    that collapse an ENSG-indexed matrix directly (e.g. per-sample CTA matrices)."""
    return dict(member_to_canonical("cdna"))


def cdna_canonical_to_symbol() -> dict[str, str]:
    """Alias: ``dict(canonical_to_symbol('cdna'))``."""
    return dict(canonical_to_symbol("cdna"))


def protein_member_to_canonical() -> dict[str, str]:
    """Alias: ``dict(member_to_canonical('protein'))``."""
    return dict(member_to_canonical("protein"))


def protein_canonical_id_to_symbol() -> dict[str, str]:
    """Alias: ``dict(canonical_to_symbol('protein'))``."""
    return dict(canonical_to_symbol("protein"))


def cdna_symbol_to_canonical_symbol() -> dict[str, str]:
    """Alias: ``dict(symbol_to_canonical('cdna'))`` — for collapsing a symbol-keyed
    matrix (e.g. the percentile-reference transcriptome TSV)."""
    return dict(symbol_to_canonical("cdna"))


def collapse_cdna_identical_loci_long(
    df: pd.DataFrame,
    *,
    group_keys: list[str],
    sum_cols: list[str],
    id_col: str = "Ensembl_Gene_ID",
    symbol_col: str = "Symbol",
    max_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Sum cDNA-identical loci (+curated overrides) in a long table — the
    universal read-recovery collapse. Thin wrapper over
    :func:`collapse_protein_identical_loci_long` with the cDNA maps."""
    return collapse_protein_identical_loci_long(
        df, group_keys=group_keys, sum_cols=sum_cols, id_col=id_col,
        symbol_col=symbol_col, max_cols=max_cols,
        member_to_canonical=_cdna_member_to_canonical(),
        canonical_to_symbol=_cdna_canonical_to_symbol())
