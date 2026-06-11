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
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from pirlygenes.load_dataset import get_data


@lru_cache(maxsize=1)
def protein_identical_groups() -> pd.DataFrame:
    """The derived protein-identical gene-group table (one row per member)."""
    return get_data("protein-identical-gene-groups")


@lru_cache(maxsize=1)
def _member_to_canonical() -> dict[str, str]:
    df = protein_identical_groups()
    return dict(zip(df["ensembl_gene_id"].astype(str),
                    df["group_canonical_ensembl_gene_id"].astype(str)))


@lru_cache(maxsize=1)
def _canonical_id_to_symbol() -> dict[str, str]:
    df = protein_identical_groups()
    return dict(zip(df["group_canonical_ensembl_gene_id"].astype(str),
                    df["group_canonical_symbol"].astype(str)))


@lru_cache(maxsize=1)
def canonical_symbol_map() -> dict[str, str]:
    """``{member_symbol_upper: group_canonical_symbol}`` for folding a gene-symbol
    set (e.g. a CTA panel) onto its protein-identical representatives, so a
    panel and a collapsed matrix agree on which symbol carries the group."""
    df = protein_identical_groups()
    out = {}
    for sym, canon in zip(df["symbol"], df["group_canonical_symbol"]):
        s = str(sym).strip().upper()
        if s:
            out[s] = str(canon)
    return out


def fold_symbols_to_canonical(symbols) -> list[str]:
    """Map each symbol to its protein-identical group's canonical symbol
    (identity if ungrouped) and de-duplicate, preserving order. Use to collapse
    a panel (CTA, lineage, …) the same way the matrix was collapsed."""
    m = canonical_symbol_map()
    seen, out = set(), []
    for s in symbols:
        c = m.get(str(s).strip().upper(), s)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


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
) -> pd.DataFrame:
    """Collapse protein-identical loci in a **long** table (one row per gene per
    context).

    ``group_keys`` are the columns that, together with the gene, identify a row
    (e.g. ``["cancer_code", "source_cohort", "normalization"]``). Within each
    (protein-group canonical id, ``*group_keys``) the ``sum_cols`` are summed in
    **linear space** (``min_count=1``: ``NaN`` members ignored, all-``NaN`` stays
    ``NaN``); ``max_cols`` take the max (e.g. a per-gene detection count); every
    other column is taken from the canonical member's row. The merged row is
    keyed by the group's canonical Ensembl id + symbol. Genes in no multi-member
    group are unchanged. Returns a new long DataFrame with the same columns.
    """
    if df.empty:
        return df
    m2c = _member_to_canonical()
    csym = _canonical_id_to_symbol()
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
    # key the merged row by the canonical id + symbol
    out[id_col] = out["_canon"]
    out[symbol_col] = [csym.get(c, s)
                       for c, s in zip(out["_canon"], out[symbol_col])]
    return out.sort_values("_ord").reset_index(drop=True)[list(df.columns)]


@lru_cache(maxsize=1)
def _cta_symbol_to_group() -> dict[str, str]:
    """``{member_symbol: proteoform_group}`` over the curated CTA protein groups
    (``cta-protein-groups``, identical/near-identical >=90% aa)."""
    g = get_data("cta-protein-groups")
    return {str(m): str(grp)
            for m, grp in zip(g["member_symbol"], g["protein_group"])}


def collapse_cta_proteoforms_long(
    df: pd.DataFrame,
    *,
    group_keys: list[str],
    sum_cols: list[str],
    id_col: str = "Ensembl_Gene_ID",
    symbol_col: str = "Symbol",
    max_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """CTA-scoped sibling of :func:`collapse_protein_identical_loci_long`.

    Sums the curated CTA proteoform-group members (``cta-protein-groups``,
    identical/near-identical **>=90% aa** — the CT47A cluster, MAGEA3/MAGEA6,
    NY-ESO CTAG1A/CTAG1B, …) in **linear space** per (proteoform group,
    ``*group_keys``); non-CTA genes pass through unchanged (only antigens roll
    up, never housekeeping clusters). The merged row keeps a **representative
    member**'s symbol + id (the member whose symbol equals the group key when the
    key is itself a symbol — e.g. ``MAGEA9`` — else the lowest-id member). The
    group key may be an *alias* (``NY-ESO-1``), so the merged row deliberately
    keeps a real member **symbol** (e.g. ``CTAG1B``), not the alias, so that
    ``cta_paralog_symbols`` / matrix-column lookups still resolve it. >=90%
    identity is the right grain for antigen abundance and catches paralogs a
    strict byte-identical rule misses (MAGEA3/6 are 95.9%).
    """
    if df.empty:
        return df
    s2g = _cta_symbol_to_group()
    work = df.reset_index(drop=True).copy()
    work["_ord"] = range(len(work))
    sym = work[symbol_col].astype(str)
    # CTA members -> their proteoform group; NON-CTA rows -> their own ENSG (NOT
    # their symbol), so a CTA member whose symbol maps to several ENSGs still
    # sums (HNRNPCL1/2 etc.), but two distinct-protein genes that merely SHARE a
    # symbol (e.g. HERC3's two ENSGs) are NOT collapsed together.
    work["_grp"] = sym.map(s2g).fillna(work[id_col].astype(str))
    work["_is_rep"] = sym == work["_grp"]
    full_keys = ["_grp", *group_keys]
    present_sum = [c for c in sum_cols if c in work.columns]
    present_max = [c for c in max_cols if c in work.columns]
    rep = (work.sort_values(full_keys + ["_is_rep", id_col],
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
    # NB: keep the representative member's symbol+id (do NOT relabel to _grp,
    # which may be a non-symbol alias like 'NY-ESO-1').
    return out.sort_values("_ord").reset_index(drop=True)[list(df.columns)]
