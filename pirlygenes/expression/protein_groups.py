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
"""

from __future__ import annotations

import os
import re
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
    suffixes = [s[len(stem):] for s in syms]
    if stem and all(suffixes):
        return stem + "/".join(suffixes)
    return "/".join(syms)


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
    # A row is a FOLDED proteoform only where >=2 member loci were present in
    # this context and summed; it then leaves the ENSG key space and is keyed by
    # the group's proteoform ID (``csym[canon]`` — e.g. ``CTAG1A/B``) in both the
    # id and symbol columns. Single-locus rows keep their real ENSG + symbol.
    sizes = work.groupby(full_keys).size().rename("_n").reset_index()
    out = out.merge(sizes, on=full_keys, how="left")
    pid = out["_canon"].map(csym)
    folded = (out["_n"] >= 2) & pid.notna()
    out[id_col] = out[id_col].mask(folded, pid)
    out[symbol_col] = out[symbol_col].mask(folded, pid)
    return out.sort_values("_ord").reset_index(drop=True)[list(df.columns)]


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


@lru_cache(maxsize=1)
def _cdna_member_to_canonical() -> dict[str, str]:
    """``{member_ensg: canonical_ensg}`` from the cDNA-identical groups, with the
    curated overrides applied last (they force-collapse whole protein-identical
    groups, superseding any cDNA-subgroup split)."""
    cd = cdna_identical_groups()
    m = dict(zip(cd["ensembl_gene_id"].astype(str),
                 cd["group_canonical_ensembl_gene_id"].astype(str)))
    overrides = set(get_data("proteoform-collapse-overrides")
                    ["group_canonical_ensembl_gene_id"].astype(str))
    if overrides:
        pg = protein_identical_groups()
        for canon, sub in pg.groupby("group_canonical_ensembl_gene_id"):
            if str(canon) in overrides:
                for member in sub["ensembl_gene_id"].astype(str):
                    m[member] = str(canon)
    return m


@lru_cache(maxsize=1)
def _cdna_canonical_to_symbol() -> dict[str, str]:
    cd = cdna_identical_groups()
    out = dict(zip(cd["group_canonical_ensembl_gene_id"].astype(str),
                   cd["group_canonical_symbol"].astype(str)))
    ov = get_data("proteoform-collapse-overrides")
    out.update(dict(zip(ov["group_canonical_ensembl_gene_id"].astype(str),
                        ov["group_symbol"].astype(str))))
    return out


@lru_cache(maxsize=1)
def _cdna_symbol_to_canonical_symbol() -> dict[str, str]:
    """``{member_symbol_upper: canonical_symbol}`` for folding a panel (e.g. CTA)
    onto the cDNA-collapsed matrix's canonical symbols."""
    m2c = _cdna_member_to_canonical()
    c2s = _cdna_canonical_to_symbol()
    cd = cdna_identical_groups()
    pg = protein_identical_groups()
    out = {}
    for df in (cd, pg):
        for sym, ensg in zip(df["symbol"], df["ensembl_gene_id"].astype(str)):
            s = str(sym).strip().upper()
            if s and ensg in m2c:
                out[s] = c2s.get(m2c[ensg], sym)
    return out


def fold_to_cdna_canonical_symbol(symbols) -> list[str]:
    """Map symbols onto their cDNA-identical (+override) canonical symbol and
    de-duplicate, preserving order — fold a panel the way the matrix collapsed."""
    m = _cdna_symbol_to_canonical_symbol()
    seen, out = set(), []
    for s in symbols:
        c = m.get(str(s).strip().upper(), s)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def cdna_member_to_canonical() -> dict[str, str]:
    """Public ``{member_ensg: canonical_ensg}`` for the cDNA-identical collapse
    (+ curated overrides) — for consumers that collapse an ENSG-indexed matrix
    directly (e.g. per-sample CTA matrices)."""
    return dict(_cdna_member_to_canonical())


def cdna_canonical_to_symbol() -> dict[str, str]:
    """Public ``{canonical_ensg: canonical_symbol}`` companion to
    :func:`cdna_member_to_canonical`."""
    return dict(_cdna_canonical_to_symbol())


def cdna_symbol_to_canonical_symbol() -> dict[str, str]:
    """Public ``{member_symbol_upper: canonical_symbol}`` for collapsing a
    **symbol**-keyed matrix (e.g. the percentile-reference transcriptome TSV) so
    a cDNA-identical group is one entry in the ranking universe."""
    return dict(_cdna_symbol_to_canonical_symbol())


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
