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
