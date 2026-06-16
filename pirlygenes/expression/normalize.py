"""Expression matrix transforms — distinct from QC classification.

QC (:mod:`pirlygenes.expression.qc`) answers *which features are
usable*. This module answers *how to rescale what survives* and how to
convert between common quantification scales:

- :func:`normalize_expression` — zero technical-RNA features (mtDNA,
  rRNA-like, mt-pseudogenes) and renormalize the remaining mass back to
  the original per-column total so a comparison is not driven by
  technical-RNA denominator drift.
- :func:`fpkm_to_tpm` — rescale each expression column so it sums to
  10⁶, the standard TPM convention. Combine with
  :func:`normalize_expression` to get the analysis-view matrix.
- :func:`add_tpm_columns_from_fpkm` — append TPM-scale companion columns
  while preserving raw FPKM provenance columns.
- :func:`percentile_rank_expression` — map expression columns to
  within-column percentile ranks.
- :func:`tpm_to_housekeeping_normalized` — divide each column by the
  geometric mean of a curated housekeeping panel, producing a unit-free
  ratio scale that survives library-prep totals.
- :func:`renormalize_to_million` — bare utility: rescale columns to
  sum to 10⁶ without dropping anything.

The technical-RNA group definition lives in :mod:`.qc` and is imported
from there; this module never reclassifies genes.
"""

from __future__ import annotations

import functools
from typing import Iterable

from .qc import (
    OTHER_TECHNICAL_FRACTION,
    RIBOSOMAL_PROTEIN_FRACTION,
    TECHNICAL_FRACTION,
    TECHNICAL_RNA_GROUPS,
    classify_gene_qc,
)


# Default removal panel. Mirrors :data:`expression_qc.TECHNICAL_RNA_GROUPS`
# — mtDNA, NUMT-like pseudogenes, nuclear rRNA-like, and the
# polyadenylation-bias lncRNA panel (MALAT1, NEAT1). See the
# expression_qc module for the per-group citations.
_DEFAULT_NORMALIZE_REMOVE_GROUPS = TECHNICAL_RNA_GROUPS
_VALUE_COL_PREFIXES = ("TPM", "nTPM_", "FPKM_")
_VALUE_COL_SUFFIXES = (
    "_TPM",
    "_nTPM",
    "_FPKM",
    "_TPM_log1p",
    "_nTPM_log1p",
    "_TPM_clean",
    "_nTPM_clean",
    "_TPM_clean_log1p",
    "_nTPM_clean_log1p",
    "_TPM_hk",
    "_nTPM_hk",
    "_TPM_percentile",
    "_nTPM_percentile",
)
_RAW_VALUE_COL_PREFIXES = ("TPM_raw_", "nTPM_raw_")
_RAW_VALUE_COL_SUFFIXES = ("_TPM_raw", "_nTPM_raw")


def _is_expression_value_col(col: object) -> bool:
    name = str(col)
    return (
        name.startswith(_VALUE_COL_PREFIXES)
        or name.endswith(_VALUE_COL_SUFFIXES)
    ) and not (
        name.startswith(_RAW_VALUE_COL_PREFIXES)
        or name.endswith(_RAW_VALUE_COL_SUFFIXES)
    )


_KEEP_NONCODING_NORMALIZATION_BIOTYPES = frozenset(
    {
        "protein_coding",
        "protein_coding_cds_not_defined",
        "ig_c_gene",
        "ig_d_gene",
        "ig_j_gene",
        "ig_v_gene",
        "tr_c_gene",
        "tr_d_gene",
        "tr_j_gene",
        "tr_v_gene",
    }
)
_BIOTYPE_COLUMN_CANDIDATES = (
    "biotype",
    "gene_biotype",
    "gene_type",
    "gene_type_name",
    "feature_biotype",
    "feature_type",
)


def _coerce_bool_mask(values):
    import pandas as pd

    return pd.Series(values).fillna(False).astype(bool)


def _infer_biotype_col(df, explicit: str | None = None) -> str | None:
    if explicit and explicit in df.columns:
        return explicit
    for col in _BIOTYPE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _is_kept_biotype(value: object) -> bool:
    import pandas as pd

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    token = str(value).strip().lower()
    if not token:
        return True
    if token.startswith("protein_coding"):
        return True
    return token in _KEEP_NONCODING_NORMALIZATION_BIOTYPES


def _clean_tpm_normalize(
    out,
    *,
    label_col,
    id_col,
    value_cols,
    group_cols,
    censored_fill,
    technical_fraction,
    exclude_ribosomal_proteins,
    protect,
):
    """Apply the shared reference clean-TPM transform (:func:`clean_tpm_matrix`)
    to ``out`` in place, per (group × value column), and return
    ``(out, info)``. Each value column is a per-gene vector; treating it as a
    one-sample gene×sample matrix lets the runtime path reuse the identical
    builder transform so the result lands on the v4 basis (biological 750k).
    """
    import pandas as pd

    # Clean-TPM censoring is ENSG-keyed (the canonical censored-gene list). A
    # symbol-only frame can't be censored, and silently censoring nothing while
    # still rescaling to the fixed-fraction budget would misnormalize — so
    # require a real Ensembl_Gene_ID column for any non-zero censored_fill. The
    # legacy censored_fill="zero" path (classify_gene_qc, symbol-capable) does
    # not reach here.
    if not id_col or id_col not in out.columns:
        raise ValueError(
            f"clean-TPM normalization (censored_fill={censored_fill!r}) needs an "
            f"'{id_col}' Ensembl-gene-id column — the censored-gene list is "
            "ENSG-keyed; resolve symbols to Ensembl ids first, or use "
            "censored_fill='zero' for the legacy symbol-based transform")
    gene_table = pd.DataFrame(
        {
            "Symbol": out[label_col].fillna("").astype(str),
            "Ensembl_Gene_ID": out[id_col].fillna("").astype(str),
        },
        index=out.index,
    )
    removable = clean_tpm_removal_mask(
        gene_table,
        exclude_ribosomal_proteins=exclude_ribosomal_proteins,
        protect=protect,
    )

    def _apply(indices):
        idx = list(indices)
        block = out.loc[idx, value_cols].apply(pd.to_numeric, errors="coerce")
        cleaned = clean_tpm_matrix(
            block,
            removable=removable.loc[idx],
            gene_table=gene_table.loc[idx],
            exclude_ribosomal_proteins=exclude_ribosomal_proteins,
            censored_fill=censored_fill,
            technical_fraction=technical_fraction,
        )
        out.loc[idx, value_cols] = cleaned

    resolved_group_cols: list[str] = []
    if group_cols is None:
        _apply(out.index)
    else:
        resolved_group_cols = [str(c) for c in group_cols if str(c) in out.columns]
        if not resolved_group_cols:
            return out, {
                "applied": False,
                "reason": "missing grouping columns",
                "columns": {},
                "groups": {},
            }
        for _key, idx in out.groupby(resolved_group_cols, dropna=False).groups.items():
            _apply(idx)

    return out, {
        "applied": True,
        "reason": f"clean-TPM ({censored_fill}) reference transform applied",
        "columns": {},
        "groups": {},
        "label_col": label_col,
        "value_cols": list(value_cols),
        "group_cols": resolved_group_cols,
        "censored_fill": censored_fill,
        "technical_fraction": (
            technical_fraction if censored_fill == "fixed_fraction" else None
        ),
        # the actually-applied per-compartment split (default fixed_fraction path)
        "ribosomal_protein_fraction": (
            RIBOSOMAL_PROTEIN_FRACTION
            if censored_fill == "fixed_fraction" and exclude_ribosomal_proteins
            else None
        ),
        "other_technical_fraction": (
            OTHER_TECHNICAL_FRACTION
            if censored_fill == "fixed_fraction" and exclude_ribosomal_proteins
            else None
        ),
        "exclude_ribosomal_proteins": bool(exclude_ribosomal_proteins),
        "removed_technical_gene_count": int(removable.sum()),
        "removed_feature_mode": censored_fill,
    }


def normalize_expression(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols: Iterable[str] | None = None,
    group_cols: Iterable[str] | None = None,
    biotype_col: str | None = None,
    remove_noncoding: bool = False,
    remove_groups: Iterable[str] = _DEFAULT_NORMALIZE_REMOVE_GROUPS,
    censored_fill: str = "zero",
    technical_fraction: float = TECHNICAL_FRACTION,
    exclude_ribosomal_proteins: bool = True,
    protect=None,
):
    """Normalize technical-RNA features and rescale each column's mass.

    The default (``censored_fill="zero"``, legacy) transform zeroes
    mitochondrial transcripts, NUMT-like mitochondrial pseudogenes,
    rRNA/rRNA-pseudogene rows, and the polyadenylation-bias lncRNA panel
    (MALAT1, NEAT1), then rescales every expression vector so the remaining
    non-missing TPM mass stays on the original per-column total. Raw
    expression should be kept for QC and provenance; this helper defines the
    comparable biology view used after QC.

    Set ``censored_fill`` to a non-``"zero"`` mode to apply the **same**
    reference clean-TPM transform the cohort builders use
    (:func:`clean_tpm_matrix`), so a consumer normalizing its own inputs
    matches how the packaged ``cancer_reference_expression`` references were
    built (#311):

    - ``"fixed_fraction"`` (clean_tpm) — three-compartment: the censored
      (technical) block is pinned to ``technical_fraction`` of the 1e6 budget
      (default 25%) and the biological block to the rest (75%), each
      renormalized within its group. This is the basis the references ship on.
    - ``"reference"`` / ``"typical"`` — the other :func:`clean_tpm_matrix`
      modes.

    In every non-``"zero"`` mode the censored set is the **clean-TPM removal
    set** (:func:`clean_tpm_removal_mask`: technical RNA **plus**
    ribosomal-protein mRNA/pseudogenes, minus curated targets), not the
    technical-only ``remove_groups`` of the legacy zero path — matching the
    references by construction. ``exclude_ribosomal_proteins=False`` narrows
    it to technical-only; ``protect`` overrides the protected-target symbols;
    ``remove_noncoding`` / ``remove_groups`` / ``biotype_col`` apply only to
    the legacy zero path.

    Classification is done via :func:`classify_gene_qc`, which prefers
    ENSG-id lookup against the curated pirlygenes QC tables and falls
    back to symbol-level regex. When ``id_col`` is present in the
    frame, ENSG IDs are passed through to the classifier; otherwise
    only ``label_col`` is consulted. Versioned IDs
    (``ENSG00000251562.5``) are stripped to unversioned form before
    lookup.

    ``remove_noncoding=True`` additionally zeroes rows with noncoding
    biotypes when a biotype column is present, while keeping
    protein-coding, immunoglobulin, and TCR biotypes. Off by default
    because lncRNAs and small RNAs can be real biology in some assays.
    """
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}, "groups": {}}
    if label_col not in df.columns:
        return df.copy(), {
            "applied": False,
            "reason": f"label column {label_col!r} not present",
            "columns": {},
            "groups": {},
        }

    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if _is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
            "groups": {},
        }

    if censored_fill == "fixed_fraction":
        return _clean_tpm_normalize(
            out,
            label_col=label_col,
            id_col=id_col,
            value_cols=value_cols,
            group_cols=group_cols,
            censored_fill=censored_fill,
            technical_fraction=technical_fraction,
            exclude_ribosomal_proteins=exclude_ribosomal_proteins,
            protect=protect,
        )
    if censored_fill != "zero":
        raise ValueError(
            "normalize_expression supports censored_fill='fixed_fraction' (the "
            "single clean-TPM contract) or 'zero' (legacy technical-RNA drop); "
            f"got {censored_fill!r} — the reference / typical modes were removed.")

    labels = out[label_col].fillna("").astype(str).str.strip()
    remove_group_set = {str(group) for group in remove_groups}
    if id_col and id_col in out.columns:
        ids = out[id_col].fillna("").astype(str).str.split(".").str[0].str.strip()
        qc_classes = [
            classify_gene_qc(sym, ensembl_id=ensg)
            for sym, ensg in zip(labels, ids)
        ]
        qc_classes = pd.Series(qc_classes, index=out.index)
    else:
        qc_classes = labels.map(classify_gene_qc)
    technical_mask = qc_classes.map(lambda qc: qc.group in remove_group_set).astype(bool)
    biotype_col = _infer_biotype_col(out, biotype_col)
    noncoding_mask = _coerce_bool_mask([False] * len(out))
    noncoding_mask.index = out.index
    if remove_noncoding and biotype_col is not None:
        noncoding_mask = ~out[biotype_col].map(_is_kept_biotype).astype(bool)
    removable = (technical_mask | noncoding_mask).astype(bool)
    technical_count = int(technical_mask.sum())
    noncoding_count = int(noncoding_mask.sum()) if remove_noncoding else 0

    def _normalize_indices(indices, record_target):
        nonlocal out
        any_applied_inner = False
        idx = list(indices)
        idx_set = set(idx)
        group_removable = removable.loc[idx]
        group_technical = technical_mask.loc[idx]
        group_noncoding = noncoding_mask.loc[idx]
        for col in value_cols:
            vals = pd.to_numeric(out.loc[idx, col], errors="coerce")
            valid = vals.notna()
            removable_valid = group_removable & valid
            keep_valid = (~group_removable) & valid
            raw_sum = float(vals.sum())
            removed = float(vals[removable_valid].sum())
            remaining = raw_sum - removed
            removed_fraction = removed / raw_sum if raw_sum > 0 else 0.0
            record_target[col] = {
                "input_sum": raw_sum,
                "removed_tpm": removed,
                "removed_fraction": removed_fraction,
                "removed_gene_count": int(group_removable.sum()),
                "removed_technical_gene_count": int(group_technical.sum()),
                "removed_noncoding_gene_count": int(group_noncoding.sum()),
                "renormalization_factor": (
                    float(raw_sum / remaining) if raw_sum > 0 and remaining > 0 else 1.0
                ),
            }
            if raw_sum <= 0 or removed <= 0:
                continue
            remove_idx = [
                i for i in removable_valid[removable_valid].index if i in idx_set
            ]
            if remaining <= 0:
                out.loc[remove_idx, col] = 0.0
                any_applied_inner = True
                continue
            scale = raw_sum / remaining
            keep_idx = [i for i in keep_valid[keep_valid].index if i in idx_set]
            out.loc[remove_idx, col] = 0.0
            out.loc[keep_idx, col] = vals.loc[keep_idx] * scale
            any_applied_inner = True
        return any_applied_inner

    column_records = {}
    group_records = {}
    any_applied = False
    if group_cols is None:
        any_applied = _normalize_indices(out.index, column_records)
    else:
        group_cols = [str(c) for c in group_cols if str(c) in out.columns]
        if not group_cols:
            return out, {
                "applied": False,
                "reason": "missing grouping columns",
                "columns": {},
                "groups": {},
            }
        grouped = out.groupby(group_cols, dropna=False).groups
        for key, idx in grouped.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_label = "|".join(str(part) for part in key_tuple)
            group_records[key_label] = {}
            any_applied = _normalize_indices(idx, group_records[key_label]) or any_applied

    reason = (
        "technical/noncoding expression rows zeroed and remaining expression renormalized"
        if any_applied and remove_noncoding
        else (
            "technical RNA features zeroed and remaining expression renormalized"
            if any_applied
            else "no removable technical/noncoding burden"
        )
    )
    return out, {
        "applied": any_applied,
        "reason": reason,
        "columns": column_records,
        "groups": group_records,
        "label_col": label_col,
        "value_cols": value_cols,
        "group_cols": list(group_cols or []),
        "biotype_col": biotype_col,
        "remove_groups": sorted(remove_group_set),
        "remove_noncoding": bool(remove_noncoding),
        "removed_technical_gene_count": technical_count,
        "removed_noncoding_gene_count": noncoding_count,
        "removed_feature_mode": "zeroed_then_renormalized",
    }


def normalize_technical_rna_columns(
    df,
    *,
    label_col: str = "Symbol",
    value_cols: Iterable[str] | None = None,
    censored_fill: str = "zero",
    technical_fraction: float = TECHNICAL_FRACTION,
):
    """Normalize mtDNA/rRNA-like features and renormalize every expression column.

    Shared comparability transform for reference matrices. With the default
    ``censored_fill="zero"`` it zeroes technical-RNA features and preserves
    each column's total mass (legacy behavior). Set
    ``censored_fill="fixed_fraction"`` to apply the clean_tpm three-compartment
    transform the cohort builders use, so your normalized inputs match how the
    packaged references were built (#311) — see :func:`normalize_expression`.

    Per-sample raw-expression QC narration (TPM-share summaries, top-K
    concentration, rescue summaries) is part of the analysis layer and lives in
    ``trufflepig`` — record it before calling this transform if you need
    provenance for the un-cleaned state.
    """
    return normalize_expression(
        df,
        label_col=label_col,
        value_cols=value_cols,
        remove_noncoding=False,
        censored_fill=censored_fill,
        technical_fraction=technical_fraction,
    )


def normalize_technical_rna_long_table(
    df,
    *,
    label_col: str = "symbol",
    group_cols: Iterable[str] = ("cancer_code", "subtype"),
    value_cols: Iterable[str] = ("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
    censored_fill: str = "zero",
    technical_fraction: float = TECHNICAL_FRACTION,
):
    """Apply technical-RNA normalization within each long-table cohort group.

    ``censored_fill="fixed_fraction"`` applies the clean_tpm_v4 reference
    transform per cohort group (matching the packaged references) instead of
    the legacy zero-and-renormalize (#311); see :func:`normalize_expression`.
    """
    return normalize_expression(
        df,
        label_col=label_col,
        group_cols=group_cols,
        value_cols=value_cols,
        remove_noncoding=False,
        censored_fill=censored_fill,
        technical_fraction=technical_fraction,
    )


# ---------- builder clean-TPM (wide gene×sample matrix) ----------
#
# The single definition of the reference "clean TPM" transform that every
# cohort builder shares: zero the technical-RNA rows, then renormalize
# each sample column to 1e6. Previously copy-pasted into ~12 builders and
# scripts; this is now the one home.


# The clean-TPM censored set = the technical-RNA groups (mtDNA + mt-like
# pseudogene + rRNA-like + the two polyA-bias lncRNAs MALAT1/NEAT1) PLUS the
# ribosomal-protein family (mRNA + pseudogenes). These are the most-expressed,
# most multi-mapping-sensitive genes (huge pseudogene families) that split the
# zero-sum TPM budget very differently across quantifiers (RSEM vs STAR ~3-4x
# per gene) and are not tumor-specific signal. Nothing else is censored —
# other housekeeping (translation factors, ferritin, ubiquitin) is kept.
_RIBOSOMAL_PROTEIN_GROUPS = frozenset(
    {"ribosomal_protein", "ribosomal_protein_pseudogene"}
)


@functools.lru_cache(maxsize=2)
def _clean_tpm_censored_ids(include_ribosomal: bool) -> frozenset:
    """Unversioned ENSGs censored by the clean-TPM transform, read from the
    single canonical explicit list ``clean-tpm-censored-genes.csv`` (materialized
    once from :func:`classify_gene_qc`, categorized, and CTA-excluded — see
    ``scripts/generate_clean_tpm_censored_genes.py``). ``include_ribosomal``
    toggles the ``ribosomal_protein`` category on top of the always-included
    ``technical`` category."""
    from ..load_dataset import get_data
    df = get_data("clean-tpm-censored-genes", copy=False)
    if not include_ribosomal:
        df = df[df["category"].astype(str) == "technical"]
    return frozenset(df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0])


def _ensg_unversioned(gene_table):
    return gene_table["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]


def technical_rna_mask(gene_table):
    """Boolean Series — True for *technical-RNA* rows only (mtDNA / mt-like
    pseudogene / rRNA-like / polyA-bias lncRNA), via the canonical censored-gene
    list (the ``technical`` category). The strict technical set; for the default
    clean-TPM removal set (which also drops ribosomal proteins) use
    :func:`clean_tpm_removal_mask`.

    ``gene_table`` is any frame with an ``Ensembl_Gene_ID`` column; the returned
    mask is indexed like it.
    """
    return _ensg_unversioned(gene_table).isin(_clean_tpm_censored_ids(False))


@functools.lru_cache(maxsize=1)
def _default_protected_symbols():
    """Curated cancer-target symbols that must NEVER be censored even if their
    symbol matches a censored QC group — they are signal we score on. The
    canonical case is ``RPL10L`` (a testis CTA that is a ribosomal-protein
    paralog). Union of the CTA panel + surfaceome / key-gene / lineage / fusion
    panels; degrades gracefully if a source is unavailable."""
    protected: set[str] = set()
    try:
        from .. import gene_sets_cancer as gsc
        protected |= set(gsc.CTA_evidence()["Symbol"].astype(str))
    except Exception:
        pass
    try:
        from ..load_dataset import get_data
        for ds, cols in [("surface-proteins", ["Symbol"]),
                         ("cancer-key-genes", ["symbol"]),
                         ("cancer-lineage-panels", ["Symbol", "Child_Code"]),
                         ("cancer-fusions", ["gene_5prime", "gene_3prime"])]:
            try:
                df = get_data(ds)
            except Exception:
                continue
            for c in cols:
                if c in df.columns:
                    protected |= set(df[c].dropna().astype(str))
    except Exception:
        pass
    return frozenset(protected)


def clean_tpm_removal_mask(gene_table, *, exclude_ribosomal_proteins: bool = True,
                           protect=None):
    """Boolean Series of rows zeroed by the clean-TPM transform — the genes in
    the canonical censored-gene list (:func:`_clean_tpm_censored_ids`).

    The list is the **single source of truth** for the technical/biological
    split: technical-RNA (mtDNA + mt-like pseudogene + rRNA-like + the polyA-bias
    lncRNAs MALAT1/NEAT1) plus, when ``exclude_ribosomal_proteins`` (the
    default), ribosomal-protein mRNA + pseudogenes — and nothing else. It is
    CTA-safe by construction: curated cancer targets are excluded at generation,
    so a CTA ribosomal-protein paralog (``RPL10L``) or histone CTA (``H1-6``) is
    never censored (no runtime special-case needed).

    ``exclude_ribosomal_proteins=False`` gives the strict technical-only set
    (== :func:`technical_rna_mask`). ``protect`` optionally protects *additional*
    symbols beyond the list's baked-in cancer targets (it can only keep more
    genes, never censor a target).
    """
    if "Ensembl_Gene_ID" not in gene_table.columns:
        raise ValueError(
            "clean_tpm_removal_mask needs an 'Ensembl_Gene_ID' column — the "
            "canonical censored-gene list is keyed on ENSG (resolve symbols to "
            "Ensembl ids first)")
    ids = _clean_tpm_censored_ids(bool(exclude_ribosomal_proteins))
    mask = _ensg_unversioned(gene_table).isin(ids)
    if protect:
        keep = ~gene_table["Symbol"].astype(str).isin(set(protect))
        mask = mask & keep
    return mask


def drop_technical_genes(df, *, label_col: str = "Symbol",
                         id_col: str = "Ensembl_Gene_ID",
                         exclude_ribosomal_proteins: bool = True, protect=None):
    """Return ``df`` with the clean-TPM censored rows removed — the biology-only
    view of a gene×value frame.

    Uses the same canonical censored-gene list as the clean-TPM transform
    (:func:`clean_tpm_removal_mask`): technical RNA plus ribosomal proteins by
    default (``exclude_ribosomal_proteins=False`` keeps the technical-only set),
    CTA-safe by construction. ``df`` must carry ``id_col`` (Ensembl_Gene_ID);
    all other columns pass through untouched.

    Intended for distance/clustering and cross-sample comparison that should
    ride on biological signal rather than the technical compartment — in
    particular it is **insensitive to the clean_tpm_v4 fixed-fraction floor**
    (#304), since the inflated technical rows are removed before any distance is
    computed. (Distinct from :func:`pirlygenes.expression.filter_technical_rna`,
    which drops only the strict curated technical-RNA family set.)
    """
    import pandas as pd

    if id_col not in df.columns:
        raise ValueError(f"drop_technical_genes needs an {id_col!r} column")
    gene_table = pd.DataFrame(
        {
            "Symbol": (df[label_col].astype(str) if label_col in df.columns
                       else pd.Series([""] * len(df), index=df.index)),
            "Ensembl_Gene_ID": df[id_col].astype(str),
        },
        index=df.index,
    )
    removable = clean_tpm_removal_mask(
        gene_table, exclude_ribosomal_proteins=exclude_ribosomal_proteins,
        protect=protect)
    return df.loc[~removable.to_numpy()].reset_index(drop=True)


def clean_tpm_matrix(values, removable=None, *, gene_table=None,
                     exclude_ribosomal_proteins: bool = True,
                     censored_fill: str = "fixed_fraction",
                     technical_fraction: float = TECHNICAL_FRACTION,
                     ribosomal_protein_fraction: float = RIBOSOMAL_PROTEIN_FRACTION,
                     other_technical_fraction: float = OTHER_TECHNICAL_FRACTION):
    """The ONE clean-TPM transform on a gene×sample matrix (16/9/75 fixed_fraction).

    ``values`` is genes (rows) × samples (cols). Provide either an explicit
    boolean ``removable`` mask (aligned to ``values.index``) or a ``gene_table``
    (``Symbol`` + ``Ensembl_Gene_ID``, row-aligned to ``values``) — in which
    case the mask is built via :func:`clean_tpm_removal_mask`, which **censors
    ribosomal proteins by default** (``exclude_ribosomal_proteins=True`` removes
    them from the biological signal; pass ``False`` for technical-only).

    THREE-compartment normalization — force the **ribosomal-protein** block to
    ``ribosomal_protein_fraction`` (~16%), the **other-technical** block to
    ``other_technical_fraction`` (~9%), and the kept **biological** block to the
    remaining ~75% of the 1e6 budget, **renormalizing within each compartment**
    so relative expression inside each is preserved. Pinning ribosomal and
    other-technical *separately* (cancerdata's 16/9 refinement of the old
    lumped-25% v4) keeps one compartment's cross-sample/pipeline variation from
    bleeding into the other's budget — e.g. a sample with heavy residual rRNA no
    longer compresses its ribosomal-protein block. Fixing the BIOLOGICAL
    compartment to a constant ~750k budget is what makes biological clean-TPM
    cross-sample comparable — the one property the transform exists to provide.
    Cohort-independent (no reference table). An empty compartment stays at 0 (the
    others still hit their targets). The ribosomal sub-block is the censored
    ribosomal-protein category; with ``exclude_ribosomal_proteins=False`` (or no
    ``gene_table``) there is no ribosomal compartment and the censored block falls
    back to a single ``technical_fraction`` split.

    ``censored_fill`` accepts only ``"fixed_fraction"`` (the single clean-TPM
    contract); the legacy ``"zero"`` / ``"reference"`` / ``"typical"`` modes were
    removed. (Plain technical-RNA dropping — the old ``"zero"`` behavior — still
    lives in :func:`normalize_expression`, which is a different operation.)
    """
    if removable is None:
        if gene_table is None:
            raise ValueError("clean_tpm_matrix needs either removable or gene_table")
        removable = clean_tpm_removal_mask(
            gene_table, exclude_ribosomal_proteins=exclude_ribosomal_proteins)
    import pandas as pd

    if censored_fill != "fixed_fraction":
        raise ValueError(
            "clean_tpm_matrix only supports the single clean-TPM contract "
            f"censored_fill='fixed_fraction' (got {censored_fill!r}); the legacy "
            "zero / reference / typical modes were removed.")
    # The single clean-TPM contract. The censored block (mtDNA / rRNA-like /
    # mt-pseudogene / polyA-bias lncRNA + ribosomal-protein mRNA & pseudogenes) is
    # FORCED to a constant fraction of the 1e6 budget, split into separately pinned
    # ribosomal (~16%) and other-technical (~9%) compartments; biology fills the
    # constant remaining ~75%. Within each compartment relative expression is
    # preserved (cohort-independent, no reference table). Fixing the BIOLOGICAL
    # compartment to a constant 750k budget is what makes biological clean-TPM
    # cross-sample comparable — the one property the transform exists to provide.
    rem = removable.to_numpy()
    if not 0.0 < technical_fraction < 1.0:
        raise ValueError("technical_fraction must be in (0, 1)")

    def _scale_to_budget(mask, fraction):
        """Per-column scale that forces ``mask``'s mass to ``fraction`` of 1e6;
        0 where the compartment has no mass (it stays empty)."""
        s = values.loc[mask].sum(axis=0)
        out = pd.Series(0.0, index=values.columns, dtype=float)
        pos = s > 0
        out.loc[pos] = (fraction * 1_000_000.0) / s.loc[pos]
        return out

    clean = values.astype(float).copy()
    # The ribosomal sub-block is the gene_table ribosomal-protein CATEGORY (the
    # full censored list minus the technical-only list — the single source of
    # truth) INTERSECTED with the actual removable rows, so an explicitly-supplied
    # ``removable`` that disagrees with ``gene_table`` can't miscompartmentalize a
    # non-ribosomal gene into the 16% block.
    ribo_mask = None
    if exclude_ribosomal_proteins and gene_table is not None:
        full = clean_tpm_removal_mask(gene_table).to_numpy()
        tech_only = clean_tpm_removal_mask(
            gene_table, exclude_ribosomal_proteins=False).to_numpy()
        ribo_mask = rem & full & ~tech_only
    if ribo_mask is not None and ribo_mask.any():
        other_mask = rem & ~ribo_mask
        bio_frac = 1.0 - ribosomal_protein_fraction - other_technical_fraction
        if bio_frac <= 0.0:
            raise ValueError(
                "ribosomal_protein_fraction + other_technical_fraction must be "
                "< 1 (biology needs a positive budget)")
        clean.loc[ribo_mask] = values.loc[ribo_mask].mul(
            _scale_to_budget(ribo_mask, ribosomal_protein_fraction), axis=1)
        clean.loc[other_mask] = values.loc[other_mask].mul(
            _scale_to_budget(other_mask, other_technical_fraction), axis=1)
        clean.loc[~rem] = values.loc[~rem].mul(
            _scale_to_budget(~rem, bio_frac), axis=1)
        return clean.fillna(0.0)
    # technical-only view (no ribosomal compartment): single censored block pinned
    # to technical_fraction, biology to the rest.
    clean.loc[rem] = values.loc[rem].mul(
        _scale_to_budget(rem, technical_fraction), axis=1)
    clean.loc[~rem] = values.loc[~rem].mul(
        _scale_to_budget(~rem, 1.0 - technical_fraction), axis=1)
    return clean.fillna(0.0)


# ---------- cross-source transforms (#293) ----------
#
# Absolute clean TPM is NOT comparable across quantification pipelines (Toil/
# RSEM vs GDC/STAR): the per-gene offset is gene-specific and multiplicative
# (genome-wide median ~1.7x), so renormalizing to 1e6 doesn't fix it. But
# *rank* is preserved (log2(TPM+1) rank r~0.99). The documented rule:
#   - within a source: absolute clean TPM is fine;
#   - across sources:  compare in rank/z space (below), never average raw TPM
#     across pipelines.


def rank_normalize(values):
    """Within-sample percentile rank per gene (0-100), pipeline-robust (#293).

    ``values`` is genes (rows) × samples (cols); each column is ranked
    independently so a gene's value is its percentile among that sample's genes.
    On paired GDC-vs-Treehouse samples this agreed to within ~0.2 percentile
    points — the right representation for "is this gene high in this sample?"
    across pipelines (loses absolute units). NaNs are left as NaN."""
    import pandas as pd

    values = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    num = values.apply(pd.to_numeric, errors="coerce")
    return num.rank(axis=0, pct=True) * 100.0


def zscore_normalize(values, *, log: bool = True):
    """Within-sample z-score per gene, on log2(x+1) by default (#293).

    ``values`` is genes × samples; each column is standardized
    ``(x - mean) / std`` independently (population std, ddof=0). With
    ``log=True`` (default) the standardization is on ``log2(values + 1)`` — the
    recommended continuous representation for cross-source pooling (paired
    GDC-vs-Treehouse agreed to within ~0.1 sd). Zero-variance columns map to 0.
    """
    import numpy as np
    import pandas as pd

    values = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    x = values.apply(pd.to_numeric, errors="coerce")
    if log:
        x = np.log2(x + 1.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    z = x.sub(mean, axis=1)
    safe = std.replace(0.0, np.nan)
    z = z.div(safe, axis=1)
    return z.fillna(0.0)


# ---------- conversions ----------


def renormalize_to_million(
    df,
    *,
    value_cols: Iterable[str] | None = None,
):
    """Rescale each expression column so its non-NaN sum equals 10⁶.

    Bare utility — drops nothing. Use this after
    :func:`normalize_expression` if you also want the post-filter total
    pinned at exactly 10⁶ (the standard TPM convention). The default
    technical-RNA normalization preserves the *input* total, which may
    or may not already be 10⁶ depending on how the upstream quantifier
    handled denominator features.
    """
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if _is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
        }

    columns = {}
    any_applied = False
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        col_sum = float(vals.sum())
        columns[col] = {"input_sum": col_sum}
        if col_sum <= 0:
            columns[col]["scale"] = 1.0
            continue
        scale = 1e6 / col_sum
        out[col] = vals * scale
        columns[col]["scale"] = scale
        columns[col]["output_sum"] = 1e6
        any_applied = True
    return out, {
        "applied": any_applied,
        "reason": "rescaled to TPM convention (sum = 1e6)" if any_applied else "no positive column sums",
        "columns": columns,
        "value_cols": value_cols,
    }


def fpkm_to_tpm(
    df,
    *,
    value_cols: Iterable[str] | None = None,
):
    """Convert FPKM-scale expression columns to TPM by per-column rescaling.

    For each column, ``TPM_i = FPKM_i / sum(FPKM_j) * 1e6`` over rows
    with a finite value. Pair this with :func:`normalize_expression` to
    drop technical-RNA features before renormalizing — the standard
    flow is:

        1. :func:`fpkm_to_tpm` — convert quantifier output to TPM scale.
        2. ``trufflepig.expression_qc.raw_qc_profile`` (analysis layer) —
           record raw rRNA / mt / top-k composition for the report
           before any filtering.
        3. :func:`normalize_expression` — drop technical RNA and
           renormalize the remaining mass.
        4. :func:`renormalize_to_million` (optional) — pin the
           post-filter total at exactly 10⁶.

    This is mathematically identical to :func:`renormalize_to_million`
    but takes an FPKM-named argument and exists as a self-documenting
    entry point in the pipeline.
    """
    return renormalize_to_million(df, value_cols=value_cols)


def _default_fpkm_value_cols(
    df,
    *,
    source_prefix: str,
    source_suffix: str,
) -> list[str]:
    return [
        str(c) for c in df.columns
        if str(c).startswith(source_prefix) or str(c).endswith(source_suffix)
    ]


def _tpm_target_col_from_fpkm(
    source_col: str,
    *,
    source_prefix: str,
    target_prefix: str,
    source_suffix: str,
    target_suffix: str,
) -> str:
    if source_col.startswith(source_prefix):
        return target_prefix + source_col[len(source_prefix):]
    if source_col.endswith(source_suffix):
        return source_col[:-len(source_suffix)] + target_suffix
    return target_prefix + source_col


def add_tpm_columns_from_fpkm(
    df,
    *,
    value_cols: Iterable[str] | None = None,
    source_prefix: str = "FPKM_",
    target_prefix: str = "TPM_",
    source_suffix: str = "_FPKM",
    target_suffix: str = "_TPM",
    overwrite: bool = False,
):
    """Append TPM-scale companion columns for FPKM columns.

    The source FPKM columns are preserved. Each selected column is
    independently rescaled to sum to 10⁶, then written to a target
    column whose name is derived by replacing ``source_prefix`` with
    ``target_prefix`` or ``source_suffix`` with ``target_suffix``. For
    example, ``FPKM_BRCA`` becomes ``TPM_BRCA`` and ``BRCA_FPKM``
    becomes ``BRCA_TPM``.

    This is useful for reference tables where raw FPKM should remain
    available for provenance, while downstream analysis wants a
    deterministic TPM-scale view. Existing target columns are left
    unchanged unless ``overwrite=True``.
    """
    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}

    out = df.copy()
    if value_cols is None:
        value_cols = _default_fpkm_value_cols(
            out,
            source_prefix=source_prefix,
            source_suffix=source_suffix,
        )
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no FPKM expression value columns",
            "columns": {},
        }

    converted, record = fpkm_to_tpm(out, value_cols=value_cols)
    columns = {}
    for source_col in value_cols:
        target_col = _tpm_target_col_from_fpkm(
            source_col,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
        )
        if target_col in out.columns and not overwrite:
            columns[source_col] = {
                "target_column": target_col,
                "skipped": True,
                "reason": "target column already exists",
            }
            continue
        out[target_col] = converted[source_col]
        columns[source_col] = {
            "target_column": target_col,
            **record.get("columns", {}).get(source_col, {}),
        }

    return out, {
        "applied": bool(columns),
        "reason": "added TPM companion columns from FPKM columns",
        "columns": columns,
        "source_prefix": source_prefix,
        "target_prefix": target_prefix,
        "source_suffix": source_suffix,
        "target_suffix": target_suffix,
    }


def percentile_rank_expression(
    df,
    *,
    value_cols: Iterable[str] | None = None,
):
    """Map expression columns to within-column percentile ranks (0–100)."""
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if _is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
        }

    columns = {}
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        out[col] = vals.rank(pct=True) * 100
        ranked = out[col].dropna()
        columns[col] = {
            "n_ranked": int(ranked.shape[0]),
            "min": float(ranked.min()) if not ranked.empty else None,
            "max": float(ranked.max()) if not ranked.empty else None,
        }
    return out, {
        "applied": True,
        "reason": "converted expression columns to percentile ranks",
        "columns": columns,
        "value_cols": value_cols,
    }


def _housekeeping_panel_symbols(panel: Iterable[str] | None = None) -> set[str]:
    if panel is not None:
        return {str(s).strip().upper() for s in panel if str(s).strip()}
    from pirlygenes import housekeeping_gene_names

    return {str(s).strip().upper() for s in housekeeping_gene_names() if str(s).strip()}


def _housekeeping_panel_ensembl_ids(panel: Iterable[str] | None = None) -> set[str]:
    """Unversioned ENSG IDs for the housekeeping panel."""
    if panel is not None:
        return {str(s).split(".", 1)[0].strip() for s in panel if str(s).strip()}
    from pirlygenes import housekeeping_gene_ids

    return {str(s).split(".", 1)[0].strip() for s in housekeeping_gene_ids() if str(s).strip()}


def tpm_to_housekeeping_normalized(
    df,
    *,
    label_col: str = "Symbol",
    id_col: str | None = "Ensembl_Gene_ID",
    value_cols: Iterable[str] | None = None,
    panel: Iterable[str] | None = None,
    panel_ids: Iterable[str] | None = None,
    pseudocount: float = 0.1,
):
    """Divide each expression column by the geometric mean of a HK panel.

    Uses :func:`pirlygenes.housekeeping_gene_names` by default (~22
    classic housekeeping genes, e.g. ACTB, GAPDH, B2M, HPRT1, PGK1).
    Pass ``panel`` to override.

    A small ``pseudocount`` is added before taking the log to make the
    geometric mean robust to zeros in the panel. Returns the
    HK-normalized frame plus a record dict with the panel-symbol
    coverage and per-column denominator value, so callers can audit
    whether the panel was actually present.

    The output is on a unit-free ratio scale (gene expression relative
    to the housekeeping baseline). It survives library-prep total
    drift in a way that TPM does not — useful when comparing samples
    across very different sequencing depth or capture chemistry.
    """
    import numpy as np
    import pandas as pd

    if df is None:
        return None, {"applied": False, "reason": "no table", "columns": {}}
    if label_col not in df.columns:
        return df.copy(), {
            "applied": False,
            "reason": f"label column {label_col!r} not present",
            "columns": {},
        }

    out = df.copy()
    if value_cols is None:
        value_cols = [c for c in out.columns if _is_expression_value_col(c)]
    value_cols = [str(c) for c in value_cols if str(c) in out.columns]
    if not value_cols:
        return out, {
            "applied": False,
            "reason": "no expression value columns",
            "columns": {},
        }

    # ENSG-first matching when both an id_col is present and we have
    # ENSG IDs from pirlygenes; falls back to symbol matching when
    # either side is missing.
    if id_col and id_col in out.columns:
        panel_ensgs = _housekeeping_panel_ensembl_ids(panel_ids)
        ids = out[id_col].fillna("").astype(str).str.split(".").str[0].str.strip()
        hk_mask = ids.isin(panel_ensgs)
        match_mode = "ensembl_id"
        panel_size = len(panel_ensgs)
    else:
        panel_syms = _housekeeping_panel_symbols(panel)
        labels = out[label_col].fillna("").astype(str).str.strip().str.upper()
        hk_mask = labels.isin(panel_syms)
        match_mode = "symbol"
        panel_size = len(panel_syms)
    n_hk_present = int(hk_mask.sum())
    if n_hk_present == 0:
        return out, {
            "applied": False,
            "reason": f"no housekeeping panel genes found via {match_mode}",
            "columns": {},
            "panel_size": panel_size,
            "match_mode": match_mode,
        }

    columns = {}
    any_applied = False
    for col in value_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        hk_vals = vals[hk_mask].dropna()
        if hk_vals.empty:
            columns[col] = {"hk_geomean": None, "scale": 1.0, "n_hk_used": 0}
            continue
        log_vals = np.log(hk_vals.astype(float) + pseudocount)
        hk_geomean = float(np.exp(log_vals.mean()) - pseudocount)
        if hk_geomean <= 0:
            columns[col] = {"hk_geomean": hk_geomean, "scale": 1.0, "n_hk_used": int(len(hk_vals))}
            continue
        out[col] = vals / hk_geomean
        columns[col] = {
            "hk_geomean": hk_geomean,
            "scale": 1.0 / hk_geomean,
            "n_hk_used": int(len(hk_vals)),
        }
        any_applied = True
    return out, {
        "applied": any_applied,
        "reason": "divided by housekeeping geomean per column" if any_applied else "no positive HK geomeans",
        "columns": columns,
        "value_cols": value_cols,
        "panel_size": panel_size,
        "panel_present_in_table": n_hk_present,
        "match_mode": match_mode,
        "pseudocount": pseudocount,
    }


__all__ = [
    "normalize_expression",
    "normalize_technical_rna_columns",
    "normalize_technical_rna_long_table",
    "renormalize_to_million",
    "fpkm_to_tpm",
    "add_tpm_columns_from_fpkm",
    "percentile_rank_expression",
    "tpm_to_housekeeping_normalized",
    "technical_rna_mask",
    "clean_tpm_removal_mask",
    "clean_tpm_matrix",
]
