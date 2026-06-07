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

from .qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc


# Default removal panel. Mirrors :data:`expression_qc._TECHNICAL_RNA_GROUPS`
# — mtDNA, NUMT-like pseudogenes, nuclear rRNA-like, and the
# polyadenylation-bias lncRNA panel (MALAT1, NEAT1). See the
# expression_qc module for the per-group citations.
_DEFAULT_NORMALIZE_REMOVE_GROUPS = _TECHNICAL_RNA_GROUPS
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
    reference,
):
    """Apply the shared reference clean-TPM transform (:func:`clean_tpm_matrix`)
    to ``out`` in place, per (group × value column), and return
    ``(out, info)``. Each value column is a per-gene vector; treating it as a
    one-sample gene×sample matrix lets the runtime path reuse the identical
    builder transform so the result lands on the v4 basis (biological 750k).
    """
    import pandas as pd

    gene_table = pd.DataFrame(
        {
            "Symbol": out[label_col].fillna("").astype(str),
            "Ensembl_Gene_ID": (
                out[id_col].fillna("").astype(str)
                if id_col and id_col in out.columns
                else pd.Series([""] * len(out), index=out.index)
            ),
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
            reference=reference,
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
    technical_fraction: float = 0.25,
    exclude_ribosomal_proteins: bool = True,
    protect=None,
    reference=None,
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

    - ``"fixed_fraction"`` (clean_tpm_v4) — two-compartment: the censored
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

    if censored_fill != "zero":
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
            reference=reference,
        )

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
    technical_fraction: float = 0.25,
):
    """Normalize mtDNA/rRNA-like features and renormalize every expression column.

    Shared comparability transform for reference matrices. With the default
    ``censored_fill="zero"`` it zeroes technical-RNA features and preserves
    each column's total mass (legacy behavior). Set
    ``censored_fill="fixed_fraction"`` to apply the clean_tpm_v4 two-compartment
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
    technical_fraction: float = 0.25,
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


def _qc_group_mask(gene_table, groups):
    """Boolean Series — True for rows whose gene QC group is in ``groups``."""
    import pandas as pd

    want = {str(g) for g in groups}
    qc = [
        classify_gene_qc(symbol, ensembl_id=ensg)
        for symbol, ensg in zip(gene_table["Symbol"], gene_table["Ensembl_Gene_ID"])
    ]
    return pd.Series([k.group in want for k in qc], index=gene_table.index)


def technical_rna_mask(gene_table):
    """Boolean Series — True for *technical-RNA* rows only (mtDNA / mt-like
    pseudogene / rRNA-like / polyA-bias lncRNA). This is the strict technical
    set; for the default clean-TPM removal set (which also drops ribosomal
    proteins) use :func:`clean_tpm_removal_mask`.

    ``gene_table`` is any frame with ``Symbol`` and ``Ensembl_Gene_ID``
    columns; the returned mask is indexed like it.
    """
    return _qc_group_mask(gene_table, _TECHNICAL_RNA_GROUPS)


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
    """Boolean Series of rows zeroed by the default clean-TPM transform: the
    technical-RNA groups (mtDNA + mt-like pseudogene + rRNA-like + the two
    polyA-bias lncRNAs MALAT1/NEAT1) plus, when ``exclude_ribosomal_proteins``
    (the default), ribosomal-protein mRNA + pseudogenes — and **nothing else** —
    minus any curated cancer-target gene (a target is never censored even if its
    symbol matches a censored group; e.g. the CTA ``RPL10L``).

    These are housekeeping, not tumor-specific, and — being the most-expressed,
    most multi-mapping-prone genes — they destabilize the zero-sum TPM
    denominator across quantification pipelines. Other housekeeping (translation
    factors, ferritin, ubiquitin) is intentionally NOT censored. Pass
    ``exclude_ribosomal_proteins=False`` for the strict technical-only set
    (equivalent to :func:`technical_rna_mask`); ``protect`` overrides the default
    protected-target symbols.
    """
    groups = set(_TECHNICAL_RNA_GROUPS)  # mtDNA/mt-pseudogene + rRNA + polyA-lncRNA
    if exclude_ribosomal_proteins:
        groups |= set(_RIBOSOMAL_PROTEIN_GROUPS)
    mask = _qc_group_mask(gene_table, groups)
    prot = _default_protected_symbols() if protect is None else set(protect)
    if prot:
        keep = ~gene_table["Symbol"].astype(str).isin(prot).to_numpy()
        mask = mask & keep
    return mask


def censored_gene_reference():
    """``{Symbol: reference_tpm}`` — each censored gene's fixed surrogate value,
    its median TPM across the Treehouse 25.01 PolyA compendium (the reference
    cohort). Used by :func:`clean_tpm_matrix` ``censored_fill="reference"`` so a
    censored gene holds the **same** value in every cohort/pipeline. Generated
    by ``scripts/generate_censored_gene_reference.py``."""
    from ..load_dataset import get_data
    df = get_data("censored-gene-reference-tpm")
    return dict(zip(df["Symbol"].astype(str), df["reference_tpm"].astype(float)))


def clean_tpm_matrix(values, removable=None, *, gene_table=None,
                     exclude_ribosomal_proteins: bool = True,
                     censored_fill: str = "fixed_fraction", censored_budget=None,
                     reference=None, technical_fraction: float = 0.25):
    """Reference clean-TPM transform on a gene×sample matrix.

    ``values`` is genes (rows) × samples (cols). Provide either an explicit
    boolean ``removable`` mask (aligned to ``values.index``) or a ``gene_table``
    (``Symbol`` + ``Ensembl_Gene_ID``, row-aligned to ``values``) — in which
    case the mask is built via :func:`clean_tpm_removal_mask`, which **excludes
    ribosomal proteins by default**.

    ``censored_fill`` controls what happens to the censored (removable) rows
    (``"fixed_fraction"`` is the default as of clean_tpm_v4 — see below):

    - ``"reference"``: replace **each** censored gene with its own
      **fixed reference value** — its median TPM across the Treehouse PolyA
      compendium (:func:`censored_gene_reference`) — then renormalize each
      column to 1e6. Because the censored block is per-gene and identical in
      every cohort/pipeline, the kept genes are scaled by ``1e6 / (kept_sum +
      reference_sum)`` with ``reference_sum`` constant, so a sample's variable
      (often pipeline-driven) technical/ribosomal fraction no longer inflates
      its other genes, and the censored values are comparable across sources.
      Requires ``gene_table`` (or an explicit ``reference`` Symbol->TPM map) for
      the per-gene lookup; genes absent from the reference get 0.
    - ``"typical"``: hold the censored block at one constant ``censored_budget``
      (median censored TPM sum across ``values`` if ``None``) split equally
      across censored genes — cohort-derived, single value.
    - ``"fixed_fraction"`` (**default**, clean_tpm_v4): two-compartment
      normalization — force the censored (technical) block to a constant
      ``technical_fraction`` of the 1e6 budget (default 25%) and the kept
      (biological) block to the remaining
      ``1 - technical_fraction`` (75%), **renormalizing within each group** so
      relative expression inside each compartment is preserved. This removes the
      cross-sample/pipeline *technical-fraction* confound (every sample is 25%
      technical) while keeping each sample's real within-compartment ratios —
      unlike ``"reference"``, which erases within-technical variation by pinning
      each gene to a fixed value. Cohort-independent (no reference table). The
      biological compartment lands on a constant 750k budget in every sample, so
      kept genes are directly comparable across samples and sources. A sample
      with no technical mass keeps technical at 0 (biological still fills 75%).
    - ``"zero"`` (legacy): drop the censored rows and renormalize the remainder
      (inflates survivors by ``1/(1-censored_fraction)``).

    Columns whose post-fill denominator is <= 0 collapse to zero.
    """
    if removable is None:
        if gene_table is None:
            raise ValueError("clean_tpm_matrix needs either removable or gene_table")
        removable = clean_tpm_removal_mask(
            gene_table, exclude_ribosomal_proteins=exclude_ribosomal_proteins)
    import numpy as np
    import pandas as pd

    rem = removable.to_numpy()
    if censored_fill == "reference":
        # Censored genes are pinned to their FIXED per-gene reference value
        # (identical in every cohort/pipeline) and NOT renormalized; the kept
        # genes are scaled to fill the remaining budget (1e6 - reference_sum).
        if reference is None:
            reference = censored_gene_reference()
        if gene_table is None:
            raise ValueError("censored_fill='reference' needs gene_table (or "
                             "reference=) for the per-gene surrogate lookup")
        ref_row = gene_table["Symbol"].astype(str).map(reference).fillna(0.0).to_numpy()
        ref_sum = float(ref_row[rem].sum())
        kept_raw_sum = values.loc[~rem].sum(axis=0)
        kept_scale = pd.Series(0.0, index=values.columns, dtype=float)
        pos = kept_raw_sum > 0
        kept_scale.loc[pos] = max(0.0, 1_000_000.0 - ref_sum) / kept_raw_sum.loc[pos]
        clean = values.astype(float).copy()
        clean.loc[~rem] = values.loc[~rem].mul(kept_scale, axis=1)
        ci = np.where(rem)[0]
        if len(ci):  # exact reference value, broadcast across sample columns
            clean.iloc[ci, :] = ref_row[ci][:, None]
        return clean.fillna(0.0)

    if censored_fill == "fixed_fraction":
        # Two-compartment: technical -> technical_fraction*1e6, biological ->
        # the rest, each renormalized WITHIN its group (relative expression
        # preserved). Removes the technical-fraction confound without erasing
        # within-compartment signal; cohort-independent (no reference table).
        if not 0.0 < technical_fraction < 1.0:
            raise ValueError("technical_fraction must be in (0, 1)")
        tech_budget = technical_fraction * 1_000_000.0
        bio_budget = (1.0 - technical_fraction) * 1_000_000.0
        tech_sum = values.loc[rem].sum(axis=0)
        bio_sum = values.loc[~rem].sum(axis=0)
        tscale = pd.Series(0.0, index=values.columns, dtype=float)
        tpos = tech_sum > 0
        tscale.loc[tpos] = tech_budget / tech_sum.loc[tpos]
        bscale = pd.Series(0.0, index=values.columns, dtype=float)
        bpos = bio_sum > 0
        bscale.loc[bpos] = bio_budget / bio_sum.loc[bpos]
        clean = values.astype(float).copy()
        clean.loc[rem] = values.loc[rem].mul(tscale, axis=1)
        clean.loc[~rem] = values.loc[~rem].mul(bscale, axis=1)
        return clean.fillna(0.0)

    if censored_fill == "zero":
        clean = values.copy()
        clean.loc[rem, :] = 0.0
        denom = clean.sum(axis=0)
    elif censored_fill == "typical":
        kept_sum = values.loc[~rem].sum(axis=0)
        if censored_budget is None:
            censored_budget = float(values.loc[rem].sum(axis=0).median()) if rem.any() else 0.0
        clean = values.copy()
        if rem.any():
            clean.loc[rem, :] = censored_budget / int(rem.sum())
        denom = kept_sum + censored_budget
    else:
        raise ValueError("censored_fill must be 'reference', 'typical', or 'zero'")

    # only a genuinely empty column (no mass at all) collapses to zero
    positive = denom > 0
    scale = pd.Series(np.nan, index=denom.index, dtype=float)
    scale.loc[positive] = 1_000_000.0 / denom.loc[positive]
    return clean.mul(scale, axis=1).fillna(0.0)


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
    "censored_gene_reference",
    "clean_tpm_matrix",
]
