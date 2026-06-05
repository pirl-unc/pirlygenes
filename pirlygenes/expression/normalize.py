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
):
    """Zero technical-RNA features and rescale each column's remaining mass.

    The default transform zeroes mitochondrial transcripts, NUMT-like
    mitochondrial pseudogenes, rRNA/rRNA-pseudogene rows, and the
    polyadenylation-bias lncRNA panel (MALAT1, NEAT1), then rescales
    every expression vector so the remaining non-missing TPM mass
    stays on the original per-column total. Raw expression should be
    kept for QC and provenance; this helper defines the comparable
    biology view used after QC.

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
):
    """Zero mtDNA/rRNA-like features and renormalize every expression column.

    Shared comparability transform for reference matrices. Preserves
    each column's total expression mass after removing technical-RNA
    features. Per-sample raw-expression QC narration (TPM-share
    summaries, top-K concentration, rescue summaries) is part of the
    analysis layer and lives in ``trufflepig`` — record it before
    calling this transform if you need provenance for the un-cleaned
    state.
    """
    return normalize_expression(
        df,
        label_col=label_col,
        value_cols=value_cols,
        remove_noncoding=False,
    )


def normalize_technical_rna_long_table(
    df,
    *,
    label_col: str = "symbol",
    group_cols: Iterable[str] = ("cancer_code", "subtype"),
    value_cols: Iterable[str] = ("tumor_tpm_median", "tumor_tpm_q1", "tumor_tpm_q3"),
):
    """Apply technical-RNA normalization within each long-table cohort group."""
    return normalize_expression(
        df,
        label_col=label_col,
        group_cols=group_cols,
        value_cols=value_cols,
        remove_noncoding=False,
    )


# ---------- builder clean-TPM (wide gene×sample matrix) ----------
#
# The single definition of the reference "clean TPM" transform that every
# cohort builder shares: zero the technical-RNA rows, then renormalize
# each sample column to 1e6. Previously copy-pasted into ~12 builders and
# scripts; this is now the one home.


# Ribosomal-protein mRNA (RPL*/RPS*/RACK1/FAU) + their processed pseudogenes.
# Among the most-expressed genes AND the most multi-mapping-sensitive (huge
# pseudogene families), so different quantifiers split the zero-sum TPM budget
# into them very differently (RSEM vs STAR can differ ~3-4x per RP gene). Since
# TPM is closed, that variable "tax" deflates/inflates every other gene and
# breaks cross-source comparability. They are not tumor-specific signal, so the
# clean-TPM denominator excludes them by default (see clean_tpm_removal_mask).
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


def clean_tpm_removal_mask(gene_table, *, exclude_ribosomal_proteins: bool = True):
    """Boolean Series of rows zeroed by the default clean-TPM transform: the
    technical-RNA set (always) plus, when ``exclude_ribosomal_proteins`` (the
    default), ribosomal-protein mRNA and its pseudogenes.

    Excluding ribosomal proteins is the default because they are housekeeping,
    not tumor-specific, and — being the most-expressed, most multi-mapping-prone
    genes — they dominate and destabilize the zero-sum TPM denominator across
    quantification pipelines (see :data:`_RIBOSOMAL_PROTEIN_GROUPS`). Pass
    ``exclude_ribosomal_proteins=False`` for the strict technical-only set
    (equivalent to :func:`technical_rna_mask`).
    """
    groups = set(_TECHNICAL_RNA_GROUPS)
    if exclude_ribosomal_proteins:
        groups |= set(_RIBOSOMAL_PROTEIN_GROUPS)
    return _qc_group_mask(gene_table, groups)


def clean_tpm_matrix(values, removable=None, *, gene_table=None,
                     exclude_ribosomal_proteins: bool = True,
                     censored_fill: str = "typical", censored_budget=None):
    """Reference clean-TPM transform on a gene×sample matrix.

    ``values`` is genes (rows) × samples (cols). Provide either an explicit
    boolean ``removable`` mask (aligned to ``values.index``) or a ``gene_table``
    — in which case the mask is built via :func:`clean_tpm_removal_mask`, which
    **excludes ribosomal proteins by default**.

    ``censored_fill`` controls what happens to the censored (removable) rows:

    - ``"typical"`` (default): hold the censored block at a **constant typical
      polyA-capture budget** ``censored_budget`` (a single TPM total split
      equally across the censored genes), then renormalize each column to 1e6.
      Because the censored slot is sample-independent, the kept genes are scaled
      by ``1e6 / (kept_sum + budget)`` instead of ``1e6 / kept_sum`` — so a
      sample with an unusually large (often pipeline-driven) technical/ribosomal
      fraction no longer **inflates** its other genes. If ``censored_budget`` is
      ``None`` it defaults to the median censored TPM sum across ``values``'
      columns (the cohort's typical capture); pass a stored reference constant
      when cleaning a single sample.
    - ``"zero"`` (legacy): drop the censored rows entirely and renormalize the
      remainder to 1e6 (inflates survivors by ``1/(1-censored_fraction)``).

    Columns whose post-removal denominator is <= 0 become all-zero.
    """
    if removable is None:
        if gene_table is None:
            raise ValueError("clean_tpm_matrix needs either removable or gene_table")
        removable = clean_tpm_removal_mask(
            gene_table, exclude_ribosomal_proteins=exclude_ribosomal_proteins)
    import numpy as np
    import pandas as pd

    rem = removable.to_numpy()
    if censored_fill == "zero":
        clean = values.copy()
        clean.loc[rem, :] = 0.0
        denom = clean.sum(axis=0)
    elif censored_fill == "typical":
        kept_sum = values.loc[~rem].sum(axis=0)
        if censored_budget is None:
            censored_budget = float(values.loc[rem].sum(axis=0).median()) if rem.any() else 0.0
        n_cens = int(rem.sum())
        clean = values.copy()
        # constant typical value per censored gene (sample-independent budget):
        # every censored gene always receives the surrogate value (it is never
        # zeroed), so even an all-censored column keeps its surrogate signal.
        if n_cens:
            clean.loc[rem, :] = censored_budget / n_cens
        denom = kept_sum + censored_budget
    else:
        raise ValueError("censored_fill must be 'typical' or 'zero'")

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
    "clean_tpm_matrix",
]
