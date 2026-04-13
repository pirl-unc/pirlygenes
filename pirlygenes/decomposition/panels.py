# Licensed under the Apache License, Version 2.0

"""Tumor-biased / matched-normal-biased / shared-lineage gene panels.

Supports the matched-normal lineage-splitting work in issue #50. For each
epithelial cancer type with a defensible matched-normal tissue, we want
three gene categories:

- **tumor-biased**: genes meaningfully higher in the TCGA cancer cohort
  than in the matched-normal bulk tissue. These carry the signal that a
  three-component decomposition relies on to distinguish tumor from
  benign parent tissue.

- **matched-normal-biased**: genes meaningfully higher in the matched
  normal than in the cancer cohort. Useful for sanity-checking that a
  sample actually contains benign parent tissue and not just stroma.

- **shared-lineage**: genes expressed comparably in both, e.g. retained
  luminal / glandular markers. Explicitly excluded from drive-the-fit
  marker sets — these cannot discriminate tumor from matched normal.

These utilities operate on the bundled ``pan_cancer_expression`` table
(FPKM_<cancer> cancer cohort medians and nTPM_<tissue> bulk normal
references). They are currently provided for downstream calibration /
inspection and are NOT wired into the decomposition marker selection.

That wiring — using tumor-biased genes as anchoring markers so the NNLS
can reliably allocate between tumor and matched_normal compartments — is
the step that will let the matched-normal feature be turned on by
default. See issue #50 step 2–3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..gene_sets_cancer import pan_cancer_expression
from ..tumor_purity import TCGA_MEDIAN_PURITY
from .templates import EPITHELIAL_MATCHED_NORMAL_TISSUE


def _load_cohort_vs_tissue(cancer_code, tissue=None):
    """Return a DataFrame with per-gene tumor-cell and matched-normal estimates.

    TCGA cohort medians are bulk, not tumor-cell-only (median TCGA purity
    is 0.4–0.8 depending on cancer type). A raw ``FPKM_<cancer> /
    nTPM_<tissue>`` ratio therefore understates the tumor-vs-normal
    difference for genes where benign admixed parent tissue contributes
    meaningfully to the TCGA bulk average — the exact class of genes a
    matched-normal panel cares about (AMACR in PRAD, CDX2 in COAD, ...).

    This loader deconvolves the TCGA cohort median to a crude tumor-cell-
    only estimate using the published TCGA median purity and the matched-
    normal bulk as the non-tumor background:

        tumor_cell ≈ max(0, (FPKM_cancer - (1 - p) * nTPM_tissue) / p)

    The TCGA_MEDIAN_PURITY dict is the same calibration constant used in
    ``estimate_tumor_expression_ranges`` to build the per-gene cohort
    prior, so the two code paths use a consistent definition.

    Returns ``(df, tissue)`` where ``df`` has columns ``symbol``,
    ``tumor_fpkm`` (deconvolved tumor-cell estimate), ``normal_ntpm``,
    and ``tcga_bulk_fpkm`` (the raw cohort median, retained for
    inspection).
    """
    if tissue is None:
        tissue = EPITHELIAL_MATCHED_NORMAL_TISSUE.get(cancer_code)
        if tissue is None:
            raise ValueError(
                f"{cancer_code} has no default matched-normal tissue; "
                f"pass `tissue=` explicitly or extend "
                f"EPITHELIAL_MATCHED_NORMAL_TISSUE."
            )

    ref = pan_cancer_expression().drop_duplicates(subset="Symbol")
    cancer_col = f"FPKM_{cancer_code}"
    tissue_col = f"nTPM_{tissue}"
    missing = [c for c in (cancer_col, tissue_col) if c not in ref.columns]
    if missing:
        raise KeyError(f"Missing reference columns: {missing}")

    cohort_bulk = ref[cancer_col].astype(float)
    normal_ntpm = ref[tissue_col].astype(float)
    purity = float(TCGA_MEDIAN_PURITY.get(cancer_code, 0.7))
    tumor_cell = np.maximum(
        0.0, (cohort_bulk - (1.0 - purity) * normal_ntpm) / purity,
    )

    out = pd.DataFrame(
        {
            "symbol": ref["Symbol"].astype(str),
            "tumor_fpkm": tumor_cell,
            "normal_ntpm": normal_ntpm,
            "tcga_bulk_fpkm": cohort_bulk,
        }
    )
    return out, tissue


def _log2_ratio(a, b, floor=0.1):
    """Symmetric, floor-stabilized log2 ratio ``log2((a + floor) / (b + floor))``."""
    return np.log2((np.asarray(a, dtype=float) + floor) / (np.asarray(b, dtype=float) + floor))


def build_tumor_biased_panel(
    cancer_code,
    tissue=None,
    delta_log2=1.0,
    min_tumor_expression=1.0,
):
    """Genes meaningfully higher in the TCGA cancer cohort than in the
    matched-normal tissue.

    Parameters
    ----------
    cancer_code : str
        TCGA cancer code (e.g. ``"PRAD"``).
    tissue : str, optional
        Matched-normal tissue (e.g. ``"prostate"``). Defaults to
        :data:`EPITHELIAL_MATCHED_NORMAL_TISSUE`.
    delta_log2 : float
        Minimum log2(tumor / normal) ratio. ``1.0`` ≡ ≥2× tumor-biased.
    min_tumor_expression : float
        Require ``FPKM_<cancer> >= min_tumor_expression`` so we don't
        retain noisy ratios driven purely by the floor term.

    Returns
    -------
    DataFrame with columns ``symbol, tumor_fpkm, normal_ntpm,
    log2_ratio``, sorted by ``log2_ratio`` descending.
    """
    df, _tissue = _load_cohort_vs_tissue(cancer_code, tissue=tissue)
    df["log2_ratio"] = _log2_ratio(df["tumor_fpkm"], df["normal_ntpm"])
    keep = (df["tumor_fpkm"] >= min_tumor_expression) & (df["log2_ratio"] >= delta_log2)
    return df[keep].sort_values("log2_ratio", ascending=False).reset_index(drop=True)


def build_matched_normal_biased_panel(
    cancer_code,
    tissue=None,
    delta_log2=1.0,
    min_normal_expression=1.0,
):
    """Genes meaningfully higher in the matched-normal tissue than in
    the TCGA cancer cohort.

    Same parameters as :func:`build_tumor_biased_panel`, direction
    reversed.
    """
    df, _tissue = _load_cohort_vs_tissue(cancer_code, tissue=tissue)
    df["log2_ratio"] = _log2_ratio(df["normal_ntpm"], df["tumor_fpkm"])
    keep = (df["normal_ntpm"] >= min_normal_expression) & (df["log2_ratio"] >= delta_log2)
    return df[keep].sort_values("log2_ratio", ascending=False).reset_index(drop=True)


def build_shared_lineage_panel(
    cancer_code,
    tissue=None,
    tolerance_log2=1.0,
    min_expression=5.0,
):
    """Genes expressed at similar levels in tumor-cohort bulk and matched-normal bulk.

    Useful to flag retained-lineage markers (e.g. KLK3 in PRAD, SFTPB in
    LUAD) that cannot distinguish tumor from benign parent tissue.

    Note: the comparison uses **raw bulk** (``tcga_bulk_fpkm`` vs
    ``nTPM_tissue``), not the deconvolved tumor-cell estimate. A naive
    TCGA deconvolution subtracts the full normal contribution, which by
    construction pulls lineage-shared genes toward zero and would remove
    them from a "shared" panel — precisely the wrong answer. Raw bulk
    comparison captures the intended semantic: both cohorts express the
    gene at comparable magnitudes.

    Parameters
    ----------
    tolerance_log2 : float
        Max abs log2 ratio to count as "shared". ``1.0`` ≡ within ~2x.
    min_expression : float
        Require both cohort bulk and normal reference to exceed this
        floor so the ratio is meaningful.

    Returns
    -------
    DataFrame with columns ``symbol, tumor_fpkm, normal_ntpm,
    tcga_bulk_fpkm, abs_log2_ratio``.
    """
    df, _tissue = _load_cohort_vs_tissue(cancer_code, tissue=tissue)
    df["abs_log2_ratio"] = np.abs(_log2_ratio(df["tcga_bulk_fpkm"], df["normal_ntpm"]))
    keep = (
        (df["tcga_bulk_fpkm"] >= min_expression)
        & (df["normal_ntpm"] >= min_expression)
        & (df["abs_log2_ratio"] <= tolerance_log2)
    )
    return df[keep].sort_values("abs_log2_ratio").reset_index(drop=True)


def summarize_panels(cancer_code, tissue=None):
    """Convenience snapshot of panel sizes for a cancer type.

    Returns a dict with counts of tumor-biased, matched-normal-biased,
    and shared-lineage genes at the default thresholds. Useful for
    eyeballing whether a cancer type has enough discriminative signal
    to drive a three-component decomposition.
    """
    return {
        "cancer_code": cancer_code,
        "tissue": tissue or EPITHELIAL_MATCHED_NORMAL_TISSUE.get(cancer_code),
        "tumor_biased": len(build_tumor_biased_panel(cancer_code, tissue=tissue)),
        "matched_normal_biased": len(
            build_matched_normal_biased_panel(cancer_code, tissue=tissue)
        ),
        "shared_lineage": len(build_shared_lineage_panel(cancer_code, tissue=tissue)),
    }
