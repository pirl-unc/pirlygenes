# Licensed under the Apache License, Version 2.0

"""Core decomposition engine using constrained NNLS.

For each gene g in a sample:
    expression_g = sum_k (fraction_k * reference_g_k)

All expression vectors are HK-normalized (fold-over-housekeeping median)
before NNLS to make HPA nTPM, TCGA FPKM, and sample TPM comparable.
The final per-gene attribution is converted back to TPM scale.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signature import build_signature_matrix
from .templates import (
    TEMPLATES,
    get_template_components,
)
from ..gene_sets_cancer import pan_cancer_expression, housekeeping_gene_ids
from ..tumor_purity import TCGA_MEDIAN_PURITY


@dataclass
class DecompositionResult:
    """Result of decomposing a sample into cell type components."""
    template: str
    cancer_type: str
    fractions: dict  # {component: fraction}
    purity: float  # tumor fraction
    reconstruction_error: float  # NNLS residual (in fold-HK space)
    gene_attribution: pd.DataFrame  # per-gene decomposition (TPM scale)
    score: float  # overall quality score
    description: str = ""


def _hk_normalize(values, genes, hk_gene_set):
    """Normalize an expression vector by its HK gene median.

    Returns (normalized_values, hk_median).
    """
    hk_vals = [values[i] for i, g in enumerate(genes) if g in hk_gene_set and values[i] > 0]
    hk_med = float(np.median(hk_vals)) if hk_vals else 1.0
    if hk_med <= 0:
        hk_med = 1.0
    return values / hk_med, hk_med


def _get_tumor_profile(cancer_type, genes, hk_gene_set):
    """Get HK-normalized tumor reference profile for a cancer type.

    Uses TCGA median FPKM / HK_median_FPKM / cohort_purity.
    """
    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")

    cancer_col = f"FPKM_{cancer_type}"
    if cancer_col not in ref_dedup.columns:
        return np.zeros(len(genes))

    purity = TCGA_MEDIAN_PURITY.get(cancer_type, 0.5)

    # Raw FPKM profile
    profile = np.zeros(len(genes))
    for i, gid in enumerate(genes):
        if gid in ref_dedup.index:
            raw = float(ref_dedup.loc[gid, cancer_col])
            if np.isfinite(raw):
                profile[i] = raw

    # HK-normalize, then adjust for purity
    profile_hk, _ = _hk_normalize(profile, genes, hk_gene_set)
    # Rough deconvolution: tumor_fold ≈ bulk_fold / purity
    # (assumes TME contribution is small relative to HK normalization)
    return profile_hk / max(purity, 0.1)


def _constrained_nnls(A, b, sum_to_one_weight=10.0):
    """NNLS with soft sum-to-one constraint."""
    from scipy.optimize import nnls

    penalty_row = np.full((1, A.shape[1]), sum_to_one_weight)
    A_aug = np.vstack([A, penalty_row])
    b_aug = np.append(b, sum_to_one_weight)

    fractions, _ = nnls(A_aug, b_aug)

    # Normalize to sum to 1
    total = fractions.sum()
    if total > 0:
        fractions = fractions / total

    # Residual on original system
    reconstructed = A @ fractions
    residual = float(np.sqrt(np.mean((b - reconstructed) ** 2)))

    return fractions, residual


def _decompose_one(sample_expr_hk, genes, symbols, sig_matrix_hk,
                    comp_names, template_name, cancer_type, sample_hk_median):
    """Two-stage decomposition for one (template, cancer_type) hypothesis.

    Stage 1: Fit TME components (immune + stromal + tissue) via NNLS.
             No tumor column — we solve for the non-tumor fraction only.
    Stage 2: Residual (observed - reconstructed TME) = tumor attribution.

    All inputs are in HK-fold space. Output attribution is in TPM.
    """
    # Observed vector in HK-fold space
    observed = np.array([sample_expr_hk.get(gid, 0.0) for gid in genes], dtype=float)

    # Filter to genes with signal in either sample or reference
    mask = (observed > 0.001) | (sig_matrix_hk.max(axis=1) > 0.01)
    if mask.sum() < 50:
        mask = observed > 0

    # Stage 1: fit TME components (no tumor column)
    # Use NNLS WITHOUT sum-to-one constraint — fractions can sum to <1,
    # and the deficit is the tumor fraction.
    from scipy.optimize import nnls
    A = sig_matrix_hk[mask]
    b = observed[mask]
    raw_fracs, _ = nnls(A, b)

    # Compute TME total and tumor fraction
    reconstructed_full = sig_matrix_hk @ raw_fracs
    tme_total_hk = reconstructed_full.sum()
    sample_total_hk = observed[observed > 0].sum()
    tme_fraction = min(tme_total_hk / max(sample_total_hk, 1e-10), 1.0)
    tumor_fraction = max(0.0, 1.0 - tme_fraction)

    # Normalize TME fractions to sum to (1 - tumor_fraction)
    raw_total = raw_fracs.sum()
    if raw_total > 0:
        normed_fracs = raw_fracs * (1.0 - tumor_fraction) / raw_total
    else:
        normed_fracs = raw_fracs

    # Build fractions dict
    frac_dict = {"tumor": float(tumor_fraction)}
    for comp, f in zip(comp_names, normed_fracs):
        frac_dict[comp] = float(f)

    # Residual: how well TME explains the sample
    residual = float(np.sqrt(np.mean((b - A @ raw_fracs) ** 2)))

    # Stage 2: per-gene attribution — TME explained + residual = tumor
    attribution_rows = []
    for i, (gid, sym) in enumerate(zip(genes, symbols)):
        obs_hk = sample_expr_hk.get(gid, 0.0)
        obs_tpm = obs_hk * sample_hk_median
        if obs_tpm < 0.01:
            continue

        row = {"gene_id": gid, "symbol": sym, "observed_tpm": round(obs_tpm, 2)}
        tme_explained_hk = 0.0
        for j, comp in enumerate(comp_names):
            attr_hk = raw_fracs[j] * sig_matrix_hk[i, j]
            row[comp] = round(float(attr_hk * sample_hk_median), 2)
            tme_explained_hk += attr_hk

        # Tumor = residual (observed minus TME), clamped to >= 0
        tumor_hk = max(0.0, obs_hk - tme_explained_hk)
        tumor_tpm = tumor_hk * sample_hk_median
        row["tumor"] = round(float(tumor_tpm), 2)
        row["residual"] = 0.0  # by construction, tumor absorbs the residual
        row["tumor_fraction_of_total"] = round(
            float(tumor_tpm / obs_tpm) if obs_tpm > 0 else 0, 4)
        attribution_rows.append(row)

    attr_df = pd.DataFrame(attribution_rows)
    if not attr_df.empty:
        attr_df = attr_df.sort_values("tumor", ascending=False).reset_index(drop=True)

    score = 1.0 / (1.0 + residual)
    tmpl_desc = TEMPLATES.get(template_name, {}).get("description", template_name)

    return DecompositionResult(
        template=template_name,
        cancer_type=cancer_type,
        fractions=frac_dict,
        purity=tumor_fraction,
        reconstruction_error=residual,
        gene_attribution=attr_df,
        score=score,
        description=f"{cancer_type} — {tmpl_desc}",
    )


def decompose_sample(
    df_gene_expr,
    cancer_types=None,
    templates=None,
    top_k=3,
):
    """Decompose a sample across multiple (template, cancer_type) hypotheses.

    Parameters
    ----------
    df_gene_expr : DataFrame
        Expression data with Ensembl gene ID and TPM columns.
    cancer_types : list of str, optional
        Candidate cancer types. If None, uses top 3 from cancer type detection.
    templates : list of str, optional
        Candidate templates. If None, uses heuristics to select plausible ones.
    top_k : int
        Number of top decompositions to return.

    Returns
    -------
    list of DecompositionResult, sorted by score descending.
    """
    from ..plot import _sample_expression_by_symbol
    from .signature import _load_hpa_cell_types

    # --- HK gene set ---
    hk_ids = housekeeping_gene_ids()

    # --- Sample expression ---
    sample_raw, _ = _sample_expression_by_symbol(df_gene_expr)

    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    sym_to_eid = ref_dedup["Ensembl_Gene_ID"].to_dict()

    # Convert to Ensembl IDs
    sample_by_eid = {}
    for sym, tpm in sample_raw.items():
        eid = sym_to_eid.get(sym)
        if eid:
            sample_by_eid[eid] = tpm

    # Gene list from HPA
    hpa = _load_hpa_cell_types()
    genes = hpa["Ensembl_Gene_ID"].tolist()

    # Auto-detect cancer types
    if cancer_types is None:
        from ..plot import _compute_cancer_type_signature_stats
        stats = _compute_cancer_type_signature_stats(df_gene_expr)
        cancer_types = [s["code"] for s in stats[:3]]

    if templates is None:
        templates = ["solid_primary", "met_lymph_node", "met_liver", "met_lung"]

    # Intersect gene IDs: only genes present in BOTH HPA and sample
    gene_set = set(genes) & set(sample_by_eid.keys())

    # Run all combinations
    results = []
    for ct in cancer_types:
        for tmpl in templates:
            try:
                components = get_template_components(tmpl, ct)
                non_tumor = [c for c in components if c != "tumor"]

                # Build signature matrix (filtered to shared genes)
                filt_genes, filt_symbols, sig_raw, comp_names = build_signature_matrix(
                    non_tumor, gene_subset=gene_set)

                # HK-normalize sample to match filtered gene list
                filt_sample_vec = np.array([sample_by_eid.get(g, 0.0) for g in filt_genes])
                filt_sample_hk, filt_hk_median = _hk_normalize(
                    filt_sample_vec, filt_genes, hk_ids)
                filt_sample_expr_hk = {
                    g: float(v) for g, v in zip(filt_genes, filt_sample_hk)}

                # HK-normalize each column of the signature matrix
                sig_hk = np.zeros_like(sig_raw)
                for col_idx in range(sig_raw.shape[1]):
                    sig_hk[:, col_idx], _ = _hk_normalize(
                        sig_raw[:, col_idx], filt_genes, hk_ids)

                result = _decompose_one(
                    filt_sample_expr_hk, filt_genes, filt_symbols,
                    sig_hk, comp_names,
                    tmpl, ct, filt_hk_median,
                )
                results.append(result)
            except Exception:
                continue

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
