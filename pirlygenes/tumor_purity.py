# Licensed under the Apache License, Version 2.0

"""Tumor purity estimation from bulk RNA-seq expression data.

Uses within-sample gene-set enrichment ratios to estimate tumor content.
The key insight: the ratio (gene_set_TPM_sum / housekeeping_TPM_sum) is
platform-independent (works across TPM, FPKM, etc.) because both numerator
and denominator come from the same sample.

Comparing this ratio to the TCGA cohort reference gives a purity estimate:

    purity ≈ (tumor_signal / HK)_sample / (tumor_signal / HK)_reference

Multiple gene sets are scored independently:
- Cancer-type signature genes (auto-detected or specified)
- ESTIMATE stromal genes (Yoshihara et al. 2013, Nat Commun)
- ESTIMATE immune genes (Yoshihara et al. 2013, Nat Commun)

Higher stromal/immune scores → lower tumor purity.
"""

import numpy as np

from .gene_sets_cancer import (
    pan_cancer_expression,
    housekeeping_gene_ids,
)
from .load_dataset import get_data
from .plot import _guess_gene_cols
from .plot_data_helpers import _strip_ensembl_version


# -------------------- cancer type → normal tissue mapping --------------------

CANCER_TO_TISSUE = {
    "ACC": "adrenal_gland",
    "BLCA": "urinary_bladder",
    "BRCA": "breast",
    "CESC": "cervix",
    "CHOL": "gallbladder",
    "COAD": "colon",
    "DLBC": "lymph_node",
    "ESCA": "esophagus",
    "GBM": "cerebral_cortex",
    "HNSC": "tongue",
    "KICH": "kidney",
    "KIRC": "kidney",
    "KIRP": "kidney",
    "LAML": "bone_marrow",
    "LGG": "cerebral_cortex",
    "LIHC": "liver",
    "LUAD": "lung",
    "LUSC": "lung",
    "MESO": "lung",
    "OV": "ovary",
    "PAAD": "pancreas",
    "PCPG": "adrenal_gland",
    "PRAD": "prostate",
    "READ": "rectum",
    "SARC": "smooth_muscle",
    "SKCM": "skin",
    "STAD": "stomach",
    "TGCT": "testis",
    "THCA": "thyroid_gland",
    "THYM": "thymus",
    "UCEC": "endometrium",
    "UCS": "endometrium",
    "UVM": "retina",
}

# Median TCGA tumor purity by cancer type (from Aran et al. 2015, Nat Commun;
# consensus purity estimates across ABSOLUTE, ESTIMATE, LUMP, IHC).
# Used to calibrate: TCGA reference ≈ this purity, not 100%.
TCGA_MEDIAN_PURITY = {
    "ACC": 0.79, "BLCA": 0.59, "BRCA": 0.73, "CESC": 0.49,
    "CHOL": 0.68, "COAD": 0.59, "DLBC": 0.94, "ESCA": 0.50,
    "GBM": 0.83, "HNSC": 0.60, "KICH": 0.84, "KIRC": 0.72,
    "KIRP": 0.78, "LAML": 0.95, "LGG": 0.87, "LIHC": 0.73,
    "LUAD": 0.56, "LUSC": 0.67, "MESO": 0.55, "OV": 0.72,
    "PAAD": 0.42, "PCPG": 0.69, "PRAD": 0.69, "READ": 0.60,
    "SARC": 0.66, "SKCM": 0.65, "STAD": 0.40, "TGCT": 0.75,
    "THCA": 0.72, "THYM": 0.78, "UCEC": 0.71, "UCS": 0.65,
    "UVM": 0.85,
}


# -------------------- helpers --------------------


def _build_sample_tpm_by_symbol(df_gene_expr):
    """Return {symbol: max_TPM} from expression data (no normalization)."""
    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    ref = pan_cancer_expression()
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))

    result = {}
    for _, row in df.iterrows():
        gid = str(row[gene_id_col])
        sym = id_to_sym.get(gid)
        if sym is None:
            continue
        tpm = float(row[tpm_col])
        if sym not in result or tpm > result[sym]:
            result[sym] = tpm
    return result


def _geneset_hk_ratio(genes, hk_symbols, expr_by_symbol):
    """Sum of gene set expression / sum of housekeeping expression.

    This ratio is platform-independent (cancels out the TPM/FPKM
    scaling factor since both come from the same sample/column).
    """
    gs_sum = sum(expr_by_symbol.get(g, 0) for g in genes)
    hk_sum = sum(expr_by_symbol.get(g, 0) for g in hk_symbols)
    if hk_sum <= 0:
        return 0.0
    return gs_sum / hk_sum


def _select_tumor_specific_genes(cancer_code, n=30):
    """Select genes highly expressed in cancer but NOT in matched normal tissue.

    Returns list of gene symbols sorted by tumor specificity.
    """
    ref = pan_cancer_expression(normalize="housekeeping")
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]

    cancer_col = f"FPKM_{cancer_code}"
    if cancer_col not in ref_by_sym.columns:
        return []

    tissue = CANCER_TO_TISSUE.get(cancer_code)
    normal_col = f"nTPM_{tissue}" if tissue else None

    # Z-score across cancer types for initial ranking
    expr_matrix = ref_by_sym[fpkm_cols].astype(float)
    gene_mean = expr_matrix.mean(axis=1)
    gene_std = expr_matrix.std(axis=1).replace(0, np.nan)
    z_scores = ((expr_matrix[cancer_col] - gene_mean) / gene_std).fillna(0)

    # Candidates: high z-score AND meaningful expression
    candidates = z_scores[expr_matrix[cancer_col] > 0.01].nlargest(n * 5)

    markers = []
    for gene in candidates.index:
        cancer_hk = float(ref_by_sym.loc[gene, cancer_col])
        if normal_col and normal_col in ref_by_sym.columns:
            normal_hk = float(ref_by_sym.loc[gene, normal_col])
        else:
            normal_hk = 0.0

        # Require cancer expression to meaningfully exceed normal tissue
        delta = cancer_hk - normal_hk
        if delta > 0 and delta / (cancer_hk + 0.001) >= 0.3:
            markers.append(gene)
            if len(markers) >= n:
                break

    return markers


# -------------------- main estimation --------------------


def estimate_tumor_purity(df_gene_expr, cancer_type=None):
    """Estimate tumor purity from expression data.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression data with gene ID, gene name, and TPM columns.
    cancer_type : str or None
        TCGA cancer type code or alias. If None, auto-detected.

    Returns
    -------
    dict
        cancer_type : str — TCGA code
        overall_estimate : float — purity estimate (0–1)
        overall_lower : float — lower bound
        overall_upper : float — upper bound
        components : dict — per-component details
    """
    from .plot import (
        _compute_cancer_type_signature_stats,
        resolve_cancer_type,
    )

    # Auto-detect cancer type
    if cancer_type is None:
        stats = _compute_cancer_type_signature_stats(df_gene_expr)
        cancer_code = stats[0]["code"]
        cancer_score = stats[0]["score"]
    else:
        cancer_code = resolve_cancer_type(cancer_type)
        cancer_score = None

    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)

    # Reference expression by symbol (raw FPKM, within-dataset)
    ref = pan_cancer_expression()
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ref_expr = ref_by_sym[f"FPKM_{cancer_code}"].to_dict()

    # HK gene symbols
    hk_ids = housekeeping_gene_ids()
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    hk_syms = [id_to_sym[gid] for gid in hk_ids if gid in id_to_sym]

    # TCGA reference purity (the TCGA cohort is NOT 100% pure)
    tcga_purity = TCGA_MEDIAN_PURITY.get(cancer_code, 0.7)

    # ---- Component 1: Cancer-type signature genes ----
    sig_genes = _select_tumor_specific_genes(cancer_code, n=30)
    sig_sample_ratio = _geneset_hk_ratio(sig_genes, hk_syms, sample_tpm)
    sig_ref_ratio = _geneset_hk_ratio(sig_genes, hk_syms, ref_expr)

    # Per-gene estimates for bounds
    per_gene = []
    for gene in sig_genes:
        s = sample_tpm.get(gene, 0)
        s_hk = _geneset_hk_ratio([gene], hk_syms, sample_tpm)
        r_hk = _geneset_hk_ratio([gene], hk_syms, ref_expr)
        if r_hk > 0.001:
            raw_p = s_hk / r_hk
            # Calibrate: TCGA reference ≈ tcga_purity, not 100%
            calibrated_p = raw_p * tcga_purity
            per_gene.append({
                "gene": gene,
                "sample_tpm": s,
                "purity_raw": float(raw_p),
                "purity": float(np.clip(calibrated_p, 0, 1)),
            })

    if sig_ref_ratio > 0:
        sig_purity_raw = sig_sample_ratio / sig_ref_ratio
        sig_purity = float(np.clip(sig_purity_raw * tcga_purity, 0, 1))
    else:
        sig_purity = None

    per_gene_purities = [g["purity"] for g in per_gene]
    sig_lower = float(np.percentile(per_gene_purities, 25)) if per_gene_purities else None
    sig_upper = float(np.percentile(per_gene_purities, 75)) if per_gene_purities else None

    # ---- Component 2: ESTIMATE stromal genes ----
    try:
        est_df = get_data("estimate-signatures")
        stromal_genes = est_df[est_df["Category"] == "Stromal"]["Symbol"].tolist()
        immune_genes = est_df[est_df["Category"] == "Immune"]["Symbol"].tolist()
    except Exception:
        stromal_genes = []
        immune_genes = []

    stromal_sample = _geneset_hk_ratio(stromal_genes, hk_syms, sample_tpm)
    stromal_ref = _geneset_hk_ratio(stromal_genes, hk_syms, ref_expr)
    immune_sample = _geneset_hk_ratio(immune_genes, hk_syms, sample_tpm)
    immune_ref = _geneset_hk_ratio(immune_genes, hk_syms, ref_expr)

    # Stromal/immune enrichment relative to TCGA reference.
    # In TCGA, (1 - tcga_purity) is already stroma+immune.
    # If sample has more stromal signal → lower purity.
    stromal_enrichment = stromal_sample / stromal_ref if stromal_ref > 0 else 1.0
    immune_enrichment = immune_sample / immune_ref if immune_ref > 0 else 1.0

    # Estimate non-tumor fraction from enrichment
    # TCGA non-tumor fraction = 1 - tcga_purity
    # Sample non-tumor fraction ≈ (1 - tcga_purity) × enrichment
    tcga_nontumor = 1 - tcga_purity
    stromal_nontumor = min(tcga_nontumor * stromal_enrichment, 1.0)
    immune_nontumor = min(tcga_nontumor * immune_enrichment, 1.0)
    estimate_purity = float(np.clip(
        1.0 - (stromal_nontumor + immune_nontumor) / 2,
        0, 1,
    ))

    # ---- Combine estimates ----
    estimates = []
    if sig_purity is not None:
        estimates.append(sig_purity)
    if stromal_genes:
        estimates.append(estimate_purity)

    if estimates:
        overall = float(np.median(estimates))
        all_bounds = []
        if sig_lower is not None:
            all_bounds.extend([sig_lower, sig_upper])
        all_bounds.append(estimate_purity)
        if sig_purity is not None:
            all_bounds.append(sig_purity)
        overall_lower = float(min(all_bounds))
        overall_upper = float(max(all_bounds))
    else:
        overall = overall_lower = overall_upper = None

    return {
        "cancer_type": cancer_code,
        "cancer_type_score": cancer_score,
        "tissue": CANCER_TO_TISSUE.get(cancer_code),
        "tcga_median_purity": tcga_purity,
        "overall_estimate": overall,
        "overall_lower": overall_lower,
        "overall_upper": overall_upper,
        "components": {
            "signature": {
                "genes": sig_genes,
                "purity": sig_purity,
                "lower": sig_lower,
                "upper": sig_upper,
                "per_gene": per_gene,
            },
            "stromal": {
                "enrichment": stromal_enrichment,
                "n_genes": len(stromal_genes),
            },
            "immune": {
                "enrichment": immune_enrichment,
                "n_genes": len(immune_genes),
            },
            "estimate_purity": estimate_purity,
        },
    }


# -------------------- plotting --------------------


def plot_tumor_purity(
    df_gene_expr,
    cancer_type=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 8),
):
    """Plot tumor purity estimation with all components.

    Shows:
    - Left panel: per-gene purity estimates from cancer-type signature
    - Right panel: component summary (signature, stromal, immune, combined)
    """
    import matplotlib.pyplot as plt

    result = estimate_tumor_purity(df_gene_expr, cancer_type=cancer_type)
    cancer_code = result["cancer_type"]
    comp = result["components"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})

    # ---- Left: per-gene purity estimates ----
    per_gene = comp["signature"]["per_gene"]
    if per_gene:
        per_gene_sorted = sorted(per_gene, key=lambda g: -g["purity"])
        genes = [g["gene"] for g in per_gene_sorted]
        purities = [g["purity"] * 100 for g in per_gene_sorted]
        y = np.arange(len(genes))

        colors = [plt.cm.RdYlGn(p / 100) for p in purities]
        ax1.barh(y, purities, color=colors, edgecolor="none", height=0.7)

        ax1.set_yticks(y)
        ax1.set_yticklabels(genes, fontsize=8)
        ax1.set_xlabel("Purity estimate (%)", fontsize=10)
        ax1.set_title(
            f"{cancer_code} signature gene purity estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, calibrated for "
            f"TCGA median purity {result['tcga_median_purity']:.0%})",
            fontsize=10,
        )
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.axvline(
            x=comp["signature"]["purity"] * 100 if comp["signature"]["purity"] else 0,
            color="black", linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"Aggregate: {comp['signature']['purity']:.0%}" if comp["signature"]["purity"] else "",
        )
        ax1.legend(loc="lower right", fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No tumor-specific signature genes found",
                 ha="center", va="center", transform=ax1.transAxes)

    # ---- Right: component summary ----
    components = []

    if comp["signature"]["purity"] is not None:
        components.append((
            f"Tumor signature\n({len(comp['signature']['genes'])} genes)",
            comp["signature"]["purity"] * 100,
            comp["signature"]["lower"] * 100 if comp["signature"]["lower"] is not None else None,
            comp["signature"]["upper"] * 100 if comp["signature"]["upper"] is not None else None,
            "#2166ac",
        ))

    components.append((
        f"ESTIMATE stromal\n({comp['stromal']['n_genes']} genes)",
        None,  # not a direct purity
        None, None,
        "#d6604d",
    ))
    components.append((
        f"ESTIMATE immune\n({comp['immune']['n_genes']} genes)",
        None,
        None, None,
        "#4393c3",
    ))

    if comp.get("estimate_purity") is not None:
        components.append((
            "ESTIMATE combined\n(1 − infiltration)",
            comp["estimate_purity"] * 100,
            None, None,
            "#762a83",
        ))

    if result["overall_estimate"] is not None:
        components.append((
            "Overall estimate",
            result["overall_estimate"] * 100,
            result["overall_lower"] * 100,
            result["overall_upper"] * 100,
            "#1a1a1a",
        ))

    y_positions = []
    y_labels = []
    y_pos = 0
    for name, purity, lower, upper, color in components:
        y_positions.append(y_pos)
        y_labels.append(name)

        if purity is not None:
            ax2.barh(y_pos, purity, color=color, edgecolor="none", height=0.6, alpha=0.8)
            ax2.text(purity + 1, y_pos, f"{purity:.0f}%", va="center", fontsize=9, fontweight="bold")

            if lower is not None and upper is not None:
                ax2.plot([lower, upper], [y_pos, y_pos], color=color,
                         linewidth=3, alpha=0.4, solid_capstyle="round")
        else:
            # Show enrichment for stromal/immune
            if "stromal" in name.lower():
                enr = comp["stromal"]["enrichment"]
                label = f"{enr:.1f}× vs TCGA"
                bar_val = min(enr / 5 * 100, 100)  # scale to 0-100
            elif "immune" in name.lower():
                enr = comp["immune"]["enrichment"]
                label = f"{enr:.1f}× vs TCGA"
                bar_val = min(enr / 5 * 100, 100)
            else:
                continue
            ax2.barh(y_pos, bar_val, color=color, edgecolor="none", height=0.6, alpha=0.4)
            ax2.text(bar_val + 1, y_pos, label, va="center", fontsize=9)

        y_pos += 1

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Purity (%) / enrichment", fontsize=10)
    ax2.set_title("Purity components", fontsize=11)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"Tumor purity estimate: {result['overall_estimate']:.0%} "
        f"[{result['overall_lower']:.0%}–{result['overall_upper']:.0%}]"
        if result["overall_estimate"] is not None
        else "Tumor purity estimate: N/A",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, result
