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


# Lineage genes per cancer type — genes retained in metastases and specific
# enough to calibrate purity.  Only genes with low TME background and high
# expression in the origin tissue should be listed.
LINEAGE_GENES = {
    # Genitourinary
    "PRAD": ["STEAP1", "STEAP2", "FOLH1", "TMPRSS2", "KLK3", "KLK2", "NKX3-1", "HOXB13", "AR"],
    "BLCA": ["UPK1A", "UPK2", "UPK3A", "KRT20", "GATA3", "PPARG"],
    "TGCT": ["POU5F1", "NANOG", "SOX17", "TFAP2C", "KIT"],
    # Breast / gynecologic
    "BRCA": ["ESR1", "GATA3", "FOXA1", "TFF1", "TFF3", "AGR2"],
    "OV":   ["PAX8", "WT1", "MUC16", "MSLN", "FOLR1"],
    "UCEC": ["PAX8", "ESR1", "PGR", "MSX1", "HOXA10"],
    "UCS":  ["PAX8", "ESR1", "PGR", "MSX1"],
    "CESC": ["TP63", "SOX2", "KRT17", "CDKN2A", "DSG3"],
    # Lung
    "LUAD": ["NKX2-1", "NAPSA", "SFTPB", "SFTPC"],
    "LUSC": ["TP63", "SOX2", "KRT5", "KRT14"],
    "MESO": ["MSLN", "WT1", "CALB2", "BAP1"],
    # GI
    "COAD": ["CDX2", "MUC2", "VIL1", "CDH17"],
    "READ": ["CDX2", "MUC2", "VIL1", "CDH17"],
    "STAD": ["MUC5AC", "MUC6", "CDX2", "CLDN18"],
    "ESCA": ["TP63", "SOX2", "KRT5", "KRT14", "CDX2"],
    "LIHC": ["ALB", "APOB", "HNF4A", "AFP"],
    "CHOL": ["KRT7", "KRT19", "SOX9", "HNF1B", "EPCAM"],
    "PAAD": ["PDX1", "PTF1A", "KRT19", "MUC1"],
    # Kidney
    "KIRC": ["CA9", "PAX8", "NDUFA4L2"],
    "KIRP": ["PAX8", "PAX2", "AMACR"],
    "KICH": ["KIT", "PAX8", "FOXI1"],
    # CNS
    "GBM":  ["GFAP", "OLIG2", "SOX2"],
    "LGG":  ["GFAP", "OLIG2", "IDH1", "ATRX"],
    # Endocrine
    "THCA": ["TG", "TPO", "PAX8", "NIS"],
    "ACC":  ["CYP11B1", "CYP11B2", "CYP21A2", "STAR", "NR5A1"],
    "PCPG": ["TH", "DBH", "CHGA", "CHGB", "PHOX2B"],
    # Skin / soft tissue
    "SKCM": ["MLANA", "PMEL", "TYR", "DCT", "MITF"],
    "UVM":  ["MLANA", "PMEL", "TYR", "MITF"],
    "SARC": ["DES", "ACTA2", "MYOD1", "MYOG"],
    # Hematologic
    "LAML": ["MPO", "CD34", "KIT", "FLT3"],
    "DLBC": ["CD19", "CD20", "PAX5", "BCL6", "IRF4"],
    "THYM": ["CD3D", "CD3E", "CD3G", "LCK", "ZAP70"],
    # Head and neck
    "HNSC": ["TP63", "SOX2", "KRT5", "KRT14", "CDKN2A"],
}


def _lineage_purity_estimates(cancer_code, sample_tpm, ref_by_sym, hk_syms, tcga_purity):
    """Estimate purity from cancer-type lineage genes using HK-normalized ratios.

    For each lineage gene, computes:
        sample_ratio = gene_sample / HK_sample
        ref_ratio    = gene_TCGA  / HK_TCGA
        tme_ratio    = median(gene_tissue / HK_tissue) across TME tissues
        true_tumor_ratio = (ref_ratio - (1-tcga_purity) * tme_ratio) / tcga_purity
        purity = (sample_ratio - tme_ratio) / (true_tumor_ratio - tme_ratio)

    Returns list of dicts with per-gene purity estimates.
    """
    genes = LINEAGE_GENES.get(cancer_code, [])
    if not genes:
        return []

    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    _repro = {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}
    ntpm_nonrepro = [c for c in ntpm_cols if c.replace("nTPM_", "") not in _repro]

    # Curated TME tissues (immune organs + stromal/connective)
    _tme = {
        "bone_marrow", "lymph_node", "spleen", "thymus", "tonsil", "appendix",
        "smooth_muscle", "skeletal_muscle", "heart_muscle", "adipose_tissue",
    }
    tme_cols = [c for c in ntpm_nonrepro if c.replace("nTPM_", "") in _tme]

    # HK symbols in reference
    hk_in_ref = [s for s in hk_syms if s in ref_dedup.index]

    # HK medians per column
    cancer_col = f"FPKM_{cancer_code}"
    if cancer_col not in ref_dedup.columns:
        return []
    ref_hk_cancer = ref_dedup.loc[hk_in_ref, cancer_col].astype(float).median()
    if ref_hk_cancer <= 0:
        return []

    tme_hk_medians = {}
    for col in tme_cols:
        tme_hk_medians[col] = ref_dedup.loc[hk_in_ref, col].astype(float).median()

    # Sample HK (median of expressed HK genes)
    sample_hk_vals = [sample_tpm[g] for g in hk_syms if sample_tpm.get(g, 0) > 0]
    sample_hk_med = float(np.median(sample_hk_vals)) if sample_hk_vals else 0.0
    if sample_hk_med <= 0:
        return []

    results = []
    for gene in genes:
        if gene not in ref_dedup.index:
            continue
        s_tpm = sample_tpm.get(gene, 0)
        if s_tpm <= 0:
            continue

        sample_ratio = s_tpm / sample_hk_med
        ref_ratio = float(ref_dedup.loc[gene, cancer_col]) / ref_hk_cancer

        # TME ratio: median across TME tissues (HK-normalized)
        tme_ratios = []
        for col in tme_cols:
            hk_m = tme_hk_medians[col]
            if hk_m > 0:
                tme_ratios.append(float(ref_dedup.loc[gene, col]) / hk_m)
        tme_ratio = float(np.median(tme_ratios)) if tme_ratios else 0.0

        # Deconvolve TCGA to get true tumor ratio
        true_tumor_ratio = (ref_ratio - (1 - tcga_purity) * tme_ratio) / tcga_purity

        if true_tumor_ratio <= tme_ratio:
            continue

        purity = (sample_ratio - tme_ratio) / (true_tumor_ratio - tme_ratio)
        purity = float(np.clip(purity, 0, 1))

        results.append({
            "gene": gene,
            "sample_tpm": s_tpm,
            "sample_ratio": float(sample_ratio),
            "ref_ratio": float(ref_ratio),
            "tme_ratio": float(tme_ratio),
            "tumor_ratio": float(true_tumor_ratio),
            "purity": purity,
        })

    return results


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

    # ---- Component 3: Lineage gene refinement ----
    lineage_per_gene = _lineage_purity_estimates(
        cancer_code, sample_tpm, ref_by_sym, hk_syms, tcga_purity,
    )
    lineage_purities = sorted(g["purity"] for g in lineage_per_gene if g["purity"] > 0)
    if len(lineage_purities) >= 3:
        # Use upper-half median: genes giving LOW estimates likely
        # de-differentiated (lost expression) rather than indicating
        # low purity.  Genes giving HIGH estimates are reliable —
        # their signal can't be explained by gene loss.
        mid = len(lineage_purities) // 2
        upper_half = lineage_purities[mid:]
        lineage_purity = float(np.median(upper_half))
        lineage_lower = float(np.percentile(upper_half, 25))
        lineage_upper = float(np.percentile(upper_half, 75))
    else:
        lineage_purity = lineage_lower = lineage_upper = None

    # ---- Combine estimates ----
    estimates = []
    if sig_purity is not None:
        estimates.append(sig_purity)
    if stromal_genes:
        estimates.append(estimate_purity)
    if lineage_purity is not None:
        estimates.append(lineage_purity)

    if estimates:
        overall = float(np.median(estimates))
        # Bounds: use lineage IQR when available (tighter), else signature + ESTIMATE
        if lineage_lower is not None:
            overall_lower = lineage_lower
            overall_upper = lineage_upper
        else:
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
            "lineage": {
                "genes": [g["gene"] for g in lineage_per_gene],
                "purity": lineage_purity,
                "lower": lineage_lower,
                "upper": lineage_upper,
                "per_gene": lineage_per_gene,
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


# -------------------- tissue scoring --------------------


def _score_normal_tissues(sample_tpm_by_symbol, top_n=10):
    """Score each HPA normal tissue by signature gene expression in sample.

    For each tissue, selects genes most specifically expressed in that tissue
    (by z-score across 50 tissues) and computes the sample's mean midrank
    percentile for those genes.

    Returns sorted list of (tissue, score, n_genes).
    """
    ref = pan_cancer_expression()
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]

    expr = ref_by_sym[ntpm_cols].astype(float)
    gene_mean = expr.mean(axis=1)
    gene_std = expr.std(axis=1).replace(0, np.nan)
    z_matrix = expr.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0)

    results = []
    for col in ntpm_cols:
        tissue = col.replace("nTPM_", "")
        z_col = z_matrix[col]
        expr_col = expr[col]
        sig_genes = list(z_col[expr_col > 0.5].nlargest(20).index)
        if len(sig_genes) < 5:
            continue

        pcts = []
        for gene in sig_genes:
            s_val = sample_tpm_by_symbol.get(gene, 0)
            if gene in expr.index:
                ref_vals = expr.loc[gene].values
                n = len(ref_vals)
                below = np.sum(ref_vals < s_val)
                equal = np.sum(np.isclose(ref_vals, s_val, atol=0.01))
                pcts.append((below + 0.5 * equal) / n)
        if pcts:
            results.append((tissue, float(np.mean(pcts)), len(pcts)))

    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def _get_mhc_expression(sample_tpm_by_symbol):
    """Get MHC class I and II expression levels."""
    mhc1_genes = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]
    mhc2_genes = ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"]

    mhc1 = {g: sample_tpm_by_symbol.get(g, 0) for g in mhc1_genes}
    mhc2 = {g: sample_tpm_by_symbol.get(g, 0) for g in mhc2_genes}
    return mhc1, mhc2


# -------------------- comprehensive summary --------------------


def analyze_sample(df_gene_expr, cancer_type=None):
    """Comprehensive sample composition analysis.

    Returns a dict with all analysis results: cancer type, purity,
    tissue context, MHC status, and narrative interpretation.
    """
    from .plot import (
        _compute_cancer_type_signature_stats,
        resolve_cancer_type,
        CANCER_TYPE_NAMES,
    )

    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)

    # 1. Cancer type
    stats = _compute_cancer_type_signature_stats(df_gene_expr)
    if cancer_type:
        cancer_code = resolve_cancer_type(cancer_type)
    else:
        cancer_code = stats[0]["code"]
    cancer_name = CANCER_TYPE_NAMES.get(cancer_code, cancer_code)
    cancer_score = stats[0]["score"] if stats[0]["code"] == cancer_code else None

    # 2. Purity
    purity = estimate_tumor_purity(df_gene_expr, cancer_type=cancer_code)

    # 3. Tissue context
    tissue_scores = _score_normal_tissues(sample_tpm)

    # 4. MHC expression
    mhc1, mhc2 = _get_mhc_expression(sample_tpm)

    # 5. Top cancer type matches
    top_cancers = [(s["code"], s["score"]) for s in stats[:5]]

    return {
        "cancer_type": cancer_code,
        "cancer_name": cancer_name,
        "cancer_score": cancer_score,
        "top_cancers": top_cancers,
        "purity": purity,
        "tissue_scores": tissue_scores,
        "mhc1": mhc1,
        "mhc2": mhc2,
    }


def plot_sample_summary(
    df_gene_expr,
    cancer_type=None,
    save_to_filename=None,
    save_dpi=300,
):
    """Comprehensive sample composition plot.

    Four-panel figure:
    - Top-left: cancer type identification (bar chart)
    - Top-right: tumor purity and microenvironment composition
    - Bottom-left: normal tissue context (where is the non-tumor signal from?)
    - Bottom-right: MHC class I and II expression
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    analysis = analyze_sample(df_gene_expr, cancer_type=cancer_type)
    purity = analysis["purity"]
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Panel 1: Cancer type identification ----
    ax1 = fig.add_subplot(gs[0, 0])
    top_cancers = analysis["top_cancers"]
    codes = [c for c, s in top_cancers]
    scores = [s for c, s in top_cancers]
    colors = ["#2166ac" if c == cancer_code else "#92c5de" for c in codes]
    y = np.arange(len(codes))
    ax1.barh(y, scores, color=colors, edgecolor="none", height=0.6)
    for i, (code, score) in enumerate(top_cancers):
        from .plot import CANCER_TYPE_NAMES as CTN
        label = f"{code} ({CTN.get(code, '')})"
        ax1.text(score + 0.01, i, f"{score:.3f}", va="center", fontsize=9)
        ax1.text(-0.01, i, label, va="center", ha="right", fontsize=9)
    ax1.set_yticks([])
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Signature similarity score", fontsize=10)
    ax1.set_title("Cancer type identification", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # ---- Panel 2: Purity and microenvironment ----
    ax2 = fig.add_subplot(gs[0, 1])
    overall = purity["overall_estimate"]
    stromal_enr = purity["components"]["stromal"]["enrichment"]
    immune_enr = purity["components"]["immune"]["enrichment"]

    # Stacked composition bar
    tumor_frac = overall if overall else 0
    stromal_frac = min(stromal_enr / (stromal_enr + immune_enr + 0.001), 1 - tumor_frac) * (1 - tumor_frac)
    immune_frac = 1 - tumor_frac - stromal_frac

    ax2.barh(0, tumor_frac * 100, color="#2166ac", height=0.5, label=f"Tumor ({tumor_frac:.0%})")
    ax2.barh(0, stromal_frac * 100, left=tumor_frac * 100, color="#d6604d", height=0.5,
             label=f"Stromal ({stromal_frac:.0%})")
    ax2.barh(0, immune_frac * 100, left=(tumor_frac + stromal_frac) * 100, color="#4393c3",
             height=0.5, label=f"Immune ({immune_frac:.0%})")

    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel("Estimated composition (%)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Add text annotations below
    lo = purity["overall_lower"]
    hi = purity["overall_upper"]
    details = [
        f"Tumor purity: {overall:.0%}" + (f" [{lo:.0%}–{hi:.0%}]" if lo is not None else ""),
        f"Stromal enrichment: {stromal_enr:.1f}x vs TCGA {cancer_code}",
        f"Immune enrichment: {immune_enr:.1f}x vs TCGA {cancer_code}",
        f"TCGA {cancer_code} median purity: {purity['tcga_median_purity']:.0%}",
    ]
    for i, txt in enumerate(details):
        ax2.text(0, -0.6 - i * 0.5, txt, transform=ax2.transData,
                 fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8) if i == 0 else None)

    ax2.set_ylim(-3.5, 0.8)
    ax2.set_title("Sample composition", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # ---- Panel 3: Tissue context ----
    ax3 = fig.add_subplot(gs[1, 0])
    tissue_scores = analysis["tissue_scores"]
    if tissue_scores:
        tissues = [t.replace("_", " ").title() for t, s, n in tissue_scores]
        t_scores = [s for t, s, n in tissue_scores]
        matched = CANCER_TO_TISSUE.get(cancer_code, "").replace("_", " ").title()
        t_colors = []
        for t, s, n in tissue_scores:
            tname = t.replace("_", " ").title()
            if tname == matched:
                t_colors.append("#2166ac")  # tumor origin tissue
            elif s > 0.7:
                t_colors.append("#b2182b")  # strong non-tumor signal
            else:
                t_colors.append("#92c5de")  # background
        y = np.arange(len(tissues))
        ax3.barh(y, t_scores, color=t_colors, edgecolor="none", height=0.6)
        ax3.set_yticks(y)
        ax3.set_yticklabels(tissues, fontsize=9)
        for i, (t, s, n) in enumerate(tissue_scores):
            ax3.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=8)
        ax3.set_xlim(0, 1.1)
        ax3.set_xlabel("Tissue signature score", fontsize=10)
        ax3.invert_yaxis()
        # Legend
        from matplotlib.patches import Patch
        ax3.legend(handles=[
            Patch(color="#2166ac", label=f"Expected origin ({matched})"),
            Patch(color="#b2182b", label="Strong signal (>0.7)"),
            Patch(color="#92c5de", label="Background"),
        ], loc="lower right", fontsize=7, framealpha=0.9)
    ax3.set_title("Normal tissue context\n(where is the non-tumor signal from?)",
                  fontsize=12, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ---- Panel 4: MHC expression ----
    ax4 = fig.add_subplot(gs[1, 1])
    mhc1 = analysis["mhc1"]
    mhc2 = analysis["mhc2"]

    all_genes = list(mhc1.keys()) + list(mhc2.keys())
    all_tpms = [mhc1.get(g, 0) for g in mhc1] + [mhc2.get(g, 0) for g in mhc2]
    y = np.arange(len(all_genes))

    # Color by class
    n1 = len(mhc1)
    colors_mhc = ["#2166ac"] * n1 + ["#b2182b"] * len(mhc2)
    ax4.barh(y, all_tpms, color=colors_mhc, edgecolor="none", height=0.6, alpha=0.8)

    ax4.set_yticks(y)
    ax4.set_yticklabels(all_genes, fontsize=9)
    for i, tpm in enumerate(all_tpms):
        if tpm > 0:
            ax4.text(tpm + max(all_tpms) * 0.02, i, f"{tpm:.0f}", va="center", fontsize=8)

    # Divider between class I and II
    ax4.axhline(y=n1 - 0.5, color="#cccccc", linewidth=0.8, linestyle="--")
    ax4.text(max(all_tpms) * 0.95, n1 / 2 - 0.5, "Class I", ha="right", va="center",
             fontsize=9, color="#2166ac", fontweight="bold")
    ax4.text(max(all_tpms) * 0.95, n1 + len(mhc2) / 2 - 0.5, "Class II", ha="right", va="center",
             fontsize=9, color="#b2182b", fontweight="bold")

    ax4.set_xlabel("TPM", fontsize=10)
    ax4.invert_yaxis()
    ax4.set_title("MHC antigen presentation", fontsize=12, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ---- Main title ----
    fig.suptitle(
        f"Sample composition analysis — {cancer_name} ({cancer_code})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, analysis
