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

from collections import Counter

import numpy as np
import pandas as pd

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

_HOST_SITE_BACKGROUND_TISSUES = {
    "bone_marrow", "lymph_node", "spleen", "thymus", "tonsil", "appendix",
    "smooth_muscle", "skeletal_muscle", "heart_muscle", "adipose_tissue",
}

_CANCER_FAMILY_PANELS = {
    "PROSTATE": ["KLK3", "KLK2", "TMPRSS2", "FOLH1", "NKX3-1", "HOXB13", "STEAP1", "STEAP2", "AR"],
    "CRC": ["GUCY2C", "TFF3", "CDH17", "HEPH", "SLC12A2", "EPHB2", "CEACAM5", "CEACAM6", "VIL1", "CDX2"],
    "GASTRIC": ["MUC5AC", "MUC6", "CLDN18", "TFF1", "TFF2", "REG4", "GKN1", "GKN2"],
    "ESCA_SQ": ["TP63", "SOX2", "KRT5", "KRT14", "DSG3", "PPL", "KRT17", "FAM83H"],
    "SQUAMOUS": ["TP63", "SOX2", "KRT5", "KRT14", "DSG3", "PPL", "KRT17", "CLCA2", "KRT6A"],
    "MESENCHYMAL": ["COL1A1", "COL1A2", "COL3A1", "DCN", "THBS4", "POSTN", "TAGLN", "ACTA2", "MYLK", "DES"],
    "RENAL": ["PAX8", "PAX2", "CA9", "NDUFA4L2", "SLC22A12", "KCNJ1", "AMACR"],
    "GLIAL": ["GFAP", "OLIG2", "AQP4", "ALDH1L1", "SLC1A3", "SOX2"],
    "MELANOCYTIC": ["MLANA", "PMEL", "TYR", "DCT", "MITF"],
}

_CANCER_FAMILY_BY_CODE = {
    "PRAD": "PROSTATE",
    "COAD": "CRC",
    "READ": "CRC",
    "STAD": "GASTRIC",
    "ESCA": "ESCA_SQ",
    "HNSC": "SQUAMOUS",
    "LUSC": "SQUAMOUS",
    "CESC": "SQUAMOUS",
    "SARC": "MESENCHYMAL",
    "UCS": "MESENCHYMAL",
    "KIRC": "RENAL",
    "KIRP": "RENAL",
    "KICH": "RENAL",
    "GBM": "GLIAL",
    "LGG": "GLIAL",
    "SKCM": "MELANOCYTIC",
    "UVM": "MELANOCYTIC",
}

_CANCER_FAMILY_CODE_COUNTS = Counter(_CANCER_FAMILY_BY_CODE.values())
_CANCER_FAMILY_GROUP = {
    "ESCA_SQ": "SQUAMOUS",
}
_CANCER_FAMILY_GROUP_CODE_COUNTS = Counter(
    _CANCER_FAMILY_GROUP.get(family, family)
    for family in _CANCER_FAMILY_BY_CODE.values()
)
_CANCER_FAMILY_DISPLAY = {
    "CRC": "CRC",
    "ESCA_SQ": "esophageal squamous",
    "GASTRIC": "gastric",
    "GLIAL": "glial",
    "MELANOCYTIC": "melanocytic",
    "MESENCHYMAL": "mesenchymal / sarcoma-like",
    "PROSTATE": "prostate",
    "RENAL": "renal",
    "SQUAMOUS": "squamous",
}

TUMOR_PURITY_PARAMETERS = {
    "lineage": {
        "missing_support_factor": 0.35,
        "detection_fraction_threshold": 0.05,
    },
    "tumor_specific_markers": {
        "delta_min": 0.02,
        "normal_fraction_max": 0.5,
        "tme_fraction_max": 0.5,
        "cancer_expression_min": 0.5,
        "fallback_expression_min": 0.1,
        "zscore_min": 1.0,
        "fallback_zscore_min": 0.25,
        "specificity_min": 1.5,
    },
    "host_background": {
        "expression_min": 0.05,
        "zscore_min": 1.5,
        "specificity_min": 2.0,
        "top_genes": 20,
    },
    "purity_combination": {
        "signature_only_estimate_floor": 0.05,
        "tumor_anchor_weight": 0.7,
        "estimate_weight": 0.3,
        "signature_conflict_ratio": 0.75,
        "signature_stability_min": 0.45,
        "signature_weight_floor": 0.35,
    },
    "family_scoring": {
        "presence_scale": 0.15,
        "within_family_base": 0.35,
        "within_family_gain": 0.65,
        "non_family_penalty": 0.85,
        "min_factor": 0.05,
        "support_norm_floor": 0.05,
        "signature_stability_floor": 0.2,
        "family_display_fraction": 0.4,
        "candidate_panel_min_score": 0.05,
        "candidate_panel_top_n": 2,
        "non_penalizing_families": ["MESENCHYMAL"],
        "soft_family_penalty_gain": 0.75,
    },
}

_SIGNATURE_PANEL_CACHE = {}
_REARRANGED_GENE_PREFIXES = ("IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD")
_GENERIC_SIGNATURE_EXCLUDE_PREFIXES = ("MT-", "RPL", "RPS", "HLA-")


def get_tumor_purity_parameters():
    """Return the current tumor-purity and family-scoring free parameters."""
    return TUMOR_PURITY_PARAMETERS

_CANCER_NORMAL_TISSUES = {
    "COAD": ["colon", "rectum", "appendix", "small_intestine", "duodenum"],
    "READ": ["rectum", "colon", "appendix", "small_intestine", "duodenum"],
    "STAD": ["stomach", "duodenum", "esophagus", "gallbladder"],
    "ESCA": ["esophagus", "stomach"],
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


def _sample_hk_median(sample_tpm):
    """Return the sample housekeeping median on raw TPM scale."""
    ref = pan_cancer_expression()
    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    hk_syms = [id_to_sym[gid] for gid in housekeeping_gene_ids() if gid in id_to_sym]
    sample_hk_vals = [sample_tpm[g] for g in hk_syms if sample_tpm.get(g, 0) > 0]
    return float(np.median(sample_hk_vals)) if sample_hk_vals else 0.0


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


def _summarize_lineage_support(lineage_per_gene):
    """Summarize whether the observed lineage pattern matches the candidate tumor.

    A single shared marker can produce a misleadingly high lineage purity. We
    therefore score the *pattern* of lineage genes, not just their median
    purity, using a weighted cosine similarity between the observed lineage
    excess and the candidate's expected tumor lineage profile.
    """
    if not lineage_per_gene:
        return {
            "concordance": None,
            "detection_fraction": 0.0,
            "support_factor": TUMOR_PURITY_PARAMETERS["lineage"]["missing_support_factor"],
        }

    sample_excess = np.array(
        [max(0.0, row["sample_ratio"] - row["tme_ratio"]) for row in lineage_per_gene],
        dtype=float,
    )
    tumor_excess = np.array(
        [max(0.0, row["tumor_ratio"] - row["tme_ratio"]) for row in lineage_per_gene],
        dtype=float,
    )
    weights = np.sqrt(np.maximum(tumor_excess, 1e-6))

    sample_weighted = sample_excess * weights
    tumor_weighted = tumor_excess * weights
    denom = float(np.linalg.norm(sample_weighted) * np.linalg.norm(tumor_weighted))
    if denom > 0:
        concordance = float(np.clip(sample_weighted.dot(tumor_weighted) / denom, 0.0, 1.0))
    else:
        concordance = 0.0

    detected = sample_excess >= (
        TUMOR_PURITY_PARAMETERS["lineage"]["detection_fraction_threshold"]
        * np.maximum(tumor_excess, 1e-6)
    )
    detection_fraction = float(np.mean(detected))

    # Pattern match matters more than raw detection count. A candidate with a
    # few expressed genes but the wrong overall shape should be penalized hard.
    support_factor = float(np.sqrt(concordance) * (0.5 + 0.5 * detection_fraction))

    return {
        "concordance": concordance,
        "detection_fraction": detection_fraction,
        "support_factor": support_factor,
    }


def _select_tumor_specific_genes(cancer_code, n=30):
    """Select genes highly expressed in cancer but NOT in matched normal tissue.

    Returns list of gene symbols sorted by tumor specificity.
    """
    return _select_tumor_specific_genes_for_panel(
        cancer_code,
        n=n,
        exclude_lineage=True,
    )


def _is_excluded_signature_gene(symbol):
    """Exclude gene families that are brittle or driven by non-tumor admixture."""
    if not symbol:
        return True
    symbol = str(symbol)
    return symbol.startswith(_REARRANGED_GENE_PREFIXES) or symbol.startswith(
        _GENERIC_SIGNATURE_EXCLUDE_PREFIXES
    )


def _select_tumor_specific_genes_for_panel(cancer_code, n=30, exclude_lineage=True):
    """Select robust cancer-signature genes for purity and subtype panels.

    The panel builder is deliberately conservative:
    - drop rearranged immune receptor loci and generic housekeeping-like genes
    - require meaningful expression in the target cancer type
    - score genes by cancer-type specificity *and* visibility above TME / normal
    - relax thresholds only if the strict tier leaves a type under-covered
    """
    cache_key = (cancer_code, int(n), bool(exclude_lineage))
    cached = _SIGNATURE_PANEL_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    ref = pan_cancer_expression(normalize="housekeeping")
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}
    ]

    cancer_col = f"FPKM_{cancer_code}"
    if cancer_col not in ref_by_sym.columns:
        return []

    normal_tissues = list(_CANCER_NORMAL_TISSUES.get(cancer_code, []))
    tissue = CANCER_TO_TISSUE.get(cancer_code)
    if tissue and tissue not in normal_tissues:
        normal_tissues.append(tissue)
    normal_cols = [f"nTPM_{t}" for t in sorted(set(normal_tissues)) if f"nTPM_{t}" in ref_by_sym.columns]
    tme_cols = [
        f"nTPM_{t}"
        for t in sorted(_HOST_SITE_BACKGROUND_TISSUES)
        if f"nTPM_{t}" in ref_by_sym.columns
    ]
    lineage_genes = set(LINEAGE_GENES.get(cancer_code, [])) if exclude_lineage else set()

    # Z-score across cancer types for initial ranking
    expr_matrix = ref_by_sym[fpkm_cols].astype(float)
    gene_mean = expr_matrix.mean(axis=1)
    gene_std = expr_matrix.std(axis=1).replace(0, np.nan)
    z_scores = ((expr_matrix[cancer_col] - gene_mean) / gene_std).fillna(0)
    cancer_hk = expr_matrix[cancer_col]
    matched_normal_hk = (
        ref_by_sym[normal_cols].astype(float).max(axis=1)
        if normal_cols
        else pd.Series(0.0, index=ref_by_sym.index, dtype=float)
    )
    broad_normal_hk = (
        ref_by_sym[ntpm_nonrepro].astype(float).max(axis=1)
        if ntpm_nonrepro
        else pd.Series(0.0, index=ref_by_sym.index, dtype=float)
    )
    normal_hk = pd.concat(
        [matched_normal_hk.rename("matched"), broad_normal_hk.rename("broad")],
        axis=1,
    ).max(axis=1)
    tme_hk = (
        ref_by_sym[tme_cols].astype(float).max(axis=1)
        if tme_cols
        else pd.Series(0.0, index=ref_by_sym.index, dtype=float)
    )
    background_hk = pd.concat([normal_hk.rename("normal"), tme_hk.rename("tme")], axis=1).max(axis=1)

    score = (
        z_scores.clip(lower=0.0)
        * np.log2(cancer_hk + 1.0)
        * np.log2((cancer_hk + 0.01) / (background_hk + 0.01) + 1.0)
    )
    normal_frac = normal_hk / (cancer_hk + 0.001)
    tme_frac = tme_hk / (cancer_hk + 0.001)
    specificity = (cancer_hk + 0.001) / (background_hk + 0.001)
    excluded = ref_by_sym.index.to_series().map(_is_excluded_signature_gene)
    if lineage_genes:
        excluded = excluded | ref_by_sym.index.to_series().isin(lineage_genes)

    params = TUMOR_PURITY_PARAMETERS["tumor_specific_markers"]
    tiers = [
        {
            "expr_min": params["cancer_expression_min"],
            "zscore_min": params["zscore_min"],
            "normal_frac_max": params["normal_fraction_max"],
            "tme_frac_max": params["tme_fraction_max"],
            "specificity_min": params["specificity_min"],
        },
        {
            "expr_min": max(params["fallback_expression_min"], params["cancer_expression_min"] * 0.5),
            "zscore_min": max(0.0, params["fallback_zscore_min"]),
            "normal_frac_max": min(0.8, params["normal_fraction_max"] + 0.15),
            "tme_frac_max": min(0.8, params["tme_fraction_max"] + 0.2),
            "specificity_min": max(1.1, params["specificity_min"] - 0.3),
        },
        {
            "expr_min": params["fallback_expression_min"],
            "zscore_min": 0.0,
            "normal_frac_max": 1.0,
            "tme_frac_max": 1.0,
            "specificity_min": 1.0,
        },
    ]

    markers = []
    seen = set()
    for tier in tiers:
        keep = (
            (cancer_hk > tier["expr_min"])
            & (z_scores > tier["zscore_min"])
            & ((cancer_hk - normal_hk) > params["delta_min"])
            & (normal_frac <= tier["normal_frac_max"])
            & (tme_frac <= tier["tme_frac_max"])
            & (specificity >= tier["specificity_min"])
            & ~excluded
        )
        candidates = score[keep].sort_values(ascending=False)
        for gene in candidates.index:
            if gene in seen:
                continue
            seen.add(gene)
            markers.append(gene)
            if len(markers) >= n:
                _SIGNATURE_PANEL_CACHE[cache_key] = tuple(markers[:n])
                return markers[:n]

    fallback = score[(cancer_hk > params["fallback_expression_min"]) & ~excluded].sort_values(ascending=False)
    for gene in fallback.index:
        if gene in seen:
            continue
        seen.add(gene)
        markers.append(gene)
        if len(markers) >= n:
            break

    _SIGNATURE_PANEL_CACHE[cache_key] = tuple(markers[:n])
    return markers


def _summarize_gene_level_purity(per_gene_purities, strategy="winsorized_median"):
    """Summarize per-gene purity estimates robustly.

    Signature genes can contain amplified or noisy outliers, while lineage
    genes can contain low outliers from de-differentiation. The summary should
    reflect the stable center of the distribution rather than a few extremes.
    """
    vals = np.array(sorted(float(p) for p in per_gene_purities if p is not None and p > 0), dtype=float)
    if len(vals) == 0:
        return None, None, None, None

    lower = float(np.percentile(vals, 25))
    upper = float(np.percentile(vals, 75))
    if strategy == "upper_half":
        core = vals[len(vals) // 2:] if len(vals) >= 3 else vals
    elif strategy == "winsorized_median" and len(vals) >= 4:
        core = np.clip(vals, lower, upper)
    else:
        core = vals

    overall = float(np.median(core))
    stability = float(np.clip((lower + 0.02) / (upper + 0.02), 0.0, 1.0))
    return overall, lower, upper, stability


def _combine_purity_estimates(
    sig_purity,
    sig_lower,
    sig_upper,
    estimate_purity,
    lineage_purity,
    lineage_lower,
    lineage_upper,
    sig_stability=None,
):
    """Combine purity signals while keeping ESTIMATE as context, not destiny.

    The ESTIMATE-derived purity is useful as an infiltration warning, but in
    highly inflamed metastases it often collapses to ~0 and should not erase a
    coherent tumor/lineage signal. When lineage support exists, combine it with
    the tumor-specific signature directly; otherwise fall back to the available
    evidence.
    """
    has_sig = sig_purity is not None
    has_lineage = lineage_purity is not None
    signature_params = TUMOR_PURITY_PARAMETERS["purity_combination"]
    deprioritize_signature = _signature_conflicts_with_lineage(
        sig_purity=sig_purity,
        lineage_purity=lineage_purity,
        sig_stability=sig_stability,
    )

    if has_sig and has_lineage:
        if deprioritize_signature:
            tumor_anchor = float(lineage_purity)
        else:
            sig_weight = float(max(sig_stability or 1.0, signature_params["signature_weight_floor"]))
            lineage_weight = 1.0
            tumor_anchor = float(
                np.exp(
                    (
                        sig_weight * np.log(max(sig_purity, 1e-6))
                        + lineage_weight * np.log(max(lineage_purity, 1e-6))
                    )
                    / (sig_weight + lineage_weight)
                )
            )
    elif has_lineage:
        tumor_anchor = float(lineage_purity)
    elif has_sig:
        tumor_anchor = float(sig_purity)
    else:
        tumor_anchor = None

    if tumor_anchor is not None and estimate_purity is not None:
        # Tumor-positive and infiltration-negative signals should both matter,
        # but the tumor-specific anchor gets slightly more weight because
        # ESTIMATE can undercall purity in inflamed metastases.
        if has_sig and has_lineage and estimate_purity <= 0:
            overall = float(tumor_anchor)
        elif has_sig and not has_lineage:
            estimate_floor = signature_params["signature_only_estimate_floor"]
            overall = float(np.sqrt(max(tumor_anchor, 0.0) * max(estimate_purity, estimate_floor)))
        else:
            estimate_floor = signature_params["signature_only_estimate_floor"]
            overall = float(
                (max(tumor_anchor, 0.0) ** signature_params["tumor_anchor_weight"])
                * (max(estimate_purity, estimate_floor) ** signature_params["estimate_weight"])
            )
    elif tumor_anchor is not None:
        overall = float(tumor_anchor)
    elif estimate_purity is not None:
        overall = float(estimate_purity)
    else:
        return None, None, None

    lower_candidates = [overall]
    upper_candidates = [overall]

    for value in (lineage_lower,):
        if value is not None:
            lower_candidates.append(float(value))
    if not deprioritize_signature:
        for value in (sig_lower,):
            if value is not None:
                lower_candidates.append(float(value))
    for value in (lineage_upper,):
        if value is not None:
            upper_candidates.append(float(value))
    if not deprioritize_signature:
        for value in (sig_upper,):
            if value is not None:
                upper_candidates.append(float(value))

    if estimate_purity is not None and (estimate_purity > 0 or (has_sig and not has_lineage)):
        lower_candidates.append(float(estimate_purity))

    overall_lower = float(np.clip(min(lower_candidates), 0.0, 1.0))
    overall_upper = float(np.clip(max(upper_candidates), 0.0, 1.0))
    overall = float(np.clip(overall, overall_lower, overall_upper))
    return overall, overall_lower, overall_upper


def _signature_conflicts_with_lineage(sig_purity, lineage_purity, sig_stability):
    """Return True when a weak signature should not drag down a coherent lineage call."""
    if sig_purity is None or lineage_purity is None:
        return False
    params = TUMOR_PURITY_PARAMETERS["purity_combination"]
    stability = float(sig_stability if sig_stability is not None else 1.0)
    return (
        float(sig_purity) < float(lineage_purity) * params["signature_conflict_ratio"]
        and stability < params["signature_stability_min"]
    )


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
    sig_purity_robust, sig_lower, sig_upper, sig_stability = _summarize_gene_level_purity(
        per_gene_purities,
        strategy="winsorized_median",
    )
    if sig_purity_robust is not None:
        sig_purity = sig_purity_robust

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

    # Convert stromal/immune enrichment into a tumor-vs-background odds model
    # rather than a linear fraction model, which otherwise collapses to 0 on
    # inflamed samples.
    tcga_nontumor_odds = (1 - tcga_purity) / max(tcga_purity, 1e-6)
    stromal_purity = 1.0 / (1.0 + tcga_nontumor_odds * max(stromal_enrichment, 0.0))
    immune_purity = 1.0 / (1.0 + tcga_nontumor_odds * max(immune_enrichment, 0.0))
    estimate_purity = float(np.clip(np.sqrt(stromal_purity * immune_purity), 0.0, 1.0))

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
        lineage_purity, lineage_lower, lineage_upper, lineage_stability = _summarize_gene_level_purity(
            upper_half,
            strategy="upper_half",
        )
    else:
        lineage_purity = lineage_lower = lineage_upper = lineage_stability = None
    lineage_support = _summarize_lineage_support(lineage_per_gene)

    # ---- Combine estimates ----
    overall, overall_lower, overall_upper = _combine_purity_estimates(
        sig_purity=sig_purity,
        sig_lower=sig_lower,
        sig_upper=sig_upper,
        estimate_purity=estimate_purity if stromal_genes else None,
        lineage_purity=lineage_purity,
        lineage_lower=lineage_lower,
        lineage_upper=lineage_upper,
        sig_stability=sig_stability,
    )
    signature_deprioritized = _signature_conflicts_with_lineage(
        sig_purity=sig_purity,
        lineage_purity=lineage_purity,
        sig_stability=sig_stability,
    )
    if signature_deprioritized:
        integration_source = "lineage"
    elif sig_purity is not None and lineage_purity is not None:
        integration_source = "signature+lineage"
    elif lineage_purity is not None:
        integration_source = "lineage"
    elif sig_purity is not None:
        integration_source = "signature"
    elif estimate_purity is not None:
        integration_source = "estimate"
    else:
        integration_source = None

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
                "stability": sig_stability,
                "per_gene": per_gene,
            },
            "lineage": {
                "genes": [g["gene"] for g in lineage_per_gene],
                "purity": lineage_purity,
                "lower": lineage_lower,
                "upper": lineage_upper,
                "stability": lineage_stability,
                "concordance": lineage_support["concordance"],
                "detection_fraction": lineage_support["detection_fraction"],
                "support_factor": lineage_support["support_factor"],
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
            "integration": {
                "source": integration_source,
                "signature_deprioritized": signature_deprioritized,
            },
        },
    }


# -------------------- plotting --------------------


def plot_tumor_purity(
    df_gene_expr,
    cancer_type=None,
    sample_mode="auto",
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
    if sample_mode == "auto":
        try:
            from .decomposition import infer_sample_mode
            sample_mode = infer_sample_mode(cancer_types=[cancer_code], sample_mode="auto")
        except Exception:
            sample_mode = "solid"

    if sample_mode == "heme":
        metric_label = "Fraction estimate"
        component_title = "Fraction / context components"
        summary_title = "Malignant-lineage fraction estimate"
        signature_label = "Malignant signature"
        overall_label = "Overall fraction proxy"
        left_title = (
            f"{cancer_code} lineage-signature fraction estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, calibrated for "
            f"TCGA median purity {result['tcga_median_purity']:.0%})"
        )
    elif sample_mode == "pure":
        metric_label = "Consistency estimate"
        component_title = "Consistency / context components"
        summary_title = "Population consistency estimate"
        signature_label = "Population signature"
        overall_label = "Overall consistency"
        left_title = (
            f"{cancer_code} lineage-profile consistency estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, not interpreted as bulk admixture)"
        )
    else:
        metric_label = "Purity estimate"
        component_title = "Purity components"
        summary_title = "Tumor purity estimate"
        signature_label = "Tumor signature"
        overall_label = "Overall estimate"
        left_title = (
            f"{cancer_code} signature gene purity estimates\n"
            f"(gene TPM / HK TPM vs TCGA reference, calibrated for "
            f"TCGA median purity {result['tcga_median_purity']:.0%})"
        )

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
        ax1.set_xlabel(f"{metric_label} (%)", fontsize=10)
        ax1.set_title(left_title, fontsize=10)
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
            f"{signature_label}\n({len(comp['signature']['genes'])} genes)",
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
            overall_label,
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
    ax2.set_xlabel(f"{metric_label.split()[0]} (%) / enrichment", fontsize=10)
    ax2.set_title(component_title, fontsize=11)
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"{summary_title}: {result['overall_estimate']:.0%} "
        f"[{result['overall_lower']:.0%}–{result['overall_upper']:.0%}]"
        if result["overall_estimate"] is not None
        else f"{summary_title}: N/A",
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


def _score_host_tissues(sample_tpm_by_symbol, tissues=None, top_n=None):
    """Score host tissues using site-specific genes depleted from TME overlap.

    This is stricter than `_score_normal_tissues()`: genes must be specific for
    the candidate host tissue relative to other tissues and relative to generic
    immune/stromal backgrounds. This prevents lymph node from winning simply
    because a sample is immune-rich.
    """
    ref = pan_cancer_expression()
    ref_by_sym = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    expr = ref_by_sym[ntpm_cols].astype(float)
    hk_median = _sample_hk_median(sample_tpm_by_symbol)
    if hk_median <= 0:
        hk_median = 1.0

    id_to_sym = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))
    hk_gene_symbols = [id_to_sym[gid] for gid in housekeeping_gene_ids() if gid in id_to_sym and id_to_sym[gid] in ref_by_sym.index]
    if hk_gene_symbols:
        ref_hk_medians = expr.loc[hk_gene_symbols].median(axis=0).replace(0, np.nan)
    else:
        ref_hk_medians = pd.Series(1.0, index=expr.columns, dtype=float)
    expr_hk = expr.div(ref_hk_medians, axis=1).fillna(0.0)

    gene_mean = expr_hk.mean(axis=1)
    gene_std = expr_hk.std(axis=1).replace(0, np.nan)
    z_matrix = expr_hk.sub(gene_mean, axis=0).div(gene_std, axis=0).fillna(0)

    results = []
    for col in ntpm_cols:
        tissue = col.replace("nTPM_", "")
        if tissues is not None and tissue not in tissues:
            continue

        background_cols = [
            other for other in ntpm_cols
            if other != col and other.replace("nTPM_", "") in _HOST_SITE_BACKGROUND_TISSUES
        ]
        if background_cols:
            background_max = expr_hk[background_cols].max(axis=1)
        else:
            background_max = pd.Series(0.0, index=expr.index)

        tissue_expr = expr_hk[col]
        z_col = z_matrix[col]
        specificity = (tissue_expr + 1e-6) / (background_max + 1e-6)
        score = z_col * np.log2(specificity + 1.0)
        keep = (
            (tissue_expr > TUMOR_PURITY_PARAMETERS["host_background"]["expression_min"])
            & (z_col > TUMOR_PURITY_PARAMETERS["host_background"]["zscore_min"])
            & (specificity > TUMOR_PURITY_PARAMETERS["host_background"]["specificity_min"])
        )
        sig_genes = list(
            score[keep]
            .sort_values(ascending=False)
            .head(TUMOR_PURITY_PARAMETERS["host_background"]["top_genes"])
            .index
        )
        if len(sig_genes) < 5:
            continue

        pcts = []
        for gene in sig_genes:
            s_val = sample_tpm_by_symbol.get(gene, 0.0) / hk_median
            ref_vals = expr_hk.loc[gene].values
            n = len(ref_vals)
            below = np.sum(ref_vals < s_val)
            equal = np.sum(np.isclose(ref_vals, s_val, atol=1e-6))
            pcts.append((below + 0.5 * equal) / n)
        if pcts:
            results.append((tissue, float(np.mean(pcts)), len(pcts)))

    results.sort(key=lambda x: -x[1])
    if top_n is None:
        return results
    return results[:top_n]


def _score_cancer_family_panels(sample_tpm_by_symbol):
    """Score broad cancer families before attempting fine subtype ranking."""
    hk_median = _sample_hk_median(sample_tpm_by_symbol)
    if hk_median <= 0:
        return {family: 0.0 for family in _CANCER_FAMILY_PANELS}

    scores = {}
    for family, genes in _CANCER_FAMILY_PANELS.items():
        values = [sample_tpm_by_symbol.get(g, 0.0) / hk_median for g in genes]
        if not values:
            scores[family] = 0.0
            continue
        values = sorted(values)
        upper_half = values[len(values) // 2:] if len(values) >= 3 else values
        scores[family] = float(np.median(upper_half)) if upper_half else 0.0
    return scores


def _get_mhc_expression(sample_tpm_by_symbol):
    """Get MHC class I and II expression levels."""
    mhc1_genes = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"]
    mhc2_genes = ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1"]

    mhc1 = {g: sample_tpm_by_symbol.get(g, 0) for g in mhc1_genes}
    mhc2 = {g: sample_tpm_by_symbol.get(g, 0) for g in mhc2_genes}
    return mhc1, mhc2


def rank_cancer_type_candidates(
    df_gene_expr,
    candidate_codes=None,
    top_k=5,
):
    """Rank cancer-type hypotheses by signature evidence and purity plausibility.

    Pure signature similarity tends to overcall stromal or immune-rich cancer
    types when the sample has heavy admixture. We rank candidates by:

    - cancer-type signature similarity
    - a purity anchor that combines tumor-specific and lineage evidence
    - lineage pattern concordance when lineage genes are available

    This keeps "one of these two is plausible" ambiguity visible while
    downweighting types whose purity model does not fit the sample.
    """
    from .plot import _compute_cancer_type_signature_stats, resolve_cancer_type

    stats = _compute_cancer_type_signature_stats(df_gene_expr)
    signature_score_map = {row["code"]: float(row["score"]) for row in stats}
    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)
    family_scores = _score_cancer_family_panels(sample_tpm)
    family_params = TUMOR_PURITY_PARAMETERS["family_scoring"]
    soft_families = set(family_params.get("non_penalizing_families", []))
    hard_family_scores = {
        family: score for family, score in family_scores.items()
        if family not in soft_families
    }
    max_family_score = max(hard_family_scores.values(), default=0.0)
    sorted_family_scores = sorted(hard_family_scores.values(), reverse=True)
    top_family_score = sorted_family_scores[0] if sorted_family_scores else 0.0
    second_family_score = sorted_family_scores[1] if len(sorted_family_scores) > 1 else 0.0
    family_presence = float(np.clip(top_family_score / family_params["presence_scale"], 0.0, 1.0))
    family_specificity = 0.0
    if top_family_score > 0:
        family_specificity = float(
            np.clip((top_family_score - second_family_score) / top_family_score, 0.0, 1.0)
        )
    ranked_families = sorted(
        family_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    hard_ranked_families = sorted(
        hard_family_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    if candidate_codes is None:
        candidate_codes = [row["code"] for row in stats[:8]]
        for family, score in ranked_families[: family_params["candidate_panel_top_n"]]:
            if score < family_params["candidate_panel_min_score"]:
                continue
            if family in soft_families and top_family_score >= family_params["presence_scale"]:
                continue
            family_codes = [
                code for code, family_label in _CANCER_FAMILY_BY_CODE.items()
                if family_label == family
            ]
            candidate_codes.extend(family_codes)
    else:
        candidate_codes = [resolve_cancer_type(code) for code in candidate_codes]

    seen = set()
    ordered_codes = []
    for code in candidate_codes:
        if code not in seen:
            seen.add(code)
            ordered_codes.append(code)

    rows = []
    top_family_label = hard_ranked_families[0][0] if hard_ranked_families else None
    non_penalizing_families = soft_families
    for code in ordered_codes:
        purity_result = estimate_tumor_purity(df_gene_expr, cancer_type=code)
        purity_estimate = float(purity_result["overall_estimate"] or 0.0)
        signature_score = float(signature_score_map.get(code, 0.0))
        signature_stability = float(
            purity_result.get("components", {}).get("signature", {}).get("stability") or 1.0
        )
        lineage = purity_result.get("components", {}).get("lineage", {})
        lineage_support_factor = float(lineage.get("support_factor") or 1.0)
        lineage_concordance = lineage.get("concordance")
        lineage_detection_fraction = lineage.get("detection_fraction")
        family_label = _CANCER_FAMILY_BY_CODE.get(code)
        if family_label is not None:
            if family_label in non_penalizing_families:
                family_factor = float(
                    np.clip(
                        1.0 - family_params["soft_family_penalty_gain"] * family_presence,
                        family_params["min_factor"],
                        1.0,
                    )
                )
            else:
                family_relative = (
                    float(family_scores.get(family_label, 0.0) / max_family_score)
                    if max_family_score > 0
                    else 0.0
                )
                family_factor = float(
                    np.clip(
                        family_params["within_family_base"] + family_params["within_family_gain"] * family_presence * family_relative,
                        family_params["min_factor"],
                        1.0,
                    )
                )
        elif family_label is None and family_presence > 0:
            if top_family_label in non_penalizing_families:
                family_factor = 1.0
            else:
                family_factor = float(
                    np.clip(
                        1.0 - family_params["non_family_penalty"] * family_presence,
                        family_params["min_factor"],
                        1.0,
                    )
                )
        else:
            family_factor = 1.0
        support_score = (
            signature_score
            * max(purity_estimate, family_params["support_norm_floor"])
            * lineage_support_factor
            * max(signature_stability, family_params["signature_stability_floor"])
            * max(family_factor, family_params["min_factor"])
        )
        rows.append(
            {
                "code": code,
                "signature_score": signature_score,
                "signature_stability": signature_stability,
                "purity_estimate": purity_estimate,
                "lineage_purity": lineage.get("purity"),
                "lineage_concordance": lineage_concordance,
                "lineage_detection_fraction": lineage_detection_fraction,
                "lineage_support_factor": lineage_support_factor,
                "family_label": family_label,
                "family_score": family_scores.get(family_label) if family_label is not None else None,
                "family_presence": family_presence,
                "family_specificity": family_specificity,
                "family_factor": family_factor,
                "support_score": support_score,
                "purity_result": purity_result,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["support_score"],
            -row["signature_score"],
            row["code"],
        )
    )
    if rows:
        best_family_group = _CANCER_FAMILY_GROUP.get(
            rows[0].get("family_label"),
            rows[0].get("family_label"),
        )
        if best_family_group and _CANCER_FAMILY_GROUP_CODE_COUNTS.get(best_family_group, 0) > 1:
            same_family = [
                row for row in rows[1:]
                if _CANCER_FAMILY_GROUP.get(row.get("family_label"), row.get("family_label")) == best_family_group
            ]
            if same_family:
                other_rows = [
                    row for row in rows[1:]
                    if _CANCER_FAMILY_GROUP.get(row.get("family_label"), row.get("family_label")) != best_family_group
                ]
                rows = [rows[0]] + same_family + other_rows

    max_support = max((row["support_score"] for row in rows), default=0.0)
    for row in rows:
        row["support_norm"] = (
            float(row["support_score"] / max_support) if max_support > 0 else 0.0
        )

    return rows[:top_k]


def _summarize_candidate_family(candidate_trace):
    """Summarize family-level ambiguity from ranked cancer candidates."""
    if not candidate_trace:
        return {
            "label": None,
            "codes": [],
            "display": None,
            "subtype_clause": None,
        }

    best = candidate_trace[0]
    family = best.get("family_label")
    if family is None:
        return {
            "label": None,
            "codes": [best["code"]],
            "display": None,
            "subtype_clause": None,
        }
    family_group = _CANCER_FAMILY_GROUP.get(family, family)

    family_rows = [
        row for row in candidate_trace
        if _CANCER_FAMILY_GROUP.get(row.get("family_label"), row.get("family_label")) == family_group
        and row["support_score"] >= best["support_score"] * TUMOR_PURITY_PARAMETERS["family_scoring"]["family_display_fraction"]
    ]
    family_codes = [row["code"] for row in family_rows]
    if _CANCER_FAMILY_GROUP_CODE_COUNTS.get(family_group, 0) < 2:
        return {
            "label": family_group,
            "codes": family_codes or [best["code"]],
            "display": None,
            "subtype_clause": None,
        }
    display_name = _CANCER_FAMILY_DISPLAY.get(family_group, family_group)
    display = f"{display_name} family"
    subtype_clause = None
    if len(family_codes) >= 2:
        display = f"{display_name} family ({' > '.join(family_codes[:3])})"
        subtype_clause = f"{family_codes[0]} > {family_codes[1]}"
    elif family_codes:
        subtype_clause = family_codes[0]

    return {
        "label": family_group,
        "codes": family_codes,
        "display": display,
        "subtype_clause": subtype_clause,
    }


def _summarize_fit_quality(candidate_trace, signature_stats):
    """Describe whether TCGA references provide a focused subtype fit."""
    if not candidate_trace:
        return {
            "label": "unknown",
            "signature_gap": None,
            "support_ratio": None,
            "message": "No cancer candidates were available.",
        }

    best = candidate_trace[0]
    second = candidate_trace[1] if len(candidate_trace) > 1 else None
    support_ratio = None
    if second is not None and second["support_score"] > 0:
        support_ratio = float(best["support_score"] / second["support_score"])

    top_signature = float(signature_stats[0]["score"]) if signature_stats else 0.0
    reference_idx = min(4, len(signature_stats) - 1) if signature_stats else 0
    reference_signature = float(signature_stats[reference_idx]["score"]) if signature_stats else 0.0
    signature_gap = float(top_signature - reference_signature)

    if signature_gap < 0.05:
        return {
            "label": "weak",
            "signature_gap": signature_gap,
            "support_ratio": support_ratio,
            "message": (
                "Subtype fit is weak: the sample sits in a flat TCGA signature landscape, "
                "so broad family interpretation is more trustworthy than the exact top label."
            ),
        }
    if support_ratio is not None and support_ratio < 1.35:
        return {
            "label": "ambiguous",
            "signature_gap": signature_gap,
            "support_ratio": support_ratio,
            "message": "Top subtype candidates remain close; treat the leading label as provisional.",
        }
    return {
        "label": "focused",
        "signature_gap": signature_gap,
        "support_ratio": support_ratio,
        "message": "The leading TCGA reference is materially separated from alternatives.",
    }


# -------------------- comprehensive summary --------------------


def analyze_sample(df_gene_expr, cancer_type=None):
    """Comprehensive sample composition analysis.

    Returns a dict with all analysis results: cancer type, purity,
    background signatures, MHC status, and narrative interpretation.
    """
    from .plot import (
        _compute_cancer_type_signature_stats,
        resolve_cancer_type,
        CANCER_TYPE_NAMES,
    )

    sample_tpm = _build_sample_tpm_by_symbol(df_gene_expr)

    # 1. Cancer type
    stats = _compute_cancer_type_signature_stats(df_gene_expr)
    default_candidates = [row["code"] for row in stats[:8]]
    if cancer_type:
        cancer_code = resolve_cancer_type(cancer_type)
        candidate_trace = rank_cancer_type_candidates(
            df_gene_expr,
            candidate_codes=[cancer_code] + default_candidates,
            top_k=8,
        )
    else:
        candidate_trace = rank_cancer_type_candidates(
            df_gene_expr,
            candidate_codes=None,
            top_k=8,
        )
        cancer_code = candidate_trace[0]["code"] if candidate_trace else stats[0]["code"]
    cancer_name = CANCER_TYPE_NAMES.get(cancer_code, cancer_code)
    candidate_lookup = {row["code"]: row for row in candidate_trace}
    selected_candidate = candidate_lookup.get(cancer_code)
    cancer_score = selected_candidate["support_score"] if selected_candidate else None
    family_summary = _summarize_candidate_family(candidate_trace)
    fit_quality = _summarize_fit_quality(candidate_trace, stats)

    # 2. Purity
    if selected_candidate is not None:
        purity = selected_candidate["purity_result"]
    else:
        purity = estimate_tumor_purity(df_gene_expr, cancer_type=cancer_code)

    # 3. Residual background signatures
    tissue_scores = _score_host_tissues(sample_tpm, top_n=10)

    # 4. MHC expression
    mhc1, mhc2 = _get_mhc_expression(sample_tpm)

    # 5. Top cancer type matches
    top_cancers = [(row["code"], row["support_score"]) for row in candidate_trace[:5]]
    signature_top_cancers = [(s["code"], s["score"]) for s in stats[:5]]

    return {
        "cancer_type": cancer_code,
        "cancer_name": cancer_name,
        "cancer_score": cancer_score,
        "top_cancers": top_cancers,
        "signature_top_cancers": signature_top_cancers,
        "candidate_trace": candidate_trace,
        "family_summary": family_summary,
        "fit_quality": fit_quality,
        "purity": purity,
        "tissue_scores": tissue_scores,
        "mhc1": mhc1,
        "mhc2": mhc2,
    }


def plot_sample_summary(
    df_gene_expr,
    cancer_type=None,
    sample_mode="auto",
    save_to_filename=None,
    save_dpi=300,
):
    """Comprehensive sample composition plot.

    Four-panel figure:
    - Top-left: cancer type identification (bar chart)
    - Top-right: tumor purity and microenvironment composition
    - Bottom-left: residual background signatures (where is the non-tumor signal from?)
    - Bottom-right: MHC class I and II expression
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    analysis = analyze_sample(df_gene_expr, cancer_type=cancer_type)
    purity = analysis["purity"]
    cancer_code = analysis["cancer_type"]
    cancer_name = analysis["cancer_name"]
    if sample_mode == "auto":
        try:
            from .decomposition import infer_sample_mode
            sample_mode = infer_sample_mode(
                candidate_rows=analysis.get("candidate_trace"),
                cancer_types=[cancer_code],
                sample_mode="auto",
            )
        except Exception:
            sample_mode = "solid"

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
    ax1.set_xlabel("Support score (signature x purity)", fontsize=10)
    fit_quality = analysis.get("fit_quality", {})
    fit_label = fit_quality.get("label")
    title = "Cancer type hypotheses"
    if fit_label in {"weak", "ambiguous"}:
        title += f" ({fit_label} fit)"
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    label_options = [top_cancers[0][0]] if top_cancers else []
    if len(top_cancers) >= 2 and fit_label in {"weak", "ambiguous"}:
        label_options.append(top_cancers[1][0])
    if fit_label or label_options:
        lines = []
        if fit_label:
            lines.append(f"Fit: {fit_label}")
        if len(label_options) == 2:
            lines.append(f"Possible labels: {label_options[0]} or {label_options[1]}")
        elif len(label_options) == 1:
            lines.append(f"Lead label: {label_options[0]}")
        ax1.text(
            0.02,
            0.02,
            "\n".join(lines),
            transform=ax1.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )

    # ---- Panel 2: Purity and microenvironment ----
    ax2 = fig.add_subplot(gs[0, 1])
    overall = purity["overall_estimate"]
    stromal_enr = purity["components"]["stromal"]["enrichment"]
    immune_enr = purity["components"]["immune"]["enrichment"]

    # Stacked composition bar
    tumor_frac = overall if overall else 0
    stromal_frac = min(stromal_enr / (stromal_enr + immune_enr + 0.001), 1 - tumor_frac) * (1 - tumor_frac)
    immune_frac = 1 - tumor_frac - stromal_frac

    if sample_mode == "heme":
        main_label = "Malignant-like"
        stromal_label = "Stromal context"
        immune_label = "Immune context"
        comp_title = "Heme Composition Context"
        comp_xlabel = "Estimated fraction / context (%)"
        detail_prefix = "Malignant-lineage fraction proxy"
    elif sample_mode == "pure":
        main_label = "Dominant population"
        stromal_label = "Residual stromal"
        immune_label = "Residual immune"
        comp_title = "Population Coherence"
        comp_xlabel = "Estimated population / context (%)"
        detail_prefix = "Population consistency"
    else:
        main_label = "Tumor"
        stromal_label = "Stromal"
        immune_label = "Immune"
        comp_title = "Sample Composition"
        comp_xlabel = "Estimated composition (%)"
        detail_prefix = "Tumor purity"

    ax2.barh(0, tumor_frac * 100, color="#2166ac", height=0.5, label=f"{main_label} ({tumor_frac:.0%})")
    ax2.barh(0, stromal_frac * 100, left=tumor_frac * 100, color="#d6604d", height=0.5,
             label=f"{stromal_label} ({stromal_frac:.0%})")
    ax2.barh(0, immune_frac * 100, left=(tumor_frac + stromal_frac) * 100, color="#4393c3",
             height=0.5, label=f"{immune_label} ({immune_frac:.0%})")

    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel(comp_xlabel, fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Add text annotations below
    lo = purity["overall_lower"]
    hi = purity["overall_upper"]
    details = [f"{detail_prefix}: {overall:.0%}" + (f" [{lo:.0%}–{hi:.0%}]" if lo is not None else "")]
    if sample_mode == "solid":
        details.extend(
            [
                f"Stromal enrichment: {stromal_enr:.1f}x vs TCGA {cancer_code}",
                f"Immune enrichment: {immune_enr:.1f}x vs TCGA {cancer_code}",
                f"TCGA {cancer_code} median purity: {purity['tcga_median_purity']:.0%}",
            ]
        )
    elif sample_mode == "heme":
        details.extend(
            [
                f"Stromal context: {stromal_enr:.1f}x vs TCGA {cancer_code}",
                f"Immune context: {immune_enr:.1f}x vs TCGA {cancer_code}",
                "Interpretation: lineage/background context, not a strict tumor-vs-immune split",
            ]
        )
    else:
        details.extend(
            [
                f"Residual stromal context: {stromal_enr:.1f}x vs TCGA {cancer_code}",
                f"Residual immune context: {immune_enr:.1f}x vs TCGA {cancer_code}",
                "Interpretation: consistency vs matched lineage profile, not bulk admixture",
            ]
        )
    for i, txt in enumerate(details):
        ax2.text(0, -0.6 - i * 0.5, txt, transform=ax2.transData,
                 fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8) if i == 0 else None)

    ax2.set_ylim(-3.5, 0.8)
    ax2.set_title(comp_title, fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # ---- Panel 3: Background tissue signatures ----
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
        ax3.set_xlabel("Background signature score", fontsize=10)
        ax3.invert_yaxis()
        # Legend
        from matplotlib.patches import Patch
        ax3.legend(handles=[
            Patch(color="#2166ac", label=f"Expected origin family ({matched})"),
            Patch(color="#b2182b", label="Strong residual background"),
            Patch(color="#92c5de", label="Background"),
        ], loc="lower right", fontsize=7, framealpha=0.9)
    if sample_mode == "heme":
        bg_title = "Lineage / Background Context\n(residual hematopoietic and tissue programs)"
    elif sample_mode == "pure":
        bg_title = "Residual Background Check\n(contamination / off-target context)"
    else:
        bg_title = "Background Tissue Signatures\n(residual non-tumor context)"
    ax3.set_title(bg_title, fontsize=12, fontweight="bold")
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
    if sample_mode == "heme":
        mode_title = "hematologic / lymphoid bulk"
    elif sample_mode == "pure":
        mode_title = "pure population / cell culture"
    else:
        mode_title = "solid tumor / metastatic bulk"
    fig.suptitle(
        f"Sample composition analysis — {cancer_name} ({cancer_code}) [{mode_title}]",
        fontsize=15, fontweight="bold", y=0.98,
    )

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig, analysis
