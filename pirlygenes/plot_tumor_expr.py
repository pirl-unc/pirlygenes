# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

from .common import _guess_gene_cols
from .plot_data_helpers import _strip_ensembl_version
from .gene_sets_cancer import (
    pan_cancer_expression,
    housekeeping_gene_ids,
    is_extended_housekeeping_symbol,
    CTA_gene_id_to_name,
    therapy_target_gene_id_to_name,
    cancer_surfaceome_gene_id_to_name,
)
from .plot_scatter import resolve_cancer_type
from .plot_therapy import (
    _summarize_fn1_edb_transcript_support,
    _apply_therapy_support_gate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPRODUCTIVE_TISSUES = {"testis", "epididymis", "seminal_vesicle", "placenta", "ovary"}

_STROMAL_TISSUES = {"smooth_muscle", "skeletal_muscle", "heart_muscle", "adipose_tissue"}

# Immune/lymphoid tissues that represent TME infiltrate.  Curated to
# exclude epithelial organs that merely contain resident immune cells
# (which would inflate the TME background estimate).
_IMMUNE_TISSUES = {
    "bone_marrow", "lymph_node", "spleen", "thymus", "tonsil", "appendix",
}

_TME_TISSUES = _STROMAL_TISSUES | _IMMUNE_TISSUES

# Met-site tissue augmentation (#13). When a biopsy was taken from a
# specific metastatic site, its host-tissue contribution should be
# represented in the TME background so tumor-expression estimates are
# not inflated by unsubtracted host-tissue signal. Each entry lists
# tissues to **add** to the TME reference; passing an empty set (for
# `primary`) leaves the default set untouched.
MET_SITE_TISSUE_AUGMENTATION = {
    "primary":    set(),
    "lymph_node": {"lymph_node", "spleen", "thymus", "tonsil"},
    "liver":      {"liver"},
    "brain":      {"cerebral_cortex", "cerebellum"},
    "lung":       {"lung"},
    "bone":       {"bone_marrow"},
}

MET_SITES = tuple(MET_SITE_TISSUE_AUGMENTATION.keys())

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _sample_expression_by_symbol(df_gene_expr):
    import pandas as pd

    gene_id_col, gene_name_col = _guess_gene_cols(df_gene_expr)
    df = df_gene_expr.copy()
    df[gene_id_col] = df[gene_id_col].astype(str).map(_strip_ensembl_version)

    tpm_col = "TPM" if "TPM" in df.columns else next(
        (c for c in df.columns if c.lower() == "tpm"), None
    )
    if tpm_col is None:
        raise KeyError(f"No TPM column found. Columns: {list(df.columns)}")

    raw_values = df[tpm_col].astype(float)
    hk_mask = df[gene_id_col].isin(housekeeping_gene_ids())
    hk_median = df.loc[hk_mask, tpm_col].astype(float).median()
    if not (hk_median > 0):  # catches NaN and <= 0
        hk_median = 1.0
    hk_values = raw_values / hk_median

    # Resolve symbols from Ensembl IDs via pan-cancer reference
    ref_lookup = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    id_to_symbol = dict(zip(ref_lookup["Ensembl_Gene_ID"], ref_lookup["Symbol"]))
    if "canonical_gene_name" in df.columns:
        fallback = df["canonical_gene_name"].fillna("").astype(str)
    else:
        fallback = df[gene_name_col].fillna("").astype(str)
    symbols = df[gene_id_col].map(id_to_symbol).fillna(fallback)

    expr_df = pd.DataFrame(
        {
            "gene_id": df[gene_id_col],
            "Symbol": symbols,
            "sample_raw": raw_values,
            "sample_hk": hk_values,
        }
    )
    expr_df = expr_df[expr_df["Symbol"].astype(str).str.strip().ne("")]
    # Aggregate by Ensembl ID (unique), then map to symbol.
    # Sum across rows with same ID (alt-haplotype reads are split by aligner).
    grouped = expr_df.groupby("gene_id", as_index=False, sort=False).agg(
        {"Symbol": "first", "sample_raw": "sum", "sample_hk": "sum"}
    )
    return (
        dict(zip(grouped["Symbol"], grouped["sample_raw"])),
        dict(zip(grouped["Symbol"], grouped["sample_hk"])),
    )


def estimate_tumor_expression(
    df_gene_expr,
    cancer_type,
    purity,
):
    """Estimate true tumor cell expression by deconvolving TME contribution.

    For each gene: ``tumor_expr = (observed - (1-purity) * tme_ref) / purity``

    Genes are categorized into:
    - **CTA**: cancer-testis antigens (vaccination targets)
    - **therapy_target**: genes with active therapy trials
    - **surface**: known surface proteins (ADC/CAR-T/bispecific targets)
    - **other**: remaining genes with meaningful tumor signal

    Returns a DataFrame with columns: gene_id, symbol, category,
    observed_tpm, tme_expected, tumor_adjusted, tcga_median,
    tcga_percentile, is_surface, therapies.
    """
    import pandas as pd
    from .gene_sets_cancer import (
        surface_protein_gene_ids,
    )

    cancer_code = resolve_cancer_type(cancer_type)

    # Sample expression
    sample_raw, _ = _sample_expression_by_symbol(df_gene_expr)

    # Reference data
    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    ntpm_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    # TME tissues
    ptprc_row = ref_dedup.loc["PTPRC"] if "PTPRC" in ref_dedup.index else None
    if ptprc_row is not None:
        ptprc_vals = ptprc_row[ntpm_nonrepro].astype(float)
        immune_cols = [c for c in ntpm_nonrepro if ptprc_vals[c] > ptprc_vals.median()]
    else:
        immune_cols = []
    stromal_cols = [
        c for c in ntpm_nonrepro
        if c.replace("nTPM_", "") in _STROMAL_TISSUES
    ]
    tme_cols = list(set(immune_cols + stromal_cols))

    # Cancer type origin tissue: map cancer type to closest normal tissue
    cancer_col = f"FPKM_{cancer_code}"
    tcga_expr = ref_dedup[cancer_col].astype(float) if cancer_col in ref_dedup.columns else None

    # Build gene lookup sets
    cta_map = CTA_gene_id_to_name()  # {ensembl_id: name}
    cta_symbols = set(cta_map.values())

    # Therapy targets across all therapy types
    _all_therapy_keys = [
        "ADC", "ADC-approved", "CAR-T", "CAR-T-approved",
        "TCR-T", "TCR-T-approved", "bispecific-antibodies",
        "bispecific-antibodies-approved", "radioligand",
    ]
    gene_therapies = {}  # symbol -> set of base therapy types
    for tt in _all_therapy_keys:
        try:
            tmap = therapy_target_gene_id_to_name(tt)
            base = tt.replace("-approved", "").replace("-trials", "")
            for gid, gname in tmap.items():
                gene_therapies.setdefault(gname, set()).add(base)
        except Exception:
            pass
    fn1_support = _summarize_fn1_edb_transcript_support(df_gene_expr)

    # Surface proteins
    try:
        surf_ids = surface_protein_gene_ids()
        cancer_surf = cancer_surfaceome_gene_id_to_name()
        ref_flat = ref.drop_duplicates(subset="Ensembl_Gene_ID")
        eid_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
        surf_symbols = {eid_to_sym.get(eid, "") for eid in surf_ids}
        surf_symbols |= set(cancer_surf.values())
        surf_symbols.discard("")
    except Exception:
        surf_symbols = set()

    # TME reference: mean across TME tissues for each gene
    if tme_cols:
        tme_mean = ref_dedup[tme_cols].astype(float).mean(axis=1)
    else:
        tme_mean = pd.Series(0, index=ref_dedup.index)

    # TCGA distribution for percentile calculation
    cancer_expr_all = ref_dedup[fpkm_cols].astype(float)

    # Build result rows — only process genes that the sample expresses
    # or that are in a known target category
    interesting_symbols = set(cta_symbols) | set(gene_therapies.keys())
    interesting_symbols |= {s for s, v in sample_raw.items() if v > 0.1}

    rows = []
    purity_clamp = max(purity, 0.01)  # avoid division by zero

    for symbol in interesting_symbols:
        if symbol not in ref_dedup.index:
            continue
        observed = sample_raw.get(symbol, 0.0)
        tme_ref = float(tme_mean.get(symbol, 0))
        tcga_med = float(tcga_expr[symbol]) if tcga_expr is not None else 0.0

        # Purity adjustment
        tumor_adj = max(0, (observed - (1 - purity_clamp) * tme_ref) / purity_clamp)

        # TCGA percentile
        ref_vals = cancer_expr_all.loc[symbol].values
        n = len(ref_vals)
        below = np.sum(ref_vals < tumor_adj)
        equal = np.sum(np.isclose(ref_vals, tumor_adj, atol=0.01))
        pctile = float((below + 0.5 * equal) / n)

        # Categorize
        is_cta = symbol in cta_symbols
        is_surface = symbol in surf_symbols
        therapies, therapy_supported, therapy_support_note, therapy_support_tpm, therapy_support_fraction, therapy_supporting_transcripts = _apply_therapy_support_gate(
            symbol,
            gene_therapies.get(symbol, set()),
            fn1_support,
        )
        is_therapy = bool(therapies)

        # Filter: only include genes with meaningful tumor signal
        # or that are in a known category
        if tumor_adj < 0.5 and not is_cta and not is_therapy:
            continue

        if is_cta:
            category = "CTA"
        elif is_therapy:
            category = "therapy_target"
        elif is_surface and tumor_adj > 1:
            category = "surface"
        else:
            category = "other"

        eid = ref_dedup.loc[symbol, "Ensembl_Gene_ID"] if "Ensembl_Gene_ID" in ref_dedup.columns else ""

        rows.append({
            "gene_id": eid,
            "symbol": symbol,
            "category": category,
            "observed_tpm": round(observed, 2),
            "tme_expected": round(tme_ref, 2),
            "tumor_adjusted": round(tumor_adj, 2),
            "tcga_median": round(tcga_med, 2),
            "tcga_percentile": round(pctile, 3),
            "is_surface": is_surface,
            "is_cta": is_cta,
            "therapy_supported": therapy_supported,
            "therapy_support_note": therapy_support_note,
            "therapy_support_tpm": round(therapy_support_tpm, 2) if therapy_support_tpm is not None else None,
            "therapy_support_fraction": round(therapy_support_fraction, 3) if therapy_support_fraction is not None else None,
            "therapy_supporting_transcripts": therapy_supporting_transcripts,
            "therapies": ", ".join(sorted(therapies)) if therapies else "",
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("tumor_adjusted", ascending=False).reset_index(drop=True)
    return result


def estimate_tumor_expression_ranges(
    df_gene_expr,
    cancer_type,
    purity_result,
    decomposition_results=None,
    met_site=None,
):
    """Estimate tumor-specific expression with uncertainty bounds.

    For each gene, computes a 3x3 grid of estimates by crossing
    (low, med, high) TME background with (low, med, high) purity:

        tumor_expr = max(0, (observed - (1-purity) * tme_bg) / purity)

    TME bounds come from either:

    - the 25th / 50th / 75th percentile across TME reference tissues, or
    - the 25th / 50th / 75th percentile across candidate decomposition
      hypotheses when ``decomposition_results`` is provided.

    Purity bounds come from the ``overall_lower`` / ``overall_estimate`` /
    ``overall_upper`` fields of ``estimate_tumor_purity()``.

    Parameters
    ----------
    df_gene_expr : DataFrame
        Sample gene expression with TPM column.
    cancer_type : str
        TCGA cancer type code or name.
    purity_result : dict
        Return value of ``estimate_tumor_purity()``.
    decomposition_results : list, optional
        Candidate ``DecompositionResult`` objects from
        ``pirlygenes.decomposition.decompose_sample()``.

    Returns
    -------
    DataFrame with columns: symbol, category, observed_tpm,
        tme_lo, tme_med, tme_hi,
        est_1 ... est_9 (the 3x3 grid, ascending order),
        median_est, therapies, is_surface, is_cta.
    """
    import pandas as pd
    from .gene_sets_cancer import (
        surface_protein_gene_ids,
    )

    from .gene_sets_cancer import housekeeping_gene_ids
    from .tumor_purity import TCGA_MEDIAN_PURITY

    cancer_code = resolve_cancer_type(cancer_type)

    # --- Sample expression (raw TPM and HK-normalized) ---
    sample_raw, sample_hk = _sample_expression_by_symbol(df_gene_expr)

    # Sample HK median (for converting back from fold-HK to TPM)
    hk_ids = housekeeping_gene_ids()
    ref_full = pan_cancer_expression()
    ref_flat = ref_full.drop_duplicates(subset="Ensembl_Gene_ID")
    id_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
    hk_syms = {id_to_sym[gid] for gid in hk_ids if gid in id_to_sym}
    sample_hk_vals = [sample_raw[s] for s in hk_syms if sample_raw.get(s, 0) > 0]
    sample_hk_median = float(np.median(sample_hk_vals)) if sample_hk_vals else 1.0

    # --- Reference data ---
    ref_dedup = ref_full.drop_duplicates(subset="Symbol").set_index("Symbol")
    ntpm_cols = [c for c in ref_full.columns if c.startswith("nTPM_")]
    fpkm_cols = [c for c in ref_full.columns if c.startswith("FPKM_")]
    ntpm_nonrepro = [
        c for c in ntpm_cols
        if c.replace("nTPM_", "") not in _REPRODUCTIVE_TISSUES
    ]

    # TME tissues (curated immune + stromal). Met-site aware: when the
    # caller supplies ``met_site``, union in the host tissues for that
    # biopsy site so the TME background reflects, e.g., lymph-node
    # infiltrate for a nodal met (#13). Uniform median falls back to
    # the default curated set.
    effective_tme_tissues = set(_TME_TISSUES)
    if met_site:
        effective_tme_tissues |= MET_SITE_TISSUE_AUGMENTATION.get(met_site, set())
    tme_cols = [c for c in ntpm_nonrepro if c.replace("nTPM_", "") in effective_tme_tissues]

    # --- HK-normalize reference columns ---
    # Each column (nTPM tissue or FPKM cancer type) gets its own HK median
    hk_in_ref = sorted(hk_syms & set(ref_dedup.index))
    ref_hk_medians = {}
    for col in tme_cols + fpkm_cols:
        ref_hk_medians[col] = ref_dedup.loc[hk_in_ref, col].astype(float).median()

    # TME reference in HK-fold space: per gene, per tissue or decomposition.
    # When decomposition hypotheses are available, use their inferred
    # non-tumor background profile instead of a generic TME tissue panel.
    decomp_backgrounds = []
    if decomposition_results:
        for result in decomposition_results:
            bg = getattr(result, "tme_background_hk", None)
            if bg:
                decomp_backgrounds.append(bg)

    # Per-gene max across healthy tissues (precomputed vector; used to
    # flag rows where sample signal could be entirely TME-explained).
    # Vectorized once here to avoid a per-gene `.loc[].max()` inside the
    # main loop — that was a measured ~30x slowdown on full panels.
    if ntpm_nonrepro:
        _max_healthy = ref_dedup[ntpm_nonrepro].astype(float).max(axis=1)
        max_healthy_tpm_by_symbol = _max_healthy.to_dict()
    else:
        max_healthy_tpm_by_symbol = {}

    # --- Full-coverage TPM-space TME background (fixes issue #45) --------
    #
    # The decomposition's `tme_background_hk` dict only covers genes that
    # were in the decomposition's signature panel. For target-list genes
    # like FN1 / COL1A1 / IGKC that AREN'T signature genes but ARE clearly
    # stromal/immune expressed, `tme_background_hk.get(sym, 0.0)` returns 0
    # -> no TME subtraction -> `tumor_tpm ~ sample_tpm / purity` which
    # inflates the "Tumor TPM" reported in the target table.
    #
    # Fix: when we have a decomposition result with a `fractions` dict,
    # build a full-gene signature matrix using the same cell-type /
    # bulk-tissue references the decomposition engine uses, and compute:
    #
    #   tme_bg_tpm[g] = sum_{c != tumor} fractions[c] * ref_tpm[g, c]
    #
    # This gives a per-gene TPM-scale TME background for every gene in
    # the reference, not just signature genes. The formula in the loop
    # prefers this TPM-space path when available and falls back to the
    # HK-fold path when it isn't.
    #
    # Matched-normal split (issue #50): when the decomposition included a
    # `matched_normal_<tissue>` component (epithelial primaries with
    # `use_matched_normal=True`), its contribution is tracked separately
    # as `matched_normal_tpm_by_symbol`. This lets the target report
    # distinguish "subtracted stromal/immune background" from "subtracted
    # benign parent-tissue contribution" per gene. The formula sums
    # them -- total non-tumor background is still `tme_only + matched_normal`.
    tme_bg_tpm_by_symbol = None
    tme_only_tpm_by_symbol = None
    matched_normal_tpm_by_symbol = None
    matched_normal_component_name = None
    matched_normal_tissue = None
    matched_normal_fraction_global = 0.0
    per_compartment_tpm_by_symbol = None  # #108: per-gene per-compartment TPM
    if decomposition_results:
        top_result = decomposition_results[0]
        top_fractions = getattr(top_result, "fractions", None) or {}
        non_tumor_components = [
            c for c, f in top_fractions.items() if c != "tumor" and f > 0
        ]
        matched_normal_component_name = getattr(
            top_result, "matched_normal_tissue", None,
        )
        if matched_normal_component_name is not None:
            matched_normal_tissue = matched_normal_component_name
            matched_normal_component_name = f"matched_normal_{matched_normal_tissue}"
            matched_normal_fraction_global = float(
                getattr(top_result, "matched_normal_fraction", 0.0) or 0.0
            )
        if non_tumor_components:
            from .decomposition.signature import build_signature_matrix
            try:
                _genes, sym_list, matrix, _cols = build_signature_matrix(
                    non_tumor_components,
                    gene_subset=None,
                    sample_by_eid=None,
                )
                non_tumor_fracs = np.array(
                    [float(top_fractions[c]) for c in non_tumor_components],
                    dtype=float,
                )
                # Per-gene expected non-tumor contribution to sample TPM:
                # sum_c fractions[c] * ref_tpm[g, c]. `fractions[c]` is
                # already scaled by (1 - tumor_fraction), so summing
                # directly gives absolute non-tumor TPM -- no extra
                # (1 - p) multiplier needed at the formula site.
                tme_tpm_vec = matrix @ non_tumor_fracs
                tme_bg_tpm_by_symbol = {
                    str(sym): float(val)
                    for sym, val in zip(sym_list, tme_tpm_vec)
                }
                # #108: keep the per-compartment breakdown so target
                # rendering can show attribution columns instead of only
                # a collapsed TME total. Each entry maps a gene symbol to
                # a {compartment: attributed_tpm} dict (non-zero only).
                per_comp_mat = matrix * non_tumor_fracs[np.newaxis, :]
                per_compartment_tpm_by_symbol = {}
                for i, sym in enumerate(sym_list):
                    breakdown = {}
                    for j, comp in enumerate(non_tumor_components):
                        val = float(per_comp_mat[i, j])
                        if val >= 0.01:
                            breakdown[comp] = round(val, 2)
                    if breakdown:
                        per_compartment_tpm_by_symbol[str(sym)] = breakdown
                # Split out the matched-normal contribution so the report
                # can distinguish TME-only subtraction from parent-tissue
                # subtraction (issue #50). The matched-normal column is
                # present only when the decomposition was run with
                # `use_matched_normal=True` on an epithelial primary.
                if matched_normal_component_name in non_tumor_components:
                    mn_idx = non_tumor_components.index(matched_normal_component_name)
                    mn_frac = float(non_tumor_fracs[mn_idx])
                    mn_vec = matrix[:, mn_idx] * mn_frac
                    matched_normal_tpm_by_symbol = {
                        str(sym): float(val)
                        for sym, val in zip(sym_list, mn_vec)
                    }
                    tme_only_vec = tme_tpm_vec - mn_vec
                    tme_only_tpm_by_symbol = {
                        str(sym): float(max(0.0, val))
                        for sym, val in zip(sym_list, tme_only_vec)
                    }
                else:
                    tme_only_tpm_by_symbol = dict(tme_bg_tpm_by_symbol)
            except Exception:
                tme_bg_tpm_by_symbol = None
                tme_only_tpm_by_symbol = None
                matched_normal_tpm_by_symbol = None

    # --- Purity-adjusted TCGA (HK-normalized, then deconvolved) ---
    # For each FPKM cancer-type column, compute:
    #   tcga_hk = FPKM / FPKM_HK_median
    #   tme_hk  = median TME tissue fold (same as sample TME reference)
    #   tcga_tumor_hk = (tcga_hk - (1-tcga_purity) * tme_hk) / tcga_purity
    # We'll compute this per-gene in the loop.

    # --- Purity bounds ---
    p_lo = max(purity_result.get("overall_lower") or 0.01, 0.01)
    p_med = max(purity_result.get("overall_estimate") or 0.05, 0.01)
    p_hi = max(purity_result.get("overall_upper") or p_med, 0.01)
    p_lo, p_med, p_hi = sorted([p_lo, p_med, p_hi])

    # --- Gene category lookups ---
    cta_symbols = set(CTA_gene_id_to_name().values())

    _all_therapy_keys = [
        "ADC", "ADC-approved", "CAR-T", "CAR-T-approved",
        "TCR-T", "TCR-T-approved", "bispecific-antibodies",
        "bispecific-antibodies-approved", "radioligand",
    ]
    gene_therapies = {}
    for tt in _all_therapy_keys:
        try:
            tmap = therapy_target_gene_id_to_name(tt)
            base = tt.replace("-approved", "").replace("-trials", "")
            for gid, gname in tmap.items():
                gene_therapies.setdefault(gname, set()).add(base)
        except Exception:
            pass
    fn1_support = _summarize_fn1_edb_transcript_support(df_gene_expr)

    try:
        surf_ids = surface_protein_gene_ids()
        cancer_surf = cancer_surfaceome_gene_id_to_name()
        eid_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
        surf_symbols = {eid_to_sym.get(eid, "") for eid in surf_ids}
        surf_symbols |= set(cancer_surf.values())
        surf_symbols.discard("")
    except Exception:
        surf_symbols = set()

    # --- Compute 9-point estimates for every expressed gene ---
    cancer_expr_all = ref_dedup[fpkm_cols].astype(float)
    rows = []
    for symbol in sample_raw:
        if symbol not in ref_dedup.index:
            continue
        observed = sample_raw[symbol]
        if observed < 0.01:
            continue

        # HK-normalize sample
        sample_fold = observed / sample_hk_median

        # TME background in HK-fold space.
        if decomp_backgrounds:
            tme_folds = [float(bg.get(symbol, 0.0)) for bg in decomp_backgrounds]
        else:
            tme_folds = []
            for col in tme_cols:
                hk_m = ref_hk_medians.get(col, 0)
                if hk_m > 0:
                    tme_folds.append(float(ref_dedup.loc[symbol, col]) / hk_m)
        if not tme_folds:
            tme_folds = [0.0]
        tme_fold_lo = float(np.percentile(tme_folds, 25))
        tme_fold_med = float(np.median(tme_folds))
        tme_fold_hi = float(np.percentile(tme_folds, 75))

        # TME-explainability flag: the max TPM this gene reaches in ANY
        # single healthy reference tissue. Used below to decide how
        # aggressively to clamp toward the cohort prior.
        max_healthy_tpm = float(max_healthy_tpm_by_symbol.get(symbol, 0.0))
        tme_explainable = max_healthy_tpm >= observed * 0.5

        # Cohort prior for tumor expression in this cancer type. Computed
        # by deconvolving the TCGA cancer-cohort median against the cohort's
        # assumed median purity, then rescaling to the sample's TPM scale.
        # This prior implicitly includes the "reactive stroma" signal
        # typical for this cancer type (TCGA medians are bulk tumor
        # samples, so tissue-specific cancer-associated fibroblast and
        # infiltrate contributions are baked in). Used for empirical-
        # Bayes shrinkage of the sample-based estimate at low purity.
        cancer_col = f"FPKM_{cancer_code}"
        cohort_prior_tpm = 0.0
        tcga_tumor_fold = 0.0
        if (
            cancer_col in ref_dedup.columns
            and cancer_col in ref_hk_medians
            and ref_hk_medians[cancer_col] > 0
        ):
            cancer_hk_m = ref_hk_medians[cancer_col]
            tcga_fold = float(ref_dedup.loc[symbol, cancer_col]) / cancer_hk_m
            tcga_p = TCGA_MEDIAN_PURITY.get(cancer_code, 0.7)
            tcga_tumor_fold = max(
                0.0, (tcga_fold - (1 - tcga_p) * tme_fold_med) / tcga_p
            )
            cohort_prior_tpm = tcga_tumor_fold * sample_hk_median

        # Empirical-Bayes shrinkage weight. At high purity the sample-
        # based estimate is reliable (w_sample -> 1). At low purity the
        # 1/p division in the deconvolution inflates noise; shrinkage
        # pulls estimates back toward the cohort prior (w_sample -> 0
        # as purity -> 0). `k_shrinkage` is the purity at which weights
        # are 50/50 -- tuned so that CTAs and real tumor markers at
        # moderate purity (~0.4) are mostly sample-driven, while
        # low-purity stromal-like genes get anchored to cohort.
        #
        # `k_shrinkage` doubles for tme_explainable genes (to ~2x
        # stronger shrinkage), reflecting the extra uncertainty about
        # whether the signal is genuinely tumor-cell-derived.
        k_shrinkage = 0.20
        if tme_explainable:
            k_shrinkage = 0.40

        # Shrinkage floor: when the cohort prior is near-zero (e.g.
        # CTAs -- activated in a minority of samples, so median = 0),
        # the prior mean is uninformative. Empirical-Bayes shrinkage
        # toward 0 would wrongly pull real sample-specific signal
        # down. Skip shrinkage in that regime and trust the sample.
        skip_shrinkage = cohort_prior_tpm < 1.0

        # 9-point estimate grid. TPM-space path (preferred) uses the
        # decomposition's per-gene expected TME contribution directly.
        # HK-fold path is the fallback when no decomposition is
        # available or the gene is absent from the reference. Every
        # grid point is shrunk toward the cohort prior and clamped at
        # `observed_tpm` when tme_explainable=True (tumor cells can't
        # contribute more than the observed signal if a single healthy
        # tissue could explain it alone).
        def _apply_priors(raw_tumor_tpm, purity_used):
            if skip_shrinkage:
                shrunk = raw_tumor_tpm
            else:
                w_sample = float(purity_used) / (float(purity_used) + k_shrinkage)
                shrunk = (
                    w_sample * raw_tumor_tpm
                    + (1.0 - w_sample) * cohort_prior_tpm
                )
            if tme_explainable:
                shrunk = min(shrunk, observed)
            return max(0.0, shrunk)

        estimates = []
        if tme_bg_tpm_by_symbol is not None and symbol in tme_bg_tpm_by_symbol:
            bg_tpm_mid = tme_bg_tpm_by_symbol[symbol]
            # Uncertainty on the TME estimate itself: +/-50%. Reference
            # cell-type profiles may under- or over-estimate the actual
            # infiltrate composition in the specific sample.
            for bg_tpm in [bg_tpm_mid * 0.5, bg_tpm_mid, bg_tpm_mid * 1.5]:
                for p in [p_lo, p_med, p_hi]:
                    raw = max(0.0, (observed - bg_tpm)) / p
                    estimates.append(_apply_priors(raw, p))
        else:
            for bg in [tme_fold_lo, tme_fold_med, tme_fold_hi]:
                for p in [p_lo, p_med, p_hi]:
                    tumor_fold = max(0.0, (sample_fold - (1 - p) * bg) / p)
                    raw = tumor_fold * sample_hk_median
                    estimates.append(_apply_priors(raw, p))
        estimates.sort()
        median_est = float(np.median(estimates))

        ref_vals = cancer_expr_all.loc[symbol].values
        n = len(ref_vals)
        below = np.sum(ref_vals < median_est)
        equal = np.sum(np.isclose(ref_vals, median_est, atol=0.01))
        tcga_percentile = float((below + 0.5 * equal) / n)

        # Ratio vs purity-adjusted TCGA median for matched cancer type.
        # Uses the already-computed cohort tumor fold above.
        #
        # Three outcomes, each consumed by a distinct plot label:
        #   - finite positive -> fold-change vs TCGA ("0.3x", "1.5x", ...)
        #   - inf             -> sample expresses the gene but the TCGA
        #                       cohort tumor-component is essentially zero.
        #                       Rendered as a red "absent in TCGA" alert --
        #                       flags atypical expression for this cancer.
        #   - None            -> both the sample tumor-component and TCGA
        #                       tumor-component are essentially zero.
        #                       Rendered as a gray "0 in TCGA" -- nothing to
        #                       compare.
        #
        # `tcga_tumor_fold` clips to exactly 0.0 whenever the TCGA cohort
        # median is explainable by TME alone; previously the <= 0 branch
        # collapsed to None unconditionally, so a CTA with FPKM_<cancer>
        # median ~ 0 and strong sample expression rendered as a quiet gray
        # label instead of the intended red "absent in TCGA" alert. The
        # sample-side check below restores the intended semantics.
        our_tumor_fold = median_est / sample_hk_median
        if tcga_tumor_fold > 0.001:
            vs_tcga = float(our_tumor_fold / tcga_tumor_fold)
        elif our_tumor_fold > 0.001:
            vs_tcga = float("inf")
        else:
            vs_tcga = None

        # Categorize
        is_cta = symbol in cta_symbols
        is_surface = symbol in surf_symbols
        therapies, therapy_supported, therapy_support_note, therapy_support_tpm, therapy_support_fraction, therapy_supporting_transcripts = _apply_therapy_support_gate(
            symbol,
            gene_therapies.get(symbol, set()),
            fn1_support,
        )

        if is_cta:
            category = "CTA"
        elif therapies:
            category = "therapy_target"
        elif is_surface:
            category = "surface"
        else:
            category = "other"

        eid = ref_dedup.loc[symbol, "Ensembl_Gene_ID"] if "Ensembl_Gene_ID" in ref_dedup.columns else ""

        # Per-gene matched-normal split reporting (issue #50). Zero when
        # no matched-normal component is active or the gene isn't in the
        # signature matrix. `estimation_path` records which branch
        # produced the estimate, so the target report can annotate each
        # gene with its provenance.
        mn_tpm = 0.0
        if matched_normal_tpm_by_symbol is not None:
            mn_tpm = float(matched_normal_tpm_by_symbol.get(symbol, 0.0))
        if tme_only_tpm_by_symbol is not None:
            tme_only_tpm = float(tme_only_tpm_by_symbol.get(symbol, 0.0))
        elif tme_bg_tpm_by_symbol is not None:
            tme_only_tpm = float(tme_bg_tpm_by_symbol.get(symbol, 0.0))
        else:
            tme_only_tpm = 0.0

        if tme_explainable:
            estimation_path = "clamped"
        elif mn_tpm > 0.0:
            estimation_path = "matched_normal_split"
        elif tme_bg_tpm_by_symbol is not None and symbol in tme_bg_tpm_by_symbol:
            estimation_path = "tme_only"
        else:
            estimation_path = "tme_fold_fallback"

        # #60: mark extended-housekeeping symbols so downstream
        # target tables can filter them out without silently dropping
        # rows from the TSV (power users still see them with the flag).
        excluded_from_ranking = bool(
            is_extended_housekeeping_symbol(symbol, scope="ranking")
        )

        # #108: per-compartment attribution. When decomposition ran,
        # apportion the observed TPM across TME compartments using the
        # per-compartment reference × fitted fractions, then derive:
        #   attr_tumor_tpm = max(0, observed - sum(attr_compartments))
        # and tumor fraction of total. These drive the Attribution
        # column in targets.md and the per-target stacked-bar figure.
        attribution = {}
        if per_compartment_tpm_by_symbol is not None:
            raw = per_compartment_tpm_by_symbol.get(symbol)
            if raw:
                attribution = dict(raw)
        attr_tme_total = sum(attribution.values())
        attr_tumor_tpm = max(0.0, observed - attr_tme_total)
        attr_tumor_fraction = (
            float(attr_tumor_tpm / observed) if observed > 0 else 0.0
        )
        if attribution:
            attr_top_comp, attr_top_tpm = max(
                attribution.items(), key=lambda kv: kv[1]
            )
        else:
            attr_top_comp, attr_top_tpm = "", 0.0

        # #35: a gene whose observed TPM is mostly explained by non-
        # tumor compartments is a low-confidence tumor-expression claim,
        # especially risky at low purity where residual TPM is divided
        # by a small number and amplified. When decomposition attribution
        # is available, fire the flag from `attr_tumor_fraction < 0.3`
        # so it's grounded in the fitted compartments instead of a
        # generic TME-fold. Fall back to the old formula when no
        # decomposition ran.
        if attribution:
            tme_dominant = observed > 0 and attr_tumor_fraction < 0.30
        else:
            tme_dominant = (
                observed > 0
                and round(tme_fold_med, 4) * sample_hk_median >= round(0.7 * observed, 4)
            )
        low_confidence_tumor = bool(tme_dominant)

        rows.append({
            "gene_id": eid,
            "symbol": symbol,
            "category": category,
            "observed_tpm": round(observed, 2),
            "tme_fold_lo": round(tme_fold_lo, 4),
            "tme_fold_med": round(tme_fold_med, 4),
            "tme_fold_hi": round(tme_fold_hi, 4),
            "max_healthy_tpm": round(max_healthy_tpm, 2),
            "tme_explainable": bool(tme_explainable),
            "tme_dominant": tme_dominant,
            "low_confidence_tumor": low_confidence_tumor,
            "cohort_prior_tpm": round(cohort_prior_tpm, 2),
            "tme_only_tpm": round(tme_only_tpm, 2),
            "matched_normal_tpm": round(mn_tpm, 2),
            "matched_normal_tissue": matched_normal_tissue or "",
            "matched_normal_fraction": round(matched_normal_fraction_global, 4),
            "estimation_path": estimation_path,
            # #108: per-compartment attribution. `attribution` is a dict
            # of {compartment: attributed_tpm}; `attr_tumor_tpm` is the
            # residual after subtracting those compartments; the top-
            # compartment shortcut keeps the common case cheap for
            # markdown rendering.
            "attribution": attribution,
            "attr_tumor_tpm": round(attr_tumor_tpm, 2),
            "attr_tumor_fraction": round(attr_tumor_fraction, 4),
            "attr_top_compartment": attr_top_comp,
            "attr_top_compartment_tpm": round(float(attr_top_tpm), 2),
            **{f"est_{i+1}": round(estimates[i], 2) for i in range(9)},
            "median_est": round(median_est, 2),
            "pct_cancer_median": round(vs_tcga, 2) if vs_tcga is not None else None,
            "tcga_percentile": round(tcga_percentile, 3),
            "is_surface": is_surface,
            "is_cta": is_cta,
            "excluded_from_ranking": excluded_from_ranking,
            "therapy_supported": therapy_supported,
            "therapy_support_note": therapy_support_note,
            "therapy_support_tpm": round(therapy_support_tpm, 2) if therapy_support_tpm is not None else None,
            "therapy_support_fraction": round(therapy_support_fraction, 3) if therapy_support_fraction is not None else None,
            "therapy_supporting_transcripts": therapy_supporting_transcripts,
            "therapies": ", ".join(sorted(therapies)) if therapies else "",
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("median_est", ascending=False).reset_index(drop=True)
    return result


def plot_matched_normal_attribution(
    df_ranges,
    cancer_type,
    category,
    top_n=15,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Stacked horizontal-bar plot of per-gene tumor / matched-normal / TME
    attribution for a single target category (issue #55).

    For each of the top ``top_n`` genes in ``category`` (ranked by
    ``median_est``), draw a horizontal bar broken into:

    - ``tumor``: ``observed_tpm - matched_normal_tpm - tme_only_tpm``
    - ``matched_normal_tpm``: benign parent-tissue contribution
    - ``tme_only_tpm``: stromal / immune / host-tissue contribution

    Only useful when ``df_ranges`` carries non-zero ``matched_normal_tpm``
    for at least one gene in the category (i.e. the decomposition ran
    with ``use_matched_normal=True`` for an epithelial primary). Returns
    ``None`` otherwise -- the CLI uses that to skip emitting an empty
    figure.

    Emitted as a standalone PNG (one per category) rather than as a
    panel in a composite figure, following the project's plot-crowding
    preference.
    """

    cancer_code = resolve_cancer_type(cancer_type)

    sub = df_ranges[df_ranges["category"] == category].head(top_n).copy()
    if sub.empty:
        return None
    if "matched_normal_tpm" not in sub.columns:
        return None
    if (sub["matched_normal_tpm"].astype(float) <= 0).all():
        return None

    sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)
    n = len(sub)
    if figsize is None:
        figsize = (10, max(3.0, 0.4 * n + 1.5))

    observed = sub["observed_tpm"].astype(float).values
    mn = sub["matched_normal_tpm"].astype(float).values
    tme = sub["tme_only_tpm"].astype(float).values
    # Tumor-cell attribution is whatever observed signal remains after
    # TME and matched-normal are subtracted. Floor at 0 to guard against
    # tiny over-subtractions from solver jitter.
    tumor_attr = np.maximum(0.0, observed - mn - tme)

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, tumor_attr, color="#e74c3c", label="tumor cells")
    ax.barh(y, mn, left=tumor_attr, color="#3498db", label="matched-normal tissue")
    ax.barh(y, tme, left=tumor_attr + mn, color="#95a5a6", label="other TME (stromal/immune)")

    # Symbols on the y-axis. Therapy-target annotation appended when present.
    labels = []
    for _, row in sub.iterrows():
        sym = str(row["symbol"])
        if row.get("therapies"):
            sym = f"{sym}  [{row['therapies']}]"
        flags = []
        if row.get("tme_explainable"):
            flags.append("\u26a0")
        path = str(row.get("estimation_path", ""))
        if path == "clamped":
            flags.append("clamp")
        if flags:
            sym = f"{sym}  {' '.join(flags)}"
        labels.append(sym)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    # Cohort-prior marker: small tick on each bar so reviewers can see
    # where the TCGA cohort would have placed tumor-cell expression.
    if "cohort_prior_tpm" in sub.columns:
        priors = sub["cohort_prior_tpm"].astype(float).values
        for i, prior in enumerate(priors):
            if prior > 0:
                ax.plot([prior], [i], marker="|", color="black", markersize=14, markeredgewidth=2)

    ax.set_xlabel("TPM (stacked: tumor + matched-normal + other TME = observed)", fontsize=10)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    mn_tissue = ""
    if "matched_normal_tissue" in sub.columns:
        nonempty = sub["matched_normal_tissue"].astype(str).replace("nan", "")
        nonempty = [v for v in nonempty.unique() if v]
        if nonempty:
            mn_tissue = nonempty[0]
    title = f"Matched-normal attribution \u2014 {cancer_code} {category}"
    if mn_tissue:
        title += f"\n(benign {mn_tissue} subtracted before purity division; black tick = TCGA cohort prior)"
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_target_attribution(
    df_ranges,
    cancer_type,
    category,
    top_n=15,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Per-target compositional attribution stacked bars (#108).

    For each of the top ``top_n`` targets in ``category`` (ranked by
    ``median_est``), draw a horizontal bar broken into the tumor core
    plus each non-tumor compartment that contributes at least 1% of the
    observed TPM. Compartments are read from the ``attribution`` column
    of ``df_ranges`` (a dict of {compartment: TPM}); tumor-core TPM is
    taken from ``attr_tumor_tpm``.

    Emitted as a standalone PNG (one per category, per project
    plot-crowding preference). Returns ``None`` and does not write a
    file when no row in the category has an attribution breakdown (e.g.
    decomposition didn't run, or none of the top targets overlapped the
    reference matrix).
    """
    cancer_code = resolve_cancer_type(cancer_type)
    sub = df_ranges[df_ranges["category"] == category].head(top_n).copy()
    if sub.empty or "attribution" not in sub.columns:
        return None

    def _has_breakdown(v):
        return isinstance(v, dict) and len(v) > 0

    if not sub["attribution"].apply(_has_breakdown).any():
        return None

    sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)
    n = len(sub)
    if figsize is None:
        figsize = (11, max(3.0, 0.4 * n + 1.5))

    # Collect all compartments that appear across the shown targets, in
    # descending order of aggregate contribution so the legend ranks the
    # most-impactful compartments first.
    totals = {}
    for attr in sub["attribution"]:
        if not isinstance(attr, dict):
            continue
        for comp, tpm in attr.items():
            totals[comp] = totals.get(comp, 0.0) + float(tpm)
    compartments = sorted(totals, key=lambda c: -totals[c])
    palette = plt.cm.tab20.colors
    comp_colors = {c: palette[i % len(palette)] for i, c in enumerate(compartments)}

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    tumor_attr = sub["attr_tumor_tpm"].astype(float).values
    ax.barh(y, tumor_attr, color="#e74c3c", label="tumor core")
    left = tumor_attr.copy()
    for comp in compartments:
        vals = np.array([
            float(attr.get(comp, 0.0)) if isinstance(attr, dict) else 0.0
            for attr in sub["attribution"]
        ])
        if not np.any(vals > 0):
            continue
        ax.barh(y, vals, left=left, color=comp_colors[comp],
                label=comp.replace("_", " "))
        left = left + vals

    labels = []
    for _, row in sub.iterrows():
        sym = str(row["symbol"])
        if row.get("therapies"):
            sym = f"{sym}  [{row['therapies']}]"
        flags = []
        if row.get("tme_dominant"):
            flags.append("\u26a0\u26a0")
        elif row.get("tme_explainable"):
            flags.append("\u26a0")
        if flags:
            sym = f"{sym}  {' '.join(flags)}"
        labels.append(sym)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel(
        "TPM (stacked: tumor core + non-tumor compartments = observed)",
        fontsize=10,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=1)
    ax.set_title(
        f"Per-target compositional attribution \u2014 {cancer_code} {category}\n"
        "(\u26a0\u26a0 = tumor < 30% of observed; \u26a0 = single-tissue-explainable)",
        fontsize=10, fontweight="bold",
    )

    plt.tight_layout()
    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig


def plot_tumor_expression_ranges(
    df_ranges,
    purity_result,
    cancer_type,
    top_n=15,
    categories=None,
    save_to_filename=None,
    save_dpi=300,
    figsize=None,
):
    """Strip plot of 9-point tumor expression estimates per gene.

    Parameters
    ----------
    df_ranges : DataFrame
        Output of ``estimate_tumor_expression_ranges()``.
    purity_result : dict
        Output of ``estimate_tumor_purity()``.
    cancer_type : str
        Cancer type code for title.
    top_n : int
        Max genes per category panel.
    categories : list of str, optional
        Which categories to plot. Default: CTA, therapy_target, surface.
    save_to_filename : str, optional
        Path to save the figure.
    """

    if categories is None:
        categories = ["therapy_target", "CTA", "surface"]

    cancer_code = resolve_cancer_type(cancer_type)
    p_lo = max(purity_result.get("overall_lower") or 0.01, 0.01)
    p_med = max(purity_result.get("overall_estimate") or 0.05, 0.01)
    p_hi = max(purity_result.get("overall_upper") or p_med, 0.01)

    cat_titles = {
        "CTA": "Cancer-Testis Antigens",
        "therapy_target": "Therapeutic Targets",
        "surface": "Surface Proteins",
        "other": "Other Tumor Genes",
    }
    cat_colors = {
        "CTA": "#e74c3c",
        "therapy_target": "#3498db",
        "surface": "#2ecc71",
        "other": "#95a5a6",
    }
    # Count genes per panel to size adaptively
    panel_counts = []
    for cat in categories:
        n = min(top_n, len(df_ranges[df_ranges["category"] == cat]))
        panel_counts.append(max(n, 1))
    total_genes = sum(panel_counts)

    n_panels = len(categories)
    if figsize is None:
        # ~0.4 inches per gene row, minimum 2 inches per panel
        panel_heights = [max(2.0, 0.4 * n) for n in panel_counts]
        figsize = (14, sum(panel_heights) + 1.5)

    fig, axes = plt.subplots(
        n_panels, 2, figsize=figsize, squeeze=False,
        gridspec_kw={
            "width_ratios": [3, 1],
            "height_ratios": panel_counts,
        },
    )
    est_cols = [f"est_{i+1}" for i in range(9)]

    # Adaptive font/marker sizes: larger when fewer genes
    base_font = min(12, max(8, int(200 / max(total_genes, 1))))
    marker_s = min(80, max(30, int(600 / max(total_genes, 1))))
    diamond_s = min(120, max(50, int(900 / max(total_genes, 1))))

    for ax_idx, cat in enumerate(categories):
        ax_strip = axes[ax_idx, 0]
        ax_pct = axes[ax_idx, 1]
        sub = df_ranges[df_ranges["category"] == cat].head(top_n).copy()
        sub = sub.sort_values("median_est", ascending=True).reset_index(drop=True)

        if sub.empty:
            ax_strip.set_title(cat_titles.get(cat, cat))
            ax_strip.text(0.5, 0.5, "No genes", ha="center", va="center",
                          transform=ax_strip.transAxes, fontsize=base_font, color="gray")
            ax_pct.set_visible(False)
            continue

        color = cat_colors.get(cat, "#95a5a6")
        y_positions = np.arange(len(sub))

        # --- Left panel: 9-point strip plot ---
        for i, (_, row) in enumerate(sub.iterrows()):
            vals = [row[c] for c in est_cols]
            vals_plot = [max(v, 0.01) for v in vals]
            median_v = max(row["median_est"], 0.01)

            ax_strip.scatter(vals_plot, [i] * 9, color=color, alpha=0.4,
                             s=marker_s, zorder=3)
            ax_strip.scatter([median_v], [i], color=color, marker="D",
                             s=diamond_s, edgecolors="black", linewidths=0.7,
                             zorder=5)
            ax_strip.plot([min(vals_plot), max(vals_plot)], [i, i],
                          color=color, alpha=0.3, linewidth=2.5, zorder=2)

        labels = []
        for _, row in sub.iterrows():
            label = row["symbol"]
            if row["therapies"]:
                label += f"  [{row['therapies']}]"
            labels.append(label)

        ax_strip.set_yticks(y_positions)
        ax_strip.set_yticklabels(labels, fontsize=base_font)
        ax_strip.set_xscale("log")
        ax_strip.set_xlabel("Tumor-specific expression (TPM)", fontsize=base_font)
        ax_strip.set_title(cat_titles.get(cat, cat), fontsize=base_font + 2,
                           fontweight="bold", color=color)
        ax_strip.set_ylim(-0.5, len(sub) - 0.5)
        ax_strip.grid(axis="x", alpha=0.2)

        # --- Right panel: % of cancer type median (log scale) ---
        for i, (_, row) in enumerate(sub.iterrows()):
            pct = row.get("pct_cancer_median")
            if pct is None or (isinstance(pct, float) and np.isnan(pct)):
                # Gene not in TCGA reference for this cancer type
                ax_pct.text(0.5, i, "0 in TCGA", fontsize=base_font - 2,
                            color="gray", ha="center", va="center",
                            transform=ax_pct.get_yaxis_transform())
                continue
            if isinstance(pct, float) and np.isinf(pct):
                # Sample expresses the gene but the TCGA cohort reference is
                # zero -- so this gene is absent from TCGA {cancer_code} but
                # present in THIS sample. Draw a solid dark-red band across
                # the row with white text inside so the reader can't mistake
                # it for a tissue-decomposition claim or a general property
                # of the cancer type.
                ax_pct.axhspan(i - 0.35, i + 0.35, color="#6b0000",
                               alpha=1.0, zorder=3, linewidth=0)
                ax_pct.text(0.5, i, f"absent in TCGA {cancer_code}",
                            fontsize=base_font - 2, color="white",
                            fontweight="bold", ha="center", va="center",
                            zorder=4,
                            transform=ax_pct.get_yaxis_transform())
                continue
            bar_color = color if pct >= 0.5 else "#d4a017"
            ax_pct.barh(i, max(pct, 0.001), color=bar_color, alpha=0.7, height=0.6)
            lbl = f"{pct:.1f}\u00d7" if pct < 10 else f"{pct:.0f}\u00d7"
            ax_pct.text(max(pct, 0.001) * 1.2, i, lbl, fontsize=base_font - 1,
                        va="center", color="black")

        ax_pct.set_xscale("log")
        ax_pct.axvline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax_pct.set_yticks([])
        ax_pct.set_xlabel(f"vs {cancer_code} median", fontsize=base_font)
        ax_pct.set_title(f"vs {cancer_code}", fontsize=base_font, color="gray")
        ax_pct.set_ylim(-0.5, len(sub) - 0.5)
        ax_pct.grid(axis="x", alpha=0.15)

    # Suptitle with purity info and caveat
    fig.suptitle(
        f"Purity-adjusted tumor expression \u2014 {cancer_code}\n"
        f"Purity: {p_lo:.0%} / {p_med:.0%} / {p_hi:.0%} (low / est / high)\n"
        f"Values are deconvolved estimates \u2014 may overstate expression at low purity",
        fontsize=10, fontweight="bold", y=1.04,
    )
    plt.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")

    return fig


def plot_purity_adjusted_targets(
    df_gene_expr,
    cancer_type,
    purity,
    save_to_filename=None,
    save_dpi=300,
    figsize=(14, 10),
    top_n=40,
):
    """Plot purity-adjusted tumor expression for key gene categories.

    Shows observed vs purity-adjusted expression for CTAs, therapy
    targets, and surface proteins, with TCGA percentile context.
    """
    import pandas as pd

    adj = estimate_tumor_expression(df_gene_expr, cancer_type, purity)
    cancer_code = resolve_cancer_type(cancer_type)

    # Select top genes per category
    categories = ["CTA", "therapy_target", "surface"]
    selected = []
    for cat in categories:
        sub = adj[adj["category"] == cat].head(top_n // len(categories))
        selected.append(sub)
    selected = pd.concat(selected, ignore_index=True) if selected else adj.head(0)
    # Add high-expression "other" if space remains
    remaining = top_n - len(selected)
    if remaining > 0:
        other = adj[(adj["category"] == "other") & (adj["tumor_adjusted"] > 10)]
        selected = pd.concat([selected, other.head(remaining)], ignore_index=True)

    if selected.empty:
        return None

    selected = selected.sort_values(
        ["category", "tumor_adjusted"], ascending=[True, False]
    ).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )

    # Left: horizontal bar chart of purity-adjusted expression
    y = np.arange(len(selected))
    cat_colors = {
        "CTA": "#e74c3c",
        "therapy_target": "#3498db",
        "surface": "#2ecc71",
        "other": "#95a5a6",
    }
    colors = [cat_colors.get(c, "#95a5a6") for c in selected["category"]]

    ax1.barh(y, selected["tumor_adjusted"], color=colors, alpha=0.8, height=0.7)
    # Overlay observed as dots
    ax1.scatter(
        selected["observed_tpm"], y,
        color="black", s=20, zorder=5, label="observed TPM"
    )
    ax1.set_yticks(y)
    labels = []
    for _, row in selected.iterrows():
        suffix = ""
        if row["is_surface"]:
            suffix += " [S]"
        if row["therapies"]:
            suffix += f" ({row['therapies']})"
        labels.append(f"{row['symbol']}{suffix}")
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Expression (TPM)")
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_title(f"Purity-adjusted tumor expression \u2014 {cancer_code} (purity={purity:.0%})")
    ax1.invert_yaxis()
    ax1.legend(fontsize=8, loc="lower right")

    # Right: TCGA percentile heatmap
    pctiles = selected["tcga_percentile"].values
    ax2.barh(y, pctiles, color=colors, alpha=0.8, height=0.7)
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("TCGA percentile")
    ax2.set_title("vs TCGA cancer types")
    ax2.invert_yaxis()

    # Category legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="CTA (vaccination target)"),
        Patch(facecolor="#3498db", label="Therapy target (in trials)"),
        Patch(facecolor="#2ecc71", label="Surface protein"),
        Patch(facecolor="#95a5a6", label="Other tumor gene"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")

    return fig
