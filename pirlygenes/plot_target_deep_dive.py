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

"""Deep-dive therapy-target and CTA plots.

Three families of visualisation:

1. **Actionable surface targets** — observed TPM + tumor-adjusted TPM
   for a curated panel vs the matched TCGA cancer type, healthy vital
   tissues, and pan-cancer TCGA.
2. **CTA deep dive** — same treatment for cancer-testis antigens.
3. **Tumor vs TME attribution** — per-gene bar showing how much of
   the observed signal comes from tumor vs microenvironment.

All three are generalised: a cancer-type-aware panel function selects
the right genes for any TCGA code.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .gene_sets_cancer import (
    pan_cancer_expression,
    CTA_gene_id_to_name,
)
from .plot_scatter import resolve_cancer_type

# ── Essential tissues for safety context ─────────────────────────────────

VITAL_TISSUES = [
    "heart_muscle", "liver", "kidney", "lung", "cerebral_cortex",
    "colon", "bone_marrow", "pancreas", "stomach",
]

# ── Per-cancer-type actionable surface target panels ─────────────────────
#
# Curated: genes that are current or emerging therapy targets for each
# cancer type.  Includes approved targets, late-phase trial targets, and
# diagnostic biomarkers that are actionable for patient selection.

_CANCER_SURFACE_TARGETS = {
    "PRAD": [
        "FOLH1",    # PSMA — 177Lu-PSMA-617 (Pluvicto), approved
        "KLK3",     # PSA — diagnostic, AR target readout
        "KLK2",     # hK2 — diagnostic, AR target
        "STEAP1",   # STEAP1 — AMG 509 (xaluritamig), phase III
        "STEAP2",   # STEAP2 — emerging ADC/bispecific target
        "PSCA",     # PSCA — CAR-T trials
        "TACSTD2",  # TROP2 — sacituzumab govitecan (Trodelvy)
        "CD276",    # B7-H3 — enoblituzumab, CAR-T
        "ERBB2",    # HER2 — T-DXd (Enhertu), expanding indications
        "CD46",     # CD46 — FOR46 ADC, PRAD trials
        "DLL3",     # DLL3 — tarlatamab (Imdelltra), NEPC
        "GRPR",     # GRP receptor — 177Lu-RM2 PSMA alternative
        "AR",       # Androgen receptor — enzalutamide/abiraterone readout
        "NKX3-1",   # Prostate lineage TF — diagnostic
        "HOXB13",   # Prostate lineage — diagnostic, germline marker
        "TMPRSS2",  # TMPRSS2 — fusion partner, AR target readout
    ],
    "BRCA": [
        "ERBB2",    # HER2 — trastuzumab, T-DXd
        "ESR1",     # ER — endocrine therapy readout
        "PGR",      # PR — endocrine therapy readout
        "TACSTD2",  # TROP2 — sacituzumab govitecan
        "NECTIN4",  # Nectin-4 — enfortumab vedotin (expanding)
        "CD276",    # B7-H3
        "FOLR1",    # FRα — mirvetuximab soravtansine
        "MUC16",    # CA-125 — diagnostic, ADC target
        "MUC1",     # MUC1 — CAR-T trials
        "CEACAM5",  # CEA — tusamitamab ravtansine
    ],
    "LUAD": [
        "EGFR",     # EGFR — osimertinib, amivantamab
        "ERBB2",    # HER2 — T-DXd
        "MET",      # MET — capmatinib, tepotinib
        "TACSTD2",  # TROP2 — datopotamab deruxtecan
        "CEACAM5",  # CEA — tusamitamab ravtansine
        "CD276",    # B7-H3
        "MSLN",     # Mesothelin — CAR-T
        "NECTIN4",  # Nectin-4
        "DLL3",     # DLL3 — SCLC/NEPC, expanding
        "FOLR1",    # FRα
    ],
    "LUSC": [
        "EGFR",
        "NECTIN4",
        "TACSTD2",
        "CD276",
        "FGFR1",   # FGFR — erdafitinib
        "DLL3",
    ],
    "COAD": [
        "CEACAM5",  # CEA
        "ERBB2",    # HER2
        "EGFR",     # EGFR — cetuximab, panitumumab
        "TACSTD2",
        "CD276",
        "GPC3",     # Glypican-3 — trials
        "MET",
        "LGR5",     # Stem cell marker
        "GUCY2C",   # Guanylyl cyclase C — CAR-T trials
    ],
    "SKCM": [
        "CD274",    # PD-L1
        "CTLA4",    # CTLA-4 — ipilimumab
        "PDCD1",    # PD-1
        "MLANA",    # MART-1 — TCR-T / TIL therapy
        "PMEL",     # gp100 — tebentafusp
        "TYR",      # Tyrosinase — TCR-T
        "TACSTD2",
        "CD276",
    ],
    "OV": [
        "FOLR1",    # FRα — mirvetuximab soravtansine (approved)
        "MUC16",    # CA-125 — diagnostic, ADC
        "MSLN",     # Mesothelin
        "TACSTD2",  # TROP2
        "ERBB2",
        "NECTIN4",
        "NaPi2b",   # SLC34A2 — lifastuzumab vedotin
        "CD276",
    ],
    "LIHC": [
        "GPC3",     # Glypican-3 — CAR-T, bispecific trials
        "CD274",    # PD-L1
        "AFP",      # AFP — diagnostic
        "EPCAM",    # EpCAM — catumaxomab
        "TACSTD2",
        "MET",
        "FGFR4",   # FGFR4 — fisogatinib
    ],
    "GBM": [
        "EGFR",     # EGFRvIII — rindopepimut, CAR-T
        "IL13RA2",  # IL-13Rα2 — CAR-T
        "DLL3",     # DLL3 — NEPC/neuroendocrine
        "CD276",    # B7-H3
        "GD2",      # GD2 — dinutuximab (repurposing)
        "PDGFRA",
    ],
    "BLCA": [
        "NECTIN4",  # Nectin-4 — enfortumab vedotin (approved)
        "TACSTD2",  # TROP2 — sacituzumab govitecan
        "ERBB2",    # HER2
        "FGFR3",   # FGFR3 — erdafitinib (approved)
        "CD274",    # PD-L1
        "CD276",
    ],
}

# Default fallback for cancer types without a curated panel
_DEFAULT_SURFACE_TARGETS = [
    "TACSTD2", "CD276", "ERBB2", "NECTIN4", "CEACAM5",
    "MSLN", "FOLR1", "EPCAM", "MUC1", "CD274",
]


def actionable_surface_targets(cancer_type):
    """Return the curated actionable surface target panel for a cancer type."""
    code = resolve_cancer_type(cancer_type)
    return list(_CANCER_SURFACE_TARGETS.get(code, _DEFAULT_SURFACE_TARGETS))


# ── Reference data helpers ───────────────────────────────────────────────


def _build_reference_context(symbols, cancer_code):
    """Build a reference context dict for a list of gene symbols.

    Returns {symbol: {cancer_fpkm, origin_tissue_ntpm,
    vital_tissues: {tissue: ntpm}, all_cancer_median, all_cancer_max}}.
    """
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    fpkm_cols = [c for c in ref.columns if c.startswith("FPKM_")]
    from .tumor_purity import CANCER_TO_TISSUE
    origin_tissue = CANCER_TO_TISSUE.get(cancer_code)

    result = {}
    for sym in symbols:
        if sym not in ref.index:
            continue
        row = ref.loc[sym]
        cancer_col = f"FPKM_{cancer_code}"
        cancer_fpkm = float(row[cancer_col]) if cancer_col in ref.columns else 0.0

        origin_ntpm = 0.0
        if origin_tissue:
            origin_col = f"nTPM_{origin_tissue}"
            if origin_col in ref.columns:
                origin_ntpm = float(row[origin_col])

        vital = {}
        for tissue in VITAL_TISSUES:
            col = f"nTPM_{tissue}"
            if col in ref.columns:
                vital[tissue] = float(row[col])

        all_cancer_vals = [float(row[c]) for c in fpkm_cols if c in ref.columns]
        result[sym] = {
            "cancer_fpkm": cancer_fpkm,
            "origin_tissue_ntpm": origin_ntpm,
            "vital_tissues": vital,
            "all_cancer_median": float(np.median(all_cancer_vals)) if all_cancer_vals else 0.0,
            "all_cancer_max": float(np.max(all_cancer_vals)) if all_cancer_vals else 0.0,
        }
    return result


def _get_sample_tpm_by_symbol(df_gene_expr):
    """Return {symbol: tpm} for the sample, preferring direct symbol column."""
    from .sample_context import _build_tpm_by_symbol
    return _build_tpm_by_symbol(df_gene_expr)


def _estimate_tumor_tpm(observed, tme_ref, purity):
    """Single-point tumor-adjusted TPM."""
    p = max(purity, 0.01)
    return max(0.0, (observed - (1.0 - p) * tme_ref) / p)


def _get_tme_reference(symbols, cancer_code):
    """Return {symbol: tme_ntpm} — mean expression across TME tissues."""
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    from .plot_tumor_expr import _TME_TISSUES
    tme_cols = [f"nTPM_{t}" for t in _TME_TISSUES if f"nTPM_{t}" in ref.columns]
    result = {}
    for sym in symbols:
        if sym in ref.index and tme_cols:
            result[sym] = float(ref.loc[sym, tme_cols].astype(float).mean())
        else:
            result[sym] = 0.0
    return result


# ── Plot 1: Actionable surface targets deep dive ────────────────────────


def plot_actionable_targets(
    df_gene_expr,
    cancer_type,
    purity_estimate=None,
    custom_genes=None,
    save_to_filename=None,
    save_dpi=300,
    title=None,
):
    """Dot plot: observed TPM + tumor-adjusted TPM for actionable targets
    vs TCGA cancer-type median, healthy tissues, and pan-cancer context.

    Each gene gets one row. Columns show:
    - Sample observed TPM (black dot)
    - Tumor-adjusted TPM (red dot, if purity available)
    - TCGA cancer-type median (blue bar)
    - Max vital-tissue expression (gray bar — safety signal)

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression data with gene ID / symbol + TPM.
    cancer_type : str
        TCGA code or alias.
    purity_estimate : float or None
        Overall purity estimate (0–1). If None, tumor-adjusted not shown.
    custom_genes : list[str] or None
        Override the default curated panel.
    """
    cancer_code = resolve_cancer_type(cancer_type)
    genes = custom_genes or actionable_surface_targets(cancer_code)

    sample_tpm = _get_sample_tpm_by_symbol(df_gene_expr)
    ref_ctx = _build_reference_context(genes, cancer_code)
    tme_ref = _get_tme_reference(genes, cancer_code)

    # Filter to genes present in reference
    genes = [g for g in genes if g in ref_ctx]
    if not genes:
        return None

    # Build data
    rows = []
    for sym in genes:
        ctx = ref_ctx[sym]
        obs = sample_tpm.get(sym, 0.0)
        tme = tme_ref.get(sym, 0.0)
        tumor_adj = _estimate_tumor_tpm(obs, tme, purity_estimate) if purity_estimate else None
        max_vital = max(ctx["vital_tissues"].values()) if ctx["vital_tissues"] else 0.0
        rows.append({
            "symbol": sym,
            "observed": obs,
            "tumor_adjusted": tumor_adj,
            "cancer_median": ctx["cancer_fpkm"],
            "max_vital_tissue": max_vital,
            "origin_tissue": ctx["origin_tissue_ntpm"],
            "tme_background": tme,
        })

    # Sort by observed TPM descending
    rows.sort(key=lambda r: r["observed"], reverse=True)
    n = len(rows)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * n)))
    y_pos = np.arange(n)
    symbols = [r["symbol"] for r in rows]

    # Background bars
    ax.barh(y_pos, [r["cancer_median"] for r in rows],
            height=0.35, color="#4A90D9", alpha=0.3, label=f"TCGA {cancer_code} median")
    ax.barh(y_pos + 0.35, [r["max_vital_tissue"] for r in rows],
            height=0.25, color="#999999", alpha=0.3, label="Max vital tissue (safety)")

    # Sample dots
    ax.scatter([r["observed"] for r in rows], y_pos,
               color="black", s=60, zorder=5, label="Sample observed TPM")

    if purity_estimate:
        tumor_vals = [r["tumor_adjusted"] for r in rows]
        ax.scatter(tumor_vals, y_pos,
                   color="#E74C3C", s=60, marker="D", zorder=5,
                   label=f"Tumor-adjusted (purity={purity_estimate:.0%})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel("TPM (log scale)")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or f"Actionable Surface Targets — {cancer_code}")
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


# ── Plot 2: Tumor vs TME attribution ────────────────────────────────────


def plot_tumor_attribution(
    df_gene_expr,
    cancer_type,
    purity_estimate,
    custom_genes=None,
    category="surface",
    top_n=20,
    save_to_filename=None,
    save_dpi=300,
):
    """Stacked horizontal bar: for each gene, show how much of the
    observed TPM is attributable to tumor vs TME background.

    tumor_component = purity × tumor_adjusted
    tme_component = (1 - purity) × tme_reference
    """
    cancer_code = resolve_cancer_type(cancer_type)

    if category == "CTA":
        cta_map = CTA_gene_id_to_name()
        genes = custom_genes or sorted(cta_map.values())
    else:
        genes = custom_genes or actionable_surface_targets(cancer_code)

    sample_tpm = _get_sample_tpm_by_symbol(df_gene_expr)
    tme_ref = _get_tme_reference(genes, cancer_code)
    ref_ctx = _build_reference_context(genes, cancer_code)

    genes = [g for g in genes if g in ref_ctx]
    if not genes:
        return None

    purity = max(purity_estimate, 0.01)
    rows = []
    for sym in genes:
        obs = sample_tpm.get(sym, 0.0)
        tme = tme_ref.get(sym, 0.0)
        tumor_adj = _estimate_tumor_tpm(obs, tme, purity)
        tumor_component = purity * tumor_adj
        tme_component = (1.0 - purity) * tme
        rows.append({
            "symbol": sym,
            "observed": obs,
            "tumor_component": tumor_component,
            "tme_component": tme_component,
            "tumor_adjusted": tumor_adj,
            "pct_tumor": tumor_component / max(obs, 0.01),
        })

    # Sort by observed, take top N
    rows.sort(key=lambda r: r["observed"], reverse=True)
    rows = [r for r in rows if r["observed"] > 0.1][:top_n]
    if not rows:
        return None

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * n)))
    y_pos = np.arange(n)
    symbols = [r["symbol"] for r in rows]

    tumor_vals = [r["tumor_component"] for r in rows]
    tme_vals = [r["tme_component"] for r in rows]

    ax.barh(y_pos, tumor_vals, color="#E74C3C", alpha=0.8, label="Tumor")
    ax.barh(y_pos, tme_vals, left=tumor_vals, color="#4A90D9", alpha=0.5, label="TME background")

    # Annotate with percentage
    for i, r in enumerate(rows):
        pct = r["pct_tumor"]
        if r["observed"] > 0.5:
            ax.text(r["observed"] + 0.5, i, f"{pct:.0%} tumor",
                    va="center", fontsize=7, color="#555555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel("TPM")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    cat_label = "CTAs" if category == "CTA" else "Actionable Targets"
    ax.set_title(f"Tumor vs TME Attribution — {cat_label} — {cancer_code} (purity={purity:.0%})")
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig


# ── Plot 3: CTA deep dive ───────────────────────────────────────────────


def plot_cta_deep_dive(
    df_gene_expr,
    cancer_type,
    purity_estimate=None,
    top_n=20,
    save_to_filename=None,
    save_dpi=300,
):
    """Same layout as plot_actionable_targets but for CTAs.

    CTAs are cancer-testis antigens — normally restricted to testis/placenta,
    so any expression in somatic tissue is tumor-specific. The safety context
    (vital tissues) is especially important here because true CTAs should
    have near-zero expression in all vital tissues.
    """
    cancer_code = resolve_cancer_type(cancer_type)
    cta_map = CTA_gene_id_to_name()
    cta_symbols = sorted(cta_map.values())

    sample_tpm = _get_sample_tpm_by_symbol(df_gene_expr)
    ref_ctx = _build_reference_context(cta_symbols, cancer_code)
    tme_ref = _get_tme_reference(cta_symbols, cancer_code)

    cta_symbols = [g for g in cta_symbols if g in ref_ctx]

    rows = []
    for sym in cta_symbols:
        ctx = ref_ctx[sym]
        obs = sample_tpm.get(sym, 0.0)
        tme = tme_ref.get(sym, 0.0)
        tumor_adj = _estimate_tumor_tpm(obs, tme, purity_estimate) if purity_estimate else None
        max_vital = max(ctx["vital_tissues"].values()) if ctx["vital_tissues"] else 0.0
        rows.append({
            "symbol": sym,
            "observed": obs,
            "tumor_adjusted": tumor_adj,
            "cancer_median": ctx["cancer_fpkm"],
            "max_vital_tissue": max_vital,
            "tme_background": tme,
        })

    rows.sort(key=lambda r: r["observed"], reverse=True)
    rows = rows[:top_n]
    if not rows:
        return None

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * n)))
    y_pos = np.arange(n)
    symbols = [r["symbol"] for r in rows]

    ax.barh(y_pos, [r["cancer_median"] for r in rows],
            height=0.35, color="#27AE60", alpha=0.3, label=f"TCGA {cancer_code} median")
    ax.barh(y_pos + 0.35, [r["max_vital_tissue"] for r in rows],
            height=0.25, color="#999999", alpha=0.3, label="Max vital tissue (safety)")

    ax.scatter([r["observed"] for r in rows], y_pos,
               color="black", s=60, zorder=5, label="Sample observed TPM")
    if purity_estimate:
        tumor_vals = [r["tumor_adjusted"] for r in rows]
        ax.scatter(tumor_vals, y_pos,
                   color="#E74C3C", s=60, marker="D", zorder=5,
                   label=f"Tumor-adjusted (purity={purity_estimate:.0%})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel("TPM (log scale)")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Cancer-Testis Antigens — {cancer_code}")
    fig.tight_layout()

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    return fig
