# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stage-0 healthy-vs-tumor gate (#149).

Runs BEFORE cancer-type classification. Races the sample's expression
profile against every HPA normal tissue (50 columns) and every TCGA
cohort (33 columns) already shipped in :func:`pan_cancer_expression`.
If the best-matching HPA tissue correlates more strongly than the
best-matching TCGA cohort by a comfortable margin, and there is no
corroborating proliferation signal (MKI67), the sample is flagged as
likely healthy rather than force-classified into a TCGA code.

This is the minimum-viable gate that fixes the GTEx-normals-called-as-
DLBC/SARC/GBM failure without requiring a new classifier or new data.
The HPA nTPM reference is already shipped; the TCGA FPKM reference is
already shipped; the correlation pass is a few hundred milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .gene_sets_cancer import pan_cancer_expression


# Sample needs at least this many genes overlapping the reference to
# produce a meaningful correlation; falls below the noise floor
# otherwise. Empirically ~5K genes is enough; set conservatively.
_MIN_REFERENCE_GENES = 2000

# Margin by which the best HPA tissue must beat the best TCGA cohort
# before we even consider a healthy call. Too tight: false positives
# on cold-immune cancers or low-purity tumors where the sample's
# stroma-and-normal fraction dominates.  Too loose: misses the GTEx-
# style mis-classifications this gate exists to catch.
_HEALTHY_MARGIN_CONFIDENT = 0.05
_HEALTHY_MARGIN_AMBIGUOUS = 0.03

# Proliferation-panel veto. Any malignancy coordinately upregulates
# multiple cell-cycle / mitotic-checkpoint genes; healthy tissue
# does not, even when one member (MKI67 in germinal-center spleen,
# colon crypts) is elevated in isolation. We require the geomean
# of the panel on log-TPM to be below ``_PROLIFERATION_VETO_LOG2``.
# Empirically ~3.5 log2-TPM separates healthy solid tissues from
# primary solid tumors; heme-normal vs heme-cancer remains
# ambiguous because normal lymphoid tissue has cycling B-cells.
_PROLIFERATION_PANEL = ("MKI67", "TOP2A", "CCNB1", "BIRC5", "AURKA")
_PROLIFERATION_VETO_LOG2 = 3.5


@dataclass
class HealthyVsTumorResult:
    """Outcome of the Stage-0 gate.

    ``call`` takes one of three values:

    - ``"healthy"`` — confident healthy-tissue match: HPA clearly
      beats TCGA AND proliferation panel is quiet.
    - ``"ambiguous"`` — HPA beats TCGA by a modest margin or
      proliferation panel is modest; the sample could be a low-purity
      tumor or normal tissue.  Downstream must treat the cancer call
      as soft-confidence.
    - ``"tumor-consistent"`` — no evidence against cancer (default).
    """

    call: str
    best_hpa_tissue: str
    hpa_correlation: float
    best_tcga_code: str
    tcga_correlation: float
    margin: float
    proliferation_log2_mean: float
    proliferation_genes_observed: int
    verdict: str
    n_reference_genes: int

    @property
    def likely_healthy(self) -> bool:
        """Back-compat: True for confident healthy calls only."""
        return self.call == "healthy"

    def brief_banner(self) -> str | None:
        """Short one-line banner for the brief when healthy/ambiguous; ``None`` otherwise."""
        if self.call == "tumor-consistent":
            return None
        tissue = self.best_hpa_tissue.replace("nTPM_", "").replace("_", " ")
        cohort = self.best_tcga_code.replace("FPKM_", "")
        if self.call == "healthy":
            return (
                f"**⚠ Sample may not be cancer.** Expression profile matches "
                f"normal **{tissue}** (ρ={self.hpa_correlation:.2f}) more than "
                f"any TCGA cohort (best {cohort} ρ={self.tcga_correlation:.2f}); "
                f"proliferation panel is quiet ({self.proliferation_log2_mean:.1f} "
                f"log2-TPM mean across "
                f"{self.proliferation_genes_observed}/{len(_PROLIFERATION_PANEL)} "
                f"cell-cycle genes)."
            )
        # ambiguous
        return (
            f"**Ambiguous healthy-vs-tumor signal.** Expression correlates "
            f"slightly more with normal **{tissue}** (ρ={self.hpa_correlation:.2f}) "
            f"than TCGA {cohort} (ρ={self.tcga_correlation:.2f}) — sample may be "
            f"normal tissue or a low-purity tumor. Treat the cancer call as "
            f"soft-confidence pending orthogonal evidence."
        )


def _extract_sample_tpm_by_symbol(df_expr: pd.DataFrame) -> dict[str, float]:
    """Extract {symbol: TPM} from a pirlygenes expression frame.

    Accepts the same schema the rest of the pipeline uses: a TPM
    column (case-insensitive 'TPM' or 'tpm') plus either a gene_id
    (Ensembl) or gene_name column.
    """
    tpm_col = next(
        (c for c in df_expr.columns if c.upper() == "TPM"),
        None,
    )
    if tpm_col is None:
        return {}
    gene_id_col = next(
        (c for c in df_expr.columns if c.lower() in
         ("gene_id", "ensembl_gene_id", "canonical_gene_id", "geneid")),
        None,
    )
    gene_name_col = next(
        (c for c in df_expr.columns if c.lower() in
         ("gene_name", "canonical_gene_name", "gene_symbol", "symbol", "hugo_symbol")),
        None,
    )
    ref = pan_cancer_expression()[["Ensembl_Gene_ID", "Symbol"]].drop_duplicates(
        subset="Ensembl_Gene_ID"
    )
    eid_to_symbol = dict(zip(ref["Ensembl_Gene_ID"], ref["Symbol"]))

    symbols = None
    if gene_id_col:
        bare_ids = df_expr[gene_id_col].astype(str).str.split(".", n=1).str[0]
        symbols = [eid_to_symbol.get(eid, "") for eid in bare_ids]
    if symbols is None or not any(symbols):
        if gene_name_col:
            symbols = df_expr[gene_name_col].astype(str).tolist()
    if symbols is None:
        return {}

    tpm_vals = df_expr[tpm_col].astype(float).tolist()
    out: dict[str, float] = {}
    for sym, tpm in zip(symbols, tpm_vals):
        if not sym or str(sym).strip() == "":
            continue
        out[str(sym)] = out.get(str(sym), 0.0) + float(tpm)
    return out


def assess_healthy_vs_tumor(
    df_expr: pd.DataFrame,
    confident_margin: float = _HEALTHY_MARGIN_CONFIDENT,
    ambiguous_margin: float = _HEALTHY_MARGIN_AMBIGUOUS,
    proliferation_veto_log2: float = _PROLIFERATION_VETO_LOG2,
) -> HealthyVsTumorResult:
    """Race the sample against HPA-normal-tissue vs TCGA-cohort references.

    Returns a :class:`HealthyVsTumorResult`.  The sample is classified:

    - ``"healthy"`` when HPA correlation exceeds TCGA by
      ``confident_margin`` AND the proliferation-panel geomean is
      below ``proliferation_veto_log2``.
    - ``"ambiguous"`` when either the correlation margin is between
      ``ambiguous_margin`` and ``confident_margin`` OR the panel is
      borderline — the sample may be a low-purity tumor or normal.
    - ``"tumor-consistent"`` otherwise.
    """
    sample_by_symbol = _extract_sample_tpm_by_symbol(df_expr)
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")

    hpa_cols = [c for c in ref.columns if c.startswith("nTPM_")]
    tcga_cols = [c for c in ref.columns if c.startswith("FPKM_")]

    shared_symbols = [s for s in sample_by_symbol if s in ref.index]
    n_overlap = len(shared_symbols)

    prolif_tpms = [
        sample_by_symbol.get(g, 0.0) for g in _PROLIFERATION_PANEL
        if g in sample_by_symbol
    ]
    if prolif_tpms:
        prolif_log2 = float(np.mean(np.log2(np.array(prolif_tpms) + 1.0)))
    else:
        prolif_log2 = 0.0
    n_prolif_obs = len(prolif_tpms)

    if n_overlap < _MIN_REFERENCE_GENES:
        return HealthyVsTumorResult(
            call="tumor-consistent",
            best_hpa_tissue="",
            hpa_correlation=0.0,
            best_tcga_code="",
            tcga_correlation=0.0,
            margin=0.0,
            proliferation_log2_mean=prolif_log2,
            proliferation_genes_observed=n_prolif_obs,
            verdict=(
                f"Insufficient reference overlap ({n_overlap} < "
                f"{_MIN_REFERENCE_GENES} genes); cannot race healthy vs tumor."
            ),
            n_reference_genes=n_overlap,
        )

    sample_log = np.log2(np.array(
        [sample_by_symbol[s] + 1.0 for s in shared_symbols]
    ))

    def _best_match(cols: list[str]) -> tuple[str, float]:
        best_col, best_rho = "", -1.0
        for col in cols:
            ref_vals = ref.loc[shared_symbols, col].astype(float).to_numpy()
            if not np.isfinite(ref_vals).any() or float(np.nanmax(ref_vals)) <= 0:
                continue
            ref_log = np.log2(np.nan_to_num(ref_vals) + 1.0)
            rho = float(_spearman_rho(sample_log, ref_log))
            if rho > best_rho:
                best_rho, best_col = rho, col
        return best_col, best_rho

    best_hpa, hpa_rho = _best_match(hpa_cols)
    best_tcga, tcga_rho = _best_match(tcga_cols)
    margin = hpa_rho - tcga_rho

    prolif_quiet = prolif_log2 < proliferation_veto_log2
    prolif_borderline = prolif_log2 < (proliferation_veto_log2 + 1.0)

    if margin >= confident_margin and prolif_quiet:
        call = "healthy"
    elif margin >= ambiguous_margin and prolif_borderline:
        call = "ambiguous"
    else:
        call = "tumor-consistent"

    if call == "healthy":
        verdict = (
            f"Confident healthy-tissue call: HPA "
            f"{best_hpa.replace('nTPM_', '')} ρ={hpa_rho:.3f} vs TCGA "
            f"{best_tcga.replace('FPKM_', '')} ρ={tcga_rho:.3f} "
            f"(margin {margin:+.3f}); proliferation panel mean "
            f"{prolif_log2:.1f} log2-TPM below veto {proliferation_veto_log2:.1f}."
        )
    elif call == "ambiguous":
        verdict = (
            f"Ambiguous healthy-vs-tumor: HPA ρ={hpa_rho:.3f}, TCGA "
            f"ρ={tcga_rho:.3f} (margin {margin:+.3f}); proliferation panel "
            f"{prolif_log2:.1f} log2-TPM. Could be normal tissue or a "
            f"low-purity tumor — treat the cancer call as soft."
        )
    else:
        verdict = (
            f"Tumor-consistent: HPA ρ={hpa_rho:.3f} does not clearly beat "
            f"TCGA ρ={tcga_rho:.3f}; proliferation panel "
            f"{prolif_log2:.1f} log2-TPM."
        )

    return HealthyVsTumorResult(
        call=call,
        best_hpa_tissue=best_hpa,
        hpa_correlation=hpa_rho,
        best_tcga_code=best_tcga,
        tcga_correlation=tcga_rho,
        margin=margin,
        proliferation_log2_mean=prolif_log2,
        proliferation_genes_observed=n_prolif_obs,
        verdict=verdict,
        n_reference_genes=n_overlap,
    )


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation on two equal-length vectors.

    Uses scipy if available; falls back to a Pearson-on-ranks
    implementation that has no scipy dependency.
    """
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y, nan_policy="omit").statistic)
    except Exception:
        return _pearson_on_ranks(x, y)


def _pearson_on_ranks(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    rx = _rank(x[mask])
    ry = _rank(y[mask])
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(v) + 1)
    return ranks
