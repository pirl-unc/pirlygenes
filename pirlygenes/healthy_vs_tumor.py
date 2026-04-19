# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Stage-0 tissue-composition + cancer-hint gate (#149, refined).

Runs BEFORE cancer-type classification. Produces the coarsest
possible reading of "what kind of tissue is this, and is there a
hint of cancer?" and propagates the result forward so later stages
can refine. Explicitly does NOT make a binary healthy-vs-cancer call
on the first pass — healthy and low-purity-tumor look similar here
and will be distinguished downstream once lineage markers, purity,
and therapy-axis signals have been read.

Output:

- ``top_normal_tissues``: the three HPA normal-tissue columns that
  best correlate with the sample's log-TPM profile, each with a
  Spearman rho.
- ``top_tcga_cohorts``: the three TCGA cohorts that best correlate.
- ``proliferation_log2_mean``: geomean on log-TPM of the five-gene
  proliferation panel (MKI67 / TOP2A / CCNB1 / BIRC5 / AURKA).
- ``cancer_hint``: one of ``"tumor-consistent"``, ``"possibly-tumor"``,
  ``"healthy-dominant"`` — a coarse read, not a commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .gene_sets_cancer import pan_cancer_expression


_MIN_REFERENCE_GENES = 2000

# Coordinate cell-cycle / mitosis panel. Elevated as a group only in
# proliferating (malignant or regenerating) tissue; individual members
# can be up in normal tissue (MKI67 in germinal-center spleen, colon
# crypts, bone marrow) without the full panel firing.
_PROLIFERATION_PANEL = ("MKI67", "TOP2A", "CCNB1", "BIRC5", "AURKA")
_PROLIFERATION_HIGH_LOG2 = 4.5   # panel mean > this → "tumor-consistent"
_PROLIFERATION_QUIET_LOG2 = 3.5  # panel mean < this → proliferation quiet

# The margin by which the top HPA tissue must beat the top TCGA cohort
# correlation before we nudge toward "healthy-dominant". In combination
# with the proliferation panel.
_HPA_MARGIN_STRONG = 0.05
_HPA_MARGIN_WEAK = 0.02

# Lymphoid-tissue normals whose expression profile is structurally
# indistinguishable from lymphoid malignancy by bulk RNA — the TCGA
# DLBC reference is itself >90% lymph-node tissue + the malignant
# clone, so correlation alone can't separate them. When the top HPA
# match falls in this set AND the top TCGA match is a heme-lymphoid
# cancer, we always flag as "possibly-tumor" regardless of
# proliferation (germinal centers + bone marrow are normally
# cycling) and regardless of correlation margin.
_LYMPHOID_NORMAL_TISSUES = frozenset({
    "nTPM_lymph_node", "nTPM_spleen", "nTPM_thymus",
    "nTPM_bone_marrow", "nTPM_appendix", "nTPM_tonsil",
})
_HEME_LYMPHOID_TCGA_COHORTS = frozenset({
    "FPKM_DLBC", "FPKM_LAML", "FPKM_THYM",
})


@dataclass
class TissueCompositionSignal:
    """Stage-0 coarse reading of tissue composition + cancer hint.

    ``cancer_hint`` is one of:

    - ``"tumor-consistent"`` — proliferation panel is clearly active
      OR TCGA cohort correlation exceeds HPA tissue correlation by
      more than the margin (typical primary tumor).
    - ``"possibly-tumor"`` — proliferation panel is not loud AND HPA
      correlation is close to TCGA; could be tumor or normal.
    - ``"healthy-dominant"`` — proliferation panel is quiet AND HPA
      correlation clearly exceeds TCGA.

    This is a coarse hint designed to be refined downstream, not a
    final classification.
    """

    top_normal_tissues: list[tuple[str, float]] = field(default_factory=list)
    top_tcga_cohorts: list[tuple[str, float]] = field(default_factory=list)
    proliferation_log2_mean: float = 0.0
    proliferation_genes_observed: int = 0
    cancer_hint: str = "tumor-consistent"
    n_reference_genes: int = 0
    verdict: str = ""
    # True when the top HPA / top TCGA pair is structurally
    # indistinguishable by bulk RNA (lymphoid normal vs heme-lymphoid
    # cancer). Downstream tumor-evidence gates should NOT suppress
    # the Stage-0 banner in this case — the "purity" estimate on a
    # normal lymphoid sample is spurious anchor.
    structural_ambiguity: bool = False

    def summary_line(self) -> str:
        """One-sentence description suitable for a brief or summary.md."""
        if not self.top_normal_tissues:
            return "Tissue-composition signal unavailable (reference overlap too small)."

        def _fmt_tissue(name, rho):
            return f"{name.replace('nTPM_', '').replace('_', ' ')} (ρ={rho:.2f})"

        def _fmt_cancer(name, rho):
            return f"{name.replace('FPKM_', '')} (ρ={rho:.2f})"

        tissues = ", ".join(_fmt_tissue(n, r) for n, r in self.top_normal_tissues[:3])
        cohorts = ", ".join(_fmt_cancer(n, r) for n, r in self.top_tcga_cohorts[:3])
        return (
            f"Top normal-tissue matches: {tissues}. "
            f"Top TCGA cohorts: {cohorts}. "
            f"Proliferation panel {self.proliferation_log2_mean:.1f} log2-TPM "
            f"(of {self.proliferation_genes_observed}/{len(_PROLIFERATION_PANEL)} "
            f"genes observed). Hint: **{self.cancer_hint}**."
        )

    def brief_banner(
        self,
        purity: float | None = None,
        signature_score: float | None = None,
    ) -> str | None:
        """Terse banner for the brief — shown for every non-tumor-consistent hint.

        Propagates the Stage-0 coarse reading forward so a clinician
        reading the brief sees the tissue context and the cancer-hint
        confidence up front rather than just the downstream TCGA label.

        When ``purity`` (Stage 2) and/or ``signature_score`` (Stage 1)
        are supplied, corroborating tumor evidence can suppress the
        banner: we don't want to shout "composition ambiguous" on a
        sample that has a solid lineage-matched TCGA cohort at moderate
        purity. The healthy-dominant banner has a higher bar to
        suppress than the possibly-tumor banner — it takes strong
        evidence (purity ≥ 0.5 AND signature ≥ 0.75) to override a
        confident Stage-0 healthy call.
        """
        if self.cancer_hint == "tumor-consistent":
            return None
        if not self.top_normal_tissues:
            return None

        if (purity is not None or signature_score is not None) \
                and not self.structural_ambiguity:
            # Structural-ambiguity cases (lymphoid normal vs heme-
            # lymphoid cancer) always keep the banner — the purity
            # and signature estimates are spurious in that regime
            # because the reference itself is lymphoid tissue.
            p = float(purity) if purity is not None else 0.0
            s = float(signature_score) if signature_score is not None else 0.0
            if self.cancer_hint == "healthy-dominant":
                # Require strong corroborating tumor evidence to
                # override a confident healthy signal.
                if p >= 0.50 and s >= 0.75:
                    return None
            elif self.cancer_hint == "possibly-tumor":
                # Modest corroborating evidence suffices to suppress
                # the soft ambiguity banner — the downstream cancer
                # call is already well-supported.
                if p >= 0.30 or s >= 0.75:
                    return None

        tissue, rho = self.top_normal_tissues[0]
        tissue_name = tissue.replace("nTPM_", "").replace("_", " ")
        cohort = self.top_tcga_cohorts[0][0].replace("FPKM_", "") if self.top_tcga_cohorts else "—"
        cohort_rho = self.top_tcga_cohorts[0][1] if self.top_tcga_cohorts else 0.0
        if self.cancer_hint == "healthy-dominant":
            return (
                f"**⚠ Stage-0 hint: healthy tissue dominant.** Sample profile "
                f"correlates with normal **{tissue_name}** (ρ={rho:.2f}) more "
                f"than any TCGA cohort (best {cohort} ρ={cohort_rho:.2f}); "
                f"proliferation panel quiet "
                f"({self.proliferation_log2_mean:.1f} log2-TPM). Downstream "
                f"cancer-type call is soft-confidence."
            )
        # possibly-tumor
        if self.structural_ambiguity:
            return (
                f"**Stage-0 hint: structural ambiguity (lymphoid).** Top normal "
                f"match is **{tissue_name}** (ρ={rho:.2f}); top TCGA match is "
                f"{cohort} (ρ={cohort_rho:.2f}). Normal lymphoid tissue and "
                f"lymphoid malignancy are indistinguishable by bulk-RNA "
                f"correlation — treat the downstream cancer call as soft-"
                f"confidence regardless of purity (the purity estimate itself "
                f"is unreliable in this regime)."
            )
        return (
            f"**Stage-0 hint: composition ambiguous.** Top normal-tissue match "
            f"is **{tissue_name}** (ρ={rho:.2f}); top TCGA match is {cohort} "
            f"(ρ={cohort_rho:.2f}). Could be normal tissue or a low-purity "
            f"tumor — the downstream cancer call is soft-confidence pending "
            f"lineage / purity / therapy-axis evidence."
        )


# Back-compat alias: old name some callers used.
HealthyVsTumorResult = TissueCompositionSignal


def _extract_sample_tpm_by_symbol(df_expr: pd.DataFrame) -> dict[str, float]:
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


def assess_tissue_composition(df_expr: pd.DataFrame) -> TissueCompositionSignal:
    """Race the sample against HPA-normal-tissue and TCGA-cohort columns.

    Returns the top-3 matches on each side + proliferation-panel
    geomean + a coarse cancer-hint call. Intended as the first stage
    in a coarse-to-fine pipeline: it gives downstream stages a
    tissue-composition prior without committing to healthy / cancer.
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
        return TissueCompositionSignal(
            proliferation_log2_mean=prolif_log2,
            proliferation_genes_observed=n_prolif_obs,
            cancer_hint="tumor-consistent",
            n_reference_genes=n_overlap,
            verdict=(
                f"Insufficient reference overlap ({n_overlap} < "
                f"{_MIN_REFERENCE_GENES} genes); tissue-composition signal "
                f"unavailable — defer to downstream cancer-type inference."
            ),
        )

    sample_log = np.log2(np.array(
        [sample_by_symbol[s] + 1.0 for s in shared_symbols]
    ))

    def _top_matches(cols: list[str], k: int = 3) -> list[tuple[str, float]]:
        scored = []
        for col in cols:
            ref_vals = ref.loc[shared_symbols, col].astype(float).to_numpy()
            if not np.isfinite(ref_vals).any() or float(np.nanmax(ref_vals)) <= 0:
                continue
            ref_log = np.log2(np.nan_to_num(ref_vals) + 1.0)
            rho = float(_spearman_rho(sample_log, ref_log))
            scored.append((col, rho))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    top_normal = _top_matches(hpa_cols, k=3)
    top_tcga = _top_matches(tcga_cols, k=3)

    best_normal_rho = top_normal[0][1] if top_normal else 0.0
    best_tcga_rho = top_tcga[0][1] if top_tcga else 0.0
    margin = best_normal_rho - best_tcga_rho

    # Structural-ambiguity override: when the top HPA match is a
    # lymphoid normal AND the top TCGA match is a heme-lymphoid
    # cancer, bulk-RNA correlation cannot distinguish them. Always
    # flag as possibly-tumor so the downstream reader sees the
    # ambiguity rather than a false tumor-consistent call.
    top_hpa_name = top_normal[0][0] if top_normal else ""
    top_tcga_name = top_tcga[0][0] if top_tcga else ""
    lymphoid_ambiguity = (
        top_hpa_name in _LYMPHOID_NORMAL_TISSUES
        and top_tcga_name in _HEME_LYMPHOID_TCGA_COHORTS
    )

    structural_ambiguity = False
    if lymphoid_ambiguity:
        cancer_hint = "possibly-tumor"
        structural_ambiguity = True
    elif prolif_log2 >= _PROLIFERATION_HIGH_LOG2:
        cancer_hint = "tumor-consistent"
    elif prolif_log2 < _PROLIFERATION_QUIET_LOG2 and margin >= _HPA_MARGIN_STRONG:
        cancer_hint = "healthy-dominant"
    elif margin >= _HPA_MARGIN_WEAK:
        cancer_hint = "possibly-tumor"
    else:
        cancer_hint = "tumor-consistent"

    if cancer_hint == "healthy-dominant":
        verdict = (
            f"Healthy-dominant: best HPA "
            f"{top_normal[0][0].replace('nTPM_', '')} ρ={best_normal_rho:.3f} "
            f"beats best TCGA {top_tcga[0][0].replace('FPKM_', '') if top_tcga else '-'} "
            f"ρ={best_tcga_rho:.3f} by {margin:+.3f}; proliferation "
            f"{prolif_log2:.1f} log2-TPM quiet."
        )
    elif cancer_hint == "possibly-tumor":
        verdict = (
            f"Composition ambiguous: HPA ρ={best_normal_rho:.3f} vs "
            f"TCGA ρ={best_tcga_rho:.3f} (margin {margin:+.3f}); "
            f"proliferation {prolif_log2:.1f} log2-TPM moderate."
        )
    else:
        verdict = (
            f"Tumor-consistent: proliferation panel {prolif_log2:.1f} log2-TPM "
            f"or TCGA ρ={best_tcga_rho:.3f} dominant vs HPA ρ={best_normal_rho:.3f}."
        )

    return TissueCompositionSignal(
        top_normal_tissues=top_normal,
        top_tcga_cohorts=top_tcga,
        proliferation_log2_mean=prolif_log2,
        proliferation_genes_observed=n_prolif_obs,
        cancer_hint=cancer_hint,
        n_reference_genes=n_overlap,
        verdict=verdict,
        structural_ambiguity=structural_ambiguity,
    )


# Back-compat alias: callers wrote ``assess_healthy_vs_tumor`` before
# the rename. Keep the old entry point working.
def assess_healthy_vs_tumor(
    df_expr: pd.DataFrame, *_args, **_kwargs,
) -> TissueCompositionSignal:
    return assess_tissue_composition(df_expr)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
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
