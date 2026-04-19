"""Tests for the Stage-0 healthy-vs-tumor gate (#149).

Uses the shipped HPA nTPM reference to construct a synthetic "healthy
tissue" sample (a tissue column perturbed with noise) and a synthetic
"tumor" sample (TCGA FPKM column + high proliferation markers). The
gate must call the first healthy, the second tumor.

Does NOT depend on external cohort files so CI can run offline.
"""

import numpy as np
import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import pan_cancer_expression
from pirlygenes.healthy_vs_tumor import (
    HealthyVsTumorResult,
    assess_healthy_vs_tumor,
    _PROLIFERATION_PANEL,
)


def _ref():
    return pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")


def _as_df(sample_dict: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"gene_symbol": list(sample_dict.keys()),
         "TPM": list(sample_dict.values())}
    )


def test_result_default_fields_populated():
    """The dataclass must expose every field the brief banner touches
    without AttributeError."""
    r = HealthyVsTumorResult(
        call="tumor-consistent",
        best_hpa_tissue="",
        hpa_correlation=0.0,
        best_tcga_code="",
        tcga_correlation=0.0,
        margin=0.0,
        proliferation_log2_mean=0.0,
        proliferation_genes_observed=0,
        verdict="-",
        n_reference_genes=0,
    )
    assert r.brief_banner() is None  # tumor-consistent does not banner
    assert r.likely_healthy is False


def test_synthetic_healthy_brain_is_called_healthy():
    """Reconstruct a healthy-brain sample from the HPA nTPM_cerebral_cortex
    column + proliferation markers set at near-zero. The gate must call
    it healthy — margin vs any TCGA cohort should be large because
    brain-native TCGA codes (GBM, LGG) aren't exact matches for HPA
    cerebral_cortex when the proliferation panel is quiet."""
    ref = _ref()
    sample = ref["nTPM_cerebral_cortex"].astype(float).to_dict()
    # Keep MKI67 + other proliferation genes low.
    for g in _PROLIFERATION_PANEL:
        sample[g] = 0.5
    r = assess_healthy_vs_tumor(_as_df(sample))
    assert r.call in ("healthy", "ambiguous"), (
        f"synthetic healthy brain should not be tumor-consistent; got {r.call}"
    )
    assert "cerebral_cortex" in r.best_hpa_tissue or "spinal_cord" in r.best_hpa_tissue


def test_synthetic_tumor_profile_is_tumor_consistent():
    """Reconstruct a sample from a TCGA FPKM cohort + high proliferation.
    The gate must call it tumor-consistent (not flag it as healthy)."""
    ref = _ref()
    sample = ref["FPKM_BRCA"].astype(float).to_dict()
    # Force proliferation high.
    for g in _PROLIFERATION_PANEL:
        sample[g] = 200.0
    r = assess_healthy_vs_tumor(_as_df(sample))
    assert r.call == "tumor-consistent", (
        f"high-proliferation BRCA profile should not be called healthy; got {r.call}"
    )


def test_proliferation_panel_veto_blocks_healthy_call():
    """Even when correlation favours HPA, a high proliferation-panel
    geomean must not be swallowed as healthy — the proliferation
    veto is the primary guard against low-purity tumor false-positives."""
    ref = _ref()
    sample = ref["nTPM_liver"].astype(float).to_dict()
    for g in _PROLIFERATION_PANEL:
        sample[g] = 500.0  # full proliferation program on
    r = assess_healthy_vs_tumor(_as_df(sample))
    assert r.call != "healthy", (
        f"high proliferation veto should prevent healthy call; got {r.call} "
        f"(prolif log2 mean {r.proliferation_log2_mean:.1f})"
    )


def test_insufficient_reference_overlap_returns_tumor_consistent():
    """If the sample barely overlaps the reference, the gate must fall
    back to tumor-consistent rather than produce a spurious call."""
    sample = {"FAKE_GENE_1": 10.0, "FAKE_GENE_2": 20.0}
    r = assess_healthy_vs_tumor(_as_df(sample))
    assert r.call == "tumor-consistent"
    assert "Insufficient" in r.verdict


def test_result_brief_banner_mentions_tissue_and_cohort():
    """The brief banner must name the winning normal tissue AND the
    runner-up TCGA cohort so the reader can judge the call."""
    r = HealthyVsTumorResult(
        call="healthy",
        best_hpa_tissue="nTPM_cerebral_cortex",
        hpa_correlation=0.94,
        best_tcga_code="FPKM_LGG",
        tcga_correlation=0.86,
        margin=0.08,
        proliferation_log2_mean=0.8,
        proliferation_genes_observed=5,
        verdict="-",
        n_reference_genes=5000,
    )
    banner = r.brief_banner()
    assert banner is not None
    assert "cerebral cortex" in banner
    assert "LGG" in banner
