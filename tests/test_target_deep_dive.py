# Licensed under the Apache License, Version 2.0

"""Tests for therapy target deep-dive and subtype signature plots."""

import pandas as pd

from pirlygenes.gene_sets_cancer import pan_cancer_expression
from pirlygenes.plot_target_deep_dive import (
    actionable_surface_targets,
    plot_actionable_targets,
    plot_curated_target_evidence,
    plot_priority_target_context,
    plot_priority_targets,
    plot_tumor_attribution,
    plot_cta_deep_dive,
    _CANCER_SURFACE_TARGETS,
)
from pirlygenes.plot_subtype_signature import (
    SUBTYPE_CONTRASTS,
    plot_subtype_signature,
)


def _tcga_sample(cancer_code):
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame({
        "ensembl_gene_id": ref["Ensembl_Gene_ID"],
        "gene_symbol": ref["Symbol"],
        "TPM": ref[f"FPKM_{cancer_code}"].astype(float),
    })


# ── actionable_surface_targets ───────────────────────────────────────────


def test_prad_targets_include_psma():
    targets = actionable_surface_targets("PRAD")
    assert "FOLH1" in targets


def test_brca_targets_include_her2():
    targets = actionable_surface_targets("BRCA")
    assert "ERBB2" in targets


def test_unknown_type_uses_default():
    targets = actionable_surface_targets("ACC")
    assert len(targets) > 0


def test_curated_panels_cover_major_types():
    for code in ["PRAD", "BRCA", "LUAD", "COAD", "SKCM", "OV", "LIHC", "GBM", "BLCA"]:
        assert code in _CANCER_SURFACE_TARGETS
        assert len(_CANCER_SURFACE_TARGETS[code]) >= 5


# ── plot_actionable_targets ──────────────────────────────────────────────


def test_plot_actionable_targets_saves_png(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "targets.png"
    fig = plot_actionable_targets(df, "PRAD", purity_estimate=0.6,
                                  save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_actionable_targets_without_purity(tmp_path):
    df = _tcga_sample("BRCA")
    out = tmp_path / "targets.png"
    fig = plot_actionable_targets(df, "BRCA", save_to_filename=str(out))
    assert fig is not None


# ── plot_tumor_attribution ───────────────────────────────────────────────


def test_plot_tumor_attribution_surface(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "attrib.png"
    fig = plot_tumor_attribution(df, "PRAD", purity_estimate=0.6,
                                  save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_tumor_attribution_cta(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "attrib_cta.png"
    fig = plot_tumor_attribution(df, "PRAD", purity_estimate=0.6,
                                  category="CTA",
                                  save_to_filename=str(out))
    # May be None if no CTAs expressed
    if fig is not None:
        assert out.exists()


# ── plot_cta_deep_dive ───────────────────────────────────────────────────


def test_plot_cta_deep_dive_saves_png(tmp_path):
    df = _tcga_sample("SKCM")  # melanoma has high CTA expression
    out = tmp_path / "cta.png"
    fig = plot_cta_deep_dive(df, "SKCM", purity_estimate=0.65,
                              save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_curated_target_evidence_saves_png(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 28.0,
                "attr_tumor_tpm_high": 39.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.32,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "category": "therapy_target",
            },
            {
                "symbol": "STEAP2",
                "observed_tpm": 90.0,
                "attr_tumor_tpm": 13.0,
                "attr_tumor_tpm_low": 8.0,
                "attr_tumor_tpm_high": 17.0,
                "attr_tumor_fraction": 0.14,
                "attr_tumor_fraction_low": 0.09,
                "attr_tumor_fraction_high": 0.19,
                "attr_support_fraction": 0.0,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 78.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": True,
                "matched_normal_over_predicted": False,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
            {
                "symbol": "STEAP2",
                "agent": "experimental ADC",
                "agent_class": "ADC",
                "phase": "phase_2",
                "indication": "mCRPC",
            },
        ]
    )
    out = tmp_path / "curated-evidence.png"
    fig = plot_curated_target_evidence(
        ranges_df,
        target_panel,
        "PRAD",
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()


def test_plot_priority_targets_saves_png(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 18.0,
                "attr_tumor_tpm_high": 40.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.21,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 0.67,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "ADC, radioligand",
                "tcga_percentile": 0.97,
                "category": "therapy_target",
            },
            {
                "symbol": "FAP",
                "observed_tpm": 44.0,
                "attr_tumor_tpm": 41.0,
                "attr_tumor_tpm_low": 37.0,
                "attr_tumor_tpm_high": 44.0,
                "attr_tumor_fraction": 0.94,
                "attr_tumor_fraction_low": 0.83,
                "attr_tumor_fraction_high": 1.0,
                "attr_support_fraction": 1.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "attr_top_compartment": "endothelial",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "ADC, radioligand",
                "tcga_percentile": 1.0,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
        ]
    )
    out = tmp_path / "priority-targets.png"
    fig = plot_priority_targets(
        ranges_df,
        "PRAD",
        target_panel=target_panel,
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()


def test_plot_priority_target_context_saves_png(tmp_path):
    ranges_df = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "observed_tpm": 87.0,
                "attr_tumor_tpm": 34.0,
                "attr_tumor_tpm_low": 18.0,
                "attr_tumor_tpm_high": 40.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.21,
                "attr_tumor_fraction_high": 0.45,
                "attr_support_fraction": 0.67,
                "matched_normal_tissue": "prostate",
                "matched_normal_tpm": 46.0,
                "attr_top_compartment": "matched_normal_prostate",
                "tme_explainable": True,
                "tme_dominant": False,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "ADC, radioligand",
                "tcga_percentile": 0.97,
                "category": "therapy_target",
            },
            {
                "symbol": "FAP",
                "observed_tpm": 44.0,
                "attr_tumor_tpm": 17.0,
                "attr_tumor_tpm_low": 10.0,
                "attr_tumor_tpm_high": 24.0,
                "attr_tumor_fraction": 0.39,
                "attr_tumor_fraction_low": 0.23,
                "attr_tumor_fraction_high": 0.55,
                "attr_support_fraction": 0.0,
                "matched_normal_tissue": "",
                "matched_normal_tpm": 0.0,
                "attr_top_compartment": "fibroblast",
                "tme_explainable": True,
                "tme_dominant": True,
                "matched_normal_over_predicted": False,
                "therapy_supported": True,
                "therapies": "radioligand",
                "tcga_percentile": 1.0,
                "category": "therapy_target",
            },
        ]
    )
    target_panel = pd.DataFrame(
        [
            {
                "symbol": "FOLH1",
                "agent": "177Lu-PSMA-617",
                "agent_class": "radioligand",
                "phase": "approved",
                "indication": "mCRPC",
            },
        ]
    )
    out = tmp_path / "priority-target-context.png"
    fig = plot_priority_target_context(
        ranges_df,
        "PRAD",
        target_panel=target_panel,
        save_to_filename=str(out),
    )
    assert fig is not None
    assert out.exists()
    ax_range = fig.axes[0]
    assert ax_range.get_xscale() == "linear"
    assert "log10(TPM+1)" in ax_range.get_xlabel()
    assert fig.legends


# ── subtype signatures ───────────────────────────────────────────────────


def test_subtype_contrasts_exist_for_prad():
    assert "PRAD" in SUBTYPE_CONTRASTS
    assert SUBTYPE_CONTRASTS["PRAD"][0]["axis_a"] == "AR_signaling"
    assert SUBTYPE_CONTRASTS["PRAD"][0]["axis_b"] == "NE_differentiation"


def test_subtype_contrasts_exist_for_brca():
    assert "BRCA" in SUBTYPE_CONTRASTS


def test_plot_subtype_signature_prad(tmp_path):
    df = _tcga_sample("PRAD")
    out = tmp_path / "subtype.png"
    fig = plot_subtype_signature(df, "PRAD", save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_subtype_signature_brca(tmp_path):
    df = _tcga_sample("BRCA")
    out = tmp_path / "subtype.png"
    fig = plot_subtype_signature(df, "BRCA", save_to_filename=str(out))
    assert fig is not None
    assert out.exists()


def test_plot_subtype_signature_no_contrast_returns_none():
    df = _tcga_sample("ACC")
    fig = plot_subtype_signature(df, "ACC")
    assert fig is None
