# Licensed under the Apache License, Version 2.0

"""Tests for therapy target deep-dive and subtype signature plots."""

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import pan_cancer_expression
from pirlygenes.plot_target_deep_dive import (
    actionable_surface_targets,
    plot_actionable_targets,
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
