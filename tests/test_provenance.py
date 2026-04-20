"""Smoke tests for the provenance page (#106)."""

import pandas as pd

from pirlygenes.provenance import build_provenance_md, plot_provenance_funnel


class _Ctx:
    def __init__(self, prep="exome_capture", preservation="ffpe",
                 severity="mild", index=0.55, confidence=0.9):
        self.library_prep = prep
        self.library_prep_confidence = confidence
        self.preservation = preservation
        self.preservation_confidence = 0.85
        self.degradation_severity = severity
        self.degradation_index = index
        self.missing_mt = False
        self.signals = {}
        self.flags = []


class _Decomp:
    def __init__(self, purity=0.28):
        self.purity = purity
        self.fractions = {
            "tumor": purity,
            "T_cell": 0.03,
            "endothelial": 0.03,
            "fibroblast": 0.25,
            "myeloid": 0.15,
            "matched_normal_prostate": 0.26,
        }
        self.component_trace = pd.DataFrame([
            {"component": "T_cell", "fraction": 0.03},
            {"component": "matched_normal_prostate", "fraction": 0.26},
        ])


def _ranges_df():
    return pd.DataFrame([
        {"symbol": "FOLH1", "observed_tpm": 142.0,
         "attribution": {"endothelial": 12.0},
         "attr_tumor_tpm": 128.0, "attr_tumor_fraction": 0.90},
        {"symbol": "STEAP1", "observed_tpm": 78.0,
         "attribution": {"fibroblast": 10.0},
         "attr_tumor_tpm": 62.0, "attr_tumor_fraction": 0.79},
    ])


def test_provenance_md_walks_the_five_steps():
    analysis = {
        "sample_context": _Ctx(),
        "purity": {"overall_estimate": 0.28, "overall_lower": 0.19, "overall_upper": 0.40},
    }
    md = build_provenance_md(
        analysis, _ranges_df(), [_Decomp()],
        cancer_code="PRAD", sample_id="sample_X",
    )
    # Each numbered step must appear.
    for heading in ["1. Library prep", "2. Preservation",
                    "3. Coarse composition", "4. Subtype refinements",
                    "5. Tumor-specific core"]:
        assert heading in md, f"missing step heading: {heading}"
    assert "exome capture" in md
    assert "FOLH1" in md or "tumor-core" in md.lower()


def test_provenance_handles_missing_decomposition():
    analysis = {
        "sample_context": _Ctx(),
        "purity": {"overall_estimate": 0.28, "overall_lower": 0.19, "overall_upper": 0.40},
    }
    md = build_provenance_md(
        analysis, _ranges_df(), decomp_results=[],
        cancer_code="PRAD",
    )
    # Still renders — just with a no-decomposition note.
    assert "No decomposition result" in md


def test_provenance_funnel_renders_png(tmp_path):
    analysis = {
        "sample_context": _Ctx(),
        "purity": {"overall_estimate": 0.28, "overall_lower": 0.19, "overall_upper": 0.40},
    }
    out = tmp_path / "prov.png"
    result = plot_provenance_funnel(
        analysis, _ranges_df(), [_Decomp()],
        save_to_filename=str(out),
    )
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000
