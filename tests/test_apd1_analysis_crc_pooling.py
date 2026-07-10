"""Regression tests for CRC source-scope pooling in aPD1/ICI analyses."""

import math
from pathlib import Path
import sys

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import cancer_apd1_response, cancer_tmb
from pirlygenes.load_dataset import get_data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analyses"))

import _apd1_factors as factors  # noqa: E402
from _apd1_factors import (  # noqa: E402
    apd1_map,
    indel_map,
    pool_colorectal_axis,
    tmb_map,
)

_CRC_SOURCE_CODES = {"COAD", "READ", "COAD_MSI", "READ_MSI", "COAD_MSS", "READ_MSS"}


def test_crc_subtype_lookups_resolve_via_oncoref():
    """Clinical aPD1/TMB rows are owned by oncoref (pirlygenes#541), which folds
    the COAD/READ microsatellite source rows into the ``CRC_MSI`` anchor. The
    finer source codes are no longer raw rows in the re-exported frame, but a
    lookup for one still resolves to the CRC anchor value through oncoref's
    source-scope resolver."""
    apd1_codes = set(get_data("cancer-apd1-response.csv")["cancer_code"])
    tmb_codes = set(get_data("cancer-tmb.csv")["cancer_code"])

    # the CRC molecular-subtype anchor is present; the COAD/READ MSI source
    # split rows are folded into it, not carried as independent rows.
    assert "CRC_MSI" in apd1_codes and "CRC_MSI" in tmb_codes
    assert not ({"COAD_MSI", "READ_MSI"} & apd1_codes)
    assert not ({"COAD_MSI", "READ_MSI"} & tmb_codes)

    # source-scope subtype lookups still resolve to the CRC anchor value.
    assert cancer_apd1_response("COAD_MSI") == cancer_apd1_response("CRC_MSI")
    assert cancer_apd1_response("READ_MSI") == cancer_apd1_response("CRC_MSI")
    assert cancer_tmb("COAD_MSI") == cancer_tmb("CRC_MSI")
    assert cancer_tmb("READ_MSI") == cancer_tmb("CRC_MSI")


def test_clinical_anchor_maps_hide_crc_source_split_rows():
    """ORR/TMB/indel axes are CRC-scoped clinical anchors, not duplicate points."""
    apd1 = apd1_map()
    tmb = tmb_map()
    indel = indel_map()

    assert apd1["CRC_MSI"] == pytest.approx(cancer_apd1_response("CRC_MSI"))
    assert apd1["CRC_MSS"] == pytest.approx(0.0)
    assert tmb["CRC_MSI"] == pytest.approx(46.0)
    assert indel["CRC_MSI"] == pytest.approx(2.0)
    assert indel["CRC_MSS"] == pytest.approx(0.0)

    for mapping in (apd1, tmb, indel):
        assert not (_CRC_SOURCE_CODES & set(mapping))


def test_pooling_helper_can_retain_sources_for_measured_feature_axes():
    """Measured subtype features can keep source rows while adding CRC tiers."""
    raw = {"COAD_MSI": 1.0, "READ_MSI": 3.0, "COAD": 5.0, "READ": 7.0}
    pooled = pool_colorectal_axis(raw, keep_source_codes=True)

    assert pooled["COAD_MSI"] == pytest.approx(1.0)
    assert pooled["READ_MSI"] == pytest.approx(3.0)
    assert pooled["CRC_MSI"] == pytest.approx(2.0)
    assert pooled["CRC"] == pytest.approx(6.0)


def test_cohort_gene_matrix_expands_requested_crc_tiers(monkeypatch):
    """Pooled clinical keys still fetch the underlying measured expression rows."""
    seen = {}

    def fake_reference_expression(cancer_types, **_kwargs):
        seen["cancer_types"] = list(cancer_types)
        rows = []
        values = {
            "COAD_MSI": 10.0,
            "READ_MSI": 30.0,
            "COAD_MSS": 2.0,
            "READ_MSS": 6.0,
            "UCEC": 1.0,
            "UCS": 1.0,
        }
        for code, value in values.items():
            if code in cancer_types:
                rows.append({
                    "cancer_code": code,
                    "source_cohort": "fake",
                    "n_samples": 1,
                    "processing_pipeline": "rnaseq",
                    "Symbol": "GENE1",
                    "expression": value,
                })
        return pd.DataFrame(rows)

    monkeypatch.setattr(factors, "cancer_reference_expression", fake_reference_expression)
    monkeypatch.setattr(factors, "_ucec_subtype_tpm", lambda: None)

    mat = factors.cohort_gene_matrix(["CRC_MSI", "CRC_MSS"])

    assert {"COAD_MSI", "READ_MSI", "COAD_MSS", "READ_MSS"} <= set(
        seen["cancer_types"])
    assert "COAD_MSI" not in mat.index
    assert "READ_MSI" not in mat.index
    assert mat.loc["CRC_MSI", "GENE1"] == pytest.approx(math.log10(21.0))
    assert mat.loc["CRC_MSS", "GENE1"] == pytest.approx(math.log10(5.0))
