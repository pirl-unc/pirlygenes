"""Contract tests for pirlygenes.expression — the reference data +
mechanical-transforms layer added in 5.1.0 (issues #246, #247).

Covers:
  - Each accessor returns a non-empty frame with expected columns
  - Topiary's call pattern (load_all_dataframes_dict()[…]) still works
  - normalize_expression, fpkm_to_tpm, tpm_to_housekeeping_normalized
    produce expected shapes
  - classify_gene_qc gives the correct family for representative genes
  - filter_technical_rna sources drop-ids from gene_families
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pirlygenes import load_all_dataframes_dict
import pirlygenes.expression.accessors as expression_accessors
from pirlygenes.expression import (
    GeneQcClass,
    add_tpm_columns_from_fpkm,
    aggregate_gene_expression,
    available_cancer_expression_references,
    cancer_enriched_genes,
    cancer_expression,
    cancer_expression_reference_status,
    cancer_expression_source_candidates,
    cancer_reference_expression,
    classify_gene_qc,
    estimate_signatures,
    filter_technical_rna,
    filter_to_genes,
    fpkm_to_tpm,
    heme_tumor_up_vs_matched_normal,
    hpa_cell_type_expression,
    is_rescue_feature,
    log1p_transform,
    log2_transform,
    normalize_expression,
    normalize_to_housekeeping,
    pan_cancer_expression,
    percentile_rank_expression,
    renormalize_to_million,
    technical_rna_gene_ids,
    tumor_up_vs_matched_normal,
    tpm_to_housekeeping_normalized,
)


# ---------- reference accessors ----------


def _local_pan_cancer_fixture():
    """Small pre-canonical pan matrix + persisted rollup-view inputs."""
    fpkm = np.array([2.0, 8.0, 1.0, 10.0, 5.0, 3.0, 4.0, 3.0])
    raw = pd.DataFrame({
        "Ensembl_Gene_ID": [
            "ENSG00000146648",  # EGFR
            "ENSG00000141510",  # TP53
            "ENSG00000198888",  # MT-ND1 (technical)
            "ENSG00000111640",  # GAPDH (housekeeping)
            "ENSG00000147604",  # RPL7 (ribosomal protein)
            "ENSG00000148362",  # PAXX legacy id
            "ENSG00000015479",  # canonical MATR3 row
            "ENSG00000280987",  # retired MATR3 row
        ],
        "Symbol": [
            "EGFR", "TP53", "MT-ND1", "GAPDH", "RPL7", "PAXX", "MATR3", "MATR3",
        ],
        "nTPM_liver": [20.0, 10.0, 5.0, 40.0, 30.0, 2.0, 12.0, 7.0],
        "FPKM_LUAD": fpkm,
    })

    # PAXX and MATR3 are deliberately in pirlygenes' old key space. TP53 is
    # absent from every CRC member, exercising unavailable-vs-zero semantics.
    rollup = raw[["Ensembl_Gene_ID", "Symbol"]].copy()
    rollup["CHOL"] = [10.0, 20.0, 5.0, 40.0, 30.0, 6.0, 12.0, 3.0]
    rollup["COAD"] = [100.0, np.nan, 50.0, 250.0, 150.0, 25.0, 125.0, 100.0]
    rollup["READ"] = [200.0, np.nan, 60.0, 260.0, 160.0, 35.0, 225.0, 75.0]
    for code, offset in (
        ("NET_PANCREAS", 0.0),
        ("NET_MIDGUT", 10.0),
        ("NET_RECTAL", 20.0),
        ("NET_LUNG", 30.0),
        ("LUAD", 40.0),
        ("LUSC", 50.0),
        ("ADCC", 60.0),
    ):
        rollup[code] = rollup["CHOL"] + offset

    weights = {
        "CHOL": 36,
        "COAD": 3,
        "READ": 1,
        "NET_PANCREAS": 33,
        "NET_MIDGUT": 81,
        "NET_RECTAL": 18,
        "NET_LUNG": 118,
        "LUAD": 515,
        "LUSC": 498,
        "ADCC": 99,
    }
    persisted = rollup[["Ensembl_Gene_ID"]].copy()
    for aggregate, members in (
        ("BTC", ("CHOL",)),
        ("CRC", ("COAD", "READ")),
        ("NET", ("NET_PANCREAS", "NET_MIDGUT", "NET_RECTAL", "NET_LUNG")),
        ("NSCLC", ("LUAD", "LUSC")),
        ("SGC", ("ADCC",)),
    ):
        member_weights = pd.Series({code: weights[code] for code in members})
        values = rollup[list(members)]
        numerator = values.mul(member_weights, axis="columns").sum(
            axis=1, min_count=1
        )
        denominator = values.notna().mul(member_weights, axis="columns").sum(axis=1)
        persisted[f"TPM_{aggregate}"] = numerator.div(
            denominator.where(denominator > 0)
        )
    return raw, persisted


@pytest.fixture
def local_pan_cancer(monkeypatch):
    """Stub the two local artifacts and isolate the canonical-frame cache."""
    import oncoref

    raw, rollup = _local_pan_cancer_fixture()
    calls = {"pan": 0, "rollup": 0}

    def fake_get_data(name, *, copy=True):
        if name == "pan-cancer-expression":
            calls["pan"] += 1
            return raw.copy()
        if name == "pan-cancer-expression-rollups":
            calls["rollup"] += 1
            return rollup.copy()
        raise AssertionError(f"unexpected dataset {name!r}")

    def forbidden_eager_call(*args, **kwargs):
        raise AssertionError("the pan adapter must not call oncoref's eager accessor")

    expression_accessors._pan_reference_frame.cache_clear()
    expression_accessors._load_pan_rollup_frame.cache_clear()
    monkeypatch.setattr(expression_accessors, "get_data", fake_get_data)
    monkeypatch.setattr(oncoref, "pan_cancer_expression", forbidden_eager_call)
    yield raw, rollup, calls
    expression_accessors._pan_reference_frame.cache_clear()
    expression_accessors._load_pan_rollup_frame.cache_clear()


def test_pan_reference_frame_uses_local_artifacts_once_without_eager_oncoref(
    local_pan_cancer,
):
    raw, _rollup, calls = local_pan_cancer

    first = expression_accessors._pan_reference_frame()
    second = expression_accessors._pan_reference_frame()

    assert first is second
    assert calls == {"pan": 1, "rollup": 1}
    assert {"nTPM_liver", "FPKM_LUAD", "TPM_LUAD", "TPM_CRC"} <= set(first.columns)
    assert len(first) == len(raw) - 1  # the two MATR3 rows collapse
    assert first["TPM_LUAD"].sum() == pytest.approx(1_000_000)


def test_pan_cancer_adapter_preserves_all_normalize_contracts(
    local_pan_cancer,
):
    _raw, _rollup, calls = local_pan_cancer
    canonical = expression_accessors._pan_reference_frame()
    base_cols = {"liver_nTPM", "LUAD_FPKM", "LUAD_TPM", "CRC_TPM"}

    default = pan_cancer_expression()
    explicit_clean = pan_cancer_expression(normalize="tpm_clean")
    pd.testing.assert_frame_equal(default, explicit_clean)

    raw_view = pan_cancer_expression(normalize=None)
    tpm_view = pan_cancer_expression(normalize="tpm")
    raw_log = pan_cancer_expression(normalize="tpm_log1p")
    clean_log = pan_cancer_expression(normalize="tpm_clean_log1p")
    combined = pan_cancer_expression(
        normalize=["tpm_clean", "hk", "percentile"],
    )

    for frame in (default, raw_view, tpm_view, raw_log, clean_log, combined):
        assert base_cols <= set(frame.columns)
        assert not any(c.startswith(("nTPM_", "FPKM_", "TPM_")) for c in frame.columns)
        assert not any(c.endswith("_raw") for c in frame.columns)
        assert frame["LUAD_FPKM"].tolist() == canonical["FPKM_LUAD"].tolist()
        assert frame["LUAD_TPM"].tolist() == pytest.approx(canonical["TPM_LUAD"])

    normalized_cols = {
        "liver_nTPM_clean", "LUAD_TPM_clean", "CRC_TPM_clean",
    }
    assert normalized_cols <= set(default.columns)
    assert not normalized_cols & set(raw_view.columns)
    assert list(raw_view.columns) == list(tpm_view.columns)
    assert {"liver_nTPM_log1p", "LUAD_TPM_log1p", "CRC_TPM_log1p"} <= set(
        raw_log.columns
    )
    assert {
        "liver_nTPM_clean_log1p", "LUAD_TPM_clean_log1p", "CRC_TPM_clean_log1p",
    } <= set(clean_log.columns)
    assert {
        "liver_nTPM_clean", "LUAD_TPM_clean", "CRC_TPM_clean",
        "liver_nTPM_hk", "LUAD_TPM_hk", "CRC_TPM_hk",
        "liver_nTPM_percentile", "LUAD_TPM_percentile", "CRC_TPM_percentile",
    } <= set(combined.columns)

    # Every mode shares one cheap read of each persisted local artifact.
    assert calls == {"pan": 1, "rollup": 1}


def test_pan_cancer_canonical_rows_rollups_and_legacy_gene_filters(
    local_pan_cancer,
):
    _raw, _rollup, calls = local_pan_cancer

    by_symbol = pan_cancer_expression(genes="EGFR", normalize="tpm")
    by_unversioned_id = pan_cancer_expression(
        genes=["ENSG00000146648"], normalize="tpm",
    )
    by_versioned_id = pan_cancer_expression(
        genes=["ENSG00000146648.17"], normalize="tpm",
    )
    for frame in (by_symbol, by_unversioned_id, by_versioned_id):
        assert frame["Symbol"].tolist() == ["EGFR"]

    paxx = pan_cancer_expression(genes=["PAXX"], normalize="tpm")
    assert paxx["Ensembl_Gene_ID"].tolist() == ["ENSG00000310560"]
    assert "ENSG00000148362" not in set(paxx["Ensembl_Gene_ID"])
    assert paxx["CRC_TPM"].tolist() == pytest.approx([(25.0 * 3 + 35.0) / 4])
    assert paxx[["BTC_TPM", "CRC_TPM", "NET_TPM", "NSCLC_TPM", "SGC_TPM"]].notna().all(axis=None)

    paxx_legacy = pan_cancer_expression(
        genes=["ENSG00000148362.9"], normalize="tpm",
    )
    assert paxx_legacy["Ensembl_Gene_ID"].tolist() == ["ENSG00000310560"]

    matr3 = pan_cancer_expression(genes=["MATR3"], normalize="tpm")
    assert len(matr3) == 1
    assert matr3["Ensembl_Gene_ID"].tolist() == ["ENSG00000015479"]
    assert matr3["liver_nTPM"].tolist() == [19.0]
    assert matr3["CRC_TPM"].tolist() == pytest.approx([(225.0 * 3 + 300.0) / 4])

    matr3_retired = pan_cancer_expression(
        genes=["ENSG00000280987"], normalize="tpm",
    )
    assert matr3_retired["Ensembl_Gene_ID"].tolist() == ["ENSG00000015479"]
    assert calls == {"pan": 1, "rollup": 1}


def test_pan_cancer_unavailable_rollup_stays_missing_in_every_derivative(
    local_pan_cancer,
):
    for mode, derived in (
        ("tpm_clean", "CRC_TPM_clean"),
        ("tpm_clean_log1p", "CRC_TPM_clean_log1p"),
        ("hk", "CRC_TPM_hk"),
        ("percentile", "CRC_TPM_percentile"),
    ):
        row = pan_cancer_expression(genes=["TP53"], normalize=mode).iloc[0]
        assert pd.isna(row["CRC_TPM"])
        assert pd.isna(row[derived])


def test_pan_cancer_expression_returns_wide_frame_with_tpm_companions():
    df = pan_cancer_expression()
    assert not df.empty
    assert "Ensembl_Gene_ID" in df.columns
    assert "Symbol" in df.columns
    # <tissue>_nTPM from HPA, raw <code>_FPKM from TCGA, and deterministic
    # <code>_TPM companions derived from FPKM all coexist.
    assert any(c.endswith("_nTPM") for c in df.columns)
    assert any(c.endswith("_FPKM") for c in df.columns)
    assert any(c.endswith("_TPM") for c in df.columns)
    assert not any(c.startswith("tcga_") for c in df.columns)


def test_pan_cancer_expression_subset_filters_to_named_genes():
    df = pan_cancer_expression(genes=["KLK3", "MYC"])
    assert len(df) >= 2  # both symbols + maybe alias variants
    assert set(df["Symbol"].str.upper()) >= {"KLK3", "MYC"}


def test_pan_cancer_expression_housekeeping_rescales_to_unit_baseline():
    df = pan_cancer_expression(normalize="hk")
    tpm_cols = [c for c in df.columns if c.endswith("_TPM_hk")]
    source_tpm_cols = [
        col
        for col in tpm_cols
        if f"{col[:-len('_TPM_hk')]}_FPKM" in df.columns
    ]
    assert len(source_tpm_cols) == 33
    # Median-of-ratios does not rescale each HK gene to 1.0 (that was the old
    # geomean behavior); it puts the whole sample on the reference-profile
    # scale. The defining invariant is that the median over HK genes of
    # normalized_tpm / reference_tpm sits at ~1.0. Check that instead.
    from pirlygenes import housekeeping_gene_ids
    from oncoref import housekeeping_reference_profile

    ref = housekeeping_reference_profile()
    ref_tpm = dict(
        zip(ref["Ensembl_Gene_ID"].astype(str), ref["reference_tpm"].astype(float))
    )
    hk = housekeeping_gene_ids()
    hk_rows = df[df["Ensembl_Gene_ID"].isin(hk)]
    for col in tpm_cols[:3]:  # spot-check first few columns
        ratios = [
            value / ref_tpm[gene]
            for gene, value in zip(
                hk_rows["Ensembl_Gene_ID"].astype(str),
                hk_rows[col].astype(float),
            )
            if ref_tpm.get(gene, 0.0) > 0
        ]
        med = float(np.median(ratios))
        assert med == pytest.approx(1.0, rel=0.05), (
            f"{col} median housekeeping ratio-to-reference is {med}, expected ~1.0"
        )


def test_cancer_enriched_genes_excludes_computed_rollups_from_background(
    monkeypatch,
):
    frame = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000146648"],
        "Symbol": ["EGFR"],
        "LUAD_FPKM": [1.0],
        "LUSC_FPKM": [1.0],
        "COAD_FPKM": [1.0],
        "LUAD_TPM_hk": [12.0],
        "LUSC_TPM_hk": [4.0],
        "COAD_TPM_hk": [8.0],
        # Deliberately extreme rollups: none may affect the source background.
        "NSCLC_TPM_hk": [1_000.0],
        "CRC_TPM_hk": [2_000.0],
        "BTC_TPM_hk": [3_000.0],
    })
    monkeypatch.setattr(
        expression_accessors,
        "pan_cancer_expression",
        lambda **_kwargs: frame.copy(),
    )

    enriched = cancer_enriched_genes("LUAD", min_fold=0.0, min_expression=0.0)

    assert enriched.loc[0, "other_median"] == pytest.approx(6.0)
    assert enriched.loc[0, "fold_change"] == pytest.approx(
        (12.0 + 0.001) / (6.0 + 0.001)
    )


def test_cancer_enriched_genes_excludes_members_of_aggregate_target(monkeypatch):
    frame = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000146648"],
        "Symbol": ["EGFR"],
        "LUAD_FPKM": [1.0],
        "LUSC_FPKM": [1.0],
        "COAD_FPKM": [1.0],
        "LUAD_TPM_hk": [100.0],
        "LUSC_TPM_hk": [200.0],
        "COAD_TPM_hk": [8.0],
        "NSCLC_TPM_hk": [12.0],
    })
    monkeypatch.setattr(
        expression_accessors,
        "pan_cancer_expression",
        lambda **_kwargs: frame.copy(),
    )

    enriched = cancer_enriched_genes("NSCLC", min_fold=0.0, min_expression=0.0)

    assert enriched.loc[0, "other_median"] == pytest.approx(8.0)


def test_cancer_expression_returns_per_symbol_expression_column():
    df = cancer_expression("PRAD")
    assert {"Ensembl_Gene_ID", "Symbol", "expression"} <= set(df.columns)
    assert not df.empty


def test_cancer_expression_tcga_default_is_clean_tpm_not_housekeeping():
    # PRAD now has bundled per-sample data via Treehouse 25.01 PolyA
    # TCGA subset (source_cohort=TREEHOUSE_POLYA_25_01_TCGA_SUBSET);
    # cancer_expression prefers that over the legacy pan-cancer FPKM
    # medians. Default should still be TPM_clean (not housekeeping).
    default = cancer_expression("PRAD", genes=["KLK3"])
    assert {"Ensembl_Gene_ID", "Symbol", "expression"} <= set(default.columns)
    # KLK3 is the canonical prostate-secreted gene; should be highly
    # expressed in the PRAD reference.
    klk3 = default[default["Symbol"] == "KLK3"]
    assert not klk3.empty
    assert float(klk3["expression"].iloc[0]) > 100

    # `normalize="hk"` should fall through to pan_cancer_expression
    # because TPM_hk is not a column on the per-sample-built reference.
    explicit_hk = cancer_expression("PRAD", genes=["KLK3"], normalize="hk")
    pan = pan_cancer_expression(genes=["KLK3"], normalize=["hk"])
    expected_hk = pan[["Ensembl_Gene_ID", "Symbol", "PRAD_TPM_hk"]].rename(
        columns={"PRAD_TPM_hk": "expression"}
    )
    pd.testing.assert_frame_equal(
        explicit_hk.reset_index(drop=True),
        expected_hk.reset_index(drop=True),
    )
    # Default (clean TPM from Treehouse) ≠ HK-normalized.
    assert not np.allclose(default["expression"], explicit_hk["expression"])


def test_hpa_cell_type_expression_long_form(monkeypatch):
    import oncoref

    expected = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1"],
            "Symbol": ["A"],
            "T-cells": [3.0],
        }
    )
    monkeypatch.setattr(oncoref, "hpa_cell_type_expression", lambda: expected.copy())
    monkeypatch.setattr(
        expression_accessors,
        "get_data",
        lambda name: pytest.fail(f"unexpected local dataset read: {name}"),
    )

    df = hpa_cell_type_expression()

    pd.testing.assert_frame_equal(df, expected)


def test_estimate_signatures_has_stromal_and_immune_classes():
    df = estimate_signatures()
    assert not df.empty


def test_available_cancer_expression_references_includes_cllmap():
    df = available_cancer_expression_references()
    cll = df[df["cancer_code"] == "CLL"]
    assert not cll.empty
    row = cll.iloc[0]
    assert row["source_cohort"] == "CLLMAP_2022"
    assert row["n_samples"] == 708


def test_available_cancer_expression_references_includes_mmrf():
    df = available_cancer_expression_references()
    mm = df[df["cancer_code"] == "MM"]
    assert not mm.empty
    row = mm.iloc[0]
    assert row["source_cohort"] == "MMRF_COMMPASS"
    assert row["n_samples"] == 764


def test_available_cancer_expression_references_includes_target_all_lineages():
    df = available_cancer_expression_references()
    refs = df[df["cancer_code"].isin(["B_ALL", "T_ALL"])].set_index("cancer_code")
    assert {"B_ALL", "T_ALL"} <= set(refs.index)
    assert refs.loc["B_ALL", "source_cohort"] == "TARGET_ALL_2018"
    assert refs.loc["T_ALL", "source_cohort"] == "TARGET_ALL_2018"
    assert refs.loc["B_ALL", "n_samples"] == 154
    assert refs.loc["T_ALL", "n_samples"] == 264


def test_available_cancer_expression_references_includes_imported_specific_cohorts():
    df = available_cancer_expression_references()

    def _canonical(code: str) -> str:
        # available_cancer_expression_references orders rows so primary
        # cohorts come before mixed/metastasis within a cancer_code, so
        # taking the first row yields the canonical reference.
        rows = df[df["cancer_code"] == code]
        assert not rows.empty, f"no packaged reference for {code}"
        return str(rows.iloc[0]["source_cohort"])

    def _n_samples(code: str) -> int:
        rows = df[df["cancer_code"] == code]
        return int(rows.iloc[0]["n_samples"])

    assert _canonical("SARC_OS") == "TREEHOUSE_POLYA_25_01"
    assert _n_samples("SARC_OS") == 262
    assert _canonical("NET_PANCREAS") == "GSE118014_ALVAREZ_2018"
    assert _canonical("SARC_CHON") == "GSE299759_MEIJER_2026"
    assert _canonical("SARC_DDLPS") == "GSE30929_SINGER_2007_LPS"
    assert _canonical("LAML_APL") == "BEATAML_OHSU_2022"
    assert _canonical("RB") == "TREEHOUSE_RIBOD_25_01"


def test_available_cancer_expression_references_includes_acquirable_heme_cohorts():
    df = available_cancer_expression_references()
    refs = df.set_index("cancer_code")

    expected = {
        "BL": ("CGCI_BLGSP", 184),
        "CML": ("GSE100026_DING_2017", 5),
        "CTCL": ("GSE171811_ECCITE_CTCL", 7),
        "MCL": ("GSE271664_BODOR_2025", 51),
        "MDS": ("GSE114922_SHIOZAWA_2018", 82),
        "MPN": ("GSE283710_WASHU_2024", 45),
    }
    for code, (source_cohort, n_samples) in expected.items():
        assert refs.loc[code, "source_cohort"] == source_cohort
        assert refs.loc[code, "n_samples"] == n_samples


def test_imported_specific_reference_keeps_one_default_source_per_code():
    df = available_cancer_expression_references()
    # RT remains a single-source code (TARGET_RT_2017 summary import).
    assert df[df["cancer_code"] == "RT"]["source_cohort"].tolist() == [
        "TARGET_RT_2017"
    ]
    # GBM and LGG are populated via the TCGA-via-Treehouse glioma
    # split (sweep_treehouse_tcga_glioma_split.py landed in commit
    # c45b9ca). Both should have exactly one packaged source row each.
    gbm_sources = df[df["cancer_code"] == "GBM"]["source_cohort"].tolist()
    lgg_sources = df[df["cancer_code"] == "LGG"]["source_cohort"].tolist()
    assert gbm_sources == ["TREEHOUSE_POLYA_25_01_TCGA_SUBSET"]
    assert lgg_sources == ["TREEHOUSE_POLYA_25_01_TCGA_SUBSET"]


def test_cancer_reference_expression_returns_cll_clean_tpm_by_default():
    df = cancer_reference_expression(cancer_types=["CLL"], genes=["MS4A1", "MALAT1"])
    assert {"Ensembl_Gene_ID", "Symbol", "cancer_code", "source_cohort"} <= set(
        df.columns
    )
    assert set(df["normalization"]) == {"TPM_clean"}
    ms4a1 = df[df["Symbol"] == "MS4A1"].iloc[0]
    malat1 = df[df["Symbol"] == "MALAT1"].iloc[0]
    assert ms4a1["expression"] > 100
    # clean_tpm_16_9_75 normalizes the technical block to a fixed 25% fraction
    # (not zeroed as in v1), so MALAT1 carries a suppressed-but-nonzero value.
    assert malat1["expression"] > 0


def test_cancer_reference_expression_expands_sarc_aggregate_to_subtype_union():
    """The pan-sarcoma ``SARC`` aggregate has no frozen shard; asking for it
    returns the UNION of its member subtype rows (each keeping its own subtype
    cancer_code + source_cohort), not an empty frame or a fabricated pooled
    row (Phase C union view)."""
    df = cancer_reference_expression(cancer_types="SARC", genes=["TP53"])
    codes = set(df["cancer_code"].astype(str))
    # the umbrella code itself never appears as a row (no SARC shard)
    assert "SARC" not in codes
    # several real histology atoms are present
    assert {"SARC_LMS", "SARC_DDLPS", "SARC_OS"} <= codes
    # the common-name alias expands identically
    via_alias = cancer_reference_expression(cancer_types="sarcoma", genes=["TP53"])
    assert set(via_alias["cancer_code"].astype(str)) == codes


def test_cancer_reference_expression_expands_histology_rollups():
    """``SARC_RMS`` / ``SARC_LPS`` are aggregate-only codes (not registry
    codes); the accessor expands them to their member subtypes' rows."""
    rms = cancer_reference_expression(cancer_types="SARC_RMS", genes=["TP53"])
    assert set(rms["cancer_code"].astype(str)) <= {
        "SARC_RMS_ERMS", "SARC_RMS_ARMS", "SARC_RMS_PRMS", "SARC_RMS_SSRMS",
    }
    assert not rms.empty
    lps = cancer_reference_expression(cancer_types="SARC_LPS", genes=["TP53"])
    assert "SARC_DDLPS" in set(lps["cancer_code"].astype(str))


def test_cancer_reference_expression_non_aggregate_code_unaffected():
    """A plain leaf code resolves to exactly itself (aggregate expansion does
    not leak into ordinary single-cohort lookups)."""
    df = cancer_reference_expression(cancer_types="SARC_LMS", genes=["TP53"])
    assert set(df["cancer_code"].astype(str)) == {"SARC_LMS"}


def test_cancer_reference_expression_can_return_raw_and_clean_wide():
    df = cancer_reference_expression(
        cancer_types="CLL",
        genes=["MS4A1"],
        normalize=["tpm", "tpm_clean"],
        format="wide",
    )
    assert {"CLL_TPM", "CLL_TPM_clean"} <= set(df.columns)
    row = df.iloc[0]
    assert row["CLL_TPM"] > 100
    assert row["CLL_TPM_clean"] > row["CLL_TPM"]


def test_cancer_reference_expression_empty_gene_subset_keeps_long_schema():
    df = cancer_reference_expression(cancer_types="CLL", genes=["NO_SUCH_GENE"])
    assert df.empty
    assert list(df.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "Proteoform_ID",
        "Member_Ensembl_Gene_IDs",
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
        "normalization",
        "expression",
        "q1",
        "q3",
    ]
    assert "TPM_clean_median" not in df.columns


def test_cancer_reference_expression_empty_gene_subset_keeps_wide_schema():
    df = cancer_reference_expression(
        cancer_types="CLL",
        genes=["NO_SUCH_GENE"],
        normalize=["tpm", "tpm_clean"],
        format="wide",
    )
    assert df.empty
    assert list(df.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "CLL_TPM",
        "CLL_TPM_clean",
    ]


def test_cancer_reference_expression_validates_format_for_empty_filters():
    with pytest.raises(ValueError, match="format must be 'long' or 'wide'"):
        cancer_reference_expression(
            cancer_types="CLL",
            genes=["NO_SUCH_GENE"],
            format="matrix",
        )


def test_cancer_expression_resolves_non_tcga_reference():
    df = cancer_expression("CLL", genes=["FCER2"])
    assert list(df["Symbol"]) == ["FCER2"]
    assert df["expression"].iloc[0] > 100


def test_cancer_expression_does_not_read_legacy_reference_bundle(monkeypatch):
    expression_accessors._oncoref_reference_code_set.cache_clear()
    monkeypatch.setattr(
        expression_accessors,
        "_load_cancer_reference_expression",
        lambda: (_ for _ in ()).throw(
            AssertionError("delegated cancer_expression read the legacy bundle")
        ),
    )
    df = cancer_expression("CLL", genes=["FCER2"])
    assert list(df["Symbol"]) == ["FCER2"]
    expression_accessors._oncoref_reference_code_set.cache_clear()


def test_cancer_expression_resolves_mmrf_reference():
    df = cancer_expression("MM", genes=["TNFRSF17", "SDC1", "GPRC5D"])
    assert set(df["Symbol"]) == {"TNFRSF17", "SDC1", "GPRC5D"}
    values = dict(zip(df["Symbol"], df["expression"]))
    # Thresholds reflect clean_tpm_16_9_75 three-compartment renormalization (the
    # biological block shares 75% of the budget, slightly below the v1 zeroing
    # levels): TNFRSF17 ~296, SDC1 ~317, GPRC5D ~130.
    assert values["TNFRSF17"] > 250
    assert values["SDC1"] > 300
    assert values["GPRC5D"] > 100


def test_cancer_reference_expression_mm_markers_and_clean_artifacts():
    markers = [
        "SDC1",
        "TNFRSF17",
        "CD38",
        "SLAMF7",
        "MZB1",
        "XBP1",
        "FCRH5",
        "GPRC5D",
        "MALAT1",
        "MT-CO1",
    ]
    df = cancer_reference_expression(
        cancer_types="MM",
        genes=markers,
        normalize=["tpm", "tpm_clean"],
    )
    clean = df[df["normalization"] == "TPM_clean"].set_index("Symbol")
    expected_symbols = set(markers) - {"FCRH5"} | {"FCRL5"}
    assert expected_symbols <= set(clean.index)
    for marker in ["SDC1", "TNFRSF17", "CD38", "SLAMF7", "MZB1", "XBP1"]:
        assert clean.loc[marker, "expression"] > 50
    assert clean.loc["FCRL5", "expression"] > 1
    assert clean.loc["GPRC5D", "expression"] > 100
    # clean_tpm_16_9_75 renormalizes the technical block to a fixed 25% fraction
    # rather than zeroing it, so MALAT1 / MT-CO1 are nonzero (and in a
    # technically-light cohort like CD138-sorted MM, even inflated toward the
    # 25% floor) — but they no longer ride the raw library-prep variation.
    assert clean.loc["MALAT1", "expression"] > 0
    assert clean.loc["MT-CO1", "expression"] > 0


def test_cancer_expression_resolves_target_all_references():
    b_all = cancer_expression("B_ALL", genes=["CD79A", "PAX5", "MALAT1"])
    b_values = dict(zip(b_all["Symbol"], b_all["expression"]))
    assert b_values["CD79A"] > 1000
    assert b_values["PAX5"] > 100
    # clean_tpm_16_9_75 normalizes technical RNA to a fixed 25% fraction (not zeroed).
    assert b_values["MALAT1"] > 0

    t_all = cancer_expression("T_ALL", genes=["CD3D", "BCL11B", "MALAT1"])
    t_values = dict(zip(t_all["Symbol"], t_all["expression"]))
    # CD3D ~972 under clean-TPM's 75% biological budget (was >1000 under v1 zeroing).
    assert t_values["CD3D"] > 900
    assert t_values["BCL11B"] > 100
    assert t_values["MALAT1"] > 0


def test_cancer_reference_expression_target_all_lineage_markers_separate():
    df = cancer_reference_expression(
        cancer_types=["B_ALL", "T_ALL"],
        genes=["CD79A", "CD3D"],
        include_provenance=False,
    )
    pivot = df.pivot_table(
        index="Symbol",
        columns="cancer_code",
        values="expression",
        aggfunc="first",
    )
    assert pivot.loc["CD79A", "B_ALL"] > pivot.loc["CD79A", "T_ALL"] * 10
    assert pivot.loc["CD3D", "T_ALL"] > pivot.loc["CD3D", "B_ALL"] * 10


def test_cancer_reference_expression_imported_os_reference_is_clean_by_default():
    # OS now built from Treehouse 25.01 PolyA per-sample data (262
    # samples). AARS1 / PRXL2C used to come in via the historical-
    # symbol-rescue path in the old summary import (AARS → AARS1,
    # AAED1 → PRXL2C); the new Treehouse builder uses strict
    # pyensembl 112 mapping so old symbols are dropped. Historical
    # rescue for Treehouse is queued (see audit "Open questions").
    df = cancer_reference_expression(
        cancer_types="SARC_OS",
        genes=["COL1A2", "RUNX2", "MALAT1", "MT-CO1"],
        normalize=["tpm", "tpm_clean"],
    )
    pivot = df.pivot_table(
        index="Symbol",
        columns="normalization",
        values="expression",
        aggfunc="first",
    )
    assert pivot.loc["COL1A2", "TPM"] > 7000
    assert pivot.loc["COL1A2", "TPM_clean"] > pivot.loc["COL1A2", "TPM"]
    assert pivot.loc["RUNX2", "TPM_clean"] > 80
    # clean_tpm_16_9_75 normalizes the technical block to a fixed 25% fraction
    # (not zeroed), so MALAT1 / MT-CO1 are suppressed but nonzero.
    assert pivot.loc["MALAT1", "TPM_clean"] > 0
    assert pivot.loc["MT-CO1", "TPM_clean"] > 0

    out = cancer_expression("SARC_OS", genes=["COL1A2", "MALAT1"])
    values = dict(zip(out["Symbol"], out["expression"]))
    assert values["COL1A2"] == pivot.loc["COL1A2", "TPM_clean"]
    assert values["MALAT1"] > 0


def test_cancer_reference_expression_acquirable_heme_markers_are_distinct():
    df = cancer_reference_expression(
        cancer_types=["BL", "CML", "MCL", "MDS", "MPN"],
        genes=[
            "MS4A1",
            "CD79A",
            "MYC",
            "CCND1",
            "SOX11",
            "ABL1",
            "MPO",
            "CD34",
            "MPL",
            "MALAT1",
        ],
        normalize="tpm_clean",
        include_provenance=False,
    )
    pivot = df.pivot_table(
        index="Symbol",
        columns="cancer_code",
        values="expression",
        aggfunc="first",
    )
    assert pivot.loc["MS4A1", "BL"] > 100
    assert pivot.loc["CD79A", "BL"] > 500
    assert pivot.loc["MYC", "BL"] > 100
    assert pivot.loc["CCND1", "MCL"] > 100
    assert pivot.loc["SOX11", "MCL"] > 50
    assert pivot.loc["MPO", "CML"] > 1000
    assert pivot.loc["ABL1", "CML"] > 10
    # CD34+ sorted HSPCs: CD34 is distinctly elevated (vs bulk CML ~4 TPM).
    # The v5.9 recount3 rebuild (STAR coverage / Gencode v26) puts the median
    # at ~66 TPM vs the prior author-HTSeq build's >100; both mark MDS as
    # CD34-high. Threshold tracks the recount3 source.
    assert pivot.loc["CD34", "MDS"] > 50
    assert pivot.loc["CD34", "MDS"] > 5 * pivot.loc["CD34", "CML"]
    assert pivot.loc["MPO", "MDS"] > 1000
    assert pivot.loc["CD34", "MPN"] > 50
    assert pivot.loc["MPL", "MPN"] > 100
    # clean_tpm_16_9_75 normalizes the technical block to a fixed 25% fraction
    # (not zeroed), so MALAT1 is nonzero across every heme cohort.
    assert (pivot.loc["MALAT1"] > 0).all()


def test_cancer_reference_expression_ctcl_scrna_markers_and_artifacts():
    df = cancer_reference_expression(
        cancer_types="CTCL",
        genes=["CD3D", "CD3E", "CCR4", "KIR3DL2", "TOX", "MALAT1", "MS4A1"],
        normalize=["tpm", "tpm_clean"],
    )
    pivot = df.pivot_table(
        index="Symbol",
        columns="normalization",
        values="expression",
        aggfunc="first",
    )

    # CD3D ~964 under clean-TPM's 75% biological budget (was >1000 under v1 zeroing).
    assert pivot.loc["CD3D", "TPM_clean"] > 900
    assert pivot.loc["CD3E", "TPM_clean"] > 1000
    assert pivot.loc["KIR3DL2", "TPM_clean"] > 200
    assert pivot.loc["TOX", "TPM_clean"] > 200
    assert pivot.loc["CCR4", "TPM_clean"] > 50
    assert pivot.loc["MS4A1", "TPM_clean"] < 20
    assert pivot.loc["MALAT1", "TPM"] > 1000
    # clean_tpm_16_9_75 normalizes technical RNA to a fixed 25% fraction (not zeroed).
    assert pivot.loc["MALAT1", "TPM_clean"] > 0


def test_imported_symbol_only_references_use_historical_symbol_rescue():
    # SCLC remains a summary-only import (SCLC_UCOLOGNE_2015 via
    # scripts/import_cancer_specific_expression.py) and still
    # benefits from the historical symbol rescue in that import path.
    # OS / RMS / SARC etc. moved to per-sample Treehouse builds and
    # use strict pyensembl 112 matching (no historical rescue yet).
    refs = load_all_dataframes_dict()["cancer-reference-expression.csv"]
    nbl_ref = refs[refs["cancer_code"].eq("NBL_MYCNnonamp")]
    assert len(nbl_ref) > 12_000
    # AARS1 (formerly AARS) and PRXL2C (formerly AAED1) are recovered
    # by the historical-Ensembl-name lookup in the summary importer.
    assert {
        "ENSG00000090861",  # source symbol AARS -> current AARS1
        "ENSG00000158122",  # source symbol AAED1 -> current PRXL2C
    } <= set(nbl_ref["Ensembl_Gene_ID"])
    symbols = nbl_ref.set_index("Ensembl_Gene_ID")["Symbol"].to_dict()
    assert symbols["ENSG00000090861"] == "AARS1"
    assert symbols["ENSG00000158122"] == "PRXL2C"


def test_imported_specific_reference_recovers_expected_cohort_markers():
    df = cancer_reference_expression(
        cancer_types=["SARC_DDLPS", "SARC_CHON", "NET_PANCREAS"],
        genes=["MDM2", "CDK4", "COL2A1", "ACAN", "CHGA"],
        normalize="tpm_clean",
        include_provenance=False,
    )
    pivot = df.pivot_table(
        index="Symbol",
        columns="cancer_code",
        values="expression",
        aggfunc="first",
    )
    # SARC_DDLPS thresholds reflect the microarray-TPM-proxy dynamic range
    # of GSE30929 (Affymetrix HG-U133A), where the 12q13-15 amplicon shows
    # at the median as CDK4 ~1400 and MDM2 ~70 (vs ~10 in non-amplified
    # MYXLPS/PLEOLPS); thresholds chosen to discriminate amp vs non-amp.
    # CHON / NET_PANCREAS assertions use RNA-seq sources and keep the original
    # >1000 thresholds.
    assert pivot.loc["MDM2", "SARC_DDLPS"] > 50
    assert pivot.loc["CDK4", "SARC_DDLPS"] > 1000
    assert pivot.loc["COL2A1", "SARC_CHON"] > 1000
    assert pivot.loc["ACAN", "SARC_CHON"] > 1000
    assert pivot.loc["CHGA", "NET_PANCREAS"] > 1000


def test_expression_gene_filters_accept_aliases():
    df = cancer_expression("MM", genes=["BCMA", "FCRH5"])
    assert set(df["Symbol"]) == {"TNFRSF17", "FCRL5"}
    values = dict(zip(df["Symbol"], df["expression"]))
    # TNFRSF17 ~296 under clean_tpm_16_9_75's 75% biological budget (was >300 under v1).
    assert values["TNFRSF17"] > 250
    assert values["FCRL5"] > 1


def test_cancer_expression_defaults_to_clean_tpm_for_all_packaged_references():
    refs = available_cancer_expression_references()
    assert not refs.empty

    for code in refs["cancer_code"].drop_duplicates():
        ref = cancer_reference_expression(
            cancer_types=code,
            normalize="tpm_clean",
            include_provenance=False,
        )
        gene_id = ref["Ensembl_Gene_ID"].iloc[0]
        expected = ref[ref["Ensembl_Gene_ID"] == gene_id][
            ["Ensembl_Gene_ID", "Symbol", "expression"]
        ].reset_index(drop=True)

        actual = cancer_expression(code, genes=[gene_id]).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


def test_reference_expression_default_is_source_generic(monkeypatch):
    fake = pd.DataFrame(
        [
            {
                "Ensembl_Gene_ID": "ENSG000001",
                "Symbol": "FAKE1",
                "cancer_code": "CLL",
                "source_cohort": "FAKE_CLL",
                "source_project": "fake",
                "source_version": "test",
                "TPM_median": 7.0,
                "TPM_q1": 5.0,
                "TPM_q3": 9.0,
                "TPM_mean": 7.5,
                "TPM_clean_median": 11.0,
                "TPM_clean_q1": 10.0,
                "TPM_clean_q3": 12.0,
                "n_samples": 3,
                "n_detected": 3,
                "processing_pipeline": "test",
                "notes": "",
            },
            {
                "Ensembl_Gene_ID": "ENSG000002",
                "Symbol": "FAKE2",
                "cancer_code": "MM",
                "source_cohort": "FAKE_MM",
                "source_project": "fake",
                "source_version": "test",
                "TPM_median": 13.0,
                "TPM_q1": 12.0,
                "TPM_q3": 14.0,
                "TPM_mean": 13.5,
                "TPM_clean_median": 17.0,
                "TPM_clean_q1": 16.0,
                "TPM_clean_q3": 18.0,
                "n_samples": 4,
                "n_detected": 4,
                "processing_pipeline": "test",
                "notes": "",
            },
        ]
    )
    monkeypatch.setattr(
        expression_accessors,
        "_load_cancer_reference_expression",
        lambda: fake.copy(),
    )

    # cancer_expression() discovers reference availability through the local
    # compatibility metadata above, then reads values through the delegated
    # public accessor (#557). Keep both halves of this synthetic unit test in
    # the same fake source without changing the separately imported real
    # cancer_reference_expression used by the wide empty-schema assertion below.
    def fake_delegated_reference(cancer_types=None, genes=None, **_kwargs):
        codes = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
        out = fake[fake["cancer_code"].isin(codes)].copy()
        if genes is not None:
            wanted = set(genes)
            out = out[
                out["Ensembl_Gene_ID"].isin(wanted) | out["Symbol"].isin(wanted)
            ]
        return out.assign(
            normalization="TPM_clean",
            expression=out["TPM_clean_median"],
            q1=out["TPM_clean_q1"],
            q3=out["TPM_clean_q3"],
        )

    monkeypatch.setattr(
        expression_accessors,
        "cancer_reference_expression",
        fake_delegated_reference,
    )

    mm = cancer_expression("MM", genes=["FAKE2"])
    assert mm["expression"].tolist() == [17.0]

    wide = cancer_reference_expression(
        cancer_types=["CLL", "MM"],
        genes=["NO_SUCH_GENE"],
        format="wide",
    )
    assert wide.empty
    assert list(wide.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "CLL_TPM_clean",
        "MM_TPM_clean",
    ]


def test_reference_expression_delegates_to_oncoref_without_fallback(monkeypatch):
    import oncoref

    calls = []

    def fake_oncoref(cancer_types=None, **kwargs):
        calls.append((cancer_types, kwargs))
        normalization = kwargs["normalize"]
        source_label = "tpm_raw" if normalization == "tpm" else normalization
        offset = 0.0 if normalization == "tpm" else 4.0
        rows = []
        for code, value in (("CLL", 7.0), ("MM", 13.0)):
            rows.append({
                "Ensembl_Gene_ID": f"ENSG_{code}",
                "Symbol": f"FAKE_{code}",
                "Proteoform_ID": f"ENSG_{code}",
                "Member_Ensembl_Gene_IDs": f"ENSG_{code}",
                "cancer_code": code,
                "source_cohort": f"FAKE_{code}",
                "source_project": "fake",
                "source_version": "test",
                "n_samples": 3,
                "n_detected": 3,
                "processing_pipeline": "test",
                "notes": "",
                "normalization": source_label,
                "expression": value + offset,
                "q1": value + offset - 1,
                "q3": value + offset + 1,
            })
        out = pd.DataFrame(rows)
        out.attrs["availability"] = [
            {
                "cancer_code": code,
                "normalization": source_label,
                "available": True,
            }
            for code in ("CLL", "MM")
        ]
        out.attrs["reference_source"] = "summary_rows_all"
        return out

    monkeypatch.setattr(oncoref, "cancer_reference_expression", fake_oncoref)

    df = cancer_reference_expression(
        cancer_types=["CLL", "MM"],
        genes=["FAKE"],
        normalize=["tpm_log1p", "tpm_clean"],
        source_kind="geo",
        source_cohort=["FAKE_CLL", "FAKE_MM"],
        exclude_microarray_proxy=True,
        collapse_cdna_identical=True,
    )

    assert len(calls) == 2
    assert {kwargs["normalize"] for _, kwargs in calls} == {"tpm", "tpm_clean"}
    for codes, kwargs in calls:
        assert codes == ["CLL", "MM"]
        assert kwargs["sample_qc"] == "all"
        assert kwargs["reference_source"] == "summary_rows_all"
        assert kwargs["gene_id_style"] == "pirlygenes"
        assert kwargs["gene_universe"] == "pirlygenes"
        assert kwargs["on_missing"] == "empty"
        # Pirlygenes owns the compatibility kind map, so kinds are resolved to
        # exact cohort IDs before oncoref filters/pools. These deliberately fake
        # cohorts are absent from that registry and therefore become an exact
        # nonmatching filter rather than leaking through as unfiltered rows.
        assert kwargs["source_kind"] is None
        assert kwargs["source_cohort"] == [
            expression_accessors._EMPTY_REFERENCE_SOURCE_COHORT
        ]
        assert kwargs["exclude_microarray_proxy"] is True
        assert kwargs["collapse_cdna_identical"] is True

    assert set(df["normalization"]) == {"TPM_log1p", "TPM_clean"}
    raw_log = df[df["normalization"] == "TPM_log1p"].set_index("cancer_code")
    assert raw_log.loc["CLL", "expression"] == pytest.approx(np.log1p(7.0))
    assert raw_log.loc["MM", "q3"] == pytest.approx(np.log1p(14.0))
    assert df.attrs["reference_backend"] == "oncoref"
    assert df.attrs["reference_source"] == "summary_rows_all"


@pytest.mark.parametrize(
    ("cancer_type", "gene", "expected_codes"),
    [
        ("LUAD", "TP53", {"LUAD"}),
        ("CLL", "MS4A1", {"CLL"}),
        ("SARC_DDLPS", "MDM2", {"SARC_DDLPS"}),
        ("BRCA_Basal", "ESR1", {"BRCA_Basal"}),
        ("CRC", "TP53", {"COAD", "READ"}),
    ],
)
def test_reference_expression_oncoref_parity_surfaces(
    cancer_type,
    gene,
    expected_codes,
):
    df = cancer_reference_expression(
        cancer_type,
        genes=[gene],
        normalize=["tpm", "tpm_clean"],
    )
    assert not df.empty
    assert set(df["cancer_code"].astype(str)) == expected_codes
    assert set(df["normalization"].astype(str)) == {"TPM", "TPM_clean"}
    assert df.attrs["reference_backend"] == "oncoref"


def test_cancer_expression_empty_gene_subset_is_uniform_across_sources():
    expected_cols = ["Ensembl_Gene_ID", "Symbol", "expression"]

    cll = cancer_expression("CLL", genes=["NO_SUCH_GENE"])
    assert cll.empty
    assert list(cll.columns) == expected_cols

    mm = cancer_expression("MM", genes=["NO_SUCH_GENE"])
    assert mm.empty
    assert list(mm.columns) == expected_cols

    prad = cancer_expression("PRAD", genes=["NO_SUCH_GENE"])
    assert prad.empty
    assert list(prad.columns) == expected_cols


def test_cancer_expression_uses_parent_reference_for_child_labels():
    # PCN (Solitary Plasmacytoma) has no direct reference cohort and
    # should fall back to its parent MM (CoMMpass). This is the
    # canonical parent-fallback test.
    pcn = cancer_expression("PCN", genes=["TNFRSF17"])
    mm = cancer_expression("MM", genes=["TNFRSF17"])
    assert pcn.reset_index(drop=True).equals(mm.reset_index(drop=True))


def test_cancer_expression_prefers_direct_reference_over_parent():
    # BRCA_Basal has its own direct reference cohort
    # (TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50, from the PAM50 split
    # landed in commit f358f5e). It should NOT fall back to the BRCA
    # umbrella — that would dilute the basal-specific signal with
    # luminal/HER2/etc. samples. ERBB2 expression differs sharply
    # between basal (~60 TPM) and the BRCA umbrella average (~135
    # TPM, driven by HER2-enriched cases).
    basal = cancer_expression("BRCA_Basal", genes=["ERBB2"])
    parent = cancer_expression("BRCA", genes=["ERBB2"])
    assert not basal.reset_index(drop=True).equals(
        parent.reset_index(drop=True)
    ), (
        "BRCA_Basal should use its direct PAM50 reference, not fall "
        "back to the BRCA umbrella."
    )


def test_cancer_expression_source_candidates_cover_requested_gaps():
    requested = {
        "BRCA_Basal",
        "BRCA_HER2",
        "BRCA_LumA",
        "BRCA_LumB",
        "BRCA_Normal",
        "HNSC_HPVpos",
        "HNSC_HPVneg",
        "LUAD_EGFR",
        "LUAD_KRAS",
        "LUAD_STK11",
        "MBL_G3",
        "MBL_G4",
        "MBL_SHH",
        "MBL_WNT",
        "SCLC_ASCL1",
        "SCLC_NEUROD1",
        "SCLC_POU2F3",
        "SCLC_YAP1",
        "FL",
        "HCL",
        "HL",
        "PCN",
        "NET_LUNG",
        "NEC_LUNG_LARGECELL",
        "NET_MIDGUT",
        "MTC",
        "NPC",
        "NEC_MERKEL",
        "ADCC",
        "ACINIC",
        "SARC_ESS_HG",
        "SARC_ESS_LG",
        "SARC_GCTB",
        "SARC_ANGIO",
        "SARC_ASPS",
        "SARC_CCS",
        "SARC_DFSP",
        "SARC_DSRCT",
        "SARC_EHE",
        "SARC_EMC",
        "SARC_EPITH",
        "SARC_GIST",
        "SARC_IFS",
        "SARC_IMT",
        "SARC_KS",
        "SARC_MPNST",
        "SARC_MYXLPS",
        "SARC_PEC",
        "SARC_SFT",
        "SARC_WDLPS",
    }
    df = cancer_expression_source_candidates()
    required_cols = {
        "cancer_code",
        "source_status",
        "reference_code",
        "source_project",
        "source_cohort",
        "accession",
        "source_url",
        "assay",
        "source_scope",
        "estimated_samples",
        "processing_plan",
        "gene_id_plan",
        "normalization_plan",
        "notes",
    }
    assert required_cols.issubset(df.columns)
    assert not (requested - set(df["cancer_code"]))
    assert df["source_status"].notna().all()
    assert df["processing_plan"].fillna("").str.len().gt(0).all()
    text_cols = [c for c in df.columns if c != "estimated_samples"]
    assert not df[text_cols].isna().any().any()

    ready = df[df["source_status"].str.contains("candidate_ready")]
    assert ready["source_url"].str.startswith("https://").all()
    assert ready["accession"].str.len().gt(0).all()


def test_cancer_expression_reference_status_is_uniform_for_parent_labels():
    status = cancer_expression_reference_status(
        ["BRCA_Basal", "PCN", "SARC_GIST", "CLL"],
    ).set_index("cancer_code")

    # BRCA_Basal has its own PAM50 direct-reference shard since
    # commit f358f5e — should NOT fall back to BRCA parent.
    assert status.loc["BRCA_Basal", "reference_status"] == "direct_reference"
    assert status.loc["BRCA_Basal", "reference_code"] == "BRCA_Basal"
    # PCN still has no direct cohort, falls back to MM.
    assert status.loc["PCN", "reference_status"] == "parent_reference"
    assert status.loc["PCN", "reference_code"] == "MM"
    # SARC_GIST has its own direct reference since commit e3cc372.
    assert status.loc["SARC_GIST", "reference_status"] == "direct_reference"
    assert status.loc["SARC_GIST", "reference_code"] == "SARC_GIST"
    assert status.loc["CLL", "reference_status"] == "direct_reference"


def test_cancer_expression_reference_status_avoids_full_reference(monkeypatch):
    def raise_if_used(*_args, **_kwargs):
        raise AssertionError("status loaded or rescanned the full reference")

    monkeypatch.setattr(
        expression_accessors,
        "_load_cancer_reference_expression",
        raise_if_used,
    )
    monkeypatch.setattr(
        expression_accessors,
        "_pan_expression_codes",
        lambda: {"BRCA", "SARC"},
    )
    monkeypatch.setattr(
        expression_accessors,
        "_has_cancer_reference",
        raise_if_used,
    )
    monkeypatch.setattr(
        expression_accessors,
        "_reference_cohort_summary",
        raise_if_used,
    )

    status = cancer_expression_reference_status().set_index("cancer_code")

    assert status.loc["CLL", "reference_status"] == "direct_reference"
    assert status.loc["PCN", "reference_code"] == "MM"
    assert status.loc["BRCA_Basal", "reference_code"] == "BRCA_Basal"


def test_cancer_reference_expression_cll_sample_manifest_tracks_exclusions():
    samples = load_all_dataframes_dict()["cancer-reference-expression-samples.csv"]
    samples = samples[samples["source_cohort"] == "CLLMAP_2022"]
    included = samples["included"].astype(str).str.lower().eq("true")
    excluded = samples[~included]
    assert len(samples) == 715
    assert len(excluded) == 7
    assert set(excluded["sample_id"]) == {
        "CRC-0007",
        "CRC-0011",
        "CRC-0028",
        "CRC-0033",
        "DFCI-5053",
        "JB-0010",
        "GCLL-0136",
    }
    assert (
        excluded.set_index("sample_id").loc["GCLL-0136", "exclusion_reason"]
        == "suspected_mcl"
    )


def test_cancer_reference_expression_mm_sample_manifest_tracks_exclusions():
    samples = load_all_dataframes_dict()["cancer-reference-expression-samples.csv"]
    samples = samples[samples["source_cohort"] == "MMRF_COMMPASS"]
    included = samples["included"].astype(str).str.lower().eq("true")
    excluded = samples[~included]
    assert len(samples) == 859
    assert included.sum() == 764
    assert len(excluded) == 95
    assert set(excluded["exclusion_reason"]) == {"not_primary_bm_cd138pos"}
    assert samples.loc[included, "sample_type"].eq(
        "Primary Blood Derived Cancer - Bone Marrow"
    ).all()
    assert samples.loc[included, "sample_id"].str.endswith("BM_CD138pos").all()
    assert set(samples.loc[included, "lineage_label"]) == {"MM"}


def test_cancer_reference_expression_target_all_sample_manifest_tracks_lineage():
    samples = load_all_dataframes_dict()["cancer-reference-expression-samples.csv"]
    samples = samples[samples["source_cohort"] == "TARGET_ALL_2018"]
    included = samples["included"].astype(str).str.lower().eq("true")
    excluded = samples[~included]

    assert len(samples) == 679
    assert included.sum() == 418
    assert samples.loc[included, "cancer_code"].value_counts().to_dict() == {
        "T_ALL": 264,
        "B_ALL": 154,
    }
    assert excluded["exclusion_reason"].value_counts().to_dict() == {
        "no_b_or_t_lineage": 181,
        "not_primary_diagnostic_blood_or_marrow": 69,
        "duplicate_primary_sample": 11,
    }
    assert set(samples.loc[included, "source_project_id"]) <= {
        "TARGET-ALL-P1",
        "TARGET-ALL-P2",
    }
    assert not samples.loc[included, "source_project_id"].eq("TARGET-ALL-P3").any()
    assert samples.loc[included, "sample_type"].isin(
        {
            "Primary Blood Derived Cancer - Bone Marrow",
            "Primary Blood Derived Cancer - Peripheral Blood",
        }
    ).all()
    assert set(samples.loc[included, "lineage_label"]) == {"B_ALL", "T_ALL"}


def test_cancer_reference_expression_heme_sample_manifest_tracks_inclusions():
    samples = load_all_dataframes_dict()["cancer-reference-expression-samples.csv"]
    samples = samples[samples["cancer_code"].isin(["BL", "CML", "MCL", "MDS", "MPN"])]
    included = samples["included"].astype(str).str.lower().eq("true")

    assert samples.loc[included, "cancer_code"].value_counts().to_dict() == {
        "BL": 184,
        "MDS": 82,
        "MCL": 51,
        "MPN": 45,
        "CML": 5,
    }

    excluded = samples[~included]
    assert not excluded.empty
    assert "healthy_control" in set(excluded["exclusion_reason"])
    assert "secondary_aml_not_chronic_phase_mpn" in set(
        excluded["exclusion_reason"]
    )
    assert "not_primary_burkitt_tumor" in set(excluded["exclusion_reason"])


def test_cancer_reference_expression_ctcl_scrna_manifest_tracks_clones():
    samples = load_all_dataframes_dict()["cancer-reference-expression-samples.csv"]
    samples = samples[samples["source_cohort"] == "GSE171811_ECCITE_CTCL"]
    included = samples["included"].astype(str).str.lower().eq("true")
    excluded = samples[~included]

    assert len(samples) == 14
    assert included.sum() == 12
    assert samples.loc[included, "case_id"].nunique() == 7
    assert set(samples.loc[included, "primary_diagnosis"]) == {"SS", "leukemic MF"}
    assert set(samples.loc[included, "raw_unit"]) == {"scRNA UMI counts"}
    assert excluded["sample_id"].tolist() == [
        "GSM5234576_HC1_Blood",
        "GSM5234577_HC1_Skin",
    ]
    assert set(excluded["exclusion_reason"]) == {"healthy_control"}
    assert samples.loc[included, "lineage_evidence_source"].str.contains(
        "dominant TCR beta clone",
    ).all()


# ---------- topiary call pattern ----------


def test_topiary_load_all_dataframes_dict_pattern_works():
    """Topiary's `pce = load_all_dataframes_dict()["pan-cancer-expression.csv"]`
    must keep working — that pattern broke between 5.0.0 and 5.0.2 when
    the CSVs were stripped. 5.1.0 restores it."""
    pce = load_all_dataframes_dict()["pan-cancer-expression.csv"]
    assert isinstance(pce, pd.DataFrame)
    assert not pce.empty
    assert "Ensembl_Gene_ID" in pce.columns


def test_matched_normal_marker_tables_are_packaged_and_filterable():
    dataframes = load_all_dataframes_dict()
    assert "tumor-up-vs-matched-normal.csv" in dataframes
    assert "heme-tumor-up-vs-matched-normal.csv" in dataframes

    os_markers = tumor_up_vs_matched_normal("SARC_OS")
    assert len(os_markers) == 10
    assert os_markers["cancer_code"].eq("SARC_OS").all()
    assert os_markers.iloc[0]["symbol"] == "COL1A2"
    assert os_markers.iloc[0]["ensembl_gene_id"] == "ENSG00000164692"
    assert os_markers.iloc[0]["fold_change_vs_matched_normal"] > 4000

    laml_markers = heme_tumor_up_vs_matched_normal("LAML")
    assert len(laml_markers) == 10
    assert laml_markers["cancer_code"].eq("LAML").all()
    assert {"tumor_tpm", "matched_normal_ntpm", "max_non_lymphoid_ntpm"} <= set(
        laml_markers.columns
    )


# ---------- rescaling primitives ----------


def test_renormalize_to_million_rescales_columns_to_sum_1e6():
    df = pd.DataFrame({
        "Symbol": ["A", "B", "C"],
        "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
        "TPM_S1": [10.0, 20.0, 30.0],
        "TPM_S2": [1.0, 2.0, 3.0],
    })
    out, _record = renormalize_to_million(df, value_cols=["TPM_S1", "TPM_S2"])
    for col in ("TPM_S1", "TPM_S2"):
        assert abs(out[col].sum() - 1_000_000) < 1e-3


def test_fpkm_to_tpm_round_trip_preserves_relative_ranks():
    df = pd.DataFrame({
        "Symbol": ["A", "B", "C"],
        "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
        "FPKM_S1": [1.0, 2.0, 3.0],
    })
    out, _record = fpkm_to_tpm(df, value_cols=["FPKM_S1"])
    # FPKM → TPM is monotonic per column, so rank-correlation is 1.
    assert (out["FPKM_S1"].rank() == df["FPKM_S1"].rank()).all()


def test_add_tpm_columns_from_fpkm_preserves_source_columns():
    df = pd.DataFrame({
        "Symbol": ["A", "B", "C"],
        "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
        "FPKM_S1": [1.0, 2.0, 3.0],
    })
    out, record = add_tpm_columns_from_fpkm(df)
    assert "FPKM_S1" in out.columns
    assert "TPM_S1" in out.columns
    assert out["FPKM_S1"].tolist() == [1.0, 2.0, 3.0]
    assert out["TPM_S1"].sum() == pytest.approx(1_000_000)
    assert record["columns"]["FPKM_S1"]["target_column"] == "TPM_S1"


def test_add_tpm_columns_from_fpkm_accepts_entity_first_columns():
    df = pd.DataFrame({
        "Symbol": ["A", "B", "C"],
        "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
        "BRCA_FPKM": [1.0, 2.0, 3.0],
    })
    out, record = add_tpm_columns_from_fpkm(df)
    assert "BRCA_FPKM" in out.columns
    assert "BRCA_TPM" in out.columns
    assert out["BRCA_FPKM"].tolist() == [1.0, 2.0, 3.0]
    assert out["BRCA_TPM"].sum() == pytest.approx(1_000_000)
    assert record["columns"]["BRCA_FPKM"]["target_column"] == "BRCA_TPM"


def test_percentile_rank_expression_is_reusable():
    df = pd.DataFrame({"TPM_S1": [10.0, 20.0, 30.0]})
    out, record = percentile_rank_expression(df, value_cols=["TPM_S1"])
    assert out["TPM_S1"].tolist() == pytest.approx([100 / 3, 200 / 3, 100])
    assert record["columns"]["TPM_S1"]["n_ranked"] == 3


def test_log1p_transform_is_reusable():
    df = pd.DataFrame({"TPM_S1": [0.0, 1.0, 9.0]})
    out = log1p_transform(df, value_cols=["TPM_S1"])
    assert out["TPM_S1"].tolist() == pytest.approx(np.log1p([0.0, 1.0, 9.0]))


def test_normalize_expression_drops_technical_rna_rows_and_renormalizes():
    df = pd.DataFrame({
        "Symbol": ["MT-CO1", "MALAT1", "MYC", "KLK3"],
        "Ensembl_Gene_ID": ["ENSG00000198804", "ENSG00000251562",
                            "ENSG00000136997", "ENSG00000142515"],
        "TPM_S1": [400_000.0, 100_000.0, 250_000.0, 250_000.0],
    })
    out, record = normalize_expression(df, value_cols=["TPM_S1"])
    # MT-CO1 and MALAT1 are technical-RNA; MYC and KLK3 survive.
    surviving = set(out.loc[out["TPM_S1"] > 0, "Symbol"])
    assert surviving == {"MYC", "KLK3"}
    # The surviving rows are renormalized back to the original total
    # (or close to it; small floating-point drift is fine).
    assert 999_999 < out["TPM_S1"].sum() < 1_000_001
    assert record["applied"] is True
    assert record["removed_technical_gene_count"] == 2
    assert "mt_dna" in record["remove_groups"]


def test_tpm_to_housekeeping_normalized_divides_by_size_factor():
    """tpm_to_housekeeping_normalized divides each column by its
    median-of-ratios housekeeping size factor. Result stays non-negative."""
    pce = pan_cancer_expression()
    fpkm_cols = [c for c in pce.columns if c.endswith("_FPKM")][:2]
    out, _record = tpm_to_housekeeping_normalized(pce, value_cols=fpkm_cols)
    for col in fpkm_cols:
        assert col in out.columns
    # Outputs are positive (or NaN), no negatives.
    for col in fpkm_cols:
        vals = out[col].dropna()
        if not vals.empty:
            assert (vals >= 0).all()


# ---------- classifier ----------


def test_classify_gene_qc_mt_dna_via_symbol():
    cls = classify_gene_qc("MT-CO1")
    assert isinstance(cls, GeneQcClass)
    assert cls.group == "mt_dna"


def test_classify_gene_qc_polya_bias_lncrna():
    assert classify_gene_qc("MALAT1").group == "polyadenylation_bias_lncrna"
    assert classify_gene_qc("NEAT1").group == "polyadenylation_bias_lncrna"


def test_classify_gene_qc_via_ensembl_id_fallback():
    """ENSG00000251562 is MALAT1; classify_gene_qc must resolve it via
    the pirlygenes.gene_families lookup even without the symbol."""
    cls = classify_gene_qc(symbol=None, ensembl_id="ENSG00000251562")
    assert cls.group == "polyadenylation_bias_lncrna"


def test_classify_gene_qc_protein_coding_returns_other():
    assert classify_gene_qc("MYC").group == "other"


def test_is_rescue_feature_true_for_mt_and_polyA_lncrnas():
    assert is_rescue_feature("MT-CO1") is True
    assert is_rescue_feature("MALAT1") is True
    assert is_rescue_feature("NEAT1") is True
    assert is_rescue_feature("MYC") is False


# ---------- filters / convenience ----------


def test_technical_rna_gene_ids_includes_mt_and_malat1():
    ids = technical_rna_gene_ids()
    assert "ENSG00000198804" in ids   # MT-CO1
    assert "ENSG00000251562" in ids   # MALAT1
    # Plenty of mt and rRNA pseudogenes — sanity floor.
    assert len(ids) > 500


def test_filter_technical_rna_removes_mt_rows_from_pan_cancer():
    df = pan_cancer_expression()
    n_total = len(df)
    out = filter_technical_rna(df)
    n_after = len(out)
    assert n_after < n_total
    # No surviving row should be an mtDNA ENSG.
    drop = technical_rna_gene_ids()
    assert not set(out["Ensembl_Gene_ID"]) & drop


def test_filter_to_genes_subsets_by_symbol_or_ensg():
    df = pan_cancer_expression()
    # Versioned ENSG queries match the canonical unversioned row.
    out = filter_to_genes(df, ["KLK3", "ENSG00000136997.17"])
    assert not out.empty
    syms = set(out["Symbol"].str.upper())
    assert {"KLK3", "MYC"} <= syms


def test_normalize_to_housekeeping_handles_explicit_value_cols():
    df = pan_cancer_expression()
    fpkm_cols = [c for c in df.columns if c.endswith("_FPKM")][:2]
    out = normalize_to_housekeeping(df, value_cols=fpkm_cols)
    expected, record = tpm_to_housekeeping_normalized(df, value_cols=fpkm_cols)
    assert record["applied"]
    pd.testing.assert_frame_equal(out, expected)
    # The non-rescaled columns should be untouched.
    other_col = next(c for c in df.columns if c.endswith("_nTPM"))
    pd.testing.assert_series_equal(
        df[other_col].reset_index(drop=True),
        out[other_col].reset_index(drop=True),
    )


def test_normalize_to_housekeeping_uses_ensembl_ids_without_symbol_column():
    from pirlygenes import housekeeping_gene_ids

    hk_ids = sorted(housekeeping_gene_ids())[:5]
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [*hk_ids, "ENSG00000141510"],
            "sample_TPM": [10.0, 10.0, 10.0, 10.0, 10.0, 100.0],
        }
    )

    out = normalize_to_housekeeping(df, value_cols=["sample_TPM"])
    _, record = tpm_to_housekeeping_normalized(df, value_cols=["sample_TPM"])

    assert record["applied"]
    assert record["panel"] == "pirlygenes_active_housekeeping"
    # median-of-ratios size factor = median(sample_tpm / reference_tpm) over
    # the detected HK genes; divide the non-panel gene (TP53) by it. Read the
    # denominator from the record rather than hardcoding, so the assertion
    # tracks the reference profile instead of the old geomean(10 + 0.1).
    denom = record["columns"]["sample_TPM"]["denominator"]
    assert out["sample_TPM"].iloc[-1] == pytest.approx(100.0 / denom)


def test_normalize_to_housekeeping_blanks_columns_with_no_panel_genes():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "sample_TPM": [100.0],
        }
    )

    out = normalize_to_housekeeping(df, value_cols=["sample_TPM"])

    assert out["sample_TPM"].isna().all()


def test_normalize_to_housekeeping_blanks_columns_with_all_zero_panel():
    from pirlygenes import housekeeping_gene_ids

    hk_ids = sorted(housekeeping_gene_ids())[:5]
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": [*hk_ids, "ENSG00000141510"],
            "sample_TPM": [0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
        }
    )

    out = normalize_to_housekeeping(df, value_cols=["sample_TPM"])

    assert out["sample_TPM"].isna().all()


def test_log2_transform_idempotent_on_zero_pseudocount_input():
    df = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG1", "ENSG2"],
        "FPKM_X": [0.0, 1.0],
    })
    out = log2_transform(df, value_cols=["FPKM_X"], pseudocount=1.0)
    # log2(0+1)=0, log2(1+1)=1
    assert out["FPKM_X"].iloc[0] == pytest.approx(0.0)
    assert out["FPKM_X"].iloc[1] == pytest.approx(1.0)


# ---------- aggregate (transcript → gene) ----------


def test_aggregate_gene_expression_sums_transcripts_to_genes():
    df = pd.DataFrame({
        "transcript_id": ["ENST1.1", "ENST1.2", "ENST2.1"],
        "TPM": [10.0, 5.0, 8.0],
    })
    tx_to_gene = {"ENST1.1": "GENEA", "ENST1.2": "GENEA", "ENST2.1": "GENEB"}
    out = aggregate_gene_expression(df, tx_to_gene_name=tx_to_gene)
    out_indexed = out.set_index("gene")
    assert out_indexed.loc["GENEA", "TPM"] == pytest.approx(15.0)
    assert out_indexed.loc["GENEB", "TPM"] == pytest.approx(8.0)


# ---------- pipeline ordering inside accessor kwargs ----------


def test_accessor_pipeline_drops_technical_rna_before_gene_subset():
    """The family filter must run before the gene-list subset.

    Otherwise ``pan_cancer_expression(genes=["MT-CO1", ...],
    drop_technical_rna=True)`` would happily keep MT-CO1 because it
    matches the explicit gene list. Lock the order in so a future
    reorder surfaces here rather than as silent mis-ranking
    downstream.
    """
    df = pan_cancer_expression(
        genes=["MT-CO1", "MYC", "KLK3"],
        drop_technical_rna=True,
    )
    syms = set(df["Symbol"].str.upper())
    assert "MT-CO1" not in syms
    assert {"MYC", "KLK3"} <= syms


def test_accessor_pipeline_applies_log_after_normalize():
    """log_transform runs after normalize.

    The logged column must equal ``log2(normalized_linear + 1)`` elementwise.
    An order swap (log-then-normalize) would log the raw TPM first and then
    divide by the housekeeping size factor, so the logged values would no
    longer be a pure log of the linear normalized values. Method-agnostic:
    this holds under median-of-ratios exactly as it did under the old geomean.
    """
    linear = pan_cancer_expression(normalize="hk", log_transform=False)
    logged = pan_cancer_expression(normalize="hk", log_transform=True)

    tpm_col = next(c for c in logged.columns if c.endswith("_TPM_hk"))
    lin = linear.set_index("Ensembl_Gene_ID")[tpm_col].astype(float)
    log = logged.set_index("Ensembl_Gene_ID")[tpm_col].astype(float)
    common = lin.index.intersection(log.index)
    max_diff = (log.loc[common] - np.log2(lin.loc[common] + 1.0)).abs().max()
    assert max_diff == pytest.approx(0.0, abs=1e-9), (
        f"logged column is not log2(normalized + 1); max diff {max_diff}"
    )


# ---------- normalize_expression: noncoding biotype path ----------


def test_normalize_expression_remove_noncoding_with_biotype_column():
    """``remove_noncoding=True`` drops both technical-RNA rows and rows
    whose biotype falls outside the protein-coding / Ig / TCR keep-list.

    The two filter paths run together: MALAT1 goes via the technical-
    RNA family (polyadenylation-bias lncRNA), and LINC123 goes via
    the biotype gate. MYC survives both. After dropping, the kept
    column total is renormalized back to the original 1e6.
    """
    df = pd.DataFrame({
        "Symbol": ["MYC", "MALAT1", "LINC123"],
        "Ensembl_Gene_ID": [
            "ENSG00000136997",   # MYC
            "ENSG00000251562",   # MALAT1 — caught as technical RNA
            "ENSG_LINC123",      # placeholder; dropped via biotype
        ],
        "biotype": ["protein_coding", "lincRNA", "antisense"],
        "TPM_S1": [400_000.0, 300_000.0, 300_000.0],
    })
    out, record = normalize_expression(
        df, value_cols=["TPM_S1"], remove_noncoding=True,
    )

    surviving = set(out.loc[out["TPM_S1"] > 0, "Symbol"])
    assert surviving == {"MYC"}
    # MALAT1 is caught by the technical-RNA family filter. The biotype
    # gate also catches it (lincRNA isn't on the keep-list), so its
    # row is counted in both ``removed_technical_gene_count`` and
    # ``removed_noncoding_gene_count`` — overlap is expected. LINC123
    # is biotype-only.
    assert record["removed_technical_gene_count"] == 1
    assert record["removed_noncoding_gene_count"] == 2
    assert out["TPM_S1"].sum() == pytest.approx(1_000_000)


# ---------- normalize= preset on the accessors ----------


def test_pan_cancer_expression_normalize_default_is_tpm_clean():
    """The default is the clean TPM analysis view."""
    df = pan_cancer_expression()
    assert any(c.endswith("_FPKM") for c in df.columns)
    assert any(c.endswith("_TPM") for c in df.columns)
    assert any(c.endswith("_nTPM") for c in df.columns)
    assert any(c.endswith("_TPM_clean") for c in df.columns)
    assert any(c.endswith("_nTPM_clean") for c in df.columns)
    assert not any(c.startswith("tcga_") for c in df.columns)
    mt_mask = df["Symbol"].astype(str).str.startswith("MT-")
    value_cols = [
        c for c in df.columns
        if c.endswith(("_TPM_clean", "_nTPM_clean"))
    ]
    # clean TPM is the fixed_fraction 16/9/75 contract (identical to
    # cancer_reference_expression): technical RNA is PINNED to ~9%, not zeroed.
    for col in value_cols:
        assert df.loc[mt_mask, col].astype(float).sum() > 0


def test_pan_cancer_expression_normalize_none_keeps_raw_and_tpm_columns():
    """``normalize=None`` keeps provenance FPKM and adds TPM analysis columns."""
    df = pan_cancer_expression(normalize=None)
    assert any(c.endswith("_FPKM") for c in df.columns)
    assert any(c.endswith("_TPM") for c in df.columns)
    assert any(c.endswith("_nTPM") for c in df.columns)
    assert not any(
        c.endswith(("_TPM_clean", "_nTPM_clean", "_TPM_hk", "_nTPM_hk"))
        for c in df.columns
    )
    assert not any(c.startswith("tcga_") for c in df.columns)


def test_pan_cancer_expression_normalize_none_log_transform_uses_tpm_values():
    """The raw log path transforms TPM/nTPM analysis columns, not FPKM."""
    raw = pan_cancer_expression(normalize=None)
    logged = pan_cancer_expression(normalize=None, log_transform=True)

    fpkm_col = next(c for c in raw.columns if c.endswith("_FPKM"))
    tpm_col = next(c for c in raw.columns if c.endswith("_TPM"))
    ntpm_col = next(c for c in raw.columns if c.endswith("_nTPM"))

    pd.testing.assert_series_equal(
        raw[fpkm_col].reset_index(drop=True),
        logged[fpkm_col].reset_index(drop=True),
        check_names=False,
    )
    assert logged[tpm_col].iloc[:20].tolist() == pytest.approx(
        np.log2(raw[tpm_col].astype(float).iloc[:20] + 1.0)
    )
    assert logged[ntpm_col].iloc[:20].tolist() == pytest.approx(
        np.log2(raw[ntpm_col].astype(float).iloc[:20] + 1.0)
    )


def test_pan_cancer_expression_normalize_tpm_preserves_fpkm_and_adds_tpm():
    """``normalize="tpm"`` is an explicit alias for TPM companions."""
    df = pan_cancer_expression(normalize="tpm")
    assert any(c.endswith("_FPKM") for c in df.columns)
    assert any(c.endswith("_TPM") for c in df.columns)
    assert any(c.endswith("_nTPM") for c in df.columns)
    assert not any(
        c.endswith(("_TPM_clean", "_nTPM_clean", "_TPM_hk", "_nTPM_hk"))
        for c in df.columns
    )
    assert not any(c.startswith("tcga_") for c in df.columns)


def test_pan_cancer_expression_normalize_uppercase_tpm_adds_tpm():
    """``normalize="TPM"`` is accepted as an alias for ``"tpm"``."""
    lower = pan_cancer_expression(normalize="tpm")
    upper = pan_cancer_expression(normalize="TPM")
    assert list(lower.columns) == list(upper.columns)


def test_pan_cancer_expression_normalize_tpm_rescales_fpkm_to_million():
    """After ``normalize="tpm"`` each former FPKM column sums to 10⁶."""
    df = pan_cancer_expression(normalize="tpm")
    # Computed rollups (BTC/CRC/NET/NSCLC/SGC) are TPM-only, sample-weighted
    # cohort medians and therefore do not carry the per-sample sum-to-million
    # invariant. Check only deterministic companions with paired FPKM provenance.
    tpm_cols = [
        c for c in df.columns
        if c.endswith("_TPM") and f"{c[:-len('_TPM')]}_FPKM" in df.columns
    ]
    assert tpm_cols
    for col in tpm_cols:
        col_sum = float(pd.to_numeric(df[col], errors="coerce").sum())
        if col_sum > 0:
            assert col_sum == pytest.approx(1_000_000, rel=1e-6)


def test_pan_cancer_expression_normalize_percentile_keeps_native_names():
    """``normalize="percentile"`` adds percentile columns while leaving
    raw FPKM and TPM-scale provenance untouched."""
    raw = pan_cancer_expression(normalize="tpm")
    df = pan_cancer_expression(normalize="percentile")
    assert any(c.endswith("_FPKM") for c in df.columns)
    fpkm_col = next(c for c in raw.columns if c.endswith("_FPKM"))
    pd.testing.assert_series_equal(
        raw[fpkm_col].reset_index(drop=True),
        df[fpkm_col].reset_index(drop=True),
    )
    tpm_col = next(c for c in raw.columns if c.endswith("_TPM"))
    pd.testing.assert_series_equal(
        raw[tpm_col].reset_index(drop=True),
        df[tpm_col].reset_index(drop=True),
    )
    tpm_col = next(c for c in df.columns if c.endswith("_TPM_percentile"))
    vals = pd.to_numeric(df[tpm_col], errors="coerce").dropna()
    assert vals.min() >= 0
    assert vals.max() <= 100


def test_pan_cancer_expression_normalize_housekeeping_alias_works():
    df = pan_cancer_expression(normalize="housekeeping")
    assert not df.empty
    assert any(c.endswith("_TPM_hk") for c in df.columns)


def test_pan_cancer_expression_normalize_tpm_clean_fixed_fraction():
    """``normalize="tpm_clean"`` applies the ONE fixed_fraction 16/9/75 contract
    (identical to cancer_reference_expression): the censored block is PINNED, not
    zeroed — ribosomal proteins to ~16%, other technical RNA to ~9% — and biology
    fills the constant remaining ~75%."""
    from pirlygenes.load_dataset import get_data
    raw = pan_cancer_expression(normalize=None)
    df = pan_cancer_expression(normalize="tpm_clean")
    # Authoritative category split — the canonical censored-gene list (the same
    # source of truth the transform is built on), keyed by ENSG. Using the list
    # DIRECTLY (not the removal-mask helper the implementation also calls) makes
    # this an independent check of which genes land in each compartment.
    censored = get_data("clean-tpm-censored-genes")
    ribo_ids = set(censored.loc[censored["category"] == "ribosomal_protein",
                                "Ensembl_Gene_ID"].astype(str))
    tech_ids = set(censored.loc[censored["category"] == "technical",
                                "Ensembl_Gene_ID"].astype(str))
    ids = df["Ensembl_Gene_ID"].astype(str)
    ribo = ids.isin(ribo_ids).to_numpy()
    other = ids.isin(tech_ids).to_numpy()
    rem = ribo | other
    assert ribo.any() and other.any()
    value_cols = [
        c for c in df.columns
        if c.endswith(("_TPM_clean", "_nTPM_clean"))
    ]
    for col in value_cols:
        v = df[col].astype(float)
        total = float(v.sum())
        if total <= 0:
            continue
        assert v[other].sum() > 0                        # technical PINNED, not zeroed
        assert v[ribo].sum() / total == pytest.approx(0.16, abs=0.015)
        assert v[other].sum() / total == pytest.approx(0.09, abs=0.015)
        assert v[~rem].sum() / total == pytest.approx(0.75, abs=0.015)
    fpkm_col = next(c for c in df.columns if c.endswith("_FPKM"))
    pd.testing.assert_series_equal(
        raw[fpkm_col].reset_index(drop=True),
        df[fpkm_col].reset_index(drop=True),
    )


def test_pan_cancer_expression_normalize_tpm_clean_preserves_base_tpm_columns():
    """``tpm_clean`` keeps base TPM/nTPM values and adds clean companions."""
    raw = pan_cancer_expression(normalize="tpm")
    df = pan_cancer_expression(normalize="tpm_clean")

    tpm_col = next(c for c in raw.columns if c.endswith("_TPM"))
    ntpm_col = next(c for c in raw.columns if c.endswith("_nTPM"))
    for source_col, clean_col in (
        (tpm_col, tpm_col.replace("_TPM", "_TPM_clean", 1)),
        (ntpm_col, ntpm_col.replace("_nTPM", "_nTPM_clean", 1)),
    ):
        assert clean_col in df.columns
        pd.testing.assert_series_equal(
            raw[source_col].reset_index(drop=True),
            df[source_col].reset_index(drop=True),
            check_names=False,
        )


def test_pan_cancer_expression_normalize_tpm_clean_pins_cols_to_million():
    """After ``normalize="tpm_clean"`` every clean TPM/nTPM analysis column
    sums to 10⁶."""
    df = pan_cancer_expression(normalize="tpm_clean")
    value_cols = [
        c for c in df.columns
        if c.endswith(("_TPM_clean", "_nTPM_clean"))
    ]
    assert value_cols
    for col in value_cols:
        col_sum = float(pd.to_numeric(df[col], errors="coerce").sum())
        if col_sum > 0:
            assert col_sum == pytest.approx(1_000_000, rel=1e-6)


def test_pan_cancer_expression_normalize_tpm_log1p_adds_columns():
    raw = pan_cancer_expression(normalize="tpm")
    df = pan_cancer_expression(normalize="tpm_log1p")

    tpm_col = next(c for c in raw.columns if c.endswith("_TPM"))
    log_col = tpm_col.replace("_TPM", "_TPM_log1p", 1)
    assert log_col in df.columns
    assert df[log_col].iloc[:20].tolist() == pytest.approx(
        np.log1p(raw[tpm_col].astype(float).iloc[:20])
    )

    fpkm_col = next(c for c in raw.columns if c.endswith("_FPKM"))
    pd.testing.assert_series_equal(
        raw[fpkm_col].reset_index(drop=True),
        df[fpkm_col].reset_index(drop=True),
        check_names=False,
    )


def test_pan_cancer_expression_normalize_tpm_clean_log1p_adds_columns():
    clean = pan_cancer_expression(normalize="tpm_clean")
    df = pan_cancer_expression(normalize="tpm_clean_log1p")

    clean_col = next(c for c in clean.columns if c.endswith("_TPM_clean"))
    log_col = clean_col.replace("_TPM_clean", "_TPM_clean_log1p", 1)
    assert clean_col in df.columns
    assert log_col in df.columns
    assert df[log_col].iloc[:20].tolist() == pytest.approx(
        np.log1p(clean[clean_col].astype(float).iloc[:20])
    )


def test_pan_cancer_expression_normalize_default_matches_singleton_list():
    default = pan_cancer_expression()
    list_mode = pan_cancer_expression(normalize=["tpm_clean"])
    pd.testing.assert_frame_equal(default, list_mode)


def test_pan_cancer_expression_normalize_list_combines_modes():
    df = pan_cancer_expression(normalize=["tpm_clean", "hk", "percentile"])
    tumor_entities = {
        c[:-len("_TPM")] for c in df.columns if c.endswith("_TPM")
    }
    normal_entities = {
        c[:-len("_nTPM")] for c in df.columns if c.endswith("_nTPM")
    }
    assert {"BTC", "CRC", "NET", "NSCLC", "SGC"} <= tumor_entities

    # FPKM is optional provenance, but every tumor and normal entity has the
    # same requested analysis derivatives.
    for entity in tumor_entities:
        assert {
            f"{entity}_TPM_clean",
            f"{entity}_TPM_hk",
            f"{entity}_TPM_percentile",
        } <= set(df.columns)
    for entity in normal_entities:
        assert {
            f"{entity}_nTPM_clean",
            f"{entity}_nTPM_hk",
            f"{entity}_nTPM_percentile",
        } <= set(df.columns)


def test_pan_cancer_expression_normalize_rejects_invalid_token():
    with pytest.raises(ValueError, match="normalize must be None"):
        pan_cancer_expression(normalize="raw")


def test_pan_cancer_expression_normalize_rejects_invalid_list_token():
    with pytest.raises(ValueError, match="normalize must be None"):
        pan_cancer_expression(normalize=["tpm", "raw"])


def test_pan_cancer_expression_rejects_removed_normalization_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        pan_cancer_expression(normalization="hk")


def test_pan_cancer_expression_rejects_removed_legacy_kwargs():
    for kwargs in (
        {"technical_rna_normalize": True},
        {"remove_noncoding": True},
        {"renormalize_to_million": True},
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            pan_cancer_expression(**kwargs)


def test_pan_cancer_expression_rejects_removed_legacy_positional_kwargs():
    with pytest.raises(TypeError):
        pan_cancer_expression(None, None, False, True)


def test_pan_cancer_expression_proteoform_duality():
    """pan_cancer_expression carries the gene/proteoform bridge columns and can
    collapse identical loci in linear space (uniform with cancer_reference_expression)."""
    from pirlygenes.expression.accessors import pan_cancer_expression
    genes = ["CTAG1A", "CTAG1B", "PRAME"]
    base = pan_cancer_expression(genes=genes, normalize="tpm")
    assert {"Proteoform_ID", "Member_Ensembl_Gene_IDs"} <= set(base.columns)
    # gene view: CTAG1B bridges to its proteoform
    assert base.loc[base.Symbol == "CTAG1B", "Proteoform_ID"].iloc[0] == "CTAG1A/B"
    # collapse: CTAG1A + CTAG1B summed into one CTAG1A/B row, PRAME untouched
    coll = pan_cancer_expression(genes=genes, normalize="tpm",
                                 collapse_protein_identical=True)
    assert "CTAG1A/B" in set(coll.Symbol) and "CTAG1B" not in set(coll.Symbol)
    vcol = next(c for c in coll.columns if c.endswith("_TPM"))
    summed = coll.loc[coll.Symbol == "CTAG1A/B", vcol].iloc[0]
    parts = base.loc[base.Symbol.isin(["CTAG1A", "CTAG1B"]), vcol].sum()
    assert abs(float(summed) - float(parts)) < 1e-6
    # a member-symbol gene filter still hits the folded row when collapsing
    only = pan_cancer_expression(genes=["CTAG1B"], normalize="tpm",
                                 collapse_protein_identical=True)
    assert set(only.Symbol) == {"CTAG1A/B"}
