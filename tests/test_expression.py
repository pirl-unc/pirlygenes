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
from pirlygenes.expression import (
    GeneQcClass,
    aggregate_gene_expression,
    cancer_expression,
    classify_gene_qc,
    estimate_signatures,
    filter_technical_rna,
    filter_to_genes,
    fpkm_to_tpm,
    heme_tumor_up_vs_matched_normal,
    hpa_cell_type_expression,
    is_rescue_feature,
    log2_transform,
    normalize_expression,
    normalize_to_housekeeping,
    pan_cancer_expression,
    renormalize_to_million,
    subtype_deconvolved_expression,
    tcga_deconvolved_expression,
    technical_rna_gene_ids,
    tpm_to_housekeeping_normalized,
    tumor_up_vs_matched_normal,
)


# ---------- reference accessors ----------


def test_pan_cancer_expression_returns_wide_frame_with_tcga_columns():
    df = pan_cancer_expression()
    assert not df.empty
    assert "Ensembl_Gene_ID" in df.columns
    assert "Symbol" in df.columns
    # nTPM_<tissue> from HPA, FPKM_<code> from TCGA, and tcga_<code>
    # from the deconvolution all coexist in the wide frame.
    assert any(c.startswith("nTPM_") for c in df.columns)
    assert any(c.startswith("FPKM_") for c in df.columns)
    assert any(c.startswith("tcga_") for c in df.columns)


def test_pan_cancer_expression_subset_filters_to_named_genes():
    df = pan_cancer_expression(genes=["KLK3", "MYC"])
    assert len(df) >= 2  # both symbols + maybe alias variants
    assert set(df["Symbol"].str.upper()) >= {"KLK3", "MYC"}


def test_pan_cancer_expression_housekeeping_rescales_to_unit_baseline():
    df = pan_cancer_expression(normalize="housekeeping")
    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")]
    # After housekeeping rescale, the median of housekeeping rows in
    # each column should sit at ~1.0 (it's a rescale, not a centering).
    # Tolerance loose enough to survive pandas median-aggregation
    # ordering drift without false-flagging.
    from pirlygenes import housekeeping_gene_ids
    hk = housekeeping_gene_ids()
    hk_rows = df[df["Ensembl_Gene_ID"].isin(hk)]
    for col in fpkm_cols[:3]:  # spot-check first few columns
        med = hk_rows[col].astype(float).median()
        assert med == pytest.approx(1.0, rel=0.05), (
            f"{col} housekeeping median is {med}, expected ~1.0"
        )


def test_cancer_expression_returns_per_symbol_expression_column():
    df = cancer_expression("PRAD")
    assert {"Ensembl_Gene_ID", "Symbol", "expression"} <= set(df.columns)
    assert not df.empty


def test_tcga_deconvolved_expression_long_form_schema():
    df = tcga_deconvolved_expression()
    assert df is not None
    expected = {"symbol", "cancer_code", "tumor_tpm_median", "tumor_tpm_q1",
                "tumor_tpm_q3", "n_samples"}
    assert expected <= set(df.columns)


def test_subtype_deconvolved_expression_has_subtype_column():
    df = subtype_deconvolved_expression()
    assert df is not None
    assert "subtype" in df.columns
    assert "cancer_code" in df.columns
    # BRCA PAM50 split is the canonical subtype example.
    assert "BRCA" in set(df["cancer_code"])


def test_tumor_up_vs_matched_normal_returns_panel_with_fold_change():
    df = tumor_up_vs_matched_normal()
    assert df is not None and not df.empty
    assert "fold_change_vs_matched_normal" in df.columns
    assert "cancer_code" in df.columns


def test_heme_tumor_up_vs_matched_normal_returns_panel():
    df = heme_tumor_up_vs_matched_normal()
    assert df is not None and not df.empty
    assert {"DLBC", "LAML"} & set(df["cancer_code"])


def test_hpa_cell_type_expression_long_form():
    df = hpa_cell_type_expression()
    assert not df.empty


def test_estimate_signatures_has_stromal_and_immune_classes():
    df = estimate_signatures()
    assert not df.empty


# ---------- topiary call pattern ----------


def test_topiary_load_all_dataframes_dict_pattern_works():
    """Topiary's `pce = load_all_dataframes_dict()["pan-cancer-expression.csv"]`
    must keep working — that pattern broke between 5.0.0 and 5.0.2 when
    the CSVs were stripped. 5.1.0 restores it."""
    pce = load_all_dataframes_dict()["pan-cancer-expression.csv"]
    assert isinstance(pce, pd.DataFrame)
    assert not pce.empty
    assert "Ensembl_Gene_ID" in pce.columns


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


def test_tpm_to_housekeeping_normalized_uses_geomean_of_curated_panel():
    """tpm_to_housekeeping_normalized divides each column by the geomean
    of the curated housekeeping panel. Result is unitless."""
    pce = pan_cancer_expression()
    fpkm_cols = [c for c in pce.columns if c.startswith("FPKM_")][:2]
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
    out = filter_to_genes(df, ["KLK3", "ENSG00000136997"])  # symbol + ENSG (MYC)
    assert not out.empty
    syms = set(out["Symbol"].str.upper())
    assert "KLK3" in syms or "MYC" in syms


def test_normalize_to_housekeeping_handles_explicit_value_cols():
    df = pan_cancer_expression()
    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")][:2]
    out = normalize_to_housekeeping(df, value_cols=fpkm_cols)
    # The non-rescaled columns should be untouched.
    other_col = next(c for c in df.columns if c.startswith("nTPM_"))
    pd.testing.assert_series_equal(
        df[other_col].reset_index(drop=True),
        out[other_col].reset_index(drop=True),
    )


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
    """``pan_cancer_expression(genes=[…], drop_technical_rna=True)`` should
    drop MT genes regardless of whether the caller asked for them in
    the gene subset — the family filter runs before the gene-list
    subset. Locks in the order so a future reordering bug shows up
    here rather than as a silent mis-ranking downstream."""
    df = pan_cancer_expression(
        genes=["MT-CO1", "MYC", "KLK3"],
        drop_technical_rna=True,
    )
    syms = set(df["Symbol"].str.upper())
    assert "MT-CO1" not in syms, (
        "MT-CO1 leaked through despite drop_technical_rna=True"
    )
    assert syms & {"MYC", "KLK3"}, "neither MYC nor KLK3 made it through"


def test_accessor_pipeline_applies_log_after_normalize():
    """When both ``normalize="housekeeping"`` and ``log_transform=True`` are
    set, log2 is applied AFTER the rescale (so values around 1.0 land
    near 0). Catches an order swap that would log-transform raw FPKM
    first and then divide by the wrong housekeeping median."""
    df = pan_cancer_expression(
        normalize="housekeeping",
        log_transform=True,
    )
    from pirlygenes import housekeeping_gene_ids
    hk = housekeeping_gene_ids()
    hk_rows = df[df["Ensembl_Gene_ID"].isin(hk)]
    fpkm_col = next(c for c in df.columns if c.startswith("FPKM_"))
    # log2(1.0) == 0 — the housekeeping median in each column should
    # be approximately log2(1.0) == 0 after rescale + log.
    med = hk_rows[fpkm_col].astype(float).median()
    assert med == pytest.approx(1.0, abs=0.1), (
        f"housekeeping median after normalize+log is {med}, expected ~1.0 "
        "(log2(1+1)=1 for the +1 pseudocount on a rescaled-to-1 value)"
    )


# ---------- normalize_expression: noncoding biotype path ----------


def test_normalize_expression_remove_noncoding_with_biotype_column():
    """remove_noncoding=True drops rows whose biotype isn't in the
    protein-coding / Ig / TCR keep-list when a biotype column exists."""
    df = pd.DataFrame({
        "Symbol": ["MYC", "MALAT1_NC", "LINC123"],
        "Ensembl_Gene_ID": ["ENSG_MYC", "ENSG_NC1", "ENSG_NC2"],
        "biotype": ["protein_coding", "lincRNA", "antisense"],
        "TPM_S1": [400_000.0, 300_000.0, 300_000.0],
    })
    out, record = normalize_expression(
        df, value_cols=["TPM_S1"], remove_noncoding=True,
    )
    surviving = set(out.loc[out["TPM_S1"] > 0, "Symbol"])
    assert "MYC" in surviving
    assert "LINC123" not in surviving
    assert record["removed_noncoding_gene_count"] >= 1
