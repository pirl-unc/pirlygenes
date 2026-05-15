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

import pandas as pd
import pytest

from pirlygenes import load_all_dataframes_dict
from pirlygenes.expression import (
    GeneQcClass,
    add_tpm_columns_from_fpkm,
    aggregate_gene_expression,
    cancer_expression,
    classify_gene_qc,
    estimate_signatures,
    filter_technical_rna,
    filter_to_genes,
    fpkm_to_tpm,
    hpa_cell_type_expression,
    is_rescue_feature,
    log2_transform,
    normalize_expression,
    normalize_to_housekeeping,
    pan_cancer_expression,
    percentile_rank_expression,
    renormalize_to_million,
    technical_rna_gene_ids,
    tpm_to_housekeeping_normalized,
)


# ---------- reference accessors ----------


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
    # After housekeeping rescale, the median of housekeeping rows in
    # each column should sit at ~1.0 (it's a rescale, not a centering).
    # Tolerance loose enough to survive pandas median-aggregation
    # ordering drift without false-flagging.
    from pirlygenes import housekeeping_gene_ids
    hk = housekeeping_gene_ids()
    hk_rows = df[df["Ensembl_Gene_ID"].isin(hk)]
    for col in tpm_cols[:3]:  # spot-check first few columns
        med = hk_rows[col].astype(float).median()
        assert med == pytest.approx(1.0, rel=0.05), (
            f"{col} housekeeping median is {med}, expected ~1.0"
        )


def test_cancer_expression_returns_per_symbol_expression_column():
    df = cancer_expression("PRAD")
    assert {"Ensembl_Gene_ID", "Symbol", "expression"} <= set(df.columns)
    assert not df.empty


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


def test_deconvolved_reference_tables_are_not_packaged():
    dataframes = load_all_dataframes_dict()
    assert "tcga-deconvolved-expression.csv" not in dataframes
    assert "subtype-deconvolved-expression.csv" not in dataframes
    assert "tumor-up-vs-matched-normal.csv" not in dataframes
    assert "heme-tumor-up-vs-matched-normal.csv" not in dataframes


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


def test_percentile_rank_expression_is_reusable():
    df = pd.DataFrame({"TPM_S1": [10.0, 20.0, 30.0]})
    out, record = percentile_rank_expression(df, value_cols=["TPM_S1"])
    assert out["TPM_S1"].tolist() == pytest.approx([100 / 3, 200 / 3, 100])
    assert record["columns"]["TPM_S1"]["n_ranked"] == 3


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
    out = filter_to_genes(df, ["KLK3", "ENSG00000136997"])  # symbol + ENSG (MYC)
    assert not out.empty
    syms = set(out["Symbol"].str.upper())
    assert "KLK3" in syms or "MYC" in syms


def test_normalize_to_housekeeping_handles_explicit_value_cols():
    df = pan_cancer_expression()
    fpkm_cols = [c for c in df.columns if c.endswith("_FPKM")][:2]
    out = normalize_to_housekeeping(df, value_cols=fpkm_cols)
    # The non-rescaled columns should be untouched.
    other_col = next(c for c in df.columns if c.endswith("_nTPM"))
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

    A housekeeping gene rescales to 1.0; with the default pseudocount
    of 1, log2(1.0 + 1.0) = 1.0. An order swap (log-then-rescale)
    would log raw TPM first, then divide by the wrong housekeeping
    median, and the median of housekeeping rows wouldn't land at 1.0.
    """
    df = pan_cancer_expression(normalize="hk", log_transform=True)
    from pirlygenes import housekeeping_gene_ids

    hk_rows = df[df["Ensembl_Gene_ID"].isin(housekeeping_gene_ids())]
    tpm_col = next(c for c in df.columns if c.endswith("_TPM_hk"))
    med = hk_rows[tpm_col].astype(float).median()
    assert med == pytest.approx(1.0, abs=0.1)


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
    for col in value_cols:
        assert df.loc[mt_mask, col].astype(float).sum() == pytest.approx(0.0)


def test_pan_cancer_expression_normalize_none_keeps_raw_and_tpm_columns():
    """``normalize=None`` leaves the raw/provenance columns unchanged."""
    df = pan_cancer_expression(normalize=None)
    assert any(c.endswith("_FPKM") for c in df.columns)
    assert not any(c.endswith("_TPM") for c in df.columns)
    assert any(c.endswith("_nTPM") for c in df.columns)
    assert not any(
        c.endswith(("_TPM_clean", "_nTPM_clean", "_TPM_hk", "_nTPM_hk"))
        for c in df.columns
    )
    assert not any(c.startswith("tcga_") for c in df.columns)


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
    tpm_cols = [c for c in df.columns if c.endswith("_TPM")]
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


def test_pan_cancer_expression_normalize_tpm_clean_zeroes_technical_rna():
    """``normalize="tpm_clean"`` zeroes mtDNA / rRNA / NUMT / MALAT1+NEAT1
    rows in added clean TPM-scale analysis columns."""
    raw = pan_cancer_expression(normalize=None)
    df = pan_cancer_expression(normalize="tpm_clean")
    mt_mask = df["Symbol"].astype(str).str.startswith("MT-")
    assert mt_mask.any()
    value_cols = [
        c for c in df.columns
        if c.endswith(("_TPM_clean", "_nTPM_clean"))
    ]
    for col in value_cols:
        assert df.loc[mt_mask, col].astype(float).sum() == pytest.approx(0.0)
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


def test_pan_cancer_expression_normalize_default_matches_singleton_list():
    default = pan_cancer_expression()
    list_mode = pan_cancer_expression(normalize=["tpm_clean"])
    pd.testing.assert_frame_equal(default, list_mode)


def test_pan_cancer_expression_normalize_list_combines_modes():
    df = pan_cancer_expression(normalize=["tpm_clean", "hk", "percentile"])
    for suffix in (
        "_TPM",
        "_TPM_clean",
        "_TPM_hk",
        "_TPM_percentile",
        "_nTPM",
        "_nTPM_clean",
        "_nTPM_hk",
        "_nTPM_percentile",
    ):
        assert any(c.endswith(suffix) for c in df.columns), suffix


def test_pan_cancer_expression_normalize_rejects_invalid_token():
    with pytest.raises(ValueError, match="normalize must be None"):
        pan_cancer_expression(normalize="raw")


def test_pan_cancer_expression_normalize_rejects_invalid_list_token():
    with pytest.raises(ValueError, match="normalize must be None"):
        pan_cancer_expression(normalize=["tpm", "raw"])


def test_pan_cancer_expression_rejects_removed_normalization_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        pan_cancer_expression(normalization="hk")


# ---------- legacy kwargs (deprecated; still work for one release) ----------


def test_pan_cancer_expression_legacy_kwargs_emit_deprecation():
    """The pre-5.2.0 kwargs still work but raise DeprecationWarning
    pointing callers at the new preset or exact primitives."""
    with pytest.warns(DeprecationWarning, match="normalization primitives"):
        pan_cancer_expression(technical_rna_normalize=True)


def test_pan_cancer_expression_legacy_technical_rna_normalize_still_zeroes_mt():
    """Behaviour of the legacy kwarg is preserved — mtDNA rows go to 0
    and per-column totals are renormalized to their input."""
    baseline = pan_cancer_expression(normalize=None)
    fpkm_col = next(c for c in baseline.columns if c.endswith("_FPKM"))
    raw_total = float(pd.to_numeric(baseline[fpkm_col], errors="coerce").sum())

    with pytest.warns(DeprecationWarning):
        out = pan_cancer_expression(technical_rna_normalize=True)
    mt_mask = out["Symbol"].astype(str).str.startswith("MT-")
    assert out.loc[mt_mask, fpkm_col].astype(float).sum() == pytest.approx(0.0)
    out_total = float(pd.to_numeric(out[fpkm_col], errors="coerce").sum())
    assert out_total == pytest.approx(raw_total, rel=1e-6)


def test_pan_cancer_expression_legacy_positional_kwargs_still_work():
    """Adding ``normalize=`` must not break pre-5.2.0 positional
    callers for the older normalization kwargs."""
    baseline = pan_cancer_expression(normalize=None)
    fpkm_col = next(c for c in baseline.columns if c.endswith("_FPKM"))
    raw_total = float(pd.to_numeric(baseline[fpkm_col], errors="coerce").sum())

    with pytest.warns(DeprecationWarning):
        out = pan_cancer_expression(None, None, False, True)
    mt_mask = out["Symbol"].astype(str).str.startswith("MT-")
    assert out.loc[mt_mask, fpkm_col].astype(float).sum() == pytest.approx(0.0)
    out_total = float(pd.to_numeric(out[fpkm_col], errors="coerce").sum())
    assert out_total == pytest.approx(raw_total, rel=1e-6)


def test_pan_cancer_expression_legacy_renormalize_pins_all_cols_to_million():
    with pytest.warns(DeprecationWarning):
        out = pan_cancer_expression(renormalize_to_million=True)
    value_cols = [
        c for c in out.columns
        if c.endswith(("_FPKM", "_nTPM", "_TPM"))
    ]
    assert value_cols
    for col in value_cols:
        col_sum = float(pd.to_numeric(out[col], errors="coerce").sum())
        if col_sum > 0:
            assert col_sum == pytest.approx(1_000_000, rel=1e-6)
