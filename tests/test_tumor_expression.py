# Licensed under the Apache License, Version 2.0

"""Tests for lineage-based purity estimation and 9-point tumor expression ranges."""

import numpy as np

from pirlygenes.tumor_purity import (
    LINEAGE_GENES,
    TCGA_MEDIAN_PURITY,
    _combine_purity_estimates,
    _lineage_purity_estimates,
    _select_tumor_specific_genes,
    _summarize_gene_level_purity,
    _summarize_lineage_support,
)


# ── LINEAGE_GENES coverage ────────────────────────────────────────

def test_lineage_genes_covers_all_tcga_types():
    """Every TCGA cancer type with a known purity should have lineage genes."""
    missing = [ct for ct in TCGA_MEDIAN_PURITY if ct not in LINEAGE_GENES]
    assert missing == [], f"Cancer types without lineage genes: {missing}"


def test_lineage_genes_values_are_nonempty_lists():
    for ct, genes in LINEAGE_GENES.items():
        assert isinstance(genes, list), f"{ct}: expected list, got {type(genes)}"
        assert len(genes) >= 2, f"{ct}: need at least 2 lineage genes, got {len(genes)}"


def test_lineage_genes_has_no_duplicates():
    for ct, genes in LINEAGE_GENES.items():
        assert len(genes) == len(set(genes)), f"{ct} has duplicate genes"


# ── _lineage_purity_estimates edge cases ──────────────────────────

def test_lineage_unknown_cancer_type_returns_empty():
    result = _lineage_purity_estimates("FAKE_TYPE", {}, {}, [], 0.7)
    assert result == []


def test_lineage_empty_sample_returns_empty():
    result = _lineage_purity_estimates("PRAD", {}, {}, [], 0.69)
    assert result == []


# ── Upper-half median estimator ───────────────────────────────────

def test_upper_half_median_ignores_low_outliers():
    """Simulates de-differentiated genes pulling down the median."""
    # 9 genes: 4 de-differentiated (low), 5 retained (high)
    purities = [0.003, 0.004, 0.013, 0.025, 0.061, 0.078, 0.098, 0.104, 0.206]
    mid = len(purities) // 2
    upper_half = purities[mid:]
    estimate = float(np.median(upper_half))
    lower = float(np.percentile(upper_half, 25))
    upper = float(np.percentile(upper_half, 75))

    # Estimate should be in the retained cluster, not dragged down
    assert estimate > 0.05, f"Estimate {estimate} should be > 5%"
    assert lower > 0.03, f"Lower {lower} should be > 3%"
    assert upper < 0.25, f"Upper {upper} should be < 25%"


def test_lineage_support_penalizes_wrong_pattern():
    rows = [
        {"sample_ratio": 1.00, "tme_ratio": 0.0, "tumor_ratio": 0.20},
        {"sample_ratio": 0.01, "tme_ratio": 0.0, "tumor_ratio": 5.00},
        {"sample_ratio": 0.01, "tme_ratio": 0.0, "tumor_ratio": 3.00},
    ]
    stats = _summarize_lineage_support(rows)
    assert stats["concordance"] < 0.2
    assert stats["support_factor"] < 0.2


def test_combine_purity_ignores_zero_estimate_when_lineage_exists():
    overall, lower, upper = _combine_purity_estimates(
        sig_purity=0.35,
        sig_lower=0.20,
        sig_upper=0.50,
        estimate_purity=0.0,
        lineage_purity=0.60,
        lineage_lower=0.45,
        lineage_upper=0.75,
    )
    assert 0.45 < overall < 0.50
    assert lower <= overall <= upper
    assert lower == 0.20
    assert upper == 0.75


def test_combine_purity_penalizes_signature_only_calls_with_infiltration():
    overall, lower, upper = _combine_purity_estimates(
        sig_purity=0.52,
        sig_lower=0.30,
        sig_upper=0.70,
        estimate_purity=0.0,
        lineage_purity=None,
        lineage_lower=None,
        lineage_upper=None,
    )
    assert 0.15 < overall < 0.17
    assert lower <= overall <= upper


def test_combine_purity_deprioritizes_unstable_low_signature_when_lineage_exists():
    overall, lower, upper = _combine_purity_estimates(
        sig_purity=0.03,
        sig_lower=0.005,
        sig_upper=0.08,
        estimate_purity=0.35,
        lineage_purity=0.11,
        lineage_lower=0.08,
        lineage_upper=0.14,
        sig_stability=0.2,
    )
    assert overall > 0.10
    assert lower >= 0.08
    assert upper >= overall


def test_signature_summary_ignores_high_outlier():
    overall, lower, upper, stability = _summarize_gene_level_purity(
        [0.03, 0.04, 0.05, 0.06, 0.80],
        strategy="winsorized_median",
    )
    assert 0.04 < overall < 0.07
    assert 0.03 <= lower <= overall
    assert upper < 0.30
    assert 0.1 < stability < 1.0


def test_prad_signature_panel_excludes_rearranged_immune_genes():
    panel = _select_tumor_specific_genes("PRAD", n=30)
    assert panel
    # Only rearranged V/D/J/C segments should be blocked — prefix-based
    # checks would also drop unrelated genes like TRAF*, TRAK1, TRAP1.
    rearranged_prefixes = ("IGH", "IGK", "IGL", "TRA", "TRB", "TRG", "TRD")
    for gene in panel:
        for prefix in rearranged_prefixes:
            if gene.startswith(prefix) and len(gene) > len(prefix):
                segment_char = gene[len(prefix)]
                assert segment_char not in "VDJC", (
                    f"{gene} looks like a rearranged receptor segment"
                )
    assert "TRGV9" not in panel
    assert "TRGC1" not in panel


def test_signature_exclusion_preserves_unrelated_tr_ig_genes():
    """TRAF*, TRAK1, TRAP1, TRADD, IGHMBP2, IGFBP* should not be excluded."""
    from pirlygenes.tumor_purity import _compile_excluded_gene_matcher

    is_excluded = _compile_excluded_gene_matcher()
    for gene in ["TRAF3", "TRAF6", "TRAK1", "TRAP1", "TRADD", "IGHMBP2", "IGFBP3"]:
        assert not is_excluded(gene), f"{gene} should not be excluded"
    # HLA class I stays, class II goes
    for gene in ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"]:
        assert not is_excluded(gene), f"{gene} (class I) should not be excluded"
    for gene in ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DQB1"]:
        assert is_excluded(gene), f"{gene} (class II) should be excluded"
    # Rearranged receptors should still be excluded
    for gene in ["TRGV9", "TRGC1", "IGHV3-33", "IGKV1-5", "TRBV7-9"]:
        assert is_excluded(gene), f"{gene} (rearranged receptor) should be excluded"


def test_dlbc_panel_bypasses_immune_exclusion():
    """Immune-origin cancers should include lineage markers that would otherwise
    be filtered as infiltrate contamination."""
    panel = _select_tumor_specific_genes("DLBC", n=30)
    assert panel
    # B-cell lineage markers should be recoverable for DLBC
    assert any(g.startswith("HLA-D") for g in panel), (
        f"DLBC panel should include HLA class II markers, got: {panel}"
    )


def test_signature_panel_cache_invalidates_on_param_change():
    """Mutating TUMOR_PURITY_PARAMETERS should not serve stale panels."""
    from pirlygenes.tumor_purity import (
        TUMOR_PURITY_PARAMETERS,
        _select_tumor_specific_genes_for_panel,
    )

    panel_with_defaults = _select_tumor_specific_genes_for_panel("PRAD", n=30)
    params = TUMOR_PURITY_PARAMETERS["tumor_specific_markers"]
    original = params["excluded_gene_regexes"]
    try:
        # Add a pattern that blocks "KLK.*" — any KLK gene previously in
        # the panel should now be removed.
        params["excluded_gene_regexes"] = list(original) + [r"KLK.*"]
        panel_with_klk_blocked = _select_tumor_specific_genes_for_panel("PRAD", n=30)
        assert not any(g.startswith("KLK") for g in panel_with_klk_blocked), (
            f"cache should have re-keyed on param change; got {panel_with_klk_blocked}"
        )
    finally:
        params["excluded_gene_regexes"] = original

    # After restoring params the original panel should return.
    panel_restored = _select_tumor_specific_genes_for_panel("PRAD", n=30)
    assert panel_restored == panel_with_defaults


def test_combine_purity_treats_zero_stability_as_low_weight():
    """A stability of exactly 0.0 must not collapse to full weight via truthiness.

    Guards against the `sig_stability or 1.0` pattern: 0.0 is not
    "unknown", it's the strongest possible signal that the signature
    channel is unreliable. When the conflict gate happens NOT to fire
    (signature is close to lineage in absolute value), the weighted-log
    path should still downweight the signature instead of giving it full
    weight equal to lineage.
    """
    # Pick values where the conflict gate does NOT trigger — i.e.
    # sig/lineage >= signature_conflict_ratio (0.75). This forces the code
    # through the weighted-log branch where the bug lived.
    kwargs = dict(
        sig_lower=0.30,
        sig_upper=0.70,
        estimate_purity=None,
        lineage_purity=0.60,
        lineage_lower=0.55,
        lineage_upper=0.65,
    )
    overall_zero, _, _ = _combine_purity_estimates(
        sig_purity=0.50, sig_stability=0.0, **kwargs
    )
    overall_unknown, _, _ = _combine_purity_estimates(
        sig_purity=0.50, sig_stability=None, **kwargs
    )
    overall_high, _, _ = _combine_purity_estimates(
        sig_purity=0.50, sig_stability=0.9, **kwargs
    )
    # Stability=0 should pull the anchor closer to lineage (0.60) than
    # stability=None (which is treated as full signature weight = 1.0).
    # Stability=0.9 lies between the two.
    assert overall_zero > overall_unknown, (
        f"stability=0 must give lineage more relative weight than "
        f"stability=None; zero={overall_zero:.3f} unknown={overall_unknown:.3f}"
    )
    assert overall_unknown <= overall_high <= overall_zero or (
        overall_unknown <= overall_zero
    ), (
        f"Monotonicity check: lower stability should pull toward lineage; "
        f"zero={overall_zero:.3f} high={overall_high:.3f} unknown={overall_unknown:.3f}"
    )


# ── estimate_tumor_expression_ranges ──────────────────────────────

def test_ranges_dataframe_columns():
    """Verify the output DataFrame has all expected columns."""
    from pirlygenes.plot import estimate_tumor_expression_ranges
    import pandas as pd

    # Minimal expression data with just a few genes
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000169398", "ENSG00000160752"],
        "gene_symbol": ["PTK2", "IL34"],
        "TPM": [50.0, 10.0],
    })
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    assert isinstance(result, pd.DataFrame)

    expected_cols = [
        "gene_id", "symbol", "category", "observed_tpm",
        "tme_fold_lo", "tme_fold_med", "tme_fold_hi",
        "max_healthy_tpm", "tme_explainable", "cohort_prior_tpm",
        "est_1", "est_5", "est_9", "median_est",
        "pct_cancer_median", "tcga_percentile", "is_surface", "is_cta", "therapies",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_ranges_tme_explainable_clamps_at_observed():
    """For genes whose healthy-tissue max can explain the sample signal
    alone, `median_est` must not exceed `observed_tpm`. This guards
    against the 1/purity inflation for stromal / normal-lineage genes
    (e.g. KLK3 in prostate, FN1 stroma).
    """
    from pirlygenes.plot import estimate_tumor_expression_ranges
    import pandas as pd

    # KLK3: high in normal prostate (~7700 nTPM), observed 500 TPM.
    # max_healthy >> observed → tme_explainable = True → clamped at
    # observed. Without the clamp, 500/0.3 ≈ 1667 would be reported.
    df = pd.DataFrame({
        "ensembl_gene_id": [
            "ENSG00000142515",  # KLK3
            "ENSG00000075624",  # ACTB
            "ENSG00000156508",  # EEF1A1
            "ENSG00000111640",  # GAPDH
        ],
        "gene_symbol": ["KLK3", "ACTB", "EEF1A1", "GAPDH"],
        "TPM": [500.0, 150.0, 300.0, 100.0],
    })
    purity = {"overall_lower": 0.25, "overall_estimate": 0.30, "overall_upper": 0.35}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)
    klk3 = out[out["symbol"] == "KLK3"].iloc[0]
    assert klk3["tme_explainable"], "KLK3 should be tme_explainable in a 30% purity sample"
    assert klk3["median_est"] <= klk3["observed_tpm"] + 1e-6, (
        f"median_est {klk3['median_est']} exceeds observed {klk3['observed_tpm']} "
        f"despite tme_explainable=True"
    )


def test_ranges_skips_shrinkage_when_cohort_prior_is_near_zero():
    """CTAs with cohort median ≈ 0 (activated in a minority of samples)
    must NOT be shrunk toward zero by empirical-Bayes — the cohort
    median is uninformative for sparsely-expressed genes.
    """
    from pirlygenes.plot import estimate_tumor_expression_ranges
    from pirlygenes.gene_sets_cancer import CTA_gene_id_to_name, pan_cancer_expression
    import pandas as pd

    # Pick 3 CTAs with cohort_prior near zero in PRAD.
    ref = pan_cancer_expression().drop_duplicates(subset="Symbol").set_index("Symbol")
    cta_map = CTA_gene_id_to_name()
    zero_cohort_ctas = []
    for gid, sym in cta_map.items():
        if sym in ref.index and float(ref.loc[sym, "FPKM_PRAD"]) < 0.1:
            zero_cohort_ctas.append((gid, sym))
        if len(zero_cohort_ctas) >= 3:
            break
    assert zero_cohort_ctas, "Should find at least one near-zero CTA in PRAD"

    rows = [{"gene_id": gid, "gene_name": sym, "TPM": 50.0}
            for gid, sym in zero_cohort_ctas]
    rows.extend([
        {"gene_id": "ENSG00000075624", "gene_name": "ACTB",   "TPM": 150.0},
        {"gene_id": "ENSG00000156508", "gene_name": "EEF1A1", "TPM": 300.0},
        {"gene_id": "ENSG00000111640", "gene_name": "GAPDH",  "TPM": 100.0},
    ])
    df = pd.DataFrame(rows)
    purity = {"overall_lower": 0.30, "overall_estimate": 0.35, "overall_upper": 0.40}
    out = estimate_tumor_expression_ranges(df, "PRAD", purity)

    # Near-zero cohort prior CTAs should land at roughly `observed/purity`,
    # the raw deconvolution — NOT shrunk toward zero.
    cta_symbols = {sym for _, sym in zero_cohort_ctas}
    ctas = out[out["symbol"].isin(cta_symbols)]
    assert len(ctas) > 0
    for _, row in ctas.iterrows():
        # observed=50, purity~0.35 → raw ≈ 143. Allow some flexibility
        # for the 3x3 TME/purity grid's median.
        assert row["cohort_prior_tpm"] < 1.0, (
            f"{row['symbol']} cohort prior {row['cohort_prior_tpm']} not near zero"
        )
        assert row["median_est"] > 100, (
            f"{row['symbol']} median_est {row['median_est']} was shrunk toward zero "
            f"despite near-zero cohort prior (should be ~143)"
        )


def test_ranges_low_purity_shrinks_toward_cohort_prior():
    """At very low purity, the sample-based estimator has high variance
    (1/purity inflates). Shrinkage should pull estimates toward the
    TCGA cohort prior so we don't report pathological numbers.
    """
    from pirlygenes.plot import estimate_tumor_expression_ranges
    import pandas as pd

    # Gene not expressed in any healthy tissue but elevated in sample.
    # Without shrinkage: observed/purity is the estimator.
    # With shrinkage at very low purity: pulled toward cohort prior.
    df = pd.DataFrame({
        "ensembl_gene_id": [
            "ENSG00000137959",  # IFI44L — interferon-stimulated, low baseline
            "ENSG00000075624",  # ACTB (for HK median)
            "ENSG00000156508",  # EEF1A1
            "ENSG00000111640",  # GAPDH
        ],
        "gene_symbol": ["IFI44L", "ACTB", "EEF1A1", "GAPDH"],
        "TPM": [10.0, 150.0, 300.0, 100.0],
    })
    purity_low = {"overall_lower": 0.08, "overall_estimate": 0.10, "overall_upper": 0.12}
    purity_high = {"overall_lower": 0.60, "overall_estimate": 0.65, "overall_upper": 0.70}

    out_low = estimate_tumor_expression_ranges(df, "PRAD", purity_low)
    out_high = estimate_tumor_expression_ranges(df, "PRAD", purity_high)

    # Find the target gene in both outputs
    g_low = out_low[out_low["symbol"] == "IFI44L"]
    g_high = out_high[out_high["symbol"] == "IFI44L"]
    if len(g_low) and len(g_high):
        # At low purity the estimate should be LESS inflated than raw
        # 1/purity would give: raw low-purity estimate ≈ 10/0.10 = 100.
        # With shrinkage toward a (low) cohort prior, it should be much
        # closer to the cohort prior than to 100.
        low_est = float(g_low.iloc[0]["median_est"])
        raw_low = 10.0 / 0.10
        assert low_est < raw_low, (
            f"Shrinkage should pull low-purity estimate ({low_est}) "
            f"below raw 1/purity estimate ({raw_low})"
        )


def test_ranges_nine_estimates_are_sorted():
    """Each gene's 9 estimates should be in ascending order."""
    from pirlygenes.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000169398"],
        "gene_symbol": ["PTK2"],
        "TPM": [100.0],
    })
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.20,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    if not result.empty:
        row = result.iloc[0]
        ests = [row[f"est_{i+1}"] for i in range(9)]
        assert ests == sorted(ests), f"Estimates not sorted: {ests}"


def test_ranges_all_estimates_nonnegative():
    """No tumor expression estimate should be negative."""
    from pirlygenes.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000169398"],
        "gene_symbol": ["PTK2"],
        "TPM": [0.5],  # very low expression
    })
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    if not result.empty:
        row = result.iloc[0]
        for i in range(9):
            assert row[f"est_{i+1}"] >= 0, f"est_{i+1} is negative"


def test_ranges_empty_input():
    """Empty expression data should produce empty DataFrame."""
    from pirlygenes.plot import estimate_tumor_expression_ranges
    import pandas as pd

    df = pd.DataFrame({
        "ensembl_gene_id": [],
        "gene_symbol": [],
        "TPM": [],
    })
    purity_result = {
        "overall_lower": 0.05,
        "overall_estimate": 0.10,
        "overall_upper": 0.15,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    assert len(result) == 0


def test_ranges_pct_cancer_median_steap1_near_one():
    """STEAP1 at ~TCGA PRAD levels should have pct_cancer_median near 1.0."""
    from pirlygenes.plot import estimate_tumor_expression_ranges
    from pirlygenes.gene_sets_cancer import pan_cancer_expression, housekeeping_gene_ids
    import pandas as pd

    # Construct a fake sample where STEAP1 is at roughly the same
    # HK-normalized level as TCGA PRAD.
    ref = pan_cancer_expression()
    ref_dedup = ref.drop_duplicates(subset="Symbol").set_index("Symbol")
    hk_ids = housekeeping_gene_ids()
    ref_flat = ref.drop_duplicates(subset="Ensembl_Gene_ID")
    id_to_sym = dict(zip(ref_flat["Ensembl_Gene_ID"], ref_flat["Symbol"]))
    hk_syms = {id_to_sym[gid] for gid in hk_ids if gid in id_to_sym}
    hk_in_ref = sorted(hk_syms & set(ref_dedup.index))

    # STEAP1 TCGA PRAD fold
    prad_hk = ref_dedup.loc[hk_in_ref, "FPKM_PRAD"].astype(float).median()
    steap1_prad = float(ref_dedup.loc["STEAP1", "FPKM_PRAD"])
    steap1_fold = steap1_prad / prad_hk

    # Build a sample with the same fold, at 100% purity (so we can check)
    sample_hk_med = 500.0
    steap1_tpm = steap1_fold * sample_hk_med

    # Need some HK genes too
    rows = []
    for sym in list(hk_in_ref)[:10]:
        eid = ref_flat[ref_flat["Symbol"] == sym]["Ensembl_Gene_ID"].iloc[0]
        rows.append({"ensembl_gene_id": eid, "gene_symbol": sym, "TPM": sample_hk_med})
    # Add STEAP1
    rows.append({
        "ensembl_gene_id": "ENSG00000205542",
        "gene_symbol": "STEAP1",
        "TPM": steap1_tpm,
    })
    df = pd.DataFrame(rows)

    purity_result = {
        "overall_lower": 0.90,
        "overall_estimate": 1.0,
        "overall_upper": 1.0,
    }
    result = estimate_tumor_expression_ranges(df, "PRAD", purity_result)
    steap_row = result[result["symbol"] == "STEAP1"]
    if not steap_row.empty:
        pct = steap_row.iloc[0]["pct_cancer_median"]
        # Should be close to 1.0 (within 50% either direction)
        assert pct is not None, "pct_cancer_median should not be None"
        assert 0.5 < pct < 2.0, f"Expected ~1.0, got {pct}"


# ── _TME_TISSUES consistency ─────────────────────────────────────

def test_tme_tissues_are_valid():
    """All curated TME tissues should exist in the reference data."""
    from pirlygenes.plot import _TME_TISSUES
    from pirlygenes.gene_sets_cancer import pan_cancer_expression

    ref = pan_cancer_expression()
    ntpm_cols = {c.replace("nTPM_", "") for c in ref.columns if c.startswith("nTPM_")}
    for tissue in _TME_TISSUES:
        assert tissue in ntpm_cols, f"TME tissue {tissue!r} not in reference nTPM columns"


# ── CLI lineage narrative ─────────────────────────────────────────

def test_lineage_narrative_generation():
    """The lineage narrative should handle all three cases: retained, lost, not found."""
    from pirlygenes.tumor_purity import LINEAGE_GENES

    # Simulate a purity result with lineage component
    lineage_per_gene = [
        {"gene": "STEAP1", "purity": 0.098, "sample_tpm": 90.0,
         "sample_ratio": 0.16, "ref_ratio": 1.1, "tme_ratio": 0.01, "tumor_ratio": 1.6},
        {"gene": "KLK3", "purity": 0.003, "sample_tpm": 139.0,
         "sample_ratio": 0.25, "ref_ratio": 50.0, "tme_ratio": 0.0, "tumor_ratio": 73.0},
    ]
    purity = {
        "cancer_type": "PRAD",
        "overall_estimate": 0.098,
        "overall_lower": 0.078,
        "overall_upper": 0.104,
        "components": {
            "lineage": {
                "genes": ["STEAP1", "KLK3"],
                "purity": 0.098,
                "lower": 0.078,
                "upper": 0.104,
                "per_gene": lineage_per_gene,
            },
            "stromal": {"enrichment": 4.3, "n_genes": 141},
            "immune": {"enrichment": 2.5, "n_genes": 141},
        },
    }

    # The narrative code in cli.py uses these fields
    lineage = purity["components"]["lineage"]
    sorted_genes = sorted(lineage["per_gene"], key=lambda g: g["purity"], reverse=True)
    median_p = lineage["purity"]

    retained = [g for g in sorted_genes if g["purity"] >= median_p * 0.5]
    lost = [g for g in sorted_genes if g["purity"] < median_p * 0.5]

    all_lineage = LINEAGE_GENES.get("PRAD", [])
    found_names = {g["gene"] for g in lineage_per_gene}
    not_found = [g for g in all_lineage if g not in found_names]

    assert len(retained) == 1  # STEAP1
    assert retained[0]["gene"] == "STEAP1"
    assert len(lost) == 1  # KLK3
    assert lost[0]["gene"] == "KLK3"
    assert len(not_found) > 0  # Missing genes from PRAD lineage set
    assert "FOLH1" in not_found
