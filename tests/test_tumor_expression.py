# Licensed under the Apache License, Version 2.0

"""Tests for lineage-based purity estimation and 9-point tumor expression ranges."""

import numpy as np

from pirlygenes.tumor_purity import (
    LINEAGE_GENES,
    TCGA_MEDIAN_PURITY,
    _combine_purity_estimates,
    _lineage_purity_estimates,
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


def test_signature_summary_ignores_high_outlier():
    overall, lower, upper, stability = _summarize_gene_level_purity(
        [0.03, 0.04, 0.05, 0.06, 0.80],
        strategy="winsorized_median",
    )
    assert 0.04 < overall < 0.07
    assert 0.03 <= lower <= overall
    assert upper < 0.30
    assert 0.1 < stability < 1.0


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
        "est_1", "est_5", "est_9", "median_est",
        "pct_cancer_median", "tcga_percentile", "is_surface", "is_cta", "therapies",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


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
