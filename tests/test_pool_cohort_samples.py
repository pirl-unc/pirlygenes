"""Heterogeneity-safe cross-cohort pooling primitive (roadmap #366 remainder).

Different source cohorts measure different gene sets; pooling them must compute
each gene's stats over only the samples whose cohort measured it, with a
per-gene ``n_available`` denominator — never inner-joining down to the common
gene set and never imputing off-panel genes to zero (missing != zero).
"""

import numpy as np
import pandas as pd

from pirlygenes.expression.stats import (
    PooledCohorts,
    align_ragged_matrices,
    available_count_columns,
    pool_cohort_samples,
)


def _cohort_a():
    # measures genes A, B, C across 3 samples
    return pd.DataFrame(
        {"a1": [10.0, 100.0, 0.0], "a2": [20.0, 200.0, 5.0], "a3": [30.0, 300.0, 0.0]},
        index=["A", "B", "C"],
    )


def _cohort_b():
    # measures genes A, D only (NOT B, C; measures D which A doesn't) across 2 samples
    return pd.DataFrame(
        {"b1": [40.0, 7.0], "b2": [50.0, 9.0]},
        index=["A", "D"],
    )


def test_alignment_is_union_and_unmeasured_stays_nan():
    aligned = align_ragged_matrices([_cohort_a(), _cohort_b()])
    assert set(aligned.index) == {"A", "B", "C", "D"}  # union, not intersection
    assert list(aligned.columns) == ["a1", "a2", "a3", "b1", "b2"]
    # B/C measured only by cohort A -> NaN in cohort B's columns (never filled)
    assert aligned.loc["B", ["b1", "b2"]].isna().all()
    assert aligned.loc["C", ["b1", "b2"]].isna().all()
    # D measured only by cohort B -> NaN in cohort A's columns
    assert aligned.loc["D", ["a1", "a2", "a3"]].isna().all()
    # A measured by both -> no NaN
    assert aligned.loc["A"].notna().all()


def test_per_gene_stats_use_only_measuring_samples():
    _, stats = pool_cohort_samples([_cohort_a(), _cohort_b()])
    med = pd.Series(stats["TPM_median"], index=["A", "B", "C", "D"])
    # A pooled over all 5 samples {10,20,30,40,50} -> 30
    assert med["A"] == 30.0
    # B only cohort A's {100,200,300} -> 200 (NOT diluted by cohort-B zeros)
    assert med["B"] == 200.0
    # D only cohort B's {7,9} -> 8
    assert med["D"] == 8.0


def test_availability_aware_counts():
    counts = available_count_columns(align_ragged_matrices([_cohort_a(), _cohort_b()]))
    idx = ["A", "B", "C", "D"]
    n_samples = pd.Series(counts["n_samples"], index=idx)
    n_avail = pd.Series(counts["n_available"], index=idx)
    n_det = pd.Series(counts["n_detected"], index=idx)
    # n_samples is the pooled width (constant)
    assert (n_samples == 5).all()
    # n_available is per-gene measured count
    assert n_avail["A"] == 5 and n_avail["B"] == 3 and n_avail["C"] == 3 and n_avail["D"] == 2
    # n_detected counts measured AND >0: gene C has two zeros among 3 measured
    assert n_det["C"] == 1
    # a not-measured cell is never counted as a (zero) detection
    assert n_det["B"] == 3 and n_det["D"] == 2


def test_std_is_nan_for_single_sample_gene():
    a = pd.DataFrame({"a1": [1.0]}, index=["A"])          # 1 sample
    b = pd.DataFrame({"b1": [2.0], "b2": [4.0]}, index=["B"])  # 2 samples, different gene
    _, stats = pool_cohort_samples([a, b])
    std = pd.Series(stats["TPM_std"], index=["A", "B"])
    assert np.isnan(std["A"])        # only 1 sample measured A -> undefined std
    assert std["B"] == np.std([2.0, 4.0], ddof=1)


def test_missing_is_not_zero():
    """The whole point: a gene off one cohort's panel must not be pulled toward
    zero by that cohort's samples (the outer-join-and-fill bug)."""
    _, stats = pool_cohort_samples([_cohort_a(), _cohort_b()])
    mean = pd.Series(stats["TPM_mean"], index=["A", "B", "C", "D"])
    # B's mean over only cohort A = (100+200+300)/3 = 200; a fill-zero pool over
    # all 5 would give (100+200+300+0+0)/5 = 120 (the wrong, deflated answer).
    assert mean["B"] == 200.0
    assert mean["B"] != 120.0


def test_empty_input():
    aligned, stats = pool_cohort_samples([])
    assert aligned.empty and stats == {}


def test_mask_is_membership_not_notna_measured_but_zero():
    """A measured-but-zero gene is measured=True (the mask comes from cohort
    panel membership, not from the value being non-NaN/non-zero)."""
    pool = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()])
    # gene C measured by cohort A (values incl zeros) -> measured in a-cols,
    # not measured in b-cols
    assert pool.measured.loc["C", ["a1", "a2", "a3"]].all()
    assert not pool.measured.loc["C", ["b1", "b2"]].any()
    # the zeros are measured, so they ARE in n_available
    counts = pool.counts()
    n_avail = pd.Series(counts["n_available"], index=pool.values.index)
    assert n_avail["C"] == 3            # all three A samples measured C
    n_det = pd.Series(counts["n_detected"], index=pool.values.index)
    assert n_det["C"] == 1              # only one of them had C > 0


def test_mask_membership_vs_notna_on_dropout():
    """A measured-but-dropout cell (intra-cohort NaN) still counts in
    n_available (the cohort measured the gene) but is excluded from the stat.
    A notna-based mask would wrongly drop it from the denominator."""
    # cohort measures gene G in 3 samples but sample s2 dropped out (NaN)
    coh = pd.DataFrame({"s1": [4.0], "s2": [np.nan], "s3": [8.0]}, index=["G"])
    pool = PooledCohorts.from_cohorts([coh])
    counts = pool.counts()
    # membership mask -> all 3 measured this gene
    assert int(counts["n_available"][0]) == 3
    # but the median/mean skip the dropout NaN -> over {4, 8}
    stats = pool.stats()
    assert stats["TPM_median"][0] == 6.0
    # contrast: the notna-based degenerate helper would only see 2 available
    assert int(available_count_columns(pool.values)["n_available"][0]) == 2


def test_gene_rows_are_canonically_sorted_and_order_independent():
    """Union gene rows are sorted (lexical id) so the pool is reproducible
    regardless of input cohort order; sample columns stay grouped by cohort."""
    p_ab = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()])
    p_ba = PooledCohorts.from_cohorts([_cohort_b(), _cohort_a()])
    assert list(p_ab.values.index) == ["A", "B", "C", "D"]      # sorted union
    assert list(p_ab.values.index) == list(p_ba.values.index)   # order-independent
    # columns follow input order (grouped by cohort), not sorted
    assert list(p_ab.values.columns) == ["a1", "a2", "a3", "b1", "b2"]
    assert list(p_ba.values.columns) == ["b1", "b2", "a1", "a2", "a3"]


def test_centralized_helpers_agree_with_functional_frontdoor():
    pool = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()])
    am, summary = pool_cohort_samples([_cohort_a(), _cohort_b()])
    pd.testing.assert_frame_equal(am, pool.analysis_matrix)
    for key, arr in pool.summary().items():
        assert np.allclose(summary[key], arr, equal_nan=True)
