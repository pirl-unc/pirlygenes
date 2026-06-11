"""Heterogeneity-safe cross-cohort pooling primitive (roadmap #366 remainder).

Different source cohorts measure different gene sets; pooling them must compute
each gene's stats over only the samples whose cohort measured it, with a
per-gene ``n_available`` denominator — never inner-joining down to the common
gene set and never imputing off-panel genes to zero (missing != zero).

Every ``PooledCohorts`` result is a **gene-indexed** DataFrame, so gene identity
is by id label, never by row position.
"""

import numpy as np
import pandas as pd
import pytest

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
    # B/C measured only by cohort A -> NaN in cohort B's columns (never filled)
    assert aligned.loc["B", ["b1", "b2"]].isna().all()
    assert aligned.loc["C", ["b1", "b2"]].isna().all()
    # D measured only by cohort B -> NaN in cohort A's columns
    assert aligned.loc["D", ["a1", "a2", "a3"]].isna().all()
    # A measured by both -> no NaN
    assert aligned.loc["A"].notna().all()


def test_per_gene_stats_use_only_measuring_samples():
    _, summary = pool_cohort_samples([_cohort_a(), _cohort_b()])
    med = summary["TPM_median"]  # gene-indexed Series
    # A pooled over all 5 samples {10,20,30,40,50} -> 30
    assert med["A"] == 30.0
    # B only cohort A's {100,200,300} -> 200 (NOT diluted by cohort-B zeros)
    assert med["B"] == 200.0
    # D only cohort B's {7,9} -> 8
    assert med["D"] == 8.0


def test_availability_aware_counts():
    counts = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()]).counts()
    # n_samples is the pooled width (constant)
    assert (counts["n_samples"] == 5).all()
    # n_available is per-gene measured count
    assert counts.loc["A", "n_available"] == 5
    assert counts.loc["B", "n_available"] == 3
    assert counts.loc["C", "n_available"] == 3
    assert counts.loc["D", "n_available"] == 2
    # n_detected counts measured AND >0: gene C has two zeros among 3 measured
    assert counts.loc["C", "n_detected"] == 1
    # a not-measured cell is never counted as a (zero) detection
    assert counts.loc["B", "n_detected"] == 3
    assert counts.loc["D", "n_detected"] == 2


def test_std_is_nan_for_single_sample_gene():
    a = pd.DataFrame({"a1": [1.0]}, index=["A"])          # 1 sample
    b = pd.DataFrame({"b1": [2.0], "b2": [4.0]}, index=["B"])  # 2 samples, different gene
    _, summary = pool_cohort_samples([a, b])
    assert np.isnan(summary.loc["A", "TPM_std"])  # only 1 sample measured A
    assert summary.loc["B", "TPM_std"] == np.std([2.0, 4.0], ddof=1)


def test_missing_is_not_zero():
    """The whole point: a gene off one cohort's panel must not be pulled toward
    zero by that cohort's samples (the outer-join-and-fill bug)."""
    _, summary = pool_cohort_samples([_cohort_a(), _cohort_b()])
    # B's mean over only cohort A = (100+200+300)/3 = 200; a fill-zero pool over
    # all 5 would give (100+200+300+0+0)/5 = 120 (the wrong, deflated answer).
    assert summary.loc["B", "TPM_mean"] == 200.0
    assert summary.loc["B", "TPM_mean"] != 120.0


def test_empty_input():
    aligned, summary = pool_cohort_samples([])
    assert aligned.empty and summary.empty


def test_mask_is_membership_not_notna_measured_but_zero():
    """A measured-but-zero gene is measured=True (the mask comes from cohort
    panel membership, not from the value being non-NaN/non-zero)."""
    pool = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()])
    # gene C measured by cohort A (values incl zeros), not by cohort B
    assert pool.measured.loc["C", ["a1", "a2", "a3"]].all()
    assert not pool.measured.loc["C", ["b1", "b2"]].any()
    counts = pool.counts()
    assert counts.loc["C", "n_available"] == 3   # all three A samples measured C
    assert counts.loc["C", "n_detected"] == 1    # only one of them had C > 0


def test_mask_membership_vs_notna_on_dropout():
    """A measured-but-dropout cell (intra-cohort NaN) still counts in
    n_available (the cohort measured the gene) but is excluded from the stat.
    A notna-based mask would wrongly drop it from the denominator."""
    coh = pd.DataFrame({"s1": [4.0], "s2": [np.nan], "s3": [8.0]}, index=["G"])
    pool = PooledCohorts.from_cohorts([coh])
    # membership mask -> all 3 measured this gene
    assert pool.counts().loc["G", "n_available"] == 3
    # but the median/mean skip the dropout NaN -> over {4, 8}
    assert pool.stats().loc["G", "TPM_median"] == 6.0
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


def test_summary_is_gene_indexed_not_positional():
    """The single output surface is gene-indexed: looking up a gene's stats is
    by ENSG label, not by guessing its row position."""
    pool = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()])
    summary = pool.summary()
    assert list(summary.index) == ["A", "B", "C", "D"]
    assert summary.index.equals(pool.gene_index)
    # label lookup gives the right gene's stats (B measured only by cohort A)
    assert summary.loc["B", "TPM_median"] == 200.0
    assert summary.loc["B", "n_available"] == 3
    # stats + counts are both present, all label-aligned
    assert "TPM_p90" in summary.columns and "n_detected" in summary.columns


def test_genes_matched_by_id_never_by_position():
    """Scramble each cohort's gene row order and the per-gene pooled stats are
    unchanged — genes align by id label (concat/reindex), never by position."""
    a, b = _cohort_a(), _cohort_b()
    a_scrambled = a.loc[["C", "A", "B"]]            # reorder rows
    b_scrambled = b.loc[["D", "A"]]
    base = PooledCohorts.from_cohorts([a, b]).summary()
    scram = PooledCohorts.from_cohorts([a_scrambled, b_scrambled]).summary()
    pd.testing.assert_frame_equal(base, scram)      # identical, by id
    # cohorts disagreeing on a SHARED gene's row position is harmless too:
    a2 = a.loc[["B", "C", "A"]]                      # A's gene A now last
    pooled = PooledCohorts.from_cohorts([a2, b]).summary()
    assert pooled.loc["A", "TPM_median"] == base.loc["A", "TPM_median"]


def test_mask_value_axis_mismatch_is_rejected():
    """A hand-built pool whose mask doesn't share values' gene index/order is
    rejected — you cannot accidentally pair a mask with mis-ordered values."""
    vals = pd.DataFrame({"s1": [1.0, 2.0]}, index=["A", "B"])
    bad_mask = pd.DataFrame({"s1": [True, True]}, index=["B", "A"])  # reordered
    with pytest.raises(ValueError, match="gene index mismatch"):
        PooledCohorts(vals, bad_mask)


def test_per_cohort_availability_views():
    """Labelled cohorts expose is_measured[gene,cohort], n_measured_genes, and
    n_measured_samples (observed, dropout-aware) for correct per-gene weighting."""
    A = pd.DataFrame({"a1": [10., 100., 0.], "a2": [20., np.nan, 5.],
                      "a3": [30., 300., 0.]}, index=["A", "B", "C"])
    B = pd.DataFrame({"b1": [40., 7.], "b2": [50., 9.]}, index=["A", "D"])
    pool = PooledCohorts.from_cohorts({"cohortA": A, "cohortB": B})
    cm = pool.cohort_measured
    assert cm.loc["A", "cohortA"] and cm.loc["A", "cohortB"]        # in both
    assert cm.loc["B", "cohortA"] and not cm.loc["B", "cohortB"]    # A only
    assert cm.loc["D", "cohortB"] and not cm.loc["D", "cohortA"]    # B only
    assert pool.n_measured_genes.to_dict() == {"cohortA": 3, "cohortB": 2}
    nms = pool.n_measured_samples
    assert nms.loc["A", "cohortA"] == 3 and nms.loc["A", "cohortB"] == 2
    assert nms.loc["B", "cohortA"] == 2   # membership True but 1 sample dropped out
    assert nms.loc["C", "cohortB"] == 0   # cohortB never measured C


def test_per_cohort_views_require_labels():
    pool = PooledCohorts.from_cohorts([_cohort_a()])   # iterable -> auto-labelled
    assert pool.sample_cohort is not None               # auto-labelled cohort_0
    assert list(pool.n_measured_genes.index) == ["cohort_0"]


def test_functional_frontdoor_matches_class():
    am, summary = pool_cohort_samples([_cohort_a(), _cohort_b()])
    pool = PooledCohorts.from_cohorts([_cohort_a(), _cohort_b()])
    pd.testing.assert_frame_equal(am, pool.analysis_matrix)
    pd.testing.assert_frame_equal(summary, pool.summary())