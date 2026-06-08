"""Per-gene × cohort tail-weighted percentile vectors (#298)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from pirlygenes.expression import (
    available_percentile_cohorts,
    cohort_gene_percentiles,
)

# single source of truth for the breakpoints: the generator defines them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_cohort_gene_percentiles import BREAKPOINTS  # noqa: E402

_BP = [f"p{b}" for b in BREAKPOINTS]


def _skip_if_absent():
    if not available_percentile_cohorts():
        pytest.skip("percentile artifact not present in this checkout/cache")


def test_schema_and_26_breakpoints():
    _skip_if_absent()
    df = cohort_gene_percentiles("PRAD")
    assert {"Ensembl_Gene_ID", "Symbol"} <= set(df.columns)
    assert [c for c in df.columns if c not in ("Ensembl_Gene_ID", "Symbol")] == _BP
    assert len(df) > 10000


def test_breakpoints_are_monotone_nondecreasing():
    _skip_if_absent()
    df = cohort_gene_percentiles("PRAD")
    vals = df[_BP].to_numpy()
    # p0 <= p1 <= ... <= p100 for every gene (allow tiny float16 round noise)
    diffs = np.diff(vals, axis=1)
    assert (diffs >= -1.0).all()


def test_as_tpm_vs_log1p():
    _skip_if_absent()
    tpm = cohort_gene_percentiles("PRAD", as_tpm=True)
    log = cohort_gene_percentiles("PRAD", as_tpm=False)
    # log space restores to TPM via expm1 (within float16 tolerance)
    np.testing.assert_allclose(
        np.expm1(log[_BP].to_numpy()), tpm[_BP].to_numpy(), rtol=1e-2, atol=1.0)


def test_neuroendocrine_cohorts_have_percentiles():
    _skip_if_absent()
    cohorts = set(available_percentile_cohorts())
    assert {"NET_PANCREAS", "SCLC", "NET_LUNG", "NEC_LUNG_LARGECELL"} <= cohorts


def test_summary_only_cohort_raises():
    _skip_if_absent()
    # MTC (GSE32662 microarray) is summary-only — no per-sample matrix, so no
    # percentile vector. (CLL/MM/BL etc. now DO have per-sample data.)
    with pytest.raises(ValueError):
        cohort_gene_percentiles("MTC")
