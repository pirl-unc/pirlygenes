"""Scoped regression guard for the #557 delegation parity harness.

The full sweep lives in ``scripts/parity_reference_expression.py`` (offline,
writes a per-code report). Here we lock in the *shape* of parity on a couple of
well-behaved cohorts so a regression in the delegated compatibility projection,
or in the canonical oncoref comparison view, trips in CI.

Tolerances are deliberately loose relative to the observed deltas (PRAD median
~0.05%, p95 ~0.14%): the point is to catch a structural break (wrong join, a
unit/scale regression, a vanished cohort), not to pin float noise.
"""

import warnings

import pandas as pd
import pytest

pytest.importorskip("oncoref")

from pirlygenes.expression.accessors import cancer_reference_expression
from pirlygenes.expression.parity import parity_for_code


def _serves(code: str) -> bool:
    """oncoref must be able to compute the code from its artifact in this env
    (the bundle is present in CI, but skip rather than fail if a fetch is
    needed and unavailable)."""
    import oncoref

    for qc in ("pass", "pass_or_warn", "all"):
        try:
            oncoref.cancer_reference_expression(
                cancer_types=code, genes=["ENSG00000141510"], normalize="tpm_clean",
                sample_qc=qc,
            )
            return True
        except ValueError as err:
            if "sample_qc" in str(err).lower() and "mismatch" in str(err).lower():
                continue
            return False
        except Exception:
            return False
    return False


@pytest.fixture(scope="module")
def pg_frame():
    warnings.filterwarnings("ignore")
    return cancer_reference_expression(
        cancer_types=["PRAD", "LUAD", "MTC", "SARC_DDLPS"]
    )


@pytest.mark.parametrize("code", ["PRAD", "LUAD"])
def test_clean_cohort_parity(pg_frame, code):
    if not _serves(code):
        pytest.skip(f"oncoref cannot serve {code} in this environment")
    r = parity_for_code(code, pg_frame=pg_frame)
    assert r["status"] == "ok", r
    # The reference sample set behind each summary must be identical.
    assert r["n_samples_match"], (r["n_samples_pg"], r["n_samples_on"])
    # Well-expressed genes agree tightly; loose ceilings guard a structural break.
    assert r["rel_median"] < 0.01, r["rel_median"]
    assert r["rel_p95"] < 0.05, r["rel_p95"]
    # Most of the gene universe overlaps (the divergent tail is a small minority).
    assert r["n_genes_shared"] > 20000, r["n_genes_shared"]


def test_multi_cohort_code_paired_to_oncoref_cohort(pg_frame):
    """A code pirlygenes serves from several source_cohorts (SARC_DDLPS spans 3)
    must be paired to the single cohort oncoref computed from — matched by sample
    count — not blurred into a many-to-many join. Guards the multi-cohort bug:
    before the fix this reported n_samples 40/48 and a ~38% median delta."""
    if not _serves("SARC_DDLPS"):
        pytest.skip("oncoref cannot serve SARC_DDLPS in this environment")
    assert pg_frame[pg_frame["cancer_code"] == "SARC_DDLPS"][
        "source_cohort"
    ].nunique() > 1, "fixture precondition: SARC_DDLPS should be multi-cohort"
    r = parity_for_code("SARC_DDLPS", pg_frame=pg_frame)
    assert r["status"] == "ok", r
    assert r["n_samples_match"], (r["n_samples_pg"], r["n_samples_on"])
    assert r["rel_median"] < 0.01, r["rel_median"]


def test_group_code_multi_expansion_flagged():
    """A future pooled/group code (oncoref expands `CRC` into COAD+READ blocks
    under one label, repeating each gene) must be flagged `oncoref-multi-cohort`,
    not silently dedup'd down to whichever block sorts first. Uses a synthetic pg
    frame so the code is not `pg-empty` and reaches the guard."""
    if not _serves("CRC"):
        pytest.skip("oncoref cannot serve CRC in this environment")
    syn = pd.DataFrame(
        {
            "cancer_code": ["CRC", "CRC"],
            "normalization": ["TPM_clean", "TPM_clean"],
            "source_cohort": ["FAKE", "FAKE"],
            "n_samples": [100, 100],
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000171862"],
            "expression": [10.0, 5.0],
        }
    )
    r = parity_for_code("CRC", pg_frame=syn)
    assert r["status"] == "oncoref-multi-cohort", r


def test_qc_policy_fallback(pg_frame):
    """MTC's artifact was baked ``pass_or_warn``; the harness must fall back to
    it instead of erroring on the default ``pass`` policy."""
    if not _serves("MTC"):
        pytest.skip("oncoref cannot serve MTC in this environment")
    r = parity_for_code("MTC", pg_frame=pg_frame)
    assert r["status"] == "ok", r
    assert r["qc_used"] == "pass_or_warn", r["qc_used"]


def test_report_shape_smoke(pg_frame):
    """parity_for_code returns the documented metric keys for a served code."""
    if not _serves("PRAD"):
        pytest.skip("oncoref cannot serve PRAD in this environment")
    r = parity_for_code("PRAD", pg_frame=pg_frame)
    expected = {
        "cancer_code", "status", "n_samples_pg", "n_samples_on",
        "n_samples_match", "n_genes_shared", "rel_median", "rel_p95",
        "n_divergent",
    }
    assert expected <= set(r), expected - set(r)
