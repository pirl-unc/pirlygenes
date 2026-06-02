"""recount3 coverage gene-sums → TPM length-normalization (no network).

Locks the documented math in pirlygenes.builders.recount3.gene_sums_to_tpm:
divide coverage by exonic bp_length, renormalize each sample to 1e6, and
collapse version / _PAR_Y duplicate Gencode IDs by summation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pirlygenes.builders.recount3 import gene_sums_to_tpm


def test_length_normalize_and_renormalize_to_million():
    # Two genes, equal coverage but gene B is 2x longer → B's TPM is half A's.
    gene_sums = pd.DataFrame(
        {"S1": [1000.0, 1000.0], "S2": [0.0, 500.0]},
        index=["ENSG00000000001.5", "ENSG00000000002.3"],
    )
    bp_length = pd.Series(
        {"ENSG00000000001": 1000.0, "ENSG00000000002": 2000.0}
    )
    tpm = gene_sums_to_tpm(gene_sums, bp_length)
    # each sample column renormalizes to 1e6
    np.testing.assert_allclose(tpm.sum(axis=0).to_numpy(), [1e6, 1e6])
    # S1: rates 1.0 vs 0.5 → 2:1 split
    np.testing.assert_allclose(
        tpm["S1"].to_numpy(), [2e6 / 3, 1e6 / 3], rtol=1e-9
    )
    # S2: only gene B has coverage → it takes the whole 1e6
    np.testing.assert_allclose(tpm["S2"].to_numpy(), [0.0, 1e6])


def test_version_and_par_y_duplicates_are_summed():
    gene_sums = pd.DataFrame(
        {"S1": [600.0, 400.0]},
        index=["ENSG00000000003.1", "ENSG00000000003.1_PAR_Y"],
    )
    bp_length = pd.Series({"ENSG00000000003": 1000.0})
    tpm = gene_sums_to_tpm(gene_sums, bp_length)
    # both rows collapse to one unversioned ENSG, taking the full 1e6
    assert tpm.index.tolist() == ["ENSG00000000003"]
    np.testing.assert_allclose(tpm["S1"].to_numpy(), [1e6])


def test_genes_without_length_are_dropped():
    gene_sums = pd.DataFrame(
        {"S1": [1000.0, 1000.0]},
        index=["ENSG00000000004.2", "ENSG00000000005.2"],
    )
    bp_length = pd.Series({"ENSG00000000004": 1000.0})  # gene 5 has no length
    tpm = gene_sums_to_tpm(gene_sums, bp_length)
    assert tpm.index.tolist() == ["ENSG00000000004"]
    np.testing.assert_allclose(tpm["S1"].to_numpy(), [1e6])
