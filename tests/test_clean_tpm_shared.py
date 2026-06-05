"""The single shared clean-TPM helper (expression.normalize).

Previously copy-pasted into ~12 builders/scripts; these lock the one
definition. The default clean-TPM removal set (v2) zeroes technical-RNA rows
(mtDNA / rRNA-like / mt-like pseudogene / polyA-bias lncRNA) **and**
ribosomal-protein mRNA + pseudogenes, then renormalizes each sample column to
1e6. ``technical_rna_mask`` remains the strict technical-only subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pirlygenes.expression.normalize import (
    clean_tpm_matrix,
    clean_tpm_removal_mask,
    technical_rna_mask,
)


def _fixture():
    gene_table = pd.DataFrame({
        "Symbol": ["MT-CO1", "MT-RNR1", "MALAT1", "RPL13A", "TP53", "ACTB"],
        "Ensembl_Gene_ID": [
            "ENSG00000198804", "ENSG00000211459", "ENSG00000251562",
            "ENSG00000142541", "ENSG00000141510", "ENSG00000075624",
        ],
    })
    values = pd.DataFrame(
        np.array([[5e5, 8e5], [1e5, 5e4], [2e4, 3e4],
                  [3e5, 2e5], [5e4, 6e4], [2e5, 1e5]]),
        index=gene_table.index, columns=["S1", "S2"],
    )
    return gene_table, values


def test_technical_rna_mask_is_strict_technical_only():
    gene_table, _ = _fixture()
    mask = technical_rna_mask(gene_table)
    # strict technical set: MT-CO1, MT-RNR1 (mtDNA), MALAT1 (polyA-bias lncRNA)
    # removed; ribosomal *protein* mRNA (RPL13A) and normal genes kept.
    assert mask.tolist() == [True, True, True, False, False, False]


def test_default_removal_mask_also_drops_ribosomal_protein():
    gene_table, _ = _fixture()
    mask = clean_tpm_removal_mask(gene_table)  # default exclude_ribosomal_proteins=True
    # now RPL13A (ribosomal protein) is removed too; TP53/ACTB still kept.
    assert mask.tolist() == [True, True, True, True, False, False]
    # opting out reproduces the strict technical-only set
    strict = clean_tpm_removal_mask(gene_table, exclude_ribosomal_proteins=False)
    assert strict.tolist() == technical_rna_mask(gene_table).tolist()


def test_clean_tpm_zero_fill_drops_censored_and_renormalizes():
    gene_table, values = _fixture()
    # legacy zero fill: censored (technical + RPL13A) zeroed, remainder -> 1e6
    clean = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="zero")
    assert (clean.loc[[0, 1, 2, 3]] == 0).all().all()   # incl. RPL13A
    assert (clean.loc[[4, 5]] > 0).all().all()           # TP53, ACTB kept
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])


def test_clean_tpm_typical_fill_holds_constant_budget_and_avoids_inflation():
    gene_table, values = _fixture()
    # default "typical" fill: censored genes get a constant budget, sums to 1e6
    clean = clean_tpm_matrix(values, gene_table=gene_table)  # censored_fill="typical"
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
    # censored genes (MT/MALAT1/RPL13A) all share one constant value per sample
    cens = clean.loc[[0, 1, 2, 3]]
    for col in clean.columns:
        assert cens[col].nunique() == 1
    # with a fixed budget, kept genes are NOT inflated the way zero-fill does:
    # zero-fill scales kept by 1e6/kept_sum; typical-fill by 1e6/(kept_sum+budget)
    z = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="zero")
    assert (clean.loc[4] < z.loc[4]).all()   # TP53 less inflated under typical fill
    assert (clean.loc[5] < z.loc[5]).all()   # ACTB less inflated under typical fill


def test_technical_only_mask_keeps_ribo_protein():
    gene_table, values = _fixture()
    tech = clean_tpm_matrix(values, technical_rna_mask(gene_table), censored_fill="zero")
    assert (tech.loc[3] > 0).all()   # RPL13A kept under the strict technical set


def test_clean_tpm_all_censored_column():
    gene_table = pd.DataFrame({
        "Symbol": ["MT-CO1"], "Ensembl_Gene_ID": ["ENSG00000198804"],
    })
    values = pd.DataFrame([[123.0, 456.0]], index=[0], columns=["S1", "S2"])
    mask = technical_rna_mask(gene_table)
    # zero fill: nothing kept -> column collapses to zero
    z = clean_tpm_matrix(values, mask, censored_fill="zero")
    assert (z == 0).all().all()
    # typical fill: censored genes always get the surrogate value, so the
    # (all-censored) column is surrogate-filled and still sums to 1e6
    t = clean_tpm_matrix(values, mask, censored_fill="typical")
    np.testing.assert_allclose(t.sum(axis=0).to_numpy(), [1e6, 1e6])
    assert (t.loc[0] > 0).all()
