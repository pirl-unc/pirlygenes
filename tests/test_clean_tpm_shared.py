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


def test_default_removal_mask_is_mitochondrial_and_ribosomal_only():
    gene_table, _ = _fixture()
    # default: mitochondrial (MT-CO1, MT-RNR1) + ribosomal protein (RPL13A) are
    # censored; MALAT1 (polyA-bias lncRNA) and normal genes are NOT (only the
    # mitochondrial + ribosomal families, nothing else).
    assert clean_tpm_removal_mask(gene_table).tolist() == \
        [True, True, False, True, False, False]
    # exclude_ribosomal_proteins=False -> mitochondrial + rRNA only (RPL13A kept)
    assert clean_tpm_removal_mask(gene_table, exclude_ribosomal_proteins=False).tolist() == \
        [True, True, False, False, False, False]
    # only the strict technical set still flags MALAT1
    assert technical_rna_mask(gene_table).tolist() == \
        [True, True, True, False, False, False]


def test_clean_tpm_zero_fill_drops_censored_and_renormalizes():
    gene_table, values = _fixture()
    clean = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="zero")
    assert (clean.loc[[0, 1, 3]] == 0).all().all()    # mito + RPL13A censored
    assert (clean.loc[[2, 4, 5]] > 0).all().all()      # MALAT1, TP53, ACTB kept
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])


def test_clean_tpm_typical_fill_constant_budget_avoids_inflation():
    gene_table, values = _fixture()
    clean = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="typical")
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
    cens = clean.loc[[0, 1, 3]]                         # censored share one value
    for col in clean.columns:
        assert cens[col].nunique() == 1
    z = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="zero")
    assert (clean.loc[4] < z.loc[4]).all()             # kept less inflated


def test_clean_tpm_reference_fill_pins_per_gene_and_fills_remainder():
    gene_table, values = _fixture()
    # explicit per-gene reference constants (deterministic, cohort-independent)
    ref = {"MT-CO1": 100.0, "MT-RNR1": 50.0, "RPL13A": 80.0}   # sum 230
    clean = clean_tpm_matrix(values, gene_table=gene_table,
                             censored_fill="reference", reference=ref)
    # each censored gene pinned EXACTLY to its reference, identical every sample
    assert clean.loc[0].tolist() == [100.0, 100.0]
    assert clean.loc[1].tolist() == [50.0, 50.0]
    assert clean.loc[3].tolist() == [80.0, 80.0]
    # non-censored genes (incl. MALAT1) fill the remaining 1e6 - 230 budget
    np.testing.assert_allclose(clean.loc[[2, 4, 5]].sum(axis=0).to_numpy(),
                               [1e6 - 230.0, 1e6 - 230.0])
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])


def test_reference_fill_is_cohort_independent():
    gene_table, values = _fixture()
    ref = {"MT-CO1": 100.0, "MT-RNR1": 50.0, "RPL13A": 80.0}
    a = clean_tpm_matrix(values, gene_table=gene_table,
                         censored_fill="reference", reference=ref)
    b = clean_tpm_matrix(values * 7, gene_table=gene_table,
                         censored_fill="reference", reference=ref)
    # censored genes hold the same value regardless of the (scaled) input cohort
    for i in (0, 1, 3):
        assert a.loc[i].tolist() == b.loc[i].tolist()


def test_technical_only_mask_keeps_ribo_protein():
    gene_table, values = _fixture()
    tech = clean_tpm_matrix(values, technical_rna_mask(gene_table), censored_fill="zero")
    assert (tech.loc[3] > 0).all()   # RPL13A kept under the strict technical set
