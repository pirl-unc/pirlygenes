"""The single shared clean-TPM helper (expression.normalize).

Previously copy-pasted into ~12 builders/scripts; these lock the one
definition: zero technical-RNA rows (mtDNA / rRNA-like / mt-like
pseudogene / polyA-bias lncRNA), keep everything else (incl. ribosomal
*protein* mRNA), then renormalize each sample column to 1e6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pirlygenes.expression.normalize import clean_tpm_matrix, technical_rna_mask


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


def test_technical_rna_mask_selects_mt_rrna_polyalncrna_not_ribo_protein():
    gene_table, _ = _fixture()
    mask = technical_rna_mask(gene_table)
    # MT-CO1, MT-RNR1 (mtDNA) and MALAT1 (polyA-bias lncRNA) are removed;
    # ribosomal *protein* mRNA (RPL13A) and normal genes are kept.
    assert mask.tolist() == [True, True, True, False, False, False]


def test_clean_tpm_zeroes_removable_and_renormalizes_to_million():
    gene_table, values = _fixture()
    clean = clean_tpm_matrix(values, technical_rna_mask(gene_table))
    # removed rows are zero in every sample
    assert (clean.loc[[0, 1, 2]] == 0).all().all()
    # kept rows are nonzero; each sample renormalized to 1e6
    assert (clean.loc[[3, 4, 5]] > 0).all().all()
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])


def test_clean_tpm_all_removed_column_becomes_zero():
    gene_table = pd.DataFrame({
        "Symbol": ["MT-CO1"], "Ensembl_Gene_ID": ["ENSG00000198804"],
    })
    values = pd.DataFrame([[123.0, 456.0]], index=[0], columns=["S1", "S2"])
    clean = clean_tpm_matrix(values, technical_rna_mask(gene_table))
    assert (clean == 0).all().all()
