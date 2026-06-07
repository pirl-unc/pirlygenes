"""The single shared clean-TPM helper (expression.normalize).

Previously copy-pasted into ~12 builders/scripts; these lock the one
definition. The clean-TPM removal set is technical-RNA rows (mtDNA / rRNA-like
/ mt-like pseudogene / polyA-bias lncRNA) **and** ribosomal-protein mRNA +
pseudogenes. As of clean_tpm_v4 the **default** transform is
``censored_fill="fixed_fraction"``: that removal (technical) block is forced to
25% of the 1e6 budget and the kept (biological) block to 75%, each renormalized
within its group (the ``"reference"`` / ``"typical"`` / ``"zero"`` modes remain
available). ``technical_rna_mask`` is the strict technical-only subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pirlygenes.expression.normalize import (
    clean_tpm_matrix,
    clean_tpm_removal_mask,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
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


def test_default_removal_mask_is_technical_plus_ribosomal_protein():
    gene_table, _ = _fixture()
    # default: technical RNA (MT-CO1, MT-RNR1, MALAT1) + ribosomal protein
    # (RPL13A) censored; normal genes kept. Nothing else.
    assert clean_tpm_removal_mask(gene_table).tolist() == \
        [True, True, True, True, False, False]
    # exclude_ribosomal_proteins=False -> strict technical-only (RPL13A kept)
    assert clean_tpm_removal_mask(gene_table, exclude_ribosomal_proteins=False).tolist() == \
        technical_rna_mask(gene_table).tolist() == [True, True, True, False, False, False]


def test_clean_tpm_zero_fill_drops_censored_and_renormalizes():
    gene_table, values = _fixture()
    clean = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="zero")
    assert (clean.loc[[0, 1, 2, 3]] == 0).all().all()   # technical + RPL13A
    assert (clean.loc[[4, 5]] > 0).all().all()           # TP53, ACTB kept
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])


def test_clean_tpm_typical_fill_constant_budget_avoids_inflation():
    gene_table, values = _fixture()
    clean = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="typical")
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
    cens = clean.loc[[0, 1, 2, 3]]                      # censored share one value
    for col in clean.columns:
        assert cens[col].nunique() == 1
    z = clean_tpm_matrix(values, gene_table=gene_table, censored_fill="zero")
    assert (clean.loc[4] < z.loc[4]).all()             # kept less inflated


def test_clean_tpm_reference_fill_pins_per_gene_and_fills_remainder():
    gene_table, values = _fixture()
    # explicit per-gene reference constants (deterministic, cohort-independent)
    ref = {"MT-CO1": 100.0, "MT-RNR1": 50.0, "MALAT1": 40.0, "RPL13A": 80.0}  # 270
    clean = clean_tpm_matrix(values, gene_table=gene_table,
                             censored_fill="reference", reference=ref)
    # each censored gene pinned EXACTLY to its reference, identical every sample
    assert clean.loc[0].tolist() == [100.0, 100.0]
    assert clean.loc[2].tolist() == [40.0, 40.0]
    assert clean.loc[3].tolist() == [80.0, 80.0]
    # the kept genes (TP53, ACTB) fill the remaining 1e6 - 270 budget
    np.testing.assert_allclose(clean.loc[[4, 5]].sum(axis=0).to_numpy(),
                               [1e6 - 270.0, 1e6 - 270.0])
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])


def test_reference_fill_is_cohort_independent():
    gene_table, values = _fixture()
    ref = {"MT-CO1": 100.0, "MT-RNR1": 50.0, "MALAT1": 40.0, "RPL13A": 80.0}
    a = clean_tpm_matrix(values, gene_table=gene_table,
                         censored_fill="reference", reference=ref)
    b = clean_tpm_matrix(values * 7, gene_table=gene_table,
                         censored_fill="reference", reference=ref)
    # censored genes hold the same value regardless of the (scaled) input cohort
    for i in (0, 1, 2, 3):
        assert a.loc[i].tolist() == b.loc[i].tolist()


def test_clean_tpm_fixed_fraction_two_compartment():
    gene_table, values = _fixture()
    # two samples deliberately differ in technical fraction (the confound)
    clean = clean_tpm_matrix(values, gene_table=gene_table,
                             censored_fill="fixed_fraction")
    mask = clean_tpm_removal_mask(gene_table).to_numpy()
    # every sample: technical -> 25%, biological -> 75%, total 1e6
    np.testing.assert_allclose(clean.loc[mask].sum(axis=0).to_numpy(),
                               [250_000.0, 250_000.0])
    np.testing.assert_allclose(clean.loc[~mask].sum(axis=0).to_numpy(),
                               [750_000.0, 750_000.0])
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
    # within-compartment relative expression preserved (MT-CO1:MT-RNR1 was 5:1)
    assert abs(clean.loc[0, "S1"] / clean.loc[1, "S1"]
               - values.loc[0, "S1"] / values.loc[1, "S1"]) < 1e-6
    # custom fraction honoured
    c10 = clean_tpm_matrix(values, gene_table=gene_table,
                           censored_fill="fixed_fraction", technical_fraction=0.10)
    np.testing.assert_allclose(c10.loc[mask].sum(axis=0).to_numpy(),
                               [100_000.0, 100_000.0])


def test_technical_only_mask_keeps_ribo_protein():
    gene_table, values = _fixture()
    tech = clean_tpm_matrix(values, technical_rna_mask(gene_table), censored_fill="zero")
    assert (tech.loc[3] > 0).all()   # RPL13A kept under the strict technical set


# ---- runtime path consistency with the builder transform (#311) ----

def _wide_df():
    gene_table, values = _fixture()
    vals = values.rename(columns={"S1": "TPM_S1", "S2": "TPM_S2"})
    return pd.concat([gene_table, vals], axis=1)


def test_runtime_wrapper_v4_matches_builder_clean_tpm():
    """normalize_technical_rna_columns(censored_fill='fixed_fraction') produces
    the SAME values as the builder's clean_tpm_matrix — the runtime path now
    matches how packaged references are built (#311)."""
    gene_table, values = _fixture()
    ref = clean_tpm_matrix(values, gene_table=gene_table,
                           censored_fill="fixed_fraction")
    out, info = normalize_technical_rna_columns(_wide_df(),
                                                censored_fill="fixed_fraction")
    rt = out[["TPM_S1", "TPM_S2"]].to_numpy()
    np.testing.assert_allclose(rt, ref.to_numpy())
    # biological compartment lands on the 750k v4 budget; technical on 250k
    mask = clean_tpm_removal_mask(gene_table).to_numpy()
    np.testing.assert_allclose(out[["TPM_S1", "TPM_S2"]][~mask].sum().to_numpy(),
                               [750_000.0, 750_000.0])
    assert info["removed_feature_mode"] == "fixed_fraction"


def test_runtime_wrapper_default_is_legacy_zero_unchanged():
    """Default censored_fill='zero' keeps the legacy zero-and-renormalize on the
    technical-only set (no regression): RPL13A (ribosomal protein) is NOT in the
    zero path's removal set, so it stays nonzero."""
    out, info = normalize_technical_rna_columns(_wide_df())
    assert info["removed_feature_mode"] == "zeroed_then_renormalized"
    # technical RNA zeroed (MT-CO1 row 0); ribosomal protein RPL13A (row 3) kept
    assert (out.loc[0, ["TPM_S1", "TPM_S2"]] == 0).all()
    assert (out.loc[3, ["TPM_S1", "TPM_S2"]] > 0).all()


def test_runtime_long_table_v4_per_group_budget():
    """The long-table wrapper applies the v4 transform within each cohort group:
    every group's biological compartment lands on 750k."""
    gene_table, values = _fixture()
    rows = []
    for code in ("AAA", "BBB"):
        for i in range(len(gene_table)):
            rows.append({
                "symbol": gene_table.loc[i, "Symbol"],
                "Ensembl_Gene_ID": gene_table.loc[i, "Ensembl_Gene_ID"],
                "cancer_code": code,
                "subtype": "",
                "tumor_tpm_median": float(values.iloc[i, 0] * (1 if code == "AAA" else 2)),
            })
    long = pd.DataFrame(rows)
    out, info = normalize_technical_rna_long_table(
        long, value_cols=("tumor_tpm_median",), censored_fill="fixed_fraction")
    mask_by_code = {}
    for code, g in out.groupby("cancer_code"):
        gt = g[["symbol", "Ensembl_Gene_ID"]].rename(columns={"symbol": "Symbol"})
        mask = clean_tpm_removal_mask(gt.reset_index(drop=True)).to_numpy()
        bio = g["tumor_tpm_median"].to_numpy()[~mask].sum()
        mask_by_code[code] = bio
    for code, bio in mask_by_code.items():
        assert abs(bio - 750_000.0) < 1.0, (code, bio)


def test_clean_tpm_helpers_exported_at_top_level():
    import pirlygenes as pg
    assert pg.clean_tpm_matrix is clean_tpm_matrix
    assert pg.clean_tpm_removal_mask is clean_tpm_removal_mask


# ---- cross-source transforms: extended level + rank/zscore (#293) ----

def _extended_fixture():
    gene_table = pd.DataFrame({
        "Symbol": ["MT-CO1", "RPL13A", "EEF1A1", "SRSF1", "TP53", "ACTB", "MYC"],
        "Ensembl_Gene_ID": [
            "ENSG00000198804", "ENSG00000142541", "ENSG00000156508",
            "ENSG00000136450", "ENSG00000141510", "ENSG00000075624",
            "ENSG00000136997",
        ],
    })
    values = pd.DataFrame(
        {"S1": [5e5, 3e5, 2e5, 1e5, 5e4, 2e5, 8e4],
         "S2": [8e5, 2e5, 1e5, 5e4, 6e4, 1e5, 3e4]},
        index=gene_table.index)
    return gene_table, values


def test_extended_level_adds_extended_housekeeping():
    from pirlygenes.expression.normalize import clean_tpm_matrix as ctm
    gene_table, values = _extended_fixture()
    default = clean_tpm_removal_mask(gene_table).tolist()
    extended = clean_tpm_removal_mask(gene_table, level="extended").tolist()
    syms = gene_table["Symbol"].tolist()
    # default = technical (MT-CO1) + ribosomal protein (RPL13A)
    assert {syms[i] for i, m in enumerate(default) if m} == {"MT-CO1", "RPL13A"}
    # extended also removes translation factor (EEF1A1), splicing (SRSF1),
    # classic HK (ACTB); keeps real biology (TP53, MYC)
    ext_removed = {syms[i] for i, m in enumerate(extended) if m}
    assert {"EEF1A1", "SRSF1", "ACTB"} <= ext_removed
    assert "TP53" not in ext_removed and "MYC" not in ext_removed
    # extended clean still lands biological on the 750k v4 budget
    import numpy as np
    clean = ctm(values, gene_table=gene_table, level="extended")
    em = np.array(extended)
    np.testing.assert_allclose(clean.to_numpy()[~em].sum(axis=0), [750_000.0] * 2)


def test_clean_tpm_invalid_level_raises():
    gene_table, _ = _extended_fixture()
    import pytest
    with pytest.raises(ValueError):
        clean_tpm_removal_mask(gene_table, level="bogus")


def test_rank_normalize_within_sample_percentile():
    from pirlygenes.expression.normalize import rank_normalize
    _gt, values = _extended_fixture()
    r = rank_normalize(values)
    # top gene in each sample ranks at 100; ranks are within (0, 100]
    assert r["S1"].max() == 100.0 and r["S2"].max() == 100.0
    assert (r.to_numpy() > 0).all() and (r.to_numpy() <= 100).all()
    # rank is monotone with value within a column (MT-CO1 highest in S1)
    assert r["S1"].idxmax() == values["S1"].idxmax()


def test_drop_technical_genes_biology_only_view():
    from pirlygenes.expression.normalize import drop_technical_genes
    gene_table, values = _extended_fixture()
    frame = gene_table.copy()
    for c in values.columns:
        frame[c] = values[c].values
    # default: drops technical (MT-CO1) + ribosomal protein (RPL13A)
    bio = drop_technical_genes(frame)
    assert set(bio["Symbol"]) == {"EEF1A1", "SRSF1", "TP53", "ACTB", "MYC"}
    # extended: also drops translation/splicing/classic-HK
    bio_ext = drop_technical_genes(frame, level="extended")
    assert set(bio_ext["Symbol"]) == {"TP53", "MYC"}
    # sample columns pass through untouched
    assert list(values.columns) == [c for c in bio.columns
                                    if c not in ("Symbol", "Ensembl_Gene_ID")]


def test_zscore_normalize_standardizes_each_column():
    import numpy as np
    from pirlygenes.expression.normalize import zscore_normalize
    _gt, values = _extended_fixture()
    z = zscore_normalize(values)
    np.testing.assert_allclose(z.mean(axis=0).to_numpy(), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0, ddof=0).to_numpy(), [1.0, 1.0], atol=1e-9)
    # zero-variance column -> all zeros (no NaN/inf)
    flat = pd.DataFrame({"S1": [7.0, 7.0, 7.0]})
    assert (zscore_normalize(flat)["S1"] == 0.0).all()
