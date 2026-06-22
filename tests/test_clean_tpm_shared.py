"""The single shared clean-TPM helper (expression.normalize).

Previously copy-pasted into ~12 builders/scripts; these lock the one
definition. The clean-TPM removal set is technical-RNA rows (mtDNA / rRNA-like
/ mt-like pseudogene / polyA-bias lncRNA) **and** ribosomal-protein mRNA +
pseudogenes. As of clean_tpm_16_9_75 the **default** transform is
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


def test_clean_tpm_matrix_rejects_removed_modes():
    """zero / reference / typical were removed — clean_tpm_matrix is the single
    fixed_fraction clean-TPM contract and rejects anything else."""
    import pytest
    gene_table, values = _fixture()
    for mode in ("zero", "reference", "typical"):
        with pytest.raises(ValueError, match="fixed_fraction"):
            clean_tpm_matrix(values, gene_table=gene_table, censored_fill=mode)


def test_clean_tpm_fixed_fraction_three_compartment():
    """Default clean-TPM splits the censored block into ribosomal-protein (16%)
    + other-technical (9%), biology 75% (pirlygenes' current 16/9 contract)."""
    gene_table, values = _fixture()
    clean = clean_tpm_matrix(values, gene_table=gene_table,
                             censored_fill="fixed_fraction")
    mask = clean_tpm_removal_mask(gene_table).to_numpy()
    tech_only = clean_tpm_removal_mask(
        gene_table, exclude_ribosomal_proteins=False).to_numpy()
    ribo = mask & ~tech_only          # RPL13A
    other = mask & tech_only          # MT-CO1, MT-RNR1, MALAT1
    # each compartment pinned separately; biology gets the rest; total 1e6
    np.testing.assert_allclose(clean.loc[ribo].sum(axis=0).to_numpy(),
                               [160_000.0, 160_000.0])
    np.testing.assert_allclose(clean.loc[other].sum(axis=0).to_numpy(),
                               [90_000.0, 90_000.0])
    np.testing.assert_allclose(clean.loc[~mask].sum(axis=0).to_numpy(),
                               [750_000.0, 750_000.0])
    # combined censored still 25% (back-compat with the lumped contract)
    np.testing.assert_allclose(clean.loc[mask].sum(axis=0).to_numpy(),
                               [250_000.0, 250_000.0])
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
    # within-compartment relative expression preserved (MT-CO1:MT-RNR1 was 5:1)
    assert abs(clean.loc[0, "S1"] / clean.loc[1, "S1"]
               - values.loc[0, "S1"] / values.loc[1, "S1"]) < 1e-6
    # custom per-compartment fractions honoured
    c = clean_tpm_matrix(values, gene_table=gene_table,
                         censored_fill="fixed_fraction",
                         ribosomal_protein_fraction=0.20,
                         other_technical_fraction=0.05)
    np.testing.assert_allclose(c.loc[ribo].sum(axis=0).to_numpy(),
                               [200_000.0, 200_000.0])
    np.testing.assert_allclose(c.loc[other].sum(axis=0).to_numpy(),
                               [50_000.0, 50_000.0])
    # technical-only view (ribosomal kept in biology): single censored block at
    # technical_fraction, no ribosomal compartment
    t = clean_tpm_matrix(values, gene_table=gene_table,
                         censored_fill="fixed_fraction",
                         exclude_ribosomal_proteins=False, technical_fraction=0.10)
    np.testing.assert_allclose(t.loc[tech_only].sum(axis=0).to_numpy(),
                               [100_000.0, 100_000.0])


def test_technical_only_mask_keeps_ribo_protein():
    gene_table, values = _fixture()
    # under the strict technical-only mask, RPL13A (ribosomal protein) is NOT
    # censored, so it stays in the biological compartment (non-zero) after clean.
    tech = clean_tpm_matrix(values, technical_rna_mask(gene_table))
    assert (tech.loc[3] > 0).all()   # RPL13A kept under the strict technical set


# ---- runtime path consistency with the builder transform (#311) ----

def _wide_df():
    gene_table, values = _fixture()
    vals = values.rename(columns={"S1": "TPM_S1", "S2": "TPM_S2"})
    return pd.concat([gene_table, vals], axis=1)


def test_runtime_wrapper_matches_builder_clean_tpm():
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
    # biological compartment lands on the 750k biological budget; technical on 250k
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


def test_clean_tpm_runtime_rejects_symbol_only_input():
    """A symbol-only frame can't be censored (the list is ENSG-keyed), so the
    clean-TPM runtime path RAISES rather than silently censoring nothing while
    still rescaling to the fixed-fraction budget (#317 follow-up). The legacy
    zero path stays symbol-capable."""
    import pytest
    df = pd.DataFrame({"Symbol": ["MT-CO1", "TP53"], "TPM_S1": [5e5, 1e5]})
    with pytest.raises(ValueError):
        normalize_technical_rna_columns(df, censored_fill="fixed_fraction")
    # legacy zero path (classify_gene_qc, symbol-capable) does not raise
    out, info = normalize_technical_rna_columns(df)  # default censored_fill="zero"
    assert "removed_feature_mode" in info


def test_runtime_long_table_per_group_budget():
    """The long-table wrapper applies the clean-TPM transform within each cohort group:
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


def test_default_mask_is_technical_plus_ribosomal_only():
    # Translation factor (EEF1A1) and splicing (SRSF1) are NOT in the clean-TPM
    # removal set — only technical + ribosomal protein are (no extended level).
    gene_table, _ = _extended_fixture()
    removed = {gene_table["Symbol"][i] for i, m
               in enumerate(clean_tpm_removal_mask(gene_table).tolist()) if m}
    assert removed == {"MT-CO1", "RPL13A"}
    assert "EEF1A1" not in removed and "SRSF1" not in removed


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
    # drops technical (MT-CO1) + ribosomal protein (RPL13A); keeps biology
    # (translation/splicing are not censored — no extended level)
    bio = drop_technical_genes(frame)
    assert set(bio["Symbol"]) == {"EEF1A1", "SRSF1", "TP53", "ACTB", "MYC"}
    # technical-only keeps the ribosomal protein too
    bio_tech = drop_technical_genes(frame, exclude_ribosomal_proteins=False)
    assert "RPL13A" in set(bio_tech["Symbol"])
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


# ---- canonical censored-gene list: single source of truth + CTA-safe ----

def test_canonical_list_schema_and_categories():
    from pirlygenes.load_dataset import get_data
    df = get_data("clean-tpm-censored-genes")
    assert {"Ensembl_Gene_ID", "Symbol", "category"} <= set(df.columns)
    assert set(df["category"]) == {"technical", "ribosomal_protein"}
    assert len(df) > 2000


def test_canonical_list_is_cta_safe():
    """The list is CTA-excluded by construction, so the ribosomal-protein CTA
    (RPL10L) and the histone CTA (H1-6) are never censored — no runtime
    protect-subtract needed."""
    import pandas as pd
    from pirlygenes.load_dataset import get_data
    from pirlygenes.gene_sets_cancer import CTA_evidence

    censored = get_data("clean-tpm-censored-genes")
    censored_ens = set(censored["Ensembl_Gene_ID"].astype(str))
    cta_ens = set(CTA_evidence()["Ensembl_Gene_ID"].dropna().astype(str)
                  .str.split(".").str[0])
    assert censored_ens.isdisjoint(cta_ens)
    # RPL10L (ENSG00000165496) and H1-6 (ENSG00000187475) specifically kept
    gt = pd.DataFrame({"Symbol": ["RPL10L", "H1-6", "RPL13A"],
                       "Ensembl_Gene_ID": ["ENSG00000165496", "ENSG00000187475",
                                           "ENSG00000142541"]})
    mask = clean_tpm_removal_mask(gt).tolist()
    assert mask == [False, False, True]  # CTAs kept, real ribosomal censored


def test_removal_mask_membership_matches_canonical_list():
    """clean_tpm_removal_mask is exactly the canonical list (ENSG membership),
    so there is one source of truth used everywhere."""
    import pandas as pd
    from pirlygenes.load_dataset import get_data
    df = get_data("clean-tpm-censored-genes")
    gt = df.rename(columns={"category": "_cat"})[["Symbol", "Ensembl_Gene_ID"]].copy()
    # add a couple of biological genes that must NOT be censored
    gt = pd.concat([gt, pd.DataFrame({
        "Symbol": ["TP53", "EGFR"],
        "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000146648"]})],
        ignore_index=True)
    mask = clean_tpm_removal_mask(gt)
    removed = set(gt.loc[mask.to_numpy(), "Ensembl_Gene_ID"])
    assert removed == set(df["Ensembl_Gene_ID"].astype(str))


def test_public_technical_rna_contract_exported():
    """#445/#446: the clean-TPM technical-RNA compartment is a PUBLIC contract.
    Consumers (trufflepig) import these instead of underscore-private globals."""
    from pirlygenes.expression import (
        TECHNICAL_FRACTION,
        TECHNICAL_RNA_FAMILIES,
        TECHNICAL_RNA_GROUPS,
    )
    from pirlygenes.expression import qc

    assert TECHNICAL_RNA_GROUPS == frozenset(
        {"mt_dna", "mt_like_pseudogene", "rrna_like", "polyadenylation_bias_lncrna"})
    assert "mitochondrial" in TECHNICAL_RNA_FAMILIES
    assert TECHNICAL_FRACTION == 0.25
    # private names remain as back-compat aliases pointing at the public objects
    assert qc._TECHNICAL_RNA_GROUPS is qc.TECHNICAL_RNA_GROUPS
    assert qc._TECHNICAL_RNA_FAMILIES is qc.TECHNICAL_RNA_FAMILIES


def test_normalize_defaults_track_public_fraction():
    """The technical_fraction function defaults reference the public constant, so
    changing TECHNICAL_FRACTION can't silently desync from the applied value."""
    import inspect
    from pirlygenes.expression import normalize
    from pirlygenes.expression.qc import TECHNICAL_FRACTION
    sig = inspect.signature(normalize.normalize_expression)
    assert sig.parameters["technical_fraction"].default == TECHNICAL_FRACTION
