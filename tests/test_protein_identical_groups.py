"""Protein-identical gene groups + the linear-space collapse for
protein-abundance proxies (segmental-dup paralogs, histone clusters, CTAs)."""

import numpy as np
import pandas as pd

from pirlygenes.expression.protein_groups import (
    collapse_protein_identical_loci,
    collapse_protein_identical_loci_long,
    fold_symbols_to_canonical,
    protein_identical_groups,
)


def _sym_groups():
    """{canonical_ensg: set(symbols)} from the derived table."""
    df = protein_identical_groups()
    out = {}
    for canon, grp in df.groupby("group_canonical_ensembl_gene_id"):
        out[canon] = set(grp["symbol"].astype(str))
    return out


def test_groups_table_shape_and_invariants():
    df = protein_identical_groups()
    for col in ("group_canonical_ensembl_gene_id", "ensembl_gene_id",
                "symbol", "protein_aa", "n_members"):
        assert col in df.columns
    # every group has >= 2 members and a protein at least the min length
    assert (df["n_members"] >= 2).all()
    assert (df["protein_aa"] >= 30).all()
    # the canonical id is itself a member of its group (representative present)
    for canon, grp in df.groupby("group_canonical_ensembl_gene_id"):
        assert canon in set(grp["ensembl_gene_id"])
        # canonical = lexicographically smallest member id (deterministic)
        assert canon == min(grp["ensembl_gene_id"])


def test_real_protein_identical_families_are_grouped():
    """Histone clusters and protein-identical CTAs must group together."""
    groups = _sym_groups()
    # histone H4 cluster: many loci, one identical H4 protein
    h4 = max((s for s in groups.values() if any(x.startswith("H4C") for x in s)),
             key=len, default=set())
    assert sum(x.startswith("H4C") for x in h4) >= 5
    # CT47A cancer-testis cluster
    assert any(sum(x.startswith("CT47A") for x in s) >= 5 for s in groups.values())
    # known protein-identical CTA paralog pairs share a group
    def together(a, b):
        return any({a, b} <= s for s in groups.values())
    assert together("SSX2", "SSX2B")
    assert together("MAGEA9", "MAGEA9B")


def test_distinct_protein_paralogs_not_grouped():
    """Placental / paralog families that differ at the protein level must NOT
    be merged, and unrelated genes must never be bridged by a short fragment."""
    groups = _sym_groups()
    flat = [s for s in groups.values()]
    # PSG and CSH placental families differ at protein level -> not co-grouped
    assert not any(sum(x.startswith("PSG") for x in s) >= 2 for s in flat)
    # the canonical spurious bridge (short shared isoform) must be absent
    assert not any({"FGFR1", "IFIT3"} <= s for s in flat)


def _cohort(ids, symbols, **cols):
    return pd.DataFrame({"Ensembl_Gene_ID": ids, "Symbol": symbols, **cols})


def test_collapse_sums_in_linear_space_and_ignores_nan_members():
    df = protein_identical_groups()
    grp = df[df["n_members"] == 2].iloc[0]
    canon = grp["group_canonical_ensembl_gene_id"]
    members = df[df["group_canonical_ensembl_gene_id"] == canon]["ensembl_gene_id"].tolist()
    frame = _cohort(
        members + ["ENSG00000000001"],
        ["A", "Ab", "UNREL"],
        t1=[10.0, 5.0, 100.0],
        t2=[np.nan, 8.0, 50.0],     # first member NaN in t2 -> ignored, not zero
        t3=[np.nan, np.nan, 1.0],   # both members NaN -> stays NaN
    )
    out = collapse_protein_identical_loci(frame)
    assert len(out) == 2                                   # group collapsed + unrelated
    row = out[out["Ensembl_Gene_ID"] == canon].iloc[0]
    assert row["t1"] == 15.0                               # 10 + 5
    assert row["t2"] == 8.0                                # NaN ignored, not 0
    assert np.isnan(row["t3"])                             # all-NaN stays NaN (min_count=1)
    assert out[out["Symbol"] == "UNREL"]["t1"].iloc[0] == 100.0  # untouched


def test_collapse_noop_when_single_member_present():
    df = protein_identical_groups()
    grp = df[df["n_members"] >= 2].iloc[0]
    one_member = grp["ensembl_gene_id"]
    frame = _cohort([one_member, "ENSG00000000002"], ["A", "B"], t1=[3.0, 4.0])
    out = collapse_protein_identical_loci(frame)
    assert len(out) == 2                                   # nothing to merge
    pd.testing.assert_frame_equal(
        out.sort_values("Ensembl_Gene_ID").reset_index(drop=True),
        frame.sort_values("Ensembl_Gene_ID").reset_index(drop=True),
    )


def test_long_collapse_sums_per_context_not_across():
    """The long collapse sums a group within each (cohort) context separately,
    keying the merged row by the canonical id/symbol."""
    df = protein_identical_groups()
    grp = df[df["n_members"] == 2].iloc[0]
    canon = grp["group_canonical_ensembl_gene_id"]
    members = df[df["group_canonical_ensembl_gene_id"] == canon]["ensembl_gene_id"].tolist()
    # two cohorts; member-0 NaN in cohort B
    long = pd.DataFrame({
        "Ensembl_Gene_ID": members * 2 + ["ENSG00000000003", "ENSG00000000003"],
        "Symbol": ["A", "Ab"] * 2 + ["U", "U"],
        "cancer_code": ["CA", "CA", "CB", "CB", "CA", "CB"],
        "expression": [10.0, 5.0, 7.0, np.nan, 99.0, 88.0],
    })
    out = collapse_protein_identical_loci_long(
        long, group_keys=["cancer_code"], sum_cols=["expression"])
    val = out.set_index(["Ensembl_Gene_ID", "cancer_code"])["expression"]
    assert val[(canon, "CA")] == 15.0          # 10 + 5 within cohort CA
    assert val[(canon, "CB")] == 7.0           # 7 + NaN -> 7 (NaN ignored)
    assert val[("ENSG00000000003", "CA")] == 99.0   # ungrouped untouched
    # merged row carries the canonical symbol
    assert out[out["Ensembl_Gene_ID"] == canon]["Symbol"].iloc[0] == grp["group_canonical_symbol"]


def test_fold_symbols_to_canonical():
    df = protein_identical_groups()
    grp = df[df["n_members"] >= 2].iloc[0]
    members = df[df["group_canonical_ensembl_gene_id"] ==
                 grp["group_canonical_ensembl_gene_id"]]["symbol"].tolist()
    canon_sym = grp["group_canonical_symbol"]
    folded = fold_symbols_to_canonical([m for m in members if m] + ["UNRELATEDXYZ"])
    # all members collapse to the single canonical symbol; unrelated passes through
    assert canon_sym in folded
    assert "UNRELATEDXYZ" in folded
    assert folded.count(canon_sym) == 1        # de-duplicated


def test_accessor_collapse_option_sums_paralogs():
    from pirlygenes.expression.accessors import cancer_reference_expression
    base = cancer_reference_expression(cancer_types=["SKCM"], normalize="tpm_clean")
    coll = cancer_reference_expression(cancer_types=["SKCM"], normalize="tpm_clean",
                                       collapse_protein_identical=True)
    assert len(coll) < len(base)               # loci merged
    # MAGEA9/MAGEA9B -> one canonical row whose value is their sum
    b = base[base["Symbol"].isin(["MAGEA9", "MAGEA9B"])]["expression"]
    c = coll[coll["Symbol"].isin(["MAGEA9", "MAGEA9B"])]["expression"]
    assert len(c) == 1
    assert abs(float(c.iloc[0]) - float(b.sum())) < 1e-6


def test_collapse_is_idempotent():
    df = protein_identical_groups()
    canon = df[df["n_members"] == 2].iloc[0]["group_canonical_ensembl_gene_id"]
    members = df[df["group_canonical_ensembl_gene_id"] == canon]["ensembl_gene_id"].tolist()
    frame = _cohort(members, ["A", "Ab"], t1=[2.0, 3.0])
    once = collapse_protein_identical_loci(frame)
    twice = collapse_protein_identical_loci(once)
    pd.testing.assert_frame_equal(once, twice)
    assert once["t1"].iloc[0] == 5.0
