"""Protein-identical gene groups + the linear-space collapse for
protein-abundance proxies (segmental-dup paralogs, histone clusters, CTAs)."""

import numpy as np
import pandas as pd

from pirlygenes.expression.protein_groups import (
    annotate_panel_proteoforms,
    cdna_identical_groups,
    collapse_protein_identical_loci,
    collapse_protein_identical_loci_long,
    fold_symbols_to_canonical,
    proteoform_group_of,
    proteoform_id,
    protein_identical_groups,
)


def _sym_groups():
    """{canonical_ensg: set(symbols)} from the derived table.

    Members whose symbol is unresolved (NaN in the table) carry no symbol to
    match on, so they're dropped here. ``dropna`` before ``astype(str)`` keeps
    this robust across pandas versions: older pandas coerces NaN to the literal
    ``"nan"`` string, newer pandas preserves it as a float — and a float has no
    ``.startswith`` for the prefix checks below.
    """
    df = protein_identical_groups()
    out = {}
    for canon, grp in df.groupby("group_canonical_ensembl_gene_id"):
        out[canon] = set(grp["symbol"].dropna().astype(str))
    return out


def test_every_member_and_group_has_a_symbol():
    """No blank symbols in either derived table: a locus Ensembl ships without an
    HGNC name (novel / alt-contig protein- or cDNA-identical duplicates) falls
    back to its ENSG, so the symbol columns are always a usable identifier —
    never blank (a blank round-trips through CSV as NaN, which breaks downstream
    string ops)."""
    for df in (protein_identical_groups(), cdna_identical_groups()):
        assert df["symbol"].notna().all()
        assert df["group_canonical_symbol"].notna().all()
        # the fallback is exactly the row's own ENSG; never an unrelated value
        nameless = df[df["symbol"].str.startswith("ENSG")]
        assert (nameless["symbol"] == nameless["ensembl_gene_id"]).all()


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
    # the folded rows leave the ENSG key space -> keyed by the proteoform ID
    pid = grp["group_canonical_symbol"]
    val = out.set_index(["Ensembl_Gene_ID", "cancer_code"])["expression"]
    assert val[(pid, "CA")] == 15.0            # 10 + 5 within cohort CA
    assert val[(pid, "CB")] == 7.0             # 7 + NaN -> 7 (NaN ignored)
    assert val[("ENSG00000000003", "CA")] == 99.0   # ungrouped keeps its ENSG
    # merged row's id AND symbol are the proteoform ID
    assert (out[out["Ensembl_Gene_ID"] == pid]["Symbol"] == pid).all()


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
    # MAGEA9 + MAGEA9B -> one row keyed by the proteoform ID (one symbol is a
    # prefix of the other, so the ID slashes the full names), value = their sum
    b = base[base["Symbol"].isin(["MAGEA9", "MAGEA9B"])]["expression"]
    c = coll[coll["Symbol"] == "MAGEA9/MAGEA9B"]["expression"]
    assert len(c) == 1
    assert abs(float(c.iloc[0]) - float(b.sum())) < 1e-6
    assert not coll["Symbol"].isin(["MAGEA9", "MAGEA9B"]).any()


def test_cdna_groups_table_invariants():
    from pirlygenes.expression.protein_groups import cdna_identical_groups
    df = cdna_identical_groups()
    assert (df["n_members"] >= 2).all()
    assert (df["cds_nt"] >= 90).all()
    for canon, grp in df.groupby("group_canonical_ensembl_gene_id"):
        assert canon in set(grp["ensembl_gene_id"])
        assert canon == min(grp["ensembl_gene_id"])


def test_accessor_collapse_cdna_identical_behaviour():
    """The universal read-recovery collapse: cDNA-identical antigens merge,
    cDNA-DISTINCT paralogs (MAGEA3 vs MAGEA6) stay separate, the CT47A override
    unifies the whole antigen, and the histone *cluster* is NOT swept (only its
    exact-duplicate pairs)."""
    from pirlygenes.expression.accessors import cancer_reference_expression as cre
    base = cre(cancer_types=["SKCM"], normalize="tpm")
    roll = cre(cancer_types=["SKCM"], normalize="tpm", collapse_cdna_identical=True)
    assert len(roll) < len(base)

    def nrows(df, pat):
        return int(df["Symbol"].astype(str).str.match(pat).sum())

    # NY-ESO CTAG1A/CTAG1B are cDNA-identical -> one row, summed, keyed by the
    # proteoform ID CTAG1A/B in BOTH the id and symbol columns (the member loci
    # leave the ENSG key space)
    b = base[base["Symbol"].isin(["CTAG1A", "CTAG1B"])]["expression"]
    r = roll[roll["Symbol"] == "CTAG1A/B"]
    assert len(r) == 1 and abs(float(r["expression"].iloc[0]) - float(b.sum())) < 1e-6
    assert r["Ensembl_Gene_ID"].iloc[0] == "CTAG1A/B"          # keyed by proteoform ID
    assert not roll["Symbol"].isin(["CTAG1A", "CTAG1B"]).any()  # member loci folded away
    # MAGEA3 vs MAGEA6 are cDNA-DISTINCT (95.9% aa) -> stay separate
    assert roll["Symbol"].isin(["MAGEA3", "MAGEA6"]).sum() == 2
    # CT47A: override unifies all 12 loci into one (CT47B1 is a different antigen)
    assert nrows(roll, "CT47A") == 1
    # histone cluster: only exact-duplicate pairs merge, NOT the whole cluster
    assert nrows(base, "H4C") - nrows(roll, "H4C") <= 3


def test_parameterized_fold_core():
    """The kind= core is the single implementation; the per-space aliases delegate
    to it identically."""
    from pirlygenes.expression import protein_groups as pg
    syms = ["CTAG1B", "CGB3", "PRAME"]
    ids = ["ENSG00000184033", "ENSG00000141510"]
    # kind selects the space; cDNA vs protein differ on protein-identical-but-
    # cDNA-distinct loci (CGB3 -> CGB3 cdna vs CGB3/5/8 protein)
    assert pg.fold_symbols(syms, kind="cdna") == pg.fold_to_cdna_canonical_symbol(syms)
    assert pg.fold_symbols(syms, kind="protein") == pg.fold_symbols_to_canonical(syms)
    assert pg.fold_ids(ids, kind="cdna") == pg.fold_to_cdna_canonical_id(ids)
    assert pg.fold_ids(ids, kind="protein") == pg.fold_to_protein_canonical_id(ids)
    assert pg.fold_symbols(["CGB3"], kind="cdna") == ["CGB3"]
    assert pg.fold_symbols(["CGB3"], kind="protein") == ["CGB3/5/8"]
    # maps: aliases are copies of the core
    assert pg.cdna_member_to_canonical() == pg.member_to_canonical("cdna")
    assert pg.protein_canonical_id_to_symbol() == pg.canonical_to_symbol("protein")


def test_fold_to_cdna_canonical_id():
    """ENSG analog of the symbol fold: a member ENSG -> the proteoform key, an
    ungrouped ENSG -> itself, de-duplicated."""
    from pirlygenes.expression.protein_groups import fold_to_cdna_canonical_id
    assert fold_to_cdna_canonical_id(["ENSG00000184033"]) == ["CTAG1A/B"]   # CTAG1B
    assert fold_to_cdna_canonical_id(["ENSG00000141510"]) == ["ENSG00000141510"]  # TP53
    assert fold_to_cdna_canonical_id(
        ["ENSG00000184033", "ENSG00000268651"]) == ["CTAG1A/B"]   # both -> one key


def test_dual_gene_proteoform_identifiers():
    """The accessor exposes a gene view and a proteoform view that share one
    schema and bridge via Proteoform_ID / Member_Ensembl_Gene_IDs."""
    from pirlygenes.expression.accessors import cancer_reference_expression as cre
    gene = cre(cancer_types=["SKCM"], normalize="tpm")
    prot = cre(cancer_types=["SKCM"], normalize="tpm", collapse_cdna_identical=True)
    cols = {"Ensembl_Gene_ID", "Proteoform_ID", "Member_Ensembl_Gene_IDs", "Symbol"}
    assert cols <= set(gene.columns) and cols <= set(prot.columns)
    # gene frame: CTAG1B keeps its ENSG but bridges to the proteoform
    g = gene[gene["Ensembl_Gene_ID"] == "ENSG00000184033"].iloc[0]
    assert g["Proteoform_ID"] == "CTAG1A/B"
    assert g["Member_Ensembl_Gene_IDs"] == "ENSG00000184033"
    # proteoform frame: one CTAG1A/B row carrying its constituent ENSGs
    pr = prot[prot["Proteoform_ID"] == "CTAG1A/B"]
    assert len(pr) >= 1
    assert set(pr.iloc[0]["Member_Ensembl_Gene_IDs"].split(";")) == {
        "ENSG00000184033", "ENSG00000268651"}
    # the dual columns survive pooling (pool groups on them, so they round-trip)
    pooled = cre(cancer_types=["SKCM"], normalize="tpm",
                 collapse_cdna_identical=True, pool=True)
    assert cols <= set(pooled.columns)
    pp = pooled[pooled["Proteoform_ID"] == "CTAG1A/B"]
    assert len(pp) == 1 and set(
        pp.iloc[0]["Member_Ensembl_Gene_IDs"].split(";")) == {
        "ENSG00000184033", "ENSG00000268651"}
    # Proteoform_ID is built with the maps MATCHING the collapse, so it equals the
    # row key on BOTH collapse frames (cDNA and protein), not just cDNA.
    for kw in (dict(collapse_cdna_identical=True),
               dict(collapse_protein_identical=True)):
        df = cre(cancer_types=["SKCM"], normalize="tpm", **kw)
        assert (df["Ensembl_Gene_ID"].astype(str)
                == df["Proteoform_ID"].astype(str)).all()


def test_fold_resolves_display_aliases_up_front():
    """Synonym->canonical happens up front: a panel named in display space lands
    in the SAME proteoform space as member symbols, never leaking through."""
    from pirlygenes.expression.protein_groups import (
        fold_symbols_to_canonical, fold_to_cdna_canonical_symbol)
    for fold in (fold_to_cdna_canonical_symbol, fold_symbols_to_canonical):
        assert fold(["NY-ESO-1"]) == ["CTAG1A/B"]      # display alias of a GROUP
        assert fold(["CTAG1B"]) == ["CTAG1A/B"]        # member symbol -> same
        # de-dups when alias + member of the same group are both given
        assert fold(["NY-ESO-1", "CTAG1B"]) == ["CTAG1A/B"]
        # single-locus current symbols pass through; grouped-only folding never
        # inverts a current symbol onto an old alias (regression guard for the
        # removed backwards PVRL4->NECTIN4 entry)
        assert fold(["NECTIN4"]) == ["NECTIN4"]
        assert fold(["CD274"]) == ["CD274"]


def test_cta_proteoform_panels_match_collapsed_frame():
    """The public folded CTA panels select the clustered antigens (NY-ESO etc.)
    on a proteoform-collapsed frame, so consumers don't have to fold themselves."""
    from pirlygenes.gene_sets_cancer import (CTA_proteoform_ids,
                                             CTA_proteoform_symbols)
    from pirlygenes.expression.accessors import cancer_reference_expression as cre
    psy, pid = set(CTA_proteoform_symbols()), set(CTA_proteoform_ids())
    assert "CTAG1A/B" in psy and "CTAG1A/B" in pid    # NY-ESO folded
    assert "CTAG1B" not in psy                         # member symbol folds away
    df = cre(cancer_types=["SKCM"], normalize="tpm", collapse_cdna_identical=True)
    by_sym = set(df[df["Symbol"].isin(psy)]["Proteoform_ID"])
    by_id = set(df[df["Ensembl_Gene_ID"].isin(pid)]["Proteoform_ID"])
    assert "CTAG1A/B" in by_sym and "CTAG1A/B" in by_id  # both selection paths hit it


def test_proteoform_id_construction():
    """A folded group's ID is the merged member symbols, so it shows exactly what
    was combined and is unique by construction."""
    # shared prefix -> factor it out, slash the distinct suffixes
    assert proteoform_id(["XAGE1A", "XAGE1B"]) == "XAGE1A/B"
    assert proteoform_id(["CTAG1B", "CTAG1A"]) == "CTAG1A/B"           # order-independent
    assert proteoform_id(["CT47A2", "CT47A10", "CT47A1"]) == "CT47A1/2/10"  # natural order
    # no shared prefix -> slash full symbols (natural-sorted)
    assert proteoform_id(["SOD2", "FOO9"]) == "FOO9/SOD2"
    # one symbol is a prefix of the other -> slash full symbols
    assert proteoform_id(["MAGEA9", "MAGEA9B"]) == "MAGEA9/MAGEA9B"
    # identical / duplicate names -> use one
    assert proteoform_id(["FOO", "FOO"]) == "FOO"
    # NaN/empty members ignored
    assert proteoform_id(["XAGE1A", "XAGE1B", "nan", ""]) == "XAGE1A/B"
    assert proteoform_id([]) is None


def test_group_csv_ids_unique_and_wellformed():
    """The shipped derived tables: every group's proteoform ID is unique and is
    not bounded by separator punctuation."""
    for df in (protein_identical_groups(), cdna_identical_groups()):
        per_id = df.groupby("group_canonical_symbol"
                            )["group_canonical_ensembl_gene_id"].nunique()
        assert (per_id == 1).all(), dict(per_id[per_id > 1])
        for sym in df["group_canonical_symbol"].dropna().unique():
            s = str(sym)
            assert s[0].isalnum() and s[-1].isalnum(), s


def test_collapse_is_idempotent():
    df = protein_identical_groups()
    canon = df[df["n_members"] == 2].iloc[0]["group_canonical_ensembl_gene_id"]
    members = df[df["group_canonical_ensembl_gene_id"] == canon]["ensembl_gene_id"].tolist()
    frame = _cohort(members, ["A", "Ab"], t1=[2.0, 3.0])
    once = collapse_protein_identical_loci(frame)
    twice = collapse_protein_identical_loci(once)
    pd.testing.assert_frame_equal(once, twice)
    assert once["t1"].iloc[0] == 5.0


# ---- opt-in curated-panel proteoform view (derived; never hand-curated) -------

def test_annotate_panel_proteoforms_marks_multilocus_only():
    """The panel accessor fills proteoform_id + member symbols for a multi-locus
    gene and leaves single-locus genes blank; the scalar helper agrees."""
    g = protein_identical_groups()
    grp = g[g["n_members"] >= 2].iloc[0]
    member_ensg = str(grp["ensembl_gene_id"])
    pid = str(grp["group_canonical_symbol"])
    TP53 = "ENSG00000141510"  # single-locus sanity anchor
    assert proteoform_group_of(TP53) is None

    panel = pd.DataFrame({"Ensembl_Gene_ID": [member_ensg, TP53]})
    out = annotate_panel_proteoforms(panel)
    assert out.loc[0, "proteoform_id"] == pid
    assert out.loc[0, "proteoform_members"]                       # member symbols
    assert member_ensg in out.loc[0, "proteoform_member_ensembl_ids"]
    assert out.loc[0, "proteoform_n_members"] >= 2                # >=2 loci
    assert out.loc[1, "proteoform_id"] == ""                      # TP53 single-locus
    assert out.loc[1, "proteoform_members"] == ""
    assert out.loc[1, "proteoform_member_ensembl_ids"] == ""
    assert out.loc[1, "proteoform_n_members"] == 0

    # n_members counts distinct LOCI, so same-symbol PAR X/Y groups (whose member
    # symbols dedup to one) still report >=2.
    scalar = proteoform_group_of(member_ensg)
    assert scalar["n_members"] == len(scalar["member_ensembl_gene_ids"]) >= 2

    # a ';'-joined id cell (rule-table style) resolves each ENSG independently
    out2 = annotate_panel_proteoforms(
        pd.DataFrame({"Ensembl_Gene_ID": [f"{member_ensg};{TP53}"]}))
    assert out2.loc[0, "proteoform_id"] == pid
    # original CSVs are never mutated by the accessor
    assert "proteoform_id" not in panel.columns


def test_no_curated_panel_uses_a_nameless_proteoform_member():
    """Curated gene-set panels must reference the *named* ENSG of a gene, not a
    symbol-less alt-locus member of a protein-identical group (the ECSCR /
    PCDH20 class). A nameless member means the symbol<->ENSG pairing is wrong
    (the symbol's real locus is the named one), so the panel would mis-fold."""
    import re
    from pirlygenes.load_dataset import get_data
    g = protein_identical_groups()
    nameless = {str(e) for e, s in zip(g["ensembl_gene_id"], g["symbol"])
                if str(s).strip() == str(e).strip() or str(s).strip().lower() == "nan"}
    panels = [
        "cancer-key-genes", "cancer-fusions", "fusion-surrogate-expression",
        "rare-cancer-rna-surrogates", "rare-cancer-fusion-rules",
        "fusion-expression-effects", "cancer-lineage-panels", "lineage-genes",
        "therapy-response-signatures", "cancer-family-panels",
        "cancer-compartment-panels", "cancer-supertype-panels",
        "cancer-type-discriminators", "surface-proteins",
    ]
    ensg_re = re.compile(r"ENSG\d{11}")
    bad = []
    for name in panels:
        try:
            df = get_data(name)
        except Exception:
            continue
        for c in [c for c in df.columns if re.search(r"ensembl|ensg", c, re.I)]:
            for cell in df[c].dropna().astype(str):
                for e in ensg_re.findall(cell):
                    if e in nameless:
                        bad.append(f"{name}::{c} = {e}")
    assert not bad, ("curated panels reference nameless proteoform-group members "
                     "(use the gene's named ENSG):\n  " + "\n  ".join(bad))
