"""Ontology integrity + coverage guards for the cancer-type registry.

Cross-reference invariants that keep the taxonomy 'well curated' and drift-proof
(the class of corruption #377/#379 were about), plus a printed coverage report
across the per-node data axes (lineage / TMB / indel / aPD1).
"""

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    cancer_type_subtypes_of,
    resolve_cancer_type,
)
from pirlygenes.load_dataset import get_data


def _reg():
    return cancer_type_registry().set_index("code")


def test_all_data_table_codes_resolve_to_registry():
    """Every cancer code referenced by a per-node data table must be a real
    registry code (no orphan/typo codes — the silent-drift class)."""
    bad = []
    for name, col in [("cancer-tmb", "cancer_code"),
                      ("cancer-frameshift-burden", "cancer_code"),
                      ("cancer-apd1-response", "cancer_code"),
                      ("cancer-subtype-groupings", "member_code")]:
        for code in get_data(name)[col]:
            if resolve_cancer_type(code) is None:
                bad.append((name, code))
    assert not bad, f"data tables reference unknown cancer codes: {bad}"


def test_viral_etiology_has_agent():
    """A non-``none`` viral_etiology must name its viral_agent."""
    reg = _reg()
    bad = [c for c, r in reg.iterrows()
           if str(r["viral_etiology"]) in ("defining", "subset")
           and not str(r["viral_agent"]).strip()]
    assert not bad, f"viral_etiology set but no viral_agent: {bad}"


def test_fusion_driven_has_a_fusion_row():
    """A top-level fusion-defining entity should have a cancer-fusions row
    (cross-reference between the registry flag and the cited fusion table)."""
    reg = _reg()
    fus = set(get_data("cancer-fusions")["cancer_code"])
    bad = [c for c, r in reg.iterrows()
           if str(r["fusion_driven"]) in ("defining",)
           and not str(r["parent_code"]).strip() and c not in fus]
    assert not bad, f"fusion_driven=defining but no cancer-fusions row: {bad}"


def test_computed_aggregates_have_members():
    """A computed-aggregate node must resolve to members — either parent_code
    children or an explicit cancer-cohort-aggregates mapping."""
    reg = _reg()
    agg = set(get_data("cancer-cohort-aggregates")["aggregate_code"])
    bad = []
    for c, r in reg.iterrows():
        if str(r["expression_source"]) == "computed":
            if not cancer_type_subtypes_of(c) and c not in agg:
                bad.append(c)
    assert not bad, f"computed aggregate with no members: {bad}"


def test_coverage_report():
    """Non-failing: print per-axis coverage. Denominator is the set of
    cancer types that SHOULD carry their own value — real top-level codes
    (not subtypes, which inherit from their parent, nor computed/curated-
    placeholder nodes). Raw all-159-code percentages understate coverage badly.

    indel/aPD1 are intentionally sparse: published indel-load and anti-PD-1
    *monotherapy* ORR only exist for ~TCGA / trialled cancers, so those gaps are
    largely irreducible (and must never be fabricated)."""
    reg = _reg()
    real_top = {c for c, r in reg.iterrows()
                if (pd.isna(r["parent_code"]) or not str(r["parent_code"]).strip())
                and str(r["expression_source"]) not in ("computed", "curated")}
    from pirlygenes.gene_sets_cancer import cancer_tmb

    lin = get_data("lineage-genes")
    lin_ct = {c for c, g in lin.groupby("Cancer_Type") if len(g) >= 5}
    tmb_ct = {c for c in real_top if cancer_tmb(c) is not None}  # parent-inherits
    ind_ct = set(get_data("cancer-frameshift-burden")["cancer_code"])
    apd1_ct = set(get_data("cancer-apd1-response")["cancer_code"])
    n = len(real_top)
    print("\n[ontology coverage] %d real top-level cancer types" % n)
    for label, s in [("lineage(>=5)", lin_ct), ("TMB", tmb_ct),
                     ("indel", ind_ct), ("aPD1(mono)", apd1_ct)]:
        have = len(s & real_top)
        print(f"  {label:13s} {have:3d}/{n} ({100 * have // n}%)")
    # lineage is complete for every real top-level type.
    assert lin_ct >= real_top, f"real types missing a lineage panel: {sorted(real_top - lin_ct)}"
    # TMB is complete except the documented single-driver heme neoplasms with no
    # reliable entity-median TMB (CML BCR-ABL / MPN JAK2-CALR-MPL) - deliberately
    # blank per the oncology-table lit audit, NOT fabricated.
    _TMB_IRREDUCIBLE = {"CML", "MPN"}
    assert (real_top - tmb_ct) <= _TMB_IRREDUCIBLE, \
        f"real types missing TMB (beyond documented blanks): {sorted(real_top - tmb_ct - _TMB_IRREDUCIBLE)}"
