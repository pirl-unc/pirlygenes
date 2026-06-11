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
    """Non-failing: print per-axis coverage so gaps are visible. Data gaps are
    often irreducible (indel/aPD1 only exist for ~TCGA / trialled cancers); this
    surfaces them rather than hiding them."""
    reg = _reg()
    codes = set(reg.index)
    lin = get_data("lineage-genes")
    lin_ct = {c for c, g in lin.groupby("Cancer_Type") if len(g) >= 5}
    tmb = get_data("cancer-tmb")
    tmb_ct = set(tmb[tmb["median_tmb_mut_mb"].notna()]["cancer_code"])
    ind_ct = set(get_data("cancer-frameshift-burden")["cancer_code"])
    apd1_ct = set(get_data("cancer-apd1-response")["cancer_code"])
    n = len(codes)
    print("\n[ontology coverage] %d registry codes" % n)
    for label, s in [("lineage(>=5)", lin_ct), ("TMB", tmb_ct),
                     ("indel", ind_ct), ("aPD1", apd1_ct)]:
        have = len(s & codes)
        print(f"  {label:13s} {have:3d}/{n} ({100 * have // n}%)")
    assert n > 100  # sanity
