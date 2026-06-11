"""The all:-union expression contract (#366): a computed-aggregate code expands
to member cohorts with mixed assays/pipelines and different gene universes, so
the reference must (a) preserve per-cohort provenance, (b) never pre-pool, and
(c) offer a pipeline-homogeneous view via exclude_microarray_proxy."""

from pirlygenes.expression.accessors import cancer_reference_expression


def test_union_preserves_per_cohort_pipeline_provenance():
    d = cancer_reference_expression(cancer_types="SARC", genes=["TP53"])
    # union expands to many member atoms, each tagged with its own pipeline
    assert d["cancer_code"].nunique() > 5
    assert "processing_pipeline" in d.columns
    assert d["processing_pipeline"].nunique() > 1  # heterogeneous pipelines


def test_exclude_microarray_proxy_yields_homogeneous_view():
    full = cancer_reference_expression(cancer_types="SARC", genes=["TP53"])
    homo = cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"], exclude_microarray_proxy=True)
    # the microarray-proxy members (cross-platform-incomparable TPM) are dropped
    assert full["processing_pipeline"].str.contains("microarray", na=False).any()
    assert not homo["processing_pipeline"].str.contains("microarray", na=False).any()
    assert homo["cancer_code"].nunique() < full["cancer_code"].nunique()


def test_absent_genes_are_absent_not_zero():
    # a gene present in only some member cohorts returns rows only for those
    # cohorts (not_measurable elsewhere), never fabricated 0 rows.
    d = cancer_reference_expression(cancer_types="SARC", genes=["TP53"])
    assert (d["expression"].fillna(0) >= 0).all()
    # every returned row is a real measurement (has a source cohort)
    assert d["source_cohort"].notna().all()


def test_source_kind_selector_filters_by_cohort_kind():
    """#366 source:node selector — source_kind keeps only members of the given
    cohort kind(s); None (default) is the full all:-union."""
    allu = cancer_reference_expression(cancer_types="SARC", genes=["TP53"])
    geo = cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"], source_kind="geo")
    th = cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"], source_kind="treehouse")
    assert geo["source_cohort"].nunique() < allu["source_cohort"].nunique()
    assert geo["source_cohort"].str.startswith("GSE").all()        # geo only
    assert th["source_cohort"].str.startswith("TREEHOUSE").all()   # treehouse only
    # a list of kinds unions them
    both = cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"], source_kind=["geo", "treehouse"])
    assert both["source_cohort"].nunique() == allu["source_cohort"].nunique()
