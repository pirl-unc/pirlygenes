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


def test_tcga_selects_as_treehouse_not_a_fake_kind():
    """TCGA is Treehouse-reprocessed, so it selects under source_kind='treehouse'
    (the processing source) — there is no 'tcga' kind. source_cohort= gives the
    finer origin-level precision (the specific Treehouse TCGA subset)."""
    th = cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"], source_kind="treehouse")
    assert any("TCGA" in s for s in th["source_cohort"].unique())  # TCGA in treehouse:
    assert cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"], source_kind="tcga").empty  # no tcga kind
    sub = cancer_reference_expression(
        cancer_types="SARC", genes=["TP53"],
        source_cohort="TREEHOUSE_POLYA_25_01_TCGA_SUBSET")
    assert set(sub["source_cohort"].unique()) == {"TREEHOUSE_POLYA_25_01_TCGA_SUBSET"}


def test_pool_collapses_multisource_n_weighted_per_gene_availability():
    """pool=True collapses a multi-source code's per-cohort rows into one
    n-weighted pooled row per gene, pooling only the cohorts that measured the
    gene (per-gene availability)."""
    base = cancer_reference_expression(cancer_types="NUTM", normalize="tpm")
    pooled = cancer_reference_expression(cancer_types="NUTM", normalize="tpm",
                                         pool=True)
    assert base["source_cohort"].nunique() > 1            # genuinely multi-source
    assert pooled.groupby("Ensembl_Gene_ID").size().max() == 1   # one row per gene
    assert (pooled["source_cohort"] == "POOLED").all()
    assert pooled["q1"].isna().all() and pooled["q3"].isna().all()  # quantiles dropped

    # a gene measured in BOTH cohorts: n_samples-weighted mean of per-cohort values
    avail = base.dropna(subset=["expression"])
    both = avail.groupby("Symbol")["source_cohort"].nunique()
    sym = both[both == 2].index[0]
    rows = base[base["Symbol"] == sym]
    w = rows["n_samples"]
    expect = (w * rows["expression"]).sum() / w.sum()
    got = pooled.loc[pooled["Symbol"] == sym, "expression"].iloc[0]
    assert abs(got - expect) < 1e-6
    # pooled n_samples = summed sample count of the measuring cohorts
    assert pooled.loc[pooled["Symbol"] == sym, "n_samples"].iloc[0] == w.sum()


def test_pool_is_noop_for_single_source_code():
    base = cancer_reference_expression(cancer_types="SKCM", normalize="tpm",
                                       genes=["TP53", "MLANA"])
    pooled = cancer_reference_expression(cancer_types="SKCM", normalize="tpm",
                                         genes=["TP53", "MLANA"], pool=True)
    # single source -> same gene count, values unchanged (just relabelled POOLED)
    assert len(pooled) == base["Ensembl_Gene_ID"].nunique()
    for sym in ["TP53", "MLANA"]:
        b = base.loc[base["Symbol"] == sym, "expression"].iloc[0]
        p = pooled.loc[pooled["Symbol"] == sym, "expression"].iloc[0]
        assert abs(b - p) < 1e-6
