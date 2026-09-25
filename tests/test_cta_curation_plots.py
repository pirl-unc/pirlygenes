"""Unit tests for the packaged CTA-curation figure generator
(``pirlygenes.cta_curation_plots``) — the logic behind ``pirlygenes plot
cta-curation`` and the docs/cta-curation.md figures."""
from pathlib import Path

from pirlygenes import cta_curation_plots as ccp


def test_per_source_counts_partition_sums_to_total():
    """Each source's kept_confident + kept_weak + excluded must equal its total
    (the outcome categories partition the source's genes)."""
    rows = ccp._per_source_counts(ccp._evidence())
    assert rows
    for r in rows:
        assert r["kept_confident"] + r["kept_weak"] + r["excluded"] == r["total"]
        assert r["total"] > 0
    # rows are ordered largest-source-first (drives the funnel/outcome plots)
    totals = [r["total"] for r in rows]
    assert totals == sorted(totals, reverse=True)


def test_tag_sets_cover_primary_sources():
    sets = ccp._tag_sets(ccp._evidence())
    assert set(sets) == set(ccp.PRIMARY_SOURCES)
    assert len(sets) == 10
    assert all(sets.values())


def test_render_returns_figures_and_writes_them(tmp_path: Path):
    result = ccp.render(out_dir=tmp_path)
    expected = set(ccp.FILENAMES)
    if ccp.publication_data_available():
        expected.update(ccp.PUBLICATION_FILENAMES)
    assert set(result["paths"]) == expected
    assert result["n_genes"] > 0
    for path in result["paths"].values():
        assert path.exists() and path.stat().st_size > 0
        assert path.with_suffix(".pdf").is_file()


def test_funnel_lands_on_the_actual_public_default_set():
    from oncoref.cta import cta_gene_ids, cta_unfiltered_gene_ids

    table = ccp.stage_membership()
    assert set(table.loc[table.default_panel, "Ensembl_Gene_ID"]) == cta_gene_ids()
    assert set(table.loc[table.non_cta_removed, "Ensembl_Gene_ID"]) == cta_unfiltered_gene_ids() & set(ccp._evidence().Ensembl_Gene_ID)
    assert not (table.default_panel & ~table.hpa_restriction).any()
    counts = ccp.stage_counts()
    assert sum(row["dropped"] for row in counts) + counts[-1]["remaining"] == len(table)


def test_false_strings_are_not_truthy_filter_passes():
    import pandas as pd

    assert ccp._bool_series(pd.Series(["False", "True", None, False, True])).tolist() == [
        False, True, False, False, True,
    ]


def test_overlap_matrix_is_symmetric_and_defaults_are_subsets():
    counts = ccp.source_overlap_counts()
    assert (counts.default_overlap <= counts.candidate_overlap).all()
    for field in ("candidate_overlap", "default_overlap"):
        matrix = counts.pivot(index="source_a", columns="source_b", values=field)
        assert matrix.equals(matrix.T)
    for name, ids in ccp._tag_sets(ccp._evidence()).items():
        if ids:
            diagonal = counts[(counts.source_a == name) & (counts.source_b == name)]
            assert diagonal.candidate_overlap.item() == len(ids)


def test_published_placental_overlap_uses_current_coding_identities():
    import pytest

    if not ccp.publication_data_available():
        pytest.skip("Published source tables require the updated oncoref checkout")
    gong, bradley, prior = ccp.placental_source_sets().values()
    assert (len(gong), len(bradley), len(prior)) == (70, 10, 19)
    assert len(gong & bradley) == 7
    assert len(gong & prior) == 16
    from oncoref.cta import cta_gene_ids

    assert len(gong & prior & cta_gene_ids()) == 8


def test_plot_reliability_thresholds_come_from_the_filter_owner():
    from oncoref.cta_tissues import HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS

    for label, threshold in ccp.RELIABILITY_THRESHOLD.items():
        key = "Missing" if label == "no data" else label
        assert threshold == HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS[key]


def test_outcome_counts_use_public_default_and_preserve_all_retained_genes():
    from oncoref import cta
    from oncoref.cta_provenance import candidate_provenance
    from oncoref import cta_curation_plots as owner

    assert ccp.render is owner.render
    raw = ccp._evidence()
    assert len(raw) == 2537
    assert len(cta.cta_gene_ids()) == 624
    assert "TRIM64" not in cta.cta_gene_names()
    assert candidate_provenance().paper_dois.ne("").all()
    for row in ccp._per_source_counts(raw):
        ids = ccp._tag_sets(raw)[row["source"]]
        assert row["kept_confident"] + row["kept_weak"] == len(ids & cta.cta_gene_ids())
