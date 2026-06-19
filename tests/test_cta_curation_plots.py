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
    # the big curated databases contribute genes
    assert sets["CTpedia"]
    assert sets["CTexploreR"]
    assert sets["daSilva2017_protein"]


def test_render_returns_five_figures_and_writes_them(tmp_path: Path):
    result = ccp.render(out_dir=tmp_path)
    assert set(result["paths"]) == set(ccp.FILENAMES)
    assert result["n_genes"] > 0
    for path in result["paths"].values():
        assert path.exists() and path.stat().st_size > 0
