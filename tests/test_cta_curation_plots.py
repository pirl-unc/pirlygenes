"""Unit tests for the packaged CTA-curation figure generator
(``pirlygenes.cta_curation_plots``) — the logic behind ``pirlygenes plot
cta-curation`` and the docs/cta-curation.md figures."""
import hashlib
import json
from pathlib import Path

from pirlygenes import cta_curation_plots as ccp


def test_per_source_counts_partition_sums_to_total():
    """Reviewed inclusion, HPA-only passes and exclusions partition each source."""
    from oncoref.cta import cta_gene_ids

    rows = ccp._per_source_counts(ccp._evidence())
    sets = ccp._tag_sets(ccp._evidence())
    defaults = cta_gene_ids()
    assert rows
    for r in rows:
        assert (r["default_panel"] + r["hpa_pass_outside_default"]
                + r["family_or_hpa_excluded"]) == r["total"]
        assert r["default_panel"] == len(sets[r["source"]] & defaults)
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


def test_render_returns_figures_and_writes_them(tmp_path: Path):
    import oncoref
    import pandas as pd
    from PIL import Image
    from pypdf import PdfReader

    result = ccp.render(out_dir=tmp_path)
    expected = set(ccp.FILENAMES)
    if ccp.publication_data_available():
        expected.update(ccp.PUBLICATION_FILENAMES)
    assert set(result["paths"]) == expected
    assert result["n_genes"] > 0
    for path in result["paths"].values():
        assert path.exists() and path.stat().st_size > 0
        assert path.with_suffix(".pdf").is_file()
        with Image.open(path) as im:
            assert min(im.info["dpi"]) >= 299
        assert PdfReader(path.with_suffix(".pdf")).pages[0].extract_text().strip()
    counts = pd.read_csv(tmp_path / "cta-source-outcome-counts.csv")
    assert counts.set_index("source").loc["CTpedia", "default_panel"] == 186
    provenance = json.loads((tmp_path / "cta-curation-provenance.json").read_text())
    assert provenance["oncoref_version"] == oncoref.__version__
    assert set(provenance["default_gene_ids"]) == oncoref.cta_gene_ids()
    assert provenance["stage_counts"][-1]["remaining"] == 297
    audit = Path(oncoref.__file__).parent / "data" / "cta-specificity-audit.csv"
    assert provenance["input_sha256"][audit.name] == hashlib.sha256(audit.read_bytes()).hexdigest()
    text = PdfReader(result["paths"]["filter_outcome"].with_suffix(".pdf")).pages[0].extract_text()
    assert "default panel" in text and "outside default" in text
    assert "kept (HPA-confident)" not in text


def test_trim64_is_an_hpa_pass_outside_the_reviewed_default():
    import pirlygenes.gene_sets_cancer as gsc
    from pirlygenes import coverage

    gid = "ENSG00000204450"
    table = ccp.stage_membership()
    trim64 = table.set_index("Ensembl_Gene_ID").loc[gid]
    assert trim64.passes_filters and trim64.hpa_restriction
    assert not trim64.default_panel
    assert trim64.specificity_action == "candidate_only"
    assert ccp.stage_counts()[-1]["remaining"] == 297
    assert gid not in gsc.CTA_gene_ids()
    assert gid not in gsc.CTA_filtered_gene_ids()
    assert gid not in gsc.CTA_placental_restricted_gene_ids()
    assert gid not in gsc.CTA_gene_id_to_name()
    assert gid in gsc.CTA_unfiltered_gene_ids()
    assert gid in set(gsc.CTA_evidence().Ensembl_Gene_ID)
    assert gid not in coverage.resolve_gene_set("CTA")[1]

    # Removing the raw candidate changes only the HPA-pass/outside-default bucket.
    with_trim64 = {r["source"]: r for r in ccp._per_source_counts(table)}
    without_trim64 = {r["source"]: r for r in ccp._per_source_counts(
        table.loc[table.Ensembl_Gene_ID.ne(gid)])}
    for source, members in ccp._tag_sets(table).items():
        if source not in with_trim64:
            continue
        before, after = with_trim64[source], without_trim64[source]
        assert before["default_panel"] == after["default_panel"]
        assert before["family_or_hpa_excluded"] == after["family_or_hpa_excluded"]
        assert before["hpa_pass_outside_default"] - after["hpa_pass_outside_default"] == int(gid in members)


def test_funnel_lands_on_the_actual_public_default_set():
    from oncoref.cta import cta_gene_ids, cta_unfiltered_gene_ids
    from oncoref.load_dataset import get_data

    table = ccp.stage_membership()
    raw = get_data("cancer-testis-antigens")
    assert set(table.Ensembl_Gene_ID) == set(raw.Ensembl_Gene_ID)
    assert set(table.loc[table.default_panel, "Ensembl_Gene_ID"]) == cta_gene_ids()
    assert set(table.loc[table.non_cta_removed, "Ensembl_Gene_ID"]) == cta_unfiltered_gene_ids()
    assert not (table.default_panel & ~table.hpa_restriction).any()
    counts = ccp.stage_counts()
    assert [row["remaining"] for row in counts] == [439, 431, 312, 297]
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
