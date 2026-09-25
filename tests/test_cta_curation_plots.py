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
    assert len(sets) == 10
    assert all(sets.values())


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
    assert counts.set_index("source").loc["da Silva 2017", "default_panel"] == 515
    provenance = json.loads((tmp_path / "run-manifest.json").read_text())
    assert provenance["oncoref_version"] == oncoref.__version__
    assert set(provenance["default_gene_ids"]) == oncoref.cta_gene_ids()
    assert provenance["stages"][-1][1] == 624
    audit = Path(oncoref.__file__).parent / "data" / "cta-specificity-audit.csv"
    assert provenance["input_sha256"][audit.name] == hashlib.sha256(audit.read_bytes()).hexdigest()
    text = PdfReader(result["paths"]["filter_outcome"].with_suffix(".pdf")).pages[0].extract_text()
    assert "default panel" in text and "outside default" in text
    assert "kept (HPA-confident)" not in text


def test_trim64_is_an_hpa_pass_outside_the_reviewed_default():
    import pirlygenes.gene_sets_cancer as gsc
    from pirlygenes import coverage

    gid = "ENSG00000204450"
    from oncoref.cta_provenance import legacy_only_candidates
    table = gsc.CTA_evidence()
    trim64 = table.set_index("Ensembl_Gene_ID").loc[gid]
    assert trim64.passes_filters
    assert trim64.specificity_action == "candidate_only"
    assert gid in set(legacy_only_candidates().Ensembl_Gene_ID)
    assert gid not in set(ccp._evidence().Ensembl_Gene_ID)
    assert ccp.stage_counts()[-1]["remaining"] == 624
    assert gid not in gsc.CTA_gene_ids()
    assert gid not in gsc.CTA_filtered_gene_ids()
    assert gid not in gsc.CTA_placental_restricted_gene_ids()
    assert gid not in gsc.CTA_gene_id_to_name()
    assert gid in gsc.CTA_unfiltered_gene_ids()
    assert gid not in coverage.resolve_gene_set("CTA")[1]


def test_funnel_lands_on_the_actual_public_default_set():
    from oncoref.cta import cta_gene_ids, cta_unfiltered_gene_ids
    from oncoref.load_dataset import get_data

    table = ccp.stage_membership()
    raw = get_data("cancer-testis-antigens")
    coding = set(table.loc[table.protein_coding, "Ensembl_Gene_ID"])
    assert coding == set(ccp._evidence().Ensembl_Gene_ID)
    assert coding < set(raw.Ensembl_Gene_ID)
    assert set(table.loc[table.default_panel, "Ensembl_Gene_ID"]) == cta_gene_ids()
    assert set(table.loc[table.non_cta_removed, "Ensembl_Gene_ID"]) == cta_unfiltered_gene_ids() & set(ccp._evidence().Ensembl_Gene_ID)
    assert not (table.default_panel & ~table.hpa_restriction).any()
    counts = ccp.stage_counts()
    assert [row["remaining"] for row in counts] == [3895, 3654, 2537, 2523, 880, 624]
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
