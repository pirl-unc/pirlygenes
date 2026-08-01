"""The pirlygenes cohort API is a read-only oncoref compatibility layer."""

from pathlib import Path

import pandas as pd

from pirlygenes import cohorts


def test_compatibility_registry_covers_every_owner_matrix():
    from oncoref import source_matrices

    owner = source_matrices.registry()
    actual = {cohort.code for cohort in cohorts._PER_SAMPLE_COHORTS}

    assert actual == set(owner["cancer_code"].astype(str))
    assert len(actual) == len(owner)


def test_per_sample_sources_are_derived_from_owner_registry():
    from oncoref.expression_registry import expression_sources

    owner_ids = {source.id for source in expression_sources()}

    assert owner_ids <= set(cohorts.PER_SAMPLE_SOURCES)
    assert {
        "treehouse-polya-25-01",
        "treehouse-ribod-25-01",
        "gse118014-pannet",
        "sclc-ucologne-2015",
    } <= set(cohorts.PER_SAMPLE_SOURCES)
    for source_id, (label, project) in cohorts.PER_SAMPLE_SOURCES.items():
        assert cohorts.source_label(source_id) == label
        assert cohorts.source_project(source_id) == project
    assert cohorts.source_label("not-a-source") is None


def test_compatibility_metadata_fills_incomplete_owner_records():
    assert cohorts.source_label("target-all") == "TARGET_ALL_2018"
    assert cohorts.source_project("target-all") == "TARGET ALL"
    assert cohorts.source_label("cllmap") == "CLLMAP_2022"
    assert cohorts.source_project("cllmap") == "CLL-map"
    assert cohorts.source_label("beataml-ohsu-2022") == "BEATAML_OHSU_2022"
    assert cohorts.source_project("beataml-ohsu-2022") == "BeatAML 1.0"


def test_owner_projects_supersede_retired_compatibility_fallbacks():
    from oncoref.expression_registry import expression_sources

    source_ids = {
        "gse118014-pannet",
        "drmetrics-lnen-2020",
        "gse98894-midnet",
        "gse114922-mds",
        "gse32662-mtc",
        "gse30929-lps",
    }
    owner_projects = {
        source.id: source.source_project
        for source in expression_sources()
        if source.id in source_ids
    }

    assert set(owner_projects) == source_ids
    assert all(owner_projects.values())
    assert {
        source_id: cohorts.source_project(source_id)
        for source_id in source_ids
    } == owner_projects


def test_source_filter_uses_owner_source_cohort_mapping():
    polya = cohorts.cohorts_for_source("treehouse-polya-25-01")
    sclc = cohorts.cohorts_for_source("sclc-ucologne-2015")

    assert {"ATRT", "PRAD", "BRCA_LumA"} <= set(polya)
    assert set(sclc) == {
        "SCLC",
        "SCLC_ASCL1",
        "SCLC_NEUROD1",
        "SCLC_POU2F3",
        "SCLC_YAP1",
    }
    assert all(cohort.stem == cohort.code for cohort in polya.values())
    assert set(
        cohorts.cohorts_for_source(
            "treehouse-polya-25-01",
            include_related=False,
        )
    ) < set(polya)
    assert cohorts.cohorts_for_source("not-a-source") == {}


def test_owner_build_source_ids_delegate_to_selected_matrix_resolver():
    assert set(cohorts.cohorts_for_source("beataml-ohsu-2022")) == {
        "LAML_APL",
        "LAML_ELNadv",
        "LAML_ELNfav",
        "LAML_ELNint",
    }
    assert set(cohorts.cohorts_for_source("tcga-acc")) == {"ACC"}
    assert set(cohorts.cohorts_for_source("prjna1083972-mmnst")) == {
        "SARC_MMNST"
    }
    assert set(cohorts.cohorts_for_source("target-nbl")) == {
        "NBL",
        "NBL_MYCNamp",
        "NBL_MYCNnonamp",
    }


def test_historical_source_ids_resolve_through_selected_matrix_provenance(
    monkeypatch,
):
    expected = {
        "LAML_APL",
        "LAML_ELNadv",
        "LAML_ELNfav",
        "LAML_ELNint",
    }
    assert set(cohorts.cohorts_for_source("beataml-ohsu")) == expected

    from oncoref import source_matrices

    monkeypatch.setattr(
        source_matrices,
        "is_cached",
        lambda code: code in expected,
    )
    assert set(cohorts.available_cohorts("beataml-ohsu")) == expected


def test_legacy_geo_heme_is_a_composite_read_alias():
    assert cohorts.source_label("geo-heme") == "GEO_HEME_2022"
    assert cohorts.source_project("geo-heme") == "GEO"
    actual = cohorts.cohorts_for_source("geo-heme")
    assert set(actual) == {"CML", "MDS", "MCL", "MPN"}
    assert {cohort.source_id for cohort in actual.values()} == {
        "gse100026-cml",
        "gse114922-mds",
        "gse271664-mcl",
        "gse283710-mpn",
    }


def test_available_cohorts_uses_owner_cache_state(monkeypatch):
    from oncoref import source_matrices

    monkeypatch.setattr(
        source_matrices,
        "is_cached",
        lambda code: code in {"ATRT", "PRAD"},
    )

    assert set(cohorts.available_cohorts("treehouse-polya-25-01")) == {
        "ATRT",
        "PRAD",
    }
    assert set(cohorts.all_available_cohorts()) == {"ATRT", "PRAD"}


def test_iteration_filters_by_resolved_codes_for_current_and_legacy_ids(
    monkeypatch,
):
    available = cohorts.cohorts_for_source("beataml-ohsu-2022")
    monkeypatch.setattr(cohorts, "all_available_cohorts", lambda: available)
    monkeypatch.setattr(
        cohorts,
        "read_per_sample",
        lambda cohort: pd.DataFrame({"code": [cohort.code]}),
    )

    expected = set(available)
    for source_id in ("beataml-ohsu-2022", "beataml-ohsu"):
        actual = {
            cohort.code
            for cohort, _frame in cohorts.iter_per_sample_cohorts(
                sources=[source_id]
            )
        }
        assert actual == expected


def test_read_per_sample_fetches_through_owner(tmp_path, monkeypatch):
    from oncoref import source_matrices

    path = tmp_path / "PRAD.parquet"
    expected = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "Symbol": ["TP53"],
            "sample-1": [3.0],
        }
    )
    expected.to_parquet(path, index=False)
    monkeypatch.setattr(source_matrices, "ensure", lambda code: path)

    cohort = cohorts.cohorts_for_source("treehouse-polya-25-01")["PRAD"]
    pd.testing.assert_frame_equal(cohorts.read_per_sample(cohort), expected)


def test_treehouse_groups_delegate_to_owner_builder_registry():
    from oncoref.expression_builders import treehouse_cohorts_for_group

    expected = treehouse_cohorts_for_group("tcga_brca_pam50")
    actual = cohorts.cohorts_for_group("tcga_brca_pam50")

    assert [cohort.code for cohort in actual] == [
        cohort.cancer_code for cohort in expected
    ]
    assert [cohort.selection for cohort in actual] == [
        cohort.selection for cohort in expected
    ]
    assert cohorts.cohorts_for_group("not-a-group") == []


def test_no_local_builder_fleet_or_summary_shards_remain():
    from pirlygenes.expression import stats

    root = Path(__file__).resolve().parent.parent

    # An ignored __pycache__ can survive a branch switch in an existing
    # checkout; the contract is that no importable builder source remains.
    assert not list((root / "pirlygenes" / "builders").glob("*.py"))
    assert not (
        root / "pirlygenes" / "data" / "cancer-reference-expression"
    ).exists()
    for retired in (
        "expression_sources.yaml",
        "cancer-reference-expression-samples.csv.gz",
        "cancer-expression-source-candidates.csv",
        "ncbi-symbol-synonyms.csv.gz",
    ):
        assert not (root / "pirlygenes" / "data" / retired).exists()
    assert not list((root / "scripts").glob("build_*reference_expression.py"))
    assert not list((root / "scripts").glob("sweep_*.py"))
    cta_analysis = (
        root / "analyses" / "cta_patient_counts.py"
    ).read_text(encoding="utf-8")
    assert ".cache/pirlygenes/expression" not in cta_analysis
    assert "_per_sample_tpm.parquet" not in cta_analysis
    for writer in (
        "build_reference_rows",
        "finalize_reference_rows",
        "write_reference_rows",
        "upsert_samples_manifest",
    ):
        assert not hasattr(stats, writer)
