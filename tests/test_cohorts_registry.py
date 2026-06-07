"""Centralized per-sample cohort/source registry (#329).

cohorts.py is the single source of truth for which per-sample expression sources
exist and which cohorts they hold (explicit map for treehouse-polya; code==stem
discovery from the cached derived parquets elsewhere)."""

import pandas as pd

from pirlygenes import cohorts


def test_per_sample_sources_complete():
    src = cohorts.PER_SAMPLE_SOURCES
    # all sources the per-sample pipeline produces are registered here, not in a
    # script — incl. ribod + the NE sources added by #318/#326.
    assert {"treehouse-polya-25-01", "treehouse-ribod-25-01", "gse118014-pannet",
            "sclc-ucologne-2015", "drmetrics-lnen-2020"} <= set(src)
    for sid, (label, project) in src.items():
        assert cohorts.source_label(sid) == label
        assert cohorts.source_project(sid) == project
    assert cohorts.source_label("not-a-source") is None


def test_explicit_source_uses_registry_map():
    # treehouse-polya has an explicit code->Cohort map (mixed-case stems)
    th = cohorts.cohorts_for_source("treehouse-polya-25-01")
    assert th["PRAD"].stem == "tcga_prad"           # stem != code
    assert th["ATRT"].stem == "ATRT"


def test_discovery_source_uses_stem_equals_code(tmp_path, monkeypatch):
    """A source without an explicit map discovers cohorts from its derived
    parquet stems, with code == stem."""
    derived = tmp_path / "derived"
    derived.mkdir()
    df = pd.DataFrame({"Ensembl_Gene_ID": ["ENSG00000141510"], "Symbol": ["TP53"],
                       "s1": [10.0], "s2": [20.0]})
    df.to_parquet(derived / "NET_PANCREAS_per_sample_tpm.parquet", index=False)
    monkeypatch.setattr(cohorts.downloads, "source_cache_dir",
                        lambda source_id, **k: tmp_path)
    monkeypatch.setitem(cohorts.PER_SAMPLE_SOURCES, "fake-ne",
                        ("FAKE_NE", "GEO"))
    got = cohorts.cohorts_for_source("fake-ne")
    assert set(got) == {"NET_PANCREAS"}
    assert got["NET_PANCREAS"].stem == "NET_PANCREAS"   # code == stem
    assert "NET_PANCREAS" in cohorts.all_available_cohorts()


def test_unknown_source_empty():
    assert cohorts.cohorts_for_source("nope") == {}
