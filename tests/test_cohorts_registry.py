"""Centralized per-sample cohort/source registry (#329).

cohorts.py is the single source of truth for which per-sample expression sources
exist and which cohorts they hold (explicit map for treehouse-polya; code==stem
discovery from the cached derived parquets elsewhere)."""

import importlib.util
from pathlib import Path

import pandas as pd

from pirlygenes import cohorts

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_iter_per_sample_dedups_by_code_richest_source_wins(tmp_path, monkeypatch):
    """The medoid/percentile bundle is keyed by code; when two sources both have
    a code on disk, iter_per_sample_cohorts yields it once — the richest source
    (most samples) wins, so e.g. chordoma resolves to GSE239531 (n=23) over
    treehouse-ribod (n=2), and a later source can't clobber a richer one."""
    def _mk(sub, n_samples):
        d = tmp_path / sub / "derived"
        d.mkdir(parents=True)
        cols = {"Ensembl_Gene_ID": ["ENSG00000141510"], "Symbol": ["TP53"]}
        for i in range(n_samples):
            cols[f"s{i}"] = [float(i + 1)]
        pd.DataFrame(cols).to_parquet(
            d / "SARC_CHOR_per_sample_tpm.parquet", index=False)
    _mk("src-small", 2)
    _mk("src-big", 5)
    monkeypatch.setattr(cohorts.downloads, "source_cache_dir",
                        lambda source_id, **k: tmp_path / source_id)
    # src-small is registered FIRST, but src-big has more samples and must win.
    monkeypatch.setattr(cohorts, "PER_SAMPLE_SOURCES",
                        {"src-small": ("S", "x"), "src-big": ("B", "x")})
    pairs = list(cohorts.iter_per_sample_cohorts())
    assert [c.source_id for c, _ in pairs] == ["src-big"]   # richest wins
    # opt out -> both (source-registration order)
    pairs_all = list(cohorts.iter_per_sample_cohorts(unique_by_code=False))
    assert {c.source_id for c, _ in pairs_all} == {"src-small", "src-big"}


# --- the single declarative registry of every Treehouse per-sample cohort -----

def test_registry_has_no_duplicate_codes():
    codes = [c.code for c in cohorts._PER_SAMPLE_COHORTS]
    assert len(codes) == len(set(codes)), "duplicate cohort code in registry"


def test_registry_covers_the_expected_cohort_set():
    """Lock the exact (code -> stem) set per source so any add/rename/restem is
    a deliberate, reviewed registry edit (the read path and the sweeps both read
    this — neither enumerates cohorts independently)."""
    polya = {c.code: c.stem for c in cohorts._PER_SAMPLE_COHORTS
             if c.source_id == "treehouse-polya-25-01"}
    # 26 polya-direct (stem==code) + 30 tcga-direct + 2 glioma + 13 molecular/
    # histology splits = 71.
    assert len(polya) == 71
    # spot-check each stem rule
    assert polya["ATRT"] == "ATRT"                  # pediatric direct
    assert polya["NPC"] == "NPC"                    # rare-subtype direct
    assert polya["SARC_GIST"] == "SARC_GIST"        # treehouse-direct GIST
    assert polya["PRAD"] == "tcga_prad"             # tcga direct
    assert polya["GBM"] == "tcga_gbm"               # glioma split
    assert polya["BRCA_LumA"] == "tcga_brca_luma"   # pam50
    assert polya["HNSC_HPVpos"] == "tcga_hnsc_hpv_pos"  # hpv
    assert polya["LUAD_STK11"] == "tcga_luad_stk11"     # mutation
    assert polya["SARC_WDLPS"] == "tcga_sarc_wdlps"     # histology overlay
    ribod = {c.code: c.stem for c in cohorts._PER_SAMPLE_COHORTS
             if c.source_id == "treehouse-ribod-25-01"}
    assert ribod == {"SARC_CHOR": "SARC_CHOR", "RB": "RB"}


def test_neuroendocrine_cohorts_declared_for_all_ne_sources():
    """The NE per-sample cohorts are declared in the registry (not only
    discovered from disk), so cohorts.py can enumerate every per-sample cohort
    without the cache present."""
    by_source = {sid: set(cohorts.cohorts_for_source(sid)) for sid in
                 ("gse118014-pannet", "sclc-ucologne-2015", "drmetrics-lnen-2020")}
    assert by_source == {
        "gse118014-pannet": {"NET_PANCREAS"},
        "sclc-ucologne-2015": {"SCLC", "SCLC_ASCL1", "SCLC_NEUROD1",
                               "SCLC_POU2F3", "SCLC_YAP1"},
        "drmetrics-lnen-2020": {"NET_LUNG", "NEC_LUNG_LARGECELL"}}
    # NE stems are code==stem
    for sid in by_source:
        for c in cohorts.cohorts_for_source(sid).values():
            assert c.stem == c.code


def test_groups_partition_the_registry():
    """Every cohort belongs to exactly one build group; the groups together
    cover the whole registry (no orphan rows a sweep would never build)."""
    by_group = {}
    for c in cohorts._PER_SAMPLE_COHORTS:
        by_group.setdefault(c.group, []).append(c.code)
    assert sum(len(v) for v in by_group.values()) == len(cohorts._PER_SAMPLE_COHORTS)
    assert set(by_group) == {
        "polya_pediatric", "sarc_rare_direct", "sarc_subtypes",
        "sarc_rare_overlay", "tcga_direct", "tcga_glioma",
        "tcga_brca_pam50", "tcga_hnsc_hpv", "tcga_luad_mut", "ribod",
        "neuroendocrine", "sclc_tf_subtype"}


def test_cohorts_for_group_filters_by_group():
    pam50 = cohorts.cohorts_for_group("tcga_brca_pam50")
    assert {c.code for c in pam50} == {
        "BRCA_Basal", "BRCA_HER2", "BRCA_LumA", "BRCA_LumB", "BRCA_Normal"}
    assert all(c.selection.startswith("pam50:") for c in pam50)
    assert cohorts.cohorts_for_group("not-a-group") == []


def test_static_sweeps_consume_the_registry():
    """The build sweeps build their cohort list FROM the registry (not an inline
    copy) — so the two can't drift. Verified for the three network-free sweeps
    whose COHORTS are built at import time."""
    for script, group in [
            ("sweep_treehouse_polya_cohorts", "polya_pediatric"),
            ("sweep_treehouse_ribod_cohorts", "ribod"),
            ("sweep_treehouse_tcga_cohorts", "tcga_direct")]:
        mod = _load_script(script)
        built = {c.cancer_code: c.effective_cache_stem for c in mod.COHORTS}
        expected = {c.code: c.stem for c in cohorts.cohorts_for_group(group)}
        assert built == expected, (script, built, expected)
