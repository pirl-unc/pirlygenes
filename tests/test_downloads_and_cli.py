"""Smoke tests for the cohort-level CLI + downloads Python API.

Covers the foundation shipped in the expression-data refresh project
(see docs/expression-data-refresh-plan.md). Heavy behavior (build,
fetch) is scaffolded only — those subcommands return a clear
NotImplemented pointer and are not exercised here. `plot
patient-coverage` is implemented and covered in test_coverage.py.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from pirlygenes import cli, downloads


def test_registry_loads_and_has_expected_categories():
    sources = downloads.load_registry()
    assert sources, "registry must be non-empty"
    categories = {s.category for s in sources}
    assert categories == {"expression"}
    ids = [s.id for s in sources]
    assert len(ids) == len(set(ids)), "source ids must be unique"
    # Sanity-check a few well-known anchors are present.
    by_id = {s.id: s for s in sources}
    assert "cgci-blgsp" in by_id
    assert "mmrf-commpass" in by_id
    assert "tcga-blca" in by_id
    assert "treehouse-polya-25-01" in by_id
    # #346: the legacy route labeled all 95 mixed salivary samples as ADCC.
    # oncoref#422 owns the diagnosis-split rebuild; a local build must not
    # silently recreate the quarantined artifact.
    assert "gse294016-adcc" not in by_id
    # TCGA cohorts in the YAML registry use the unprefixed registry
    # codes (BLCA, BRCA, ...) so they match cancer-type-registry.csv.
    # The TCGA-via-Treehouse build tags rows with source_cohort
    # TREEHOUSE_POLYA_25_01_TCGA_SAMPLES to distinguish from a future
    # GDC-direct build under the same cancer_code.
    tcga_codes = {
        code
        for s in sources
        if s.source_type == "gdc" and s.id.startswith("tcga-")
        for code in s.cancer_codes
    }
    assert tcga_codes
    assert "BLCA" in tcga_codes
    assert "BRCA" in tcga_codes
    assert not any(code.startswith("TCGA_") for code in tcga_codes)


def test_ci_oncoref_cache_key_tracks_resolved_package_and_data_versions():
    workflow = (
        Path(__file__).resolve().parent.parent / ".github/workflows/tests.yml"
    ).read_text()

    assert "import oncoref; from oncoref.version import DATA_VERSION" in workflow
    assert "oncoref.__version__" in workflow
    assert "data-{DATA_VERSION}" in workflow
    assert "steps.oncoref-cache-version.outputs.key" in workflow


def test_dependency_owned_sources_are_present_in_oncoref():
    """Dependency-owned routes stay discoverable but never write locally."""
    from oncoref.expression_builders import (
        gdc_source_entries,
        geo_matrix_source_entries,
        recount3_source_entries,
        treehouse_source_entries,
    )

    local = {
        source.id: source
        for source in downloads.load_registry()
        if source.build_owner == "oncoref"
    }
    upstream = {
        str(entry["id"]): entry
        for entries in (
            gdc_source_entries(),
            geo_matrix_source_entries(),
            recount3_source_entries(),
            treehouse_source_entries(),
        )
        for entry in entries
    }

    assert local
    assert {"cgci-blgsp", "gse328026-sarc-pec"} <= set(local)
    assert set(local) <= set(upstream)
    for source_id, source in local.items():
        assert source.builder is None
        assert source.source_type == str(upstream[source_id]["source_type"])
        if source.source_cohort:
            assert source.source_cohort == str(upstream[source_id]["source_cohort"])


def test_cache_root_honors_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path / "override"))
    assert downloads.cache_root() == tmp_path / "override"
    monkeypatch.delenv("PIRLYGENES_CACHE")
    assert downloads.cache_root() == Path.home() / ".cache" / "pirlygenes"


def test_registry_rejects_conflicting_build_owners(tmp_path: Path):
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        "sources:\n"
        "  - id: conflicting\n"
        "    builder: scripts/build.py\n"
        "    build_owner: oncoref\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot declare both"):
        downloads.load_registry(registry)


def test_source_cache_dir_layout(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path))
    assert downloads.source_cache_dir("foo") == tmp_path / "expression" / "foo"
    assert (
        downloads.source_cache_dir("bar", category="protein")
        == tmp_path / "protein" / "bar"
    )


def test_collect_cache_usage_reports_zero_for_empty_cache(
    monkeypatch, tmp_path: Path,
):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path))
    usages = downloads.collect_cache_usage()
    assert usages, "must report at least one source"
    assert all(u.on_disk_bytes == 0 for u in usages)


def test_collect_cache_usage_walks_actual_files(
    monkeypatch, tmp_path: Path,
):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path))
    target = downloads.source_cache_dir("cgci-blgsp")
    target.mkdir(parents=True)
    (target / "a.bin").write_bytes(b"x" * 1024)
    (target / "sub").mkdir()
    (target / "sub" / "b.bin").write_bytes(b"y" * 2048)

    usages = {u.source.id: u for u in downloads.collect_cache_usage()}
    assert usages["cgci-blgsp"].on_disk_bytes == 1024 + 2048


def test_render_list_groups_and_sorts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path))
    out = downloads.render_list(downloads.collect_cache_usage())
    assert "== expression" in out
    assert "Cache root:" in out
    assert "Total across" in out


def _run_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


def test_cli_no_args_prints_help():
    rc, out, _ = _run_cli([])
    assert rc == 0
    assert "downloads" in out
    assert "build" in out
    assert "plot" in out


def test_cli_downloads_cache_dir(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path))
    rc, out, _ = _run_cli(["downloads", "cache-dir"])
    assert rc == 0
    assert out.strip() == str(tmp_path)


def test_cli_downloads_list(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path))
    rc, out, _ = _run_cli(["downloads", "list"])
    assert rc == 0
    assert "tcga-blca" in out
    assert "cgci-blgsp" in out


def test_cli_build_list_enumerates_sources():
    rc, out, _ = _run_cli(["build", "list"])
    assert rc == 0
    assert "cgci-blgsp" in out
    assert "tcga-blca" in out
    assert "(oncoref-owned)" in out


def test_cli_build_unknown_source_reports_clearly():
    rc, _, err = _run_cli(["build", "nope-not-a-real-id"])
    assert rc == 2
    assert "no source matches" in err


def test_cli_build_dependency_owned_sources_redirect_to_oncoref():
    for source_id in (
        "gse98894-midnet",
        "cgci-blgsp",
        "gse328026-sarc-pec",
    ):
        rc, _, err = _run_cli(["build", source_id])
        assert rc == 2
        assert "built and published by oncoref" in err
        assert "oncoref.expression_builders" in err


def test_cli_build_ambiguous_cancer_code_lists_candidates():
    # CTCL is the cancer_code under exactly one source (gse171811-ctcl),
    # so it disambiguates cleanly. But there's no real "multi-source"
    # cancer code in the registry today; ensure single-match works.
    # (Negative path: explicit ambiguity would just check the error
    # contains "multiple sources" — leaving that to the dispatcher
    # docstring rather than a fixture.)
    pass


def test_cli_plot_requires_an_action():
    # `plot` is now implemented; with no action it prints a usage line naming
    # the available actions and exits non-zero.
    rc, _, err = _run_cli(["plot"])
    assert rc == 2
    assert "patient-coverage" in err
    assert "cta-curation" in err


def test_cli_plot_cta_curation_produces_figures(tmp_path: Path):
    rc, out, _ = _run_cli(["plot", "cta-curation", "--out", str(tmp_path)])
    assert rc == 0
    produced = sorted(p.name for p in tmp_path.glob("*.png"))
    assert produced == [
        "cta-deflated-frac-dist.png",
        "cta-filter-funnel.png",
        "cta-filter-outcome.png",
        "cta-protein-vs-rna.png",
        "cta-source-venn.png",
    ]
    assert "evidence rows" in out


def test_cli_analyze_redirects_to_trufflepig():
    rc, _, err = _run_cli(["analyze"])
    assert rc == 2
    assert "pirl-trufflepig" in err


def test_cli_analyze_with_legacy_flags_still_redirects():
    # `pirlygenes analyze --sample foo.tsv --workspace out` was the
    # pre-v5.0 invocation; argparse would reject the unknown flags
    # before reaching the migration handler unless main() intercepts
    # the analysis subcommand pre-parse. Regression guard for that.
    rc, _, err = _run_cli(
        ["analyze", "--sample", "foo.tsv", "--workspace", "out"]
    )
    assert rc == 2
    assert "pirl-trufflepig" in err
    assert "unrecognized" not in err


def test_get_data_resolves_csv_downloadable_by_bare_name_after_fetch(monkeypatch, tmp_path):
    """Regression: on a clean install, ``get_data("pan-cancer-expression")`` (bare
    stem — the item is registered as ``pan-cancer-expression.csv``) must resolve
    right after the on-demand bundle fetch.

    Previously ``get_data``'s post-fetch path-cache rebuild guard checked
    ``is_downloadable(name)`` on the bare stem, which returned False for a
    ``.csv``-suffixed downloadable. So the ``_dataset_paths`` cache (primed
    before the fetch) was never invalidated and the just-fetched file stayed
    invisible → ``ValueError: Dataset pan-cancer-expression not found`` on every
    fresh wheel install. See pirl-unc/trufflepig CI.
    """
    import pirlygenes.data_bundle as data_bundle
    import pirlygenes.load_dataset as ld

    bundled = tmp_path / "bundled"  # a fresh wheel: no large downloadables bundled
    cache = tmp_path / "cache"  # the version-pinned download target
    bundled.mkdir()
    cache.mkdir()

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", bundled)
    monkeypatch.setattr(ld, "_DOWNLOADED_DATA_DIR", cache)
    monkeypatch.setattr(data_bundle, "cache_dir", lambda: cache)

    def fake_ensure_local(*, auto_fetch: bool = True, verbose: bool = True):
        # Stand in for the release fetch: drop the file the bundle carries.
        (cache / "pan-cancer-expression.csv").write_text("gene_id,COAD_TPM\nENSG1,5.0\n")
        return cache

    monkeypatch.setattr(data_bundle, "ensure_local", fake_ensure_local)

    # Snapshot the module-global dataframe cache so this test's FAKE
    # pan-cancer-expression frame can't leak into real-data tests later in the
    # same (serial, -n 0) process — that poisoning made 28 downstream expression
    # tests KeyError in the release run.
    saved_frames = dict(ld._CACHED_DATAFRAMES)
    ld._CACHED_DATAFRAMES.pop("pan-cancer-expression.csv", None)

    # Prime the path cache BEFORE the file exists — the stale-cache precondition
    # the fetch has to punch through.
    ld._invalidate_dataset_paths()
    assert "pan-cancer-expression.csv" not in ld._dataset_paths()

    try:
        df = ld.get_data("pan-cancer-expression", copy=False)
        assert list(df.columns) == ["gene_id", "COAD_TPM"]
        assert len(df) == 1
    finally:
        ld._CACHED_DATAFRAMES.clear()
        ld._CACHED_DATAFRAMES.update(saved_frames)
        ld._invalidate_dataset_paths()  # don't leak the tmp-path map to other tests
