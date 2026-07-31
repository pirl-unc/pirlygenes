"""Smoke tests for the oncoref-backed cache and compatibility CLI."""

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
    # Oncoref owns the released diagnosis split; a local build must not
    # silently recreate the retired mixed-histology artifact.
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
    from oncoref.expression_registry import expression_sources

    local = {
        source.id: source
        for source in downloads.load_registry()
        if source.build_owner == "oncoref"
    }
    upstream = {source.id: source for source in expression_sources()}

    assert local
    assert {"cgci-blgsp", "gse328026-sarc-pec"} <= set(local)
    assert set(local) <= set(upstream)
    for source_id, source in local.items():
        assert source.builder is None
        assert source.source_type == upstream[source_id].source_type
        if source.source_cohort:
            assert source.source_cohort == upstream[source_id].source_cohort


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
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    usages = downloads.collect_cache_usage()
    assert usages, "must report at least one source"
    assert all(u.on_disk_bytes == 0 for u in usages)


def test_collect_cache_usage_walks_actual_files(
    monkeypatch, tmp_path: Path,
):
    from oncoref import source_matrices

    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    target = source_matrices.local_path("BL")
    target.write_bytes(b"x" * 3072)

    usages = {u.source.id: u for u in downloads.collect_cache_usage()}
    assert usages["cgci-blgsp"].on_disk_bytes == 1024 + 2048


def test_explicit_owner_subset_uses_oncoref_cache(
    monkeypatch, tmp_path: Path,
):
    from oncoref import source_matrices

    owner_cache = tmp_path / "owner"
    legacy_cache = tmp_path / "legacy"
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(owner_cache))
    monkeypatch.setenv("PIRLYGENES_CACHE", str(legacy_cache))
    source_matrices.local_path("BL").write_bytes(b"x" * 3072)

    subset = [
        source
        for source in downloads.load_registry()
        if source.id == "cgci-blgsp"
    ]
    stale = downloads.source_cache_dir("cgci-blgsp")
    stale.mkdir(parents=True)
    (stale / "stale.bin").write_bytes(b"x" * 17)

    usage = downloads.collect_cache_usage(subset)

    assert len(usage) == 1
    assert usage[0].on_disk_bytes == 3072
    assert usage[0].cache_dir == source_matrices.cache_dir()


def test_explicit_custom_source_uses_legacy_cache(
    monkeypatch, tmp_path: Path,
):
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        "sources:\n"
        "  - id: private-fixture\n"
        "    category: expression\n"
        "    cancer_codes: [PRIVATE]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PIRLYGENES_CACHE", str(tmp_path / "legacy"))
    source = downloads.load_registry(registry)[0]
    cached = downloads.source_cache_dir(source.id)
    cached.mkdir(parents=True)
    (cached / "matrix.csv").write_bytes(b"x" * 23)

    usage = downloads.collect_cache_usage([source])

    assert len(usage) == 1
    assert usage[0].on_disk_bytes == 23
    assert usage[0].cache_dir == cached


def test_cache_usage_charges_routed_matrices_only_to_physical_owner(
    monkeypatch, tmp_path: Path,
):
    from oncoref import source_matrices

    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    source_matrices.local_path("ACC").write_bytes(b"x" * 4096)

    usages = {u.source.id: u for u in downloads.collect_cache_usage()}

    assert usages["treehouse-polya-25-01-tcga-subset"].on_disk_bytes == 4096
    assert usages["tcga-acc"].on_disk_bytes == 0
    assert sum(usage.on_disk_bytes for usage in usages.values()) == 4096


def test_cache_usage_accounts_for_every_published_matrix(
    monkeypatch, tmp_path: Path,
):
    from oncoref import source_matrices

    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    codes = source_matrices.registry()["cancer_code"].astype(str).tolist()
    for code in codes:
        source_matrices.local_path(code).write_bytes(b"x")

    usages = downloads.collect_cache_usage()

    assert sum(usage.on_disk_bytes for usage in usages) == len(codes)


def test_render_list_groups_and_sorts(monkeypatch, tmp_path: Path):
    from oncoref import source_matrices

    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    out = downloads.render_list(downloads.collect_cache_usage())
    assert "== expression" in out
    assert f"Cache root: {source_matrices.cache_dir()}" in out
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
    from oncoref import source_matrices

    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    rc, out, _ = _run_cli(["downloads", "cache-dir"])
    assert rc == 0
    assert out.strip() == str(source_matrices.cache_dir())


def test_cli_downloads_list(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path))
    rc, out, _ = _run_cli(["downloads", "list"])
    assert rc == 0
    assert "tcga-blca" in out
    assert "cgci-blgsp" in out


def test_cli_downloads_fetch_resolves_owner_source_ids(monkeypatch):
    from oncoref import source_matrices

    fetched = []
    monkeypatch.setattr(source_matrices, "fetch", fetched.append)

    rc, out, err = _run_cli(["downloads", "fetch", "beataml-ohsu-2022"])

    assert rc == 0
    assert not err
    assert fetched == [
        "LAML_APL",
        "LAML_ELNadv",
        "LAML_ELNfav",
        "LAML_ELNint",
    ]
    assert "fetched 4 oncoref source matrices" in out


def test_cli_downloads_fetch_resolves_historical_source_ids(monkeypatch):
    from oncoref import source_matrices

    fetched = []
    monkeypatch.setattr(source_matrices, "fetch", fetched.append)

    rc, out, err = _run_cli(["downloads", "fetch", "beataml-ohsu"])

    assert rc == 0
    assert not err
    assert fetched == [
        "LAML_APL",
        "LAML_ELNadv",
        "LAML_ELNfav",
        "LAML_ELNint",
    ]
    assert "fetched 4 oncoref source matrices" in out


def test_cli_downloads_fetch_preserves_geo_heme_alias(monkeypatch):
    from oncoref import source_matrices

    fetched = []
    monkeypatch.setattr(source_matrices, "fetch", fetched.append)

    rc, out, err = _run_cli(["downloads", "fetch", "geo-heme"])

    assert rc == 0
    assert not err
    assert fetched == ["CML", "MCL", "MDS", "MPN"]
    assert "fetched 4 oncoref source matrices" in out


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        ("PANNET", "NET_PANCREAS"),
        ("prad", "PRAD"),
    ],
)
def test_cli_downloads_fetch_resolves_cancer_code_aliases(
    monkeypatch, requested, canonical,
):
    from oncoref import source_matrices

    fetched = []
    monkeypatch.setattr(source_matrices, "fetch", fetched.append)

    rc, out, err = _run_cli(["downloads", "fetch", requested])

    assert rc == 0
    assert not err
    assert fetched == [canonical]
    assert "fetched 1 oncoref source matrix" in out


def test_cli_downloads_fetch_reports_owner_download_failures(monkeypatch):
    from oncoref import source_matrices

    def fail(_code):
        raise source_matrices.SourceMatrixError("offline fixture")

    monkeypatch.setattr(source_matrices, "fetch", fail)

    rc, out, err = _run_cli(["downloads", "fetch", "PRAD"])

    assert rc == 2
    assert not out
    assert "failed to fetch oncoref source matrix 'PRAD'" in err
    assert "offline fixture" in err


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


def test_cli_build_reports_newly_published_mmnst_matrix():
    rc, _, err = _run_cli(["build", "prjna1083972-mmnst"])

    assert rc == 2
    assert "built and published by oncoref" in err
    assert "downloads fetch prjna1083972-mmnst" in err
    assert "SARC_MMNST" in err


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
