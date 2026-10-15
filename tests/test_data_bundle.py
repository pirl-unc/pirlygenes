"""Exercise PirlyGenes' adapter against the real DataCache archive installer."""

from __future__ import annotations

import importlib
import gzip
import io
import multiprocessing
import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

import datacache
from datacache import download
import pytest

from pirlygenes import data_bundle


FILES = {
    "cohort-expression-matrices/matrix.parquet": b"fixture matrix",
    "pan-cancer-expression.csv": b"gene,value\nfixture,1\n",
    "hpa-cell-type-expression.csv": b"gene,value\nfixture,2\n",
}


def _archive(path, files=None):
    with tarfile.open(path, mode="w:gz") as tf:
        for name, content in (FILES if files is None else files).items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tf.addfile(member, io.BytesIO(content))
    return path


def _write_cache(root, files=None):
    for name, content in (FILES if files is None else files).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _assert_cache(root, files=None):
    for name, content in (FILES if files is None else files).items():
        assert (root / name).read_bytes() == content


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / f"v{data_bundle.DATA_VERSION}"
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(root))
    source = _archive(tmp_path / "release.tar.gz")
    monkeypatch.setattr(data_bundle, "RELEASE_URL", source.as_uri())
    return root


@pytest.fixture(autouse=True)
def clear_loader_cache():
    from pirlygenes import clear_dataset_cache

    clear_dataset_cache()
    yield
    clear_dataset_cache()


def _record_transfers(monkeypatch):
    calls = []
    original = download.fetch_file

    def fetch(url, **kwargs):
        calls.append(url)
        return original(url, **kwargs)

    monkeypatch.setattr(download, "fetch_file", fetch)
    return calls


def test_threads_download_once_and_warm_reads_are_read_only(cache, monkeypatch):
    calls = _record_transfers(monkeypatch)
    barrier = threading.Barrier(2, timeout=10)
    original = data_bundle.is_local

    def simultaneous_cold_checks():
        result = original()
        barrier.wait()
        return result

    monkeypatch.setattr(data_bundle, "is_local", simultaneous_cold_checks)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(data_bundle.ensure_local, verbose=False) for _ in range(2)]
        paths = [f.result(timeout=15) for f in futures]
    assert len(calls) == 1
    assert paths[0] == paths[1]
    assert paths[0] != cache
    _assert_cache(paths[0])
    monkeypatch.setattr(data_bundle, "is_local", original)

    def forbidden(*args, **kwargs):
        pytest.fail("warm readers should not install or download")

    monkeypatch.setattr(datacache, "install_archive", forbidden)
    monkeypatch.setattr(download, "fetch_file", forbidden)
    assert data_bundle.ensure_local(verbose=False) == paths[0]
    assert data_bundle.ensure_local(auto_fetch=False) == paths[0]
    assert data_bundle.find("pan-cancer-expression.csv") == paths[0] / "pan-cancer-expression.csv"
    assert data_bundle.status()["data_dir"] == str(paths[0])


def _process_reader(root, url, barrier, count):
    os.environ["PIRLYGENES_BUNDLED_DATA"] = root
    data_bundle.RELEASE_URL = url
    is_local = data_bundle.is_local
    fetch = download.fetch_file
    checked = False

    def simultaneous_cold_checks():
        nonlocal checked
        result = is_local()
        if not checked:
            checked = True
            barrier.wait(timeout=20)
        return result

    def counted_fetch(*args, **kwargs):
        with count.get_lock():
            count.value += 1
        return fetch(*args, **kwargs)

    data_bundle.is_local = simultaneous_cold_checks
    download.fetch_file = counted_fetch
    bundle = data_bundle.ensure_local(verbose=False)
    assert data_bundle.ensure_local(verbose=False) == bundle
    _assert_cache(bundle)


def test_processes_share_datacache_lock_for_aliases(cache, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    count = context.Value("i", 0)
    readers = [
        context.Process(
            target=_process_reader,
            args=(str(root), data_bundle.RELEASE_URL, barrier, count),
        )
        for root in (cache, alias / cache.name)
    ]
    try:
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join(timeout=40)
            assert reader.exitcode == 0
    finally:
        for reader in readers:
            if reader.is_alive():
                reader.terminate()
            reader.join(timeout=5)
    assert count.value == 1
    _assert_cache(data_bundle.ensure_local(auto_fetch=False))


@pytest.mark.parametrize("legacy", [False, True])
def test_code_only_upgrade_reuses_cache_without_writes(cache, monkeypatch, legacy):
    from pirlygenes import version

    if legacy:
        _write_cache(cache)
        bundle = cache
    else:
        bundle = data_bundle.ensure_local(verbose=False)
    before = {p: p.stat().st_mtime_ns for p in cache.rglob("*")}
    url = data_bundle.RELEASE_URL
    monkeypatch.setattr(version, "__version__", "999.0.0")
    importlib.reload(data_bundle)
    monkeypatch.setattr(data_bundle, "RELEASE_URL", url)

    def forbidden(*args, **kwargs):
        pytest.fail("code-only upgrade attempted installation")

    monkeypatch.setattr(datacache, "install_archive", forbidden)
    assert data_bundle.ensure_local(verbose=False) == bundle
    assert {p: p.stat().st_mtime_ns for p in cache.rglob("*")} == before
    _assert_cache(bundle)


def test_missing_cache_inspection_is_read_only(cache):
    assert not data_bundle.is_local()
    assert data_bundle.find("pan-cancer-expression.csv") is None
    assert not data_bundle.status()["all_local"]
    with pytest.raises(FileNotFoundError, match="pirlygenes data fetch"):
        data_bundle.ensure_local(auto_fetch=False)
    assert not cache.exists()


def test_refresh_keeps_old_reader_paths_and_legacy_files(cache):
    _write_cache(cache, {name: b"legacy" for name in FILES})
    (cache / "other-owner.txt").write_text("unrelated")
    first = data_bundle.fetch(verbose=False)
    _assert_cache(first)
    second = data_bundle.fetch(verbose=False)
    assert second != first
    _assert_cache(first)
    _assert_cache(second)
    _assert_cache(cache, {name: b"legacy" for name in FILES})
    assert (cache / "other-owner.txt").read_text() == "unrelated"
    assert data_bundle.ensure_local(verbose=False) == second


@pytest.mark.parametrize("warm", [False, True])
def test_failed_archive_keeps_previous_bundle(cache, warm):
    previous = data_bundle.ensure_local(verbose=False) if warm else None
    # Change bytes at the same catalogue URL, as a failed release transfer would.
    source = cache.parent / "release.tar.gz"
    source.write_bytes(b"truncated archive")
    with pytest.raises(datacache.FileValidationError):
        data_bundle.fetch(verbose=False)
    if warm:
        assert data_bundle.ensure_local(auto_fetch=False) == previous
        _assert_cache(previous)
    else:
        assert not data_bundle.is_local()
        assert data_bundle.find("pan-cancer-expression.csv") is None
    _archive(source)
    _assert_cache(data_bundle.ensure_local(verbose=False))


def _interrupted_writer(root, url, ready):
    os.environ["PIRLYGENES_BUNDLED_DATA"] = root
    data_bundle.RELEASE_URL = url
    fetch = download.fetch_file

    def pause_after_transfer(*args, **kwargs):
        result = fetch(*args, **kwargs)
        ready.set()
        threading.Event().wait(timeout=30)
        return result

    download.fetch_file = pause_after_transfer
    data_bundle.fetch(verbose=False)


@pytest.mark.parametrize("warm", [False, True])
def test_killed_downloader_never_replaces_active_data(cache, warm):
    previous = data_bundle.ensure_local(verbose=False) if warm else None
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    writer = context.Process(
        target=_interrupted_writer,
        args=(str(cache), data_bundle.RELEASE_URL, ready),
    )
    writer.start()
    try:
        assert ready.wait(timeout=20)
        assert data_bundle.is_local() == warm
        if warm:
            assert data_bundle.ensure_local(auto_fetch=False) == previous
    finally:
        writer.terminate()
        writer.join(timeout=10)
    assert writer.exitcode is not None
    replacement = data_bundle.fetch(verbose=False)
    _assert_cache(replacement)
    if warm:
        assert replacement != previous
        _assert_cache(previous)


@pytest.mark.parametrize("previous_kind", ["missing", "legacy", "managed"])
@pytest.mark.parametrize("missing_member", list(FILES))
@pytest.mark.parametrize("empty", [False, True])
def test_incomplete_refresh_keeps_previous_cache(cache, previous_kind, missing_member, empty):
    if previous_kind == "legacy":
        _write_cache(cache)
        previous = cache
    elif previous_kind == "managed":
        previous = data_bundle.ensure_local(verbose=False)
    else:
        previous = None
    broken = dict(FILES)
    if empty:
        broken[missing_member] = b""
    else:
        del broken[missing_member]
    _archive(cache.parent / "release.tar.gz", broken)
    install = data_bundle.ensure_local if previous is None else data_bundle.fetch
    with pytest.raises(datacache.FileValidationError, match="missing required data"):
        install(verbose=False)
    if previous is None:
        assert not data_bundle.is_local()
        assert data_bundle.find("pan-cancer-expression.csv") is None
    else:
        assert data_bundle.ensure_local(auto_fetch=False) == previous
        _assert_cache(previous)
    _archive(cache.parent / "release.tar.gz")
    _assert_cache(install(verbose=False))


def test_invalid_managed_receipt_requires_explicit_repair(cache, monkeypatch):
    previous = data_bundle.ensure_local(verbose=False)
    (previous / ".datacache-manifest.json").write_text("{}")
    calls = _record_transfers(monkeypatch)
    with pytest.raises(datacache.FileValidationError, match="pirlygenes data fetch"):
        data_bundle.ensure_local(verbose=False)
    assert calls == []
    repaired = data_bundle.fetch(verbose=False)
    assert len(calls) == 1
    assert repaired != previous
    _assert_cache(repaired)
    _assert_cache(previous)


def test_managed_cache_is_listed_and_current_version_is_kept(cache):
    bundle = data_bundle.ensure_local(verbose=False)
    versions = data_bundle.list_cache_versions()
    assert len(versions) == 1
    assert versions[0]["path"] == cache
    assert versions[0]["is_current"]
    assert versions[0]["size_bytes"] >= sum(len(value) for value in FILES.values())
    assert data_bundle.prune_cache() == []
    _assert_cache(bundle)
    assert len(data_bundle.prune_cache(keep_current=False)) == 1
    assert not cache.exists()


def test_loader_resolves_bundle_published_after_path_cache_was_primed(cache, tmp_path, monkeypatch):
    from pirlygenes import load_dataset as ld

    bundled = tmp_path / "wheel-data"
    bundled.mkdir()
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", bundled)
    ld.clear_dataset_cache()
    assert "pan-cancer-expression.csv" not in ld.dataset_paths()
    frame = ld.get_data("pan-cancer-expression")
    assert frame.to_dict("list") == {"gene": ["fixture"], "value": [1]}
    assert ld.dataset_paths()["pan-cancer-expression.csv"] == data_bundle.find("pan-cancer-expression.csv")


def test_expression_accessor_uses_returned_bundle_directory(cache, tmp_path, monkeypatch):
    from pirlygenes import load_dataset as ld
    from pirlygenes.expression.accessors import _bundle_subdir

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    path = _bundle_subdir("cohort-expression-matrices")
    assert path == data_bundle.find("cohort-expression-matrices")
    assert (path / "matrix.parquet").read_bytes() == FILES["cohort-expression-matrices/matrix.parquet"]


@pytest.mark.parametrize("storage", ["csv", "csv.gz", "shards"])
@pytest.mark.parametrize("cache_error", ["corrupt", "inaccessible"])
def test_bundled_data_survives_broken_download_cache(
    cache, tmp_path, monkeypatch, storage, cache_error,
):
    from pirlygenes import load_dataset as ld

    previous = data_bundle.ensure_local(verbose=False)
    if cache_error == "corrupt":
        (previous / ".datacache-manifest.json").write_text("{}")
        expected_error = datacache.FileValidationError
    else:
        def inaccessible(*args, **kwargs):
            raise PermissionError("download cache inaccessible")
        monkeypatch.setattr(data_bundle, "data_dir", inaccessible)
        expected_error = PermissionError
    bundled = tmp_path / "wheel-data"
    bundled.mkdir()
    content = b"gene,value\nPRAME,1\n"
    if storage == "shards":
        shard_dir = bundled / "curated"
        shard_dir.mkdir()
        (shard_dir / "part.csv").write_bytes(content)
    elif storage == "csv.gz":
        with gzip.open(bundled / "curated.csv.gz", "wb") as handle:
            handle.write(content)
    else:
        (bundled / "curated.csv").write_bytes(content)
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", bundled)
    ld.clear_dataset_cache()
    for name in ("curated", "Curated.csv"):
        assert ld.get_data(name).to_dict("list") == {"gene": ["PRAME"], "value": [1]}
    # A request that actually needs the damaged cache still reports the error.
    with pytest.raises(expected_error):
        ld.get_data("pan-cancer-expression")


@pytest.mark.parametrize("sharded", [False, True])
def test_warm_bundled_reads_avoid_scans_and_invalidation_refreshes_both_roots(
    cache, tmp_path, monkeypatch, sharded,
):
    from pirlygenes import load_dataset as ld

    bundled = tmp_path / "wheel-data"
    bundled.mkdir()
    content = "gene,value\nPRAME,1\n"
    if sharded:
        (bundled / "curated").mkdir()
        (bundled / "curated" / "part.csv").write_text(content)
    else:
        (bundled / "curated.csv").write_text(content)
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", bundled)
    ld.clear_dataset_cache()

    frame = ld.get_data("curated", copy=False)
    assert "curated" in ld.dataset_paths()

    def unexpected_scan(*args, **kwargs):
        pytest.fail("a cached dataset lookup should not rescan directories")

    with monkeypatch.context() as warm:
        warm.setattr(ld, "get_all_csv_paths", unexpected_scan)
        warm.setattr(ld, "_shard_directories", unexpected_scan)
        for name in ("curated", "Curated.csv"):
            assert ld.get_data(name, copy=False) is frame
            assert ld.get_data(name).equals(frame)

    (bundled / "added.csv").write_text(content)
    cache.mkdir()
    (cache / "downloaded.csv").write_text(content)
    assert "added" not in ld.dataset_paths(bundled_only=True)
    assert "downloaded" not in ld.dataset_paths()
    ld.clear_dataset_cache()
    bundled_paths = ld.dataset_paths(bundled_only=True)
    all_paths = ld.dataset_paths()
    assert bundled_paths["added"] == bundled / "added.csv"
    assert "downloaded" not in bundled_paths
    assert all_paths["added"] == bundled / "added.csv"
    assert all_paths["downloaded"] == cache / "downloaded.csv"
    assert ld.get_data("added").equals(frame)
    assert ld.get_data("downloaded").equals(frame)


@pytest.mark.parametrize("warm_frame", [False, True])
@pytest.mark.parametrize("remove_old", [False, True])
def test_dataset_reads_follow_refreshed_bundle(cache, tmp_path, monkeypatch, warm_frame, remove_old):
    from pirlygenes import dataset_paths, get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    old_bundle = data_bundle.ensure_local(verbose=False)
    old_path = dataset_paths()["pan-cancer-expression"]
    assert old_path.parent == old_bundle
    if warm_frame:
        assert get_data("pan-cancer-expression", copy=False)["value"].tolist() == [1]
    if remove_old:
        old_path.unlink()
    replacement = dict(FILES, **{"pan-cancer-expression.csv": b"gene,value\nfixture,9\n"})
    _archive(tmp_path / "release.tar.gz", replacement)
    new_bundle = data_bundle.fetch(verbose=False)
    assert new_bundle != old_bundle
    assert dataset_paths()["pan-cancer-expression"] == new_bundle / old_path.name
    assert get_data("pan-cancer-expression", copy=False)["value"].tolist() == [9]


def test_dataset_reads_follow_external_bundle_publication(cache, tmp_path, monkeypatch):
    from pirlygenes import dataset_paths, get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    old_bundle = data_bundle.ensure_local(verbose=False)
    assert get_data("pan-cancer-expression", copy=False)["value"].tolist() == [1]
    _archive(tmp_path / "release.tar.gz", dict(
        FILES, **{"pan-cancer-expression.csv": b"gene,value\nfixture,7\n"},
    ))
    # Publish directly through DataCache, bypassing PirlyGenes' fetch lifecycle,
    # as another process would. No loader cache-reset callback can run here.
    new_bundle = datacache.install_archive(
        cache / ".datacache-archive", data_bundle.RELEASE_URL,
        force=True, verified=False,
    )
    assert new_bundle != old_bundle
    assert dataset_paths()["pan-cancer-expression"].parent == new_bundle
    assert get_data("pan-cancer-expression")["value"].tolist() == [7]


def test_public_dataset_paths_are_read_only_and_bundled_aliases_win(cache, tmp_path, monkeypatch):
    from pirlygenes import clear_dataset_cache, data_roots, dataset_paths, get_data
    from pirlygenes import load_dataset as ld

    bundled = tmp_path / "wheel-data"
    bundled.mkdir()
    (bundled / "Curated.csv").write_text("value\n1\n")
    _write_cache(cache, {"curated/part.csv": b"value\n2\n"})
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", bundled)
    assert data_roots() == (bundled, cache)
    paths = dataset_paths()
    assert paths["curated"] == bundled / "Curated.csv"
    with pytest.raises(TypeError):
        paths["curated"] = cache / "curated"
    assert get_data("curated")["value"].tolist() == [1]
    (bundled / "Curated.csv").write_text("value\n3\n")
    clear_dataset_cache()
    assert get_data("curated")["value"].tolist() == [3]


def test_refreshed_shards_do_not_reuse_previous_bundle_parquet(cache, tmp_path, monkeypatch):
    from pirlygenes import get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    monkeypatch.setattr(ld.Path, "home", lambda: tmp_path / "home")
    for value in (1, 9):
        _archive(tmp_path / "release.tar.gz", dict(
            FILES, **{"sharded/part.csv": f"gene,value\nfixture,{value}\n".encode()},
        ))
        bundle = data_bundle.fetch(verbose=False)
        # Same shard count, size and mtime across installations: only the
        # immutable directory identity distinguishes the disk-cache entries.
        os.utime(bundle / "sharded/part.csv", ns=(1_000_000_000, 1_000_000_000))
        assert get_data("sharded")["value"].tolist() == [value]
        assert list((tmp_path / "home/.cache/pirlygenes/shard_cache").rglob("data.parquet"))


def test_warm_downloaded_reads_reuse_discovered_paths(cache, tmp_path, monkeypatch):
    from pirlygenes import get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    frame = get_data("pan-cancer-expression", copy=False)

    def unexpected_scan(*args, **kwargs):
        pytest.fail("warm downloaded lookups must not rescan directories")

    monkeypatch.setattr(ld, "get_all_csv_paths", unexpected_scan)
    assert get_data("pan-cancer-expression", copy=False) is frame
    assert get_data("Pan-Cancer-Expression.csv").equals(frame)


def test_clear_dataset_cache_observes_edits_to_any_shard(cache, tmp_path, monkeypatch):
    from pirlygenes import clear_dataset_cache, get_data
    from pirlygenes import load_dataset as ld

    bundled = tmp_path / "wheel-data"
    shards = bundled / "editable"
    shards.mkdir(parents=True)
    (shards / "a.csv").write_text("value\n1\n")
    (shards / "b.csv").write_text("value\n2\n")
    os.utime(shards / "a.csv", ns=(1_000_000_000, 1_000_000_000))
    os.utime(shards / "b.csv", ns=(3_000_000_000, 3_000_000_000))
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", bundled)
    monkeypatch.setattr(ld.Path, "home", lambda: tmp_path / "home")
    assert get_data("editable")["value"].tolist() == [1, 2]

    (shards / "a.csv").write_text("value\n9\n")
    os.utime(shards / "a.csv", ns=(2_000_000_000, 2_000_000_000))
    clear_dataset_cache()
    # The total size and newest mtime did not change, but a.csv did.
    assert get_data("editable")["value"].tolist() == [9, 2]


def test_shard_cache_publication_isolated_between_bundles(cache, tmp_path, monkeypatch):
    import pandas as pd
    from pirlygenes import clear_dataset_cache, get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    monkeypatch.setattr(ld.Path, "home", lambda: tmp_path / "home")
    bundles = []
    for value in (1, 9):
        _archive(tmp_path / "release.tar.gz", dict(
            FILES, **{"sharded/part.csv": f"value\n{value}\n".encode()},
        ))
        bundles.append(data_bundle.fetch(verbose=False))
    old_bundle, new_bundle = bundles
    active = [old_bundle]
    monkeypatch.setattr(data_bundle, "data_dir", lambda: active[0])
    to_parquet = pd.DataFrame.to_parquet
    interleaved = []

    def interleave_writers(frame, path, **kwargs):
        to_parquet(frame, path, **kwargs)
        if not interleaved:
            interleaved.append(True)
            active[0] = new_bundle
            try:
                assert get_data("sharded")["value"].tolist() == [9]
            finally:
                active[0] = old_bundle

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interleave_writers)
    assert get_data("sharded")["value"].tolist() == [1]
    clear_dataset_cache()
    # An old in-flight reader and the new bundle reader must never pair one
    # installation's data with the other installation's cache signature.
    assert get_data("sharded")["value"].tolist() == [1]
    active[0] = new_bundle
    assert get_data("sharded")["value"].tolist() == [9]


def test_inflight_dataset_read_survives_bundle_cache_eviction(cache, tmp_path, monkeypatch):
    from pirlygenes import dataset_paths, get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    data_bundle.ensure_local(verbose=False)
    old_path = dataset_paths()["pan-cancer-expression"]
    _archive(tmp_path / "release.tar.gz", dict(
        FILES, **{"pan-cancer-expression.csv": b"gene,value\nfixture,9\n"},
    ))
    refreshed = []

    class InterleavedCache(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == old_path and not refreshed:
                # Deterministically schedule a refresh and second reader after
                # the first reader stores its frame but before it returns it.
                refreshed.append(data_bundle.fetch(verbose=False))
                assert get_data("pan-cancer-expression")["value"].tolist() == [9]

    monkeypatch.setattr(ld, "_CACHED_DATAFRAMES", InterleavedCache())
    assert get_data("pan-cancer-expression")["value"].tolist() == [1]
    assert get_data("pan-cancer-expression")["value"].tolist() == [9]


def test_shard_materialization_reused_after_in_memory_reset(cache, tmp_path, monkeypatch):
    import pandas as pd
    from pirlygenes import clear_dataset_cache, get_data
    from pirlygenes import load_dataset as ld

    shards = tmp_path / "wheel-data/curated"
    shards.mkdir(parents=True)
    (shards / "a.csv").write_text("value\n1\n")
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", shards.parent)
    monkeypatch.setattr(ld.Path, "home", lambda: tmp_path / "home")
    assert get_data("curated")["value"].tolist() == [1]
    clear_dataset_cache()

    def forbid_csv(*args, **kwargs):
        pytest.fail("a verified Parquet cache hit must not reparse CSV inputs")

    monkeypatch.setattr(pd, "read_csv", forbid_csv)
    assert get_data("curated")["value"].tolist() == [1]


@pytest.mark.parametrize("cache_failure", ["inaccessible", "corrupt"])
def test_shard_cache_failure_falls_back_to_source(cache, tmp_path, monkeypatch, cache_failure):
    from pirlygenes import clear_dataset_cache, get_data
    from pirlygenes import load_dataset as ld

    shards = tmp_path / "wheel-data/curated"
    shards.mkdir(parents=True)
    (shards / "a.csv").write_text("value\n1\n")
    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", shards.parent)
    monkeypatch.setattr(ld.Path, "home", lambda: tmp_path / "home")
    assert get_data("curated")["value"].tolist() == [1]
    clear_dataset_cache()
    if cache_failure == "corrupt":
        parquet = next((tmp_path / "home/.cache/pirlygenes/shard_cache").rglob("data.parquet"))
        parquet.write_bytes(b"corrupt")
    else:
        def inaccessible(*args, **kwargs):
            raise PermissionError("cache not writable")
        monkeypatch.setattr(datacache, "materialize", inaccessible)
    assert get_data("curated")["value"].tolist() == [1]


def test_relative_legacy_cache_tracks_current_directory(cache, tmp_path, monkeypatch):
    from pirlygenes import get_data
    from pirlygenes import load_dataset as ld

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", "legacy")
    for directory, value in (("first", 1), ("second", 9)):
        root = tmp_path / directory
        _write_cache(root / "legacy", dict(
            FILES, **{"pan-cancer-expression.csv": f"gene,value\nfixture,{value}\n".encode()},
        ))
        monkeypatch.chdir(root)
        assert get_data("pan-cancer-expression")["value"].tolist() == [value]
