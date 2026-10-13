"""Exercise PirlyGenes' adapter against the real DataCache archive installer."""

from __future__ import annotations

import importlib
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


def test_missing_required_release_members_are_not_usable(cache):
    _archive(cache.parent / "release.tar.gz", {"unrelated.csv": b"unrelated"})
    with pytest.raises(datacache.FileValidationError, match="missing required data"):
        data_bundle.ensure_local(verbose=False)
    assert not data_bundle.is_local()
    _archive(cache.parent / "release.tar.gz")
    _assert_cache(data_bundle.fetch(verbose=False))


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
    monkeypatch.setattr(ld, "_DOWNLOADED_DATA_DIR", cache)
    monkeypatch.setattr(ld, "_DATASET_PATHS", None)
    monkeypatch.setattr(ld, "_CACHED_DATAFRAMES", {})
    assert "pan-cancer-expression.csv" not in ld._dataset_paths()
    frame = ld.get_data("pan-cancer-expression")
    assert frame.to_dict("list") == {"gene": ["fixture"], "value": [1]}
    assert ld._dataset_paths()["pan-cancer-expression.csv"] == data_bundle.find("pan-cancer-expression.csv")


def test_expression_accessor_uses_returned_bundle_directory(cache, tmp_path, monkeypatch):
    from pirlygenes import load_dataset as ld
    from pirlygenes.expression.accessors import _bundle_subdir

    monkeypatch.setattr(ld, "_BUNDLED_DATA_DIR", tmp_path / "wheel-data")
    path = _bundle_subdir("cohort-expression-matrices")
    assert path == data_bundle.find("cohort-expression-matrices")
    assert (path / "matrix.parquet").read_bytes() == FILES["cohort-expression-matrices/matrix.parquet"]
