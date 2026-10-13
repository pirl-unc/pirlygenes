"""Network-free regression tests for concurrent bundle downloads (#636)."""

from __future__ import annotations

import importlib
import io
import multiprocessing
import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pirlygenes import data_bundle


FILES = {
    "cohort-expression-matrices/matrix.parquet": b"fixture matrix",
    "pan-cancer-expression.csv": b"gene,value\nfixture,1\n",
    "hpa-cell-type-expression.csv": b"gene,value\nfixture,2\n",
}


def _archive(files=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tf:
        for name, content in (FILES if files is None else files).items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tf.addfile(member, io.BytesIO(content))
    return stream.getvalue()


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

    def no_network(*args, **kwargs):
        pytest.fail("unexpected network access")

    monkeypatch.setattr(data_bundle.urllib.request, "urlopen", no_network)
    return root


def _serve(monkeypatch, payload):
    calls = []

    def urlopen(url):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(data_bundle.urllib.request, "urlopen", urlopen)
    return calls


def test_threads_download_once_and_warm_reads_do_not_write(cache, monkeypatch):
    calls = _serve(monkeypatch, _archive())
    barrier = threading.Barrier(2, timeout=10)
    seen = threading.local()
    is_local = data_bundle.is_local

    def simultaneous_cold_checks():
        result = is_local()
        if not getattr(seen, "checked", False):
            seen.checked = True
            barrier.wait()
        return result

    monkeypatch.setattr(data_bundle, "is_local", simultaneous_cold_checks)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(data_bundle.ensure_local, verbose=False) for _ in range(2)]
        assert [f.result(timeout=15) for f in futures] == [cache, cache]
    assert len(calls) == 1
    _assert_cache(cache)
    monkeypatch.setattr(data_bundle, "is_local", is_local)

    def no_lock(*args):
        pytest.fail("warm readers should not create a lock or write to disk")

    monkeypatch.setattr(data_bundle, "_cache_lock", no_lock)
    assert data_bundle.ensure_local(verbose=False) == cache
    assert data_bundle.ensure_local(auto_fetch=False) == cache
    assert len(calls) == 1


def _process_reader(root, payload, barrier, count):
    # Spawn starts an independent interpreter: its FileLock, imports and cache
    # state are not shared with the parent or the other reader.
    os.environ["PIRLYGENES_BUNDLED_DATA"] = root
    original = data_bundle.is_local
    checked = False

    def is_local():
        nonlocal checked
        result = original()
        if not checked:
            checked = True
            barrier.wait(timeout=20)
        return result

    def urlopen(url):
        with count.get_lock():
            count.value += 1
        return io.BytesIO(payload)

    data_bundle.is_local = is_local
    data_bundle.urllib.request.urlopen = urlopen
    data_bundle.ensure_local(verbose=False)
    data_bundle.ensure_local(verbose=False)
    _assert_cache(Path(root))


def test_processes_share_lock_for_symlink_alias(cache, tmp_path):
    # Two spellings of the same destination must not create different locks.
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    count = context.Value("i", 0)
    readers = [
        context.Process(
            target=_process_reader,
            args=(str(root), _archive(), barrier, count),
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
    _assert_cache(cache)


def test_code_only_upgrade_reuses_legacy_cache(cache, monkeypatch):
    from pirlygenes import version

    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: cache.parent))
    root = data_bundle.cache_dir()
    _write_cache(root)  # No new completion marker required for old caches.
    monkeypatch.setattr(version, "__version__", "999.0.0")
    importlib.reload(data_bundle)
    assert data_bundle.ensure_local(verbose=False) == root
    _assert_cache(root)


def test_missing_cache_read_only_inspection_does_not_write(cache):
    assert data_bundle.is_local() is False
    assert data_bundle.find("pan-cancer-expression.csv") is None
    assert data_bundle.status()["all_local"] is False
    with pytest.raises(FileNotFoundError, match="pirlygenes data fetch"):
        data_bundle.ensure_local(auto_fetch=False)
    assert list(cache.parent.iterdir()) == []


def test_repair_replaces_bundle_but_preserves_unrelated_data(cache, monkeypatch):
    _write_cache(cache, {name: b"old" for name in FILES})
    (cache / "cohort-expression-matrices/stale.parquet").write_bytes(b"stale")
    (cache / "other-owner").mkdir()
    (cache / "other-owner/data.csv").write_bytes(b"unrelated")
    calls = _serve(monkeypatch, _archive())
    assert data_bundle.fetch(verbose=False) == cache
    assert len(calls) == 1
    _assert_cache(cache)
    assert not (cache / "cohort-expression-matrices/stale.parquet").exists()
    assert (cache / "other-owner/data.csv").read_bytes() == b"unrelated"
    assert not data_bundle._backup_dir(cache).exists()


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("failure", ["truncated", "bad_crc", "missing", "empty", "wrong_type"])
def test_invalid_download_never_publishes(cache, monkeypatch, warm, failure):
    if warm:
        _write_cache(cache)
    if failure == "truncated":
        payload = _archive()[:-12]
    elif failure == "bad_crc":
        payload = bytearray(_archive())
        payload[-8] ^= 1
        payload = bytes(payload)
    else:
        files = dict(FILES)
        del files["pan-cancer-expression.csv"]
        if failure == "empty":
            files["pan-cancer-expression.csv"] = b""
        elif failure == "wrong_type":
            files["pan-cancer-expression.csv/child"] = b"wrong type"
        payload = _archive(files)
    _serve(monkeypatch, payload)
    with pytest.raises((EOFError, OSError, ValueError, tarfile.TarError)):
        data_bundle.fetch(verbose=False)
    if warm:
        _assert_cache(cache)
    else:
        assert not cache.exists()
    assert not list(cache.parent.glob("*.pirlygenes-stage-*"))
    _serve(monkeypatch, _archive())
    data_bundle.fetch(verbose=False)
    _assert_cache(cache)


def test_failed_extraction_leaves_live_cache_intact(cache, monkeypatch):
    _write_cache(cache)
    _serve(monkeypatch, _archive())
    extract = data_bundle._extract_bundle

    def interrupted(archive, staged):
        (staged / "pan-cancer-expression.csv").write_bytes(b"partial")
        _assert_cache(cache)
        raise OSError("interrupted extraction")

    monkeypatch.setattr(data_bundle, "_extract_bundle", interrupted)
    with pytest.raises(OSError, match="interrupted extraction"):
        data_bundle.fetch(verbose=False)
    _assert_cache(cache)
    monkeypatch.setattr(data_bundle, "_extract_bundle", extract)
    data_bundle.fetch(verbose=False)
    _assert_cache(cache)


def test_reader_cannot_see_incomplete_extraction(cache, monkeypatch):
    _serve(monkeypatch, _archive())
    extract = data_bundle._extract_bundle
    staged_ready = threading.Event()
    finish = threading.Event()

    def paused(archive, staged):
        extract(archive, staged)
        staged_ready.set()
        assert finish.wait(timeout=10)

    monkeypatch.setattr(data_bundle, "_extract_bundle", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(data_bundle.ensure_local, verbose=False)
        try:
            assert staged_ready.wait(timeout=10)
            assert not data_bundle.is_local()
            assert data_bundle.find("pan-cancer-expression.csv") is None
        finally:
            finish.set()
        future.result(timeout=10)
    _assert_cache(cache)


def test_failed_publication_restores_previous_cache(cache, monkeypatch):
    _write_cache(cache)
    _serve(monkeypatch, _archive({name: b"new" for name in FILES}))
    replace = os.replace

    def failed_publish(source, destination):
        if Path(source).name == "data":
            raise OSError("publication failed")
        return replace(source, destination)

    monkeypatch.setattr(data_bundle.os, "replace", failed_publish)
    with pytest.raises(OSError, match="publication failed"):
        data_bundle.fetch(verbose=False)
    _assert_cache(cache)
    assert not data_bundle._backup_dir(cache).exists()


def test_next_reader_recovers_interrupted_swap_without_download(cache):
    backup = data_bundle._backup_dir(cache)
    _write_cache(backup)
    orphan = cache.parent / f".{cache.name}.pirlygenes-stage-killed"
    orphan.mkdir()
    (orphan / "partial").write_bytes(b"incomplete")
    assert data_bundle.ensure_local(verbose=False) == cache
    _assert_cache(cache)
    assert not backup.exists()
    assert not orphan.exists()


def _interrupted_writer(root, payload, ready):
    os.environ["PIRLYGENES_BUNDLED_DATA"] = root
    data_bundle.urllib.request.urlopen = lambda url: io.BytesIO(payload)
    extract = data_bundle._extract_bundle

    def pause(archive, staged):
        extract(archive, staged)
        ready.set()
        threading.Event().wait(timeout=30)
        raise RuntimeError("test failed to terminate downloader")

    data_bundle._extract_bundle = pause
    data_bundle.ensure_local(verbose=False)


def test_cold_reader_recovers_after_downloader_is_killed(cache, monkeypatch):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    writer = context.Process(target=_interrupted_writer, args=(str(cache), _archive(), ready))
    writer.start()
    try:
        assert ready.wait(timeout=20)
        assert not data_bundle.is_local()
    finally:
        writer.terminate()
        writer.join(timeout=10)
    assert writer.exitcode is not None
    assert list(cache.parent.glob("*.pirlygenes-stage-*"))
    calls = _serve(monkeypatch, _archive())
    data_bundle.ensure_local(verbose=False)
    _assert_cache(cache)
    assert len(calls) == 1
    assert not list(cache.parent.glob("*.pirlygenes-stage-*"))


@pytest.mark.parametrize("unsafe", ["../escaped", "/absolute", "unrelated.csv", "symlink"])
def test_unsafe_archive_is_rejected(cache, monkeypatch, unsafe):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tf:
        member = tarfile.TarInfo(unsafe)
        if unsafe == "symlink":
            member.name = "pan-cancer-expression.csv"
            member.type = tarfile.SYMTYPE
            member.linkname = "../escaped"
        tf.addfile(member)
    _serve(monkeypatch, stream.getvalue())
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        data_bundle.ensure_local(verbose=False)
    assert not cache.exists()
    assert not (cache.parent / "escaped").exists()
