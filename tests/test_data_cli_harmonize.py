"""Tests for harmonized datacache location, offline mode, and data CLI."""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import datacache

from pirlygenes import cli, data_bundle, downloads, gene_ids


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


def _write_fake_bundle(version_dir: Path) -> None:
    (version_dir / "cohort-expression-matrices").mkdir(parents=True, exist_ok=True)
    (version_dir / "cohort-expression-matrices" / "a.parquet").write_bytes(b"parquet")
    (version_dir / "pan-cancer-expression.csv").write_text("gene,v\nA,1\n", encoding="utf-8")
    (version_dir / "hpa-cell-type-expression.csv").write_text("gene,v\nA,2\n", encoding="utf-8")


def test_cache_resolution_precedence_and_legacy_preservation(monkeypatch, tmp_path: Path):
    import datacache.common as dc_common

    fake_home = tmp_path / "home"
    legacy_dir = fake_home / ".cache" / "pirlygenes"
    platform_shared = tmp_path / "platform-openvax"

    monkeypatch.delenv("PIRLYGENES_CACHE", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)
    monkeypatch.delenv("OPENVAX_DATA_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(
        dc_common,
        "_platform_cache_dir",
        lambda app: str(platform_shared if app == "openvax" else tmp_path / app),
    )

    # Clean install defaults to shared platform root without creating it.
    loc = downloads.resolve_cache_location()
    assert Path(loc.path) == platform_shared / "pirlygenes"
    assert loc.origin == "default"
    assert not Path(loc.path).exists()
    assert data_bundle.cache_root() == platform_shared / "pirlygenes" / "bundled_data"

    # Populated legacy ~/.cache/pirlygenes is preserved (no forced re-downloads).
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "marker.txt").write_text("existing", encoding="utf-8")
    loc_legacy = downloads.resolve_cache_location()
    assert Path(loc_legacy.path) == legacy_dir
    assert loc_legacy.origin == "legacy"

    # OPENVAX_DATA_CACHE overrides legacy/default.
    shared_override = tmp_path / "openvax-shared"
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(shared_override))
    loc_shared = downloads.resolve_cache_location()
    assert Path(loc_shared.path) == shared_override / "pirlygenes"
    assert loc_shared.origin == "shared_env"
    assert data_bundle.cache_root() == shared_override / "pirlygenes" / "bundled_data"
    assert gene_ids._index_cache_path(112).parent == shared_override / "pirlygenes"

    # PIRLYGENES_CACHE takes highest precedence.
    pkg_override = tmp_path / "pkg-override"
    monkeypatch.setenv("PIRLYGENES_CACHE", str(pkg_override))
    loc_pkg = downloads.resolve_cache_location()
    assert Path(loc_pkg.path) == pkg_override
    assert loc_pkg.origin == "env"


def test_cli_data_dirs_is_read_only_and_supports_json(monkeypatch, tmp_path: Path):
    shared = tmp_path / "shared-cache"
    monkeypatch.delenv("PIRLYGENES_CACHE", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(shared))
    monkeypatch.delenv("OPENVAX_OFFLINE", raising=False)
    monkeypatch.delenv("DATACACHE_OFFLINE", raising=False)

    rc, out, err = _run_cli(["data", "dirs", "--json"])
    assert rc == 0
    assert not err
    assert not shared.exists(), "data dirs must remain strictly read-only"

    payload = json.loads(out)
    assert payload["package"] == "pirlygenes"
    assert payload["offline"] is False
    assert payload["cache_root"]["path"] == str(shared / "pirlygenes")
    assert payload["cache_root"]["origin"] == "shared_env"
    assert payload["cache_root"]["exists"] is False
    assert payload["bundled_data"]["path"] == str(
        shared / "pirlygenes" / "bundled_data" / f"v{data_bundle.DATA_VERSION}"
    )
    assert payload["bundled_data"]["all_local"] is False

    # Human output also works and remains read-only.
    rc2, out2, err2 = _run_cli(["data", "dirs"])
    assert rc2 == 0
    assert not err2
    assert "cache_root" in out2
    assert "bundled_data" in out2
    assert not shared.exists()


def test_cli_path_aliases_and_status_json(monkeypatch, tmp_path: Path):
    from oncoref import source_matrices

    bundle_dir = tmp_path / f"v{data_bundle.DATA_VERSION}"
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(bundle_dir))
    monkeypatch.setenv("CANCERDATA_SOURCE_MATRICES", str(tmp_path / "matrices"))

    rc, out, err = _run_cli(["data", "path"])
    assert rc == 0 and not err
    assert out.strip() == str(bundle_dir)

    rc, out, err = _run_cli(["data", "path", "--json"])
    assert rc == 0 and not err
    assert json.loads(out)["path"] == str(bundle_dir)

    rc, out, err = _run_cli(["downloads", "path", "--json"])
    assert rc == 0 and not err
    assert json.loads(out)["path"] == str(source_matrices.cache_dir())

    rc, out, err = _run_cli(["data", "status", "--json"])
    assert rc == 0 and not err
    snap = json.loads(out)
    assert snap["cache_dir"] == str(bundle_dir)
    assert snap["all_local"] is False
    assert not bundle_dir.exists()


def test_cli_data_prune_json_dry_run_and_apply(monkeypatch, tmp_path: Path):
    root = tmp_path / "bundled_data"
    current = root / f"v{data_bundle.DATA_VERSION}"
    stale = root / "v0.0.1"
    _write_fake_bundle(current)
    _write_fake_bundle(stale)
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(current))

    # Dry-run default with --json
    rc, out, err = _run_cli(["data", "prune", "--json"])
    assert rc == 0 and not err
    plan = json.loads(out)
    assert plan["dry_run"] is True
    assert [c["version"] for c in plan["candidates"]] == ["v0.0.1"]
    assert plan["deleted"] == []
    assert plan["reclaimable_bytes"] > 0
    assert plan["freed_bytes"] == 0
    assert stale.exists()

    # Apply with --yes --json
    rc2, out2, err2 = _run_cli(["data", "prune", "--yes", "--json"])
    assert rc2 == 0 and not err2
    applied = json.loads(out2)
    assert applied["dry_run"] is False
    assert [c["version"] for c in applied["deleted"]] == ["v0.0.1"]
    assert applied["freed_bytes"] > 0
    assert not stale.exists()
    assert current.exists()


def test_cli_data_fetch_offline_blocks_network(monkeypatch, tmp_path: Path):
    bundle_dir = tmp_path / f"v{data_bundle.DATA_VERSION}"
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(bundle_dir))
    monkeypatch.setattr(
        data_bundle,
        "RELEASE_URL",
        "https://example.invalid/pirlygenes-test-release.tar.gz",
    )

    rc, out, err = _run_cli(["data", "fetch", "--offline"])
    assert rc == 1
    assert not out
    assert "offline" in err.lower()
    assert not bundle_dir.exists()


def test_cli_downloads_fetch_offline_sets_env_and_handles_offline_error(monkeypatch):
    from oncoref import source_matrices

    seen_env: list[str | None] = []

    def fake_fetch(_code: str):
        seen_env.append(os.environ.get("DATACACHE_OFFLINE"))
        raise datacache.OfflineError("offline mode enabled")

    monkeypatch.delenv("DATACACHE_OFFLINE", raising=False)
    monkeypatch.setattr(source_matrices, "fetch", fake_fetch)

    rc, out, err = _run_cli(["downloads", "fetch", "--offline", "PRAD"])
    assert rc == 2
    assert not out
    assert seen_env == ["1"]
    assert "offline" in err.lower()
    assert "DATACACHE_OFFLINE" not in os.environ
