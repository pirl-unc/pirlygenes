"""Release preflight guards for versioned downloadable artifacts."""

import json
from types import SimpleNamespace

import oncoref
import pytest
from oncoref import data_bundle as oncoref_data_bundle

from pirlygenes.expression import (
    COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
    COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
)
from pirlygenes.version import DATA_VERSION
from scripts import release


def _current_metadata_payload():
    return {
        "artifact_type": COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
        "canonical_gene_ids": True,
        "schema_version": COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
        "pirlygenes_data_version": DATA_VERSION,
        "built_from": {
            "data_version": oncoref_data_bundle.DATA_VERSION,
            "package": "oncoref",
            "package_version": oncoref.__version__,
        },
        "tables": {
            "tpm": {"file": "tpm.parquet", "rows": 2},
            "clean_tpm": {"file": "clean_tpm.parquet", "rows": 2},
            "provenance": {"file": "provenance.parquet", "rows": 1},
        },
    }


def _metadata(tmp_path, monkeypatch, payload):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(release, "COHORT_MATRICES_METADATA", path)
    return path


def test_release_accepts_current_cohort_matrices_metadata(tmp_path, monkeypatch):
    _metadata(tmp_path, monkeypatch, _current_metadata_payload())
    release.validate_data_bundle_metadata()


def test_release_rejects_stale_cohort_matrices_metadata(tmp_path, monkeypatch):
    payload = _current_metadata_payload()
    payload["pirlygenes_data_version"] = "stale-data-version"
    _metadata(tmp_path, monkeypatch, payload)
    with pytest.raises(release.Abort, match="pirlygenes_data_version mismatch"):
        release.validate_data_bundle_metadata()


def test_release_rejects_noncanonical_cohort_matrices_metadata(
    tmp_path, monkeypatch,
):
    payload = _current_metadata_payload()
    payload["canonical_gene_ids"] = False
    _metadata(tmp_path, monkeypatch, payload)
    with pytest.raises(release.Abort, match="canonical_gene_ids=true"):
        release.validate_data_bundle_metadata()


def test_release_rejects_missing_or_malformed_metadata(tmp_path, monkeypatch):
    path = tmp_path / "missing.json"
    monkeypatch.setattr(release, "COHORT_MATRICES_METADATA", path)
    with pytest.raises(release.Abort, match="metadata missing"):
        release.validate_data_bundle_metadata()

    path.write_text("{ not json")
    with pytest.raises(release.Abort, match="metadata unreadable"):
        release.validate_data_bundle_metadata()

    path.write_text("[]")
    with pytest.raises(release.Abort, match="metadata invalid"):
        release.validate_data_bundle_metadata()


@pytest.mark.parametrize(
    "field",
    ["package", "data_version"],
)
def test_release_rejects_stale_owner_metadata(tmp_path, monkeypatch, field):
    payload = _current_metadata_payload()
    payload["built_from"][field] = "stale-owner"
    _metadata(tmp_path, monkeypatch, payload)

    with pytest.raises(release.Abort, match=field):
        release.validate_data_bundle_metadata()


def test_release_accepts_historical_owner_wheel_provenance(
    tmp_path,
    monkeypatch,
):
    payload = _current_metadata_payload()
    payload["built_from"]["package_version"] = "older-wheel-same-data"
    _metadata(tmp_path, monkeypatch, payload)

    release.validate_data_bundle_metadata()


def test_release_requires_owner_wheel_provenance(tmp_path, monkeypatch):
    payload = _current_metadata_payload()
    payload["built_from"].pop("package_version")
    _metadata(tmp_path, monkeypatch, payload)

    with pytest.raises(release.Abort, match="package_version"):
        release.validate_data_bundle_metadata()


def test_publish_validates_metadata_before_reusing_existing_asset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        release,
        "validate_data_bundle_metadata",
        lambda: calls.append("validate"),
    )
    monkeypatch.setattr(
        release, "release_metadata",
        lambda *_args: {"assets": [{"name": release.TARBALL.name}]},
    )

    release.publish_data(dry=True, assume_yes=True, target_sha="test-sha")

    assert calls == ["validate"]


def test_build_bootstraps_pip_when_uv_venv_omits_it(monkeypatch):
    calls = []
    monkeypatch.setattr(release.shutil, "rmtree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(
        release,
        "_run",
        lambda command, **_kwargs: calls.append(command),
    )

    release.build_wheel(dry=False)

    assert calls[0][1:] == ["-m", "ensurepip", "--upgrade"]
    assert calls[1][1:] == [
        "-m", "pip", "install", "--upgrade", "build", "twine",
    ]
    assert calls[2][1:] == ["-m", "build"]


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_github_release_lookup_failure_aborts(monkeypatch, status):
    monkeypatch.setattr(
        release.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout=json.dumps({"status": str(status)}), stderr=f"HTTP {status}",
        ),
    )
    with pytest.raises(release.Abort, match="cannot inspect GitHub release"):
        release.release_metadata("v6.0.12")


def test_github_only_404_means_missing_release(monkeypatch):
    monkeypatch.setattr(
        release.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout='{"status":"404"}', stderr="HTTP 404"),
    )
    assert release.release_metadata("v6.0.12") is None


@pytest.mark.parametrize("stdout", ["", "[]", "{}", "not json"])
def test_github_malformed_release_response_aborts(monkeypatch, stdout):
    monkeypatch.setattr(
        release.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    with pytest.raises(release.Abort):
        release.release_metadata("v6.0.12")


def test_data_upload_never_overwrites_an_existing_asset(monkeypatch):
    calls = []
    monkeypatch.setattr(release, "validate_data_bundle_metadata", lambda: None)
    monkeypatch.setattr(release, "release_metadata", lambda tag: {"assets": []})
    monkeypatch.setattr(release, "build_data_tarball", lambda **k: None)
    monkeypatch.setattr(release, "pre_extract_cache", lambda **k: None)
    monkeypatch.setattr(release, "_run", lambda command, **k: calls.append(command))
    release.publish_data(dry=False, assume_yes=True, target_sha="release-commit")
    assert calls == [["gh", "release", "upload", release.DATA_TAG, str(release.TARBALL)]]


@pytest.mark.parametrize("fail_check", [1, 2])
def test_release_stops_before_writes_when_publication_state_changes(monkeypatch, fail_check):
    from scripts.release_versions import ReleaseVersionError

    calls = []
    def check(**kwargs):
        calls.append("check")
        if calls.count("check") == fail_check:
            raise ReleaseVersionError("version was taken by another release")

    monkeypatch.setattr(release, "check_publication", check)
    monkeypatch.setattr(release, "preflight", lambda **k: None)
    monkeypatch.setattr(release.sys, "argv", ["release.py", "--execute", "--yes"])
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="release-commit"))
    for name in ("publish_data", "validate", "build_wheel", "publish_pypi", "finalize_release"):
        monkeypatch.setattr(release, name, lambda name=name, **k: calls.append(name))
    with pytest.raises(SystemExit):
        release.main()
    assert calls == (["check"] if fail_check == 1 else ["check", "publish_data", "validate", "build_wheel", "check"])


def test_finalization_does_not_push_unrelated_local_tags(monkeypatch):
    calls = []
    monkeypatch.setattr(release, "release_metadata", lambda tag: None)
    monkeypatch.setattr(release, "_run", lambda command, **k: calls.append(command))
    release.finalize_release(dry=False, assume_yes=True, target_sha="release-commit")
    assert calls == [["gh", "release", "create", release.CODE_TAG, "--target", "release-commit", "--title", release.CODE_TAG, "--generate-notes"]]


def test_deploy_wrapper_forwards_options_to_guarded_entrypoint(tmp_path):
    import os
    from pathlib import Path
    import subprocess

    invocation = tmp_path / "arguments.json"
    interpreter = tmp_path / "test python"
    interpreter.write_text(
        f"#!{release.sys.executable}\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(invocation)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    interpreter.chmod(0o755)
    subprocess.run(
        ["bash", str(Path(release.REPO_ROOT) / "deploy.sh"), "--yes", "--skip-tests"],
        cwd=tmp_path, env={**os.environ, "PYTHON": str(interpreter)}, check=True,
    )
    args = json.loads(invocation.read_text())
    assert Path(args[0]).resolve() == release.REPO_ROOT / "scripts/release.py"
    assert args[1:] == ["--execute", "--yes", "--skip-tests"]
