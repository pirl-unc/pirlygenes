"""Release preflight guards for versioned downloadable artifacts."""

import json
from types import SimpleNamespace

import oncoref
import pytest
from oncoref import data_bundle as oncoref_data_bundle

from pirlygenes.version import DATA_VERSION
from scripts import release


def _current_manifest_payload():
    return {
        "canonical_gene_ids": True,
        "data_version": DATA_VERSION,
        "source_data_version": oncoref_data_bundle.DATA_VERSION,
        "source_package": "oncoref",
        "source_package_version": oncoref.__version__,
    }


def _manifest(tmp_path, monkeypatch, payload):
    path = tmp_path / "_manifest.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(release, "COHORT_VIEWS_MANIFEST", path)
    return path


def test_release_accepts_current_cohort_views_manifest(tmp_path, monkeypatch):
    _manifest(tmp_path, monkeypatch, _current_manifest_payload())
    release.validate_data_bundle_manifests()


def test_release_rejects_stale_cohort_views_manifest(tmp_path, monkeypatch):
    payload = _current_manifest_payload()
    payload["data_version"] = "stale-data-version"
    _manifest(tmp_path, monkeypatch, payload)
    with pytest.raises(release.Abort, match="data_version mismatch"):
        release.validate_data_bundle_manifests()


def test_release_rejects_noncanonical_cohort_views_manifest(
    tmp_path, monkeypatch,
):
    payload = _current_manifest_payload()
    payload["canonical_gene_ids"] = False
    _manifest(tmp_path, monkeypatch, payload)
    with pytest.raises(release.Abort, match="canonical_gene_ids=true"):
        release.validate_data_bundle_manifests()


def test_release_rejects_missing_or_malformed_manifest(tmp_path, monkeypatch):
    path = tmp_path / "missing.json"
    monkeypatch.setattr(release, "COHORT_VIEWS_MANIFEST", path)
    with pytest.raises(release.Abort, match="manifest missing"):
        release.validate_data_bundle_manifests()

    path.write_text("{ not json")
    with pytest.raises(release.Abort, match="manifest unreadable"):
        release.validate_data_bundle_manifests()

    path.write_text("[]")
    with pytest.raises(release.Abort, match="manifest invalid"):
        release.validate_data_bundle_manifests()


@pytest.mark.parametrize(
    "field",
    ["source_package", "source_package_version", "source_data_version"],
)
def test_release_rejects_stale_owner_manifest(tmp_path, monkeypatch, field):
    payload = _current_manifest_payload()
    payload[field] = "stale-owner"
    _manifest(tmp_path, monkeypatch, payload)

    with pytest.raises(release.Abort, match=field):
        release.validate_data_bundle_manifests()


def test_publish_validates_manifest_before_reusing_existing_asset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        release,
        "validate_data_bundle_manifests",
        lambda: calls.append("validate"),
    )
    monkeypatch.setattr(release, "_release_has_asset", lambda *_args: True)

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
