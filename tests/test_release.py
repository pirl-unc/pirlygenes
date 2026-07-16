"""Release preflight guards for versioned downloadable artifacts."""

import json

import pytest

from pirlygenes.version import DATA_VERSION
from scripts import release


def _manifest(tmp_path, monkeypatch, payload):
    path = tmp_path / "_manifest.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(release, "COHORT_VIEWS_MANIFEST", path)
    return path


def test_release_accepts_current_cohort_views_manifest(tmp_path, monkeypatch):
    _manifest(tmp_path, monkeypatch, {"data_version": DATA_VERSION})
    release.validate_data_bundle_manifests()


def test_release_rejects_stale_cohort_views_manifest(tmp_path, monkeypatch):
    _manifest(tmp_path, monkeypatch, {"data_version": "5.23.19"})
    with pytest.raises(release.Abort, match="data_version mismatch"):
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
