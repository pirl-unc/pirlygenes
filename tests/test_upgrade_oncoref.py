"""Tests for the complete owner-upgrade automation."""

from pathlib import Path

import pytest
import yaml

from scripts import upgrade_oncoref


ROOT = Path(__file__).resolve().parents[1]


def test_next_patch_version_is_strict_and_deterministic():
    assert upgrade_oncoref.next_patch_version("5.23.53") == "5.23.54"
    with pytest.raises(ValueError, match="three-part numeric"):
        upgrade_oncoref.next_patch_version("5.23")


def test_prepare_updates_exact_owner_and_code_data_versions(
    tmp_path, monkeypatch,
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = [\n    "oncoref==1.8.182",\n]\n')
    version_module = tmp_path / "version.py"
    version_module.write_text(
        '__version__ = "5.23.53"\nDATA_VERSION = "5.23.53"\n'
    )
    monkeypatch.setattr(upgrade_oncoref, "PYPROJECT", pyproject)
    monkeypatch.setattr(upgrade_oncoref, "VERSION_MODULE", version_module)

    assert upgrade_oncoref.prepare("1.8.183") == "5.23.54"

    assert '"oncoref==1.8.183"' in pyproject.read_text()
    assert '__version__ = "5.23.54"' in version_module.read_text()
    assert 'DATA_VERSION = "5.23.54"' in version_module.read_text()


def test_prepare_rejects_equal_or_older_owner_release(
    tmp_path, monkeypatch,
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = [\n    "oncoref==1.8.182",\n]\n')
    version_module = tmp_path / "version.py"
    version_module.write_text(
        '__version__ = "5.23.53"\nDATA_VERSION = "5.23.53"\n'
    )
    monkeypatch.setattr(upgrade_oncoref, "PYPROJECT", pyproject)
    monkeypatch.setattr(upgrade_oncoref, "VERSION_MODULE", version_module)

    with pytest.raises(ValueError, match="is not newer"):
        upgrade_oncoref.prepare("1.8.182")
    with pytest.raises(ValueError, match="is not newer"):
        upgrade_oncoref.prepare("1.8.181")


def test_upgrade_workflow_is_scheduled_and_dispatches_full_ci():
    path = ROOT / ".github" / "workflows" / "upgrade-oncoref.yml"
    text = path.read_text()
    workflow = yaml.safe_load(text)

    # PyYAML 1.1 parses the unquoted GitHub key ``on`` as True.
    triggers = workflow.get("on", workflow.get(True))
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {
        "actions": "write",
        "contents": "write",
        "pull-requests": "write",
    }
    assert "python scripts/upgrade_oncoref.py regenerate" in text
    assert "gh workflow run tests.yml" in text
    assert "--set-upstream origin" in text


def test_normal_ci_accepts_explicit_dispatch_for_bot_prs():
    text = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert "workflow_dispatch:" in text


def test_dependabot_no_longer_opens_incomplete_pin_only_prs():
    assert not (ROOT / ".github" / "dependabot.yml").exists()
