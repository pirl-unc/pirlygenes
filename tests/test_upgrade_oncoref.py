"""Tests for the maintainer-run oncoref upgrade helper."""

import pytest

from scripts import upgrade_oncoref


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


def test_check_reports_available_owner_release(monkeypatch, capsys):
    monkeypatch.setattr(
        upgrade_oncoref, "pinned_oncoref_version", lambda: "1.8.182"
    )
    monkeypatch.setattr(
        upgrade_oncoref, "latest_oncoref_version", lambda: "1.8.183"
    )

    assert upgrade_oncoref.check() is True
    assert capsys.readouterr().out.splitlines() == [
        "current=1.8.182",
        "latest=1.8.183",
        "upgrade_required=true",
    ]


def test_regenerate_fails_before_writes_when_rollup_sources_drift(monkeypatch):
    import oncoref
    from scripts import generate_pan_cancer_expression_rollups

    def fail_source_validation():
        raise RuntimeError("source drift")

    monkeypatch.setattr(
        upgrade_oncoref,
        "pinned_oncoref_version",
        lambda: oncoref.__version__,
    )
    monkeypatch.setattr(
        generate_pan_cancer_expression_rollups,
        "validate_selected_source_shards",
        fail_source_validation,
    )
    monkeypatch.setattr(
        upgrade_oncoref,
        "_run",
        lambda *_: pytest.fail("artifact builder ran before source validation"),
    )

    with pytest.raises(RuntimeError, match="source drift"):
        upgrade_oncoref.regenerate()
