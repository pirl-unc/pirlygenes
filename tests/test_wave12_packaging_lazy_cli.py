"""Tests for Wave 12 packaging metadata, lazy package init, CLI help/completion,
and OncoRef 1.50+ compatibility."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tomllib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import oncoref
import pytest
from datacache.cli_help import HelpArgumentParser, HelpFormatter

from pirlygenes import cli

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


def test_import_pirlygenes_is_lazy_in_fresh_subprocess():
    code = """
import json
import sys
import pirlygenes

heavy = ("pandas", "numpy", "matplotlib", "oncoref", "pyensembl", "tsarina", "datacache")
loaded = [name for name in heavy if name in sys.modules]
print(json.dumps({
    "version": pirlygenes.__version__,
    "loaded": loaded,
    "dir_has_all": set(pirlygenes.__all__).issubset(set(dir(pirlygenes))),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["version"] == "6.4.0"
    assert payload["loaded"] == [], f"eager heavy imports on `import pirlygenes`: {payload['loaded']}"
    assert payload["dir_has_all"] is True


def test_lazy_init_exposes_every_all_symbol_and_rejects_unknown():
    import pirlygenes

    submodules = {"cta_policy", "downloads", "data_inventory"}
    expected_export_keys = set(pirlygenes.__all__) - {"__version__"} - submodules
    assert set(pirlygenes.EXPORT_MODULES) == expected_export_keys

    for name in pirlygenes.__all__:
        assert name in dir(pirlygenes)
        value = getattr(pirlygenes, name)
        assert value is not None, f"pirlygenes.{name} resolved to None"

    with pytest.raises(AttributeError):
        _ = pirlygenes.this_attribute_does_not_exist


def test_cli_has_argcomplete_marker_and_uses_help_argument_parser(monkeypatch):
    cli_source = (_REPO_ROOT / "pirlygenes" / "cli.py").read_bytes()
    assert b"# PYTHON_ARGCOMPLETE_OK" in cli_source[:1024]

    parser = cli._build_parser()
    assert isinstance(parser, HelpArgumentParser)
    assert parser.formatter_class is HelpFormatter

    subparsers_action = next(
        action for action in parser._actions if hasattr(action, "choices") and isinstance(action.choices, dict)
    )
    for subparser in subparsers_action.choices.values():
        assert isinstance(subparser, HelpArgumentParser)
        assert subparser.formatter_class is HelpFormatter

    seen = []
    monkeypatch.setattr(cli, "enable_argcomplete", lambda p: seen.append(p) or True)
    rc, out, err = _run_cli([])
    assert rc == 0
    assert not err
    assert "pirlygenes" in out
    assert len(seen) == 1
    assert isinstance(seen[0], HelpArgumentParser)


def test_cli_help_is_lazy_in_fresh_subprocess():
    code = """
import io
import json
import sys
from contextlib import redirect_stdout
import pirlygenes.cli as cli

buf = io.StringIO()
try:
    with redirect_stdout(buf):
        cli.main(["--help"])
except SystemExit as exc:
    exit_code = exc.code
else:
    exit_code = 0

heavy = ("pandas", "numpy", "matplotlib", "oncoref", "pyensembl", "tsarina")
loaded = [name for name in heavy if name in sys.modules]
print(json.dumps({
    "exit_code": exit_code,
    "loaded": loaded,
    "has_cache_root": "--cache-root" in buf.getvalue(),
    "has_data": "data" in buf.getvalue(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["exit_code"] == 0
    assert payload["loaded"] == [], f"heavy imports on CLI --help: {payload['loaded']}"
    assert payload["has_cache_root"] is True
    assert payload["has_data"] is True


@pytest.mark.parametrize(
    "argv_template",
    [
        ["--cache-root", "{root}", "data", "dirs", "--json"],
        ["data", "dirs", "--cache-root", "{root}", "--json"],
    ],
)
def test_cli_cache_root_flag_works_before_and_after_subcommand(
    monkeypatch, tmp_path: Path, argv_template: list[str]
):
    custom_root = tmp_path / "custom-pirlygenes-cache"
    monkeypatch.delenv("PIRLYGENES_CACHE", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)

    argv = [arg.format(root=str(custom_root)) for arg in argv_template]
    rc, out, err = _run_cli(argv)
    assert rc == 0
    assert not err
    assert not custom_root.exists(), "--cache-root data dirs must remain read-only"
    assert "PIRLYGENES_CACHE" not in os.environ

    payload = json.loads(out)
    assert payload["cache_root"]["path"] == str(custom_root)
    assert payload["bundled_data"]["path"].startswith(str(custom_root / "bundled_data"))


def test_pyproject_packaging_and_oncoref_compatibility():
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    build_system = data["build-system"]
    assert build_system["build-backend"] == "setuptools.build_meta"
    assert any(req.startswith("setuptools>=77.0.3") for req in build_system["requires"])

    project = data["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert not any("License ::" in c for c in project["classifiers"])

    deps = project["dependencies"]
    assert "datacache>=1.38.0,<2.0.0" in deps
    assert "oncoref>=1.52.0,<2.0.0" in deps

    ruff_lint = data["tool"]["ruff"]["lint"]
    assert ruff_lint["select"] == ["E4", "E7", "E9", "F"]

    major, minor, *_ = (int(part) for part in oncoref.__version__.split(".")[:3])
    assert (major, minor) >= (1, 52)
    assert major < 2
