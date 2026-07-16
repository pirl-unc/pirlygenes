"""Safety contracts for the repository's test-runner configuration."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
_THIS_FILE = Path(__file__).resolve()
_TARGET = f"{_THIS_FILE}::test_pytest_defaults_to_serial_execution"
_SERIAL_TARGET = f"{_THIS_FILE}::test_unopted_process_is_not_xdist_worker"
_PARALLEL_OPT_IN = "PIRLYGENES_ALLOW_PARALLEL_TESTS"


def _nested_pytest(*args, allow_parallel=False):
    env = os.environ.copy()
    env["PIRLYGENES_NO_ENSEMBL_INSTALL"] = "1"
    if allow_parallel:
        env[_PARALLEL_OPT_IN] = "1"
    else:
        env.pop(_PARALLEL_OPT_IN, None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=_PYPROJECT.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_pytest_defaults_to_serial_execution():
    """Plain ``pytest`` must not replicate large reference frames per worker."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)", text,
    )
    assert section, "pyproject.toml is missing [tool.pytest.ini_options]"

    configured = re.search(
        r'''(?m)^addopts\s*=\s*(["'])(.*?)\1''', section.group(1),
    )
    addopts = shlex.split(configured.group(2)) if configured else []
    parallel_options = [
        option
        for option in addopts
        if option.startswith("-n")
        or option.startswith("--numprocesses")
    ]
    assert not parallel_options, (
        "The full suite materializes multi-million-row reference frames; "
        "default xdist workers multiply peak memory. Keep the repo default "
        "serial and opt into parallelism only for lightweight subsets."
    )


def test_unopted_process_is_not_xdist_worker():
    if os.environ.get(_PARALLEL_OPT_IN) != "1":
        assert "PYTEST_XDIST_WORKER" not in os.environ


def test_xdist_auto_is_forced_to_serial_execution():
    result = _nested_pytest("-n", "auto", "-q", _SERIAL_TARGET)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output
    assert "bringing up nodes" not in output


def test_explicit_xdist_requires_memory_safety_opt_in():
    result = _nested_pytest("-n", "2", "--collect-only", _TARGET)
    output = result.stdout + result.stderr
    assert result.returncode == 4, output
    assert _PARALLEL_OPT_IN in output


def test_explicit_xdist_worker_count_is_capped():
    result = _nested_pytest(
        "-n", "3", "--collect-only", _TARGET, allow_parallel=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 4, output
    assert "at most 2 explicit xdist workers" in output


def test_bounded_xdist_opt_in_remains_available():
    result = _nested_pytest("-n", "2", "-q", _TARGET, allow_parallel=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output
