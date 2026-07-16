"""Safety contracts for the repository's test-runner configuration."""

from pathlib import Path
import re
import shlex


_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


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
