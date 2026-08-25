"""Prepare a complete pirlygenes upgrade to the latest oncoref release.

The oncoref dependency cannot safely float at install time: pirlygenes ships a
wide compatibility cache whose manifest records the exact oncoref package and
data versions used to build it.  This helper is the deterministic core of the
maintainer-run upgrade process:

    python scripts/upgrade_oncoref.py check
    python scripts/upgrade_oncoref.py prepare --version 1.8.183
    python scripts/upgrade_oncoref.py regenerate

``check`` reports whether PyPI has a newer release. ``prepare`` changes only
version declarations. After installing the newly pinned project, ``regenerate``
runs the heavyweight artifact builders in separate processes to keep peak
memory bounded and refreshes their reviewable snapshots. The resulting diff is
reviewed and submitted through the normal maintainer PR process; this helper
does not create branches, PRs, merges, or releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_MODULE = ROOT / "pirlygenes" / "version.py"
PARITY_DIR = ROOT / "analyses" / "outputs" / "reference_expression_parity"
DOC_PARITY_CSV = ROOT / "docs" / "reference-expression-delegation-557.csv"
DOC_PARITY_REPORT = ROOT / "docs" / "reference-expression-delegation-557.md"
MANIFEST_SNAPSHOT = ROOT / "tests" / "data" / "reference-availability.sha256"

_RELEASE_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_ONCOREF_PIN_RE = re.compile(r'(?m)^\s*"oncoref==(?P<version>[0-9.]+)",$')
_CODE_VERSION_RE = re.compile(
    r'(?m)^__version__ = "(?P<version>[0-9.]+)"$'
)
_DATA_VERSION_RE = re.compile(
    r'(?m)^DATA_VERSION = "(?P<version>[0-9.]+)"$'
)


def _release_tuple(value: str) -> tuple[int, int, int]:
    if not _RELEASE_RE.fullmatch(value):
        raise ValueError(f"expected a three-part numeric release, got {value!r}")
    major, minor, patch = (int(part) for part in value.split("."))
    return major, minor, patch


def _single_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}; found {len(matches)}")
    return matches[0]


def pinned_oncoref_version(pyproject: Path | None = None) -> str:
    """Return the exact oncoref version declared by this checkout."""
    pyproject = PYPROJECT if pyproject is None else pyproject
    return _single_match(
        _ONCOREF_PIN_RE,
        pyproject.read_text(),
        "oncoref pin",
    ).group("version")


def pirlygenes_version(version_module: Path | None = None) -> str:
    """Return the package version declared by this checkout."""
    version_module = VERSION_MODULE if version_module is None else version_module
    return _single_match(
        _CODE_VERSION_RE,
        version_module.read_text(),
        "pirlygenes code version",
    ).group("version")


def latest_oncoref_version() -> str:
    """Return PyPI's latest stable oncoref version using only the stdlib."""
    request = Request(
        "https://pypi.org/pypi/oncoref/json",
        headers={"User-Agent": "pirlygenes-oncoref-upgrader/1"},
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS URL
        latest = json.load(response)["info"]["version"]
    _release_tuple(latest)
    return latest


def next_patch_version(current: str) -> str:
    """Increment the final component of a three-part release."""
    major, minor, patch = _release_tuple(current)
    return f"{major}.{minor}.{patch + 1}"


def check() -> bool:
    """Print version status and return whether an upgrade is available."""
    current = pinned_oncoref_version()
    latest = latest_oncoref_version()
    upgrade_required = _release_tuple(latest) > _release_tuple(current)
    print(f"current={current}")
    print(f"latest={latest}")
    print(f"upgrade_required={str(upgrade_required).lower()}")
    return upgrade_required


def _replace_version(
    path: Path,
    pattern: re.Pattern[str],
    replacement: str,
    label: str,
) -> None:
    text = path.read_text()
    match = _single_match(pattern, text, label)
    path.write_text(text[: match.start()] + replacement + text[match.end() :])


def prepare(target: str) -> str:
    """Pin *target* and allocate a new pirlygenes code/data patch version."""
    _release_tuple(target)
    current_owner = pinned_oncoref_version()
    if _release_tuple(target) <= _release_tuple(current_owner):
        raise ValueError(
            f"target oncoref {target} is not newer than the pin {current_owner}"
        )

    current_package = pirlygenes_version()
    next_package = next_patch_version(current_package)
    _replace_version(
        PYPROJECT,
        _ONCOREF_PIN_RE,
        f'    "oncoref=={target}",',
        "oncoref pin",
    )
    _replace_version(
        VERSION_MODULE,
        _CODE_VERSION_RE,
        f'__version__ = "{next_package}"',
        "pirlygenes code version",
    )
    _replace_version(
        VERSION_MODULE,
        _DATA_VERSION_RE,
        f'DATA_VERSION = "{next_package}"',
        "pirlygenes data version",
    )
    print(
        f"prepared oncoref {current_owner} -> {target}; "
        f"pirlygenes {current_package} -> {next_package}"
    )
    return next_package


def _run(*parts: str) -> None:
    print("+", " ".join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True)


def _write_manifest_snapshot() -> None:
    """Write the canonical public availability-frame checksum."""
    import pandas as pd

    from pirlygenes.expression import available_cancer_expression_references

    result = available_cancer_expression_references().copy()
    for column in result:
        result[column] = result[column].map(
            lambda value: "<NA>" if pd.isna(value) else str(value)
        )
    payload = result.to_csv(index=False, lineterminator="\n").encode()
    MANIFEST_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_SNAPSHOT.write_text(hashlib.sha256(payload).hexdigest() + "\n")
    print(f"wrote {MANIFEST_SNAPSHOT.relative_to(ROOT)}")


def regenerate() -> None:
    """Rebuild all artifacts coupled to the exact oncoref pin."""
    import oncoref

    expected = pinned_oncoref_version()
    if oncoref.__version__ != expected:
        raise RuntimeError(
            f"installed oncoref is {oncoref.__version__}; expected {expected}"
        )

    _run(sys.executable, "scripts/generate_cohort_expression_views.py")
    _run(sys.executable, "scripts/generate_pan_cancer_expression_rollups.py")
    _run(sys.executable, "scripts/parity_reference_expression.py")
    shutil.copyfile(PARITY_DIR / "parity_by_code.csv", DOC_PARITY_CSV)
    shutil.copyfile(PARITY_DIR / "parity_report.md", DOC_PARITY_REPORT)
    _write_manifest_snapshot()
    print(
        f"regenerated owner-derived artifacts for oncoref {oncoref.__version__}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="report whether PyPI has a newer release")
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="update the exact pin and bump pirlygenes code/data versions",
    )
    prepare_parser.add_argument("--version", required=True)
    subparsers.add_parser(
        "regenerate",
        help="rebuild all artifacts coupled to the installed owner release",
    )

    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "prepare":
        prepare(args.version)
    else:
        regenerate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
