"""Enforce the checkout's oncoref pin before producing figure artifacts."""

from __future__ import annotations

import sys
from importlib import metadata

from scripts.upgrade_oncoref import pinned_oncoref_version


class FigureEnvironmentError(RuntimeError):
    """The figure batch cannot safely continue in this environment."""


def check_figure_environment(expected: str | None = None) -> str:
    """Check both installed metadata and imported code, including between jobs.

    Keep the initial return value for subsequent checks so editing the pin
    during a batch cannot silently switch that batch to a different release.
    This detects version drift; use a dedicated venv to prevent other sessions
    from replacing packages while a figure subprocess is using them.
    """
    pinned = pinned_oncoref_version()
    if expected is not None and pinned != expected:
        raise FigureEnvironmentError(
            f"oncoref pin changed during the figure batch: {expected} -> {pinned}. "
            "Start a new run after the environment and checkout agree."
        )
    try:
        installed = metadata.version("oncoref")
        import oncoref
        imported = oncoref.__version__
    except (metadata.PackageNotFoundError, ImportError) as exc:
        raise FigureEnvironmentError(
            f"Cannot load oncoref=={pinned} with {sys.executable}. "
            "Install this checkout in a dedicated figure environment; "
            "see the Figure outputs section in CLAUDE.md."
        ) from exc
    if installed != pinned or imported != pinned:
        raise FigureEnvironmentError(
            f"oncoref version mismatch: pyproject.toml pins {pinned}, "
            f"installed metadata reports {installed}, imported code reports "
            f"{imported} ({sys.executable}). "
            "Use a dedicated environment matching this checkout, for example "
            ".venv-figures/bin/python analyses/regenerate_plots.py; "
            "see the Figure outputs section in CLAUDE.md for setup."
        )
    return pinned
