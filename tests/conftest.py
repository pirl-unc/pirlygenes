"""Shared pytest setup.

Default Ensembl release: the gene-family generator and several gene-id tests
read pyensembl GTF databases. We default to **release 111** as the baseline and
ensure it's installed once, up front, so release-dependent tests are meaningful
rather than silently skipped.

Why here (and only in the xdist *controller*): the repository defaults to
serial execution, but permits at most two explicitly opted-in workers for
known-lightweight subsets. Installing inside a per-worker fixture would let
those workers race to ``pyensembl install`` into the same cache.
``pytest_configure`` runs in the controller before workers fork, so gating on
``config.workerinput`` makes the ensure-install happen exactly once.

It's best-effort: a no-op if 111 is already built (CI installs + caches it in the
workflow, so this no-ops there too), skipped when opted out via
``PIRLYGENES_NO_ENSEMBL_INSTALL`` or when the ``pyensembl`` CLI is missing, and
non-fatal if the download fails (offline) — the release-dependent tests then skip
on their own.
"""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

_DEFAULT_RELEASE = 111


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "gene_family_generator",
        Path(__file__).resolve().parent.parent / "scripts" / "generate_gene_family_sets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _release_installed(release: int) -> bool:
    try:
        return release in _generator_module()._installed_grch38_releases()
    except Exception:
        return False


def pytest_configure(config):
    # Only the xdist controller (no ``workerinput``) does the one-time install.
    if hasattr(config, "workerinput"):
        return
    # Opt-out for environments that pre-provision pyensembl or run offline.
    # (CI installs + caches release 111 in the workflow, so the check below
    # already short-circuits there — no special-casing needed.)
    if os.environ.get("PIRLYGENES_NO_ENSEMBL_INSTALL"):
        return
    if _release_installed(_DEFAULT_RELEASE) or shutil.which("pyensembl") is None:
        return
    try:
        subprocess.run(
            ["pyensembl", "install", "--release", str(_DEFAULT_RELEASE),
             "--species", "homo_sapiens"],
            check=True, timeout=900,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # non-fatal; release-dependent tests skip if 111 stays unavailable
