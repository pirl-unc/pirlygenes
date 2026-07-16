"""Repository-wide pytest memory-safety guards.

The data-heavy expression tests materialize multi-million-row reference frames
inside each process. Unbounded xdist fan-out can therefore consume several GB
per worker and make the machine unresponsive before macOS can recover.
"""

import os

import pytest


_PARALLEL_OPT_IN = "PIRLYGENES_ALLOW_PARALLEL_TESTS"
_MAX_EXPLICIT_WORKERS = 2


@pytest.hookimpl(tryfirst=True, optionalhook=True)
def pytest_xdist_auto_num_workers(config):
    """Make ``-n auto`` and ``-n logical`` safe even when passed explicitly."""
    del config
    return 0


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Reject every other accidental or unbounded xdist configuration."""
    if hasattr(config, "workerinput"):
        return

    workers = getattr(config.option, "numprocesses", None)
    transports = getattr(config.option, "tx", None) or []
    if transports and workers in (None, 0):
        raise pytest.UsageError(
            "pirlygenes disables direct xdist --tx configurations because "
            "they bypass the worker limit. Use an explicit -n 1 or -n 2 "
            f"with {_PARALLEL_OPT_IN}=1 for a known-lightweight subset."
        )
    if workers in (None, 0):
        return
    if not isinstance(workers, int) or workers > _MAX_EXPLICIT_WORKERS:
        raise pytest.UsageError(
            "pirlygenes permits at most 2 explicit xdist workers; each worker "
            "can otherwise materialize several GB of reference data."
        )
    if os.environ.get(_PARALLEL_OPT_IN) != "1":
        raise pytest.UsageError(
            "pirlygenes disables xdist workers by default because each worker "
            "can materialize several GB of reference data. Run serially, or "
            f"set {_PARALLEL_OPT_IN}=1 and use -n 1 or -n 2 only for a "
            "known-lightweight subset."
        )
