"""Figure batches must reject dependency drift before producing more output."""

import sys
from importlib import metadata
from types import SimpleNamespace

import pytest

from scripts import figure_environment


@pytest.fixture
def environment(monkeypatch):
    # Synthetic releases keep these tests independent of upstream curation.
    state = SimpleNamespace(pinned="1.2.3", installed="1.2.3")
    monkeypatch.setattr(
        figure_environment, "pinned_oncoref_version", lambda: state.pinned,
    )
    monkeypatch.setattr(
        figure_environment.metadata, "version", lambda _name: state.installed,
    )
    monkeypatch.setitem(sys.modules, "oncoref", SimpleNamespace(__version__="1.2.3"))
    return state


def test_matching_environment_is_accepted(environment):
    assert figure_environment.check_figure_environment() == "1.2.3"


@pytest.mark.parametrize("source", ["metadata", "import"])
def test_mismatched_metadata_or_import_is_rejected(environment, source):
    if source == "metadata":
        environment.installed = "1.2.2"
    else:
        sys.modules["oncoref"].__version__ = "1.2.2"
    with pytest.raises(figure_environment.FigureEnvironmentError, match="mismatch"):
        figure_environment.check_figure_environment()


def test_missing_distribution_has_actionable_error(environment, monkeypatch):
    def missing(_name):
        raise metadata.PackageNotFoundError("oncoref")

    monkeypatch.setattr(figure_environment.metadata, "version", missing)
    with pytest.raises(
        figure_environment.FigureEnvironmentError, match="Cannot load oncoref==1.2.3",
    ):
        figure_environment.check_figure_environment()


def test_pin_change_cannot_switch_a_running_batch(environment):
    expected = figure_environment.check_figure_environment()
    environment.pinned = environment.installed = "1.2.4"
    sys.modules["oncoref"].__version__ = "1.2.4"
    with pytest.raises(figure_environment.FigureEnvironmentError, match="pin changed"):
        figure_environment.check_figure_environment(expected)
