"""Figure batches must reject dependency drift before producing more output."""

from importlib import metadata
import os
import sys
from types import SimpleNamespace

import pytest

from analyses import regen_apd1_figures, regenerate_plots
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


@pytest.mark.parametrize("runner", [regenerate_plots, regen_apd1_figures])
def test_batch_refuses_drift_before_creating_outputs(
    environment, runner, tmp_path, monkeypatch,
):
    environment.installed = "1.2.2"
    monkeypatch.setattr(runner, "HERE", tmp_path)
    if runner is regenerate_plots:
        monkeypatch.setattr(runner, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(sys, "argv", ["figure-runner"])
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("figure subprocess must not start"),
    )
    with pytest.raises(figure_environment.FigureEnvironmentError, match="mismatch"):
        runner.main()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("when", ["before", "during"])
def test_subprocess_boundary_rejects_drift(environment, monkeypatch, when):
    expected = figure_environment.check_figure_environment()
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        environment.installed = "1.2.2"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(regenerate_plots.subprocess, "run", run)
    if when == "before":
        environment.installed = "1.2.2"
    with pytest.raises(figure_environment.FigureEnvironmentError, match="mismatch"):
        regenerate_plots._run([sys.executable, "figure.py"], expected)
    assert len(calls) == (0 if when == "before" else 1)


def test_apd1_batch_keeps_selected_interpreter_and_checkout(
    environment, tmp_path, monkeypatch,
):
    monkeypatch.setattr(regen_apd1_figures, "HERE", tmp_path)
    monkeypatch.setattr(regen_apd1_figures, "SCRIPTS", ["figure"])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(regen_apd1_figures.subprocess, "run", run)
    assert regen_apd1_figures.main() == 0
    [(command, kwargs)] = calls
    assert command == [sys.executable, "figure.py"]
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(regen_apd1_figures.REPO)
