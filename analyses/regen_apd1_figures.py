#!/usr/bin/env python
"""Regenerate the full aPD1 causal-factor figure batch into ONE timestamped run
directory (analyses/outputs/apd1_causal_factors/run_<ts>/) so batches don't mix.

    .venv-figures/bin/python analyses/regen_apd1_figures.py
"""
import datetime
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.figure_environment import (  # noqa: E402
    FigureEnvironmentError,
    check_figure_environment,
)

SCRIPTS = ["exclusion_vs_apd1", "apd1_causal_factors",
           "apd1_mechanism_screen", "apd1_landscape",
           "apd1_exclusion_scatters"]


def main() -> int:
    expected_oncoref = check_figure_environment()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = HERE / "outputs" / "apd1_causal_factors" / f"run_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "APD1_RUN_DIR": str(run),
        "PYTHONPATH": os.pathsep.join(
            [str(REPO)] + ([os.environ["PYTHONPATH"]]
                           if os.environ.get("PYTHONPATH") else [])),
    }
    for s in SCRIPTS:
        check_figure_environment(expected_oncoref)
        print(f"  {s} ...", flush=True)
        subprocess.run([sys.executable, f"{s}.py"], cwd=HERE, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        check_figure_environment(expected_oncoref)
    print(f"\nbatch -> {run}")
    print("\n".join("  " + p.name for p in sorted(run.glob("*.png"))))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FigureEnvironmentError as exc:
        print(f"Figure batch aborted: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
