#!/usr/bin/env python
"""Regenerate the full aPD1 causal-factor figure batch into ONE timestamped run
directory (analyses/outputs/apd1_causal_factors/run_<ts>/) so batches don't mix.

    python analyses/regen_apd1_figures.py
"""
import datetime
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["exclusion_vs_apd1", "apd1_causal_factors",
           "apd1_mechanism_screen", "apd1_landscape",
           "apd1_exclusion_scatters"]


def main() -> int:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = HERE / "outputs" / "apd1_causal_factors" / f"run_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "APD1_RUN_DIR": str(run)}
    for s in SCRIPTS:
        print(f"  {s} ...", flush=True)
        subprocess.run(["python", f"{s}.py"], cwd=HERE, env=env, check=True,
                       stdout=subprocess.DEVNULL)
    print(f"\nbatch -> {run}")
    print("\n".join("  " + p.name for p in sorted(run.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
