#!/usr/bin/env python
"""Regenerate **all** analyses plots into ONE timestamped run dir, organised by
plot family in subfolders:

    analyses/outputs/run_<YYYYMMDD-HHMMSS>/
        apd1_causal_model/          aggregate aPD1 causal-factor figures
            cta_vs_apd1_by_exclusion/
        apd1_response/              ORR bars, ORR-vs-TMB
        cta_<metric>_vs_<axis>/     CTA burden vs TMB / aPD1 / incidence / mortality
        cta_addressable/  cta_covering_set/  cta_expression_heatmaps/
        ...

This replaces the old per-script ``outputs/apd1_causal_factors/run_*`` layout
(which buried five scripts' output under one script's name) — every family now
lands in the same timestamped run dir.

    python analyses/regenerate_plots.py

Scripts that fail are reported but don't abort the batch.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

# aPD1 causal-factor batch — driven by the APD1_RUN_DIR env var, all into one
# subfolder of the run (they already group cta_vs_apd1_by_exclusion/ themselves).
APD1_BATCH = ["exclusion_vs_apd1", "apd1_causal_factors", "apd1_mechanism_screen",
              "apd1_landscape", "apd1_exclusion_scatters"]

# argparse + _run_layout scripts. (script, extra_args, target):
#   target "groups" -> writes its own family subdirs straight under the run dir
#   target "<name>" -> a flat-writing script gets its own named subdir
LAYOUT = [
    ("cta_patient_counts", [], "groups"),
    ("cta_covering_set", [], "cta_covering_set"),
    ("cta_expression_heatmaps", [], "cta_expression_heatmaps"),
    ("apd1_response_plots", [], "apd1_response"),
    ("ici_landscape", [], "ici_landscape"),
    ("apd1_ici_factor_contributions", [], "apd1_ici_contributions"),
    ("suppressor_genes_vs_apd1", [], "suppressor_genes"),
    ("antigen_or_suppression_score", [], "antigen_or_suppression"),
]


def _run(cmd, env=None):
    return subprocess.run(cmd, cwd=HERE, env=env).returncode == 0


def main() -> int:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = OUTPUTS / f"run_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []

    print(f"regenerating all plots -> {run}\n")
    env = {**os.environ, "APD1_RUN_DIR": str(run / "apd1_causal_model")}
    for s in APD1_BATCH:
        print(f"  apd1_causal_model: {s} ...", flush=True)
        (ok if _run([sys.executable, f"{s}.py"], env=env) else failed).append(s)

    for s, extra, target in LAYOUT:
        print(f"  {target}: {s} ...", flush=True)
        if target == "groups":          # writes its own group subdirs under run/
            args = ["--out-dir", str(run), "--no-timestamp"]
        else:                           # flat-writing -> its own named subdir
            args = ["--out-dir", str(run), "--run-name", target]
        (ok if _run([sys.executable, f"{s}.py", *args, *extra]) else failed).append(s)

    # cta_addressable_burden consumes cta_patient_counts' tables (written above by
    # the "groups" entry into the run root) and has its own --run-dir/--fig-dir
    # CLI, so it can't ride the LAYOUT loop. Read tables from the run root, write
    # its plots into their own family subdir.
    print("  cta_addressable: cta_addressable_burden ...", flush=True)
    addr_ok = _run([sys.executable, "cta_addressable_burden.py",
                    "--run-dir", str(run), "--fig-dir", str(run / "cta_addressable")])
    (ok if addr_ok else failed).append("cta_addressable_burden")

    pngs = sorted(run.rglob("*.png"))
    print(f"\nrun -> {run}")
    print(f"  {len(pngs)} PNGs across {len({p.parent for p in pngs})} subfolders")
    print(f"  ok: {ok}")
    if failed:
        print(f"  FAILED: {failed}")
    for d in sorted({p.parent.relative_to(run) for p in pngs}):
        print(f"    {d}/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
