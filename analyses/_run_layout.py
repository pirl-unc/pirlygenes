"""Shared per-run output layout for the analyses/ plot scripts.

A run writes its plots + tables into a timestamped ``run_<YYYYMMDD-HHMMSS>/``
subfolder under the base ``outputs/`` dir, so a fresh run never overwrites or
mixes with an older one. Reusable caches go to a stable, gitignored
``outputs/_cache/``. This mirrors the layout in ``cta_patient_counts.py`` so
every analyses script behaves the same way.

Usage::

    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    CACHE, FIGDIR = resolve_dirs(args, Path(__file__).resolve().parent / "outputs")
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def add_layout_args(ap) -> None:
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory for plots/tables (default: analyses/outputs). "
                         "Only outputs move here; the _cache/ stays at the base.")
    ap.add_argument("--run-name", default=None,
                    help="per-run subfolder name (default: run_YYYYMMDD-HHMMSS)")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="write straight into the output dir (no per-run subfolder)")


def resolve_dirs(args, base: Path):
    """Return ``(CACHE, FIGDIR)``. CACHE = ``base/_cache`` (always stable, so
    reruns reuse caches even with ``--out-dir``); FIGDIR = the per-run snapshot
    dir for this run's plots + tables, redirectable via ``--out-dir``."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    cache = base / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fig_base = base if getattr(args, "out_dir", None) is None else args.out_dir.resolve()
    if getattr(args, "no_timestamp", False):
        figdir = fig_base
    else:
        run = (getattr(args, "run_name", None)
               or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        figdir = fig_base / run
    figdir.mkdir(parents=True, exist_ok=True)
    return cache, figdir


def latest_run_dir(base, must_contain="cta_patient_counts.csv"):
    """Newest ``run_<ts>/`` under ``base`` that contains ``must_contain`` (run
    folders sort chronologically by their timestamp name). Returns ``None`` if
    none exist — for scripts that consume another script's per-run output."""
    base = Path(base)
    runs = sorted(d for d in base.glob("run_*")
                  if d.is_dir() and (d / must_contain).exists())
    return runs[-1] if runs else None


def pct_axis(ax, which):
    """Format an axis' tick numbers as percentages (50 -> '50%'), for axes whose
    values are already a 0-100 percentage. ``which`` is 'x' or 'y'."""
    from matplotlib.ticker import PercentFormatter
    getattr(ax, f"{which}axis").set_major_formatter(PercentFormatter(xmax=100, decimals=0))
