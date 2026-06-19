"""Batch-driver wrapper for the CTA curation figures.

The figure logic lives in the package (``pirlygenes.cta_curation_plots``) and is
also exposed as ``pirlygenes plot cta-curation``. This wrapper just rides the
shared analyses run layout so the figures land in the timestamped run dir
alongside every other family and participate in
``regenerate_plots.py --promote-docs``.

    python analyses/cta_curation_figures.py            # -> analyses/outputs/run_*/
    python analyses/cta_curation_figures.py --out-dir <run> --run-name cta_curation
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _run_layout import add_layout_args, resolve_dirs

from pirlygenes.cta_curation_plots import render


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, Path(__file__).resolve().parent / "outputs")
    result = render(out_dir=figdir)
    print(f"CTA curation figures from {result['n_genes']} evidence rows -> {figdir}")
    for kind, path in result["paths"].items():
        print(f"  wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
