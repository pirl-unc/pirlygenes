"""Build publication overlaps/funnels and five-gene HPA profiles in one run.

    python analyses/placental_cta_followups.py
    python analyses/placental_cta_followups.py --out-dir analyses/outputs/run_<timestamp> --no-timestamp

Install pirlygenes[viz] for vector PDF compilation. The independent oncoref
Bradley audit can be regenerated with its scripts/plot_cta_bradley_audit.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _run_layout import add_layout_args, resolve_dirs
from regenerate_plots import _build_combined_pdf
from pirlygenes.cta_curation_plots import render as render_curation
from pirlygenes.placental_gene_profiles import render as render_profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_layout_args(parser)
    args = parser.parse_args()
    _, run = resolve_dirs(args, Path(__file__).resolve().parent / "outputs")
    render_curation(run / "cta_curation")
    render_profiles(run / "placental_profiles")
    render_curation(run / "large_font" / "cta_curation", font_scale=1.5,
                    kinds=("stage_funnel", "filter_funnel"))
    print(_build_combined_pdf(run))


if __name__ == "__main__":
    main()
