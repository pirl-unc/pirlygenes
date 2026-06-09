"""Anti-PD-1 response-rate plots (curated cancer-apd1-response.csv).

Two figures (per the new aPD1 plotting axis):
  1. apd1_vs_tmb.png        — median TMB (log x) vs aPD1 monotherapy ORR (y),
                              one labelled point per cancer type, coloured by
                              lineage family. The immune-logic quadrant: high
                              TMB → high ORR (melanoma, MSI-H), low TMB → low
                              ORR (PRAD, GBM, MSS CRC).
  2. apd1_orr_bars.png      — aPD1 ORR by cancer type, sorted, coloured by
                              lineage; the documented subtype PAIRS
                              (MSI-H vs MSS CRC) are drawn adjacent so the
                              differential is visible.

Pure curation (no expression data needed): cancer_apd1_response() +
cancer_tmb(). For CTA-burden-vs-aPD1, the cta_patient_counts pipeline carries
the per-sample CTA load — pass aPD1 as the axis there.

Run:  python analyses/apd1_response_plots.py
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pirlygenes import gene_sets_cancer as gsc  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
FIGDIR = OUT   # per-run output dir; set in main() via _run_layout

# Documented aPD1-differential subtype pairs (immune-hot vs immune-cold).
_PAIRS = [("COAD_MSI", "COAD_MSS"), ("READ_MSI", "READ_MSS")]


@lru_cache(maxsize=1)
def _family_map() -> dict:
    """{code: family} over the cancer-type registry, read + indexed once."""
    fam = gsc.cancer_type_registry().set_index("code")["family"]
    return {str(code): str(family) for code, family in fam.items()}


def _family(code: str) -> str:
    return _family_map().get(code, "other")


def _color_map(codes):
    fams = sorted({_family(c) for c in codes})
    cmap = plt.get_cmap("tab20")
    return {f: cmap(i % 20) for i, f in enumerate(fams)}


def _plot_tmb_vs_apd1(orr, tmb, colors):
    pts = [(c, tmb[c], orr[c]) for c in orr if c in tmb]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xscale("log")
    for code, x, y in pts:
        ax.scatter(x, y, s=42, color=colors[_family(code)], alpha=0.9,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.annotate(code, (x, y), fontsize=6.5, xytext=(3, 3),
                    textcoords="offset points")
    # connect the subtype pairs with a thin line to show the differential
    for hi, lo in _PAIRS:
        if hi in tmb and hi in orr and lo in tmb and lo in orr:
            ax.plot([tmb[hi], tmb[lo]], [orr[hi], orr[lo]], color="0.5",
                    lw=0.8, ls="--", zorder=2)
    ax.set_xlabel("median tumor mutational burden (mut/Mb, log scale)")
    ax.set_ylabel("anti-PD-1 monotherapy ORR (%)")
    ax.set_title("Tumor mutational burden vs anti-PD-1 response, by cancer type")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-3, (max(orr.values()) * 1.08) if orr else 1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / "apd1_vs_tmb.png", dpi=300)
    plt.close(fig)
    return len(pts)


def _plot_orr_bars(orr, colors):
    items = sorted(orr.items(), key=lambda kv: kv[1])
    codes = [c for c, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(codes))))
    ax.barh(range(len(codes)), vals,
            color=[colors[_family(c)] for c in codes], edgecolor="white")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=7)
    ax.set_xlabel("anti-PD-1 monotherapy ORR (%)")
    ax.set_title("Anti-PD-1 response rate by cancer type (curated)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "apd1_orr_bars.png", dpi=300)
    plt.close(fig)


def main() -> int:
    global FIGDIR
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, FIGDIR = resolve_dirs(args, OUT)
    orr = gsc.cancer_apd1_response()           # {code: ORR%}
    tmb = {c: gsc.cancer_tmb(c) for c in orr if gsc.cancer_tmb(c) is not None}
    colors = _color_map(orr)
    n = _plot_tmb_vs_apd1(orr, tmb, colors)
    _plot_orr_bars(orr, colors)
    print(f"wrote apd1_vs_tmb.png ({n} cancers with TMB+ORR) and "
          f"apd1_orr_bars.png ({len(orr)} cancers) -> {FIGDIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
