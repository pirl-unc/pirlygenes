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
from pirlygenes.load_dataset import get_data  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs, pct_axis  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
FIGDIR = OUT   # per-run output dir; set in main() via _run_layout

# Documented aPD1-differential subtype pairs (immune-hot vs immune-cold).
_PAIRS = [("CRC_MSI", "CRC_MSS")]

# Colorectal pooling: KEYNOTE-177 (and the MSS series) report *colorectal*, so
# COAD_MSI+READ_MSI are one CRC_MSI point (identical ORR/TMB), not two — mirrors
# the causal-factors plots' CRC pooling so every aPD1 plot agrees.
_CRC_POOL = {"COAD_MSI": "CRC_MSI", "READ_MSI": "CRC_MSI",
             "COAD_MSS": "CRC_MSS", "READ_MSS": "CRC_MSS",
             "COAD": "CRC", "READ": "CRC"}


def _pool_crc(d: dict) -> dict:
    """Average the colorectal members into their CRC tier (the duplicated
    COAD/READ values are identical, so the mean is just that value); every other
    cancer code passes through unchanged."""
    groups: dict[str, list] = {}
    for code, val in d.items():
        groups.setdefault(_CRC_POOL.get(code, code), []).append(val)
    return {k: sum(v) / len(v) for k, v in groups.items()}


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


def _plot_tmb_vs_apd1(orr, tmb, colors, proxy=None, dual=None):
    proxy = proxy or set()
    dual = dual or set()
    pts = [(c, tmb[c], orr[c]) for c in orr if c in tmb]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xscale("log")
    for code, x, y in pts:
        # Marker encodes the evidence class (see drug_target):
        #   anti-PD-1 monotherapy      -> FILLED circle
        #   PD-L1 proxy (no aPD-1 ORR) -> OPEN circle, label suffix "*"
        #   dual ipi+nivo (no single-agent ORR) -> FILLED diamond, label suffix "+"
        fam_color = colors[_family(code)]
        is_proxy, is_dual = code in proxy, code in dual
        if is_proxy:
            ax.scatter(x, y, s=54, facecolors="none", edgecolors=fam_color,
                       linewidth=1.4, marker="o", zorder=3)
            suffix = "*"
        elif is_dual:
            ax.scatter(x, y, s=64, color=fam_color, edgecolor="black",
                       linewidth=0.9, marker="D", zorder=3)
            suffix = "+"
        else:
            ax.scatter(x, y, s=42, color=fam_color, edgecolor="white",
                       linewidth=0.4, marker="o", zorder=3)
            suffix = ""
        ax.annotate(f"{code}{suffix}", (x, y), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    seen = {c for c, _, _ in pts}
    handles = []
    if proxy & seen:
        handles.append(ax.scatter([], [], marker="o", facecolors="none",
                       edgecolors="0.4", linewidth=1.4,
                       label="* PD-L1 proxy (no anti-PD-1 monotherapy data)"))
    if dual & seen:
        handles.append(ax.scatter([], [], marker="D", facecolor="0.5",
                       edgecolor="black",
                       label="+ dual checkpoint, ipi+nivo (no single-agent data)"))
    if handles:
        ax.legend(loc="lower right", fontsize=8)
    # connect the subtype pairs with a thin line to show the differential
    for hi, lo in _PAIRS:
        if hi in tmb and hi in orr and lo in tmb and lo in orr:
            ax.plot([tmb[hi], tmb[lo]], [orr[hi], orr[lo]], color="0.5",
                    lw=0.8, ls="--", zorder=2)
    ax.set_xlabel("median tumor mutational burden (mut/Mb)")
    ax.set_ylabel("immune-checkpoint (ICI) objective response rate")
    pct_axis(ax, "y")
    ax.set_title("Tumor mutational burden vs immune-checkpoint (ICI) response, by cancer type")
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
    ax.set_xlabel("immune-checkpoint (ICI) objective response rate")
    pct_axis(ax, "x")
    ax.set_title("Immune-checkpoint (ICI) response rate by cancer type (curated)")
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
    orr_raw = gsc.cancer_apd1_response()       # {code: ORR%}
    tmb_raw = {c: gsc.cancer_tmb(c) for c in orr_raw
               if gsc.cancer_tmb(c) is not None}
    orr = _pool_crc(orr_raw)                   # COAD/READ MSI+MSS -> CRC tiers
    tmb = _pool_crc(tmb_raw)
    rdf = get_data("cancer-apd1-response")     # denoted fallback classes
    proxy = set(rdf.loc[rdf["drug_target"] == "PD-L1", "cancer_code"].astype(str))
    dual = set(rdf.loc[rdf["drug_target"] == "PD-1+CTLA-4", "cancer_code"].astype(str))
    colors = _color_map(orr)
    n = _plot_tmb_vs_apd1(orr, tmb, colors, proxy=proxy, dual=dual)
    _plot_orr_bars(orr, colors)
    print(f"wrote apd1_vs_tmb.png ({n} cancers with TMB+ORR) and "
          f"apd1_orr_bars.png ({len(orr)} cancers) -> {FIGDIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
