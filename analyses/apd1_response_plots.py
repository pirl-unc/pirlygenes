"""Immune-checkpoint response-rate plots (curated cancer-apd1-response.csv).

Each figure is emitted in TWO variants:
  * ``_ici``        — full immune-checkpoint view: anti-PD-1 monotherapy
                      (filled circle) + anti-PD-L1 proxies (open circle, "*")
                      + dual ipi+nivo fallbacks (double circle, "+").
  * ``_strict_pd1`` — only true single-agent anti-PD-1 ORRs (PD-L1 proxies and
                      dual-checkpoint fallbacks dropped); all filled circles.

Figures:
  1. apd1_vs_tmb_{ici,strict_pd1}.png   — median TMB (log x) vs ORR (y), one
                              labelled point per cancer type, coloured by
                              lineage (cell of origin). The immune-logic quadrant: high
                              TMB → high ORR (melanoma, MSI-H), low TMB → low
                              ORR (PRAD, GBM, MSS CRC).
  2. apd1_orr_bars_{ici,strict_pd1}.png — ORR by cancer type, sorted, coloured
                              by lineage (cell of origin); the documented subtype PAIRS
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

try:                                    # optional: nicer non-overlapping labels
    from adjustText import adjust_text
except Exception:                       # pragma: no cover - fallback if missing
    adjust_text = None

from pirlygenes import gene_sets_cancer as gsc  # noqa: E402
from pirlygenes.load_dataset import get_data  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs, pct_axis  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
FIGDIR = OUT   # per-run output dir; set in main() via _run_layout

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
def _lineage_map() -> dict:
    """{code: coarse histogenesis group} over the registry. The registry
    ``family`` column is uneven (CNS split into 6, carcinoma by organ), so we
    colour by the ~8 cell-of-origin classes from ``cancer_lineage_group``
    (Epithelial / Sarcoma / Heme / CNS / Neuroendocrine / Melanoma /
    Germ cell / Embryonal) instead."""
    codes = gsc.cancer_type_registry()["code"].astype(str)
    return {c: (gsc.cancer_lineage_group(c) or "other") for c in codes}


def _lineage(code: str) -> str:
    m = _lineage_map()
    if code in m:
        return m[code]
    # synthetic pooled CRC tiers (CRC_MSI / CRC_MSS / CRC) aren't registry codes
    # -> fall back to their umbrella's group (CRC_MSI -> CRC -> Epithelial).
    return m.get(code.split("_")[0], "other")


@lru_cache(maxsize=1)
def _lineage_color_map() -> dict:
    """Deterministic lineage-group -> colour over the FULL set of groups, so the
    palette is STABLE across variants and there are few enough groups (~8) for a
    clean, collision-free ``tab10`` legend. ``other`` gets a neutral grey."""
    groups = sorted(set(_lineage_map().values()))
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
    colors.setdefault("other", (0.6, 0.6, 0.6, 1.0))
    return colors


def _lineage_handles(codes, colors):
    """Legend handles for the lineage groups actually present (stable colour)."""
    groups = sorted({_lineage(c) for c in codes})
    return [Line2D([], [], marker="o", linestyle="", markersize=7,
                   color=colors[g], label=g) for g in groups]


def _add_family_legend(ax, codes, colors):
    """Attach the lineage colour key just outside the axes on the right. Must be
    the LAST ``ax.legend`` call so it becomes the axes' primary legend (any
    marker-class legend is added earlier via ``add_artist``)."""
    return ax.legend(handles=_lineage_handles(codes, colors),
                     title="lineage (cell of origin)", loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), fontsize=8, title_fontsize=8,
                     handletextpad=0.5, borderaxespad=0.0, ncol=1)


def _plot_tmb_vs_apd1(orr, tmb, colors, proxy=None, dual=None, *,
                      fname="apd1_vs_tmb.png", kind="immune-checkpoint (ICI)"):
    proxy = proxy or set()
    dual = dual or set()
    pts = [(c, tmb[c], orr[c]) for c in orr if c in tmb]
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_xscale("log")
    texts = []
    for code, x, y in pts:
        # Marker encodes the evidence class (see drug_target):
        #   anti-PD-1 monotherapy      -> FILLED circle
        #   PD-L1 proxy (no aPD-1 ORR) -> OPEN circle, label suffix "*"
        #   dual ipi+nivo (no single-agent ORR) -> DOUBLE circle (filled + outer
        #       ring), label suffix "+"
        fam_color = colors[_lineage(code)]
        is_proxy, is_dual = code in proxy, code in dual
        if is_proxy:
            ax.scatter(x, y, s=54, facecolors="none", edgecolors=fam_color,
                       linewidth=1.4, marker="o", zorder=3)
            suffix = "*"
        elif is_dual:
            ax.scatter(x, y, s=40, color=fam_color, edgecolor="white",
                       linewidth=0.3, marker="o", zorder=4)
            ax.scatter(x, y, s=120, facecolors="none", edgecolors=fam_color,
                       linewidth=1.1, marker="o", zorder=3)   # outer ring
            suffix = "+"
        else:
            ax.scatter(x, y, s=42, color=fam_color, edgecolor="white",
                       linewidth=0.4, marker="o", zorder=3)
            suffix = ""
        texts.append(ax.text(x, y, f"{code}{suffix}", fontsize=6))
    # de-overlap the labels. Stronger repulsion + more iterations untangles the
    # dense low-TMB/low-ORR lower-left cluster; min_arrow_len drops the short
    # leader lines (which just add clutter when a label barely moved) and keeps
    # one only when a label is pulled far enough to need its anchor.
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, expand=(1.25, 1.6),
                    force_text=(0.5, 0.8), max_move=60, iter_lim=200,
                    min_arrow_len=12,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.3))
    else:
        for t in texts:
            t.set_position((t.get_position()[0], t.get_position()[1]))
            t.set_ha("left")

    ax.set_xlabel("median tumor mutational burden (mut/Mb)")
    ax.set_ylabel(f"{kind} objective response rate")
    pct_axis(ax, "y")
    ax.set_title(f"Tumor mutational burden vs {kind} response, by cancer type")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(-3, (max(orr.values()) * 1.08) if orr else 1.0)

    # Two legends. The evidence-class marker key is built FIRST and pinned with
    # add_artist; the lineage-family colour key is built LAST so it is the axes'
    # primary legend (otherwise the second ax.legend call would drop the first).
    seen = {c for c, _, _ in pts}
    mhandles, mlabels = [], []
    if proxy & seen:
        mhandles.append(Line2D([], [], marker="o", linestyle="", color="0.4",
                        markerfacecolor="none", markeredgewidth=1.4))
        mlabels.append("* PD-L1 proxy (no anti-PD-1 monotherapy data)")
    if dual & seen:
        # double circle = overlaid filled + open-ring handles (HandlerTuple)
        inner = Line2D([], [], marker="o", linestyle="", color="0.5", markersize=4)
        outer = Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                       markeredgecolor="0.4", markeredgewidth=1.2, markersize=11)
        mhandles.append((outer, inner))
        mlabels.append("+ dual checkpoint, ipi+nivo (no single-agent data)")
    if mhandles:
        ax.add_artist(ax.legend(mhandles, mlabels, loc="lower right", fontsize=8,
                      handler_map={tuple: HandlerTuple(ndivide=None)}))
    _add_family_legend(ax, [c for c, _, _ in pts], colors)

    fig.savefig(FIGDIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(pts)


def _plot_orr_bars(orr, colors, *, fname="apd1_orr_bars.png",
                   kind="immune-checkpoint (ICI)"):
    items = sorted(orr.items(), key=lambda kv: kv[1])
    codes = [c for c, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10.5, max(6, 0.32 * len(codes))))
    ax.barh(range(len(codes)), vals,
            color=[colors[_lineage(c)] for c in codes], edgecolor="white")
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=7)
    ax.set_xlabel(f"{kind} objective response rate")
    pct_axis(ax, "x")
    ax.set_title(f"{kind[0].upper()}{kind[1:]} response rate by cancer type (curated)")
    ax.grid(axis="x", alpha=0.3)
    _add_family_legend(ax, codes, colors)
    fig.savefig(FIGDIR / fname, dpi=300, bbox_inches="tight")
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
    colors = _lineage_color_map()              # stable ~8-group lineage palette

    # Variant 1 — full ICI: anti-PD-1 monotherapy + anti-PD-L1 proxies + dual
    # ipi+nivo, each drawn with its evidence-class marker.
    n_ici = _plot_tmb_vs_apd1(orr, tmb, colors, proxy=proxy, dual=dual,
                              fname="apd1_vs_tmb_ici.png",
                              kind="immune-checkpoint (ICI)")
    _plot_orr_bars(orr, colors, fname="apd1_orr_bars_ici.png",
                   kind="immune-checkpoint (ICI)")

    # Variant 2 — strict anti-PD-1 monotherapy: drop the PD-L1 proxies and the
    # dual-checkpoint fallbacks so every point is a true single-agent anti-PD-1
    # ORR (all filled circles; no fallback markers).
    keep = {c for c in orr if c not in proxy and c not in dual}
    orr_s = {c: orr[c] for c in keep}
    tmb_s = {c: tmb[c] for c in keep if c in tmb}
    n_strict = _plot_tmb_vs_apd1(orr_s, tmb_s, colors,
                                 fname="apd1_vs_tmb_strict_pd1.png",
                                 kind="anti-PD-1 monotherapy")
    _plot_orr_bars(orr_s, colors, fname="apd1_orr_bars_strict_pd1.png",
                   kind="anti-PD-1 monotherapy")

    print(f"wrote ICI variant (apd1_vs_tmb_ici.png, {n_ici} cancers; "
          f"apd1_orr_bars_ici.png, {len(orr)} cancers) and strict-aPD1 variant "
          f"(apd1_vs_tmb_strict_pd1.png, {n_strict} cancers; "
          f"apd1_orr_bars_strict_pd1.png, {len(orr_s)} cancers) -> {FIGDIR}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
