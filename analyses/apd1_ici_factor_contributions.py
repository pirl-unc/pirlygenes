"""How each causal factor associates with anti-PD-1 (strict monotherapy) and broad
ICI objective response across cancer types: antigen/response DRIVERS (median TMB,
CTA coverage/load/9-mer payload, viral status) vs secreted SUPPRESSORS (TGFB1,
WNT11, WNT5A, IL10, and the curated TGF-beta / Wnt pathway signatures).

Two grouped Spearman-rho bars per factor — one for the strict anti-PD-1 ORR axis,
one for the broad ICI ORR axis — so you can read each factor's predictive sign and
strength against the two response definitions side by side. Drivers are expected
positive, suppressors negative.

Run:  python analyses/apd1_ici_factor_contributions.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pirlygenes import gene_sets_cancer as gsc  # noqa: E402
from pirlygenes.gene_sets_cancer import cancer_type_registry  # noqa: E402
from _apd1_factors import (apd1_map, tmb_map, viral_score, cta_metric_table,  # noqa: E402
                           cohort_gene_matrix, curated_signatures,
                           available_cta_metric_columns, CTA_FACTOR_METRICS)
from _apd1_factors import zscore, signature_score  # noqa: E402
from _panels import fold  # noqa: E402
from _run_layout import add_layout_args, resolve_dirs  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"

_TGFB = "aPD1_exclusion_TGFb_response"
_WNT = "aPD1_exclusion_Wnt"
# Individual secreted immune-EXCLUSION genes shown as their own factors.
_SECRETED = ["TGFB1", "WNT11", "WNT5A", "IL10"]
_MIN_RHO_N = 4


def _orr_maps():
    """(strict anti-PD-1 ORR, broad ICI ORR) cohort->% maps. Strict drops the
    PD-L1 proxies and PD-1+CTLA-4 dual fallbacks (mirrors apd1_response_plots /
    the cta_*_vs_apd1 axis)."""
    ici = apd1_map()
    rdf = gsc.cancer_apd1_response_df()
    drop = set(rdf.loc[rdf["drug_target"].isin(["PD-L1", "PD-1+CTLA-4"]),
                       "cancer_code"].astype(str))
    strict = {c: v for c, v in ici.items() if c not in drop}
    return strict, ici


def _factor_table():
    """Per-cohort factor values indexed by cancer code (codes with an ICI ORR)."""
    ici = apd1_map()
    mat = cohort_gene_matrix(list(ici))
    mat = mat.loc[[c for c in mat.index if c in ici]]
    reg = cancer_type_registry().set_index("code")
    tmb = tmb_map()
    cta = cta_metric_table()
    sig = curated_signatures()
    cols = {
        "median TMB": np.log10(pd.Series({c: tmb.get(c, np.nan) for c in mat.index})),
        "viral status": pd.Series({c: viral_score(c, reg) for c in mat.index}),
        "TGF-beta signature": signature_score(mat, sig.get(_TGFB, [])),
        "Wnt signature": signature_score(mat, sig.get(_WNT, [])),
    }
    cta_cols, skipped_cta = available_cta_metric_columns(
        cta, mat.index, min_n=_MIN_RHO_N)
    cols.update(cta_cols)
    for g in fold(_SECRETED):
        if g in mat.columns:
            cols[g] = zscore(mat[g])
    df = pd.DataFrame(cols)
    df.attrs["skipped_cta_factors"] = skipped_cta
    return df


# DRIVERS first (expected +), then SUPPRESSORS (expected -); plotted top->bottom.
_CTA_DRIVERS = [
    label for label, _source_col in CTA_FACTOR_METRICS
]
_DRIVERS = ["median TMB", *_CTA_DRIVERS, "viral status"]
_SUPPRESSORS = ["TGF-beta signature", "Wnt signature"] + _SECRETED


def _rho(factor: pd.Series, orr_map: dict) -> float:
    orr = pd.Series({c: orr_map.get(c, np.nan) for c in factor.index})
    ok = factor.notna() & orr.notna()
    if ok.sum() < _MIN_RHO_N:
        return np.nan
    return float(spearmanr(factor[ok], orr[ok]).statistic)


def _driver_title(factors: list[str]) -> str:
    cta_labels = [f.replace("CTA ", "") for f in _CTA_DRIVERS if f in factors]
    cta_text = f", CTA {'/'.join(cta_labels)}" if cta_labels else ""
    return f"drivers: TMB{cta_text}, viral"


def main() -> int:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT)

    df = _factor_table()
    strict, ici = _orr_maps()
    factors = [f for f in (_DRIVERS + _SUPPRESSORS) if f in df.columns]

    rows = [{"factor": f,
             "rho_apd1": _rho(df[f], strict),
             "rho_ici": _rho(df[f], ici),
             "group": "driver" if f in _DRIVERS else "suppressor"}
            for f in factors]
    res = pd.DataFrame(rows).set_index("factor")
    res.to_csv(figdir / "apd1_ici_factor_contributions.csv")

    order = list(res.index[::-1])           # barh: first factor on top
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.62 * len(order))))
    ax.barh(y + h / 2, res.loc[order, "rho_apd1"], height=h,
            color="#1b6ca8", label="anti-PD-1 monotherapy ORR")
    ax.barh(y - h / 2, res.loc[order, "rho_ici"], height=h,
            color="#f6a21e", label="broad ICI ORR")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)   # factor names are already display-ready
    # driver | suppressor divider
    n_supp = sum(1 for f in order if res.loc[f, "group"] == "suppressor")
    if 0 < n_supp < len(order):
        ax.axhline(n_supp - 0.5, color="black", lw=1.2, ls="--")
        ax.text(0.99, (n_supp - 0.5) / len(order),
                "drivers (expect +)", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color="#1b6ca8", fontweight="bold")
        ax.text(0.99, (n_supp - 0.5) / len(order),
                "suppressors (expect −)", transform=ax.transAxes, ha="right",
                va="top", fontsize=8, color="#a83232", fontweight="bold")
    ax.set_xlabel("Spearman ρ vs objective response rate (across cancer types)")
    ax.set_title("Causal-factor association with anti-PD-1 vs broad-ICI response\n"
                 f"{_driver_title(factors)}; suppressors: TGFβ/Wnt/secreted genes\n"
                 f"n={len(strict)} aPD1 / {len(ici)} ICI cohorts", fontsize=9.5)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "apd1_ici_factor_contributions.png", dpi=300)
    plt.close(fig)
    print(f"wrote apd1_ici_factor_contributions.png + .csv -> {figdir}", flush=True)
    if df.attrs.get("skipped_cta_factors"):
        skipped = ", ".join(df.attrs["skipped_cta_factors"])
        print(f"skipped unavailable CTA factors: {skipped}", flush=True)
    print(res.round(2).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
