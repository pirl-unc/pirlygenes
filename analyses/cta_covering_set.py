"""Minimal covering set of cancer-testis antigens (CTAs).

Goal: find a small panel of CTAs such that *most cancer patients* have a
cancer type expressing at least one panel member at an actionable level
(median clean-TPM above a threshold). This is weighted set cover over the
cancer-type registry.

For each cancer code we take its most-gene-rich source (same selection the
CTA heatmaps use — comparable RNA-seq TPM, not microarray proxy). A CTA
"covers" a cancer code if its median clean-TPM there exceeds ACTIONABLE_TPM.
Greedy set cover then orders CTAs by how much *additional* coverage each adds,
under two weightings:

  * by cancer TYPE (each code counts once), and
  * by PATIENTS (codes weighted by approximate US annual incidence).

Outputs a markdown report + a coverage-curve PNG to analyses/outputs/.

    python analyses/cta_covering_set.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pirlygenes.expression.accessors as accessors
import pirlygenes.gene_sets_cancer as gsc
from cta_expression_heatmaps import _representative_source
from _run_layout import add_layout_args, resolve_dirs, pct_axis

ACTIONABLE_TPM = 30.0     # "actionable" target threshold (per user intuition)
SECONDARY_TPM = 10.0      # also report coverage at a looser bar
OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Approximate US annual incidence (thousands of cases), SEER/ACS ~2024, mapped
# to pirlygenes cancer codes. Used only to weight "patients covered" — rough by
# design (fine-grained subtypes are split estimates); codes absent here get a
# small default so rare cancers still register. The TYPE-weighted view needs
# none of this.
INCIDENCE_K = {
    "BRCA": 310, "PRAD": 300, "LUAD": 160, "LUSC": 65, "COAD": 110, "READ": 45,
    "SKCM": 100, "BLCA": 83, "KIRC": 62, "KIRP": 16, "KICH": 4, "UCEC": 67,
    "THCA": 44, "PAAD": 66, "LIHC": 41, "STAD": 27, "HNSC": 71, "ESCA": 22,
    "GBM": 14, "LGG": 6, "OV": 20, "CESC": 13, "DLBC": 28, "LAML": 20,
    "LAML_APL": 1, "LAML_ELN_Fav": 6, "LAML_ELN_Int": 7, "LAML_ELN_Adv": 6,
    "MM": 35, "MDS": 20, "MPN": 20, "CML": 9, "CLL": 21, "MCL": 4, "FL": 15,
    "HL": 8, "BL": 1.5, "MESO": 3, "THYM": 1.5, "ACC": 0.2, "PCPG": 0.8,
    "UVM": 1.5, "UCS": 3, "TGCT": 9, "CHOL": 8, "B_ALL": 5, "T_ALL": 1.5,
    "CTCL": 3, "MTC": 1.5, "ADCC": 1.5, "SARC_CHON": 0.6, "SCLC": 30,
    "SCLC_POU2F3": 4, "SCLC_ASCL1": 12, "SCLC_NEUROD1": 7,
    "NBL_MYCN_amp": 0.2, "NBL_MYCN_nonamp": 0.6, "WILMS": 0.65, "RT": 0.05,
    "SARC_OS": 1, "SARC_EWS": 0.2, "MBL": 0.5, "RB": 0.3, "ATRT": 0.1, "HEPB": 0.1,
    "PANNET": 4, "MID_NET": 4, "REC_NET": 2, "LUNG_NET_LC": 4, "LUNG_NET_LCNEC": 2,
}
DEFAULT_INCIDENCE_K = 0.5   # rare/sub-typed codes not in the table


def cta_matrices() -> dict[str, pd.DataFrame]:
    """code × CTA-symbol matrices of clean-TPM, one (most-gene-rich) source
    per code, at two statistics:

      * ``median`` — the typical (≥50%) patient expresses it, and
      * ``q3``     — the upper quartile (≥25% of patients) express it.

    CTAs are *subset antigens* (heterogeneously expressed within a tumor
    type), so q3 is the clinically relevant bar for "a meaningful fraction
    of patients have this target" — median alone hides them.
    """
    df = accessors.cancer_reference_expression()
    rep = _representative_source(df)
    keep = set(zip(rep["cancer_code"], rep["source_cohort"]))
    cta_ids = set(gsc.CTA_gene_ids())
    sub = df[df["Ensembl_Gene_ID"].isin(cta_ids)]
    sub = sub[[(c, s) in keep for c, s in zip(sub["cancer_code"], sub["source_cohort"])]]
    return {
        "median (≥50% of patients)": sub.pivot_table(
            index="cancer_code", columns="Symbol", values="expression", aggfunc="max"
        ),
        "q3 (≥25% of patients)": sub.pivot_table(
            index="cancer_code", columns="Symbol", values="q3", aggfunc="max"
        ),
    }


def greedy_cover(mat: pd.DataFrame, threshold: float, weights: pd.Series):
    """Greedy weighted set cover. Returns a list of
    (cta, newly_covered_codes, cumulative_weight, cumulative_count)."""
    hits = mat > threshold                      # code × CTA boolean
    coverable = hits.index[hits.any(axis=1)]
    remaining = set(coverable)
    order = []
    while remaining:
        # weight each CTA by the incidence of the still-uncovered codes it hits
        best_cta, best_codes, best_gain = None, set(), -1.0
        for cta in hits.columns:
            covered = {c for c in hits.index[hits[cta]] if c in remaining}
            gain = float(weights.reindex(covered).fillna(DEFAULT_INCIDENCE_K).sum())
            if gain > best_gain:
                best_cta, best_codes, best_gain = cta, covered, gain
        if best_gain <= 0:
            break
        remaining -= best_codes
        covered_so_far = set(coverable) - remaining
        order.append((
            best_cta,
            sorted(best_codes),
            float(weights.reindex(covered_so_far).fillna(DEFAULT_INCIDENCE_K).sum()),
            len(covered_so_far),
        ))
    return order, coverable


def _weights_for(codes) -> pd.Series:
    return pd.Series(
        {c: INCIDENCE_K.get(c, DEFAULT_INCIDENCE_K) for c in codes}, dtype=float
    )


def _section(mat: pd.DataFrame, stat_label: str, codes, type_w, pat_w) -> list[str]:
    lines = [f"\n# Statistic: {stat_label}"]
    for wlabel, weights, total in [
        ("by cancer TYPE (each code = 1)", type_w, float(len(codes))),
        ("by PATIENTS (≈US incidence)", pat_w, float(pat_w.sum())),
    ]:
        order, coverable = greedy_cover(mat, ACTIONABLE_TPM, weights)
        cover_total = float(weights.reindex(coverable).fillna(DEFAULT_INCIDENCE_K).sum())
        lines += [
            f"\n## Covering set {wlabel}",
            f"\nCoverable: {len(coverable)}/{len(codes)} codes "
            f"({100*cover_total/total:.0f}% of weight). "
            f"Full cover needs {len(order)} CTAs.",
            "",
            "| # | CTA | cum. coverage | newly covered codes |",
            "| ---: | --- | ---: | --- |",
        ]
        for i, (cta, new_codes, cum_w, cum_n) in enumerate(order, 1):
            shown = ", ".join(new_codes[:6]) + ("…" if len(new_codes) > 6 else "")
            lines.append(
                f"| {i} | {cta} | {100*cum_w/total:.0f}% ({cum_n} codes) | {shown} |"
            )
        miles = []
        for frac in (0.5, 0.8, 0.9):
            for i, (_c, _n, cum_w, _cn) in enumerate(order, 1):
                if cum_w / total >= frac:
                    miles.append(f"{int(frac*100)}% → {i} CTAs")
                    break
        lines.append("\n**Milestones:** " + ("; ".join(miles) or "target not reached"))
        if wlabel.startswith("by PATIENTS"):
            uncov = sorted(set(codes) - set(coverable))
            lines.append(
                f"\n**No actionable CTA (>{ACTIONABLE_TPM:g} TPM at this "
                f"statistic):** " + (", ".join(uncov) if uncov else "none") + "."
            )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    add_layout_args(ap)
    args = ap.parse_args()
    _, figdir = resolve_dirs(args, OUT_DIR)
    mats = cta_matrices()
    codes = list(next(iter(mats.values())).index)
    type_w = pd.Series(1.0, index=codes)              # each cancer type = 1
    pat_w = _weights_for(codes)                        # incidence-weighted

    lines = [
        "# Covering set of CTAs — actionable target for most cancers",
        "",
        f"For each cancer code: its most-gene-rich source; a CTA covers the code "
        f"if clean-TPM > {ACTIONABLE_TPM:g} at the given statistic. Greedy set "
        f"cover. CTAs are subset antigens, so the **q3** view (target in ≥25% of "
        f"patients) is the clinically relevant one — median understates them.",
        f"\n{len(codes)} cancer codes with CTA data.",
    ]
    for stat_label, mat in mats.items():
        lines += _section(mat, stat_label, codes, type_w, pat_w)
    (figdir / "cta_covering_set.md").write_text("\n".join(lines) + "\n", "utf-8")

    # coverage curve: median vs q3, patient-weighted
    fig, ax = plt.subplots(figsize=(9, 6))
    total = float(pat_w.sum())
    for stat_label, mat in mats.items():
        order, _ = greedy_cover(mat, ACTIONABLE_TPM, pat_w)
        xs = list(range(1, len(order) + 1))
        ys = [100 * cum_w / total for (_c, _n, cum_w, _cn) in order]
        ax.plot(xs, ys, "o-", ms=4, label=f"{stat_label}")
    ax.axhline(80, color="0.7", ls="--", lw=0.8)
    ax.set_xlabel(f"# CTAs in panel (> {ACTIONABLE_TPM:g} TPM)")
    ax.set_ylabel("patients covered (incidence-weighted)")
    pct_axis(ax, "y")
    ax.set_title("CTA covering set — cumulative patient coverage")
    ax.set_xlim(left=1)
    ax.legend(title="actionable if TPM> bar at:")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "cta_covering_set.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote cta_covering_set.md + .png to {figdir}")


if __name__ == "__main__":
    main()
