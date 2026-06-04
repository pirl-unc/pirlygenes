"""Per-patient CTA expression: recurrence, breadth, and co-occurrence coverage.

The cohort-summary table only stores per-cohort percentiles, so it cannot
answer "how many *patients* express CTA X above 50 TPM" or "as I add CTAs,
how many *new* patients do I pick up". Those need per-sample values.

This script pulls the per-sample CTA TPM matrix straight from the cached
Treehouse 25.01 PolyA compendium (one uniform RSEM log2(TPM+1) pipeline, so
cohorts are mutually comparable), reusing the *exact* builder cohort filters
(`pirlygenes.builders.treehouse._filter_samples`) so the patient groups match
our registry cohorts (incl. the TCGA-subset codes and the GBM/LGG glioma
split). It then emits four artifacts:

  1. cta_patient_counts.csv         — for every CTA × cohort, the number and %
                                      of patients above 25 / 50 / 100 / 200 TPM.
  2. cta_patient_count_heatmap.png  — patients/cohort expressing each CTA above
                                      a chosen threshold (recurrence × breadth).
  3. cta_pct_bar_<COHORT>.png       — % of a cohort expressing each CTA, sorted
                                      by patient count (default GBM).
  4. cta_coverage_<COHORT>.png      — greedy co-occurrence-aware coverage: as
                                      CTAs are added, the cumulative fraction of
                                      *distinct* patients with >=1 CTA over
                                      threshold (a patient expressing several
                                      CTAs is counted once).

Run:  python analyses/cta_patient_counts.py [--threshold 25] [--cohort GBM]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pirlygenes import gene_sets_cancer as gsc
from pirlygenes import data_inventory as di
from pirlygenes.builders.treehouse import _filter_samples, TreehouseCohort

import sweep_treehouse_polya_cohorts as polya
import sweep_treehouse_tcga_cohorts as tcga

CACHE = Path.home() / ".cache" / "pirlygenes" / "expression" / "treehouse-polya-25-01"
TPM_TSV = CACHE / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
CLINICAL = CACHE / "clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv"
GLIOMA_MAP = CACHE / "derived" / "tcga_glioma_case_to_project.csv"
OUT = Path(__file__).resolve().parent / "outputs"
THRESHOLDS = [25, 50, 100, 200]


def _glioma_split_cohorts() -> list[TreehouseCohort]:
    """GBM/LGG cohorts from the cached TCGA glioma case->project map."""
    if not GLIOMA_MAP.exists():
        return []
    m = pd.read_csv(GLIOMA_MAP)
    case_to_project = dict(zip(m["submitter_id"], m["project_id"]))

    def _pred_for(project_id):
        def _pred(row):
            dsid = str(row.get("th_dataset_id", ""))
            if not dsid.startswith("TCGA"):
                return False
            return case_to_project.get("-".join(dsid.split("-")[:3])) == project_id
        return _pred

    return [
        TreehouseCohort("GBM", "glioma", sample_predicate=_pred_for("TCGA-GBM")),
        TreehouseCohort("LGG", "glioma", sample_predicate=_pred_for("TCGA-LGG")),
    ]


_DERIVED_DIR = CACHE / "derived"


def _tcga_case_subtype_cohorts(cache_csv, disease_label, key_col, code_col,
                               recode=None):
    """Build derived sub-cohorts from a cached cBioPortal patientId->code map:
    one TreehouseCohort per distinct code, matching TCGA samples whose case id
    maps to that code (same logic the sweep builders use). ``recode`` maps the
    cached label to the canonical registry code."""
    if not cache_csv.exists():
        return []
    recode = recode or {}
    m = pd.read_csv(cache_csv)
    out = []
    for raw in sorted(m[code_col].dropna().unique()):
        code = recode.get(str(raw), str(raw))
        cases = set(m.loc[m[code_col] == raw, key_col].astype(str))

        def _pred(row, cases=cases):
            dsid = str(row.get("th_dataset_id", ""))
            if not dsid.startswith("TCGA"):
                return False
            return "-".join(dsid.split("-")[:3]) in cases

        out.append(TreehouseCohort(str(code), disease_label, sample_predicate=_pred))
    return out


def _cbioportal_derived_cohorts() -> list[TreehouseCohort]:
    """BRCA PAM50 (5) + HNSC HPV (2) sub-cohorts from the cached cBioPortal
    maps — the per-sample-reconstructable sub-divisions of TCGA parents."""
    return (
        _tcga_case_subtype_cohorts(
            _DERIVED_DIR / "cbioportal_brca_pam50.csv",
            "breast invasive carcinoma", "patientId", "pam50",
            recode={"BRCA_Her2": "BRCA_HER2"})
        + _tcga_case_subtype_cohorts(
            _DERIVED_DIR / "cbioportal_hnsc_hpv.csv",
            "head & neck squamous cell carcinoma", "patientId", "hpv_subtype",
            recode={"HNSC_HPV-": "HNSC_HPV_neg", "HNSC_HPV+": "HNSC_HPV_pos"})
    )


def build_cohorts() -> list[TreehouseCohort]:
    """Pediatric/sarcoma + TCGA-subset + GBM/LGG split + cBioPortal sub-cohorts
    (BRCA PAM50, HNSC HPV) — parents AND their sub-divisions, all via the
    builders' own per-sample filters."""
    return (
        list(polya.COHORTS) + list(tcga.COHORTS)
        + _glioma_split_cohorts() + _cbioportal_derived_cohorts()
    )


def cohort_samples(clinical: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in build_cohorts():
        ids = [s for s in _filter_samples(clinical, c) if s]
        if ids:
            out[c.cancer_code] = ids
    return out


def extract_cta_matrix(symbols: dict[str, str]) -> pd.DataFrame:
    """Extract only the CTA rows from the wide compendium via awk, then load.

    Returns a CTA(ENSG)-indexed, sample-columned TPM frame."""
    sym_to_ensg = {v: k for k, v in symbols.items()}
    symfile = OUT / "_cta_symbols.txt"
    symfile.write_text("\n".join(sym_to_ensg) + "\n")
    small = OUT / "_cta_rows.tsv"
    # Reuse the cached extraction if it's newer than the source matrix — the
    # awk pass over the 6.9 GB compendium is the slow step (skip it on re-runs).
    if small.exists() and small.stat().st_mtime >= TPM_TSV.stat().st_mtime:
        print("      reusing cached _cta_rows.tsv (skip awk)", flush=True)
    else:
        # awk: keep header (NR==1) + rows whose first column is a CTA symbol.
        awk = (
            'BEGIN{while((getline l < "%s")>0) S[l]=1}'
            "NR==1||($1 in S)" % symfile
        )
        with small.open("w") as fh:
            subprocess.run(["awk", "-F\t", awk, str(TPM_TSV)], stdout=fh, check=True)
    raw = pd.read_csv(small, sep="\t")
    raw = raw.rename(columns={raw.columns[0]: "Symbol"})
    raw = raw[raw["Symbol"].isin(sym_to_ensg)].set_index("Symbol")
    # inverse log2(TPM+1) -> TPM, clamp tiny negatives
    tpm = np.power(2.0, raw.to_numpy(dtype=float)) - 1.0
    tpm = np.clip(tpm, 0.0, None)
    out = pd.DataFrame(tpm, index=raw.index, columns=raw.columns)
    out.index = [sym_to_ensg[s] for s in out.index]  # -> ENSG
    out.index.name = "Ensembl_Gene_ID"
    return out


def per_cohort_counts(mat, cohorts, ensg_to_sym):
    """Long table: CTA × cohort × threshold -> n patients, pct."""
    rows = []
    for code, samples in cohorts.items():
        cols = [s for s in samples if s in mat.columns]
        if not cols:
            continue
        sub = mat[cols]
        n = len(cols)
        for ensg in mat.index:
            vals = sub.loc[ensg].to_numpy()
            rec = {
                "cancer_code": code, "n_samples": n,
                "Ensembl_Gene_ID": ensg, "Symbol": ensg_to_sym.get(ensg, ensg),
            }
            any_hit = False
            for t in THRESHOLDS:
                k = int((vals > t).sum())
                rec[f"n_gt{t}"] = k
                rec[f"pct_gt{t}"] = round(100 * k / n, 2)
                any_hit = any_hit or k > 0
            if any_hit:
                rows.append(rec)
    return pd.DataFrame(rows)


def greedy_coverage(mat, samples, threshold):
    """Co-occurrence-aware: greedily order CTAs by marginal NEW patients (>thr).
    Returns (ordered_symbols_idx, cumulative_fraction, n_total)."""
    cols = [s for s in samples if s in mat.columns]
    n = len(cols)
    hit = (mat[cols].to_numpy() > threshold)  # CTA × patient boolean
    covered = np.zeros(n, dtype=bool)
    order, cum = [], []
    remaining = set(range(mat.shape[0]))
    while remaining:
        # marginal new coverage per remaining CTA
        best, best_gain = None, -1
        for i in remaining:
            gain = int((hit[i] & ~covered).sum())
            if gain > best_gain:
                best, best_gain = i, gain
        if best_gain <= 0:
            break
        covered |= hit[best]
        order.append(best)
        cum.append(covered.sum() / n)
        remaining.discard(best)
    return order, cum, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=25,
                    help="TPM cutoff for the heatmap / bar / coverage plots")
    ap.add_argument("--cohort", default="GBM",
                    help="cohort for the %%-bar and coverage plots")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] ENSG->Symbol for 258 CTAs", flush=True)
    ctas = set(gsc.CTA_gene_ids())
    ensg_to_sym = {}
    import glob, os
    for s in glob.glob(os.path.join(str(di._BUNDLED_REFERENCE_DIR), "*.csv.gz")):
        df = pd.read_csv(s, usecols=["Ensembl_Gene_ID", "Symbol"])
        for e, sym in zip(df.Ensembl_Gene_ID, df.Symbol):
            if e in ctas and e not in ensg_to_sym and isinstance(sym, str):
                ensg_to_sym[e] = sym
        if len(ensg_to_sym) >= len(ctas):
            break

    print("[2/5] cohort sample buckets (builder filters)", flush=True)
    clinical = pd.read_csv(CLINICAL, sep="\t", dtype=str)
    cohorts = cohort_samples(clinical)
    print(f"      {len(cohorts)} cohorts, "
          f"{sum(len(v) for v in cohorts.values())} samples", flush=True)

    print("[3/5] extract CTA per-sample TPM matrix (awk slice + log2 inverse)", flush=True)
    mat = extract_cta_matrix(ensg_to_sym)
    print(f"      matrix {mat.shape[0]} CTAs × {mat.shape[1]} samples", flush=True)

    print("[4/5] per (CTA × cohort) threshold counts -> CSV", flush=True)
    counts = per_cohort_counts(mat, cohorts, ensg_to_sym)
    counts = counts.sort_values(["cancer_code", "n_gt25"], ascending=[True, False])
    counts.to_csv(OUT / "cta_patient_counts.csv", index=False)

    print("[5/6] plots", flush=True)
    _plots(mat, cohorts, counts, ensg_to_sym, args.threshold, args.cohort)

    print("[6/7] per-cohort coverage curves (one PNG each)", flush=True)
    _coverage_every_cohort(mat, cohorts, ensg_to_sym, args.threshold)

    print("[7/7] peak-coverage bar charts (parent/child + top-CTA colored)", flush=True)
    _peak_coverage_bars(mat, cohorts, ensg_to_sym, args.threshold)
    print(f"done -> {OUT}", flush=True)


# CTA family prefixes, longest/most-specific first so e.g. MAGEA wins over a
# bare "MAGE" and SPANXN over SPANX.
_FAMILY_PREFIXES = [
    "MAGEA", "MAGEB", "MAGEC", "XAGE", "GAGE", "PAGE", "SPANXN", "SPANX",
    "SSX", "CTAG", "CT45", "CT47", "CT83", "DAZ", "SYCP", "SYCE", "SPATA",
    "CSAG", "TSPY", "PASD", "HORMAD", "PRAME", "NUTM", "GAGE12", "LUZP4",
    "ACTL", "DPPA", "DPEP", "TKTL", "PHF", "SPO", "MEIOC", "SYCN",
]
_FAMILY_LABEL = {"MAGEA": "MAGE-A", "MAGEB": "MAGE-B", "MAGEC": "MAGE-C"}


def _cta_family(symbol: str) -> str:
    """Group a CTA symbol into its antigen family (MAGE-A, GAGE, XAGE, SSX,
    CTAG, SPANX, PAGE, DAZ, …); singletons (PRAME, NUTM1) map to themselves."""
    s = symbol.upper()
    for p in _FAMILY_PREFIXES:
        if s.startswith(p):
            return _FAMILY_LABEL.get(p, p)
    stem = re.sub(r"[0-9].*$", "", s)
    return stem or s


def _shade(rgb, k):
    """Lighten (k>0) / darken (k<0) an RGB color, keeping its hue — used to
    distinguish members within one family."""
    import colorsys
    r, g, b = rgb[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1.0, max(0.0, l + k)), s)


def _parent_map():
    """code -> parent_code from the cancer-type registry (derived sub-cohorts
    only)."""
    from pirlygenes.load_dataset import get_data
    df = get_data("cancer-type-registry.csv")
    return {
        str(r.code): str(r.parent_code)
        for r in df.itertuples()
        if isinstance(r.parent_code, str) and r.parent_code.strip()
    }


def _peak_coverage_bars(mat, cohorts, ensg_to_sym, threshold):
    """Two sorted horizontal bar charts of each cohort's PEAK coverage % (the
    greedy plateau — share of patients with >=1 CTA over threshold):
      (a) colored by parent-cohort vs sub-divided (derived) cohort;
      (b) colored by the cohort's top CTA (the single most-covering CTA).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    parent_of = _parent_map()
    rows = []
    for code in cohorts:
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if not cum:
            continue
        top_cta = ensg_to_sym.get(mat.index[order[0]], mat.index[order[0]])
        rows.append({
            "code": code, "n": n, "peak": cum[-1] * 100,
            "is_derived": code in parent_of, "top_cta": top_cta,
        })
    df = pd.DataFrame(rows).sort_values("peak", ascending=True)  # ascending -> top at top after barh
    labels = [f"{c}  (n={n})" for c, n in zip(df["code"], df["n"])]
    h = max(6, len(df) * 0.26)

    # ---- (a) parent vs derived ----
    fig, ax = plt.subplots(figsize=(11, h))
    colors = ["#e85d04" if d else "#1d4e89" for d in df["is_derived"]]
    ax.barh(range(len(df)), df["peak"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"peak % of patients with ≥1 CTA > {threshold} TPM "
                  "(greedy plateau, co-occurrence-aware)")
    ax.set_xlim(0, 100)
    for y, p in enumerate(df["peak"]):
        ax.text(p + 0.6, y, f"{p:.0f}", va="center", fontsize=6)
    ax.legend(handles=[
        Patch(color="#1d4e89", label="parent / base cohort"),
        Patch(color="#e85d04", label="sub-divided (derived) cohort"),
    ], loc="lower right", fontsize=9)
    ax.set_title(f"CTA coverage ceiling by cancer type "
                 f"(> {threshold} TPM, Treehouse 25.01 PolyA)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"cta_peak_coverage_by_parent_child_t{threshold}.png", dpi=150)
    plt.close(fig)

    # ---- (b) colored by top CTA, grouped by antigen family ----
    # distinct base hue per family; members shaded within the family hue.
    distinct = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
        "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000", "#808000",
        "#000075", "#e6beff", "#aaffc3", "#ffd8b1", "#a9a9a9", "#fabed4",
    ]
    ctas = list(dict.fromkeys(df.sort_values("peak", ascending=False)["top_cta"]))
    fam_of = {c: _cta_family(c) for c in ctas}
    # families ordered by first (highest-coverage) appearance
    fam_order = list(dict.fromkeys(fam_of[c] for c in ctas))
    fam_base = {f: distinct[i % len(distinct)] for i, f in enumerate(fam_order)}
    members = {}
    for c in ctas:
        members.setdefault(fam_of[c], []).append(c)
    import matplotlib.colors as mcolors
    cta_color = {}
    for f, ms in members.items():
        base = mcolors.to_rgb(fam_base[f])
        ks = ([0.0] if len(ms) == 1
              else [(-0.18 + 0.36 * j / (len(ms) - 1)) for j in range(len(ms))])
        for c, k in zip(ms, ks):
            cta_color[c] = _shade(base, k)

    fig, ax = plt.subplots(figsize=(12, h))
    ax.barh(range(len(df)), df["peak"], color=[cta_color[c] for c in df["top_cta"]])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"peak % of patients with ≥1 CTA > {threshold} TPM")
    ax.set_xlim(0, 100)
    for y, (p, c) in enumerate(zip(df["peak"], df["top_cta"])):
        ax.text(p + 0.6, y, c, va="center", fontsize=6)
    # legend grouped by family: a family header swatch + its members
    handles = []
    for f in fam_order:
        handles.append(Patch(color=fam_base[f], label=f"{f}:"))
        for c in members[f]:
            handles.append(Patch(color=cta_color[c], label=f"   {c}"))
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              title="top CTA by family", ncol=2, handlelength=1.1,
              labelspacing=0.25, columnspacing=1.0)
    ax.set_title(f"CTA coverage ceiling by cancer type, dominant CTA colored by "
                 f"antigen family (> {threshold} TPM)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"cta_peak_coverage_by_top_cta_t{threshold}.png", dpi=150)
    plt.close(fig)
    print(f"      {len(df)} cohorts ({int(df['is_derived'].sum())} derived) "
          "-> 2 bar charts", flush=True)


def _coverage_every_cohort(mat, cohorts, ensg_to_sym, threshold):
    """One annotated greedy co-occurrence-aware coverage PNG per cancer type,
    into outputs/cta_coverage/, plus a small-multiples overview sorted by how
    fast each cohort saturates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = OUT / f"cta_coverage_t{threshold}"
    sub.mkdir(parents=True, exist_ok=True)
    summary = []  # (code, n, curve, names) for the overview + a saturation stat
    for code in sorted(cohorts):
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if not cum:
            continue
        names = [ensg_to_sym.get(mat.index[i], mat.index[i]) for i in order]
        summary.append((code, n, cum, names))

        fig, ax = plt.subplots(figsize=(9, 5.5))
        xs = range(1, len(cum) + 1)
        ax.plot(xs, [c * 100 for c in cum], marker="o", ms=3, color="#3a0ca3")
        for x, (nm, c) in enumerate(zip(names[:15], cum[:15]), start=1):
            ax.annotate(nm, (x, c * 100), fontsize=6, rotation=45,
                        textcoords="offset points", xytext=(2, 4))
        ax.set_xlabel("# CTAs added (greedy, co-occurrence-aware)")
        ax.set_ylabel(f"% of {code} patients with ≥1 CTA > {threshold} TPM")
        ax.set_title(f"{code}: cumulative distinct-patient coverage "
                     f"(> {threshold} TPM, n={n}); plateau "
                     f"{cum[-1]*100:.0f}%", fontsize=10)
        ax.set_xlim(0, min(30, len(cum) + 1))
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(sub / f"cta_coverage_{code}.png", dpi=130)
        plt.close(fig)

    # ---- small-multiples overview, sorted by plateau (broadest first) ----
    summary.sort(key=lambda t: t[2][-1], reverse=True)
    ncol = 6
    nrow = (len(summary) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 1.9),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (code, n, cum, _names) in zip(axes, summary):
        ax.plot(range(1, len(cum) + 1), [c * 100 for c in cum],
                color="#b5179e", lw=1.2)
        ax.fill_between(range(1, len(cum) + 1), [c * 100 for c in cum],
                        alpha=0.15, color="#b5179e")
        ax.set_title(f"{code} (n={n}) {cum[-1]*100:.0f}%", fontsize=7)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25)
    for ax in axes[len(summary):]:
        ax.axis("off")
    fig.suptitle(f"CTA panel coverage by cancer type — distinct patients "
                 f"with ≥1 CTA > {threshold} TPM (sorted by plateau)",
                 fontsize=11)
    fig.supxlabel("# CTAs added (greedy)", fontsize=8)
    fig.supylabel("% patients covered", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.98))
    fig.savefig(OUT / f"cta_coverage_all_cohorts_overview_t{threshold}.png", dpi=150)
    plt.close(fig)
    print(f"      wrote {len(summary)} per-cohort coverage PNGs + overview",
          flush=True)


def _plots(mat, cohorts, counts, ensg_to_sym, threshold, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tcol = f"n_gt{threshold}"
    pcol = f"pct_gt{threshold}"

    # ---- (2) patient-count heatmap: top CTAs (rows) × cohorts (cols) ----
    piv = counts.pivot_table(index="Symbol", columns="cancer_code",
                             values=tcol, fill_value=0)
    top_ctas = piv.sum(axis=1).sort_values(ascending=False).head(40).index
    top_cohorts = piv.sum(axis=0).sort_values(ascending=False).index
    piv = piv.loc[top_ctas, top_cohorts]
    fig, ax = plt.subplots(figsize=(max(10, len(top_cohorts) * 0.34),
                                    max(8, len(top_ctas) * 0.26)))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_title(f"# patients with CTA > {threshold} TPM "
                 f"(top 40 CTAs × cohorts, Treehouse 25.01 PolyA)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.5, label="patients")
    fig.tight_layout()
    fig.savefig(OUT / f"cta_patient_count_heatmap_t{threshold}.png", dpi=150)
    plt.close(fig)

    # also a %-of-cohort version (breadth, normalized for cohort size)
    pivp = counts.pivot_table(index="Symbol", columns="cancer_code",
                              values=pcol, fill_value=0).loc[top_ctas, top_cohorts]
    fig, ax = plt.subplots(figsize=(max(10, len(top_cohorts) * 0.34),
                                    max(8, len(top_ctas) * 0.26)))
    im = ax.imshow(pivp.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivp.columns)))
    ax.set_xticklabels(pivp.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivp.index)))
    ax.set_yticklabels(pivp.index, fontsize=6)
    ax.set_title(f"% of cohort with CTA > {threshold} TPM "
                 f"(top 40 CTAs × cohorts)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.5, label="% patients")
    fig.tight_layout()
    fig.savefig(OUT / f"cta_patient_pct_heatmap_t{threshold}.png", dpi=150)
    plt.close(fig)

    # ---- (3) sorted %-expressing bar for the focus cohort ----
    fc = counts[counts.cancer_code == focus].sort_values(pcol, ascending=False)
    fc = fc[fc[pcol] > 0].head(35)
    if len(fc):
        fig, ax = plt.subplots(figsize=(10, max(4, len(fc) * 0.28)))
        ax.barh(fc["Symbol"], fc[pcol], color="#b5179e")
        ax.invert_yaxis()
        ax.set_xlabel(f"% of {focus} patients > {threshold} TPM")
        n = int(fc["n_samples"].iloc[0])
        ax.set_title(f"{focus}: CTA expression breadth "
                     f"(> {threshold} TPM, n={n})", fontsize=10)
        for y, (p, k) in enumerate(zip(fc[pcol], fc[tcol])):
            ax.text(p, y, f" {k}", va="center", fontsize=6)
        fig.tight_layout()
        fig.savefig(OUT / f"cta_pct_bar_{focus}_t{threshold}.png", dpi=150)
        plt.close(fig)

    # ---- (4) co-occurrence-aware coverage curves ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for code in [focus, "SKCM", "LUAD", "BRCA", "OS"]:
        if code not in cohorts:
            continue
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if not cum:
            continue
        ax.plot(range(1, len(cum) + 1), [c * 100 for c in cum],
                marker="o", ms=2, lw=1, label=f"{code} (n={n})")
    ax.set_xlabel("# CTAs in panel (greedy, co-occurrence-aware)")
    ax.set_ylabel(f"% of patients with >=1 CTA > {threshold} TPM")
    ax.set_title(f"CTA panel coverage: distinct patients picked up "
                 f"(> {threshold} TPM)", fontsize=10)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"cta_coverage_curves_t{threshold}.png", dpi=150)
    plt.close(fig)

    # focus-cohort coverage with the CTA names annotated at each step
    order, cum, n = greedy_coverage(mat, cohorts.get(focus, []), threshold)
    if cum:
        names = [ensg_to_sym.get(mat.index[i], mat.index[i]) for i in order]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(range(1, len(cum) + 1), [c * 100 for c in cum],
                marker="o", color="#3a0ca3")
        for x, (nm, c) in enumerate(zip(names[:15], cum[:15]), start=1):
            ax.annotate(nm, (x, c * 100), fontsize=6, rotation=45,
                        textcoords="offset points", xytext=(2, 4))
        ax.set_xlabel("# CTAs added (greedy order)")
        ax.set_ylabel(f"% of {focus} patients covered")
        ax.set_title(f"{focus}: cumulative patient coverage as CTAs are added "
                     f"(> {threshold} TPM, n={n})", fontsize=10)
        ax.set_xlim(0, min(30, len(cum) + 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"cta_coverage_{focus}_t{threshold}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
