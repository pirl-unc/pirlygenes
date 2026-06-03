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


def build_cohorts() -> list[TreehouseCohort]:
    """Pediatric/sarcoma + TCGA-subset + GBM/LGG split — the comparable
    PolyA cohorts, using the builders' own filters."""
    cohorts = list(polya.COHORTS) + list(tcga.COHORTS) + _glioma_split_cohorts()
    # TCGA glioma is otherwise unsplit; drop a raw combined glioma if present.
    return cohorts


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

    print("[5/5] plots", flush=True)
    _plots(mat, cohorts, counts, ensg_to_sym, args.threshold, args.cohort)
    print(f"done -> {OUT}", flush=True)


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
    fig.savefig(OUT / "cta_patient_count_heatmap.png", dpi=150)
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
    fig.savefig(OUT / "cta_patient_pct_heatmap.png", dpi=150)
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
        fig.savefig(OUT / f"cta_pct_bar_{focus}.png", dpi=150)
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
    fig.savefig(OUT / "cta_coverage_curves.png", dpi=150)
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
        fig.savefig(OUT / f"cta_coverage_{focus}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
