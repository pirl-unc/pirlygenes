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
from pirlygenes.builders.treehouse import _filter_samples, TreehouseCohort

import sweep_treehouse_polya_cohorts as polya
import sweep_treehouse_tcga_cohorts as tcga

CACHE = Path.home() / ".cache" / "pirlygenes" / "expression" / "treehouse-polya-25-01"
TPM_TSV = CACHE / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
CLINICAL = CACHE / "clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv"
GLIOMA_MAP = CACHE / "derived" / "tcga_glioma_case_to_project.csv"
OUT = Path(__file__).resolve().parent / "outputs"
THRESHOLDS = [25, 50, 100, 200]

# Honest display labels for cohorts whose registry code is misleading about the
# actual sample composition. The bare "SARC" cohort here is the TCGA
# "leiomyosarcoma" slice (n=100, ~90% leiomyosarcoma by GDC histology), NOT the
# multi-histology TCGA-SARC umbrella its code implies — so label it as such
# until the sarcoma taxonomy rebuild makes this canonical in the registry.
_DISPLAY_CODE = {"SARC": "TCGA-LMS"}


def _display_code(code: str) -> str:
    return _DISPLAY_CODE.get(code, code)


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
    # Reuse the cached extraction only if it's newer than the source matrix AND
    # already contains every wanted symbol — otherwise a newly-added CTA (e.g.
    # XAGE5) would be silently missing because the awk pass (the slow step over
    # the 6.9 GB compendium) was skipped against a stale, smaller symbol set.
    cache_ok = False
    if small.exists() and small.stat().st_mtime >= TPM_TSV.stat().st_mtime:
        cached_syms = set(pd.read_csv(small, sep="\t", usecols=[0])
                          .iloc[:, 0].astype(str))
        missing = set(sym_to_ensg) - cached_syms
        cache_ok = not missing
        if missing:
            print(f"      cache stale ({len(missing)} new symbol(s), e.g. "
                  f"{sorted(missing)[:5]}) -> re-extracting", flush=True)
    if cache_ok:
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


# Extra per-sample cohorts pulled from other cached per-cohort parquets (linear
# TPM) so coverage isn't confined to a single compendium. The set expands as
# more sources gain materialized per-sample data (see issue #275).
_EXTRA_PARQUET_COHORTS = [
    ("treehouse-ribod-25-01", "CHOR"),
    ("treehouse-ribod-25-01", "RB"),
]


def _add_parquet_cohorts(mat, cohorts, ctas):
    """Merge extra per-sample cohorts (from cached per-cohort parquets) into the
    CTA matrix + cohort buckets, aligned to the existing CTA index."""
    cache = Path.home() / ".cache" / "pirlygenes" / "expression"
    added = []
    for source_id, code in _EXTRA_PARQUET_COHORTS:
        p = cache / source_id / "derived" / f"{code}_per_sample_tpm.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df = df[df["Ensembl_Gene_ID"].astype(str).isin(set(ctas))]
        sample_cols = [c for c in df.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
        sub = df.set_index("Ensembl_Gene_ID")[sample_cols].reindex(mat.index).fillna(0.0)
        sub.columns = [f"{code}::{c}" for c in sub.columns]  # avoid id collisions
        mat = pd.concat([mat, sub], axis=1)
        cohorts[code] = list(sub.columns)
        added.append(f"{code}(n={len(sample_cols)})")
    if added:
        print(f"      + extra per-sample cohorts: {', '.join(added)}", flush=True)
    return mat, cohorts


def _merge_proteins(mat, ensg_to_sym):
    """Collapse near-identical CTA paralogs (>=90% AA identity, + curated
    NY-ESO-1 / XAGE1) into one protein per group, SUMMING per-sample TPM (total
    expression of that protein — RNA-seq can't disambiguate identical paralogs,
    and one TCR/antibody/vaccine addresses the group). Returns the merged matrix
    indexed by protein name + an identity {name: name} display map."""
    from pirlygenes.load_dataset import get_data
    grp = get_data("cta-protein-groups")
    group_of = dict(zip(grp["member_symbol"].astype(str),
                        grp["protein_group"].astype(str)))
    row_protein = [group_of.get(s, s)
                   for s in (ensg_to_sym.get(e, e) for e in mat.index)]
    merged = mat.groupby(pd.Index(row_protein, name="protein")).sum()
    print(f"      merged {len(mat) - len(merged)} paralog rows into protein "
          f"groups -> {len(merged)} proteins", flush=True)
    return merged, {name: name for name in merged.index}


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

    # Authoritative ENSG->Symbol straight from the CTA evidence table (tsarina),
    # so every CTA in the set gets a symbol — including ones absent from the
    # bundled reference-expression CSVs (e.g. XAGE5). The compendium is keyed by
    # HUGO symbol, and CTA_evidence Symbols are HUGO, so they align for the awk.
    ctas = set(gsc.CTA_gene_ids())
    ev = gsc.CTA_evidence()[["Symbol", "Ensembl_Gene_ID"]].dropna()
    ensg_to_sym = {}
    for sym, e in zip(ev.Symbol.astype(str), ev.Ensembl_Gene_ID.astype(str)):
        if e in ctas and e not in ensg_to_sym:
            ensg_to_sym[e] = sym
    print(f"[1/5] ENSG->Symbol for {len(ensg_to_sym)} CTAs", flush=True)

    print("[2/5] cohort sample buckets (builder filters)", flush=True)
    clinical = pd.read_csv(CLINICAL, sep="\t", dtype=str)
    cohorts = cohort_samples(clinical)
    print(f"      {len(cohorts)} cohorts, "
          f"{sum(len(v) for v in cohorts.values())} samples", flush=True)

    print("[3/5] extract CTA per-sample TPM matrix (awk slice + log2 inverse)", flush=True)
    mat = extract_cta_matrix(ensg_to_sym)
    mat, cohorts = _add_parquet_cohorts(mat, cohorts, ctas)
    mat, ensg_to_sym = _merge_proteins(mat, ensg_to_sym)
    print(f"      matrix {mat.shape[0]} CTA proteins × {mat.shape[1]} samples, "
          f"{len(cohorts)} cohorts", flush=True)

    print("[4/5] per (CTA × cohort) threshold counts -> CSV", flush=True)
    counts = per_cohort_counts(mat, cohorts, ensg_to_sym)
    counts = counts.sort_values(["cancer_code", "n_gt25"], ascending=[True, False])
    counts.to_csv(OUT / "cta_patient_counts.csv", index=False)
    # per-cohort union: # patients expressing >=1 CTA protein over each threshold
    # (each patient counted once regardless of how many CTAs they express). Also
    # emit a MAGE-excluded union so addressability can show a "without MAGE" panel.
    is_mage = mat.index.to_series().astype(str).str.upper().str.startswith(
        ("MAGEA", "MAGEB", "MAGEC")).to_numpy()
    urows = []
    for code, samples in cohorts.items():
        cols = [s for s in samples if s in mat.columns]
        if not cols:
            continue
        sub = mat[cols].to_numpy()
        sub_nomage = sub[~is_mage]
        rec = {"cancer_code": code, "n_samples": len(cols)}
        for t in THRESHOLDS:
            rec[f"n_any_gt{t}"] = int((sub > t).any(axis=0).sum())
            rec[f"n_any_gt{t}_nomage"] = int((sub_nomage > t).any(axis=0).sum())
        urows.append(rec)
    pd.DataFrame(urows).to_csv(OUT / "cta_union_counts.csv", index=False)

    print("[5/9] plots", flush=True)
    _plots(mat, cohorts, counts, ensg_to_sym, args.threshold, args.cohort)

    print("[6/9] per-cohort coverage curves (one PNG each)", flush=True)
    _coverage_every_cohort(mat, cohorts, ensg_to_sym, args.threshold)

    print("[7/9] peak-coverage bar charts (parent/child + top-CTA colored)", flush=True)
    _peak_coverage_bars(mat, cohorts, ensg_to_sym, args.threshold)

    print("[8/9] stacked coverage bar (per-CTA marginal contribution)", flush=True)
    _stacked_coverage_bars(mat, cohorts, ensg_to_sym, args.threshold)

    print("[9/9] CTA coverage vs cohort TMB", flush=True)
    _cta_vs_tmb(mat, cohorts, ensg_to_sym, args.threshold)
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
    # Guard against degenerate stems: symbols like C12orf42 / H2BC1 / ZC3H11B
    # strip to "C"/"H"/"ZC", which are not real antigen families. Below a 3-char
    # stem, treat the CTA as its own (singleton) family instead of a fake group.
    return stem if len(stem) >= 3 else s


def _shade(rgb, k):
    """Lighten (k>0) / darken (k<0) an RGB color, keeping its hue — used to
    distinguish members within one family."""
    import colorsys
    r, g, b = rgb[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1.0, max(0.0, l + k)), s)


# Distinct base hues, one per antigen family; members shaded within the hue.
_FAMILY_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000", "#808000",
    "#000075", "#e6beff", "#aaffc3", "#ffd8b1", "#a9a9a9", "#fabed4",
]


def _family_color_map(ctas_ordered):
    """Map an ordered list of CTA symbols to colors: one base hue per antigen
    family (ordered by first appearance), members shaded within the family hue.

    Returns (cta_color, fam_order, fam_base, members) so callers can both color
    bars and build a family-grouped legend."""
    import matplotlib.colors as mcolors

    ctas = list(dict.fromkeys(ctas_ordered))
    fam_of = {c: _cta_family(c) for c in ctas}
    fam_order = list(dict.fromkeys(fam_of[c] for c in ctas))
    fam_base = {f: _FAMILY_PALETTE[i % len(_FAMILY_PALETTE)]
                for i, f in enumerate(fam_order)}
    members = {}
    for c in ctas:
        members.setdefault(fam_of[c], []).append(c)
    cta_color = {}
    for f, ms in members.items():
        base = mcolors.to_rgb(fam_base[f])
        ks = ([0.0] if len(ms) == 1
              else [(-0.18 + 0.36 * j / (len(ms) - 1)) for j in range(len(ms))])
        for c, k in zip(ms, ks):
            cta_color[c] = _shade(base, k)
    return cta_color, fam_order, fam_base, members


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
    labels = [f"{_display_code(c)}  (n={n})" for c, n in zip(df["code"], df["n"])]
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
                 f"(> {threshold} TPM)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"cta_peak_coverage_by_parent_child_t{threshold}.png", dpi=150)
    plt.close(fig)

    # ---- (b) colored by top CTA, grouped by antigen family ----
    # distinct base hue per family; members shaded within the family hue,
    # ordered by first (highest-coverage) appearance.
    cta_color, fam_order, fam_base, members = _family_color_map(
        df.sort_values("peak", ascending=False)["top_cta"])

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


def _stacked_coverage_bars(mat, cohorts, ensg_to_sym, threshold,
                           max_label_segments=8, min_label_pct=3.0):
    """Horizontal STACKED bar per cohort: the greedy plateau split into each
    CTA's marginal NEW-patient contribution. Because the contributions are
    co-occurrence-aware (a patient already covered by an earlier CTA isn't
    re-counted), the segments sum to the plateau and the bar never exceeds
    100%. Segments are colored by antigen family (consistent across cohorts);
    the widest segments are labelled with the CTA symbol in small font.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Per cohort: ordered (cta_symbol, marginal_pct) contributions.
    rows = []
    for code in cohorts:
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if not cum:
            continue
        segs, prev = [], 0.0
        for i, idx in enumerate(order):
            sym = ensg_to_sym.get(mat.index[idx], mat.index[idx])
            marg = (cum[i] - prev) * 100
            prev = cum[i]
            if marg > 0:
                segs.append((sym, marg))
        rows.append({"code": code, "n": n, "peak": cum[-1] * 100, "segs": segs})
    rows.sort(key=lambda r: r["peak"])  # ascending -> broadest at top after barh

    # Global CTA color map: order CTAs by total marginal contribution so the
    # most prominent antigens get the most distinct family hues.
    from collections import Counter
    tot = Counter()
    for r in rows:
        for sym, marg in r["segs"]:
            tot[sym] += marg
    cta_color, fam_order, fam_base, _members = _family_color_map(
        [c for c, _ in tot.most_common()])

    labels = [f"{_display_code(r['code'])}  (n={r['n']})" for r in rows]
    hgt = max(6, len(rows) * 0.28)
    fig, ax = plt.subplots(figsize=(13, hgt))
    for y, r in enumerate(rows):
        left = 0.0
        for j, (sym, marg) in enumerate(r["segs"]):
            ax.barh(y, marg, left=left, color=cta_color.get(sym, "#cccccc"),
                    edgecolor="white", linewidth=0.3)
            # label the widest leading segments; always try the dominant one
            if (marg >= min_label_pct and j < max_label_segments) or j == 0:
                if marg >= 1.5:
                    ax.text(left + marg / 2, y, sym, va="center", ha="center",
                            fontsize=4.5, clip_on=True)
            left += marg
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"% of patients with ≥1 CTA > {threshold} TPM "
                  "(stacked by each CTA's marginal new-patient share, greedy order)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    handles = [Patch(color=fam_base[f], label=f) for f in fam_order]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              title="antigen family", ncol=2, handlelength=1.1,
              labelspacing=0.25, columnspacing=1.0)
    ax.set_title(f"CTA coverage by cancer type, split into each CTA's marginal "
                 f"new-patient contribution (> {threshold} TPM)", fontsize=11)
    fig.text(0.99, 0.005, f"{len(rows)} per-sample cohorts (others summary-only)",
             ha="right", fontsize=6, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / f"cta_stacked_coverage_t{threshold}.png", dpi=150)
    plt.close(fig)
    print(f"      {len(rows)} cohorts -> stacked coverage bar", flush=True)


# Cohorts whose registry code is the wrong TMB key for their actual samples.
# The bare "SARC" cohort is the TCGA leiomyosarcoma slice, so look up SARC_LMS.
_TMB_CODE = {"SARC": "SARC_LMS"}

# Coarse tissue-lineage groups for coloring the CTA-vs-TMB scatter, collapsed
# from the registry `family` column (lineage-only after the Phase-C refactor;
# age is a separate `pediatric` flag, so pediatric sarcomas color as sarcoma).
_LINEAGE_COLORS = {
    "sarcoma": "#e85d04", "embryonal": "#d00000", "carcinoma": "#1d4e89",
    "melanoma": "#6a040f", "neuroendocrine": "#2a9d8f", "heme": "#9d4edd",
    "CNS": "#3a86ff", "germ cell": "#ffb703", "endocrine": "#588157",
    "other": "#9a9a9a",
}


def _registry_family_map():
    from pirlygenes.load_dataset import get_data
    df = get_data("cancer-type-registry")
    return {str(r.code): str(r.family) for r in df.itertuples()}


def _lineage_group(family: str) -> str:
    f = (family or "").lower()
    if f == "sarcoma":
        return "sarcoma"
    if f == "embryonal":
        return "embryonal"
    if f == "melanoma":
        return "melanoma"
    if "neuroendocrine" in f:
        return "neuroendocrine"
    if f.startswith("heme"):
        return "heme"
    if f == "cns":
        return "CNS"
    if f == "germ-cell":
        return "germ cell"
    if f == "endocrine":
        return "endocrine"
    if f.startswith("carcinoma"):  # incl carcinoma-other
        return "carcinoma"
    return "other"


def _cta_vs_tmb(mat, cohorts, ensg_to_sym, threshold):
    """Scatter: each cancer type's CTA coverage plateau (% of patients with ≥1
    CTA over threshold) vs its published median TMB (mut/Mb, log x). Tumors with
    high CTA coverage but low TMB are the interesting quadrant for CTA-directed
    therapy (antigen present without relying on a high neoantigen load). Cohorts
    with no curated TMB value are dropped and counted in the caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Resolve every curated TMB code to its canonical registry code so cohort
    # lookups are synonym-proof (catch-all, nothing dropped on a spelling).
    tmb_map = {}
    for c, v in gsc.cancer_tmb().items():
        try:
            rc = gsc.resolve_cancer_type(c) or c
        except ValueError:
            rc = c
        tmb_map.setdefault(rc, v)
    pts, missing = [], []
    for code in cohorts:
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if not cum:
            continue
        key = _TMB_CODE.get(code, code)  # SARC cohort is the TCGA-LMS slice
        try:
            key = gsc.resolve_cancer_type(key) or key
        except ValueError:
            pass
        tmb = tmb_map.get(key)
        if tmb is None:
            missing.append(code)
            continue
        pts.append((code, n, cum[-1] * 100, tmb))
    if not pts:
        print("      no cohorts with both coverage and TMB; skip", flush=True)
        return

    xs = [p[3] for p in pts]
    ys = [p[2] for p in pts]
    fam_map = _registry_family_map()
    groups = [_lineage_group(fam_map.get(code, "")) for code, *_ in pts]

    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xscale("log")
    x_lo, x_hi = min(xs) * 0.7, max(xs) * 1.5
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 100)

    # Shade the "antigen-rich / mutation-poor" sweet spot: high CTA coverage
    # (>=50%) at low TMB (<=3 mut/Mb), where CTA-directed therapy is attractive
    # precisely because the low neoantigen load makes checkpoint blockade weak.
    SWEET_TMB, SWEET_COV = 3.0, 50.0
    ax.fill_between([x_lo, SWEET_TMB], SWEET_COV, 100, color="#ffd166",
                    alpha=0.18, zorder=0)
    ax.text(x_lo * 1.08, 98.5, "antigen-rich / mutation-poor",
            fontsize=8, va="top", ha="left", color="#9c6f00", style="italic")

    ax.scatter(xs, ys, s=34, c=[_LINEAGE_COLORS[g] for g in groups],
               alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("median tumor mutational burden (mut/Mb, log scale)")
    ax.set_ylabel(f"% of patients with ≥1 CTA > {threshold} TPM "
                  "(coverage plateau)")
    ax.grid(alpha=0.3, which="both")

    # Repel labels so they don't overlap (thin leader lines back to points).
    texts = [ax.text(tmb, cov, _display_code(code), fontsize=6)
             for code, n, cov, tmb in pts]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    except ImportError:  # fallback: small fixed offset (may overlap)
        for t, (code, n, cov, tmb) in zip(texts, pts):
            t.set_fontsize(5)
            t.set_position((tmb * 1.02, cov + 0.6))

    # Lineage legend (only groups present), placed outside the axes.
    present = [g for g in _LINEAGE_COLORS if g in set(groups)]
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor=_LINEAGE_COLORS[g], markeredgecolor="white",
                      label=g) for g in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7, title="lineage", frameon=False)

    sub = ""
    if len(pts) >= 3:
        r = np.corrcoef(np.log10(xs), ys)[0, 1]
        sub = f"; Pearson r(log TMB, coverage)={r:.2f}"
    ax.set_title(f"CTA coverage vs TMB by cancer type "
                 f"(> {threshold} TPM, n={len(pts)} cohorts{sub})", fontsize=10)
    n_registry = len(_registry_family_map())
    ax.text(0.99, 0.01,
            f"{len(pts)} per-sample cohorts shown (of {n_registry} registry "
            f"types; others summary-only); {len(missing)} dropped (no curated TMB)",
            transform=ax.transAxes, fontsize=6, color="gray", ha="right")
    fig.tight_layout()
    fig.savefig(OUT / f"cta_coverage_vs_tmb_t{threshold}.png", dpi=150)
    plt.close(fig)
    print(f"      {len(pts)} cohorts plotted, {len(missing)} dropped (no TMB)",
          flush=True)


def _coverage_every_cohort(mat, cohorts, ensg_to_sym, threshold):
    """One annotated greedy co-occurrence-aware coverage PNG per cancer type,
    into outputs/cta_coverage/gt<threshold>/, plus a small-multiples overview
    sorted by how fast each cohort saturates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = OUT / "cta_coverage" / f"gt{threshold}"
    sub.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for code in sorted(cohorts):
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if not cum:
            continue
        names = [ensg_to_sym.get(mat.index[i], mat.index[i]) for i in order]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xs = range(1, len(cum) + 1)
        ax.plot(xs, [c * 100 for c in cum], marker="o", ms=3, color="#3a0ca3")
        for x, (nm, c) in enumerate(zip(names[:15], cum[:15]), start=1):
            ax.annotate(nm, (x, c * 100), fontsize=6, rotation=45,
                        textcoords="offset points", xytext=(2, 4))
        ax.set_xlabel("# CTAs added (greedy, co-occurrence-aware)")
        ax.set_ylabel(f"% of {_display_code(code)} patients with ≥1 CTA "
                      f"> {threshold} TPM")
        ax.set_title(f"{_display_code(code)}: cumulative distinct-patient "
                     f"coverage (> {threshold} TPM, n={n}); plateau "
                     f"{cum[-1]*100:.0f}%", fontsize=10)
        ax.set_xlim(0, len(cum) + 1)  # go to this cohort's full plateau
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(sub / f"cta_coverage_{code}.png", dpi=130)
        plt.close(fig)
        n_written += 1
    print(f"      wrote {n_written} per-cohort coverage PNGs -> {sub}", flush=True)


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
    ax.set_xticklabels([_display_code(c) for c in piv.columns],
                       rotation=90, fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_title(f"# patients with CTA > {threshold} TPM "
                 f"(top 40 CTAs × cohorts)", fontsize=9)
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
    ax.set_xticklabels([_display_code(c) for c in pivp.columns],
                       rotation=90, fontsize=6)
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
        ax.set_xlabel(f"% of {_display_code(focus)} patients > {threshold} TPM")
        n = int(fc["n_samples"].iloc[0])
        ax.set_title(f"{_display_code(focus)}: CTA expression breadth "
                     f"(> {threshold} TPM, n={n})", fontsize=10)
        for y, (p, k) in enumerate(zip(fc[pcol], fc[tcol])):
            ax.text(p, y, f" {k}", va="center", fontsize=6)
        fig.tight_layout()
        fig.savefig(OUT / f"cta_pct_bar_{focus}_t{threshold}.png", dpi=150)
        plt.close(fig)

    # ---- (4) co-occurrence-aware coverage curves: one overlaid, multi-colour
    # line per cohort. Show many cohorts and run the x-axis out to the most CTAs
    # any shown cohort needs to plateau (no arbitrary cap). ----
    curves = []
    for code in cohorts:
        order, cum, n = greedy_coverage(mat, cohorts[code], threshold)
        if cum:
            curves.append((code, n, cum))
    curves.sort(key=lambda t: t[2][-1], reverse=True)
    keep = curves[:24]
    if focus in {c for c, _, _ in curves} and focus not in {c for c, _, _ in keep}:
        keep += [t for t in curves if t[0] == focus]
    cmap = matplotlib.colormaps["tab20"]
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (code, n, cum) in enumerate(keep):
        ax.plot(range(1, len(cum) + 1), [c * 100 for c in cum],
                marker="o", ms=2, lw=1.1, color=cmap(i % 20),
                label=f"{_display_code(code)} (n={n})")
    max_x = max(len(cum) for _, _, cum in keep) + 1  # most CTAs any cohort needs
    ax.set_xlabel("# CTAs in panel (greedy, co-occurrence-aware)")
    ax.set_ylabel(f"% of patients with ≥1 CTA > {threshold} TPM")
    ax.set_title(f"CTA panel coverage: distinct patients picked up "
                 f"(> {threshold} TPM; top {len(keep)} cohorts by plateau)",
                 fontsize=10)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, ncol=3, loc="lower right")
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
        ax.set_ylabel(f"% of {_display_code(focus)} patients covered")
        ax.set_title(f"{_display_code(focus)}: cumulative patient coverage as "
                     f"CTAs are added (> {threshold} TPM, n={n})", fontsize=10)
        ax.set_xlim(0, min(30, len(cum) + 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"cta_coverage_{focus}_t{threshold}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
