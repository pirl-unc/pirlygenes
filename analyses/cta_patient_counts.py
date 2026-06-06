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
PERCENTILES = [80, 90, 95]


from dataclasses import dataclass


@dataclass(frozen=True)
class Threshold:
    """A CTA-positivity cutoff that is either an absolute TPM level
    (``kind='tpm'``) or a within-sample percentile rank after clean-TPM
    (``kind='pctile'``: a CTA is "on" in a sample if its TPM is at/above that
    sample's Nth-percentile TPM across all genes). The percentile cutoff is
    per-sample, so :meth:`cutoff` returns a vector aligned to the sample
    columns; greedy_coverage compares each column to its own cutoff via
    NumPy broadcasting."""
    kind: str   # 'tpm' | 'pctile'
    value: int

    @property
    def slug(self) -> str:
        return (f"t{self.value}" if self.kind == "tpm" else f"p{self.value}")

    @property
    def xlabel(self) -> str:
        return (f"> {self.value} TPM" if self.kind == "tpm"
                else f"≥ {self.value}th within-sample percentile")

    def cutoff(self, cols, pctile_cutoffs=None):
        """Scalar TPM (tpm mode) or per-sample cutoff array aligned to ``cols``
        (pctile mode). Samples without a percentile cutoff get +inf so no CTA
        is ever called positive there (rather than a silent false positive)."""
        if self.kind == "tpm":
            return self.value
        col = f"p{self.value}"
        series = pctile_cutoffs[col] if pctile_cutoffs is not None else None
        return np.array([
            (series.get(c, np.inf) if series is not None else np.inf)
            for c in cols
        ], dtype=float)


def per_sample_percentile_cutoffs(percentiles=PERCENTILES, *, nbins=2000,
                                  vmax_log2=20.0, cache=True) -> pd.DataFrame:
    """Per-sample TPM value at each within-sample percentile, computed over ALL
    genes in the compendium (not just CTAs) so a CTA's rank is relative to the
    whole transcriptome. Memory-bounded histogram pass over the full
    log2(TPM+1) TSV (+ the linear-TPM parquet cohorts); cached to
    ``outputs/_per_sample_pctile_cutoffs.parquet`` (keyed on TSV mtime)."""
    cachef = OUT / "_per_sample_pctile_cutoffs.parquet"
    if (cache and cachef.exists()
            and cachef.stat().st_mtime >= TPM_TSV.stat().st_mtime):
        return pd.read_parquet(cachef)

    edges = np.linspace(0.0, vmax_log2, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = None
    cols = None
    for chunk in pd.read_csv(TPM_TSV, sep="\t", chunksize=4000):
        vals = chunk.iloc[:, 1:]
        if cols is None:
            cols = list(vals.columns)
            hist = np.zeros((len(cols), nbins), dtype=np.int64)
        arr = vals.to_numpy(dtype=np.float32)            # genes(chunk) × samples
        idx = np.clip(np.searchsorted(edges, arr, side="right") - 1, 0, nbins - 1)
        # accumulate per-sample histograms: add 1 per (sample, bin) occurrence
        for j in range(idx.shape[1]):
            hist[j] += np.bincount(idx[:, j], minlength=nbins)

    cum = np.cumsum(hist, axis=1)
    total = cum[:, -1].astype(float)
    out = {"sample": cols}
    for p in percentiles:
        bins = (cum >= (total * (p / 100.0))[:, None]).argmax(axis=1)
        out[f"p{p}"] = np.power(2.0, centers[bins]) - 1.0   # log2 cutoff -> TPM
    df = pd.DataFrame(out).set_index("sample")

    # parquet cohorts ship linear TPM over their own gene set — compute their
    # per-sample percentiles directly and append.
    extra = []
    cache_dir = Path.home() / ".cache" / "pirlygenes" / "expression"
    for source_id, code in _EXTRA_PARQUET_COHORTS:
        p = cache_dir / source_id / "derived" / f"{code}_per_sample_tpm.parquet"
        if not p.exists():
            continue
        pf = pd.read_parquet(p)
        sample_cols = [c for c in pf.columns
                       if c not in ("Ensembl_Gene_ID", "Symbol")]
        m = pf[sample_cols].to_numpy(dtype=float)
        rec = {}
        for col, vec in zip(sample_cols, m.T):
            rec[col] = {f"p{q}": float(np.percentile(vec, q)) for q in percentiles}
        extra.append(pd.DataFrame(rec).T)
    if extra:
        df = pd.concat([df] + extra)
        df = df[~df.index.duplicated(keep="first")]
    if cache:
        df.to_parquet(cachef)
    return df

# Plot tick labels: the registry code is now authoritative (Phase C made SARC
# the honest pan-sarcoma grand union and split the histology atoms), so no
# special-case relabelling is needed — codes are used as-is. Kept as a thin
# hook in case a future cohort code needs an honest display override.
_DISPLAY_CODE: dict[str, str] = {}


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
            recode={"HNSC_HPV-": "HNSC_HPVneg", "HNSC_HPV+": "HNSC_HPVpos"})
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
    ("treehouse-ribod-25-01", "SARC_CHOR"),
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


def _add_aggregate_cohorts(cohorts):
    """Materialise computed-aggregate cohorts (SARC_RMS, SARC_LPS, TCGA_SARC,
    SARC_PAN) by pooling the per-sample columns of their member atoms that are
    actually present — so e.g. the four rhabdomyosarcoma subtypes show up as one
    'SARC_RMS' cohort. A patient is pooled once even if a member overlaps."""
    added = []
    for agg, members in gsc.cohort_aggregates().items():
        if agg in cohorts:
            continue  # don't shadow a real cohort of the same code
        pooled = []
        seen = set()
        present = 0
        for m in members:
            if m in cohorts:
                present += 1
                for s in cohorts[m]:
                    if s not in seen:
                        seen.add(s)
                        pooled.append(s)
        # only synthesize an aggregate that pools >=2 member cohorts of samples
        if present >= 2 and pooled:
            cohorts[agg] = pooled
            added.append(f"{agg}(n={len(pooled)}<-{present})")
    if added:
        print(f"      + aggregate cohorts: {', '.join(added)}", flush=True)
    return cohorts


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


def greedy_coverage(mat, samples, threshold, pctile_cutoffs=None):
    """Co-occurrence-aware: greedily order CTAs by marginal NEW patients (>thr).
    Returns (ordered_symbols_idx, cumulative_fraction, n_total).

    ``threshold`` is either a scalar TPM, or a :class:`Threshold` (TPM or
    within-sample percentile). For a percentile Threshold each sample column
    is compared to its own cutoff (NumPy broadcasting)."""
    cols = [s for s in samples if s in mat.columns]
    n = len(cols)
    cut = (threshold.cutoff(cols, pctile_cutoffs)
           if isinstance(threshold, Threshold) else threshold)
    hit = (mat[cols].to_numpy() > cut)  # CTA × patient boolean
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
    ap.add_argument("--no-percentiles", action="store_true",
                    help="skip the within-sample percentile-rank threshold plots")
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
    cohorts = _add_aggregate_cohorts(cohorts)
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

    tpm_thr = Threshold("tpm", args.threshold)
    print("[6/8] per-cohort coverage curves (one PNG each)", flush=True)
    _coverage_every_cohort(mat, cohorts, ensg_to_sym, tpm_thr)

    print("[7/8] stacked coverage bar (per-CTA marginal contribution)", flush=True)
    _stacked_coverage_bars(mat, cohorts, ensg_to_sym, tpm_thr)

    print("[8/8] CTA coverage vs cohort TMB", flush=True)
    _cta_vs_tmb(mat, cohorts, ensg_to_sym, tpm_thr)
    # HEPB (hepatoblastoma) sits at TMB ~0.02 — an extreme low outlier that
    # stretches the log-x axis for one point; emit a no-HEPB variant (legend
    # auto-prunes the now-empty embryonal group, x-axis auto-rescales).
    _cta_vs_tmb(mat, cohorts, ensg_to_sym, tpm_thr,
                exclude=frozenset({"HEPB"}), slug_suffix="_noHEPB")
    # Subtype-resolved variant: drop a parent category when its subtypes are
    # also plotted (no bulk BRCA/HNSC/SARC alongside their subtypes).
    _cta_vs_tmb(mat, cohorts, ensg_to_sym, tpm_thr,
                exclude=frozenset({"HEPB"}), slug_suffix="_noHEPB_subtypesonly",
                drop_covered_parents=True)

    # CTA load metrics vs TMB: (a) mean # CTAs expressed per sample, (b) mean
    # per-sample CTA-specific-9mer payload (Σ over the CTAs a sample expresses of
    # each CTA's count of 9mers absent from the whole non-CTA proteome). Weights
    # are mat-index-aligned; merged paralog groups take their best member's count
    # (one TCR/antibody addresses the group).
    from pirlygenes.load_dataset import get_data as _get_data
    spec_df = cta_specific_9mer_counts()
    sym2spec = dict(zip(spec_df["Symbol"].astype(str), spec_df["n_specific_9mers"]))
    try:
        _grp = _get_data("cta-protein-groups")
        for gname, members in _grp.groupby("protein_group"):
            vals = [sym2spec.get(m, 0) for m in members["member_symbol"].astype(str)]
            if vals:
                sym2spec.setdefault(str(gname), max(vals))
    except Exception:
        pass
    weights = np.array([float(sym2spec.get(str(name), 0)) for name in mat.index])
    payload_fn = _mean_specific_9mer_payload(weights)

    def _emit_load_metrics(thr, cutoffs=None):
        _metric_vs_tmb(mat, cohorts, thr, _mean_ctas_per_sample,
                       "mean # CTAs expressed per sample", "cta_mean_count_vs_tmb",
                       pctile_cutoffs=cutoffs, exclude=frozenset({"HEPB"}),
                       slug_suffix="_noHEPB")
        _metric_vs_tmb(mat, cohorts, thr, payload_fn,
                       "mean CTA-specific 9mers per sample", "cta_9mer_payload_vs_tmb",
                       pctile_cutoffs=cutoffs, exclude=frozenset({"HEPB"}),
                       slug_suffix="_noHEPB")

    _emit_load_metrics(tpm_thr)

    # Within-sample percentile-rank thresholds (after clean-TPM): a CTA is "on"
    # in a sample if its TPM is at/above that sample's Nth-percentile across all
    # genes. Pipeline-robust (rank harmonizes across quantification pipelines).
    # Generated once (gated on the base TPM threshold) since they don't depend
    # on --threshold; the cutoffs are cached after the first compute.
    if not args.no_percentiles and args.threshold == THRESHOLDS[0]:
        print(f"[+] within-sample percentile thresholds {PERCENTILES}", flush=True)
        cutoffs = per_sample_percentile_cutoffs()
        for p in PERCENTILES:
            pthr = Threshold("pctile", p)
            print(f"    p{p}: coverage curves + stacked bars + vs-TMB", flush=True)
            _coverage_every_cohort(mat, cohorts, ensg_to_sym, pthr, cutoffs)
            _stacked_coverage_bars(mat, cohorts, ensg_to_sym, pthr, cutoffs)
            _cta_vs_tmb(mat, cohorts, ensg_to_sym, pthr, cutoffs)
            _cta_vs_tmb(mat, cohorts, ensg_to_sym, pthr, cutoffs,
                        exclude=frozenset({"HEPB"}), slug_suffix="_noHEPB")
            _cta_vs_tmb(mat, cohorts, ensg_to_sym, pthr, cutoffs,
                        exclude=frozenset({"HEPB"}),
                        slug_suffix="_noHEPB_subtypesonly",
                        drop_covered_parents=True)
            _emit_load_metrics(pthr, cutoffs)
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


def _stacked_coverage_bars(mat, cohorts, ensg_to_sym, thr, pctile_cutoffs=None,
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
        order, cum, n = greedy_coverage(mat, cohorts[code], thr, pctile_cutoffs)
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
    ax.set_xlabel(f"% of patients with ≥1 CTA {thr.xlabel} "
                  "(stacked by each CTA's new-patient share)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    handles = [Patch(color=fam_base[f], label=f) for f in fam_order]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              title="antigen family", ncol=2, handlelength=1.1,
              labelspacing=0.25, columnspacing=1.0)
    ax.set_title(f"CTA coverage by cancer type, by each CTA's new-patient "
                 f"share ({thr.xlabel})", fontsize=11)
    fig.text(0.99, 0.005, f"{len(rows)} per-sample cohorts (others summary-only)",
             ha="right", fontsize=6, color="gray")
    fig.tight_layout()
    fig.savefig(OUT / f"cta_stacked_coverage_{thr.slug}.png", dpi=150)
    plt.close(fig)
    print(f"      {len(rows)} cohorts -> stacked coverage bar", flush=True)


# Per-cohort TMB is looked up through gsc.cancer_tmb (centralized resolver +
# parent inheritance); no cohort-specific TMB key overrides are needed now that
# SARC is the honest pan-sarcoma aggregate (it resolves to its own curated
# pan-sarcoma TMB rather than borrowing leiomyosarcoma's).
_TMB_CODE: dict[str, str] = {}

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


def _parent_code_map():
    """code -> parent_code (first parent) from the cancer-type registry."""
    from pirlygenes.load_dataset import get_data
    df = get_data("cancer-type-registry.csv")
    out = {}
    for r in df.itertuples():
        parents = _registry_parents(getattr(r, "parent_code", None))
        if parents:
            out[str(r.code)] = parents[0]
    return out


def _registry_parents(value):
    if not isinstance(value, str) or not value.strip() or value.lower() == "nan":
        return []
    return [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]


def _drop_covered_parents(pts, parent_of):
    """Drop any code that is a (transitive) ancestor of another code in the
    plotted set — keeping only the deepest subtype actually present. So with
    BRCA_LumA… present, BRCA is dropped; with HNSC_HPVpos present, HNSC is
    dropped; with SARC_LMS present, SARC is dropped; if only SARC_RMS_ARMS has
    TMB (no SARC_RMS / SARC point), SARC_RMS_ARMS stays. Pure subtractive
    filter: a code with no plotted descendant is untouched."""
    plotted = {p[0] for p in pts}
    covered = set()
    for code in plotted:
        cur = parent_of.get(code)
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            if cur in plotted:
                covered.add(cur)
            cur = parent_of.get(cur)
    return [p for p in pts if p[0] not in covered]


def _cta_vs_tmb(mat, cohorts, ensg_to_sym, thr, pctile_cutoffs=None,
                exclude=frozenset(), slug_suffix="", drop_covered_parents=False):
    """Scatter: each cancer type's CTA coverage plateau (% of patients with ≥1
    CTA over threshold) vs its published median TMB (mut/Mb, log x). Tumors with
    high CTA coverage but low TMB are the interesting quadrant for CTA-directed
    therapy (antigen present without relying on a high neoantigen load). Cohorts
    with no curated TMB value are dropped and counted in the caption.

    ``drop_covered_parents=True`` emits the subtype-resolved variant: a parent
    category is dropped whenever one of its subtypes is also plotted (no bulk
    BRCA when PAM50 subtypes show, no HNSC without HPV status, no bulk SARC),
    falling back to whatever level actually carries a curated TMB."""
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
    # Computed sarcoma aggregates (SARC, SARC_RMS, SARC_LPS, TCGA_SARC, …) have
    # no single honest TMB: their expression is a grand union over subtypes but
    # published TMB spans 0.6–2.2 mut/Mb across those subtypes, so one umbrella
    # number conflates a 4× spread. Plot only subtype-level points; the atoms
    # each carry their own curated TMB. (SARC-TMB scope fix, option C.)
    aggregate_codes = set(gsc.cohort_aggregates().keys())
    pts, missing = [], []
    for code in cohorts:
        if code in aggregate_codes or code in exclude:
            continue
        order, cum, n = greedy_coverage(mat, cohorts[code], thr, pctile_cutoffs)
        if not cum:
            continue
        key = _TMB_CODE.get(code, code)
        try:
            key = gsc.resolve_cancer_type(key) or key
        except ValueError:
            pass
        tmb = tmb_map.get(key)
        if tmb is None:
            missing.append(code)
            continue
        pts.append((code, n, cum[-1] * 100, tmb))
    if drop_covered_parents:
        pts = _drop_covered_parents(pts, _parent_code_map())
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
    # Headroom above 100% so cohorts at the coverage ceiling (e.g. ATRT, NUTM)
    # show their full marker instead of a clipped half-dot.
    Y_TOP = 105
    ax.set_ylim(0, Y_TOP)

    # Shade the "antigen-rich / mutation-poor" sweet spot: high CTA coverage
    # (>=50%) at low TMB (<=3 mut/Mb), where CTA-directed therapy is attractive
    # precisely because the low neoantigen load makes checkpoint blockade weak.
    # Fill to the top of the (extended) y-axis so the band isn't capped at 100.
    SWEET_TMB, SWEET_COV = 3.0, 50.0
    ax.fill_between([x_lo, SWEET_TMB], SWEET_COV, Y_TOP, color="#ffd166",
                    alpha=0.18, zorder=0)
    # Sit the label a little below the band's top edge so it clears the
    # ceiling-coverage dots (ATRT/NUTM) that now sit near y=100.
    ax.text(x_lo * 1.08, 93, "antigen-rich / mutation-poor",
            fontsize=8, va="top", ha="left", color="#9c6f00", style="italic")

    ax.scatter(xs, ys, s=34, c=[_LINEAGE_COLORS[g] for g in groups],
               alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("median tumor mutational burden (mut/Mb, log scale)")
    ax.set_ylabel(f"% of patients with ≥1 CTA {thr.xlabel} "
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

    ex_note = f"; excludes {', '.join(sorted(exclude))}" if exclude else ""
    sub_note = "; subtype-resolved" if drop_covered_parents else ""
    ax.set_title(f"CTA coverage vs TMB by cancer type "
                 f"({thr.xlabel}, n={len(pts)} cohorts{ex_note}{sub_note})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"cta_coverage_vs_tmb_{thr.slug}{slug_suffix}.png", dpi=150)
    plt.close(fig)
    print(f"      {len(pts)} cohorts plotted{ex_note}, "
          f"{len(missing)} dropped (no TMB)", flush=True)


def cta_specific_9mer_counts(*, ensembl_release=112, k=9, paralog_overlap=0.8,
                             refresh=False):
    """Per expressed-CTA count of k-mers (default 9mer) that occur in the CTA's
    protein but in NO non-CTA protein — a sequence-level tumor-specificity score.

    Negative set = every protein-coding gene's canonical (longest) protein,
    EXCLUDING (a) the full CTA universe (``CTA_unfiltered_gene_ids`` — so the
    low-expression "in-between" CTAs are kept out) AND (b) near-identical
    **paralog copies** of universe genes, defined as any non-universe gene with
    ≥ ``paralog_overlap`` of its 9mers already in the universe proteome. Without
    (b), un-curated amplicon copies (DAZ2/DAZ4 vs DAZ1/DAZ3, CT47A8-10 vs the
    CT47 CTAs, CT45A5/6/8/9, MAGEA2B, …) sit in the negative set and 100%-cancel
    their family's specificity. For each expressed CTA we count its 9mers absent
    from the (cleaned) negative set. Cached to ``outputs/cta_specific_9mers.csv``.
    """
    cache = OUT / "cta_specific_9mers.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    from pyensembl import EnsemblRelease
    genome = EnsemblRelease(ensembl_release)
    pos = set(gsc.CTA_gene_ids())
    universe = set(gsc.CTA_unfiltered_gene_ids())
    id2name = gsc.CTA_gene_id_to_name()
    longest: dict[str, str] = {}
    for tr in genome.transcripts():
        if tr.biotype != "protein_coding":
            continue
        seq = tr.protein_sequence
        if not seq or len(seq) < k:
            continue
        seq = seq.rstrip("*")
        if tr.gene_id not in longest or len(seq) > len(longest[tr.gene_id]):
            longest[tr.gene_id] = seq

    def kmers(s):
        return {s[i:i + k] for i in range(len(s) - k + 1)}

    # (1) union of universe 9mers, used both as the specificity reference and to
    # detect paralog copies that should not count as "normal" background.
    universe_kmers = set()
    for gid in universe:
        seq = longest.get(gid)
        if seq:
            universe_kmers |= kmers(seq)
    # (2) negative = non-universe genes that are NOT near-identical paralogs.
    negative = set()
    n_paralogs = 0
    for gid, seq in longest.items():
        if gid in universe:
            continue
        km = kmers(seq)
        if km and sum(1 for x in km if x in universe_kmers) / len(km) >= paralog_overlap:
            n_paralogs += 1   # amplicon/paralog copy of a CTA -> not background
            continue
        negative |= km
    rows = []
    for gid in sorted(pos):
        seq = longest.get(gid)
        km = kmers(seq) if seq else set()
        rows.append({
            "Ensembl_Gene_ID": gid,
            "Symbol": id2name.get(gid, gid),
            "n_9mers": len(km),
            "n_specific_9mers": sum(1 for x in km if x not in negative),
        })
    df = pd.DataFrame(rows).sort_values("n_specific_9mers", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"      CTA-specific 9mers: {len(df)} CTAs vs {len(negative):,} "
          f"non-CTA 9mers ({n_paralogs} paralog copies excluded; "
          f"median {int(df.n_specific_9mers.median())})", flush=True)
    return df


def _cohort_on_matrix(mat, cols, thr, pctile_cutoffs):
    """Boolean CTA × patient 'on' matrix for one cohort at a Threshold."""
    cut = (thr.cutoff(cols, pctile_cutoffs)
           if isinstance(thr, Threshold) else thr)
    return mat[cols].to_numpy() > cut


def _metric_vs_tmb(mat, cohorts, thr, value_fn, ylabel, slug, *,
                   pctile_cutoffs=None, exclude=frozenset(), slug_suffix=""):
    """Generic scatter of a per-cohort per-sample CTA metric vs median TMB
    (log-x), styled like :func:`_cta_vs_tmb` but with a free (auto) y-axis and
    no sweet-spot band. ``value_fn(mat, cols, thr, pctile_cutoffs) -> float`` is
    the cohort's metric (e.g. mean CTAs/sample, mean CTA-specific-9mer payload)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tmb_map = {}
    for c, v in gsc.cancer_tmb().items():
        try:
            rc = gsc.resolve_cancer_type(c) or c
        except ValueError:
            rc = c
        tmb_map.setdefault(rc, v)
    aggregate_codes = set(gsc.cohort_aggregates().keys())
    pts = []
    for code, samples in cohorts.items():
        if code in aggregate_codes or code in exclude:
            continue
        cols = [s for s in samples if s in mat.columns]
        if not cols:
            continue
        key = _TMB_CODE.get(code, code)
        try:
            key = gsc.resolve_cancer_type(key) or key
        except ValueError:
            pass
        tmb = tmb_map.get(key)
        if tmb is None:
            continue
        pts.append((code, len(cols), value_fn(mat, cols, thr, pctile_cutoffs), tmb))
    if not pts:
        print("      no cohorts with metric + TMB; skip", flush=True)
        return
    xs = [p[3] for p in pts]
    ys = [p[2] for p in pts]
    fam_map = _registry_family_map()
    groups = [_lineage_group(fam_map.get(code, "")) for code, *_ in pts]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xscale("log")
    ax.set_xlim(min(xs) * 0.7, max(xs) * 1.5)
    ax.set_ylim(0, max(ys) * 1.08)
    ax.scatter(xs, ys, s=34, c=[_LINEAGE_COLORS[g] for g in groups],
               alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("median tumor mutational burden (mut/Mb, log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    texts = [ax.text(tmb, y, _display_code(code), fontsize=6)
             for code, n, y, tmb in pts]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    except ImportError:
        for t, (code, n, y, tmb) in zip(texts, pts):
            t.set_fontsize(5)
            t.set_position((tmb * 1.02, y))
    present = [g for g in _LINEAGE_COLORS if g in set(groups)]
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor=_LINEAGE_COLORS[g], markeredgecolor="white",
                      label=g) for g in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7, title="lineage", frameon=False)
    ex_note = f"; excludes {', '.join(sorted(exclude))}" if exclude else ""
    ax.set_title(f"{ylabel} vs TMB by cancer type "
                 f"({thr.xlabel}, n={len(pts)} cohorts{ex_note})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}_{thr.slug}{slug_suffix}.png", dpi=150)
    plt.close(fig)
    print(f"      {slug}: {len(pts)} cohorts plotted{ex_note}", flush=True)


def _mean_ctas_per_sample(mat, cols, thr, pctile_cutoffs):
    on = _cohort_on_matrix(mat, cols, thr, pctile_cutoffs)
    return float(on.sum(axis=0).mean())


def _mean_specific_9mer_payload(weights):
    """Build a value_fn: mean over patients of the summed CTA-specific-9mer
    count across the CTAs that patient expresses. ``weights`` is a per-CTA
    (mat-index-aligned) vector of specific-9mer counts."""
    def value_fn(mat, cols, thr, pctile_cutoffs):
        on = _cohort_on_matrix(mat, cols, thr, pctile_cutoffs)
        return float((on * weights[:, None]).sum(axis=0).mean())
    return value_fn


def _coverage_every_cohort(mat, cohorts, ensg_to_sym, thr, pctile_cutoffs=None):
    """One annotated greedy co-occurrence-aware coverage PNG per cancer type,
    into outputs/cta_coverage/<slug>/, plus a small-multiples overview
    sorted by how fast each cohort saturates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = OUT / "cta_coverage" / thr.slug
    sub.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for code in sorted(cohorts):
        order, cum, n = greedy_coverage(mat, cohorts[code], thr, pctile_cutoffs)
        if not cum:
            continue
        names = [ensg_to_sym.get(mat.index[i], mat.index[i]) for i in order]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xs = range(1, len(cum) + 1)
        ax.plot(xs, [c * 100 for c in cum], marker="o", ms=3, color="#3a0ca3")
        for x, (nm, c) in enumerate(zip(names[:15], cum[:15]), start=1):
            ax.annotate(nm, (x, c * 100), fontsize=6, rotation=45,
                        textcoords="offset points", xytext=(2, 4))
        ax.set_xlabel("# CTAs added")
        ax.set_ylabel(f"% of {_display_code(code)} patients with ≥1 CTA "
                      f"{thr.xlabel}")
        ax.set_title(f"{_display_code(code)}: cumulative patient coverage "
                     f"({thr.xlabel}, n={n})", fontsize=10)
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
    ax.set_title(f"# patients expressing each CTA (> {threshold} TPM)", fontsize=9)
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
    ax.set_title(f"% of cohort expressing each CTA (> {threshold} TPM)", fontsize=9)
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
    ax.set_xlabel("# CTAs in panel")
    ax.set_ylabel(f"% of patients with ≥1 CTA > {threshold} TPM")
    ax.set_title(f"CTA panel coverage by cancer type "
                 f"(> {threshold} TPM; top {len(keep)} cohorts)",
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
        ax.set_xlabel("# CTAs added")
        ax.set_ylabel(f"% of {_display_code(focus)} patients covered")
        ax.set_title(f"{_display_code(focus)}: cumulative patient coverage "
                     f"(> {threshold} TPM, n={n})", fontsize=10)
        ax.set_xlim(0, min(30, len(cum) + 1))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"cta_coverage_{focus}_t{threshold}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
