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
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pirlygenes import gene_sets_cancer as gsc
from pirlygenes.builders.treehouse import _filter_samples, TreehouseCohort
from pirlygenes.coverage import greedy_coverage as _pkg_greedy_coverage

import sweep_treehouse_polya_cohorts as polya
import sweep_treehouse_tcga_cohorts as tcga

# Expression source cache (Treehouse compendium TPM + clinical + cBioPortal
# derived maps). Distinct from the output CACHE below — keep the names separate
# (a 5.22.18 collision silently broke _DERIVED_DIR / the cBioPortal splits).
EXPR_CACHE = Path.home() / ".cache" / "pirlygenes" / "expression" / "treehouse-polya-25-01"
TPM_TSV = EXPR_CACHE / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
CLINICAL = EXPR_CACHE / "clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv"
GLIOMA_MAP = EXPR_CACHE / "derived" / "tcga_glioma_case_to_project.csv"
# Three roles, three locations:
#  * OUT     — the stable base directory.
#  * CACHE   — OUT/_cache: expensive, regenerable, reused across runs (percentile
#              cutoffs, CTA-specific-9mer counts, symbol intermediates). Gitignored.
#  * FIGDIR  — a per-run subfolder (run_<ts>/) holding EVERYTHING a run produces:
#              the plots AND this run's summary tables (cta_patient_counts.csv,
#              cta_union_counts.csv). So a run is one self-contained snapshot and a
#              fresh run never overwrites/mixes with an older one.
# CACHE and FIGDIR are (re)assigned in main() from CLI args.
OUT = Path(__file__).resolve().parent / "outputs"
CACHE = OUT / "_cache"
FIGDIR = OUT
THRESHOLDS = [25, 50, 100, 200]
PERCENTILES = [80, 90, 95]


def _out_path(group: str, name: str) -> Path:
    """``<FIGDIR>/<group>/<name>.png``, creating ``<group>/``.

    Plot families that differ only by threshold / percentile / focus cohort /
    x-axis (``t25``, ``t50``, ``p90``, ``GBM_t25`` …) live together in a folder
    named for what the plot shows, instead of a flat soup of suffixed filenames.
    FIGDIR is the (optionally timestamped) per-run plot destination."""
    d = FIGDIR / group
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.png"


def _flatten_singletons(figdir):
    """Collapse any plot subfolder that ended up holding exactly one file into a
    flat ``<foldername><ext>`` (e.g. ``foo/only.png`` -> ``foo.png``), so a lone
    plot isn't buried in its own directory. Skips folders that contain
    subfolders. Run as a final pass (after all invocations into a run dir) so
    per-threshold folders that accumulate >=2 files aren't prematurely flattened."""
    figdir = Path(figdir)
    if not figdir.is_dir():
        return
    for d in sorted(p for p in figdir.iterdir() if p.is_dir()):
        entries = list(d.iterdir())
        if len(entries) == 1 and entries[0].is_file():
            f = entries[0]
            dest = figdir / f"{d.name}{f.suffix}"
            if not dest.exists():
                f.rename(dest)
                d.rmdir()
                print(f"      flattened {d.name}/{f.name} -> {dest.name}",
                      flush=True)


def _tqdm(iterable, desc, **kw):
    """tqdm progress bar with a graceful no-op fallback if tqdm isn't installed.
    ``desc`` labels the category currently being rendered."""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, leave=False, **kw)


def _pct_axis(ax, which):
    """Format an axis' tick numbers as percentages (50 -> '50%'), for axes whose
    values are already a 0-100 percentage (coverage plateau, aPD-1 ORR, burden
    share). ``which`` is 'x' or 'y'."""
    from matplotlib.ticker import PercentFormatter
    getattr(ax, f"{which}axis").set_major_formatter(PercentFormatter(xmax=100, decimals=0))


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
    log2(TPM+1) TSV (+ the linear-TPM parquet cohorts); cached.

    cDNA-identical loci are **collapsed (summed) into one entry before ranking**,
    so the ranking universe holds one XAGE1 (=XAGE1A+XAGE1B), not the individual
    members — otherwise a collapsed CTA would be ranked against a universe that
    still contains its own components (#21)."""
    cachef = CACHE / "_per_sample_pctile_cutoffs_v3.parquet"   # v2: cDNA-collapsed
    if (cache and cachef.exists()
            and cachef.stat().st_mtime >= TPM_TSV.stat().st_mtime):
        return pd.read_parquet(cachef)
    from pirlygenes.expression.protein_groups import (
        cdna_symbol_to_canonical_symbol)
    sym2canon = {k.upper(): v for k, v in cdna_symbol_to_canonical_symbol().items()}

    edges = np.linspace(0.0, vmax_log2, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = None
    cols = None
    accum: dict[str, np.ndarray] = {}      # canonical symbol -> per-sample LINEAR sum
    for chunk in pd.read_csv(TPM_TSV, sep="\t", chunksize=4000):
        genes = chunk.iloc[:, 0].astype(str).str.upper()
        vals = chunk.iloc[:, 1:]
        if cols is None:
            cols = list(vals.columns)
            hist = np.zeros((len(cols), nbins), dtype=np.int64)
        arr = vals.to_numpy(dtype=np.float32)            # genes(chunk) × samples (log2)
        canon = genes.map(sym2canon)                     # canonical symbol or NaN
        member = canon.notna().to_numpy()
        # histogram the genes that are NOT in a cDNA-identical group, as-is
        if (~member).any():
            nm_idx = np.clip(np.searchsorted(edges, arr[~member], side="right") - 1,
                             0, nbins - 1)
            for j in range(nm_idx.shape[1]):
                hist[j] += np.bincount(nm_idx[:, j], minlength=nbins)
        # accumulate group members (linear) into their canonical's per-sample sum
        if member.any():
            lin = np.power(2.0, arr[member].astype(np.float64)) - 1.0
            for gi, cs in enumerate(canon[member].to_numpy()):
                accum.setdefault(cs, np.zeros(len(cols)))
                accum[cs] += lin[gi]
    # each collapsed group contributes ONE entry per sample (its summed value)
    for linsum in accum.values():
        log2v = np.log2(linsum + 1.0)
        bins_c = np.clip(np.searchsorted(edges, log2v, side="right") - 1, 0, nbins - 1)
        hist[np.arange(len(cols)), bins_c] += 1

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
        # collapse cDNA-identical loci (sum) before ranking, same as the TSV pass
        canon = pf["Symbol"].astype(str).str.upper().map(sym2canon)
        pf = pf.assign(_g=canon.fillna(pf["Symbol"].astype(str)))
        m = pf.groupby("_g")[sample_cols].sum().to_numpy(dtype=float)
        rec = {}
        for col, vec in zip(sample_cols, m.T):
            # match the prefixed column names _add_parquet_cohorts assigns
            # ("{code}::{sample}"), else the cutoff lookup misses -> +inf -> the
            # cohort's within-sample-percentile coverage is wrongly zero.
            rec[f"{code}::{col}"] = {f"p{q}": float(np.percentile(vec, q))
                                     for q in percentiles}
        extra.append(pd.DataFrame(rec).T)
    if extra:
        df = pd.concat([df] + extra)
        df = df[~df.index.duplicated(keep="first")]
    if cache:
        df.to_parquet(cachef)
    return df

# Plot tick labels: the registry code is now authoritative (Phase C made SARC
# the honest pan-sarcoma grand union and split the histology atoms), so no
# special-case relabelling is needed. The only transform is the shared
# pos/neg -> superscript formatting (HNSC_HPVpos -> HNSC_HPV⁺).
def _display_code(code: str) -> str:
    return gsc.format_cancer_code_label(code)


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


_DERIVED_DIR = EXPR_CACHE / "derived"


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
        + _tcga_case_subtype_cohorts(
            _DERIVED_DIR / "cbioportal_ucec_subtype.csv",
            "uterine corpus endometrioid carcinoma", "patientId", "ucec_subtype",
            recode={"UCEC_CN_LOW": "UCEC_CNL", "UCEC_CN_HIGH": "UCEC_CNH"})
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
    CACHE.mkdir(parents=True, exist_ok=True)
    symfile = CACHE / "_cta_symbols.txt"
    symfile.write_text("\n".join(sym_to_ensg) + "\n")
    small = CACHE / "_cta_rows.tsv"
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
    # colorectal MSI/MSS subtypes (per-sample) so the coverage plots can resolve
    # CRC_MSI / CRC_MSS (pooled from these by _add_crc_subtype_cohorts)
    ("treehouse-polya-25-01", "COAD_MSI"),
    ("treehouse-polya-25-01", "READ_MSI"),
    ("treehouse-polya-25-01", "COAD_MSS"),
    ("treehouse-polya-25-01", "READ_MSS"),
]

# KEYNOTE-177 etc. report *colorectal*, so COAD+READ of a given microsatellite
# status are one CRC tier (the causal-factors / apd1 plots pool the same way).
_CRC_SUBTYPE_POOL = {"CRC_MSI": ["COAD_MSI", "READ_MSI"],
                     "CRC_MSS": ["COAD_MSS", "READ_MSS"]}


def _add_crc_subtype_cohorts(cohorts):
    """Synthesize CRC_MSI / CRC_MSS by pooling the per-sample COAD/READ MSI/MSS
    cohorts (de-duplicated). Bulk CRC=[COAD,READ] is already a registry
    aggregate; this adds the microsatellite-resolved tiers."""
    added = []
    for agg, members in _CRC_SUBTYPE_POOL.items():
        pooled, seen = [], set()
        for m in members:
            for s in cohorts.get(m, []):
                if s not in seen:
                    seen.add(s)
                    pooled.append(s)
        if pooled:
            cohorts[agg] = pooled
            added.append(f"{agg}(n={len(pooled)})")
    if added:
        print(f"      + CRC subtype cohorts: {', '.join(added)}", flush=True)
    return cohorts


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
    """Collapse **cDNA-identical** loci (byte-identical canonical CDS, + the
    curated overrides such as CT47A) into one entry per group, SUMMING per-sample
    TPM — the universal read-recovery collapse (a quantifier can't disambiguate
    identical-CDS paralogs, so each is split and only the sum is reliable). This
    is the SAME collapse the reference accessor uses (``collapse_cdna_identical``,
    pirlygenes #417), so the per-sample and cohort-summary pipelines agree. NOT a
    >=90% near-identical rollup — distinct proteins (MAGEA3 vs MAGEA6) stay
    separate. Returns the merged matrix (canonical-Ensembl-indexed) + a
    ``{ensg: symbol}`` display map."""
    from pirlygenes.expression.protein_groups import (cdna_canonical_to_symbol,
                                                      cdna_member_to_canonical)
    m2c = cdna_member_to_canonical()
    c2s = cdna_canonical_to_symbol()
    row_canon = [m2c.get(str(e).split(".")[0], str(e).split(".")[0])
                 for e in mat.index]
    merged = mat.groupby(pd.Index(row_canon, name="gene")).sum()
    print(f"      merged {len(mat) - len(merged)} cDNA-identical rows -> "
          f"{len(merged)} genes", flush=True)
    display = {e: c2s.get(e, ensg_to_sym.get(e, e)) for e in merged.index}
    return merged, display


def per_cohort_counts(mat, cohorts, ensg_to_sym, pctile_cutoffs=None):
    """Long table: CTA × cohort × threshold -> n patients, pct.

    Columns ``n_gt{t}``/``pct_gt{t}`` for the absolute TPM thresholds, and (when
    ``pctile_cutoffs`` is supplied) ``n_p{q}``/``pct_p{q}`` for the WITHIN-SAMPLE
    percentile thresholds — a CTA is 'on' in a sample if its TPM is at/above that
    sample's qth-percentile across all genes (each sample compared to its own
    cutoff), not a per-cohort threshold."""
    rows = []
    for code, samples in cohorts.items():
        cols = [s for s in samples if s in mat.columns]
        if not cols:
            continue
        sub = mat[cols]
        n = len(cols)
        pcuts = {}
        if pctile_cutoffs is not None:
            for q in PERCENTILES:
                pcuts[q] = Threshold("pctile", q).cutoff(cols, pctile_cutoffs)
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
            for q, cut in pcuts.items():
                k = int((vals > cut).sum())
                rec[f"n_p{q}"] = k
                rec[f"pct_p{q}"] = round(100 * k / n, 2)
                any_hit = any_hit or k > 0
            if any_hit:
                rows.append(rec)
    return pd.DataFrame(rows)


def greedy_coverage(mat, samples, threshold, pctile_cutoffs=None):
    """Co-occurrence-aware: greedily order CTAs by marginal NEW patients (>thr).
    Returns (ordered_row_positions, cumulative_fraction, n_total).

    Thin CTA-specific wrapper around :func:`pirlygenes.coverage.greedy_coverage`
    (the single source of truth for the greedy marginal-gain loop): it subsets
    the matrix to this cohort's sample columns and resolves ``threshold`` —
    either a scalar TPM, or a :class:`Threshold` (TPM / within-sample
    percentile) whose per-sample cutoff vector broadcasts against the matrix —
    then delegates the loop. Row positions are relative to ``mat`` (the column
    subset preserves row order), so existing ``mat.index[i]`` callers are
    unaffected."""
    cols = [s for s in samples if s in mat.columns]
    cut = (threshold.cutoff(cols, pctile_cutoffs)
           if isinstance(threshold, Threshold) else threshold)
    return _pkg_greedy_coverage(mat[cols], cut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=25,
                    help="TPM cutoff for the heatmap / bar / coverage plots")
    ap.add_argument("--cohort", default="GBM",
                    help="cohort for the %%-bar and coverage plots")
    ap.add_argument("--no-percentiles", action="store_true",
                    help="skip the within-sample percentile-rank threshold plots")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory for the PNG plots (default: analyses/outputs). "
                         "Only the plots move here — the caches "
                         "(_per_sample_pctile_cutoffs.parquet, cta_specific_9mers.csv) "
                         "and summary tables always stay in analyses/outputs, so a "
                         "new --out-dir never forces a recompute.")
    ap.add_argument("--run-name", default=None,
                    help="name of the per-run plot subfolder under the plot dir "
                         "(default: a timestamp, run_YYYYMMDD-HHMMSS)")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="write plots straight into the plot dir (no per-run "
                         "subfolder) — flat layout, mixes with prior runs")
    ap.add_argument("--flatten", action="store_true",
                    help="final pass: collapse any plot folder left with a single "
                         "file into <folder>.png. Pass on the LAST invocation into "
                         "a run dir (after t25+t50 etc.), not a mid-run one.")
    args = ap.parse_args()

    # CACHE is the FIXED stable cache dir (OUT/_cache) — never moved by --out-dir,
    # so reruns reuse the cached cutoffs/9mers. FIGDIR is the per-run snapshot
    # (plots + this run's tables), redirectable via --out-dir.
    global FIGDIR, CACHE
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE = OUT / "_cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    fig_base = args.out_dir.resolve() if args.out_dir is not None else OUT
    if args.no_timestamp:
        FIGDIR = fig_base
    else:
        run = args.run_name or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        FIGDIR = fig_base / run
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"[0/5] cache -> {CACHE}\n      run (plots+tables) -> {FIGDIR}",
          flush=True)

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
    cohorts = _add_crc_subtype_cohorts(cohorts)
    cohorts = _add_aggregate_cohorts(cohorts)
    cohorts = _add_subtype_intermediates(cohorts)
    mat, ensg_to_sym = _merge_proteins(mat, ensg_to_sym)
    print(f"      matrix {mat.shape[0]} CTA proteins × {mat.shape[1]} samples, "
          f"{len(cohorts)} cohorts", flush=True)

    # Within-sample percentile cutoffs (computed once, cached) so the counts
    # CSVs carry n_p80/90/95 alongside n_gt25/50/100/200 — downstream plots
    # (addressability) can then offer percentile versions too.
    pctile_cuts = None if args.no_percentiles else per_sample_percentile_cutoffs()

    print("[4/5] per (CTA × cohort) threshold counts -> CSV", flush=True)
    counts = per_cohort_counts(mat, cohorts, ensg_to_sym, pctile_cuts)
    counts = counts.sort_values(["cancer_code", "n_gt25"], ascending=[True, False])
    counts.to_csv(FIGDIR / "cta_patient_counts.csv", index=False)
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
        if pctile_cuts is not None:
            for q in PERCENTILES:
                cut = Threshold("pctile", q).cutoff(cols, pctile_cuts)  # per-sample
                rec[f"n_any_p{q}"] = int((sub > cut).any(axis=0).sum())
                rec[f"n_any_p{q}_nomage"] = int((sub_nomage > cut).any(axis=0).sum())
        urows.append(rec)
    union_df = pd.DataFrame(urows)
    union_df.to_csv(FIGDIR / "cta_union_counts.csv", index=False)
    # also a STABLE, tracked copy so other analyses (the aPD1 cta_burden) can read
    # the %-patients->=1-CTA->=95th-percentile coverage without hunting run dirs.
    union_df.to_csv(OUT / "_cta_union_counts.csv", index=False)

    print("[5/9] plots", flush=True)
    _plots(mat, cohorts, counts, ensg_to_sym, args.threshold, args.cohort)

    tpm_thr = Threshold("tpm", args.threshold)
    print("[6/8] per-cohort coverage curves (one PNG each)", flush=True)
    _coverage_every_cohort(mat, cohorts, ensg_to_sym, tpm_thr)

    print("[7/8] stacked coverage bar (per-CTA marginal contribution)", flush=True)
    _stacked_coverage_bars(mat, cohorts, ensg_to_sym, tpm_thr)

    print("[8/8] CTA coverage / load vs cohort TMB / aPD-1 / burden axes",
          flush=True)
    _emit_coverage(mat, cohorts, ensg_to_sym, tpm_thr)
    _tmb_vs_apd1()   # cohort-level TMB-vs-aPD-1 context scatter (once per run)

    # CTA load metrics vs TMB: (a) mean # CTAs expressed per sample, (b) mean
    # per-sample CTA-specific-9mer load (Σ over the CTAs a sample expresses of
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
    load_fn = _mean_specific_9mer_load(weights)

    def _emit_load_metrics(thr, cutoffs=None):
        # Full matrix: each load metric (mean #CTAs/sample, mean CTA-specific-9mer
        # load/sample) vs every x-axis (TMB, aPD-1, the four burden axes). Each
        # gets base + collapsed (molecular leaf subtypes -> one-level joint
        # cohorts). TMB additionally gets no-HEPB variants (the TMB~0.02 outlier),
        # a TMB plot colored by aPD-1 ORR, and — for the 9mer load — a log-y
        # companion (load spans ~90x: PAAD ~15 → SKCM ~1300).
        for value_fn, ylabel, slug_base in (
            (_mean_ctas_per_sample, "mean # CTAs expressed per sample",
             "cta_mean_count"),
            (load_fn, "mean CTA-specific 9mers per sample",
             "cta_9mer_load"),
            (_mean_total_cta_tpm, "mean total CTA TPM per sample",
             "cta_total_tpm"),
        ):
            for xa in _tqdm(_all_axes(), f"{slug_base} {thr.slug}"):
                _metric_vs_x(mat, cohorts, thr, value_fn, ylabel, slug_base,
                               pctile_cutoffs=cutoffs, xaxis=xa)
                _metric_vs_x(mat, cohorts, thr, value_fn, ylabel, slug_base,
                               pctile_cutoffs=cutoffs, xaxis=xa,
                               slug_suffix="_collapsed", collapse_subtypes=True)
                # log-scale metrics (9mer load, total CTA TPM) span orders of
                # magnitude -> add a log-y companion.
                is_load = slug_base in ("cta_9mer_load", "cta_total_tpm")
                if xa.short == "tmb":
                    _metric_vs_x(mat, cohorts, thr, value_fn, ylabel, slug_base,
                                   pctile_cutoffs=cutoffs, xaxis=xa,
                                   exclude=frozenset({"HEPB"}),
                                   slug_suffix="_noHEPB")
                    # metric vs TMB, points colored by anti-PD-1 ORR (+ log-y for
                    # the 9mer load, which spans ~90x)
                    _metric_vs_x(mat, cohorts, thr, value_fn, ylabel, slug_base,
                                   pctile_cutoffs=cutoffs, xaxis=xa,
                                   color_by=_apd1_axis())
                    if is_load:
                        _metric_vs_x(mat, cohorts, thr, value_fn, ylabel,
                                       slug_base, pctile_cutoffs=cutoffs, xaxis=xa,
                                       color_by=_apd1_axis(), log_y=True)
                # log-y companion of the 9mer load on every axis (regular)
                if is_load:
                    _metric_vs_x(mat, cohorts, thr, value_fn, ylabel, slug_base,
                                   pctile_cutoffs=cutoffs, xaxis=xa, log_y=True)

    _emit_load_metrics(tpm_thr)

    # Within-sample percentile-rank thresholds (after clean-TPM): a CTA is "on"
    # in a sample if its TPM is at/above that sample's Nth-percentile across all
    # genes. Pipeline-robust (rank harmonizes across quantification pipelines).
    # Generated once (gated on the base TPM threshold) since they don't depend
    # on --threshold; the cutoffs are cached after the first compute.
    if not args.no_percentiles and args.threshold == THRESHOLDS[0]:
        print(f"[+] within-sample percentile thresholds {PERCENTILES}", flush=True)
        cutoffs = per_sample_percentile_cutoffs()
        for p in _tqdm(PERCENTILES, "percentile thresholds"):
            pthr = Threshold("pctile", p)
            _coverage_every_cohort(mat, cohorts, ensg_to_sym, pthr, cutoffs)
            _stacked_coverage_bars(mat, cohorts, ensg_to_sym, pthr, cutoffs)
            _emit_coverage(mat, cohorts, ensg_to_sym, pthr, cutoffs)
            _emit_load_metrics(pthr, cutoffs)
    if args.flatten:
        _flatten_singletons(FIGDIR)
    print(f"done -> plots in {FIGDIR}", flush=True)


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
    ax.set_xlabel(f"Patients with ≥1 CTA {thr.xlabel}")
    ax.set_xlim(0, 100)
    _pct_axis(ax, "x")
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
    fig.savefig(_out_path("cta_stacked_coverage", thr.slug), dpi=300)
    plt.close(fig)
    print(f"      {len(rows)} cohorts -> stacked coverage bar", flush=True)


# Coarse tissue-lineage groups for coloring the CTA-vs-TMB scatter, collapsed
# from the registry `family` column (lineage-only after the Phase-C refactor;
# age is a separate `pediatric` flag, so pediatric sarcomas color as sarcoma).
_LINEAGE_COLORS = {
    "sarcoma": "#e85d04", "embryonal": "#d00000", "carcinoma": "#1d4e89",
    "melanoma": "#6a040f", "neuroendocrine": "#2a9d8f", "heme": "#9d4edd",
    "CNS": "#3a86ff", "germ cell": "#ffb703", "endocrine": "#588157",
    "other": "#9a9a9a",
}


@lru_cache(maxsize=1)
def _registry_family_map():
    from pirlygenes.load_dataset import get_data
    df = get_data("cancer-type-registry")
    fam = {str(r.code): str(r.family) for r in df.itertuples()}
    # Aggregate / one-level-intermediate cohorts (SARC, SARC_RMS, SARC_LPS, …)
    # aren't registry codes — inherit a member leaf's family so they color by
    # lineage (e.g. SARC_RMS -> sarcoma) instead of falling through to "other".
    for agg, members in gsc.cohort_aggregates().items():
        if agg not in fam:
            for m in members:
                if m in fam:
                    fam[agg] = fam[m]
                    break
    for leaf, inter in _subtype_collapse_map().items():
        if inter not in fam and leaf in fam:
            fam[inter] = fam[leaf]
    return fam


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


@lru_cache(maxsize=1)
def _subtype_collapse_map() -> dict:
    """``leaf code -> one-level joint cohort``. Collapses the deepest molecular
    sub-subtypes onto a single big-n cohort one level up: the curated
    intermediate aggregates (SARC_RMS, SARC_LPS — every cohort-aggregate except
    the grand SARC union), plus name-prefix groups for the remaining 2-underscore
    codes (SARC_ESS_LG/HG -> SARC_ESS, NEC_LUNG_LARGECELL -> NEC_LUNG). So a
    collapsed plot shows SARC_RMS (joint, large n) instead of the four RMS atoms,
    while the leaf-level plot still breaks them out."""
    out = {}
    for agg, members in gsc.cohort_aggregates().items():
        if agg == "SARC":          # grand union, not a one-level intermediate
            continue
        for m in members:
            out[m] = agg
    for c in gsc.cancer_type_registry()["code"].astype(str):
        if c.count("_") >= 2 and c not in out:
            out[c] = "_".join(c.split("_")[:2])
    return out


def _add_subtype_intermediates(cohorts):
    """Ensure every one-level joint cohort in :func:`_subtype_collapse_map`
    exists as a pooled cohort (union of its leaves' samples). SARC_RMS / SARC_LPS
    are already pooled by :func:`_add_aggregate_cohorts`; this fills in the rest
    (SARC_ESS, NEC_LUNG)."""
    leaves_of = {}
    for leaf, inter in _subtype_collapse_map().items():
        leaves_of.setdefault(inter, []).append(leaf)
    added = []
    for inter, leaves in leaves_of.items():
        if inter in cohorts:
            continue
        pooled, seen = [], set()
        for lf in leaves:
            for s in cohorts.get(lf, []):
                if s not in seen:
                    seen.add(s)
                    pooled.append(s)
        if pooled:
            cohorts[inter] = pooled
            added.append(f"{inter}(n={len(pooled)})")
    if added:
        print(f"      + subtype intermediates: {', '.join(added)}", flush=True)
    return cohorts


def _select_codes(cohorts, val_map, exclude, collapse_subtypes):
    """Return ``[(code, x)]`` for the codes to plot. Drops grand aggregates and
    codes with no x value. When ``collapse_subtypes``, molecular leaf subtypes
    are replaced by their one-level joint cohort (:func:`_subtype_collapse_map`)
    with a sample-weighted-mean x value across the leaves."""
    agg = set(gsc.cohort_aggregates().keys())
    collapse = _subtype_collapse_map() if collapse_subtypes else {}
    intermediates = set(collapse.values())
    inter_x = {}
    for inter in intermediates:
        num = den = 0.0
        for leaf, tgt in collapse.items():
            if tgt != inter:
                continue
            x = val_map.get(_resolve_code(leaf))
            n = len(cohorts.get(leaf, []))
            if x is not None and n > 0:
                num += x * n
                den += n
        if den:
            inter_x[inter] = num / den
    out = []
    for code in cohorts:
        if code in exclude:
            continue
        if collapse_subtypes:
            if code in collapse:          # a leaf -> shown via its intermediate
                continue
            if code in intermediates:
                x = inter_x.get(code)
            elif code in agg:             # grand aggregate (SARC, …)
                continue
            else:
                x = val_map.get(_resolve_code(code))
        else:
            if code in agg:
                continue
            x = val_map.get(_resolve_code(code))
        if x is None:
            continue
        out.append((code, x))
    return out


def _scatter_points(fig, ax, xs, ys, pts, color_by):
    """Plot the points either colored by lineage group (``color_by=None``, with a
    lineage legend) or by a continuous cohort-level axis value (``color_by`` an
    :class:`_XAxis`, e.g. anti-PD-1 ORR, with a colorbar; cohorts lacking that
    value are drawn light-grey)."""
    from matplotlib.lines import Line2D
    if color_by is None:
        fam = _registry_family_map()
        groups = [_lineage_group(fam.get(c, "")) for c, *_ in pts]
        ax.scatter(xs, ys, s=34, c=[_LINEAGE_COLORS[g] for g in groups],
                   alpha=0.9, edgecolor="white", linewidth=0.4, zorder=3)
        present = [g for g in _LINEAGE_COLORS if g in set(groups)]
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor=_LINEAGE_COLORS[g],
                          markeredgecolor="white", label=g) for g in present]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=7, title="lineage", frameon=False)
        return
    cvals = [color_by.values.get(_resolve_code(c)) for c, *_ in pts]
    gx = [x for x, v in zip(xs, cvals) if v is None]
    gy = [y for y, v in zip(ys, cvals) if v is None]
    if gx:
        ax.scatter(gx, gy, s=30, c="#dddddd", edgecolor="white", linewidth=0.4,
                   zorder=2, label=f"no {color_by.short}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.92), fontsize=7,
                  frameon=False)
    hx = [x for x, v in zip(xs, cvals) if v is not None]
    hy = [y for y, v in zip(ys, cvals) if v is not None]
    hc = [v for v in cvals if v is not None]
    if hx:
        sc = ax.scatter(hx, hy, s=40, c=hc, cmap="viridis", edgecolor="white",
                        linewidth=0.4, zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(color_by.label)


class _XAxis(NamedTuple):
    """An x-axis for the per-cohort CTA scatters — either published median TMB
    (log) or curated anti-PD-1 ORR (linear). ``values`` maps canonical cancer
    code -> x value; ``sweet_x_max`` is the right edge of the shaded
    "antigen-rich / therapy-attractive" band (low TMB, or low aPD-1 response),
    the quadrant where CTA-directed therapy is attractive because the
    alternative (checkpoint blockade riding a high neoantigen load) is weak."""
    short: str          # slug/title fragment: "tmb" / "apd1" / "incidence"
    title: str          # title phrase: "TMB" / "anti-PD-1 response"
    label: str          # x-axis label
    log: bool
    values: dict        # {canonical_code: x_value}
    sweet_x_max: float  # band edge: x <= this (sweet_high=False) or x >= this (True)
    sweet_label: str
    # When False the therapy-attractive band is the LOW-x side (low TMB / low
    # aPD-1 ORR). When True it's the HIGH-x side — for incidence the attractive
    # quadrant is a *large* patient population (high incidence) at high coverage.
    sweet_high: bool = False
    # pct=True -> the axis values are percentages: format the tick numbers with a
    # trailing "%" (and the label drops the "(%)" suffix).
    pct: bool = False


def _resolve_code(code: str) -> str:
    return gsc.resolve_cancer_type(code, strict=False) or code


def _axis_value_map(raw: dict) -> dict:
    """Resolve every curated code to its canonical registry code (synonym-proof);
    first value wins on a collision."""
    out: dict = {}
    for c, v in raw.items():
        out.setdefault(_resolve_code(c), v)
    return out


@lru_cache(maxsize=1)
def _tmb_axis() -> _XAxis:
    # Computed sarcoma aggregates (SARC, SARC_RMS, …) have no single honest TMB —
    # their expression is a grand union over subtypes spanning a ~4× TMB range —
    # so they are excluded by the aggregate-code filter and only subtype-level
    # atoms (each with its own curated TMB) are plotted. (SARC-TMB scope fix.)
    return _XAxis(
        "tmb", "TMB",
        "median tumor mutational burden (mut/Mb)", True,
        _axis_value_map(gsc.cancer_tmb()), 3.0, "antigen-rich / mutation-poor")


@lru_cache(maxsize=1)
def _apd1_axis() -> _XAxis:
    # Empty sweet_label -> no shaded sweet-spot band on the aPD-1 axis.
    return _XAxis(
        "apd1", "anti-PD-1 response",
        "anti-PD-1 monotherapy ORR", False,
        _axis_value_map(gsc.cancer_apd1_response()), 10.0, "", pct=True)


# Disease-burden x-axes: each cancer type's share (%) of annual cancer
# {incidence, mortality}, {global = GLOBOCAN 2022 world, usa = ACS}. Burden is
# curated per tissue ``burden_category``, so every code under a category inherits
# that category's share (resolved via the registry burden-category crosswalk).
# Unlike TMB/aPD-1, the therapy-attractive band is the HIGH-x side: a large
# addressable / high-unmet-need population at high CTA coverage is the compelling
# quadrant. Linear x.
_BURDEN_COL = {
    ("incidence", "global"): "world_incidence_pct",
    ("incidence", "usa"): "us_incidence_pct",
    ("mortality", "global"): "world_mortality_pct",
    ("mortality", "usa"): "us_mortality_pct",
}


@lru_cache(maxsize=8)
def _burden_axis(metric: str, scope: str) -> _XAxis:
    col = _BURDEN_COL[(metric, scope)]
    burden = gsc.cancer_burden(metric=col)  # {category: pct}
    raw = {}
    for code in gsc.cancer_type_registry()["code"].astype(str):
        cat = gsc.burden_category(code)
        pct = burden.get(cat) if cat else None
        if pct is not None and pd.notna(pct):
            raw[code] = float(pct)
    region = "world" if scope == "global" else "US"
    band = "high-incidence / large-population" if metric == "incidence" \
        else "high-mortality / unmet-need"
    return _XAxis(
        f"{metric}_{scope}", f"{region} {metric}",
        f"share of annual cancer {metric}, {region}", False,
        _axis_value_map(raw), 5.0, band, sweet_high=True, pct=True)


def _incidence_axis() -> _XAxis:           # back-compat alias
    return _burden_axis("incidence", "global")


def _burden_axes() -> list:
    return [_burden_axis(m, s) for m in ("incidence", "mortality")
            for s in ("global", "usa")]


def _all_axes() -> list:
    """Every cohort-level x-axis the scatters sweep: TMB, anti-PD-1 ORR, and the
    four disease-burden axes (incidence/mortality × global/usa)."""
    return [_tmb_axis(), _apd1_axis()] + _burden_axes()


def _xlim(xs, xaxis: _XAxis):
    if xaxis.log:
        return min(xs) * 0.7, max(xs) * 1.5
    pad = max(1.0, 0.04 * (max(xs) - min(xs)))
    return min(xs) - pad, max(xs) + pad


def _emit_coverage(mat, cohorts, ensg_to_sym, thr, cutoffs=None):
    """CTA coverage-plateau scatter vs every x-axis (TMB, aPD-1, and the four
    burden axes). Each axis gets: base, subtype-resolved (drop registry parents),
    and collapsed (molecular leaf subtypes -> one-level joint cohorts like
    SARC_RMS). TMB additionally gets its no-HEPB variants (the TMB~0.02 log-x
    outlier). The TMB-axis coverage plot is also emitted colored by aPD-1 ORR."""
    for xa in _tqdm(_all_axes(), f"coverage scatters {thr.slug}"):
        is_tmb = xa.short == "tmb"
        _cta_vs_x(mat, cohorts, ensg_to_sym, thr, cutoffs, xaxis=xa)
        _cta_vs_x(mat, cohorts, ensg_to_sym, thr, cutoffs, xaxis=xa,
                    slug_suffix="_subtypesonly", drop_covered_parents=True)
        _cta_vs_x(mat, cohorts, ensg_to_sym, thr, cutoffs, xaxis=xa,
                    slug_suffix="_collapsed", drop_covered_parents=True,
                    collapse_subtypes=True)
        if is_tmb:
            _cta_vs_x(mat, cohorts, ensg_to_sym, thr, cutoffs, xaxis=xa,
                        exclude=frozenset({"HEPB"}), slug_suffix="_noHEPB")
            _cta_vs_x(mat, cohorts, ensg_to_sym, thr, cutoffs, xaxis=xa,
                        exclude=frozenset({"HEPB"}),
                        slug_suffix="_noHEPB_subtypesonly",
                        drop_covered_parents=True)
            # coverage vs TMB, points colored by anti-PD-1 ORR
            _cta_vs_x(mat, cohorts, ensg_to_sym, thr, cutoffs, xaxis=xa,
                        color_by=_apd1_axis())


def _cta_vs_x(mat, cohorts, ensg_to_sym, thr, pctile_cutoffs=None,
                exclude=frozenset(), slug_suffix="", drop_covered_parents=False,
                xaxis=None, collapse_subtypes=False, color_by=None):
    """Scatter: each cancer type's CTA coverage plateau (% of patients with ≥1
    CTA over threshold) vs an x-axis metric (``xaxis``, default published median
    TMB; pass :func:`_apd1_axis` for anti-PD-1 ORR). Tumors with high CTA
    coverage but a low x value (low TMB / low aPD-1 response) are the interesting
    quadrant for CTA-directed therapy. Cohorts with no x value are dropped and
    counted in the caption.

    ``drop_covered_parents=True`` emits the subtype-resolved variant: a parent
    category is dropped whenever one of its subtypes is also plotted (no bulk
    BRCA when PAM50 subtypes show, no HNSC without HPV status, no bulk SARC),
    falling back to whatever level actually carries a curated x value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xaxis = xaxis or _tmb_axis()
    plan = _select_codes(cohorts, xaxis.values, exclude, collapse_subtypes)
    pts = []
    for code, val in plan:
        order, cum, n = greedy_coverage(mat, cohorts[code], thr, pctile_cutoffs)
        if not cum:
            continue
        pts.append((code, n, cum[-1] * 100, val))
    if drop_covered_parents:
        pts = _drop_covered_parents(pts, _parent_code_map())
    if not pts:
        print(f"      no cohorts with both coverage and {xaxis.short}; skip",
              flush=True)
        return

    xs = [p[3] for p in pts]
    ys = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    if xaxis.log:
        ax.set_xscale("log")
    x_lo, x_hi = _xlim(xs, xaxis)
    ax.set_xlim(x_lo, x_hi)
    # Headroom above 100% so cohorts at the coverage ceiling (e.g. ATRT, NUTM)
    # show their full marker instead of a clipped half-dot.
    Y_TOP = 105
    ax.set_ylim(0, Y_TOP)

    # Shade the "therapy-attractive" sweet spot: high CTA coverage (>=50%) at the
    # attractive x side. Low-x for TMB (CTA therapy wins where the tumor is
    # mutation-poor); high-x for incidence/mortality (large / high-unmet-need
    # population). Axes with an empty sweet_label (e.g. aPD-1) draw no band.
    SWEET_COV = 50.0
    if xaxis.sweet_label:
        if xaxis.sweet_high:
            band_x, ha = [xaxis.sweet_x_max, x_hi], "right"
            label_x = x_hi - 0.01 * (x_hi - x_lo)
        else:
            band_x = [x_lo, xaxis.sweet_x_max]
            label_x = x_lo * 1.08 if xaxis.log else x_lo + 0.01 * (x_hi - x_lo)
            ha = "left"
        ax.fill_between(band_x, SWEET_COV, Y_TOP, color="#ffd166",
                        alpha=0.18, zorder=0)
        # Sit the label a little below the band's top edge so it clears the
        # ceiling-coverage dots (ATRT/NUTM) that now sit near y=100.
        ax.text(label_x, 93, xaxis.sweet_label,
                fontsize=8, va="top", ha=ha, color="#9c6f00", style="italic")

    _scatter_points(fig, ax, xs, ys, pts, color_by)
    ax.set_xlabel(xaxis.label)
    ax.set_ylabel(f"Patients with ≥1 CTA {thr.xlabel}")
    _pct_axis(ax, "y")                       # coverage plateau is a percentage
    if xaxis.pct:
        _pct_axis(ax, "x")
    ax.grid(alpha=0.3, which="both")

    # Repel labels so they don't overlap (thin leader lines back to points).
    texts = [ax.text(val, cov, _display_code(code), fontsize=6)
             for code, n, cov, val in pts]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    except ImportError:  # fallback: small fixed offset (may overlap)
        for t, (code, n, cov, val) in zip(texts, pts):
            t.set_fontsize(5)
            t.set_position((val * 1.02 if xaxis.log else val + 0.5, cov + 0.6))

    ex_note = f"; excludes {', '.join(sorted(exclude))}" if exclude else ""
    sub_note = "; subtype-resolved" if drop_covered_parents else ""
    col_note = "; collapsed subtypes" if collapse_subtypes else ""
    ax.set_title(f"CTA coverage vs {xaxis.title} by cancer type "
                 f"({thr.xlabel}, n={len(pts)} cohorts{ex_note}{sub_note}{col_note})",
                 fontsize=10)
    fig.tight_layout()
    slug_dir = f"cta_coverage_vs_{xaxis.short}"
    if color_by is not None:
        slug_dir += f"_colorby_{color_by.short}"
    fig.savefig(_out_path(slug_dir, f"{thr.slug}{slug_suffix}"), dpi=300)
    plt.close(fig)
    print(f"      {slug_dir}: {len(pts)} cohorts plotted{ex_note}", flush=True)


def cta_specific_9mer_counts(*, ensembl_release=112, k=9, refresh=False):
    """Per expressed-CTA count of k-mers (default 9mer) that occur in the CTA's
    protein but in NO non-CTA protein — a sequence-level tumor-specificity score.

    Negative set = every protein-coding gene's canonical (longest) protein EXCEPT
    the full CTA universe (``CTA_unfiltered_gene_ids``), so the low-expression
    "in-between" CTAs are kept out of the negative (per the spec). **tsarina is
    the authority on CTA membership** — it already weighs paralogs and normal-
    tissue expression — so we trust its universe verbatim and do NOT second-guess
    it here: if a near-identical paralog copy (e.g. DAZ2/DAZ4, CT47A8-10) is
    absent from the universe, its 9mers count as background and its CTA sibling
    scores low. That is the honest result for the current curation; missing
    paralog copies are a tsarina curation question (filed upstream), not a
    pirlygenes workaround. Cached to ``outputs/cta_specific_9mers.csv``.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "cta_specific_9mers.csv"
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

    negative = set()
    for gid, seq in longest.items():
        if gid in universe:
            continue
        negative |= kmers(seq)
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
          f"non-CTA 9mers (median {int(df.n_specific_9mers.median())})", flush=True)
    return df


def _cohort_on_matrix(mat, cols, thr, pctile_cutoffs):
    """Boolean CTA × patient 'on' matrix for one cohort at a Threshold."""
    cut = (thr.cutoff(cols, pctile_cutoffs)
           if isinstance(thr, Threshold) else thr)
    return mat[cols].to_numpy() > cut


def _metric_vs_x(mat, cohorts, thr, value_fn, ylabel, slug_base, *,
                   pctile_cutoffs=None, exclude=frozenset(), slug_suffix="",
                   drop_covered_parents=False, xaxis=None, log_y=False,
                   collapse_subtypes=False, color_by=None):
    """Generic scatter of a per-cohort per-sample CTA metric vs an x-axis metric
    (``xaxis``, default median TMB log-x; pass :func:`_apd1_axis` for anti-PD-1
    ORR), styled like :func:`_cta_vs_x` but with a free (auto) y-axis and no
    sweet-spot band. ``value_fn(mat, cols, thr, pctile_cutoffs) -> float`` is the
    cohort's metric (e.g. mean CTAs/sample, mean CTA-specific-9mer load).

    ``log_y=True`` plots the y-axis on a log scale (useful when the metric spans
    orders of magnitude, e.g. CTA-specific-9mer load runs ~15→1300) and adds a
    ``_logy`` filename suffix. The figure is written to
    ``{slug_base}_vs_{xaxis.short}/{thr.slug}{suffix}[_logy].png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xaxis = xaxis or _tmb_axis()
    plan = _select_codes(cohorts, xaxis.values, exclude, collapse_subtypes)
    pts = []
    for code, val in plan:
        cols = [s for s in cohorts[code] if s in mat.columns]
        if not cols:
            continue
        pts.append((code, len(cols), value_fn(mat, cols, thr, pctile_cutoffs), val))
    if drop_covered_parents:
        pts = _drop_covered_parents(pts, _parent_code_map())
    if not pts:
        print(f"      no cohorts with metric + {xaxis.short}; skip", flush=True)
        return
    xs = [p[3] for p in pts]
    ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    if xaxis.log:
        ax.set_xscale("log")
    ax.set_xlim(*_xlim(xs, xaxis))
    if log_y:
        ax.set_yscale("log")
        pos = [y for y in ys if y > 0]
        ax.set_ylim((min(pos) * 0.7) if pos else 1.0, max(ys) * 1.5)
    else:
        ax.set_ylim(0, max(ys) * 1.08)
    _scatter_points(fig, ax, xs, ys, pts, color_by)
    ax.set_xlabel(xaxis.label)
    ax.set_ylabel(ylabel)
    if xaxis.pct:                            # aPD-1 / burden x-axis is a percentage
        _pct_axis(ax, "x")
    ax.grid(alpha=0.3, which="both")
    texts = [ax.text(val, y, _display_code(code), fontsize=6)
             for code, n, y, val in pts]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    except ImportError:
        for t, (code, n, y, val) in zip(texts, pts):
            t.set_fontsize(5)
            t.set_position((val * 1.02 if xaxis.log else val + 0.5, y))
    ex_note = f"; excludes {', '.join(sorted(exclude))}" if exclude else ""
    log_note = "; log y" if log_y else ""
    col_note = "; collapsed subtypes" if collapse_subtypes else ""
    ax.set_title(f"{ylabel} vs {xaxis.title} by cancer type "
                 f"({thr.xlabel}, n={len(pts)} cohorts{ex_note}{log_note}{col_note})",
                 fontsize=10)
    fig.tight_layout()
    slug = f"{slug_base}_vs_{xaxis.short}"
    if color_by is not None:
        slug += f"_colorby_{color_by.short}"
    name = f"{thr.slug}{slug_suffix}{'_logy' if log_y else ''}"
    fig.savefig(_out_path(slug, name), dpi=300)
    plt.close(fig)
    print(f"      {slug}: {len(pts)} cohorts plotted{ex_note}", flush=True)


def _tmb_vs_apd1():
    """Cohort-level context scatter: median TMB (log-x) vs anti-PD-1 monotherapy
    ORR (y), one point per cancer type with both curated. Shows the expected
    TMB->checkpoint-response gradient and flags off-diagonal cancers (high TMB
    yet low ORR, or low TMB yet responsive). Threshold-independent — emitted
    once per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmb, apd1 = _tmb_axis().values, _apd1_axis().values
    pts = [(c, 0, apd1[c], tmb[c]) for c in apd1 if c in tmb]  # (code, n, y, x)
    if not pts:
        return
    xs = [p[3] for p in pts]
    ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xscale("log")
    ax.set_xlim(*_xlim(xs, _tmb_axis()))
    ax.set_ylim(0, max(ys) * 1.08)
    _scatter_points(fig, ax, xs, ys, pts, None)
    ax.set_xlabel(_tmb_axis().label)
    ax.set_ylabel(_apd1_axis().label)
    _pct_axis(ax, "y")                       # aPD-1 ORR is a percentage
    ax.grid(alpha=0.3, which="both")
    texts = [ax.text(x, y, _display_code(c), fontsize=6) for c, n, y, x in pts]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4))
    except ImportError:
        pass
    ax.set_title(f"anti-PD-1 ORR vs TMB by cancer type (n={len(pts)} cohorts)",
                 fontsize=10)
    fig.tight_layout()
    # single-plot output: write flat (apd1_vs_tmb.png), no one-file folder
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "apd1_vs_tmb.png", dpi=300)
    plt.close(fig)
    print(f"      apd1_vs_tmb: {len(pts)} cohorts", flush=True)


def _mean_ctas_per_sample(mat, cols, thr, pctile_cutoffs):
    on = _cohort_on_matrix(mat, cols, thr, pctile_cutoffs)
    return float(on.sum(axis=0).mean())


def _mean_total_cta_tpm(mat, cols, thr, pctile_cutoffs):
    """Mean over patients of the summed TPM across the CTAs that are ON at the
    threshold — the total CTA transcriptional burden per sample. Threshold-aware
    like the other load metrics (so it gets t25/t50/p80/p90/p95 variants): only
    CTAs above the cutoff contribute their TPM. Spans orders of magnitude, so
    plotted log-y."""
    on = _cohort_on_matrix(mat, cols, thr, pctile_cutoffs)
    return float((mat[cols].to_numpy() * on).sum(axis=0).mean())


def _mean_specific_9mer_load(weights):
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

    sub = FIGDIR / "cta_coverage" / thr.slug
    sub.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for code in _tqdm(sorted(cohorts), f"coverage curves {thr.slug}"):
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
        ax.set_ylabel(f"{_display_code(code)} patients with ≥1 CTA {thr.xlabel}")
        ax.set_title(f"{_display_code(code)}: cumulative patient coverage "
                     f"({thr.xlabel}, n={n})", fontsize=10)
        ax.set_xlim(0, len(cum) + 1)  # go to this cohort's full plateau
        ax.set_ylim(0, 100)
        _pct_axis(ax, "y")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(sub / f"cta_coverage_{code}.png", dpi=300)
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
    fig.savefig(_out_path("cta_patient_count_heatmap", f"t{threshold}"), dpi=300)
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
    fig.savefig(_out_path("cta_patient_pct_heatmap", f"t{threshold}"), dpi=300)
    plt.close(fig)

    # ---- (3) sorted %-expressing bar for the focus cohort ----
    fc = counts[counts.cancer_code == focus].sort_values(pcol, ascending=False)
    fc = fc[fc[pcol] > 0].head(35)
    if len(fc):
        fig, ax = plt.subplots(figsize=(10, max(4, len(fc) * 0.28)))
        ax.barh(fc["Symbol"], fc[pcol], color="#b5179e")
        ax.invert_yaxis()
        ax.set_xlabel(f"{_display_code(focus)} patients > {threshold} TPM")
        _pct_axis(ax, "x")
        n = int(fc["n_samples"].iloc[0])
        ax.set_title(f"{_display_code(focus)}: CTA expression breadth "
                     f"(> {threshold} TPM, n={n})", fontsize=10)
        for y, (p, k) in enumerate(zip(fc[pcol], fc[tcol])):
            ax.text(p, y, f" {k}", va="center", fontsize=6)
        fig.tight_layout()
        fig.savefig(_out_path("cta_pct_bar", f"{focus}_t{threshold}"), dpi=300)
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
    ax.set_ylabel(f"Patients with ≥1 CTA > {threshold} TPM")
    ax.set_title(f"CTA panel coverage by cancer type "
                 f"(> {threshold} TPM; top {len(keep)} cohorts)",
                 fontsize=10)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 100)
    _pct_axis(ax, "y")
    ax.legend(fontsize=6, ncol=3, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_out_path("cta_coverage_curves", f"t{threshold}"), dpi=300)
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
        fig.savefig(_out_path("cta_coverage_curves", f"{focus}_t{threshold}"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
