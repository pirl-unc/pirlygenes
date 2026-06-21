"""Shared per-cohort data assembly for the anti-PD-1 causal-factor analyses.

Used by ``exclusion_vs_apd1.py`` (gene screen + exclusion-panel refinement) and
``apd1_causal_factors.py`` (multi-factor model). One place builds the
cohort x gene expression matrix and the per-cohort factor table so the two
scripts can't drift.

Granularity rule: the anti-PD-1 response table carries molecular subtypes where
they matter (UCEC_MSI/CNL/CNH, COAD_MSI/MSS, BRCA_Basal, HNSC_HPVpos/neg) and
coarse codes elsewhere, but clinical anchors with colorectal source scope are
pooled to CRC tiers before plotting. For example, KEYNOTE-177 TMB/ORR rows are
CRC_MSI anchors, not independent COAD_MSI and READ_MSI estimates. Measured
feature axes (expression/CTA) can still use organ/subtype rows where real
per-cohort data exist, and those rows are pooled separately in the matrix layer.

The reference expression matrix only has *bulk* UCEC, so UCEC is split into its
four TCGA molecular subtypes from the per-sample TCGA-UCEC parquet + the
cBioPortal SUBTYPE map (same artifacts the CTA plots use). If those per-sample
artifacts are absent we fall back to bulk UCEC + UCS aggregate (no subtype
split). Fine-grained subtypes that lack an aPD1 ORR simply don't join and drop
out, so e.g. SARC rolls up to whatever coarse code has a response number (none,
here).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import cancer_reference_expression
from pirlygenes.expression.protein_groups import fold_to_cdna_canonical_symbol
from pirlygenes.gene_sets_cancer import CTA_gene_names, cancer_subtype_group
from pirlygenes.load_dataset import get_data


def zscore(s: pd.Series) -> pd.Series:
    """Z-score a series across cohorts (population std, ddof=0). The one
    z-score the aPD1/ICI analyses share."""
    return (s - s.mean()) / s.std(ddof=0)


def signature_score(mat: pd.DataFrame, genes) -> pd.Series:
    """Per-cohort mean z-scored log-expression over a gene signature, restricted
    to its proteoform-folded members present in the cohort×gene matrix; NaN where
    none are present. The single place the analyses score a gene set."""
    present = [g for g in fold_to_cdna_canonical_symbol(genes) if g in mat.columns]
    if not present:
        return pd.Series(np.nan, index=mat.index)
    return mat[present].apply(zscore).mean(axis=1)

_EXPR_CACHE = (Path.home() / ".cache" / "pirlygenes" / "expression"
               / "treehouse-polya-25-01")
_DERIVED = _EXPR_CACHE / "derived"
_UCEC_PARQUET = _DERIVED / "tcga_ucec_per_sample_tpm.parquet"
_UCEC_SUBTYPE_MAP = _DERIVED / "cbioportal_ucec_subtype.csv"
# cBioPortal SUBTYPE -> aPD1-table code
_UCEC_RECODE = {
    "UCEC_POLE": "UCEC_POLE", "UCEC_MSI": "UCEC_MSI",
    "UCEC_CN_LOW": "UCEC_CNL", "UCEC_CN_HIGH": "UCEC_CNH",
}


# Rectum is curated as colorectal (READ inherits COAD's aPD1 ORR + TMB at every
# tier), so COAD/READ are NOT independent clinical anchors. We pool the
# microsatellite subtypes COAD/READ x {MSI,MSS} -> CRC_MSI / CRC_MSS (averaging
# the COAD+READ expression shards, as TCGA's COADREAD does) and DROP the bulk
# all-comer COAD/READ entirely (the bulk ORR is just a mixture of its MSI/MSS
# components, which we already model separately).
# Derived from the registry hierarchy (CRC -> COAD/READ) + the MSI/MSS
# cross-cutting groupings (cancer-subtype-groupings.csv), so the pool is a
# single source of truth, not a hardcoded list:
#   {COAD_MSI: CRC_MSI, READ_MSI: CRC_MSI, COAD_MSS: CRC_MSS, READ_MSS: CRC_MSS}
_CRC_TIER_MEMBERS = {
    f"CRC_{grp}": cancer_subtype_group(grp, under="CRC")
    for grp in ("MSI", "MSS")
}
_COLORECTAL_POOL = {
    member: tier
    for tier, members in _CRC_TIER_MEMBERS.items()
    for member in members
}

# Full colorectal re-key map shared by EVERY aPD1 plot (causal-factors here +
# response bars in apd1_response_plots): the registry-derived COAD/READ x
# {MSI,MSS} -> CRC_MSI/CRC_MSS pool plus bulk COAD/READ -> CRC. Exported so the
# plots import this one source of truth instead of each hardcoding the map.
CRC_POOL = {**_COLORECTAL_POOL, "COAD": "CRC", "READ": "CRC"}

# A bulk all-comer code whose finer clinical anchors carry distinct aPD1
# ORR/TMB is a *mixture* of those anchors (like bulk COAD/READ over MSI/MSS).
# When every listed subtype is present in the cohort x gene matrix, drop the
# bulk row so it is not counted ~Nx alongside its near-duplicate subtype rows
# in the cross-cohort Spearman/OLS. HNSC_HPVpos/HNSC_HPVneg respond very
# differently to anti-PD-1, so the HPV split is the right grain; bulk HNSC is
# the mixture. (UCEC's bulk is already replaced by its subtypes upstream.)
_BULK_DROP_IF_SUBTYPES = {"HNSC": ("HNSC_HPVpos", "HNSC_HPVneg")}
# analysis-only pooled codes (NOT registry codes) - must be kept out of any
# cancer_reference_expression fetch.
_POOLED_CODES = set(_COLORECTAL_POOL.values()) | {"CRC"}
_CRC_TIERS = _POOLED_CODES


def pool_colorectal_axis(d: dict, *, keep_source_codes: bool) -> dict:
    """Return a copy with COAD/READ source rows pooled into CRC tiers.

    Use ``keep_source_codes=False`` for clinical/source-scope axes such as
    aPD1/ICI ORR, TMB, and mechanistic indel class: those curated COAD/READ
    subtype rows are CRC-level anchors copied for lookup convenience, so plots
    should expose only ``CRC`` / ``CRC_MSI`` / ``CRC_MSS``. Use
    ``keep_source_codes=True`` only for measured feature axes where the
    organ/subtype rows remain valid independent measurements.
    """
    out = dict(d)
    groups: dict = {}
    for src, tgt in CRC_POOL.items():
        if src in out and pd.notna(out[src]):
            groups.setdefault(tgt, []).append(out[src])
    for tgt, vals in groups.items():
        out[tgt] = sum(vals) / len(vals)
    if not keep_source_codes:
        for src in CRC_POOL:
            out.pop(src, None)
    return out


def _clinical_anchor_map(d: dict) -> dict:
    """Pool CRC-scoped clinical anchors and hide the source split rows."""
    return pool_colorectal_axis(d, keep_source_codes=False)


def apd1_map() -> dict[str, float]:
    """``{cancer_code: aPD1 ORR %}`` from the curated response table.

    Colorectal ORR rows are source-scoped as CRC-level clinical anchors, so the
    returned analysis map exposes ``CRC`` / ``CRC_MSI`` / ``CRC_MSS`` rather
    than independent COAD/READ source rows.
    """
    df = get_data("cancer-apd1-response.csv")
    return _clinical_anchor_map(
        dict(zip(df["cancer_code"], df["apd1_orr_pct"].astype(float))))


_VIRAL_SCORE = {"defining": 1.0, "subset": 0.5, "none": 0.0}


def with_parent(d: dict, code: str, default=None):
    """Look up ``code`` in ``d``, falling back to its base code (before the
    first ``_``) — e.g. ``COAD_MSI`` -> ``COAD`` for a coarse-grained value."""
    for k in (code, code.split("_")[0]):
        if k in d and pd.notna(d[k]):
            return d[k]
    return default


def tmb_map() -> dict[str, float]:
    """``{cancer_code: median TMB mut/Mb}`` from ``cancer-tmb.csv``.

    CRC-scoped MSI-H TMB values are exposed as ``CRC_MSI`` only. Scripts that
    need a CRC_MSS TMB use the existing ``CRC`` fallback via :func:`with_parent`.
    """
    df = get_data("cancer-tmb.csv")
    return _clinical_anchor_map(dict(zip(df["cancer_code"], df["median_tmb_mut_mb"])))


def indel_map() -> dict[str, float]:
    """``{cancer_code: ordinal frameshift/indel-enrichment score (0/1/2)}``.

    A mechanistic class, NOT a measured per-Mb value: high = RCC lineage
    (Turajlic 2017) + dMMR/MSI-H. See ``cancer-frameshift-burden.csv``."""
    df = get_data("cancer-frameshift-burden.csv")
    return _clinical_anchor_map(
        dict(zip(df["cancer_code"], df["indel_score"].astype(float))))


def viral_score(code: str, reg) -> float:
    """Ordinal viral-antigen score from the registry ``viral_etiology``
    (defining=1.0 / subset=0.5 / none=0.0), with base-code fallback."""
    for k in (code, code.split("_")[0]):
        if k in reg.index:
            return _VIRAL_SCORE.get(str(reg.loc[k, "viral_etiology"]), 0.0)
    return 0.0


def _ucec_subtype_tpm(genes=None) -> pd.DataFrame | None:
    """Per-subtype median TPM (rows = UCEC_* codes, cols = Symbol) from the
    per-sample TCGA-UCEC parquet split by the cBioPortal molecular class.
    Returns ``None`` if the per-sample artifacts are unavailable."""
    if not (_UCEC_PARQUET.exists() and _UCEC_SUBTYPE_MAP.exists()):
        return None
    mat = pd.read_parquet(_UCEC_PARQUET)
    smap = pd.read_csv(_UCEC_SUBTYPE_MAP)
    sub = dict(zip(smap["patientId"], smap["ucec_subtype"]))
    if genes is not None:
        mat = mat[mat["Symbol"].isin(set(genes))]
    sample_cols = [c for c in mat.columns if c.startswith("TCGA-")]
    # sample "TCGA-2E-A9G8-01" -> patientId "TCGA-2E-A9G8"
    col_sub = {c: _UCEC_RECODE.get(sub.get("-".join(c.split("-")[:3])))
               for c in sample_cols}
    out = {}
    for code in set(v for v in col_sub.values() if v):
        cols = [c for c in sample_cols if col_sub[c] == code]
        if cols:
            out[code] = mat.set_index("Symbol")[cols].median(axis=1)
    if not out:
        return None
    return pd.DataFrame(out).T  # rows=codes, cols=Symbol


def cohort_gene_matrix(codes, *, ucec_subtypes: bool = True) -> pd.DataFrame:
    """cohort (cancer_code) x gene matrix of ``log10(TPM+1)``.

    Richest-source-wins per code, RNA-seq only (microarray-proxy cohorts
    dropped). When ``ucec_subtypes`` and the per-sample artifacts exist, bulk
    UCEC is replaced by its four molecular subtypes; otherwise bulk UCEC and
    UCS are kept as-is.

    Caveat: the UCEC subtype rows are summarised from the per-sample TCGA-UCEC
    parquet, a *different* pipeline than the reference-summary cohorts, so their
    absolute TPM is not strictly cross-pipeline comparable. Downstream analyses
    z-score each gene across cohorts, so this affects cohort *ranking* only
    marginally; do not compare raw subtype TPM to other cohorts' raw TPM.
    """
    # CRC_* are analysis-only pooled codes (not registry codes); when a caller
    # asks for CRC_MSI/CRC_MSS, fetch the real COAD/READ subtype shards and pool
    # them below. Do not require clinical ORR/TMB maps to expose those source
    # split rows merely to make expression fetching work.
    crc_sources = [
        member
        for tier, members in _CRC_TIER_MEMBERS.items()
        if tier in codes
        for member in members
    ]
    fetch = list(dict.fromkeys(
        [c for c in codes if c not in _CRC_TIERS] + crc_sources + ["UCEC", "UCS"]))
    # collapse_cdna_identical: sum cDNA-identical loci (+ curated overrides like
    # CT47A) CENTRALLY in the accessor — the universal read-recovery collapse,
    # applied consistently everywhere. cDNA-distinct paralogs (MAGEA3 vs MAGEA6,
    # histone clusters) stay separate; exclusion/antigen/signature genes are
    # single-copy (audited), so unaffected.
    long = cancer_reference_expression(cancer_types=fetch, normalize="tpm_clean",
                                       collapse_cdna_identical=True)
    long = long[~long["processing_pipeline"].str.contains(
        "microarray_tpm_proxy", na=False)]
    src_n = (long.groupby(["cancer_code", "source_cohort"])["n_samples"]
             .max().reset_index())
    best = src_n.sort_values("n_samples").groupby("cancer_code").tail(1)
    keep = set(zip(best["cancer_code"], best["source_cohort"]))
    long = long[[(c, s) in keep for c, s in
                 zip(long["cancer_code"], long["source_cohort"])]]
    wide = long.pivot_table(index="cancer_code", columns="Symbol",
                            values="expression", aggfunc="max")  # TPM scale
    sub = _ucec_subtype_tpm() if ucec_subtypes else None
    if sub is not None:
        wide = wide.drop(index="UCEC", errors="ignore")  # replace bulk w/ split
        wide = pd.concat([wide, sub.reindex(columns=wide.columns)])
    # pool colorectal: COAD/READ shards -> CRC tiers (mean), de-duplicating the
    # shared clinical anchors (see _COLORECTAL_POOL).
    pooled_idx = [_COLORECTAL_POOL.get(c, c) for c in wide.index]
    wide = wide.groupby(pooled_idx).mean()
    # drop bulk all-comer rows whose distinct subtype anchors are all present,
    # so the mixture is not double/triple-counted (see _BULK_DROP_IF_SUBTYPES).
    for bulk, subs in _BULK_DROP_IF_SUBTYPES.items():
        if bulk in wide.index and all(s in wide.index for s in subs):
            wide = wide.drop(index=bulk)
    return np.log10(wide + 1.0)


# "ON" threshold for a cancer-testis antigen protein: summed linear TPM >= 6.
CTA_ON_TPM = 6.0


@lru_cache(maxsize=1)
def _cta_coverage_p95() -> dict:
    """``{cancer_code: %% patients with >=1 CTA at the >=95th within-sample
    percentile}`` from cta_patient_counts' stable ``cta_union_counts.csv``. This
    is the DEFAULT CTA-burden metric. Empty dict if the table hasn't been
    generated (run ``cta_patient_counts.py`` once)."""
    path = Path(__file__).resolve().parent / "outputs" / "_cta_union_counts.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(c): 100.0 * a / n
            for c, a, n in zip(df["cancer_code"], df["n_any_p95"],
                               df["n_samples"]) if n}


def _latest_cta_patient_counts_path() -> Path | None:
    """Stable per-CTA patient-count table, falling back to the newest run dir.

    ``cta_patient_counts.py`` historically only wrote the detailed table into
    each timestamped run. Newer runs also write ``_cta_patient_counts.csv`` next
    to ``_cta_union_counts.csv``; keep the fallback so older regenerated outputs
    still support CTA-load comparisons.
    """
    out = Path(__file__).resolve().parent / "outputs"
    stable = out / "_cta_patient_counts.csv"
    if stable.exists():
        return stable
    runs = sorted(out.glob("run_*/cta_patient_counts.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


@lru_cache(maxsize=1)
def cta_metric_table() -> pd.DataFrame:
    """Per-cohort CTA metrics from the generated per-sample CTA tables.

    Columns are raw values, not z-scores:

    * ``cta_coverage_p90/p95``: percent of patients with >=1 CTA on.
    * ``cta_count_p90/p95``: mean active CTA proteins per patient.
    * ``cta_9mer_load_p90/p95``: mean CTA-specific 9-mer payload per patient.

    Empty/NaN columns are returned when the prerequisite generated tables are
    absent; callers can still render the non-CTA factors.
    """
    union_path = Path(__file__).resolve().parent / "outputs" / "_cta_union_counts.csv"
    counts_path = _latest_cta_patient_counts_path()
    out = pd.DataFrame()

    if union_path.exists():
        u = pd.read_csv(union_path)
        u = u[u["n_samples"].astype(float) > 0].copy()
        u["cancer_code"] = u["cancer_code"].astype(str)
        out = out.reindex(u["cancer_code"])
        for q in (90, 95):
            out[f"cta_coverage_p{q}"] = (
                100.0 * u[f"n_any_p{q}"].astype(float).to_numpy()
                / u["n_samples"].astype(float).to_numpy()
            )

    if counts_path is None:
        return out

    counts = pd.read_csv(counts_path)
    if counts.empty:
        return out
    counts["cancer_code"] = counts["cancer_code"].astype(str)
    n_samples = counts.groupby("cancer_code")["n_samples"].first().astype(float)
    out = out.reindex(out.index.union(n_samples.index))

    for q in (90, 95):
        col = f"n_p{q}"
        if col in counts.columns:
            out[f"cta_count_p{q}"] = (
                counts.groupby("cancer_code")[col].sum().astype(float)
                / n_samples
            )

    spec_path = Path(__file__).resolve().parent / "outputs" / "_cache" / "cta_specific_9mers.csv"
    if not spec_path.exists():
        return out

    spec = pd.read_csv(spec_path)
    sym2spec = dict(zip(spec["Symbol"].astype(str),
                        spec["n_specific_9mers"].astype(float)))
    try:
        groups = get_data("cta-protein-groups")
        for group, members in groups.groupby("protein_group"):
            weights = [sym2spec.get(str(m), 0.0)
                       for m in members["member_symbol"].astype(str)]
            if weights:
                sym2spec.setdefault(str(group), max(weights))
    except Exception:
        pass

    counts = counts.assign(
        _specific_9mers=counts["Symbol"].astype(str).map(sym2spec).fillna(0.0)
    )
    for q in (90, 95):
        col = f"n_p{q}"
        if col in counts.columns:
            weighted = (
                counts.assign(_weighted=counts[col].astype(float)
                              * counts["_specific_9mers"].astype(float))
                .groupby("cancer_code")["_weighted"].sum()
            )
            out[f"cta_9mer_load_p{q}"] = weighted / n_samples
    return out


def cta_burden(mat: pd.DataFrame, *, thr_tpm: float = CTA_ON_TPM,
               min_coverage: float = 0.5) -> pd.Series:
    """CTA burden = **% of patients in the cohort with >=1 CTA at the >=95th
    within-sample percentile** (the per-sample coverage from cta_patient_counts,
    proteoform-collapsed before ranking). Cohorts absent from that table (not in
    the poly-A compendium, e.g. NPC/SCLC) get ``NaN``.

    Falls back to a cohort-summary ON-antigen count (the matrix is already
    cDNA-collapsed; CTA panel folded onto canonical symbols) only when the
    per-sample coverage table hasn't been generated, so the analysis still runs.
    """
    cov = _cta_coverage_p95()
    if cov:
        return pd.Series({c: cov.get(c, np.nan) for c in mat.index})
    # fallback (no coverage table): ON-antigen count over the cohort summary
    genes = [g for g in fold_to_cdna_canonical_symbol(CTA_gene_names())
             if g in mat.columns]
    if not genes:
        return pd.Series(np.nan, index=mat.index)
    lin = np.power(10.0, mat[genes]) - 1.0
    measured = lin.notna()
    on = (lin >= thr_tpm) & measured
    n_meas = measured.sum(axis=1)
    rate = on.sum(axis=1) / n_meas.replace(0, np.nan)
    rate[n_meas < min_coverage * len(genes)] = np.nan
    return rate * n_meas.median()


def curated_exclusion_genes() -> dict[str, list[str]]:
    """The curated aPD1-exclusion signatures, from
    ``therapy-response-signatures.csv`` (single source of truth)."""
    sigs = get_data("therapy-response-signatures.csv")
    return {
        cls.replace("aPD1_exclusion_", ""): grp["symbol"].tolist()
        for cls, grp in sigs.groupby("therapy_class")
        if cls.startswith("aPD1_exclusion_")
    }


# axis + expected response-direction + circularity tag, keyed by therapy_class.
# axis: antigen (drives response UP) / exclusion (DOWN) / circular (outcome).
SIGNATURE_META = {
    "aPD1_antigen_presentation":   ("antigen",   +1, "borderline"),
    "aPD1_exclusion_TGFb_response": ("exclusion", -1, "causal"),
    "aPD1_exclusion_Wnt":          ("exclusion", -1, "causal"),
    "aPD1_exclusion_Wnt_target":   ("exclusion", -1, "causal"),
    "aPD1_exclusion_angiogenesis": ("exclusion", -1, "causal"),
    "aPD1_exclusion_adenosine":    ("exclusion", -1, "causal"),
    "aPD1_circular_checkpoints":   ("circular",  +1, "circular"),
    "aPD1_circular_treg":          ("circular",  +1, "circular"),
    "IFN_response":                ("circular",  +1, "circular"),
}


def curated_signatures() -> dict[str, list[str]]:
    """All curated aPD1 pathway signatures (+ IFN_response), as
    ``{therapy_class: [symbols]}``, from ``therapy-response-signatures.csv``."""
    sigs = get_data("therapy-response-signatures.csv")
    keep = set(SIGNATURE_META)
    return {
        cls: grp["symbol"].tolist()
        for cls, grp in sigs.groupby("therapy_class")
        if cls in keep
    }
