"""Shared per-cohort data assembly for the anti-PD-1 causal-factor analyses.

Used by ``exclusion_vs_apd1.py`` (gene screen + exclusion-panel refinement) and
``apd1_causal_factors.py`` (multi-factor model). One place builds the
cohort x gene expression matrix and the per-cohort factor table so the two
scripts can't drift.

Granularity rule: cohorts are matched to the anti-PD-1 response table
(``cancer-apd1-response.csv``), which defines the analysis grain — it carries
molecular subtypes where they matter (UCEC_MSI/CNL/CNH, COAD_MSI/MSS,
BRCA_Basal, HNSC_HPVpos/neg) and coarse codes elsewhere. The reference
expression matrix only has *bulk* UCEC, so UCEC is split into its four TCGA
molecular subtypes from the per-sample TCGA-UCEC parquet + the cBioPortal
SUBTYPE map (same artifacts the CTA plots use). If those per-sample artifacts
are absent we fall back to bulk UCEC + UCS aggregate (no subtype split).
Fine-grained subtypes that lack an aPD1 ORR simply don't join and drop out, so
e.g. SARC rolls up to whatever coarse code has a response number (none, here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes.expression.accessors import cancer_reference_expression
from pirlygenes.gene_sets_cancer import (CTA_gene_names, cancer_subtype_group,
                                         cta_paralog_symbols)
from pirlygenes.load_dataset import get_data

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
_COLORECTAL_POOL = {
    member: f"CRC_{grp}"
    for grp in ("MSI", "MSS")
    for member in cancer_subtype_group(grp, under="CRC")
}
_COLORECTAL_DROP = ("COAD", "READ")  # bulk all-comer = MSI/MSS mixture

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


def _pool_dict(d: dict) -> dict:
    """Pool COAD/READ subtypes into CRC_MSI/CRC_MSS (mean), and the bulk
    all-comer COAD/READ into a CRC fallback value, then drop the bulk keys.
    CRC is a *fallback* (e.g. CRC_MSS TMB has no explicit row, so it resolves to
    base CRC), not a modeled cohort (the matrix only carries CRC_MSI/CRC_MSS)."""
    pool = {**_COLORECTAL_POOL, "COAD": "CRC", "READ": "CRC"}
    groups: dict = {}
    for src, tgt in pool.items():
        if src in d and pd.notna(d[src]):
            groups.setdefault(tgt, []).append(d[src])
    for tgt, vals in groups.items():
        d[tgt] = sum(vals) / len(vals)
    for bulk in _COLORECTAL_DROP:
        d.pop(bulk, None)
    return d


def apd1_map() -> dict[str, float]:
    """``{cancer_code: aPD1 ORR %}`` from the curated response table (colorectal
    pooled to CRC tiers)."""
    df = get_data("cancer-apd1-response.csv")
    return _pool_dict(dict(zip(df["cancer_code"], df["apd1_orr_pct"].astype(float))))


_VIRAL_SCORE = {"defining": 1.0, "subset": 0.5, "none": 0.0}


def with_parent(d: dict, code: str, default=None):
    """Look up ``code`` in ``d``, falling back to its base code (before the
    first ``_``) — e.g. ``COAD_MSI`` -> ``COAD`` for a coarse-grained value."""
    for k in (code, code.split("_")[0]):
        if k in d and pd.notna(d[k]):
            return d[k]
    return default


def tmb_map() -> dict[str, float]:
    """``{cancer_code: median TMB mut/Mb}`` from ``cancer-tmb.csv`` (CRC pooled)."""
    df = get_data("cancer-tmb.csv")
    return _pool_dict(dict(zip(df["cancer_code"], df["median_tmb_mut_mb"])))


def indel_map() -> dict[str, float]:
    """``{cancer_code: ordinal frameshift/indel-enrichment score (0/1/2)}``.

    A mechanistic class, NOT a measured per-Mb value: high = RCC lineage
    (Turajlic 2017) + dMMR/MSI-H. See ``cancer-frameshift-burden.csv``."""
    df = get_data("cancer-frameshift-burden.csv")
    return _pool_dict(dict(zip(df["cancer_code"], df["indel_score"].astype(float))))


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
    # CRC_* are analysis-only pooled codes (not registry codes); fetch the real
    # COAD/READ subtype shards and pool them after.
    fetch = list(dict.fromkeys(
        [c for c in codes if c not in _CRC_TIERS] + ["UCEC", "UCS"]))
    # NB: protein-identical paralogs are NOT collapsed here (that would also sum
    # housekeeping clusters like histones). CTA paralog summing is scoped to the
    # CTA panel inside cta_burden via the curated cta-protein-groups; nothing
    # else collapses. Exclusion/antigen/signature genes are single-copy (audited).
    long = cancer_reference_expression(cancer_types=fetch, normalize="tpm_clean")
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


def _cta_protein_groups_in(mat: pd.DataFrame) -> dict[str, list[str]]:
    """``{representative: [member symbols present in mat]}`` over the curated CTA
    protein groups (``cta-protein-groups``, identical/near-identical >=90% aa —
    NY-ESO-1 = CTAG1A+CTAG1B, MAGEA3+MAGEA6, the CT47A cluster, …). CTA-scoped:
    only CTA antigens group; nothing else (histones etc.) is touched."""
    groups, seen = {}, set()
    for sym in CTA_gene_names():
        if sym in seen:
            continue
        members = cta_paralog_symbols(sym) or [sym]
        seen.update(members)
        present = [m for m in members if m in mat.columns]
        if present:
            groups[sorted(members)[0]] = present
    return groups


def cta_burden(mat: pd.DataFrame, *, thr_tpm: float = CTA_ON_TPM,
               min_coverage: float = 0.5) -> pd.Series:
    """Coverage-aware count of "ON" cancer-testis-antigen **proteins** per cohort.

    A multi-locus antigen (the CT47A cluster, MAGEA3/MAGEA6, NY-ESO CTAG1A/B, …)
    is summed in **linear TPM** across its curated protein-group members so it
    counts as ONE protein, not N near-identical paralogs whose reads were split.
    The grouping is CTA-scoped (``cta-protein-groups``) — only antigens collapse,
    never housekeeping clusters (histones etc.).

    Respects *missing != zero*: a CTA group a cohort never measured (no member
    present) is excluded from both the ON count and the denominator. We take the
    ON-*rate* among the antigen groups a cohort actually measured, rescaled to
    the median measured-group count so it stays count-like for display/z-scoring;
    a cohort below ``min_coverage`` of the groups gets ``NaN``.
    """
    groups = _cta_protein_groups_in(mat)
    if not groups:
        return pd.Series(np.nan, index=mat.index)
    lin = np.power(10.0, mat) - 1.0  # de-log to linear TPM before summing loci
    on = pd.DataFrame(index=mat.index)
    measured = pd.DataFrame(index=mat.index)
    for key, members in groups.items():
        summed = lin[members].sum(axis=1, min_count=1)   # NaN members ignored
        measured[key] = summed.notna()
        on[key] = summed >= thr_tpm
    n_meas = measured.sum(axis=1)
    n_on = (on & measured).sum(axis=1)
    rate = n_on / n_meas.replace(0, np.nan)
    rate[n_meas < min_coverage * len(groups)] = np.nan
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
