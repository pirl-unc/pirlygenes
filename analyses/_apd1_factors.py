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


def apd1_map() -> dict[str, float]:
    """``{cancer_code: aPD1 ORR %}`` from the curated response table."""
    df = get_data("cancer-apd1-response.csv")
    return dict(zip(df["cancer_code"], df["apd1_orr_pct"].astype(float)))


_VIRAL_SCORE = {"defining": 1.0, "subset": 0.5, "none": 0.0}


def with_parent(d: dict, code: str, default=None):
    """Look up ``code`` in ``d``, falling back to its base code (before the
    first ``_``) — e.g. ``COAD_MSI`` -> ``COAD`` for a coarse-grained value."""
    import pandas as pd  # local: keep top-level imports lean
    for k in (code, code.split("_")[0]):
        if k in d and pd.notna(d[k]):
            return d[k]
    return default


def tmb_map() -> dict[str, float]:
    """``{cancer_code: median TMB mut/Mb}`` from ``cancer-tmb.csv``."""
    df = get_data("cancer-tmb.csv")
    return dict(zip(df["cancer_code"], df["median_tmb_mut_mb"]))


def indel_map() -> dict[str, float]:
    """``{cancer_code: ordinal frameshift/indel-enrichment score (0/1/2)}``.

    A mechanistic class, NOT a measured per-Mb value: high = RCC lineage
    (Turajlic 2017) + dMMR/MSI-H. See ``cancer-frameshift-burden.csv``."""
    df = get_data("cancer-frameshift-burden.csv")
    return dict(zip(df["cancer_code"], df["indel_score"].astype(float)))


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
    """
    fetch = list(dict.fromkeys(list(codes) + ["UCEC", "UCS"]))
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
    return np.log10(wide + 1.0)


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
