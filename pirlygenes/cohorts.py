# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-sample cohort registry: which expression cohorts a build *source*
materialises, and how each registry cancer-type code maps to its on-disk
per-sample-TPM parquet stem.

The builders (:mod:`pirlygenes.builders.treehouse`) write one per-sample TPM
matrix per cohort to ``<source-cache>/derived/<stem>_per_sample_tpm.parquet``.
Most cohorts use their own code as the stem; the TCGA-via-Treehouse cohorts use
``tcga_<lower(code)>``; the molecular / histology sub-cohorts use mixed-case
codes that don't round-trip from a lowercased stem (e.g. ``BRCA_LumA`` ->
``tcga_brca_luma``), so that mapping is enumerated explicitly here rather than
inferred. This is the package-level home for cohort definitions that previously
lived only in ``scripts/sweep_treehouse_*`` — the cohort-level CLI
(``pirlygenes plot``) and notebooks read per-sample data through it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import downloads


@dataclass(frozen=True)
class Cohort:
    """One materialised per-sample cohort within a build source."""

    code: str        # registry cancer-type code (e.g. "GBM", "BRCA_LumA")
    stem: str        # per-sample parquet stem (filename minus _per_sample_tpm.parquet)
    source_id: str   # expression source id (downloads registry)


# --- treehouse-polya-25-01 -------------------------------------------------
# Pediatric / sarcoma / rare cohorts: parquet stem == cancer code.
_TH_DIRECT = [
    "ATRT", "SARC_EWS", "HEPB", "MBL", "NPC", "NUTM", "SARC_OS",
    "SARC_RMS_ARMS", "SARC_RMS_ERMS", "SARC_RMS_PRMS", "SARC_RMS_SSRMS",
    "SARC_ANGIO", "SARC_ASPS", "SARC_DSRCT", "SARC_EHE", "SARC_EPITH",
    "SARC_GIST", "SARC_IFS", "SARC_IMT", "SARC_LGFMS", "SARC_LMS",
    "SARC_LPS_UNSPEC", "SARC_MPNST", "SARC_MYXFIB", "SARC_SYN", "SARC_UPS",
]
# TCGA-via-Treehouse direct projects: stem == "tcga_<lower(code)>".
_TH_TCGA = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]
# Molecular / histology sub-cohorts (mixed-case codes -> explicit stems).
_TH_DERIVED = {
    "BRCA_Basal": "tcga_brca_basal", "BRCA_HER2": "tcga_brca_her2",
    "BRCA_LumA": "tcga_brca_luma", "BRCA_LumB": "tcga_brca_lumb",
    "BRCA_Normal": "tcga_brca_normal",
    "HNSC_HPVneg": "tcga_hnsc_hpv_neg", "HNSC_HPVpos": "tcga_hnsc_hpv_pos",
    "LUAD_EGFR": "tcga_luad_egfr", "LUAD_KRAS": "tcga_luad_kras",
    "LUAD_STK11": "tcga_luad_stk11",
    "SARC_DDLPS": "tcga_sarc_ddlps", "SARC_PLEOLPS": "tcga_sarc_pleolps",
    "SARC_WDLPS": "tcga_sarc_wdlps",
}


def _treehouse_polya_25_01() -> dict[str, Cohort]:
    src = "treehouse-polya-25-01"
    out: dict[str, Cohort] = {}
    for code in _TH_DIRECT:
        out[code] = Cohort(code, code, src)
    for code in _TH_TCGA:
        out[code] = Cohort(code, f"tcga_{code.lower()}", src)
    for code, stem in _TH_DERIVED.items():
        out[code] = Cohort(code, stem, src)
    return out


_REGISTRY: dict[str, dict[str, Cohort]] = {
    "treehouse-polya-25-01": _treehouse_polya_25_01(),
}


def cohorts_for_source(source_id: str) -> dict[str, Cohort]:
    """All registered cohorts for ``source_id`` ({code: Cohort}); empty if the
    source has no per-sample cohort registry."""
    return dict(_REGISTRY.get(source_id, {}))


def parquet_path(cohort: Cohort):
    """Path to a cohort's per-sample TPM parquet (may not exist on disk)."""
    return (downloads.source_cache_dir(cohort.source_id) / "derived"
            / f"{cohort.stem}_per_sample_tpm.parquet")


def available_cohorts(source_id: str) -> dict[str, Cohort]:
    """Registered cohorts whose per-sample parquet is actually present in the
    local cache (i.e. usable right now without a build/download)."""
    return {code: c for code, c in cohorts_for_source(source_id).items()
            if parquet_path(c).exists()}
