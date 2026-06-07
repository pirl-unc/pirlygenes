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

    @property
    def source_kind(self) -> str:
        """Pipeline-family prefix derived from ``source_id`` (e.g.
        ``treehouse-polya-25-01`` -> ``treehouse``). Parallels the cohort
        registry ``kind`` so per-sample cohorts address the same way as the
        reference-manifest atoms."""
        return self.source_id.split("-")[0]

    @property
    def atom(self) -> str:
        """Source-prefixed cohort atom ``"<source_kind>:<code>"`` (#292), e.g.
        ``treehouse:SARC_LMS``. Lets a new pipeline (GDC, …) slot in as a
        parallel ``gdc:<code>`` without disturbing existing cohorts; a cancer
        category is the union of its source-prefixed atoms across kinds."""
        return f"{self.source_kind}:{self.code}"


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
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]
# NB: no "SARC" — the bare SARC code is the computed pan-sarcoma grand union (its
# TCGA-SARC leiomyosarcoma samples are already in the SARC_LMS atom), not a
# concrete per-sample cohort.
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


# Single source of truth for the per-sample expression sources: ``source_id ->
# (source_cohort label, project)``. The generators and the package accessors all
# read this — don't duplicate the list in scripts. ``label`` is the
# ``source_cohort`` value used in provenance; ``project`` is the data origin.
PER_SAMPLE_SOURCES: dict[str, tuple[str, str]] = {
    "treehouse-polya-25-01": ("TREEHOUSE_POLYA_25_01", "Treehouse"),
    "treehouse-ribod-25-01": ("TREEHOUSE_RIBOD_25_01", "Treehouse"),
    # Neuroendocrine axis (#318/#326) — per-sample parquets built by
    # scripts/build_ne_per_sample_parquets.py from the cached NE source data.
    "gse118014-pannet": ("GSE118014_ALVAREZ_2018", "GEO"),
    "sclc-ucologne-2015": ("SCLC_UCOLOGNE_2015", "UCologne"),
    "drmetrics-lnen-2020": ("DRMETRICS_ALCALA_2019_LNEN", "IARC LNEN"),
}

# Sources with an explicit code→Cohort map (mixed-case stems that don't
# round-trip from the parquet name). Every other source in PER_SAMPLE_SOURCES
# discovers its cohorts from disk with ``code == stem``.
_REGISTRY: dict[str, dict[str, Cohort]] = {
    "treehouse-polya-25-01": _treehouse_polya_25_01(),
}


def source_label(source_id: str) -> str | None:
    """The ``source_cohort`` provenance label for a per-sample source."""
    entry = PER_SAMPLE_SOURCES.get(source_id)
    return entry[0] if entry else None


def source_project(source_id: str) -> str | None:
    """The data-origin project for a per-sample source (Treehouse/GEO/…)."""
    entry = PER_SAMPLE_SOURCES.get(source_id)
    return entry[1] if entry else None


def _discovered_cohorts(source_id: str) -> dict[str, Cohort]:
    """Cohorts inferred from a source's cached ``derived`` parquet stems, with
    ``code == stem`` — for sources without an explicit registry map."""
    derived = downloads.source_cache_dir(source_id) / "derived"
    out: dict[str, Cohort] = {}
    if derived.exists():
        for p in sorted(derived.glob("*_per_sample_tpm.parquet")):
            stem = p.name[: -len("_per_sample_tpm.parquet")]
            out[stem] = Cohort(stem, stem, source_id)
    return out


def cohorts_for_source(source_id: str) -> dict[str, Cohort]:
    """All cohorts for ``source_id`` ({code: Cohort}). Explicit-map sources
    (treehouse-polya) return their registered map; other per-sample sources
    discover cohorts from the cached ``derived`` parquets (code == stem); an
    unknown source returns empty."""
    if source_id in _REGISTRY:
        return dict(_REGISTRY[source_id])
    if source_id in PER_SAMPLE_SOURCES:
        return _discovered_cohorts(source_id)
    return {}


def parquet_path(cohort: Cohort):
    """Path to a cohort's per-sample TPM parquet (may not exist on disk)."""
    return (downloads.source_cache_dir(cohort.source_id) / "derived"
            / f"{cohort.stem}_per_sample_tpm.parquet")


def available_cohorts(source_id: str) -> dict[str, Cohort]:
    """Registered cohorts whose per-sample parquet is actually present in the
    local cache (i.e. usable right now without a build/download)."""
    return {code: c for code, c in cohorts_for_source(source_id).items()
            if parquet_path(c).exists()}


def all_available_cohorts() -> dict[str, Cohort]:
    """Every per-sample cohort present on disk across **all**
    :data:`PER_SAMPLE_SOURCES`. On a code collision the first source in
    registration order wins (treehouse-polya is canonical)."""
    out: dict[str, Cohort] = {}
    for source_id in PER_SAMPLE_SOURCES:
        for code, cohort in available_cohorts(source_id).items():
            out.setdefault(code, cohort)
    return out
