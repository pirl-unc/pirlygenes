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

import pandas as pd

from . import downloads

# Canonical per-sample-parquet layout — the single source of truth for the
# filename suffix and the non-sample id columns, so every reader/writer agrees.
PER_SAMPLE_SUFFIX = "_per_sample_tpm.parquet"
ID_COLS = ("Ensembl_Gene_ID", "Symbol")


@dataclass(frozen=True)
class Cohort:
    """One materialised per-sample cohort within a build source.

    The first three fields (``code``, ``stem``, ``source_id``) are all the
    read path needs. The remaining fields are the **build-time** definition
    that previously lived only in the ``scripts/sweep_treehouse_*`` scripts;
    hoisting them here makes this registry the single source of truth that
    *both* the read accessors and the build sweeps consume (so the two can
    never drift):

    * ``group`` — which build sweep materialises this cohort (the sweep
      filters the registry by it via :func:`cohorts_for_group`).
    * ``disease_label`` — the Treehouse clinical ``disease`` label the sweep
      filters samples on.
    * ``selection`` — a declarative subset selector the sweep turns into a
      ``sample_predicate``. Grammar (``""`` = no predicate, take every sample
      with ``disease_label``):

        ``tcga``                    TCGA-only subset (th_dataset_id startswith TCGA)
        ``gdc_project:TCGA-GBM``    TCGA-only + GDC case→project membership
        ``pam50:BRCA_Basal``        TCGA-only + cBioPortal PAM50 call
        ``hpv:HNSC_HPV+``           TCGA-only + cBioPortal HPV call
        ``mutation:STK11,KEAP1``    TCGA-only + cBioPortal mutation in ANY gene
        ``histology:Dedifferentiated liposarcoma``
                                    TCGA-only + GDC primary_diagnosis
    """

    code: str        # registry cancer-type code (e.g. "GBM", "BRCA_LumA")
    stem: str        # per-sample parquet stem (filename minus _per_sample_tpm.parquet)
    source_id: str   # expression source id (downloads registry)
    group: str = "direct"      # build sweep that materialises it
    disease_label: str = ""    # Treehouse clinical `disease` label to filter on
    selection: str = ""        # declarative subset selector (see class docstring)

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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# The single declarative registry of every Treehouse per-sample cohort —
# (code, stem, disease_label, selection) per build group. This is the one
# source of truth the read accessors AND the ``scripts/sweep_treehouse_*``
# build sweeps both consume; neither side enumerates cohorts independently, so
# they can't drift. ``group`` names the sweep that materialises the rows.
# ---------------------------------------------------------------------------

_POLYA = "treehouse-polya-25-01"
_RIBOD = "treehouse-ribod-25-01"

# Pediatric / sarcoma / rare cohorts registered as their own Treehouse disease
# label; parquet stem == code, no sample predicate (whole disease label).
# group "polya_pediatric" -> scripts/sweep_treehouse_polya_cohorts.py
_POLYA_PEDIATRIC = [
    ("ATRT", "atypical teratoid/rhabdoid tumor"),
    ("SARC_EWS", "Ewing sarcoma"),
    ("HEPB", "hepatoblastoma"),
    ("MBL", "medulloblastoma"),
    ("NUTM", "NUT midline carcinoma"),
    ("SARC_OS", "osteosarcoma"),
    ("SARC_RMS_ARMS", "alveolar rhabdomyosarcoma"),
    ("SARC_RMS_ERMS", "embryonal rhabdomyosarcoma"),
    ("SARC_RMS_PRMS", "pleomorphic rhabdomyosarcoma"),
    ("SARC_RMS_SSRMS", "spindle cell/sclerosing rhabdomyosarcoma"),
    ("SARC_LMS", "leiomyosarcoma"),
    ("SARC_LPS_UNSPEC", "liposarcoma"),
    ("SARC_MYXFIB", "myxofibrosarcoma"),
    ("SARC_SYN", "synovial sarcoma"),
    ("SARC_UPS", "undifferentiated pleomorphic sarcoma"),
]
# Rare-subtype Treehouse-direct codes (zero-download, own disease labels).
# group "sarc_rare_direct" -> scripts/sweep_sarc_rare_subtypes.py (direct path).
# NPC is not a sarcoma but uses the same Treehouse-direct pattern.
_SARC_RARE_DIRECT = [
    ("SARC_ANGIO", "angiosarcoma"),
    ("SARC_ASPS", "alveolar soft part sarcoma"),
    ("SARC_DSRCT", "desmoplastic small round cell tumor"),
    ("SARC_EPITH", "epithelioid sarcoma"),
    ("SARC_IMT", "inflammatory myofibroblastic tumor"),
    ("SARC_IFS", "infantile fibrosarcoma"),
    ("SARC_MPNST", "malignant peripheral nerve sheath tumor"),
    ("SARC_EHE", "epithelioid hemangioendothelioma"),
    ("SARC_LGFMS", "low grade fibromyxoid sarcoma"),
    ("NPC", "nasopharyngeal carcinoma"),
]
# TCGA-via-Treehouse direct projects: stem == "tcga_<lower(code)>", TCGA-only.
# group "tcga_direct" -> scripts/sweep_treehouse_tcga_cohorts.py.
# NB: GBM/LGG are split out (tcga_glioma group); the bare "SARC" code is the
# computed pan-sarcoma grand union, not a concrete per-sample cohort.
_TCGA_DIRECT = [
    ("ACC", "adrenocortical carcinoma"),
    ("BLCA", "bladder urothelial carcinoma"),
    ("BRCA", "breast invasive carcinoma"),
    ("CESC", "cervical & endocervical cancer"),
    ("CHOL", "cholangiocarcinoma"),
    ("COAD", "colon adenocarcinoma"),
    ("DLBC", "diffuse large B-cell lymphoma"),
    ("ESCA", "esophageal carcinoma"),
    ("HNSC", "head & neck squamous cell carcinoma"),
    ("KICH", "kidney chromophobe"),
    ("KIRC", "kidney clear cell carcinoma"),
    ("KIRP", "kidney papillary cell carcinoma"),
    ("LAML", "acute myeloid leukemia"),
    ("LIHC", "hepatocellular carcinoma"),
    ("LUAD", "lung adenocarcinoma"),
    ("LUSC", "lung squamous cell carcinoma"),
    ("MESO", "mesothelioma"),
    ("OV", "ovarian serous cystadenocarcinoma"),
    ("PAAD", "pancreatic adenocarcinoma"),
    ("PCPG", "pheochromocytoma & paraganglioma"),
    ("PRAD", "prostate adenocarcinoma"),
    ("READ", "rectum adenocarcinoma"),
    ("SKCM", "skin cutaneous melanoma"),
    ("STAD", "stomach adenocarcinoma"),
    ("TGCT", "testicular germ cell tumor"),
    ("THCA", "thyroid carcinoma"),
    ("THYM", "thymoma"),
    ("UCEC", "uterine corpus endometrioid carcinoma"),
    ("UCS", "uterine carcinosarcoma"),
    ("UVM", "uveal melanoma"),
]


def _treehouse_registry() -> tuple[Cohort, ...]:
    rows: list[Cohort] = []
    for code, label in _POLYA_PEDIATRIC:
        rows.append(Cohort(code, code, _POLYA, group="polya_pediatric",
                           disease_label=label))
    for code, label in _SARC_RARE_DIRECT:
        rows.append(Cohort(code, code, _POLYA, group="sarc_rare_direct",
                           disease_label=label))
    # gastrointestinal stromal tumour: Treehouse-direct (not in TCGA-SARC).
    rows.append(Cohort("SARC_GIST", "SARC_GIST", _POLYA, group="sarc_subtypes",
                       disease_label="gastrointestinal stromal tumor"))
    for code, label in _TCGA_DIRECT:
        rows.append(Cohort(code, f"tcga_{code.lower()}", _POLYA,
                           group="tcga_direct", disease_label=label,
                           selection="tcga"))
    # glioma: Treehouse's single "glioma" label split TCGA-GBM vs TCGA-LGG.
    for code in ("GBM", "LGG"):
        rows.append(Cohort(code, f"tcga_{code.lower()}", _POLYA,
                           group="tcga_glioma", disease_label="glioma",
                           selection=f"gdc_project:TCGA-{code}"))
    # TCGA-BRCA PAM50 molecular subtypes (cBioPortal calls). selection holds the
    # cBioPortal PAM50 label (case differs from the registry code, e.g. Her2).
    for code, pam50 in [("BRCA_Basal", "BRCA_Basal"), ("BRCA_HER2", "BRCA_Her2"),
                        ("BRCA_LumA", "BRCA_LumA"), ("BRCA_LumB", "BRCA_LumB"),
                        ("BRCA_Normal", "BRCA_Normal")]:
        suffix = code.removeprefix("BRCA_").lower()
        rows.append(Cohort(code, f"tcga_brca_{suffix}", _POLYA,
                           group="tcga_brca_pam50",
                           disease_label="breast invasive carcinoma",
                           selection=f"pam50:{pam50}"))
    # TCGA-HNSC HPV split (cBioPortal calls).
    for code, hpv, suffix in [("HNSC_HPVpos", "HNSC_HPV+", "hpv_pos"),
                              ("HNSC_HPVneg", "HNSC_HPV-", "hpv_neg")]:
        rows.append(Cohort(code, f"tcga_hnsc_{suffix}", _POLYA,
                           group="tcga_hnsc_hpv",
                           disease_label="head & neck squamous cell carcinoma",
                           selection=f"hpv:{hpv}"))
    # TCGA-LUAD driver-mutation subtypes (cBioPortal MAF; any gene in the set).
    for code, genes in [("LUAD_EGFR", "EGFR"), ("LUAD_KRAS", "KRAS"),
                        ("LUAD_STK11", "STK11,KEAP1")]:
        suffix = code.removeprefix("LUAD_").lower()
        rows.append(Cohort(code, f"tcga_luad_{suffix}", _POLYA,
                           group="tcga_luad_mut",
                           disease_label="lung adenocarcinoma",
                           selection=f"mutation:{genes}"))
    # TCGA-SARC histology overlays (GDC primary_diagnosis). WDLPS is built by
    # the sarc_subtypes sweep; DDLPS/PLEOLPS by the sarc_rare overlay path.
    for code, diagnosis, grp in [
            ("SARC_WDLPS", "Liposarcoma, well differentiated", "sarc_subtypes"),
            ("SARC_DDLPS", "Dedifferentiated liposarcoma", "sarc_rare_overlay"),
            ("SARC_PLEOLPS", "Pleomorphic liposarcoma", "sarc_rare_overlay")]:
        suffix = code.removeprefix("SARC_").lower()
        rows.append(Cohort(code, f"tcga_sarc_{suffix}", _POLYA, group=grp,
                           disease_label="liposarcoma",
                           selection=f"histology:{diagnosis}"))
    # treehouse-ribod-25-01 (ribo-depleted release): stem == code.
    rows.append(Cohort("SARC_CHOR", "SARC_CHOR", _RIBOD, group="ribod",
                       disease_label="chordoma"))
    rows.append(Cohort("RB", "RB", _RIBOD, group="ribod",
                       disease_label="retinoblastoma"))
    return tuple(rows)


_TREEHOUSE_COHORTS: tuple[Cohort, ...] = _treehouse_registry()


# Neuroendocrine per-sample cohorts, built by
# scripts/build_ne_per_sample_parquets.py from cached NE source data (GEO
# log2TPM pancreatic NET, UCologne FPKM small-cell lung, IARC LNEN counts lung
# NET/NEC). Declared here — with ``stem == code`` — so the registry is the
# *complete* per-sample source of truth: the read path can enumerate them
# without the cache present, rather than only discovering them from disk. Their
# sample→code *selection* lives in the source-specific builders (e.g. the LNEN
# histology map), exactly as the Treehouse predicates do; the registry owns the
# cohort *list*, not the per-source build mechanism.
_NE_COHORTS: tuple[Cohort, ...] = (
    Cohort("NET_PANCREAS", "NET_PANCREAS", "gse118014-pannet", group="neuroendocrine"),
    Cohort("SCLC", "SCLC", "sclc-ucologne-2015", group="neuroendocrine"),
    Cohort("NET_LUNG", "NET_LUNG", "drmetrics-lnen-2020", group="neuroendocrine"),
    Cohort("NEC_LUNG_LARGECELL", "NEC_LUNG_LARGECELL", "drmetrics-lnen-2020",
           group="neuroendocrine"),
)

# The single declarative registry of EVERY per-sample cohort across all sources.
_PER_SAMPLE_COHORTS: tuple[Cohort, ...] = _TREEHOUSE_COHORTS + _NE_COHORTS


def _registry_for_source(source_id: str) -> dict[str, Cohort]:
    return {c.code: c for c in _PER_SAMPLE_COHORTS if c.source_id == source_id}


# Sources with an explicit code→Cohort map. A source that has rows in the
# registry above uses that map; any other source in PER_SAMPLE_SOURCES (e.g. a
# newly added one) still discovers its cohorts from disk with ``code == stem``.
_REGISTRY: dict[str, dict[str, Cohort]] = {
    sid: _registry_for_source(sid)
    for sid in {c.source_id for c in _PER_SAMPLE_COHORTS}
}


def cohorts_for_group(group: str) -> list[Cohort]:
    """All registry cohorts in a given build group (in registry order). The
    build sweeps call this instead of enumerating cohorts inline, so the
    (code, stem, disease_label, selection) of every cohort lives here once."""
    return [c for c in _PER_SAMPLE_COHORTS if c.group == group]


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
        for p in sorted(derived.glob(f"*{PER_SAMPLE_SUFFIX}")):
            stem = p.name[: -len(PER_SAMPLE_SUFFIX)]
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
            / f"{cohort.stem}{PER_SAMPLE_SUFFIX}")


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


# --- canonical per-sample matrix I/O (single source of truth) --------------

def sample_columns(df: pd.DataFrame) -> list[str]:
    """The per-sample value columns of a per-sample frame (everything that
    isn't an :data:`ID_COLS` id column)."""
    return [c for c in df.columns if c not in ID_COLS]


def read_per_sample(cohort: Cohort) -> pd.DataFrame:
    """Read a cohort's per-sample TPM parquet (the canonical open + error).

    Returns the raw frame: ``ID_COLS`` + one column per sample (linear TPM).
    Use :func:`sample_columns` to get the value columns. Raises a helpful
    ``FileNotFoundError`` if the matrix isn't cached."""
    path = parquet_path(cohort)
    if not path.exists():
        raise FileNotFoundError(
            f"no per-sample parquet for {cohort.code} at {path} — run "
            f"`pirlygenes build {cohort.source_id}` or `downloads fetch` first")
    return pd.read_parquet(path)


def iter_per_sample_cohorts(*, sources=None):
    """Yield ``(cohort, df)`` for every per-sample cohort present on disk across
    ``sources`` (default: all :data:`PER_SAMPLE_SOURCES`, in registration order).
    The single iteration point shared by the representatives / percentile
    generators so the source list and parquet layout live in one place."""
    for source_id in (sources or list(PER_SAMPLE_SOURCES)):
        for _code, cohort in available_cohorts(source_id).items():
            yield cohort, read_per_sample(cohort)


def write_per_sample(gene_table: pd.DataFrame, values: pd.DataFrame,
                     source_id: str, code: str):
    """Write a cohort's per-sample TPM parquet in the canonical layout
    (``ID_COLS`` + sample columns) to ``<source-cache>/derived/``; returns the
    path. Shared by the per-sample builders so the on-disk format has one
    writer."""
    derived = downloads.source_cache_dir(source_id) / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    out = pd.concat(
        [gene_table[list(ID_COLS)].reset_index(drop=True),
         values.reset_index(drop=True)],
        axis=1,
    )
    path = derived / f"{code}{PER_SAMPLE_SUFFIX}"
    out.to_parquet(path, index=False)
    return path
