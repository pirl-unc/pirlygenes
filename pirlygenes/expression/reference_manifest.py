"""Lightweight compatibility manifest for cancer-expression references.

The public manifest predates the delegated oncoref read path and intentionally
keeps pirlygenes' cohort labels and display provenance.  Its row spine comes
from the tiny cohort-view provenance sidecar; oncoref and the cohort registry
only fill metadata, so constructing it never materializes expression values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


PUBLIC_COLUMNS = (
    "cancer_code",
    "source_cohort",
    "source_project",
    "source_version",
    "n_samples",
    "processing_pipeline",
    "tumor_origin",
    "metastasis_site",
)


# Compatibility display labels from the former summary-frame projection.  The
# cohort registry deliberately uses shorter owner labels, while this public API
# has historically exposed source-specific descriptions.
_SOURCE_PROJECT_COMPAT = {
    "BEATAML_OHSU_2022": "BeatAML 1.0",
    "CGCI_BLGSP": "CGCI Burkitt Lymphoma Genome Sequencing Project",
    "DRMETRICS_ALCALA_2019_LNEN": "IARC pan-LNEN (Alcala 2019 / Gabriel 2020)",
    "GSE114922_SHIOZAWA_2018": (
        "Shiozawa 2018 MDS (GSE114922) — recount3 Gencode v26"
    ),
    "GSE118014_ALVAREZ_2018": (
        "Alvarez 2018 primary PanNET (GSE118014) — recount3 Gencode v26"
    ),
    "GSE120328_LAMPRECHT_2018": (
        "Lamprecht 2018 cHL LCM (GSE120328) — recount3 Gencode v26"
    ),
    "GSE98894_ALVAREZ_2018_NET": (
        "Alvarez 2018 GEP-NET (GSE98894) — recount3 Gencode v26"
    ),
    "SCLC_UCOLOGNE_2015_TF_DOMINANCE": (
        "University of Cologne (TF-dominance)"
    ),
    "TARGET_ALL_2018": "TARGET ALL",
    "TARGET_NBL_2018": "TARGET Neuroblastoma",
    "TARGET_RT_2017": "TARGET Rhabdoid Tumor",
    "TARGET_WT_2015": "TARGET Wilms Tumor",
    "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS": (
        "Treehouse (MBL subgroup markers)"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50": (
        "Treehouse (TCGA-BRCA) × cBioPortal PAM50"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI": (
        "Treehouse (TCGA-COADREAD) × cBioPortal MSI"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV": (
        "Treehouse (TCGA-HNSC) × cBioPortal HPV"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT": (
        "Treehouse (TCGA-LUAD) × cBioPortal MAF"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY": "Treehouse (TCGA samples)",
    "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE": (
        "Treehouse (TCGA-STAD) x cBioPortal molecular subtype"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_SUBSET": "Treehouse (TCGA samples)",
    "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE": (
        "Treehouse (TCGA-UCEC) x cBioPortal molecular subtype"
    ),
}

_GLIOMA_VERSION = (
    "Treehouse Tumor Compendium 25.01 PolyA, TCGA-GBM/LGG split via GDC "
    "case→project lookup; downloaded from public.gi.ucsc.edu/~ekephart/public-data/; "
    "HUGO symbols harmonized to Ensembl release 112; log2(TPM+1) inverse-transformed"
)
_SARC_VERSION = (
    "Treehouse Tumor Compendium 25.01 PolyA, TCGA-SARC histology split via GDC "
    "primary_diagnosis lookup; HUGO symbols harmonized to Ensembl release 112; "
    "log2(TPM+1) inverse-transformed"
)
_SARC_DOWNLOADED_VERSION = (
    "Treehouse Tumor Compendium 25.01 PolyA, TCGA-SARC histology split via GDC "
    "primary_diagnosis lookup; downloaded from "
    "public.gi.ucsc.edu/~ekephart/public-data/; HUGO symbols harmonized to Ensembl "
    "release 112; log2(TPM+1) inverse-transformed"
)
_STAD_VERSION = (
    "Treehouse Childhood Cancer Initiative, Tumor Compendium 25.01 PolyA, "
    "TCGA-STAD subset split by cBioPortal molecular subtype calls from "
    "stad_tcga_pan_can_atlas_2018. PMID 25079317.\n; unit=log2(TPM+1); "
    "canonicalized with oncoref expression_engine; clean_tpm=16/9/75"
)
_UCEC_VERSION = (
    "Treehouse Childhood Cancer Initiative, Tumor Compendium 25.01 PolyA, "
    "TCGA-UCEC subset split by cBioPortal molecular subtype calls from "
    "ucec_tcga_pan_can_atlas_2018. PMID 23636398.\n; unit=log2(TPM+1); "
    "canonicalized with oncoref expression_engine; clean_tpm=16/9/75"
)

_SOURCE_VERSION_COMPAT = {
    (code, "TREEHOUSE_POLYA_25_01_TCGA_SUBSET"): _GLIOMA_VERSION
    for code in ("GBM", "LGG")
}
_SOURCE_VERSION_COMPAT.update({
    ("SARC_PLEOLPS", "TREEHOUSE_POLYA_25_01_TCGA_SUBSET"): _SARC_VERSION,
    (
        "SARC_WDLPS",
        "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
    ): _SARC_DOWNLOADED_VERSION,
})
_SOURCE_VERSION_COMPAT.update({
    (code, "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE"): _STAD_VERSION
    for code in ("STAD_CIN", "STAD_EBV", "STAD_GS", "STAD_MSI")
})
_SOURCE_VERSION_COMPAT.update({
    (code, "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE"): _UCEC_VERSION
    for code in ("UCEC_CNH", "UCEC_CNL", "UCEC_MSI", "UCEC_POLE")
})

_TUMOR_ORIGIN_COMPAT = {
    "GSE235092_MERKEL_2024": "primary",
    "GSE239531_VANOOST_2024": "primary",
    "GSE75885_DELESPAUL_2017": "mixed",
    "GSE98894_ALVAREZ_2018_NET": "metastasis",
    "TARGET_NBL_2018": "primary",
    "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS": "primary",
    "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI": "primary",
    "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY": "mixed",
    "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE": "primary",
    "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE": "primary",
    "UNC_NUTM1": "primary",
}
_METASTASIS_SITE_COMPAT = {"GSE98894_ALVAREZ_2018_NET": "liver"}
_PROCESSING_PIPELINE_COMPAT = {
    "TREEHOUSE_POLYA_25_01_TCGA_STAD_SUBTYPE": (
        "treehouse_polya_25_01_tcga_stad_subtype_log2_tpm_1_to_tpm_"
        "oncoref_canonical_clean_tpm_16_9_75"
    ),
    "TREEHOUSE_POLYA_25_01_TCGA_UCEC_SUBTYPE": (
        "treehouse_polya_25_01_tcga_ucec_subtype_log2_tpm_1_to_tpm_"
        "oncoref_canonical_clean_tpm_16_9_75"
    ),
}


def _present(value) -> bool:
    return value is not None and not pd.isna(value) and str(value) != ""


def _first_present(*values, default=np.nan):
    for value in values:
        if _present(value):
            return value
    return default


def _records_by_key(df: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    if df.empty or not set(columns) <= set(df.columns):
        return {}
    records = {}
    for _, row in df.drop_duplicates(list(columns)).iterrows():
        records[tuple(str(row[column]) for column in columns)] = row
    return records


def build_reference_manifest(
    provenance: pd.DataFrame,
    availability: pd.DataFrame,
    registry: pd.DataFrame,
    classify_source_cohort: Callable[[str], tuple[str | None, str | None]],
) -> pd.DataFrame:
    """Build the legacy eight-column manifest without expression values."""
    required = {"cancer_code", "source_cohort", "processing_pipeline", "n_samples"}
    missing = required - set(provenance.columns)
    if missing:
        raise ValueError(
            "cohort-view provenance is missing required columns: "
            + ", ".join(sorted(missing))
        )

    available = availability
    if "available" in available.columns:
        available = available[available["available"].fillna(False)]
    availability_by_key = _records_by_key(
        available, ("cancer_code", "source_cohort"),
    )
    registry_by_cohort = _records_by_key(registry, ("cohort_id",))

    rows = []
    for _, source in provenance.iterrows():
        code = str(source["cancer_code"])
        cohort = str(source["source_cohort"])
        key = (code, cohort)
        delegated = availability_by_key.get(key, {})
        registered = registry_by_cohort.get((cohort,), {})
        classified_origin, classified_site = classify_source_cohort(cohort)

        rows.append({
            "cancer_code": code,
            "source_cohort": cohort,
            "source_project": _first_present(
                source.get("source_project"),
                _SOURCE_PROJECT_COMPAT.get(cohort),
                registered.get("source_project"),
                delegated.get("source_project"),
            ),
            "source_version": _first_present(
                source.get("source_version"),
                _SOURCE_VERSION_COMPAT.get(key),
                registered.get("provenance"),
                delegated.get("source_version"),
            ),
            "n_samples": _first_present(
                source.get("n_samples"),
                delegated.get("n_samples"),
                delegated.get("n_reference_samples"),
            ),
            "processing_pipeline": _first_present(
                _PROCESSING_PIPELINE_COMPAT.get(cohort),
                source.get("processing_pipeline"),
                delegated.get("processing_pipeline"),
            ),
            "tumor_origin": _first_present(
                source.get("tumor_origin"),
                _TUMOR_ORIGIN_COMPAT.get(cohort),
                classified_origin,
                delegated.get("tumor_origin"),
            ),
            "metastasis_site": _first_present(
                source.get("metastasis_site"),
                _METASTASIS_SITE_COMPAT.get(cohort),
                classified_site,
                delegated.get("metastasis_site"),
            ),
        })

    out = pd.DataFrame(rows, columns=PUBLIC_COLUMNS)
    out["n_samples"] = pd.to_numeric(out["n_samples"], errors="raise").astype("int64")
    out["source_version"] = out["source_version"].astype("category")
    out["processing_pipeline"] = out["processing_pipeline"].astype("category")
    out["cancer_code"] = out["cancer_code"].astype("string")
    for column in (
        "source_cohort",
        "source_project",
        "tumor_origin",
        "metastasis_site",
    ):
        out[column] = out[column].astype(object)
    return out
