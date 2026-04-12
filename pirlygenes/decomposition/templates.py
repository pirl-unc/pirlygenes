# Licensed under the Apache License, Version 2.0

"""Sample decomposition templates.

Templates define broad non-tumor compartments that are plausible for a sample.
They deliberately avoid unsupported fine-grained immune splits and avoid
including an "origin normal epithelium" column that steals retained lineage
signal away from tumor cells in mixed samples.
"""

from ..tumor_purity import CANCER_TO_TISSUE

# ── Hierarchical tissue categories ──────────────────────────────────────
# Site-specific host components are scored against a *category* of bulk
# tissues rather than a single HPA cell type.  For the NNLS, the best-
# matching member tissue is selected as the reference column.
#
# Why: HPA single-cell profiles for astrocytes and neurons have
# housekeeping-gene medians of ~20 nTPM (vs ~350 for immune/stroma),
# likely from harsh brain-tissue dissociation.  HK-normalisation inflates
# these references ~17×, making the NNLS think a 75%-brain sample has
# <1% brain signal.  Bulk tissue references (cerebral_cortex etc.) have
# normal HK medians (~380) and avoid this distortion entirely.
#
# The best-match selection within each category captures within-category
# variation (e.g. a cerebellar met matches cerebellum while a cortical
# met matches cerebral_cortex).  Correlation is computed only on
# category-enriched genes (>2× cross-tissue median) to prevent broadly
# expressed genes from dominating the match.

TISSUE_CATEGORIES = {
    "CNS": [
        "cerebral_cortex", "cerebellum", "hippocampal_formation", "amygdala",
        "basal_ganglia", "hypothalamus", "midbrain", "choroid_plexus",
        "spinal_cord",
    ],
    "liver": ["liver"],
    "lung": ["lung"],
    "kidney": ["kidney"],
    "GI_lower": ["colon", "rectum", "appendix", "small_intestine"],
    "GI_upper": ["stomach", "esophagus", "duodenum"],
    "bone_marrow": ["bone_marrow"],
    "skin": ["skin"],
    "adrenal": ["adrenal_gland"],
    "muscle": ["heart_muscle", "skeletal_muscle", "smooth_muscle"],
    "immune_lymphoid": ["lymph_node", "spleen", "thymus", "tonsil"],
}

# Map from decomposition component name → tissue category.
# Components listed here use bulk tissue composite references (best-match
# selection) instead of HPA single-cell profiles.
# Only components whose HPA single-cell reference has a distorted HK
# median are routed through the composite approach.  Components with
# normal HK medians (osteoblast=428, marrow_stroma=428, melanocyte=393)
# keep their HPA profiles — using the broad bulk tissue reference would
# let them absorb unrelated signal (e.g. bone_marrow absorbs immune).
COMPONENT_TO_CATEGORY = {
    "hepatocyte": "liver",
    "pneumocyte": "lung",
    "astrocyte": "CNS",
    "neuron": "CNS",
    "adrenal_cortical": "adrenal",
    "keratinocyte": "skin",
}

# Shared immune components used across most solid tumor templates
_SOLID_IMMUNE = [
    "T_cell", "B_cell", "plasma", "NK", "myeloid",
]

_SOLID_STROMA = ["fibroblast", "endothelial"]

TEMPLATES = {
    "pure_population": {
        "components": [],  # just tumor; filled dynamically
        "description": "Cell line or sorted population",
    },
    "solid_primary": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        "description": "Solid tumor primary resection",
    },
    "met_lymph_node": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        "host_tissue": "lymph_node",
        "description": "Metastasis in lymph node",
    },
    "met_liver": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["hepatocyte"],
        "host_tissue": "liver",
        "description": "Metastasis in liver",
    },
    "met_brain": {
        # Both astrocyte and neuron resolve to the same bulk tissue via
        # COMPONENT_TO_CATEGORY → "CNS".  The NNLS sees two identical
        # columns and splits the weight evenly — report their fractions
        # summed as "CNS parenchyma" rather than reading them literally.
        "components": ["T_cell", "myeloid", "fibroblast", "endothelial", "astrocyte", "neuron"],
        "host_tissue": "cerebral_cortex",
        "description": "Metastasis in brain (reduced immune diversity)",
    },
    "met_lung": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["pneumocyte"],
        "host_tissue": "lung",
        "description": "Metastasis in lung",
    },
    "met_bone": {
        "components": _SOLID_IMMUNE + ["endothelial", "osteoblast", "marrow_stroma"],
        "host_tissue": "bone_marrow",
        "description": "Metastasis in bone",
    },
    "met_peritoneal": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["mesothelial"],
        "host_tissue": "appendix",
        "description": "Peritoneal metastasis",
    },
    "met_adrenal": {
        "components": ["T_cell", "myeloid", "fibroblast", "endothelial", "adrenal_cortical"],
        "host_tissue": "adrenal_gland",
        "description": "Metastasis in adrenal gland",
    },
    "met_skin": {
        "components": ["T_cell", "myeloid", "NK", "fibroblast", "endothelial", "keratinocyte", "melanocyte"],
        "host_tissue": "skin",
        "description": "Metastasis in skin",
    },
    "met_soft_tissue": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        "host_tissue": "smooth_muscle",
        "description": "Metastasis in soft tissue / retroperitoneum",
    },
    "heme_marrow": {
        "components": ["normal_myeloid", "normal_lymphoid", "erythroid",
                       "endothelial", "marrow_stroma"],
        "description": "Heme malignancy in bone marrow",
    },
    "heme_nodal": {
        "components": ["T_cell", "B_cell", "plasma", "myeloid", "fibroblast", "endothelial"],
        "host_tissue": "lymph_node",
        "description": "Heme malignancy in lymph node",
    },
    "heme_blood": {
        "components": ["T_cell", "B_cell", "NK", "normal_myeloid", "erythroid"],
        "description": "Heme malignancy in peripheral blood",
    },
}

_PRIMARY_HOST_TISSUES = {
    # Lower-GI primaries often look colon/rectum/appendix-like in bulk and
    # should not be forced into a single exact bowel segment.
    "COAD": ["colon", "rectum", "appendix"],
    "READ": ["rectum", "colon", "appendix"],
}

_TEMPLATE_HOST_TISSUES = {
    # Retroperitoneal/deep soft tissue biopsies can look like a mix of muscle
    # and adipose rather than one exact reference tissue.
    "met_soft_tissue": ["smooth_muscle", "skeletal_muscle", "adipose_tissue"],
}

# Tumor cell-of-origin → which constraint categories to relax
# (these genes become lineage markers instead of constraints)
TUMOR_ORIGIN_TYPE = {
    # Mesenchymal tumors: relax fibroblast constraints
    "SARC": "fibroblast",
    # B-cell tumors: relax IG/B-cell constraints
    "DLBC": "B_cell",
    "CLL": "B_cell",
    # T-cell tumors: relax TCR/T-cell constraints
    # (future: T-cell lymphoma types)
    # Myeloid tumors: relax myeloid constraints
    "LAML": "normal_myeloid",
    # All others: epithelial origin, no standard constraints affected
}


def get_template_components(template_name, cancer_type=None):
    """Get the full component list for a template + cancer type.

    Returns
    -------
    components : list of str
        Component names including "tumor" as the first entry.
    """
    tmpl = TEMPLATES[template_name]
    return ["tumor"] + list(tmpl["components"])


def get_template_extra_components(template_name):
    """Return components beyond the shared solid immune/stroma basis."""
    return [comp for comp in TEMPLATES[template_name]["components"] if comp not in _SOLID_IMMUNE + _SOLID_STROMA]


def get_template_host_tissues(template_name, cancer_type=None):
    """Return one or more host tissues used to score a template."""
    if template_name == "solid_primary" and cancer_type is not None:
        tissues = _PRIMARY_HOST_TISSUES.get(cancer_type)
        if tissues:
            return list(tissues)
        tissue = CANCER_TO_TISSUE.get(cancer_type)
        return [tissue] if tissue else []

    tissues = _TEMPLATE_HOST_TISSUES.get(template_name)
    if tissues:
        return list(tissues)

    tissue = TEMPLATES[template_name].get("host_tissue")
    return [tissue] if tissue else []


def get_template_host_tissue(template_name, cancer_type=None):
    """Return the matched host tissue used to score a template, if any."""
    tissues = get_template_host_tissues(template_name, cancer_type=cancer_type)
    return tissues[0] if tissues else None


def list_templates():
    """Return list of (name, description) for all templates."""
    return [(name, t["description"]) for name, t in TEMPLATES.items()]
