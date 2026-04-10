# Licensed under the Apache License, Version 2.0

"""Sample decomposition templates.

Each template defines the set of components that a sample can be
decomposed into. Templates are structural (not cancer-type-specific)
— the cancer type determines which tumor reference profile and
constraint configuration to use.
"""

from .signature import CANCER_TO_ORIGIN

# Shared immune components used across most solid tumor templates
_SOLID_IMMUNE = [
    "CD8_T", "CD4_T", "B_cell", "plasma", "NK",
    "macrophage", "DC", "neutrophil",
]

_SOLID_STROMA = ["fibroblast", "endothelial"]

TEMPLATES = {
    "pure_population": {
        "components": [],  # just tumor; filled dynamically
        "description": "Cell line or sorted population",
    },
    "solid_primary": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA,
        # + origin_tissue (added dynamically from cancer type)
        "add_origin": True,
        "description": "Solid tumor primary resection",
    },
    "met_lymph_node": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["LN_parenchyma"],
        "description": "Metastasis in lymph node",
    },
    "met_liver": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["hepatocyte"],
        "description": "Metastasis in liver",
    },
    "met_brain": {
        "components": ["CD8_T", "CD4_T", "macrophage", "DC",
                       "fibroblast", "endothelial", "astrocyte", "neuron"],
        "description": "Metastasis in brain (reduced immune diversity)",
    },
    "met_lung": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["pneumocyte"],
        "description": "Metastasis in lung",
    },
    "met_bone": {
        "components": _SOLID_IMMUNE + ["endothelial", "osteoblast", "marrow_stroma"],
        "description": "Metastasis in bone",
    },
    "met_peritoneal": {
        "components": _SOLID_IMMUNE + _SOLID_STROMA + ["mesothelial"],
        "description": "Peritoneal metastasis",
    },
    "met_adrenal": {
        "components": ["CD8_T", "CD4_T", "macrophage", "DC",
                       "fibroblast", "endothelial", "adrenal_cortical"],
        "description": "Metastasis in adrenal gland",
    },
    "met_skin": {
        "components": ["CD8_T", "CD4_T", "macrophage", "DC", "NK",
                       "fibroblast", "endothelial", "keratinocyte", "melanocyte"],
        "description": "Metastasis in skin",
    },
    "heme_marrow": {
        "components": ["normal_myeloid", "normal_lymphoid", "erythroid",
                       "endothelial", "marrow_stroma"],
        "description": "Heme malignancy in bone marrow",
    },
    "heme_nodal": {
        "components": ["CD8_T", "CD4_T", "B_cell", "macrophage", "DC",
                       "fibroblast", "endothelial"],
        "description": "Heme malignancy in lymph node",
    },
    "heme_blood": {
        "components": ["CD8_T", "B_cell", "NK", "monocyte",
                       "neutrophil", "erythroid"],
        "description": "Heme malignancy in peripheral blood",
    },
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
    components = ["tumor"] + list(tmpl["components"])

    # Add origin tissue for solid_primary
    if tmpl.get("add_origin") and cancer_type:
        origin = CANCER_TO_ORIGIN.get(cancer_type)
        if origin and origin not in components:
            components.append(origin)

    return components


def list_templates():
    """Return list of (name, description) for all templates."""
    return [(name, t["description"]) for name, t in TEMPLATES.items()]
