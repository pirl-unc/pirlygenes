# Licensed under the Apache License, Version 2.0

"""Signature matrix construction from HPA single-cell type data.

The signature matrix maps each cell type to a reference expression profile
(nTPM per gene). Components are selected based on the sample template
(e.g. met_lymph_node includes LN parenchyma; solid_primary does not).
"""

from ..load_dataset import get_data


# Map from decomposition component name → HPA cell type(s) to average
COMPONENT_TO_HPA = {
    # Immune
    "CD8_T": ["T-cells"],  # HPA doesn't split CD4/CD8; we use T-cells for both
    "CD4_T": ["T-cells"],
    "B_cell": ["B-cells"],
    "plasma": ["Plasma cells"],
    "NK": ["NK-cells"],
    "macrophage": ["Macrophages", "Kupffer cells"],
    "DC": ["dendritic cells"],
    "monocyte": ["monocytes"],
    "neutrophil": ["granulocytes"],
    # Stromal
    "fibroblast": ["Fibroblasts"],
    "endothelial": ["Endothelial cells"],
    # Site-specific parenchyma
    "LN_parenchyma": ["B-cells", "T-cells"],  # lymph node is mostly lymphocytes
    "hepatocyte": ["Hepatocytes"],
    "astrocyte": ["Astrocytes"],
    "neuron": ["Excitatory neurons", "Inhibitory neurons"],
    "pneumocyte": ["Alveolar cells type 1", "Alveolar cells type 2"],
    "osteoblast": ["Undifferentiated cells"],  # rough proxy
    "marrow_stroma": ["Undifferentiated cells"],
    "mesothelial": ["Mesothelial cells"],
    "adrenal_cortical": ["Undifferentiated cells"],  # rough proxy
    "keratinocyte": ["Basal keratinocytes", "Suprabasal keratinocytes"],
    "melanocyte": ["Melanocytes"],
    "origin_prostate": ["Prostatic glandular cells", "Basal prostatic cells"],
    "origin_breast": ["Breast glandular cells", "Breast myoepithelial cells"],
    "origin_colon": ["Distal enterocytes", "Intestinal goblet cells"],
    "origin_lung": ["Alveolar cells type 1", "Alveolar cells type 2", "Club cells"],
    "origin_liver": ["Hepatocytes", "Cholangiocytes"],
    "origin_kidney": ["Proximal tubular cells", "Distal tubular cells", "Collecting duct cells"],
    "origin_stomach": ["Gastric mucus-secreting cells"],
    "origin_skin": ["Basal keratinocytes", "Suprabasal keratinocytes", "Melanocytes"],
    "origin_brain": ["Astrocytes", "Oligodendrocytes", "Excitatory neurons"],
    "origin_thyroid": ["Glandular and luminal cells"],
    "origin_endometrium": ["Glandular and luminal cells", "Endometrial stromal cells"],
    "origin_pancreas": ["Ductal cells", "Pancreatic endocrine cells"],
    "origin_ovary": ["Granulosa cells", "Ovarian stromal cells"],
    # Heme malignancy normal counterparts
    "normal_myeloid": ["monocytes", "granulocytes"],
    "normal_lymphoid": ["B-cells", "T-cells", "NK-cells"],
    "erythroid": ["Erythroid cells"],
    "normal_blood": ["T-cells", "B-cells", "NK-cells", "monocytes", "granulocytes"],
}

# Map cancer type to origin tissue component
CANCER_TO_ORIGIN = {
    "PRAD": "origin_prostate",
    "BRCA": "origin_breast",
    "COAD": "origin_colon",
    "READ": "origin_colon",
    "LUAD": "origin_lung",
    "LUSC": "origin_lung",
    "LIHC": "origin_liver",
    "KIRC": "origin_kidney",
    "KIRP": "origin_kidney",
    "KICH": "origin_kidney",
    "STAD": "origin_stomach",
    "SKCM": "origin_skin",
    "UVM": "origin_skin",
    "GBM": "origin_brain",
    "LGG": "origin_brain",
    "THCA": "origin_thyroid",
    "UCEC": "origin_endometrium",
    "UCS": "origin_endometrium",
    "PAAD": "origin_pancreas",
    "OV": "origin_ovary",
    # Others default to no specific origin component
}


def _load_hpa_cell_types():
    """Load the HPA single-cell type nTPM matrix (gene × cell type)."""
    return get_data("hpa-cell-type-expression")


def build_signature_matrix(components, gene_subset=None):
    """Build a signature matrix for the given component list.

    Parameters
    ----------
    components : list of str
        Component names (e.g. ["CD8_T", "B_cell", "macrophage", ...]).
        Must be keys in COMPONENT_TO_HPA.
    gene_subset : set of str, optional
        If provided, only include these Ensembl gene IDs.

    Returns
    -------
    genes : list of str
        Ensembl gene IDs (rows).
    symbols : list of str
        Gene symbols (parallel to genes).
    matrix : numpy.ndarray
        Shape (n_genes, n_components), nTPM values.
    component_names : list of str
        Column labels (same order as components input).
    """
    import numpy as np

    hpa = _load_hpa_cell_types()
    if gene_subset is not None:
        hpa = hpa[hpa["Ensembl_Gene_ID"].isin(gene_subset)]

    genes = hpa["Ensembl_Gene_ID"].tolist()
    symbols = hpa["Symbol"].tolist()

    cols = []
    for comp in components:
        hpa_types = COMPONENT_TO_HPA.get(comp, [])
        if not hpa_types:
            # Unknown component → zeros
            cols.append(np.zeros(len(genes)))
            continue
        # Average across the HPA cell types for this component
        present = [t for t in hpa_types if t in hpa.columns]
        if present:
            vals = hpa[present].astype(float).fillna(0).mean(axis=1).values
        else:
            vals = np.zeros(len(genes))
        cols.append(vals)

    matrix = np.column_stack(cols)
    return genes, symbols, matrix, list(components)
