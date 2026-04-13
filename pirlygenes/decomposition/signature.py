# Licensed under the Apache License, Version 2.0

"""Signature matrix construction from HPA single-cell and tissue data.

The current decomposition intentionally works at a broad compartment level.
The bundled HPA references support T cells, B cells, plasma cells, NK cells,
myeloid cells, stromal compartments, and a handful of host-tissue contexts.
They do *not* support fine-grained T-cell subtype splits such as CD4 vs CD8
or alpha-beta vs gamma-delta in a way that is stable for bulk decomposition.
"""

from .templates import TISSUE_CATEGORIES, COMPONENT_TO_CATEGORY
from ..gene_sets_cancer import pan_cancer_expression
from ..load_dataset import get_data


# Map from decomposition component name → HPA cell type(s) to average
COMPONENT_TO_HPA = {
    # Broad immune compartments
    "T_cell": ["T-cells"],
    "B_cell": ["B-cells"],
    "plasma": ["Plasma cells"],
    "NK": ["NK-cells"],
    "myeloid": ["Macrophages", "Kupffer cells", "dendritic cells", "monocytes", "granulocytes"],
    # Stromal
    "fibroblast": ["Fibroblasts"],
    "endothelial": ["Endothelial cells"],
    # Site-specific host compartments
    "hepatocyte": ["Hepatocytes"],
    "astrocyte": ["Astrocytes"],
    "neuron": ["Excitatory neurons", "Inhibitory neurons"],
    "pneumocyte": ["Alveolar cells type 1", "Alveolar cells type 2"],
    "osteoblast": ["Undifferentiated cells"],
    "marrow_stroma": ["Undifferentiated cells"],
    "mesothelial": ["Mesothelial cells"],
    "adrenal_cortical": ["Undifferentiated cells"],
    "keratinocyte": ["Basal keratinocytes", "Suprabasal keratinocytes"],
    "melanocyte": ["Melanocytes"],
    # Heme templates
    "normal_myeloid": ["monocytes", "granulocytes"],
    "normal_lymphoid": ["B-cells", "T-cells", "NK-cells"],
    "erythroid": ["Erythroid cells"],
    "normal_blood": ["T-cells", "B-cells", "NK-cells", "monocytes", "granulocytes"],
}

# Legacy bulk-tissue fallback for components not routed through
# COMPONENT_TO_CATEGORY.  Most site-specific components now use the
# composite approach (see templates.COMPONENT_TO_CATEGORY); only
# mesothelial remains here because it lines serous cavities and
# should not match the broader GI_lower category.
COMPONENT_TO_BULK_TISSUE = {
    "mesothelial": ["appendix"],
}


COMPONENT_MARKERS = {
    "T_cell": ["CD3D", "CD3E", "TRAC", "LTB", "IL7R"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "plasma": ["IGKC", "IGLC2", "IGHG1", "JCHAIN", "MZB1", "SDC1"],
    "NK": ["NKG7", "KLRD1", "GNLY", "PRF1", "FCGR3A"],
    "myeloid": ["LYZ", "TYROBP", "FCER1G", "ITGAM", "CD68", "C1QA"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "COL3A1"],
    "endothelial": ["PECAM1", "VWF", "CDH5", "KDR", "ESAM", "EMCN"],
    "hepatocyte": ["ALB", "APOA1", "HP", "TTR", "FGB", "APOH"],
    "astrocyte": ["GFAP", "AQP4", "ALDH1L1", "SLC1A3"],
    "neuron": ["RBFOX3", "SNAP25", "SYT1", "GAP43"],
    "pneumocyte": ["SFTPA1", "SFTPA2", "SFTPB", "SFTPC", "NAPSA"],
    "osteoblast": ["BGLAP", "ALPL", "SPP1", "COL1A1"],
    "marrow_stroma": ["CXCL12", "VCAM1", "KITLG", "COL1A1"],
    "mesothelial": ["MSLN", "ITLN1", "KRT19", "KRT8"],
    "adrenal_cortical": ["STAR", "CYP11B1", "CYP21A2", "NR5A1"],
    "keratinocyte": ["KRT14", "KRT5", "KRT1", "KRT10"],
    "melanocyte": ["MLANA", "PMEL", "TYR", "DCT"],
    "normal_myeloid": ["LYZ", "S100A8", "S100A9", "FCER1G"],
    "normal_lymphoid": ["CD3D", "MS4A1", "NKG7", "LTB"],
    "erythroid": ["HBA1", "HBA2", "HBB", "ALAS2"],
    "normal_blood": ["LYZ", "S100A8", "NKG7", "MS4A1", "HBB"],
}

def _load_hpa_cell_types():
    """Load the HPA single-cell type nTPM matrix (gene × cell type)."""
    return get_data("hpa-cell-type-expression")


def _load_normal_tissue_expression():
    """Load the bulk normal-tissue expression matrix."""
    return pan_cancer_expression()


def get_component_markers(component):
    """Return curated marker genes for a broad decomposition component."""
    return list(COMPONENT_MARKERS.get(component, []))


def _select_best_category_tissue(category, sample_by_eid, genes, bulk_indexed):
    """Pick the member tissue in *category* that best matches the sample.

    For each member tissue, compute how well the sample's category-enriched
    genes correlate with that tissue.  "Category-enriched" genes are those
    where any member tissue's expression exceeds the cross-tissue median —
    this focuses the comparison on genes that are actually informative about
    category membership, not broadly-expressed genes.

    Returns (best_tissue_name, best_column_name).
    """
    import numpy as np

    tissues = TISSUE_CATEGORIES[category]
    available_cols = [f"nTPM_{t}" for t in tissues if f"nTPM_{t}" in bulk_indexed.columns]
    if not available_cols:
        return tissues[0], f"nTPM_{tissues[0]}"

    available_tissues = [col.replace("nTPM_", "") for col in available_cols]
    if len(available_cols) == 1:
        return available_tissues[0], available_cols[0]

    # Build member matrix and identify category-enriched genes
    member_matrix = np.column_stack([
        bulk_indexed[col].astype(float).reindex(genes).fillna(0.0).values
        for col in available_cols
    ])
    all_ntpm_cols = [c for c in bulk_indexed.columns if c.startswith("nTPM_")]
    background_median = np.median(
        bulk_indexed[all_ntpm_cols].astype(float).reindex(genes).fillna(0.0).values,
        axis=1,
    )
    # A gene is "category-enriched" if any member exceeds 2× the background
    category_max = member_matrix.max(axis=1)
    enriched = (category_max > background_median * 2) & (category_max > 1.0)

    sample_vec = np.array(
        [float(sample_by_eid.get(gid, 0.0)) for gid in genes], dtype=float
    )

    best_idx = 0
    best_corr = -np.inf
    for col_idx in range(member_matrix.shape[1]):
        ref_vec = member_matrix[:, col_idx]
        mask = enriched & ((sample_vec > 0.5) | (ref_vec > 0.5))
        if mask.sum() < 10:
            continue
        corr = float(np.corrcoef(
            np.log1p(sample_vec[mask]),
            np.log1p(ref_vec[mask]),
        )[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_idx = col_idx

    return available_tissues[best_idx], available_cols[best_idx]


def build_signature_matrix(components, gene_subset=None, sample_by_eid=None):
    """Build a signature matrix for the given component list.

    Parameters
    ----------
    components : list of str
        Component names (e.g. ["T_cell", "B_cell", "fibroblast", ...]).
    gene_subset : set of str, optional
        If provided, only include these Ensembl gene IDs.
    sample_by_eid : dict, optional
        Sample expression {ensembl_id: TPM}.  Used for composite tissue
        categories to select the best-matching member tissue.  If None,
        composite categories fall back to the first member tissue.

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

    hpa = _load_hpa_cell_types().drop_duplicates(subset="Ensembl_Gene_ID").copy()
    bulk = _load_normal_tissue_expression().drop_duplicates(subset="Ensembl_Gene_ID").copy()

    gene_ids = set(hpa["Ensembl_Gene_ID"].astype(str))
    gene_ids |= set(bulk["Ensembl_Gene_ID"].astype(str))
    if gene_subset is not None:
        gene_ids &= {str(gene_id) for gene_id in gene_subset}

    genes = sorted(gene_ids)
    symbol_map = dict(zip(hpa["Ensembl_Gene_ID"].astype(str), hpa["Symbol"].astype(str)))
    symbol_map.update(dict(zip(bulk["Ensembl_Gene_ID"].astype(str), bulk["Symbol"].astype(str))))
    symbols = [symbol_map.get(gene_id, gene_id) for gene_id in genes]

    hpa = hpa.set_index(hpa["Ensembl_Gene_ID"].astype(str))
    bulk = bulk.set_index(bulk["Ensembl_Gene_ID"].astype(str))

    cols = []
    for comp in components:
        # Priority 0: explicit matched-normal-epithelium convention,
        # `matched_normal_<tissue>` — used by `get_template_components`
        # for solid_primary + cancer_type to admit admixed normal-tissue
        # glands as a non-tumor compartment. Directly resolves to the
        # bulk nTPM tissue column.
        if comp.startswith("matched_normal_"):
            tissue = comp[len("matched_normal_"):]
            tissue_col = f"nTPM_{tissue}"
            if tissue_col in bulk.columns:
                vals = (
                    bulk[tissue_col]
                    .astype(float)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )
                cols.append(vals)
                continue

        # Priority 1: composite tissue category (best-match selection)
        category = COMPONENT_TO_CATEGORY.get(comp)
        if category is not None:
            if sample_by_eid is not None:
                _tissue, tissue_col = _select_best_category_tissue(
                    category, sample_by_eid, genes, bulk,
                )
            else:
                first_tissue = TISSUE_CATEGORIES[category][0]
                tissue_col = f"nTPM_{first_tissue}"
            if tissue_col in bulk.columns:
                vals = (
                    bulk[tissue_col]
                    .astype(float)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )
                cols.append(vals)
                continue

        # Priority 2: explicit bulk tissue override (legacy, for any
        # components not yet migrated to COMPONENT_TO_CATEGORY)
        bulk_tissues = COMPONENT_TO_BULK_TISSUE.get(comp, [])
        if bulk_tissues:
            bulk_cols = [f"nTPM_{tissue}" for tissue in bulk_tissues if f"nTPM_{tissue}" in bulk.columns]
            if bulk_cols:
                vals = (
                    bulk[bulk_cols]
                    .astype(float)
                    .fillna(0.0)
                    .mean(axis=1)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )
                cols.append(vals)
                continue

        # Priority 3: HPA single-cell reference
        hpa_types = COMPONENT_TO_HPA.get(comp, [])
        if hpa_types:
            present = [t for t in hpa_types if t in hpa.columns]
            if present:
                vals = (
                    hpa[present]
                    .astype(float)
                    .fillna(0.0)
                    .mean(axis=1)
                    .reindex(genes)
                    .fillna(0.0)
                    .values
                )
                cols.append(vals)
                continue

        # Unknown or unavailable component → zeros
        cols.append(np.zeros(len(genes)))

    matrix = np.column_stack(cols)
    return genes, symbols, matrix, list(components)
