"""README Python API contract test.

Every name documented in the README's Python API code block must be
importable. Prevents stale doc drift like the 5.0.0–5.0.2 phantom
``gene_id_aliases`` import that broke any user who copy-pasted the
example.

If you change the README's Python API block, mirror the change here.
"""

from pirlygenes.gene_sets_cancer import (  # noqa: F401
    CANCER_TYPE_NAMES,
    cancer_family_panel,
    cancer_family_panels,
    cancer_surfaceome_gene_names,
    cancer_type_registry,
    CTA_gene_names,
    degenerate_subtype_pairs_df,
    degradation_gene_pairs,
    disease_state_rules_df,
    fusion_expression_effect_rules_df,
    fusion_surrogate_expression_df,
    housekeeping_gene_ids,
    lineage_genes_by_cancer_type,
    mitochondrial_gene_ids,
    mutation_expression_effect_rules_df,
    narrative_gene_set,
    narrative_gene_sets_df,
    rare_cancer_fusion_rules_df,
    rare_cancer_rna_surrogate_rules_df,
    resolve_cancer_type,
    surface_protein_gene_names,
    TCGA_MEDIAN_PURITY,
    therapy_target_gene_names,
    tme_marker_gene_ids,
)
from pirlygenes.load_dataset import get_data, get_all_csv_paths  # noqa: F401
from pirlygenes.gene_ids import (  # noqa: F401
    find_canonical_gene_ids_and_names,
    get_alias_as_list,
    get_reverse_alias_as_list,
)
from pirlygenes.gene_names import aliases, display_name, short_gene_name  # noqa: F401
from pirlygenes.gene_families import (  # noqa: F401
    gene_family_for_ensembl_id,
    gene_family_for_symbol,
    gene_family_ids,
    gene_family_names,
    gene_family_symbols,
    gene_family_table,
    hemoglobin_gene_ids,
    histone_gene_ids,
    immune_receptor_segment_ids,
    nuclear_retained_lncrna_ids,
    nuclear_retained_lncrna_symbols,
    numt_pseudogene_ids,
    numt_pseudogene_symbols,
    ribosomal_protein_ids,
    ribosomal_protein_pseudogene_ids,
    rrna_and_pseudogene_ids,
    rrna_and_pseudogene_symbols,
    small_noncoding_rna_ids,
)


def test_readme_python_api_block_is_importable():
    """Smoke test — if the import block at the top of this file
    succeeds at collection time, the README's documented API matches
    reality. The actual assertions live above as imports."""
    assert True


def test_readme_cancer_family_panel_example_returns_genes():
    """README documents valid family names: PROSTATE, CRC, GASTRIC,
    ESCA_SQ, SQUAMOUS, RENAL, GLIAL, MELANOCYTIC, and the #452 adenocarcinoma
    families (LUAD, BRCA, …). Catches regressions where these names diverge from
    cancer-family-panels.csv keys (and prevents another empty-list example like
    the v5.0.0 'sarcoma' typo). MESENCHYMAL was removed in #452 (it detected
    stroma/TME, not a tumor lineage)."""
    documented = {
        "PROSTATE", "CRC", "GASTRIC", "ESCA_SQ", "SQUAMOUS",
        "RENAL", "GLIAL", "MELANOCYTIC",
        # #452 adenocarcinoma lineage families
        "LUAD", "BRCA", "PAAD", "LIHC", "OV", "UCEC", "BLCA", "THCA",
        "NEUROENDOCRINE", "HEME_BCELL", "HEME_TCELL", "HEME_MYELOID",
        "HEME_PLASMA", "EMBRYONAL", "GERM_CELL", "CNS_EMBRYONAL",
        "EPENDYMAL", "SELLAR_EPITHELIAL", "MENINGIOMA", "NERVE_SHEATH",
        "CHOROID_PLEXUS",
    }
    actual = set(cancer_family_panels().keys())
    assert documented == actual, (
        f"README documented family names drifted from cancer-family-panels.csv. "
        f"missing={documented - actual}; extra={actual - documented}"
    )
    for name in documented:
        assert cancer_family_panel(name), (
            f"cancer_family_panel({name!r}) is empty"
        )


def test_readme_bundled_table_accessors_exist():
    """The 'Primary accessor' column in the bundled-data table promises
    real importable functions for every CSV that has a named accessor.
    This pins the promise so a future refactor that renames or removes
    one of them is caught here, not by a user puzzled at a missing
    function."""
    from pirlygenes.gene_sets_cancer import (
        cancer_family_panels_df,
        cancer_key_genes_df,
        cancer_surfaceome_evidence,
        cancer_type_gene_sets,
        cancer_types_by_tissue,
        cancer_types_in_family,
        cancer_type_subtypes_of,
        CTA_evidence,
        ffpe_sensitive_markers_df,
        surface_protein_evidence,
        therapy_target_gene_ids,
    )

    for fn in (
        cancer_family_panels_df,
        cancer_key_genes_df,
        cancer_surfaceome_evidence,
        cancer_type_gene_sets,
        cancer_types_by_tissue,
        cancer_types_in_family,
        cancer_type_subtypes_of,
        CTA_evidence,
        ffpe_sensitive_markers_df,
        surface_protein_evidence,
        therapy_target_gene_ids,
    ):
        assert callable(fn), f"{fn!r} is not callable"


def test_readme_top_level_shortcuts_resolve():
    """README claims the top-level ``from pirlygenes import …`` form
    works for any of the ~50 names in ``pirlygenes.__all__``. Spot-
    check a representative sample."""
    import pirlygenes

    for name in (
        "housekeeping_gene_ids",
        "tme_marker_gene_ids",
        "numt_pseudogene_ids",
        "mitochondrial_genes_df",
        "gene_family_for_ensembl_id",
    ):
        assert name in pirlygenes.__all__, f"{name!r} missing from __all__"
        assert hasattr(pirlygenes, name), (
            f"pirlygenes.{name} listed in __all__ but not actually exposed"
        )


def test_readme_therapy_modalities_match_registry():
    """README documents the therapy-target modality set; make sure
    none of them errors and that the dominant ones return non-empty
    lists."""
    # Every key in _THERAPY_REGISTRY should accept; this covers the
    # README's listed names plus the trial/approved sub-keys.
    from pirlygenes.gene_sets_cancer import _THERAPY_REGISTRY

    for modality in _THERAPY_REGISTRY:
        # Must not raise.
        names = therapy_target_gene_names(modality)
        # All modalities ship at least one row today.
        assert len(names) > 0, f"therapy_target_gene_names({modality!r}) is empty"
