from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pirlygenes import load_dataset as ld
from pirlygenes import gene_sets_cancer as gsc


def test_get_data_from_provided_dict():
    fake = {"abc.csv": pd.DataFrame({"x": [1]})}
    df = ld.get_data("abc", _dataframes_dict=fake)
    assert list(df.columns) == ["x"]


def test_get_data_missing_raises():
    with pytest.raises(ValueError):
        ld.get_data("does-not-exist", _dataframes_dict={})


def test_get_data_returns_copy_not_cached_reference():
    # Mutating the returned DataFrame must not corrupt the cached copy that
    # subsequent callers will get (issue #29).
    first = ld.get_data("ADC-trials")
    first["_should_not_persist"] = 1
    first.drop(first.index, inplace=True)
    second = ld.get_data("ADC-trials")
    assert "_should_not_persist" not in second.columns
    assert len(second) > 0


def test_get_data_copy_false_returns_shared_cached_frame():
    # copy=False is the read-only fast path (#278): it must return the SAME
    # cached object (no full-frame copy), while the default still defends the
    # cache. Read-only callers rely on this; mutating callers must not use it.
    shared = ld.get_data("ADC-trials", copy=False)
    shared_again = ld.get_data("ADC-trials", copy=False)
    assert shared is shared_again  # identity: no copy taken
    defensive = ld.get_data("ADC-trials")
    assert defensive is not shared  # default path still copies


def test_get_all_csv_paths_contains_core_dataset():
    paths = ld.get_all_csv_paths()
    assert any(Path(p).name == "ADC-trials.csv" for p in paths)


def test_expression_effect_rule_datasets_load():
    fusion_rules = gsc.fusion_expression_effect_rules_df()
    mutation_rules = gsc.mutation_expression_effect_rules_df()
    direct_fusion_rules = gsc.rare_cancer_fusion_rules_df()

    assert {"rule_id", "gene_a", "gene_b", "expected_up_genes"}.issubset(
        set(fusion_rules.columns)
    )
    assert {"rule_id", "alteration", "expected_up_genes"}.issubset(
        set(mutation_rules.columns)
    )
    assert "nutm1_rearranged_uncertain" in set(direct_fusion_rules["rule_id"])


def test_gene_set_field_lookup_variants(monkeypatch):
    # exercises plural/lower/upper/no-underscore candidate expansion
    fake_df = pd.DataFrame({"TUMORTARGETSYMBOLS": ["A;B"]})
    monkeypatch.setattr(gsc, "get_data", lambda name: fake_df)
    out = gsc.get_field_from_gene_set("x", ["Tumor_Target_Symbol"])
    assert out == {"A", "B"}


def test_all_gene_set_wrappers(monkeypatch):
    df_generic = pd.DataFrame(
        {
            "Symbol": ["GENE1;GENE2"],
            "Gene_ID": ["ENSG1;ENSG2"],
            "Tumor_Target_Symbols": ["GENE3"],
            "Tumor_Target_Ensembl_Gene_IDs": ["ENSG3"],
        }
    )

    def fake_get_data(name):
        return df_generic

    monkeypatch.setattr(gsc, "get_data", fake_get_data)

    # ADC
    assert gsc.therapy_target_gene_names("ADC-trials")
    assert gsc.therapy_target_gene_ids("ADC-trials")
    assert gsc.therapy_target_gene_names("ADC-approved")
    assert gsc.therapy_target_gene_ids("ADC-approved")
    assert gsc.therapy_target_gene_names("ADC")
    assert gsc.therapy_target_gene_ids("ADC")

    # TCR-T
    assert gsc.therapy_target_gene_names("TCR-T-trials")
    assert gsc.therapy_target_gene_ids("TCR-T-trials")
    assert gsc.therapy_target_gene_names("TCR-T")
    assert gsc.therapy_target_gene_ids("TCR-T")

    # CAR-T
    assert gsc.therapy_target_gene_names("CAR-T")
    assert gsc.therapy_target_gene_ids("CAR-T")

    # MuTE
    assert gsc.therapy_target_gene_names("multispecific-TCE")
    assert gsc.therapy_target_gene_ids("multispecific-TCE")

    # Bispecifics
    assert gsc.therapy_target_gene_names("bispecific-antibodies")
    assert gsc.therapy_target_gene_ids("bispecific-antibodies")
    assert gsc.therapy_target_gene_id_to_name("bispecific-antibodies-approved")
    assert gsc.therapy_target_gene_id_to_name("CAR-T-approved")

    # Radioligand + CTA
    assert gsc.therapy_target_gene_names("radioligand")
    assert gsc.therapy_target_gene_ids("radioligand")
    assert gsc.CTA_gene_names()
    assert gsc.CTA_gene_ids()


def test_cta_filtered_and_evidence():
    # CTA_gene_names() = filtered + expressed (excludes never_expressed)
    expressed_names = gsc.CTA_gene_names()
    expressed_ids = gsc.CTA_gene_ids()
    # CTA_filtered includes never_expressed
    filtered_names = gsc.CTA_filtered_gene_names()
    filtered_ids = gsc.CTA_filtered_gene_ids()
    # never_expressed = filtered - expressed
    never_expr_names = gsc.CTA_never_expressed_gene_names()
    never_expr_ids = gsc.CTA_never_expressed_gene_ids()
    # unfiltered = full superset
    all_names = gsc.CTA_unfiltered_gene_names()
    all_ids = gsc.CTA_unfiltered_gene_ids()
    # excluded = fail filter
    excluded_names = gsc.CTA_excluded_gene_names()

    assert expressed_names
    assert filtered_names
    assert all_names
    assert expressed_names < filtered_names  # expressed is strict subset of filtered
    assert filtered_names < all_names  # filtered is strict subset of unfiltered
    assert expressed_names & never_expr_names == set()  # no overlap
    assert expressed_names | never_expr_names == filtered_names  # partition
    assert filtered_names | excluded_names == all_names  # partition
    assert filtered_names & excluded_names == set()  # no overlap

    evidence_df = gsc.CTA_evidence()
    assert len(evidence_df) == len(all_names)
    expected_cols = [
        "protein_reproductive",
        "protein_thymus",
        "protein_reliability",
        "rna_reproductive",
        "rna_thymus",
        "protein_strict_expression",
        "rna_reproductive_frac",
        "rna_reproductive_and_thymus_frac",
        "rna_deflated_reproductive_frac",
        "rna_deflated_reproductive_and_thymus_frac",
        "rna_80_pct_filter",
        "rna_90_pct_filter",
        "rna_95_pct_filter",
        "rna_98_pct_filter",
        "rna_99_pct_filter",
        "passes_filters",
        "filtered",
        "source_databases",
        "biotype",
        "Canonical_Transcript_ID",
        "rna_max_ntpm",
        "never_expressed",
    ]
    for col in expected_cols:
        assert col in evidence_df.columns, f"Missing column: {col}"
    assert (
        evidence_df["passes_filters"].astype(str).str.lower()
        == evidence_df["filtered"].astype(str).str.lower()
    ).all()
    assert "XAGE1B" in filtered_names


def test_cta_accessors_delegate_to_tsarina():
    # tsarina is the single source of truth (#289/#290/#291); pirlygenes
    # re-exports its evidence / gene-set / partition accessors. Assert the
    # delegation is identity, not a re-implementation that could drift.
    import tsarina.evidence as _tev
    import tsarina.gene_sets as _tgs
    import tsarina.partition as _tp

    assert gsc.CTA_gene_names() == _tgs.CTA_gene_names()
    assert gsc.CTA_filtered_gene_names() == _tgs.CTA_filtered_gene_names()
    assert gsc.CTA_unfiltered_gene_names() == _tgs.CTA_unfiltered_gene_names()
    assert gsc.CTA_gene_names is _tgs.CTA_gene_names
    assert gsc.CTAPartitionSets is _tp.CTAPartitionSets
    # CTA_evidence() returns tsarina's table verbatim
    pd_testing = __import__("pandas").testing
    pd_testing.assert_frame_equal(gsc.CTA_evidence(), _tev.CTA_evidence())


def test_cta_gene_id_to_name_preserves_row_pairing():
    mapping = gsc.CTA_gene_id_to_name()
    assert mapping["ENSG00000181323"] == "SPEM1"
    assert mapping["ENSG00000230594"] == "CT47A4"
    assert mapping["ENSG00000236126"] == "CT47A3"


def test_cta_partition():
    # gene_ids
    p = gsc.CTA_partition_gene_ids()
    assert isinstance(p, gsc.CTAPartitionSets)
    assert len(p.cta) > 200
    assert len(p.non_cta) > 15000
    assert p.cta & p.cta_never_expressed == set()
    assert p.cta & p.non_cta == set()
    assert p.cta_never_expressed & p.non_cta == set()

    # gene_names
    p2 = gsc.CTA_partition_gene_names()
    assert isinstance(p2, gsc.CTAPartitionSets)
    assert "MAGEA4" in p2.cta
    assert "TP53" in p2.non_cta

    # dataframes
    p3 = gsc.CTA_partition_dataframes()
    assert isinstance(p3, gsc.CTAPartitionDataFrames)
    assert "rna_deflated_reproductive_frac" in p3.cta.columns
    assert "Ensembl_Gene_ID" in p3.non_cta.columns

    # cta_excluded genes are in non_cta
    excluded_ids = gsc.CTA_excluded_gene_ids()
    assert excluded_ids.issubset(p.non_cta)


# ── Externalized gene sets (mitochondrial / culture / TME / degradation /
#    lineage / cancer-family) ─────────────────────────────────────────────


def test_mitochondrial_gene_loaders():
    df = gsc.mitochondrial_genes_df()
    # 13 protein-coding OXPHOS subunits + 2 rRNAs + 22 tRNAs = 37
    # (full mt-DNA gene set after #242 folded the previously-derived
    # qc-mt-dna panel back into this curated CSV).
    assert len(df) == 37
    assert set(df["Role"]) == {"protein_coding", "rRNA", "tRNA"}
    assert df["Ensembl_Gene_ID"].notna().all()
    assert "MT-CO1" in gsc.mitochondrial_gene_names()
    assert "MT-RNR1" in gsc.mitochondrial_gene_names(role="rRNA")
    assert "MT-TA" in gsc.mitochondrial_gene_names(role="tRNA")
    assert "MT-CO1" not in gsc.mitochondrial_gene_names(role="rRNA")
    assert len(gsc.mitochondrial_gene_ids()) == 37
    assert len(gsc.mitochondrial_gene_ids(role="protein_coding")) == 13
    assert len(gsc.mitochondrial_gene_ids(role="rRNA")) == 2
    assert len(gsc.mitochondrial_gene_ids(role="tRNA")) == 22


def test_culture_stress_gene_loaders():
    df = gsc.culture_stress_genes_df()
    assert len(df) >= 20
    assert "HSPA1A" in gsc.culture_stress_gene_names()
    assert "HSPA1A" in gsc.culture_stress_gene_names(category="HSP")
    assert "LDHA" not in gsc.culture_stress_gene_names(category="HSP")


def test_tme_marker_loaders():
    df = gsc.tme_markers_df()
    assert len(df) >= 15
    expected_cell_types = {"T_cell", "B_cell", "myeloid", "fibroblast", "endothelial"}
    assert expected_cell_types.issubset(set(df["Cell_Type"]))
    assert "CD3D" in gsc.tme_marker_gene_names(cell_type="T_cell")
    assert "CD3D" not in gsc.tme_marker_gene_names(cell_type="fibroblast")


def test_degradation_gene_pairs_loader():
    pairs = gsc.degradation_gene_pairs()
    assert len(pairs) == 20
    short, long, ratio = pairs[0]
    assert isinstance(short, str) and isinstance(long, str)
    assert 0.5 < ratio < 1.5


def test_lineage_gene_loaders_cover_all_tcga_codes():
    from pirlygenes.gene_sets_cancer import TCGA_MEDIAN_PURITY

    by_code = gsc.lineage_genes_by_cancer_type()
    missing = [c for c in TCGA_MEDIAN_PURITY if c not in by_code]
    assert missing == [], f"Missing lineage genes for: {missing}"
    # PRAD sanity: classic prostate markers
    prad = set(gsc.lineage_gene_symbols("PRAD"))
    assert {"KLK3", "FOLH1", "TMPRSS2"}.issubset(prad)


def test_cancer_family_panel_loader():
    families = gsc.cancer_family_panels()
    expected = {
        # adult carcinoma lineages. MESENCHYMAL was removed in #452: its panel
        # was stroma/CAF markers (TME signal present in every solid tumor, not a
        # tumor lineage) — stroma now lives in tme-markers.csv, and sarcoma
        # lineage is carried by the SARC subtype key-genes layer.
        "PROSTATE",
        "CRC",
        "GASTRIC",
        "SQUAMOUS",
        "ESCA_SQ",
        "RENAL",
        "GLIAL",
        "MELANOCYTIC",
        # adenocarcinoma families added for #452 (previously no family signal,
        # which structurally disadvantaged them in lineage ranking)
        "LUAD",
        "BRCA",
        "PAAD",
        "LIHC",
        "OV",
        "UCEC",
        "BLCA",
        "THCA",
        # lineage families added for #351 (neuroendocrine / hematolymphoid /
        # embryonal / germ-cell / CNS-embryonal) — previously-unanchored classes
        "NEUROENDOCRINE",
        "HEME_BCELL",
        "HEME_TCELL",
        "HEME_MYELOID",
        "HEME_PLASMA",
        "EMBRYONAL",
        "GERM_CELL",
        "CNS_EMBRYONAL",
        # CNS fill-in (#357): ependymal + non-glial CNS-location lineages
        "EPENDYMAL",
        "SELLAR_EPITHELIAL",
        "MENINGIOMA",
        "NERVE_SHEATH",
        "CHOROID_PLEXUS",
    }
    assert set(families) == expected
    assert "KLK3" in gsc.cancer_family_panel("PROSTATE")
    assert "GFAP" not in gsc.cancer_family_panel("PROSTATE")


def test_lineage_family_panels_351_have_expected_markers():
    """The #351 lineage panels carry canonical lineage markers and resolvable
    unversioned ENSG ids (so trufflepig can family-gate NE / heme / embryonal /
    germ-cell / CNS-embryonal tumors that previously had no family anchor)."""
    assert "SYP" in gsc.cancer_family_panel("NEUROENDOCRINE")
    assert "INSM1" in gsc.cancer_family_panel("NEUROENDOCRINE")
    assert "MS4A1" in gsc.cancer_family_panel("HEME_BCELL")
    assert "CD3D" in gsc.cancer_family_panel("HEME_TCELL")
    assert "MPO" in gsc.cancer_family_panel("HEME_MYELOID")
    assert "SDC1" in gsc.cancer_family_panel("HEME_PLASMA")
    assert "POU5F1" in gsc.cancer_family_panel("GERM_CELL")
    # every member of the new families resolves to an unversioned ENSG id
    new_fams = ["NEUROENDOCRINE", "HEME_BCELL", "HEME_TCELL", "HEME_MYELOID",
                "HEME_PLASMA", "EMBRYONAL", "GERM_CELL", "CNS_EMBRYONAL"]
    df = gsc.cancer_family_panels_df()
    sub = df[df["Family"].isin(new_fams)]
    assert (sub["Ensembl_Gene_ID"].str.match(r"^ENSG\d+$")).all()


def test_cancer_family_group_hierarchy_353():
    """family_group makes the fine->coarse taxonomy explicit (#353): fine panels
    score, family_group is the penalty boundary. ESCA_SQ rolls into SQUAMOUS, the
    four heme panels into HEMATOLYMPHOID; CNS_EMBRYONAL is its own group (kept
    separate from GLIAL/CNS for the embryonal-vs-glial hard veto)."""
    groups = gsc.cancer_family_groups()
    # every fine Family has a group; distinct tissues group to themselves
    assert groups["PROSTATE"] == "PROSTATE"
    assert groups["NEUROENDOCRINE"] == "NEUROENDOCRINE"
    # roll-ups
    assert groups["ESCA_SQ"] == "SQUAMOUS"
    assert {groups[f] for f in
            ("HEME_BCELL", "HEME_TCELL", "HEME_MYELOID", "HEME_PLASMA")} == {"HEMATOLYMPHOID"}
    # CNS coarse group holds the glial-adjacent fine panels (GLIAL + EPENDYMAL):
    # ependymoma doesn't hard-veto glioma; scoring (L1CAM/EMA/NHERF1) separates.
    assert groups["GLIAL"] == "CNS"
    assert groups["EPENDYMAL"] == "CNS"
    # #357: the primitive/stem lineages share a group (least-separable cluster ->
    # scoring picks, no internal hard-veto) but stay != CNS so MBL still vetoes GBM.
    assert {groups[f] for f in ("EMBRYONAL", "GERM_CELL", "CNS_EMBRYONAL")} == {"PRIMITIVE"}
    assert groups["CNS_EMBRYONAL"] != groups["GLIAL"]   # MBL != GBM preserved
    # non-glial CNS-location lineages hard-veto glioma (own groups)
    for f in ("SELLAR_EPITHELIAL", "MENINGIOMA", "NERVE_SHEATH", "CHOROID_PLEXUS"):
        assert groups[f] == f and groups[f] != "CNS"
    # SOX2 dropped from GLIAL (shared with squamous/embryonal — not discriminating)
    assert "SOX2" not in gsc.cancer_family_panel("GLIAL")
    # every fine family is mapped + has a display name
    assert set(groups) == set(gsc.cancer_family_panels())
    assert gsc.cancer_family_display_names()["HEME_BCELL"]


def test_cancer_lineage_panel_loader():
    """Issue #266 lineage discrimination panels — pick the right
    child cohort once a parent family has won the first-pass scoring."""
    panels = gsc.cancer_lineage_panels()
    # Every parent family from issue #266 should be present
    assert {
        "SQUAMOUS", "GI_ADENO", "HEPATOBILIARY", "LUNG",
        "ENDOCRINE", "GU_RENAL", "GYNECOLOGIC_GLANDULAR",
        "NET", "BONE_EWS", "MESENCHYMAL",
    } <= set(panels)

    # BRCA_Basal is the canonical demo from the issue — SCGB2A2 /
    # mammaglobin should be its top mammary discriminator from
    # other squamous cancers. (Child_Code matches the registry's PAM50
    # code BRCA_Basal; the bladder discriminators use the registry BLCA.)
    brca_basal = dict(gsc.cancer_lineage_panel("BRCA_Basal"))
    assert brca_basal["SCGB2A2"] == "high"
    assert brca_basal["FOXA1"] == "high"

    # PRAD lineage panel: kallikreins + TMPRSS2 + FOLH1 (PSMA)
    prad = dict(gsc.cancer_lineage_panel("PRAD"))
    assert prad["KLK3"] == "high"
    assert prad["FOLH1"] == "high"

    # Negative-discrimination: LUSC has NKX2-1 LOW (distinguishes
    # from LUAD where NKX2-1 / TTF-1 is high).
    lusc = dict(gsc.cancer_lineage_panel("LUSC"))
    assert lusc["NKX2-1"] == "low"
    assert lusc["SOX2"] == "high"

    # Filter-by-family DataFrame view
    squamous = gsc.cancer_lineage_panels_df(family="SQUAMOUS")
    assert set(squamous["Child_Code"]) == {
        "BRCA_Basal", "BLCA", "ESCA", "HNSC", "LUSC", "CESC",
    }


def test_gene_loaders_are_exported_from_package():
    """Core panel loaders should be reachable via `from pirlygenes import ...`."""
    import pirlygenes

    for name in [
        "mitochondrial_gene_ids",
        "culture_stress_gene_names",
        "tme_markers_df",
        "degradation_gene_pairs",
        "lineage_genes_by_cancer_type",
        "cancer_family_panels",
    ]:
        assert hasattr(pirlygenes, name), f"{name} not re-exported"


def test_cta_symbol_for_alias_reexported_from_tsarina():
    """pirlygenes re-exports tsarina's alias resolver (tsarina#77)."""
    from pirlygenes.gene_sets_cancer import cta_symbol_for_alias

    assert cta_symbol_for_alias("NY-ESO-1") == "CTAG1B"
    assert cta_symbol_for_alias("CT12.2") == "XAGE2"
    assert cta_symbol_for_alias("not-a-gene") is None


def test_cta_axis_accessors_reexported_from_tsarina():
    """The CTA restriction-axis + by-axes accessors are re-exported too."""
    from pirlygenes.gene_sets_cancer import (
        CTA_by_axes,
        CTA_placental_restricted_gene_names,
        CTA_testis_restricted_gene_names,
    )

    assert "CTAG1B" in CTA_testis_restricted_gene_names()
    assert len(CTA_placental_restricted_gene_names()) > 0
    assert callable(CTA_by_axes)


def test_bispecific_antibody_target_gene_ids_name_is_consistent():
    """The deprecated _ids shim uses the singular 'target', matching its
    _names sibling and the rest of the family."""
    import pirlygenes.gene_sets_cancer as gsc

    assert hasattr(gsc, "bispecific_antibody_target_gene_ids")
    assert not hasattr(gsc, "bispecific_antibody_targets_gene_ids")


def test_cta_paralog_symbols_expands_alias_to_protein_group():
    """cta_paralog_symbols returns the whole interchangeable CTA family that
    cta_symbol_for_alias collapses to one canonical symbol."""
    from pirlygenes.gene_sets_cancer import cta_paralog_symbols

    # alias, official symbol, and the second paralog all map to the full group
    assert cta_paralog_symbols("NY-ESO-1") == ["CTAG1A", "CTAG1B"]
    assert cta_paralog_symbols("CTAG1B") == ["CTAG1A", "CTAG1B"]
    assert cta_paralog_symbols("CTAG1A") == ["CTAG1A", "CTAG1B"]
    assert cta_paralog_symbols("XAGE1") == ["XAGE1A", "XAGE1B"]
    assert set(cta_paralog_symbols("MAGEA3")) == {"MAGEA3", "MAGEA6"}
    # non-CTA / unknown -> empty
    assert cta_paralog_symbols("not-a-gene") == []
    assert cta_paralog_symbols("") == []


def test_cta_paralog_symbols_singleton_returns_self():
    """A CTA with no paralog group resolves to just its own official symbol."""
    from pirlygenes.gene_sets_cancer import cta_paralog_symbols, CTA_gene_names
    from pirlygenes.load_dataset import get_data

    grouped = set(get_data("cta-protein-groups")["member_symbol"].astype(str))
    singleton = next(c for c in CTA_gene_names() if c not in grouped)
    assert cta_paralog_symbols(singleton) == [singleton]


def test_stem_cell_marker_panels_355():
    """Stem-cell marker panels (#355) — a cell-STATE layer orthogonal to the
    lineage family gate: a core PLURIPOTENT panel + tissue-specific stem
    programs, each with resolvable unversioned ENSG ids."""
    panels = gsc.stem_cell_panels()
    assert set(panels) == {
        "PLURIPOTENT", "NEURAL_STEM", "HEMATOPOIETIC_STEM",
        "NEURAL_CREST", "MESENCHYMAL_EMT",
    }
    assert "POU5F1" in gsc.stem_cell_panel("PLURIPOTENT")
    assert "SOX10" in gsc.stem_cell_panel("NEURAL_CREST")
    assert "CD34" in gsc.stem_cell_panel("HEMATOPOIETIC_STEM")
    df = gsc.stem_cell_panels_df()
    assert (df["Ensembl_Gene_ID"].str.match(r"^ENSG\d+$")).all()
    assert set(df["program"]) == {"pluripotent", "tissue_specific"}


def test_reference_provenance_columns_are_categorical():
    """Regression guard for the shard-cache memory/load-time optimization: the
    pure-text provenance columns must load as ``category`` (a future change that
    drops one from the set, or a pandas upgrade that re-materializes objects,
    would otherwise silently regress the ~10x load speedup)."""
    df = ld.get_data("cancer-reference-expression", copy=False)
    for col in ld._LOW_CARDINALITY_METADATA_COLS:
        assert isinstance(df[col].dtype, pd.CategoricalDtype), col
        # builders always stamp these, so a NaN here would signal a load bug
        assert df[col].notna().all(), col


def test_categorize_metadata_is_a_lossless_encoding(tmp_path):
    """A ``category`` is only a codes+dictionary encoding — it must never change a
    value. Guards the specific worry that the dtype switch could alter the
    strings: every value (incl. NaN, unicode, long repeated strings) survives the
    cast AND a parquet round-trip byte-identical to the original object column."""
    raw = pd.DataFrame({
        "notes": ["a long provenance sentence — with unicode é/μ", "x", np.nan,
                  "a long provenance sentence — with unicode é/μ", "x"],
        "source_version": ["v1.0", "v1.0", "v2.0", "v1.0", np.nan],
        "processing_pipeline": ["p_a", "p_b", "p_a", "p_a", "p_b"],
        "cancer_code": ["AAA", "BBB", "AAA", "CCC", "BBB"],  # not in the set
    })
    before = raw.copy()
    out = ld._categorize_metadata(raw)
    for col in ld._LOW_CARDINALITY_METADATA_COLS:
        assert isinstance(out[col].dtype, pd.CategoricalDtype)
        pd.testing.assert_series_equal(
            out[col].astype(object), before[col].astype(object), check_names=False)
    # untouched column keeps its original dtype (whatever pandas inferred —
    # object on pandas 2, the new StringDtype on pandas 3) and is NOT categorical
    assert out["cancer_code"].dtype == before["cancer_code"].dtype
    assert not isinstance(out["cancer_code"].dtype, pd.CategoricalDtype)
    # survives the parquet cache round-trip unchanged
    path = tmp_path / "roundtrip.parquet"
    out.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    for col in ld._LOW_CARDINALITY_METADATA_COLS:
        pd.testing.assert_series_equal(
            back[col].astype(object), before[col].astype(object), check_names=False)


def test_cancer_compartment_panels_coarse_tier():
    """The coarsest lineage granularity: cell-of-origin compartments. Every
    compartment has a non-empty panel of resolvable ENSG markers, and the
    melanocytic/epithelial anchors are present (regression on the curated set)."""
    panels = gsc.cancer_compartment_panels()
    expected = {"EPITHELIAL", "MESENCHYMAL", "HEMATOLYMPHOID", "MELANOCYTIC",
                "NEURAL_GLIAL", "GERM_CELL", "NEUROENDOCRINE"}
    assert set(panels) == expected
    for comp, genes in panels.items():
        assert len(genes) >= 5, comp
    assert "MLANA" in panels["MELANOCYTIC"]
    assert "EPCAM" in panels["EPITHELIAL"]
    assert "PTPRC" in panels["HEMATOLYMPHOID"]
    df = gsc.cancer_compartment_panels_df()
    assert (df["Ensembl_Gene_ID"].str.match(r"^ENSG\d+$")).all()


def test_cancer_type_discriminators():
    """Pairwise contrastive sets separate confusable types; lookup is
    order-independent and a 'low' direction encodes the negative call."""
    df = gsc.cancer_type_discriminators_df()
    assert (df["Ensembl_Gene_ID"].str.match(r"^ENSG\d+$")).all()
    assert set(df["direction"]) <= {"high", "low"}
    # order-independent pair fetch
    a = gsc.cancer_type_discriminator("BLCA", "PRAD")
    b = gsc.cancer_type_discriminator("PRAD", "BLCA")
    assert a == b and set(a) == {"BLCA", "PRAD"}
    # uroplakins favour urothelial, prostate-secretory favours prostate
    assert any(s == "UPK2" for s, _ in a["BLCA"])
    assert any(s == "KLK3" for s, _ in a["PRAD"])
    # WT1 is high-favours-OV but low-favours-UCEC (the serous discriminator)
    ov = gsc.cancer_type_discriminator("OV", "UCEC")
    assert ("WT1", "high") in ov["OV"]
    assert ("WT1", "low") in ov["UCEC"]
    # unknown pair -> empty
    assert gsc.cancer_type_discriminator("BLCA", "GBM") == {}


def test_family_panel_marker_roles_and_verified_refs():
    """Each family marker carries a role (anchor/confirmatory/negative), a source
    (tumor/immune/stroma), and a PubMed-verified PMID. Positive accessors exclude
    negatives; negative markers are fetched separately."""
    df = gsc.cancer_family_panels_df()
    assert {"role", "source", "reference"} <= set(df.columns)
    assert set(df["role"]) == {"anchor", "confirmatory", "negative"}
    assert set(df["source"]) <= {"tumor", "immune", "stroma"}
    assert df["reference"].str.match(r"PMID:\d+").all()
    # positive panel (anchor+confirmatory) excludes negatives
    thca = gsc.cancer_family_panel("THCA")
    assert "TG" in thca and "CALCA" not in thca            # TG positive, CALCA negative
    # negatives fetched separately: CALCA high -> MTC not THCA
    neg = gsc.cancer_family_negative_markers("THCA")
    assert "CALCA" in neg and "TG" not in neg
    # confirmatory = promiscuous (PAX8 spans 4 lineages); anchor = specific
    pax8 = df[(df.Family == "THCA") & (df.Symbol == "PAX8")]["role"].iloc[0]
    assert pax8 == "confirmatory"
    tg = df[(df.Family == "THCA") & (df.Symbol == "TG")]["role"].iloc[0]
    assert tg == "anchor"
    # immune-infiltrate markers are flagged as source!=tumor
    assert (gsc.cancer_family_panels_df(family="HEME_TCELL")["source"] == "immune").any()


def test_cancer_classification_ontology_and_supertypes():
    """The supertype tier promotes promiscuous markers to anchors at their level,
    and the ontology node hierarchy walks compartment->supertype->family (a DAG)."""
    st = gsc.cancer_supertype_panels()
    assert "PAX8" in st["PAX8_LINEAGE"] and "NKX2-1" in st["TTF1_LINEAGE"]
    assert "GATA3" in st["LUMINAL_GATA3"] and "TP63" in st["SQUAMOUS_PROGRAM"]
    sdf = gsc.cancer_supertype_panels_df()
    assert (sdf["role"] == "anchor").all()
    assert sdf["reference"].str.match(r"PMID:\d+").all()
    onto = gsc.cancer_classification_ontology()
    assert set(onto["tier"]) == {"compartment", "supertype", "family"}
    # level-relative role: PAX8 is anchor@supertype AND confirmatory@family(THCA)
    famdf = gsc.cancer_family_panels_df(family="THCA")
    assert (famdf[famdf.Symbol == "PAX8"]["role"] == "confirmatory").all()
    # DAG: thyroid inherits BOTH PAX8 and TTF1 programs
    path = gsc.cancer_lineage_path("THCA")
    assert path[-1] == "THCA" and path[0] == "EPITHELIAL"
    assert "PAX8_LINEAGE" in path and "TTF1_LINEAGE" in path
    assert (onto[onto.node == "NEUROENDOCRINE"]["module"] == "cross_cutting").any()
