"""Tests for the expanded cancer-type registry.

The registry is a richer superset of TCGA — covers non-TCGA heme,
pediatric, NET, and rare entities, plus expression-based subtype rows
under TCGA umbrellas (BRCA × PAM50, LAML × ELN/APL, SARC × subtype,
LUAD × mutation class, SCLC × ASCL1/NEUROD1/POU2F3/YAP1, etc.).
"""

import pytest

from pirlygenes.gene_sets_cancer import (
    CANCER_TYPE_ALIASES,
    CANCER_TYPE_NAMES,
    cancer_type_registry,
    cancer_types_in_family,
    cancer_types_by_tissue,
    cancer_type_subtypes_of,
    resolve_cancer_type,
)


def _parent_codes(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [
        part.strip()
        for part in text.replace(";", ",").split(",")
        if part.strip()
    ]


def test_registry_has_required_columns():
    df = cancer_type_registry()
    required = {
        "code",
        "name",
        "family",
        "primary_tissue",
        "primary_template",
        "parent_code",
        "expression_source",
        "notes",
    }
    missing = required - set(df.columns)
    assert not missing, f"registry missing columns: {missing}"


def test_registry_codes_are_unique():
    df = cancer_type_registry()
    dupes = df["code"][df["code"].duplicated()].tolist()
    assert not dupes, f"duplicate codes in registry: {dupes}"


def test_no_registry_code_is_a_pre_rename_alias():
    """Every registry code must be canonical — never a key of
    _RENAMED_CODE_ALIASES (the old name that resolves to a new one). Guards
    against a builder/registry re-introducing a pre-rename code (e.g. MID_NET,
    NBL_MYCN_amp, LAML_ELN_Fav) that the canonicalizer would otherwise have to
    keep mopping up at write time. See #302."""
    from pirlygenes.gene_sets_cancer import _RENAMED_CODE_ALIASES

    codes = set(cancer_type_registry()["code"].astype(str))
    offenders = codes & set(_RENAMED_CODE_ALIASES)
    assert not offenders, f"registry still carries pre-rename codes: {sorted(offenders)}"


def test_fusion_surrogate_scope_codes_are_canonical():
    """Every code in fusion-surrogate-expression.csv's ';'-separated cancer_code
    scope must resolve to the registry, except the documented allowlist: the
    'pan_cancer' sentinel and 'ALCL' (a real ALK+ entity not yet modeled —
    tracked gap). Consumers match scope by exact string, so a non-canonical
    code (e.g. 'EHE' vs 'SARC_EHE') silently misses the rule."""
    from pirlygenes.load_dataset import get_data
    from pirlygenes.gene_sets_cancer import resolve_cancer_type

    allow = {"pan_cancer"}  # the only legitimate non-registry scope sentinel
    df = get_data("fusion-surrogate-expression")
    offenders = set()
    for scope in df["cancer_code"].astype(str):
        for code in (c.strip() for c in scope.split(";")):
            if not code or code in allow:
                continue
            try:
                resolve_cancer_type(code)
            except ValueError:
                offenders.add(code)
    assert not offenders, f"non-canonical fusion-surrogate scope codes: {sorted(offenders)}"


def test_fusion_surrogate_accessor_reads_current_schema():
    """Regression: the accessor's column names must match the CSV
    (cancer_code/surrogate_gene/surrogate_role). They had drifted to
    cancer_scope/gene/role, so the accessor silently returned [] for every
    input. A known cohort must yield its curated surrogates + the pan_cancer
    NTRK fallback."""
    from pirlygenes.gene_sets_cancer import fusion_surrogate_genes_for_cancer

    ews = {d["gene"] for d in fusion_surrogate_genes_for_cancer("SARC_EWS")}
    assert {"FATE1", "NR0B1"} <= ews            # EWSR1-FLI1 ectopic surrogates
    assert {"NTRK1", "NTRK2", "NTRK3"} <= ews   # pan_cancer NTRK fallback
    # canonicalized scope now reachable by the registry code
    ehe = {d["gene"] for d in fusion_surrogate_genes_for_cancer("SARC_EHE")}
    assert {"CTGF", "CYR61"} <= ehe


def test_degenerate_subtype_pairs_use_canonical_codes():
    """Every cancer code referenced in degenerate-subtype-pairs (members and
    tiebreaker_mapping) must be canonical — no pre-rename aliases that resolve
    away to a different code. Guards #287 (the NET tiebreaker_mapping still held
    PANNET/MID_NET/LUNG_NET_LC after the Phase-C rename)."""
    import re
    from pirlygenes.load_dataset import get_data
    from pirlygenes.gene_sets_cancer import _RENAMED_CODE_ALIASES

    d = get_data("degenerate-subtype-pairs")
    # Scan the code-bearing columns only (pair_id is an opaque identifier).
    blob = "\n".join(
        d[col].astype(str).str.cat(sep="\n")
        for col in ("members", "tiebreaker_mapping") if col in d.columns
    )
    offenders = sorted(
        old for old in _RENAMED_CODE_ALIASES
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", blob)
    )
    assert not offenders, f"degenerate-subtype-pairs holds pre-rename codes: {offenders}"


def test_expression_sources_yaml_uses_canonical_cancer_codes():
    """The expression-sources registry must reference canonical cancer codes
    only — no pre-rename aliases in any source's cancer_codes. See #302."""
    from pirlygenes import downloads
    from pirlygenes.gene_sets_cancer import _RENAMED_CODE_ALIASES

    offenders = {}
    for src in downloads.load_registry():
        bad = set(src.cancer_codes) & set(_RENAMED_CODE_ALIASES)
        if bad:
            offenders[src.id] = sorted(bad)
    assert not offenders, f"sources still list pre-rename cancer_codes: {offenders}"


def test_registry_covers_all_33_tcga_codes():
    """Every TCGA code must appear in the registry or we'll lose
    compatibility with existing cancer-type detection code paths."""
    df = cancer_type_registry()
    tcga_codes = {
        "ACC",
        "BLCA",
        "BRCA",
        "CESC",
        "CHOL",
        "COAD",
        "DLBC",
        "ESCA",
        "GBM",
        "HNSC",
        "KICH",
        "KIRC",
        "KIRP",
        "LAML",
        "LGG",
        "LIHC",
        "LUAD",
        "LUSC",
        "MESO",
        "OV",
        "PAAD",
        "PCPG",
        "PRAD",
        "READ",
        "SARC",
        "SKCM",
        "STAD",
        "TGCT",
        "THCA",
        "THYM",
        "UCEC",
        "UCS",
        "UVM",
    }
    registry_codes = set(df["code"])
    missing = tcga_codes - registry_codes
    assert not missing, f"registry missing TCGA codes: {missing}"


def test_registry_includes_non_tcga_heme():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("CLL", "MM", "MCL", "FL", "HL", "BL", "CML", "MDS", "MPN", "HCL"):
        assert need in codes, f"missing heme code: {need}"


def test_registry_includes_pediatric():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in (
        "SARC_OS",
        "SARC_EWS",
        "SARC_RMS_ERMS",
        "SARC_RMS_ARMS",
        "NBL",
        "WILMS",
        "RT",
        "MBL",
        "ATRT",
        "RB",
        "HEPB",
    ):
        assert need in codes, f"missing pediatric code: {need}"


def test_registry_includes_net_axis():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("NET_PANCREAS", "NET_MIDGUT", "NET_LUNG", "SCLC", "NEC_MERKEL"):
        assert need in codes, f"missing NET code: {need}"


def test_registry_includes_rare_entities():
    df = cancer_type_registry()
    codes = set(df["code"])
    for need in ("NUTM", "ADCC", "MTC", "SARC_CHOR", "NPC"):
        assert need in codes, f"missing rare code: {need}"


def test_brca_pam50_subtypes_present():
    """BRCA's expression-based PAM50 tiles must be in the registry so
    the second-pass subtype classifier can route to them."""
    subs = cancer_type_subtypes_of("BRCA")
    assert set(subs) == {
        "BRCA_LumA",
        "BRCA_LumB",
        "BRCA_HER2",
        "BRCA_Basal",
        "BRCA_Normal",
    }


def test_sarc_subtypes_cover_main_entities():
    """SARC subtypes must at minimum include the Tranche B tiles
    plus the known tumor-biology subtypes (MPNST, angiosarcoma,
    UPS)."""
    subs = set(cancer_type_subtypes_of("SARC"))
    required = {
        "SARC_LMS",
        "SARC_DDLPS",
        "SARC_MYXLPS",
        "SARC_SYN",
        "SARC_DSRCT",
        "SARC_GIST",
        "SARC_MPNST",
        "SARC_ANGIO",
        "SARC_UPS",
    }
    missing = required - subs
    assert not missing, f"SARC subtypes missing: {missing}"


def test_laml_has_apl_and_eln_tiles():
    subs = set(cancer_type_subtypes_of("LAML"))
    assert "LAML_APL" in subs
    # ELN2017 is the modern risk-stratification that gates transplant
    # vs chemo; must be representable as a subtype tile.
    for eln in ("LAML_ELNfav", "LAML_ELNint", "LAML_ELNadv"):
        assert eln in subs


def test_bone_tissue_returns_osteosarcoma_and_ewing():
    """Site-aware hypothesis: any sample suspected of bone origin
    should be able to enumerate OS + Ewing as candidates."""
    bone_cancers = set(cancer_types_by_tissue("bone"))
    assert "SARC_OS" in bone_cancers
    assert "SARC_EWS" in bone_cancers


def test_heme_myeloid_family_contains_laml_and_related():
    """Family grouping must pull LAML + its tiles + MDS + MPN + CML
    together — they share the heme_marrow / heme_blood templates and
    need joint candidate enumeration when sample mode is heme."""
    myeloid = set(cancer_types_in_family("heme-myeloid"))
    for need in ("LAML", "MDS", "MPN", "CML"):
        assert need in myeloid


def test_neuroendocrine_family_contains_sclc_and_pannet():
    # Phase-C: the `net` family was renamed `neuroendocrine` (it spans both
    # well-diff NET and poorly-diff NEC; the differentiation column carries
    # that split). SCLC keeps its famous code; pancreatic NET is NET_PANCREAS
    # (PANNET resolves via alias; PNET avoided — it means primitive
    # neuroectodermal tumor).
    ne = set(cancer_types_in_family("neuroendocrine"))
    assert "SCLC" in ne
    assert "NET_PANCREAS" in ne
    assert "NEC_MERKEL" in ne  # Merkel cell carcinoma (NEC)


def test_parent_codes_reference_registry_entries():
    """Every non-null parent_code must reference an existing code."""
    df = cancer_type_registry()
    codes = set(df["code"])
    parents = [p for value in df["parent_code"] for p in _parent_codes(value)]
    orphan = [p for p in parents if p not in codes]
    assert not orphan, f"parent_codes not in registry: {set(orphan)}"


def test_parent_graph_is_acyclic():
    """Subtype links are a tree/forest, not an arbitrary graph with cycles."""
    df = cancer_type_registry()
    parent_map = {
        str(row["code"]): _parent_codes(row["parent_code"])
        for _, row in df.iterrows()
    }

    def visit(code, path):
        for parent in parent_map.get(code, []):
            assert parent != code, f"{code} cannot be its own parent"
            assert parent not in path, (
                "cycle in cancer-type parent graph: "
                + " -> ".join(path + [code, parent])
            )
            visit(parent, path + [code])

    for code in parent_map:
        visit(code, [])


def test_child_rows_share_parent_family():
    """Child rows can refine tissue/template, but not jump cancer family."""
    df = cancer_type_registry()
    by_code = df.set_index("code")
    mismatches = []
    for _, row in df.iterrows():
        for parent in _parent_codes(row["parent_code"]):
            parent_row = by_code.loc[parent]
            if row["family"] != parent_row["family"]:
                mismatches.append((row["code"], row["family"], parent, parent_row["family"]))
    assert not mismatches, f"parent/child family mismatches: {mismatches}"


def test_sarc_prefixed_codes_are_in_sarc_subtree():
    """Codes in the SARC namespace should not become disconnected leaves."""
    df = cancer_type_registry()
    disconnected = []
    for _, row in df.iterrows():
        code = row["code"]
        if code.startswith("SARC_") and "SARC" not in _parent_codes(row["parent_code"]):
            disconnected.append(code)
    assert not disconnected, f"SARC_* codes without SARC parent: {disconnected}"


def test_phase_c_renamed_codes_resolve_via_backward_compat_alias():
    """Old codes retired by the Phase-C rename still resolve to their new code
    (case-insensitive), so external callers don't hard-break (issue #286)."""
    from pirlygenes.gene_sets_cancer import resolve_cancer_type
    pairs = {
        "OS": "SARC_OS", "EWS": "SARC_EWS", "CHON": "SARC_CHON",
        "CHOR": "SARC_CHOR", "GCTB": "SARC_GCTB", "ESS_LG": "SARC_ESS_LG",
        "ESS_HG": "SARC_ESS_HG", "RMS_ERMS": "SARC_RMS_ERMS",
        "RMS_ARMS": "SARC_RMS_ARMS", "RMS_PRMS": "SARC_RMS_PRMS",
        "RMS_SSRMS": "SARC_RMS_SSRMS", "HNSC_HPV_pos": "HNSC_HPVpos",
        "HNSC_HPV_neg": "HNSC_HPVneg",
    }
    for old, new in pairs.items():
        assert resolve_cancer_type(old) == new
        assert resolve_cancer_type(old.lower()) == new   # case-insensitive
        assert resolve_cancer_type(new) == new           # new code still resolves


def test_known_parent_reference_labels_are_connected():
    """Fine labels without direct references should retain parent fallback links."""
    expected = {
        "BRCA_Basal": "BRCA",
        "BRCA_HER2": "BRCA",
        "BRCA_LumA": "BRCA",
        "BRCA_LumB": "BRCA",
        "BRCA_Normal": "BRCA",
        "HNSC_HPVneg": "HNSC",
        "HNSC_HPVpos": "HNSC",
        "LUAD_EGFR": "LUAD",
        "LUAD_KRAS": "LUAD",
        "LUAD_STK11": "LUAD",
        "MBL_G3": "MBL",
        "MBL_G4": "MBL",
        "MBL_SHH": "MBL",
        "MBL_WNT": "MBL",
        "PCN": "MM",
        "SCLC_ASCL1": "SCLC",
        "SCLC_NEUROD1": "SCLC",
        "SCLC_POU2F3": "SCLC",
        "SCLC_YAP1": "SCLC",
    }
    df = cancer_type_registry().set_index("code")
    missing = []
    for code, parent in expected.items():
        if parent not in _parent_codes(df.loc[code, "parent_code"]):
            missing.append((code, parent))
    assert not missing, f"expected parent links missing: {missing}"


def test_registry_expression_sources_match_packaged_references():
    """The registry's source_cohort for each code must appear among the
    packaged references for that code. Additional cohorts (e.g. liver
    metastases beyond the primary-tumor reference) are allowed: the
    registry tracks only the canonical primary reference, not every
    shard. So this test checks 'registry-canonical cohort is present'
    rather than 'every packaged shard matches the registry'."""
    from pirlygenes.expression import available_cancer_expression_references

    df = cancer_type_registry().set_index("code")
    refs = available_cancer_expression_references()
    packaged_by_code: dict[str, set[str]] = {}
    for _, ref in refs.iterrows():
        packaged_by_code.setdefault(str(ref["cancer_code"]), set()).add(
            str(ref["source_cohort"])
        )

    mismatches = []
    for code, shards in packaged_by_code.items():
        if code not in df.index:
            continue
        registry_cohort = str(df.loc[code, "source_cohort"])
        if registry_cohort and registry_cohort not in shards:
            mismatches.append((code, registry_cohort, sorted(shards)))
    assert not mismatches, (
        f"registry source_cohort missing from packaged references: {mismatches}"
    )


def test_registry_has_source_cohort_column():
    """Every curated row carries the cohort that produced its expression
    median — enables downstream tracking of which cohort + paper each
    reference value came from."""
    df = cancer_type_registry()
    assert "source_cohort" in df.columns
    assert "source_pmid" in df.columns


def test_source_cohort_values_are_canonical():
    """source_cohort should only take values from the canonical
    cohort vocabulary — rejects typos like 'TCGA_BRCA' vs 'TCGA_XENA_TOIL'."""
    df = cancer_type_registry()
    valid = {
        "",
        # Phase C: the bare SARC code is a computed pan-sarcoma grand union
        # (no frozen shard), not a materialised cohort.
        "COMPUTED_PAN_SARCOMA",
        "TCGA_XENA_TOIL",
        "TCGA_BRCA_PAM50",
        "TCGA_HNSC",
        "TCGA_LUAD",
        "BEATAML_OHSU_2022",
        "CGCI_BLGSP",
        "GSE100026_DING_2017",
        "GSE114922_SHIOZAWA_2018",
        "GSE171811_ECCITE_CTCL",
        "GSE271664_BODOR_2025",
        "GSE283710_WASHU_2024",
        "TARGET_NBL_2018",
        "TARGET_OS_2020",
        "TARGET_RMS_2014",
        "TARGET_WT_2015",
        "TARGET_RT_2017",
        "TARGET_ALL_2018",
        "TARGET_UNSPECIFIED",
        "TARGET_AML_2018",
        "SCLC_UCOLOGNE_2015",
        "MMRF_COMMPASS",
        "CLLMAP_2022",
        "ICGC",
        "LITERATURE_CURATED",
        "TREEHOUSE_v25.01",
        "TREEHOUSE_POLYA_25_01",
        "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
        "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
        "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
        "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV",
        "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI",
        "TREEHOUSE_RIBOD_25_01",
        "UNC_NUTM1",
        "GSE118014_ALVAREZ_2018",
        "GSE98894_ALVAREZ_2018_NET",
        "GSE299759_MEIJER_2026",
        "GSE75885_DELESPAUL_2017",
        "GSE120328_LAMPRECHT_2018",
        "GSE142334_FL_TFL_2021",
        "GSE241095_KS_SKIN_2023",
        "GSE248751_HUMAN_CCS_2023",
        "GSE294016_BARTL_2025_SGC",
        "GSE328026_PECOMA_2026",
        "DRMETRICS_ALCALA_2019_LNEN",
        "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS",
        "SCLC_UCOLOGNE_2015_TF_DOMINANCE",
        "SCLC_UCOLOGNE_2015",
        # v5.5.0 microarray cohorts (build_microarray_source)
        "GSE32662_PRINGLE_2012_MTC",
        # v5.6.0 LPS: 4 subtypes from one Singer 2007 MSKCC microarray
        "GSE30929_SINGER_2007_LPS",
    }
    present = set(df["source_cohort"].fillna("").astype(str).unique())
    unknown = present - valid
    assert not unknown, f"unknown source_cohort values: {unknown}"


def test_expanded_sarcomas_present():
    """The 19 sarcoma additions (WHO therapy-distinct entities) must
    be in the registry so the second-pass subtype classifier can
    route to them."""
    df = cancer_type_registry()
    codes = set(df["code"])
    required = {
        "SARC_EPITH",
        "SARC_DFSP",
        "SARC_ASPS",
        "SARC_CCS",
        "SARC_IFS",
        "SARC_EHE",
        "SARC_PEC",
        "SARC_KS",
        "SARC_MYXFIB",
        "SARC_SFT",
        "SARC_IMT",
        "SARC_GCTB",
        "SARC_ESS_LG",
        "SARC_ESS_HG",
        "SARC_LGFMS",
        "SARC_EMC",
        "SARC_PLEOLPS",
        "SARC_RMS_PRMS",
        "SARC_RMS_SSRMS",
    }
    missing = required - codes
    assert not missing, f"expanded-sarcoma codes missing: {missing}"


def test_subtype_key_maps_sarc_subtypes_to_key_genes_entries():
    """The subtype_key column must match the actual subtype values
    used in cancer-key-genes.csv, otherwise the cancers CLI
    subcommand will report bm=0 / tg=0 for curated subtypes."""
    from pirlygenes.gene_sets_cancer import (
        cancer_biomarker_genes,
        cancer_therapy_targets,
    )

    df = cancer_type_registry()
    mapped = df[df["subtype_key"].fillna("").astype(str).ne("")]
    assert len(mapped) >= 7, "expected at least 7 rows with subtype_key populated"
    for _, row in mapped.iterrows():
        parent = row["parent_code"]
        subtype = row["subtype_key"]
        bm = cancer_biomarker_genes(parent, subtype=subtype)
        tg = cancer_therapy_targets(parent, subtype=subtype)
        assert len(bm) > 0 or len(tg) > 0, (
            f"subtype_key {parent}/{subtype} (code {row['code']}) has "
            f"no key-genes rows — either the subtype_key is wrong or "
            f"cancer-key-genes.csv is missing the tile"
        )


# `pirlygenes.cli` moved to `trufflepig.main`; the corresponding CLI
# coverage tests live in trufflepig/tests/test_cancer_type_registry.py.


def test_nutm_has_actionable_curation():
    """NUT carcinoma gets the fusion-partner biomarkers (NUTM1,
    BRD4, BRD3, NSD3) plus BET-inhibitor therapy rows — these were
    added because pirlygenes is applied to NUTM1-rearranged carcinoma samples."""
    from pirlygenes.gene_sets_cancer import (
        cancer_biomarker_genes,
        cancer_therapy_targets,
    )

    bm = cancer_biomarker_genes("NUTM")
    for gene in ("NUTM1", "BRD4", "BRD3", "MYC", "TP63"):
        assert gene in bm, f"NUTM biomarker missing: {gene}"
    tg = cancer_therapy_targets("NUTM")
    agents = set(tg["agent"].astype(str).str.lower())
    # At least one BET inhibitor must be present.
    assert any(
        "bet" in row.lower()
        or "bromodomain" in row.lower()
        or "molibresib" in row.lower()
        or "birabresib" in row.lower()
        or "bms-986158" in row.lower()
        for row in list(agents) + list(tg["rationale"].astype(str).str.lower())
    )


def test_primary_templates_are_declared_or_planned():
    """Every row has a primary_template — either an implemented
    template name or a ``primary_<tissue>`` name documented as planned
    (osteosarcoma, chondrosarcoma, adipose, etc.). Catches typos like
    ``primary_bones`` or blanks."""
    df = cancer_type_registry()
    templates = df["primary_template"].dropna().unique()
    # Every template must match the convention.
    for t in templates:
        assert (
            t == "solid_primary" or t.startswith("primary_") or t.startswith("heme_")
        ), f"unknown primary_template convention: {t}"


# ---------- resolve_cancer_type / CANCER_TYPE_NAMES ----------


def test_resolve_cancer_type_canonical_codes_passthrough():
    assert resolve_cancer_type("PRAD") == "PRAD"
    assert resolve_cancer_type("BRCA") == "BRCA"
    assert resolve_cancer_type("SARC") == "SARC"


def test_resolve_cancer_type_case_insensitive():
    assert resolve_cancer_type("prad") == "PRAD"
    assert resolve_cancer_type("Prad") == "PRAD"


def test_cancer_type_info_returns_canonical_record():
    from pirlygenes.gene_sets_cancer import cancer_type_info

    # any synonym/alias/display-name -> one canonical info dict
    for q in ("prostate", "Prostate Adenocarcinoma", "PRAD", "prad"):
        info = cancer_type_info(q)
        assert info["code"] == "PRAD"
        assert info["name"] == "Prostate Adenocarcinoma"
        assert info["family"] == "carcinoma-gu"
        assert info["primary_tissue"] == "prostate"
        assert info["burden_category"] == "prostate"
        assert isinstance(info["tmb"], float)  # parent-inherited where needed
    # renamed-code back-compat flows through
    assert cancer_type_info("OS")["code"] == "SARC_OS"
    # a newly-added (#294/#295) curated type resolves end-to-end
    bcc = cancer_type_info("BCC")
    assert bcc["family"] == "carcinoma-skin"
    assert bcc["burden_category"] == "non_melanoma_skin"
    # contract matches resolve_cancer_type: None passthrough, ValueError on junk
    assert cancer_type_info(None) is None
    with pytest.raises(ValueError):
        cancer_type_info("not-a-cancer-xyz")


def test_resolve_cancer_type_common_name_alias():
    assert resolve_cancer_type("prostate") == "PRAD"
    assert resolve_cancer_type("melanoma") == "SKCM"
    assert resolve_cancer_type("colorectal") == "COAD"


def test_resolve_cancer_type_subtype_codes_from_registry():
    """Subtype codes are valid via the registry CSV, not the hand-curated
    alias dict. Catches regressions where the resolver was capped at the
    33-code TCGA list."""
    assert resolve_cancer_type("BRCA_LumA") == "BRCA_LumA"
    assert resolve_cancer_type("LAML_APL") == "LAML_APL"


def test_resolve_cancer_type_display_name_lookup():
    """Display names from the registry resolve back to their code."""
    assert resolve_cancer_type("Prostate Adenocarcinoma") == "PRAD"


def test_resolve_cancer_type_none_passthrough():
    assert resolve_cancer_type(None) is None


def test_resolve_cancer_type_unknown_raises():
    with pytest.raises(ValueError):
        resolve_cancer_type("UNKNOWN_CODE_XYZ")


def test_resolve_cancer_type_empty_raises():
    with pytest.raises(ValueError):
        resolve_cancer_type("")


def test_cancer_type_names_view_is_registry_backed():
    """CANCER_TYPE_NAMES must equal exactly the set of registry rows
    with a non-empty ``name`` — proves the view isn't capped at the old
    33-code TCGA hardcoded dict AND that NaN/blank-name rows aren't
    silently leaking through as ``"nan"`` display strings."""
    df = cancer_type_registry()
    expected = set(
        df.loc[df["name"].notna() & df["name"].astype(str).str.strip().ne(""), "code"]
    )
    view_codes = set(CANCER_TYPE_NAMES.keys())
    assert view_codes == expected, (
        f"CANCER_TYPE_NAMES drifted from registry. "
        f"missing={expected - view_codes}; extra={view_codes - expected}"
    )


def test_cancer_type_names_view_never_returns_nan_string():
    """A registry row with a missing name must not surface as the
    literal string ``"nan"`` (latent bug fixed by the DataFrame-level
    NaN filter in ``_CancerTypeNamesView._load``)."""
    for code, name in CANCER_TYPE_NAMES.items():
        assert name != "nan", (
            f"CANCER_TYPE_NAMES[{code!r}] is the literal 'nan' string — "
            "registry row has a missing name that wasn't filtered"
        )
        assert name.strip() != "", (
            f"CANCER_TYPE_NAMES[{code!r}] is empty/whitespace"
        )


def test_cancer_type_aliases_all_resolve_to_valid_registry_codes():
    """Every alias value must be a real registry code — guards against
    drift between the hand-curated alias dict and the registry CSV."""
    registry_codes = set(cancer_type_registry()["code"])
    bad = {
        alias: code
        for alias, code in CANCER_TYPE_ALIASES.items()
        if code not in registry_codes
    }
    assert not bad, f"alias values not in registry: {bad}"


# ---------- post-5.0 polish: cache invalidation + reverse-lookup cache ----------


def test_clear_cache_resets_view_and_reverse_map():
    """``_clear_caches()`` must drop both the forward and reverse
    caches so tests that monkey-patch ``get_data`` see the new data."""
    from pirlygenes.gene_sets_cancer import _clear_caches

    # Warm both caches.
    _ = CANCER_TYPE_NAMES.get("PRAD")
    _ = CANCER_TYPE_NAMES._name_to_code()
    assert CANCER_TYPE_NAMES._cache is not None
    assert CANCER_TYPE_NAMES._name_to_code_cache is not None

    _clear_caches()

    assert CANCER_TYPE_NAMES._cache is None
    assert CANCER_TYPE_NAMES._name_to_code_cache is None

    # Subsequent access rebuilds both — and still resolves correctly.
    assert resolve_cancer_type("PRAD") == "PRAD"
    assert resolve_cancer_type("Prostate Adenocarcinoma") == "PRAD"


def test_name_to_code_reverse_map_is_cached():
    """The reverse map must build once and then return the same dict
    on every call (no per-call rebuild — that was the prior bottleneck
    inside ``resolve_cancer_type``)."""
    a = CANCER_TYPE_NAMES._name_to_code()
    b = CANCER_TYPE_NAMES._name_to_code()
    assert a is b, "reverse-lookup dict should be reused, not rebuilt"


def test_resolve_cancer_type_uses_cached_reverse_map_under_load():
    """Hot-loop sanity: 10k display-name lookups must be near-instant.
    Failure mode would be a regression that rebuilds ``name_to_code``
    inside ``resolve_cancer_type`` on every call."""
    import time

    # Warm.
    CANCER_TYPE_NAMES._name_to_code()

    t0 = time.perf_counter()
    for _ in range(10_000):
        resolve_cancer_type("Prostate Adenocarcinoma")
    elapsed = time.perf_counter() - t0
    # Generous bound — actual hot-loop is ~50ms on a laptop. A
    # regression that rebuilt the 125-entry dict per call would take
    # 500ms+. 2s gives plenty of slack for slower CI hardware.
    assert elapsed < 2.0, (
        f"display-name resolve appears uncached; 10k calls took {elapsed:.2f}s"
    )


def test_lineage_genes_direction_vocab_and_schema():
    """lineage-genes carries the optional expected-high/surprising 'direction'
    (high|low) + 'reference' columns (pass-2 schema extension)."""
    from pirlygenes.load_dataset import get_data

    d = get_data("lineage-genes")
    assert {"direction", "reference"} <= set(d.columns)
    assert set(d["direction"].astype(str)) <= {"high", "low"}


def test_marker_symbol_ensg_consistency():
    """Every (Symbol, Ensembl_Gene_ID) pair in the marker datasets must agree
    with the gene's current Ensembl symbol (allowing curated display aliases).
    Locks the 5 stale-symbol fixes (NIS->SLC5A5, PVRL1->NECTIN1, ...) found in
    the gene audit and prevents regression."""
    from pyensembl import EnsemblRelease
    from pirlygenes.gene_ids import strip_version
    from pirlygenes.gene_names import display_name
    from pirlygenes.load_dataset import get_data

    g = EnsemblRelease(112)
    mism = []
    for name in ("lineage-genes", "cancer-type-genes",
                 "cancer-lineage-panels", "cancer-surfaceome"):
        d = get_data(name)
        for sym, ensg in zip(d["Symbol"].astype(str), d["Ensembl_Gene_ID"].astype(str)):
            ensg = ensg.strip()
            if not ensg.startswith("ENSG"):
                continue
            try:
                cur = g.gene_by_id(strip_version(ensg)).gene_name
            except Exception:
                continue
            if cur.upper() != sym.strip().upper():
                try:
                    ok = display_name(cur).upper() == sym.strip().upper()
                except Exception:
                    ok = False
                if not ok:
                    mism.append(f"{name}: {sym} vs ENSG {ensg} ({cur})")
    assert not mism, "symbol↔ENSG mismatches:\n  " + "\n  ".join(mism)
