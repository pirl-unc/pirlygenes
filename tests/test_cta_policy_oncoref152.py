"""Regression tests for OncoRef 1.52.0 active CTA policy and manifest adoption (#655)."""

from __future__ import annotations

import json
import warnings

import oncoref
from oncoref import antigen_sets, cta as legacy_cta, species_target_selection, target_selection
import pytest

import pirlygenes
from pirlygenes import coverage, cta_curation_plots, cta_policy, gene_sets_cancer as gsc


def test_active_cta_policy_id_and_manifest_match_live_oncoref():
    """Active CTA manifest/policy comes dynamically from oncoref.target_selection."""
    owner_meta = target_selection.metadata()

    policy_id = gsc.CTA_policy_id()
    manifest = gsc.CTA_policy_manifest()

    assert policy_id == target_selection.POLICY_ID
    assert manifest["policy_id"] == owner_meta["policy_id"] == target_selection.POLICY_ID
    assert manifest["oncoref_version"] == oncoref.__version__
    assert manifest["data_sha256"] == owner_meta["data_sha256"]
    assert manifest["source_snapshot_id"] == f"sha256:{owner_meta['data_sha256']}"
    assert manifest["source_surface"] == "oncoref.target_selection"
    assert manifest["ensembl_release"] == antigen_sets.ENSEMBL_RELEASE
    assert manifest["reference_id"] == f"homo_sapiens:GRCh38:Ensembl:{antigen_sets.ENSEMBL_RELEASE}"
    assert manifest["categories"] == list(target_selection.CATEGORIES)
    assert manifest["cta_tiers"] == list(target_selection.CTA_TIERS)
    assert manifest["sex_restriction_classes"] == list(
        species_target_selection.SEX_RESTRICTION_CLASSES
    )
    assert manifest["summary"]["categories"]["CTA"] == int(
        target_selection.results()["is_cta"].sum()
    )

    # Manifest must be detached so caller mutation cannot poison later calls.
    manifest["summary"]["categories"]["CTA"] = -1
    assert gsc.CTA_policy_manifest()["summary"]["categories"]["CTA"] > 0

    # cta_curation_plots exposes the same live policy receipt without dated-dir inference.
    assert cta_curation_plots.active_cta_policy_manifest()["policy_id"] == policy_id


def test_default_cta_membership_uses_active_policy_without_deprecation_warnings():
    """Default CTA accessors match active oncoref.target_selection with no legacy warnings."""
    owner_results = target_selection.results()
    expected_ids = {
        str(gid).split(".")[0]
        for members in owner_results.loc[owner_results["is_cta"], "member_gene_ids"]
        for gid in json.loads(members)
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids = gsc.CTA_gene_ids()
        names = gsc.CTA_gene_names()
        id_to_name = gsc.CTA_gene_id_to_name()
        filtered_ids = gsc.CTA_filtered_gene_ids()
        never_ids = gsc.CTA_never_expressed_gene_ids()

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings == []

    assert ids == expected_ids
    assert filtered_ids == ids
    assert never_ids == set()
    assert set(id_to_name) == ids
    assert set(id_to_name.values()) == names
    assert ids == set(oncoref.coverage.cta_gene_ids())
    assert id_to_name == oncoref.coverage.cta_gene_id_to_name()

    # Key OncoRef 1.52.0 membership contracts:
    # - CTAG2 (LAGE-1) is clinically cleared and admitted as an active CTA.
    # - Classic CTAs remain in active CTAs.
    # - SPEM1 and GPX5 are Reproductive Antigen only (RA, not CTA).
    # - TRIM64 and SPAG9 are Unsafe (not CTA).
    assert {"CTAG2", "CTAG1A", "CTAG1B", "MAGEA4", "PRAME", "XAGE1B"} <= names
    assert "SPEM1" not in names and "SPEM1" in gsc.CTA_reproductive_antigen_gene_names()
    assert "GPX5" not in names and "GPX5" in gsc.CTA_reproductive_antigen_gene_names()
    assert "TRIM64" not in names and "TRIM64" in gsc.CTA_excluded_gene_names()
    assert "SPAG9" not in names and "SPAG9" in gsc.CTA_excluded_gene_names()


def test_four_way_sex_restriction_partition_and_antigen_set_algebra():
    """OncoRef 1.52.0's 4-way CTA-M/F/MF/P partition is exact and disjoint."""
    part_ids = gsc.CTA_sex_partition_gene_ids()
    part_names = gsc.CTA_sex_partition_gene_names()
    assert tuple(part_ids) == ("CTA-M", "CTA-F", "CTA-MF", "CTA-P")
    assert tuple(part_names) == ("CTA-M", "CTA-F", "CTA-MF", "CTA-P")

    # Every bucket is non-empty, pairwise disjoint, and partitions CTA_gene_ids().
    for key, bucket in part_ids.items():
        assert bucket, f"{key} unexpectedly empty"
    keys = list(part_ids)
    for idx, k1 in enumerate(keys):
        for k2 in keys[idx + 1 :]:
            assert part_ids[k1].isdisjoint(part_ids[k2]), f"{k1} and {k2} overlap"
    assert set().union(*part_ids.values()) == gsc.CTA_gene_ids()
    assert set().union(*part_names.values()) == gsc.CTA_gene_names()

    # Legacy CTA-E alias resolves identically to CTA-P.
    assert gsc.CTA_antigen_set_gene_ids("CTA-E") == part_ids["CTA-P"]

    # CA / RA / CTA / CA|RA / Unsafe / Other set algebra.
    ca = gsc.CTA_cancer_antigen_gene_ids()
    ra = gsc.CTA_reproductive_antigen_gene_ids()
    cta_ids = gsc.CTA_gene_ids()
    candidates = gsc.CTA_target_candidate_gene_ids()
    unsafe = gsc.CTA_excluded_gene_ids()
    other = gsc.CTA_antigen_set_gene_ids("Other")

    assert cta_ids == (ca & ra)
    assert candidates == (ca | ra)
    assert candidates.isdisjoint(unsafe)
    assert candidates.isdisjoint(other)
    assert unsafe.isdisjoint(other)
    assert gsc.CTA_antigen_set_gene_ids("Unsafe|Other") == (unsafe | other)
    assert (candidates | unsafe | other) == gsc.CTA_unfiltered_gene_ids()

    with pytest.raises(ValueError, match="Unknown antigen set"):
        gsc.CTA_antigen_set_gene_ids("NOT_A_REAL_SET")


def test_cta_tiers_and_selection_df_filters_carry_provenance():
    """CTA_by_tier() and CTA_selection_df() expose live policy provenance and filters."""
    by_tier = {
        tier: gsc.CTA_by_tier(tier, column="Ensembl_Gene_ID")
        for tier in target_selection.CTA_TIERS
    }
    assert all(by_tier.values())
    assert set().union(*by_tier.values()) == gsc.CTA_gene_ids()
    assert "MAGEA4" in gsc.CTA_by_tier("core")
    assert "PRAME" in gsc.CTA_by_tier("core_with_extended_exposure")

    sel = gsc.CTA_selection_df(category="CTA", sex_restriction="CTA-P")
    assert not sel.empty
    assert (sel["category"] == "CTA").all()
    assert (sel["sex_restriction_class"] == "Pregnancy / peripartum-only").all()
    assert sel.attrs["policy_id"] == gsc.CTA_policy_id()
    assert sel.attrs["data_sha256"] == target_selection.metadata()["data_sha256"]

    with pytest.raises(ValueError, match="Unknown cta_tier"):
        gsc.CTA_by_tier("nonexistent_tier")
    with pytest.raises(ValueError, match="Unknown target_selection category"):
        gsc.CTA_selection_df(category="NotACategory")
    with pytest.raises(ValueError, match="Unknown sex_restriction"):
        gsc.CTA_selection_df(sex_restriction="NotASexSet")


def test_active_cta_evidence_carries_manifest_provenance_and_clinical_clearance():
    """CTA_evidence() exposes active OncoRef 1.52.0 policy columns, HPA, and Tsarina MS."""
    ev = gsc.CTA_evidence()
    assert ev.attrs["policy_id"] == gsc.CTA_policy_id()
    assert ev.attrs["oncoref_version"] == oncoref.__version__
    assert ev.attrs["source_surface"] == "oncoref.target_selection"
    assert set(ev["selection_policy_id"].unique()) == {gsc.CTA_policy_id()}
    assert set(ev.loc[ev["is_current_cta"], "Ensembl_Gene_ID"]) == gsc.CTA_gene_ids()

    by_id = ev.set_index("Ensembl_Gene_ID")
    ctag2 = by_id.loc["ENSG00000126890"]
    assert bool(ctag2["is_current_cta"]) is True
    assert ctag2["selection_category"] == "CTA"
    assert ctag2["selection_sex_restriction_code"] == "CTA-MF"
    assert ctag2["selection_clinical_clearance_scope"]
    assert "ms_restriction" in ev.columns


def test_historical_cta_helpers_and_curated_clinical_panels_remain_distinct():
    """Historical v1 HPA CTAs and PirlyGenes clinical panels remain explicitly preserved."""
    historical_ids = gsc.CTA_historical_gene_ids()
    assert historical_ids == set(legacy_cta.cta_gene_ids())
    assert len(historical_ids) == 624
    assert historical_ids != gsc.CTA_gene_ids()
    assert gsc.CTA_historical_gene_names() == set(legacy_cta.cta_gene_names())
    assert len(gsc.CTA_historical_evidence()) == len(legacy_cta.cta_evidence())

    # Curated clinical / tumor-marker panels remain owned by pirlygenes.
    assert not gsc.cancer_key_genes_df().empty
    assert "MAGEA4" in gsc.therapy_target_gene_names("TCR-T")
    assert "ERBB2" in gsc.therapy_target_gene_names("ADC-approved")
    assert "KLK3" in gsc.lineage_gene_symbols("PRAD")
    assert len(gsc.cancer_surfaceome_gene_names()) > 100


def test_coverage_resolve_gene_set_supports_active_cta_and_sex_sets():
    """coverage.resolve_gene_set resolves active CTA, 4-way sex sets, and explicit historical."""
    label, ids = coverage.resolve_gene_set("cta")
    assert label == "CTA" and ids == gsc.CTA_gene_ids()

    for short_token, expected_group in (
        ("cta-m", "CTA-M"),
        ("cta-f", "CTA-F"),
        ("cta-mf", "CTA-MF"),
        ("cta-p", "CTA-P"),
    ):
        lbl, subset = coverage.resolve_gene_set(short_token)
        assert lbl == expected_group
        assert subset == gsc.CTA_antigen_set_gene_ids(expected_group)

    lbl_ca, ca_ids = coverage.resolve_gene_set("cta:CA|RA")
    assert lbl_ca == "CA|RA" and ca_ids == gsc.CTA_target_candidate_gene_ids()

    lbl_core, core_ids = coverage.resolve_gene_set("cta:core")
    assert lbl_core == "CTA:core" and core_ids == gsc.CTA_by_tier("core", column="Ensembl_Gene_ID")

    lbl_hist, hist_ids = coverage.resolve_gene_set("cta:historical")
    assert lbl_hist == "CTA:historical" and hist_ids == gsc.CTA_historical_gene_ids()


def test_top_level_lazy_exports_include_cta_policy_surface():
    """Top-level pirlygenes lazily exposes the new CTA policy/manifest APIs."""
    assert pirlygenes.cta_policy is cta_policy
    assert pirlygenes.CTA_policy_id() == gsc.CTA_policy_id()
    assert pirlygenes.CTA_policy_manifest()["policy_id"] == gsc.CTA_policy_id()
    assert pirlygenes.CTA_gene_ids() == gsc.CTA_gene_ids()
    assert pirlygenes.CTA_sex_partition_gene_ids() == gsc.CTA_sex_partition_gene_ids()
