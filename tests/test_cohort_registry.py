"""First-class cohort vocabulary (#296) + source-prefixed atoms (#292)."""
from __future__ import annotations

import pirlygenes.gene_sets_cancer as gsc
from pirlygenes.cohorts import Cohort
from pirlygenes.expression.accessors import (
    available_cancer_expression_references,
    cancer_code_sources,
    source_prefixed_references,
)


def test_cohort_registry_schema_and_computed_aggregate():
    df = gsc.cohort_registry_df()
    required = {"cohort_id", "prefix", "kind", "source_project", "assay",
                "n_samples", "n_codes", "is_computed", "member_cohorts",
                "provenance"}
    assert required.issubset(df.columns)
    assert df["cohort_id"].is_unique
    # the computed pan-sarcoma union is a first-class cohort (the #296 point:
    # it's NOT in available_cancer_expression_references but IS a valid cohort).
    row = df.set_index("cohort_id").loc["COMPUTED_PAN_SARCOMA"]
    assert bool(row["is_computed"]) is True
    assert row["kind"] == "computed"
    members = str(row["member_cohorts"]).split(";")
    assert "SARC_LMS" in members and len(members) > 20
    assert members == gsc.cohort_aggregate_members("SARC")
    assert int(row["n_codes"]) == len(members)


def test_every_used_source_cohort_is_registered():
    """The whole point of #296: one authority. Every source_cohort that appears
    in the packaged shards OR the cancer-type registry must be in the cohort
    registry — so a consumer validating against it never rejects a real cohort
    (incl. COMPUTED_PAN_SARCOMA / LITERATURE_CURATED)."""
    known = gsc.known_cohort_ids()
    manifest = set(available_cancer_expression_references()["source_cohort"].astype(str))
    reg = set(gsc.cancer_type_registry()["source_cohort"].dropna().astype(str))
    missing = (manifest | reg) - known
    assert not missing, f"source_cohorts absent from cohort-registry: {sorted(missing)}"


def test_cohort_kind_and_prefix_documented():
    assert gsc.cohort_kind("TREEHOUSE_POLYA_25_01") == "treehouse"
    assert gsc.cohort_kind("BEATAML_OHSU_2022") == "beataml"
    assert gsc.cohort_kind("COMPUTED_PAN_SARCOMA") == "computed"
    assert gsc.cohort_kind("not-a-cohort") is None
    # microarray cohorts flagged distinctly from bulk RNA-seq
    assert gsc.cohort_registry()["GSE32662_PRINGLE_2012_MTC"]["assay"] == "microarray"


def test_artifact_only_source_is_registered_from_owner_availability():
    row = gsc.cohort_registry_df().set_index("cohort_id").loc[
        "GSE85383_YOSHIDA_2017_ESS"
    ]

    assert row["prefix"] == "GSE85383"
    assert row["kind"] == "geo"
    assert row["source_project"] == "GEO"
    assert row["n_samples"] == 13
    assert row["n_codes"] == 2
    assert "GSE85383 Yoshida 2017 ESS" in row["provenance"]
    assert "9 low-grade and 4 high-grade" in row["provenance"]


def test_new_summary_source_is_registered_from_owner_availability():
    row = gsc.cohort_registry_df().set_index("cohort_id").loc[
        "SRP493407_MMNST_2024"
    ]

    assert row["prefix"] == "SRP493407"
    assert row["kind"] == "sra"
    assert row["source_project"] == "NCBI SRA Gene Feature counts"
    assert row["n_samples"] == 3
    assert row["n_codes"] == 1
    assert "data_version=5.23.14" in row["provenance"]


def test_sparse_source_registry_keeps_physical_and_reference_counts_distinct():
    import oncoref

    registry = gsc.cohort_registry_df().set_index("cohort_id")
    expected = {
        "CGCI_BLGSP": (184, 175),
        "GSE328026_PECOMA_2026": (69, 60),
    }
    manifest = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="all",
        reference_source="summary_rows_all",
        all_sources=True,
    )

    for cohort_id, (source_samples, reference_samples) in expected.items():
        row = registry.loc[cohort_id]
        assert int(row["n_samples"]) == source_samples
        selected = manifest.loc[
            manifest["available"]
            & manifest["source_cohort"].astype(str).eq(cohort_id),
            "n_reference_samples",
        ]
        assert int(selected.sum()) == reference_samples


def test_salivary_registry_repairs_owner_code_count_from_reference_manifest():
    """The owner registry is stale in oncoref 1.8.159 (oncoref#448).

    Pirlygenes' compatibility snapshot must follow the released reference
    manifest until the dependency fixes its 123-sample/one-code row.
    """
    registry = gsc.cohort_registry_df().set_index("cohort_id")
    row = registry.loc["GSE294016_BARTL_2025_SGC"]

    assert int(row["n_samples"]) == 60
    assert int(row["n_codes"]) == 2


def test_source_prefixed_atoms_and_rollup():
    spr = source_prefixed_references()
    assert {"kind", "cohort_atom"}.issubset(spr.columns)
    # a multi-source code surfaces one atom per source kind
    ddlps = sorted(spr.loc[spr["cancer_code"] == "SARC_DDLPS", "cohort_atom"].unique())
    assert "geo:SARC_DDLPS" in ddlps and "treehouse:SARC_DDLPS" in ddlps
    # rollup keys off the resolved code (alias-proof) -> {kind: [cohort_id]}
    rollup = cancer_code_sources("PANNET")  # alias -> NET_PANCREAS
    assert "NET_PANCREAS" in rollup
    assert "geo" in rollup["NET_PANCREAS"]


def test_cohort_atom_is_source_prefixed():
    c = Cohort(code="SARC_LMS", stem="SARC_LMS", source_id="treehouse-polya-25-01")
    assert c.source_kind == "treehouse"
    assert c.atom == "treehouse:SARC_LMS"
