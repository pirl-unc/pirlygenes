"""MBL subgroup summary/artifact capability boundary (#341, oncoref#420)."""

from pirlygenes.expression import (
    available_percentile_cohorts,
    available_representative_cohorts,
    cancer_reference_expression,
    cohort_gene_percentiles,
    representative_cohort_samples,
)
from pirlygenes.gene_sets_cancer import cohort_registry_df


MBL_SUBGROUP_SAMPLES = {
    "MBL_WNT": 17,
    "MBL_SHH": 25,
    "MBL_G3": 44,
    "MBL_G4": 39,
}
MBL_SOURCE = "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS"


def test_mbl_subgroups_have_aligned_summaries_and_per_sample_artifacts():
    """The marker-derived partition exposes all delegated artifact families."""
    codes = set(MBL_SUBGROUP_SAMPLES)
    summaries = cancer_reference_expression(
        codes, genes=["TP53"], format="long"
    )

    assert set(summaries["cancer_code"]) == codes
    assert set(summaries["source_cohort"]) == {MBL_SOURCE}
    assert summaries.set_index("cancer_code")["n_samples"].to_dict() == (
        MBL_SUBGROUP_SAMPLES
    )

    assert codes <= set(available_representative_cohorts())
    assert codes <= set(available_percentile_cohorts())
    for code in MBL_SUBGROUP_SAMPLES:
        representatives = representative_cohort_samples(code)
        assert not representatives.empty
        assert {"Ensembl_Gene_ID", "Symbol"} <= set(representatives.columns)
        assert len(representatives.columns) > 2

        percentiles = cohort_gene_percentiles(code)
        assert not percentiles.empty
        assert percentiles["Ensembl_Gene_ID"].is_unique
        assert {"p0", "p50", "p100"} <= set(percentiles.columns)


def test_mbl_registry_provenance_states_marker_derived_artifacts():
    row = cohort_registry_df().set_index("cohort_id").loc[MBL_SOURCE]

    assert row["assay"] == "bulk RNA-seq"
    assert int(row["n_samples"]) == sum(MBL_SUBGROUP_SAMPLES.values())
    assert int(row["n_codes"]) == len(MBL_SUBGROUP_SAMPLES)
    assert "deterministic disjoint partition" in row["provenance"]
    assert "representative / percentile artifacts" in row["provenance"]
    assert "historical approximation" in row["provenance"]
