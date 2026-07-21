"""MBL subgroup summary/artifact capability boundary (#341, oncoref#420)."""

import pytest

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


def test_mbl_subgroups_have_summaries_but_no_per_sample_artifacts():
    """Do not confuse an aggregate RNA-seq source with redistributable matrices."""
    codes = set(MBL_SUBGROUP_SAMPLES)
    summaries = cancer_reference_expression(
        codes, genes=["TP53"], format="long"
    )

    assert set(summaries["cancer_code"]) == codes
    assert set(summaries["source_cohort"]) == {MBL_SOURCE}
    assert summaries.set_index("cancer_code")["n_samples"].to_dict() == (
        MBL_SUBGROUP_SAMPLES
    )

    assert codes.isdisjoint(available_representative_cohorts())
    assert codes.isdisjoint(available_percentile_cohorts())
    for code in MBL_SUBGROUP_SAMPLES:
        representatives = representative_cohort_samples(code)
        assert representatives.empty
        assert list(representatives.columns) == ["Ensembl_Gene_ID", "Symbol"]
        with pytest.raises(ValueError, match="no percentile vector available"):
            cohort_gene_percentiles(code)


def test_mbl_registry_provenance_states_artifact_limit():
    row = cohort_registry_df().set_index("cohort_id").loc[MBL_SOURCE]

    assert row["assay"] == "bulk RNA-seq"
    assert int(row["n_samples"]) == sum(MBL_SUBGROUP_SAMPLES.values())
    assert int(row["n_codes"]) == len(MBL_SUBGROUP_SAMPLES)
    assert "aggregate reference summaries only" in row["provenance"]
    assert "no released per-sample matrix" in row["provenance"]
