"""Released OncoRef panel and materialized-data compatibility contracts."""

import json

import oncoref
from oncoref import cta

from pirlygenes import gene_sets_cancer
from pirlygenes.expression import accessors
from pirlygenes.version import DATA_VERSION


def test_default_cta_reexports_do_not_promote_extended_only_targets():
    assert gene_sets_cancer.CTA_gene_ids() == cta.cta_gene_ids()
    assert gene_sets_cancer.CTA_unfiltered_gene_ids() == cta.cta_unfiltered_gene_ids()
    assert "GPX5" in cta.cta_extended_gene_names()
    assert "GPX5" not in gene_sets_cancer.CTA_gene_names()
    evidence = gene_sets_cancer.CTA_evidence()
    assert cta.cta_gene_ids() <= set(evidence.Ensembl_Gene_ID)
    assert oncoref.CTA_gene_ids is gene_sets_cancer.CTA_gene_ids


def test_materialized_reference_preserves_build_version_and_rejects_data_drift(tmp_path):
    from oncoref.data_bundle import DATA_VERSION as owner_data_version

    # The wheel-only owner upgrade retains the expression-data identity. A
    # fixture built by the preceding owner remains usable without relabeling
    # the builder; changing its data identity must still fail closed.
    for filename in accessors._COHORT_MATRIX_FILES.values():
        (tmp_path / filename).touch()
    metadata = {
        "artifact_type": accessors.COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
        "schema_version": accessors.COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
        "canonical_gene_ids": True,
        "pirlygenes_data_version": DATA_VERSION,
        "built_from": {
            "package": "oncoref", "package_version": "1.8.206",
            "data_version": owner_data_version,
        },
        "tables": {name: {"file": filename, "rows": 1}
                   for name, filename in accessors._COHORT_MATRIX_FILES.items()},
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    assert accessors._cohort_matrices_usable(tmp_path)
    assert json.loads(path.read_text())["built_from"]["package_version"] == "1.8.206"

    metadata["built_from"]["data_version"] = "stale-data"
    path.write_text(json.dumps(metadata))
    assert not accessors._cohort_matrices_usable(tmp_path)
