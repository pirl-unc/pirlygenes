"""Released OncoRef panel and materialized-data compatibility contracts."""

import json

import oncoref
from oncoref import cta
import pandas as pd

from pirlygenes import gene_sets_cancer
from pirlygenes.expression import accessors
from pirlygenes.version import DATA_VERSION


def test_default_cta_reexports_do_not_promote_extended_only_targets():
    assert gene_sets_cancer.CTA_gene_ids() == cta.cta_gene_ids()
    assert gene_sets_cancer.CTA_unfiltered_gene_ids() == cta.cta_unfiltered_gene_ids()
    assert "GPX5" in cta.cta_extended_gene_names()
    assert "GPX5" not in gene_sets_cancer.CTA_gene_names()
    # The newly nominated SPAG9 record remains outside the default panel.
    assert "SPAG9" in gene_sets_cancer.CTA_unfiltered_gene_names()
    assert "SPAG9" not in gene_sets_cancer.CTA_gene_names()
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


def test_cached_matrices_survive_owner_upgrade_without_rebuild_or_rewrite(
    tmp_path, monkeypatch,
):
    from oncoref.data_bundle import DATA_VERSION as owner_data_version
    from pirlygenes import data_bundle

    cache = tmp_path / "installed"
    root = cache / "cohort-expression-matrices"
    root.mkdir(parents=True)
    gene = "ENSG00000141510"
    for name, value in [("tpm", 10.0), ("clean_tpm", 8.0)]:
        pd.DataFrame({"Ensembl_Gene_ID": [gene], "Symbol": ["TP53"],
                      "CLL": [value]}).to_parquet(root / f"{name}.parquet", index=False)
    pd.DataFrame([{"cancer_code": "CLL", "source_cohort": "synthetic-old-build",
                   "processing_pipeline": "synthetic", "n_samples": 2}]).to_parquet(
        root / "provenance.parquet", index=False)
    metadata = {
        "artifact_type": accessors.COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
        "schema_version": accessors.COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
        "canonical_gene_ids": True,
        "pirlygenes_data_version": DATA_VERSION,
        "built_from": {"package": "oncoref", "package_version": "1.8.210",
                       "data_version": owner_data_version},
        "tables": {name: {"file": filename, "rows": 1}
                   for name, filename in accessors._COHORT_MATRIX_FILES.items()},
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    for path in root.iterdir():
        path.chmod(0o444)

    def snapshot():
        return {str(path.relative_to(cache)): (
            path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns,
            path.stat().st_ctime_ns, path.read_bytes() if path.is_file() else None,
        ) for path in cache.rglob("*")}

    def no_rebuild(*args, **kwargs):
        raise AssertionError("An owner package upgrade rebuilt installed matrices")

    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(cache))
    monkeypatch.setattr(accessors, "_cohort_matrices_root",
                        lambda: data_bundle.find("cohort-expression-matrices"))
    monkeypatch.setattr(accessors, "_shared_rebuilt_canonical_cohort_matrices", no_rebuild)
    monkeypatch.setattr(data_bundle, "_install", no_rebuild)
    before = snapshot()
    accessors._load_precomputed_cohort_matrices.cache_clear()
    try:
        matrices = accessors.cohort_expression_matrices("CLL", genes=[gene])
        assert matrices.tpm["CLL"].tolist() == [10.0]
        assert matrices.clean_tpm["CLL"].tolist() == [8.0]
        assert matrices.provenance.source_cohort.tolist() == ["synthetic-old-build"]
        assert accessors.cohort_expression_matrices_metadata() == metadata
        assert data_bundle.find("cohort-expression-matrices") == root
        assert snapshot() == before
    finally:
        accessors._load_precomputed_cohort_matrices.cache_clear()
