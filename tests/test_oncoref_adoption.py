"""Released OncoRef panel and materialized-data compatibility contracts."""

import json
import warnings

import datacache
import oncoref
from oncoref import cta
import pandas as pd
import pytest

from pirlygenes import gene_sets_cancer
from pirlygenes.expression import accessors
from pirlygenes.version import DATA_VERSION


def test_default_cta_accessors_adopt_active_oncoref_target_selection():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        active_ids = gene_sets_cancer.CTA_gene_ids()
        active_names = gene_sets_cancer.CTA_gene_names()
        unfiltered_names = gene_sets_cancer.CTA_unfiltered_gene_names()

    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert active_ids == set(oncoref.coverage.cta_gene_ids())
    assert gene_sets_cancer.CTA_historical_gene_ids() == set(cta.cta_gene_ids())
    assert "GPX5" in cta.cta_extended_gene_names()
    assert "GPX5" not in active_names
    # SPAG9 is classified Unsafe in OncoRef target_selection, outside active CTAs.
    assert "SPAG9" in unfiltered_names
    assert "SPAG9" not in active_names
    # Clinically cleared CTAG2 (LAGE-1) is admitted in active OncoRef 1.52.0 CTAs.
    assert "CTAG2" in active_names
    evidence = gene_sets_cancer.CTA_evidence()
    assert active_ids <= set(evidence.Ensembl_Gene_ID)
    assert evidence.attrs["policy_id"] == gene_sets_cancer.CTA_policy_id()


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

    # This fixture tests installed matrix reuse. Supply the independent owner
    # availability manifest explicitly so a cold CI run does not download the
    # entire expression bundle merely to refresh this synthetic provenance row.
    monkeypatch.setattr(
        accessors, "available_cancer_expression_references",
        lambda: pd.DataFrame(columns=[
            "cancer_code", "source_cohort", "processing_pipeline", "n_samples",
        ]),
    )
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


def _write_readonly_cached_matrix_bundle(cache, *, provenance_rows):
    from oncoref.data_bundle import DATA_VERSION as owner_data_version

    root = cache / "cohort-expression-matrices"
    root.mkdir(parents=True)
    gene = "ENSG00000141510"
    for name, gbm_val, cll_val in [("tpm", 12.0, 10.0), ("clean_tpm", 9.0, 8.0)]:
        pd.DataFrame(
            {
                "Ensembl_Gene_ID": [gene],
                "Symbol": ["TP53"],
                "GBM": [gbm_val],
                "CLL": [cll_val],
            }
        ).to_parquet(root / f"{name}.parquet", index=False)
    pd.DataFrame(provenance_rows).to_parquet(
        root / "provenance.parquet", index=False
    )
    metadata = {
        "artifact_type": accessors.COHORT_EXPRESSION_MATRICES_ARTIFACT_TYPE,
        "schema_version": accessors.COHORT_EXPRESSION_MATRICES_SCHEMA_VERSION,
        "canonical_gene_ids": True,
        "pirlygenes_data_version": DATA_VERSION,
        "built_from": {
            "package": "oncoref",
            "package_version": "1.19.2",
            "data_version": owner_data_version,
        },
        "tables": {
            name: {
                "file": filename,
                "rows": len(provenance_rows) if name == "provenance" else 1,
            }
            for name, filename in accessors._COHORT_MATRIX_FILES.items()
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    for path in root.iterdir():
        path.chmod(0o444)
    return root, metadata, gene


def _snapshot_tree(cache):
    return {
        str(path.relative_to(cache)): (
            path.stat().st_ino,
            path.stat().st_mode,
            path.stat().st_mtime_ns,
            path.stat().st_ctime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in cache.rglob("*")
    }


def test_cached_matrices_cold_offline_without_owner_bundle_or_download(
    tmp_path, monkeypatch,
):
    """Valid cached matrices must not download OncoRef's bulk archive (#654)."""
    from oncoref import data_bundle as oncoref_data_bundle

    from pirlygenes import data_bundle

    cache = tmp_path / "installed-pirlygenes"
    empty_owner_cache = tmp_path / "empty-oncoref"
    empty_owner_cache.mkdir()
    root, metadata, gene = _write_readonly_cached_matrix_bundle(
        cache,
        provenance_rows=[
            {
                "cancer_code": "GBM",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
                "processing_pipeline": "old-bundled-pipeline",
                "n_samples": 1,
            },
            {
                "cancer_code": "CLL",
                "source_cohort": "synthetic-custom-cohort",
                "processing_pipeline": "synthetic-pipeline",
                "n_samples": 2,
            },
        ],
    )

    def no_network_or_rebuild(*args, **kwargs):
        raise AssertionError("Cold local matrix read triggered download or rebuild")

    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(cache))
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(empty_owner_cache))
    monkeypatch.setenv("OPENVAX_OFFLINE", "1")
    monkeypatch.setattr(
        accessors,
        "_cohort_matrices_root",
        lambda: data_bundle.find("cohort-expression-matrices"),
    )
    monkeypatch.setattr(
        accessors, "_shared_rebuilt_canonical_cohort_matrices", no_network_or_rebuild
    )
    monkeypatch.setattr(data_bundle, "_install", no_network_or_rebuild)
    monkeypatch.setattr(oncoref_data_bundle, "ensure_local", no_network_or_rebuild)
    monkeypatch.setattr(oncoref_data_bundle, "fetch", no_network_or_rebuild)

    before = _snapshot_tree(cache)
    accessors.clear_reference_expression_cache()
    try:
        assert accessors.owner_percentile_artifacts_are_local() is False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            matrices = accessors.cohort_expression_matrices(
                ["GBM", "CLL"], genes=[gene]
            )
        assert not caught
        assert matrices.tpm["GBM"].tolist() == [12.0]
        assert matrices.clean_tpm["GBM"].tolist() == [9.0]
        assert matrices.tpm["CLL"].tolist() == [10.0]
        assert matrices.clean_tpm["CLL"].tolist() == [8.0]

        prov = matrices.provenance.set_index("source_cohort")
        assert "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES" in prov.index
        assert (
            prov.loc["TREEHOUSE_POLYA_25_01_TCGA_SAMPLES", "processing_pipeline"]
            != "old-bundled-pipeline"
        )
        assert int(prov.loc["TREEHOUSE_POLYA_25_01_TCGA_SAMPLES", "n_samples"]) > 1
        assert (
            prov.loc["synthetic-custom-cohort", "processing_pipeline"]
            == "synthetic-pipeline"
        )
        assert int(prov.loc["synthetic-custom-cohort", "n_samples"]) == 2
        assert accessors.cohort_expression_matrices_metadata() == metadata
        assert data_bundle.find("cohort-expression-matrices") == root
        assert _snapshot_tree(cache) == before
    finally:
        accessors.clear_reference_expression_cache()


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("oncoref: data bundle release download failed"),
        datacache.OfflineError("https://example.invalid/oncoref-data.tar.gz"),
        OSError("network unreachable"),
    ],
)
def test_cached_matrices_preserve_provenance_when_owner_manifest_unavailable(
    tmp_path, monkeypatch, exc,
):
    """Owner-metadata failure must not mark valid matrices corrupt (#654)."""
    from pirlygenes import data_bundle

    cache = tmp_path / "installed-pirlygenes"
    empty_owner_cache = tmp_path / "empty-oncoref"
    empty_owner_cache.mkdir()
    root, metadata, gene = _write_readonly_cached_matrix_bundle(
        cache,
        provenance_rows=[
            {
                "cancer_code": "GBM",
                "source_cohort": "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
                "processing_pipeline": "bundled-gbm-pipeline",
                "n_samples": 166,
            },
            {
                "cancer_code": "CLL",
                "source_cohort": "synthetic-offline-build",
                "processing_pipeline": "bundled-cll-pipeline",
                "n_samples": 2,
            },
        ],
    )

    def no_rebuild_or_download(*args, **kwargs):
        raise AssertionError("Unavailable owner metadata triggered rebuild or download")

    def fail_owner_manifest():
        raise exc

    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(cache))
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(empty_owner_cache))
    monkeypatch.setenv("OPENVAX_OFFLINE", "1")
    monkeypatch.setattr(
        accessors, "available_cancer_expression_references", fail_owner_manifest
    )
    monkeypatch.setattr(
        accessors,
        "_cohort_matrices_root",
        lambda: data_bundle.find("cohort-expression-matrices"),
    )
    monkeypatch.setattr(
        accessors, "_shared_rebuilt_canonical_cohort_matrices", no_rebuild_or_download
    )
    monkeypatch.setattr(data_bundle, "_install", no_rebuild_or_download)

    before = _snapshot_tree(cache)
    accessors.clear_reference_expression_cache()
    try:
        with pytest.warns(
            RuntimeWarning,
            match="Owner expression provenance metadata could not be loaded",
        ) as record:
            matrices = accessors.cohort_expression_matrices(
                ["GBM", "CLL"], genes=[gene]
            )
        messages = [str(w.message) for w in record]
        assert all("could not be read" not in msg for msg in messages)
        assert all("falling back to a full rebuild" not in msg for msg in messages)

        assert matrices.tpm["GBM"].tolist() == [12.0]
        assert matrices.clean_tpm["GBM"].tolist() == [9.0]
        assert matrices.tpm["CLL"].tolist() == [10.0]
        assert matrices.clean_tpm["CLL"].tolist() == [8.0]

        prov = matrices.provenance.set_index("source_cohort")
        assert (
            prov.loc["TREEHOUSE_POLYA_25_01_TCGA_SAMPLES", "processing_pipeline"]
            == "bundled-gbm-pipeline"
        )
        assert int(prov.loc["TREEHOUSE_POLYA_25_01_TCGA_SAMPLES", "n_samples"]) == 166
        assert (
            prov.loc["synthetic-offline-build", "processing_pipeline"]
            == "bundled-cll-pipeline"
        )
        assert int(prov.loc["synthetic-offline-build", "n_samples"]) == 2
        assert accessors.cohort_expression_matrices_metadata() == metadata
        assert data_bundle.find("cohort-expression-matrices") == root
        assert _snapshot_tree(cache) == before
    finally:
        accessors.clear_reference_expression_cache()
