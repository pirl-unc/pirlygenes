"""Fixture-driven test for scripts/build_treehouse_reference_expression.py.

The real builder consumes the Treehouse 25.01 PolyA compendium (6 GB
TSV from xena.treehouse.gi.ucsc.edu); this test exercises the same
internal helpers against a 6-gene × 5-sample mock matrix so the
inverse-log2 / symbol-harmonization / aggregation / stat-write path
is checked without any network or large-data dependency.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_builder():
    """Import the script as a module without going through scripts/ as a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_treehouse_reference_expression.py"
    spec = importlib.util.spec_from_file_location("build_treehouse", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_treehouse"] = module
    spec.loader.exec_module(module)
    return module


def _write_clinical(path: Path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_tpm_matrix(path: Path, symbols: list[str], samples: list[str], values: np.ndarray):
    df = pd.DataFrame(values, index=symbols, columns=samples)
    df.index.name = "Gene"
    df.reset_index().to_csv(path, sep="\t", index=False)


def test_read_clinical_disease_samples_filters_case_insensitive(tmp_path: Path):
    builder = _load_builder()
    clinical = tmp_path / "clinical.tsv"
    _write_clinical(
        clinical,
        [
            {"th_dataset_id": "S1", "disease": "Ewing sarcoma"},
            {"th_dataset_id": "S2", "disease": "osteosarcoma"},
            {"th_dataset_id": "S3", "disease": "ewing sarcoma"},
            {"th_dataset_id": "S4", "disease": "neuroblastoma"},
        ],
    )
    samples = builder._read_clinical_disease_samples(clinical, "Ewing sarcoma")
    assert samples == ["S1", "S3"]


def test_read_clinical_raises_when_no_match(tmp_path: Path):
    builder = _load_builder()
    clinical = tmp_path / "clinical.tsv"
    _write_clinical(
        clinical,
        [{"th_dataset_id": "S1", "disease": "glioma"}],
    )
    try:
        builder._read_clinical_disease_samples(clinical, "Ewing sarcoma")
    except RuntimeError as exc:
        assert "No samples matched" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_read_tpm_for_samples_subsets_columns(tmp_path: Path):
    builder = _load_builder()
    tpm_path = tmp_path / "tpm.tsv"
    _write_tpm_matrix(
        tpm_path,
        symbols=["A1BG", "ACTB", "GAPDH"],
        samples=["S1", "S2", "S3", "S4"],
        values=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 1.5, 2.5, 3.5],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=float,
        ),
    )
    df = builder._read_tpm_for_samples(tpm_path, ["S1", "S3"])
    assert df.index.tolist() == ["A1BG", "ACTB", "GAPDH"]
    assert df.columns.tolist() == ["S1", "S3"]
    np.testing.assert_array_equal(
        df.to_numpy(), np.array([[1.0, 3.0], [0.5, 2.5], [0.1, 0.3]])
    )


def test_read_tpm_raises_on_missing_sample(tmp_path: Path):
    builder = _load_builder()
    tpm_path = tmp_path / "tpm.tsv"
    _write_tpm_matrix(
        tpm_path,
        symbols=["A1BG"],
        samples=["S1", "S2"],
        values=np.array([[1.0, 2.0]]),
    )
    try:
        builder._read_tpm_for_samples(tpm_path, ["S1", "SX"])
    except RuntimeError as exc:
        assert "SX" in str(exc) or "1/2" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_inverse_log2_recovers_tpm():
    builder = _load_builder()
    # log2(TPM+1): TPM=0 -> 0; TPM=1 -> 1; TPM=3 -> 2; TPM=7 -> 3.
    log2_vals = pd.DataFrame([[0.0, 1.0, 2.0, 3.0]], index=["g1"], columns=list("ABCD"))
    tpm = builder._inverse_log2(log2_vals)
    np.testing.assert_allclose(tpm.iloc[0].to_numpy(), [0.0, 1.0, 3.0, 7.0])


def test_inverse_log2_clamps_tiny_negatives():
    builder = _load_builder()
    # log2(TPM+1) = -1e-20 corresponds to TPM ≈ -tiny float residual; clamp to 0.
    log2_vals = pd.DataFrame([[-1e-20]], index=["g1"], columns=["S1"])
    tpm = builder._inverse_log2(log2_vals)
    assert tpm.iloc[0, 0] >= 0.0


def test_clean_tpm_zeroes_removable_and_renormalizes():
    builder = _load_builder()
    # 2 samples, 4 genes. Gene 0 (mtRNA-like) is removable.
    values = pd.DataFrame(
        [
            [500_000.0, 800_000.0],   # removable, dominates raw mass
            [100_000.0, 50_000.0],
            [200_000.0, 100_000.0],
            [200_000.0, 50_000.0],
        ],
        index=["g0", "g1", "g2", "g3"],
        columns=["S1", "S2"],
    )
    removable = pd.Series([True, False, False, False], index=values.index)
    # legacy zero fill: removable g0 dropped to 0, remainder renormalized
    clean = builder._clean_tpm(values, removable, censored_fill="zero")
    assert (clean.iloc[0] == 0).all()
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
    # default "typical" fill (now used by the builder): g0 holds a constant
    # budget (not zero), columns still sum to 1e6, and kept genes are less
    # inflated than under zero fill (g0 dominates raw mass here).
    typ = builder._clean_tpm(values, removable)
    assert (typ.iloc[0] > 0).all()
    np.testing.assert_allclose(typ.sum(axis=0).to_numpy(), [1e6, 1e6])
    assert (typ.iloc[1] < clean.iloc[1]).all()


def test_symbol_mapping_rescues_renamed_symbol(monkeypatch, tmp_path):
    """The Treehouse symbol mapper now routes through the shared resolver,
    so a symbol the compendium spells with its *old* HGNC name (e.g.
    HIST1H1T, which pyensembl 112 doesn't know) is rescued via the synonym
    pool instead of being silently dropped."""
    from pirlygenes.builders import gene_mapping, treehouse
    from pirlygenes.builders.ncbi_gene_info import (
        GENE_INFO_SYNONYM_CONFIDENCE,
        SymbolAliasCandidate,
        SymbolAliasIndex,
    )

    class FakeGene:
        def __init__(self, gene_id, gene_name):
            self.gene_id = gene_id
            self.gene_name = gene_name

    class FakeGenome:
        def genes_by_name(self, symbol):
            table = {
                "H1-6": [FakeGene("ENSG00000187475.1", "H1-6")],
                "TP53": [FakeGene("ENSG00000141510.1", "TP53")],
            }
            return table.get(symbol, [])

    monkeypatch.setattr(treehouse, "EnsemblRelease", lambda release: FakeGenome())
    monkeypatch.setattr(
        gene_mapping, "cached_symbol_alias_index",
        lambda: SymbolAliasIndex(
            official_symbols=frozenset({"H1-6", "TP53"}),
            alias_candidates={
                "HIST1H1T": (
                    SymbolAliasCandidate("H1-6", "ncbi", GENE_INFO_SYNONYM_CONFIDENCE),
                ),
            },
        ),
    )
    gene_mapping.cached_combined_alias_index.cache_clear()
    try:
        mapping = treehouse._build_or_load_symbol_mapping(
            pd.Index(["HIST1H1T", "TP53"]),
            ensembl_release=112,
            cache_path=tmp_path / "m.parquet",
            refresh=True,
        )
    finally:
        gene_mapping.cached_combined_alias_index.cache_clear()

    by_src = dict(zip(mapping["source_symbol"], mapping["Ensembl_Gene_ID"]))
    # The old symbol was kept (source_symbol) but mapped to H1-6's ENSG.
    assert by_src["HIST1H1T"] == "ENSG00000187475"
    assert by_src["TP53"] == "ENSG00000141510"
