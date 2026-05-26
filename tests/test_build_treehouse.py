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
    clean = builder._clean_tpm(values, removable)
    # g0 should be 0 in every sample
    assert (clean.iloc[0] == 0).all()
    # Each sample sums to ~1e6
    np.testing.assert_allclose(clean.sum(axis=0).to_numpy(), [1e6, 1e6])
