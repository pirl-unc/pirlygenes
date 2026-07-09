"""Representative per-sample expression vectors (#312, delegated to oncoref #208).

The accessor now delegates selection + storage to oncoref (deterministic
farthest-first medoids); these tests pin the pirlygenes-facing contract against
the real artifact (skip when absent), plus the build-side
``select_representative_samples`` engine (still shipped + used by the builder).
Never real patient data — the artifact is the public Treehouse-derived medoids.
"""

import numpy as np
import pandas as pd
import pytest

from pirlygenes.expression import accessors
from pirlygenes.expression import select_representative_samples


def _skip_if_absent():
    if not accessors.available_representative_cohorts():
        pytest.skip("representatives artifact not present in this checkout/cache")


# --- delegated accessor contract (against the real artifact) ---

def test_available_cohorts_sorted_and_nonempty():
    _skip_if_absent()
    codes = accessors.available_representative_cohorts()
    assert codes == sorted(codes)
    assert "PRAD" in codes and "GBM" in codes


def test_wide_single_cohort_contract():
    _skip_if_absent()
    w = accessors.representative_cohort_samples("PRAD")
    assert list(w.columns[:2]) == ["Ensembl_Gene_ID", "Symbol"]
    reps = [c for c in w.columns if c.startswith("PRAD_rep")]
    assert 1 <= len(reps) <= 5
    assert reps == sorted(reps)                       # rep01, rep02, … in order
    assert w["Ensembl_Gene_ID"].nunique() == len(w)   # one row per canonical ENSG


def test_k_caps_representatives():
    _skip_if_absent()
    w = accessors.representative_cohort_samples("PRAD", k=2)
    reps = [c for c in w.columns if c.startswith("PRAD_rep")]
    assert reps == ["PRAD_rep01", "PRAD_rep02"]


def test_long_with_provenance_projects_to_pirlygenes_columns():
    _skip_if_absent()
    lng = accessors.representative_cohort_samples(
        "PRAD", format="long", include_provenance=True)
    # oncoref's provenance is a superset; the accessor projects to exactly these.
    assert list(lng.columns) == [
        "Ensembl_Gene_ID", "Symbol", "cancer_code", "representative_id",
        "expression", "source_cohort", "source_project", "n_cohort_samples"]
    assert lng["cancer_code"].unique().tolist() == ["PRAD"]


def test_log1p_normalize_matches_expm1():
    _skip_if_absent()
    raw = accessors.representative_cohort_samples("PRAD", k=1)
    lg = accessors.representative_cohort_samples(
        "PRAD", k=1, normalize="tpm_clean_log1p")
    np.testing.assert_allclose(
        lg["PRAD_rep01"].to_numpy(),
        np.log1p(raw["PRAD_rep01"].to_numpy()), rtol=1e-4, atol=1e-3)


def test_long_provenance_projection_glue(monkeypatch):
    """Projection glue is exercised even without the artifact: oncoref's
    provenance is a superset, and the accessor must trim to exactly pirlygenes'
    8 documented columns (in order), forwarding the pirlygenes-specific kwargs.
    """
    import oncoref

    superset = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000000001", "ENSG00000000002"],
        "Symbol": ["AAA", "BBB"],
        "cancer_code": ["PRAD", "PRAD"],
        "representative_id": ["PRAD_rep01", "PRAD_rep01"],
        "expression": [1.0, 2.0],
        "source_cohort": ["treehouse", "treehouse"],
        "source_project": ["TCGA-PRAD", "TCGA-PRAD"],
        "n_cohort_samples": [42, 42],
        "oncoref_only_extra": ["x", "y"],   # oncoref superset column — must be dropped
        "medoid_rank": [0, 0],              # oncoref superset column — must be dropped
    })

    def fake_delegate(*_a, **kw):
        assert kw.get("representative_id_style") == "pirlygenes"
        assert kw.get("sample_qc") == "artifact"
        assert kw.get("format") == "long" and kw.get("include_provenance") is True
        return superset

    monkeypatch.setattr(oncoref, "representative_cohort_samples", fake_delegate)
    out = accessors.representative_cohort_samples(
        "PRAD", format="long", include_provenance=True)
    assert list(out.columns) == [
        "Ensembl_Gene_ID", "Symbol", "cancer_code", "representative_id",
        "expression", "source_cohort", "source_project", "n_cohort_samples"]
    assert out["cancer_code"].tolist() == ["PRAD", "PRAD"]


def test_unknown_normalize_or_format_raises():
    # Validation happens before the oncoref call, so no artifact needed.
    with pytest.raises(ValueError):
        accessors.representative_cohort_samples("PRAD", normalize="tpm")
    with pytest.raises(ValueError):
        accessors.representative_cohort_samples("PRAD", format="tall")


def test_missing_cohort_returns_empty_schema():
    _skip_if_absent()
    # A registered code with no representatives (STAD_MSI is registered but its
    # per-sample cohort is not built) returns the empty schema, not an error.
    out = accessors.representative_cohort_samples("STAD_MSI")
    assert list(out.columns) == ["Ensembl_Gene_ID", "Symbol"]
    assert out.empty


def test_selection_is_deterministic():
    """oncoref's medoid selection is seed-free, so repeated reads are identical
    (pirlygenes' former k-means++ was seed-dependent)."""
    _skip_if_absent()
    a = accessors.representative_cohort_samples("PRAD")
    b = accessors.representative_cohort_samples("PRAD")
    pd.testing.assert_frame_equal(a, b)


def test_real_representatives_have_unique_genes():
    """One row per gene across all cohorts — no ENSG split by symbol drift."""
    _skip_if_absent()
    codes = accessors.available_representative_cohorts()
    w = accessors.representative_cohort_samples(codes, k=1, format="wide")
    assert w["Ensembl_Gene_ID"].nunique() == len(w)


def test_bundled_artifact_schema_and_v4_scale():
    _skip_if_absent()
    w = accessors.representative_cohort_samples("PRAD")
    assert {"Ensembl_Gene_ID", "Symbol"} <= set(w.columns)
    reps = [c for c in w.columns if c.startswith("PRAD_rep")]
    assert 1 <= len(reps) <= 5
    # clean_tpm_16_9_75: biological compartment lands on the 750k budget
    from pirlygenes.expression.normalize import clean_tpm_removal_mask
    mask = clean_tpm_removal_mask(w[["Symbol", "Ensembl_Gene_ID"]]).to_numpy()
    bio = w[reps[0]].to_numpy()[~mask].sum()
    assert abs(bio - 750_000.0) < 5_000.0


def test_neuroendocrine_axis_has_representatives():
    """The NE axis is represented (#318): NET_PANCREAS + SCLC ship representatives."""
    _skip_if_absent()
    cohorts = accessors.available_representative_cohorts()
    ne = [c for c in cohorts if c.startswith(("NET_", "NEC_", "SCLC", "MTC"))]
    assert {"NET_PANCREAS", "SCLC"} <= set(ne), f"NE axis under-covered: {ne}"
    w = accessors.representative_cohort_samples("NET_PANCREAS")
    assert [c for c in w.columns if c.startswith("NET_PANCREAS_rep")]


# --- select_representative_samples (the build-side clustering/medoid engine) ---

def test_select_returns_all_when_n_le_k():
    m = pd.DataFrame(np.random.default_rng(0).random((20, 3)),
                     columns=["a", "b", "c"])
    assert select_representative_samples(m, k=5) == ["a", "b", "c"]


def test_select_picks_real_columns_and_is_deterministic():
    rng = np.random.default_rng(1)
    # three well-separated blobs of samples in gene space: 5 samples x 30 genes
    blocks = [rng.normal(center, 1.0, size=(5, 30)) for center in (10.0, 100.0, 1000.0)]
    samples_by_genes = np.vstack(blocks)         # 15 samples x 30 genes
    X = samples_by_genes.T                        # genes(30) x samples(15)
    cols = [f"s{i:02d}" for i in range(15)]
    m = pd.DataFrame(X, columns=cols)
    a = select_representative_samples(m, k=3)
    b = select_representative_samples(m, k=3)
    assert a == b                      # deterministic
    assert set(a) <= set(cols)         # real columns
    assert len(a) == 3
    blobs = {i // 5 for i in (cols.index(c) for c in a)}
    assert blobs == {0, 1, 2}          # one representative per separated blob


def test_select_rejects_nonpositive_k():
    m = pd.DataFrame(np.ones((4, 4)))
    with pytest.raises(ValueError):
        select_representative_samples(m, k=0)
