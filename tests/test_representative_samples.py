"""Representative per-sample expression vectors (#312).

The accessor logic is tested against SYNTHETIC parquet shards in a tmp dir
(monkeypatched root) — never real patient data. A light schema/scale check
runs against the bundled artifact (public Treehouse-derived medoids)."""

import numpy as np
import pandas as pd
import pytest

from pirlygenes.expression import accessors
from pirlygenes.expression import select_representative_samples


_GENES = pd.DataFrame({
    "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000146648", "ENSG00000136997"],
    "Symbol": ["TP53", "EGFR", "MYC"],
})


def _shard(reps):
    df = _GENES.copy()
    for name, vec in reps.items():
        df[name] = np.asarray(vec, dtype="float32")
    return df


@pytest.fixture
def synth_reps(tmp_path, monkeypatch):
    """A tmp representatives dir with two synthetic cohorts + provenance."""
    _shard({"PRAD_rep01": [100.0, 5.0, 50.0],
            "PRAD_rep02": [90.0, 8.0, 40.0],
            "PRAD_rep03": [110.0, 3.0, 55.0]}).to_parquet(tmp_path / "PRAD.parquet")
    _shard({"GBM_rep01": [10.0, 200.0, 30.0],
            "GBM_rep02": [12.0, 180.0, 25.0]}).to_parquet(tmp_path / "GBM.parquet")
    pd.DataFrame([
        {"cancer_code": "PRAD", "representative_id": f"PRAD_rep0{i}",
         "source_cohort": "TREEHOUSE_POLYA_25_01", "source_project": "Treehouse",
         "n_cohort_samples": 496, "cluster_rank": i} for i in (1, 2, 3)
    ]).to_csv(tmp_path / "_provenance.csv", index=False)
    monkeypatch.setattr(accessors, "_representatives_root", lambda: tmp_path)
    return tmp_path


def test_available_cohorts(synth_reps):
    assert accessors.available_representative_cohorts() == ["GBM", "PRAD"]


def test_wide_single_cohort(synth_reps):
    w = accessors.representative_cohort_samples("PRAD")
    assert list(w.columns) == ["Ensembl_Gene_ID", "Symbol",
                               "PRAD_rep01", "PRAD_rep02", "PRAD_rep03"]
    assert w.shape == (3, 5)


def test_k_caps_representatives(synth_reps):
    w = accessors.representative_cohort_samples("PRAD", k=2)
    reps = [c for c in w.columns if c.startswith("PRAD_rep")]
    assert reps == ["PRAD_rep01", "PRAD_rep02"]


def test_wide_all_cohorts_outer_join(synth_reps):
    w = accessors.representative_cohort_samples()  # None -> all
    reps = [c for c in w.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert set(reps) == {"GBM_rep01", "GBM_rep02",
                         "PRAD_rep01", "PRAD_rep02", "PRAD_rep03"}


def test_wide_representatives_canonicalize_before_outer_join(tmp_path, monkeypatch):
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": [
                "ENSG00000141510",
                "ENSG00000141510",
                "ENSG00000277113",
            ],
            "Symbol": ["old_tp53_alias", "TP53", "OR2T3"],
            "AAA_rep01": [1.0, 2.0, 4.0],
        }
    ).to_parquet(tmp_path / "AAA.parquet")
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000196539"],
            "Symbol": ["OR2T3"],
            "BBB_rep01": [5.0],
        }
    ).to_parquet(tmp_path / "BBB.parquet")
    monkeypatch.setattr(accessors, "_representatives_root", lambda: tmp_path)

    w = accessors.representative_cohort_samples()

    assert w["Ensembl_Gene_ID"].tolist() == [
        "ENSG00000141510",
        "ENSG00000196539",
    ]
    assert w["AAA_rep01"].iloc[0] == 3.0
    or2t3 = w["Ensembl_Gene_ID"] == "ENSG00000196539"
    assert w.loc[or2t3, "AAA_rep01"].iloc[0] == 4.0
    assert w.loc[or2t3, "BBB_rep01"].iloc[0] == 5.0


def test_wide_baked_dedupes_divergent_symbol(tmp_path, monkeypatch):
    """Same canonical ENSG with a divergent display Symbol across shards (a real
    symbol in one cohort, the ENSG string as fallback in the other) must collapse
    to ONE row, preferring the real symbol (#474 review, P2b). The baked path
    skips read-time canonicalization, so this is where the split surfaced."""
    pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000999991"],
        "Symbol": ["TP53", "REALSYM"],
        "CLL_rep01": [10.0, 20.0],
    }).to_parquet(tmp_path / "CLL.parquet")
    pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000999991"],
        "Symbol": ["TP53", "ENSG00000999991"],   # fallback to the ENSG string
        "PRAD_rep01": [11.0, 21.0],
    }).to_parquet(tmp_path / "PRAD.parquet")
    (tmp_path / "_manifest.json").write_text('{"canonical_gene_ids": true}\n')
    monkeypatch.setattr(accessors, "_representatives_root", lambda: tmp_path)

    w = accessors.representative_cohort_samples(["CLL", "PRAD"], format="wide")

    assert w["Ensembl_Gene_ID"].nunique() == len(w)          # no duplicate ENSG
    row = w[w["Ensembl_Gene_ID"] == "ENSG00000999991"]
    assert len(row) == 1
    assert row["Symbol"].iloc[0] == "REALSYM"                # real symbol wins
    assert row["CLL_rep01"].iloc[0] == 20.0                  # both cohorts aligned
    assert row["PRAD_rep01"].iloc[0] == 21.0
    assert list(w.columns[:2]) == ["Ensembl_Gene_ID", "Symbol"]


def test_real_representatives_have_unique_genes():
    """The bundled baked representatives must yield one row per gene across all
    cohorts — no ENSG split by divergent Symbol fallbacks (#474 review, P2b)."""
    codes = accessors.available_representative_cohorts()
    if len(codes) < 2:
        pytest.skip("needs the bundled multi-cohort representatives")
    w = accessors.representative_cohort_samples(codes, k=1, format="wide")
    assert w["Ensembl_Gene_ID"].nunique() == len(w)


def test_long_with_provenance(synth_reps):
    lng = accessors.representative_cohort_samples("PRAD", format="long",
                                                  include_provenance=True)
    assert {"cancer_code", "representative_id", "expression",
            "source_cohort", "n_cohort_samples"} <= set(lng.columns)
    assert lng["cancer_code"].unique().tolist() == ["PRAD"]
    assert lng["source_cohort"].iloc[0] == "TREEHOUSE_POLYA_25_01"


def test_log1p_normalize(synth_reps):
    raw = accessors.representative_cohort_samples("PRAD", k=1)
    lg = accessors.representative_cohort_samples("PRAD", k=1,
                                                 normalize="tpm_clean_log1p")
    np.testing.assert_allclose(lg["PRAD_rep01"].to_numpy(),
                               np.log1p(raw["PRAD_rep01"].to_numpy()), rtol=1e-5)


def test_unknown_normalize_or_format_raises(synth_reps):
    with pytest.raises(ValueError):
        accessors.representative_cohort_samples("PRAD", normalize="tpm")
    with pytest.raises(ValueError):
        accessors.representative_cohort_samples("PRAD", format="tall")


def test_missing_cohort_skipped(synth_reps):
    # A code with no shard is silently skipped (not every registry code has
    # per-sample data); an empty selection returns the empty schema.
    out = accessors.representative_cohort_samples("ACC")  # no synth shard
    assert list(out.columns) == ["Ensembl_Gene_ID", "Symbol"]
    assert out.empty


# --- select_representative_samples (the clustering/medoid helper) ---

def test_select_returns_all_when_n_le_k():
    m = pd.DataFrame(np.random.default_rng(0).random((20, 3)),
                     columns=["a", "b", "c"])
    assert select_representative_samples(m, k=5) == ["a", "b", "c"]


def test_select_picks_real_columns_and_is_deterministic():
    rng = np.random.default_rng(1)
    # three well-separated blobs of samples in gene space: 5 samples x 30 genes
    # positive centers (expression is non-negative; log1p needs >= 0)
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
    # one representative from each of the three separated blobs
    blobs = {i // 5 for i in (cols.index(c) for c in a)}
    assert blobs == {0, 1, 2}


def test_select_rejects_nonpositive_k():
    m = pd.DataFrame(np.ones((4, 4)))
    with pytest.raises(ValueError):
        select_representative_samples(m, k=0)


# --- light check against the real bundled artifact (public reference) ---

def test_bundled_artifact_schema_and_v4_scale():
    cohorts = accessors.available_representative_cohorts()
    if not cohorts:
        pytest.skip("representatives artifact not present in this checkout/cache")
    assert "PRAD" in cohorts
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
    """The NE axis is represented (#318) — trufflepig's sample-level battery no
    longer leaves it unscored. NET_PANCREAS + SCLC (the canonical NE poles) ship
    representatives."""
    cohorts = accessors.available_representative_cohorts()
    if not cohorts:
        pytest.skip("representatives artifact not present in this checkout/cache")
    ne = [c for c in cohorts if c.startswith(("NET_", "NEC_", "SCLC", "MTC"))]
    assert {"NET_PANCREAS", "SCLC"} <= set(ne), f"NE axis under-covered: {ne}"
    w = accessors.representative_cohort_samples("NET_PANCREAS")
    assert [c for c in w.columns if c.startswith("NET_PANCREAS_rep")]
