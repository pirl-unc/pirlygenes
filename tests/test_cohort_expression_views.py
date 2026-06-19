"""Unified cohort normalization-views object (#319): one object bundling the
tpm / clean_tpm / clean_tpm_biological stages so a consumer can't re-normalize
inconsistently."""

import pytest

from pirlygenes.expression import (
    CohortExpressionViews,
    cohort_expression_views,
)
from pirlygenes.expression import accessors


def _disable_precomputed_views(monkeypatch, tmp_path):
    accessors._load_precomputed_cohort_views.cache_clear()
    from pathlib import Path
    root = Path(tmp_path)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root / "missing")


def test_views_bundle_three_stages_and_provenance():
    v = cohort_expression_views("CLL", genes=["MS4A1", "MALAT1", "RPL13A"])
    assert isinstance(v, CohortExpressionViews)
    for frame in (v.tpm, v.clean_tpm, v.clean_tpm_biological):
        assert {"Ensembl_Gene_ID", "Symbol"} <= set(frame.columns)
    # biological view drops the censored genes (MALAT1 technical, RPL13A ribo),
    # keeps real biology (MS4A1)
    bio = set(v.clean_tpm_biological["Symbol"])
    assert "MS4A1" in bio
    assert "MALAT1" not in bio and "RPL13A" not in bio
    # tpm/clean_tpm keep all requested genes (technical included)
    assert {"MS4A1", "MALAT1", "RPL13A"} <= set(v.clean_tpm["Symbol"])
    # provenance records the cohort + pipeline (native unit)
    assert "source_cohort" in v.provenance.columns
    assert "processing_pipeline" in v.provenance.columns
    assert len(v.provenance) >= 1


def test_views_clean_differs_from_tpm_for_technical_gene():
    """clean_tpm_16_9_75 changes the technical gene's value vs plain TPM (the whole
    point of having both stages in one object)."""
    v = cohort_expression_views("CLL", genes=["MS4A1", "MALAT1"])
    tpm = dict(zip(v.tpm["Symbol"], v.tpm["CLL"]))
    clean = dict(zip(v.clean_tpm["Symbol"], v.clean_tpm["CLL"]))
    # MALAT1 (polyA-bias technical) is suppressed under clean_tpm_16_9_75
    assert clean["MALAT1"] != tpm["MALAT1"]


def test_aggregate_code_expands_in_views():
    """An aggregate code (SARC) expands to its subtype cohorts in the views."""
    v = cohort_expression_views("SARC", genes=["TP53"])
    cohort_cols = [c for c in v.tpm.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert any(c.startswith("SARC_") for c in cohort_cols)


def test_views_canonicalize_before_pivoting_symbol_drift(monkeypatch, tmp_path):
    import pandas as pd

    fake = pd.DataFrame(
        [
            {
                "Ensembl_Gene_ID": "ENSG00000141510",
                "Symbol": "old_tp53_alias",
                "cancer_code": "AAA",
                "source_cohort": "S1",
                "source_project": "fixture",
                "source_version": "fixture-v1",
                "TPM_median": 1.0,
                "TPM_q1": 1.0,
                "TPM_q3": 1.0,
                "TPM_clean_median": 10.0,
                "TPM_clean_q1": 10.0,
                "TPM_clean_q3": 10.0,
                "n_samples": 1,
                "n_detected": 1,
                "processing_pipeline": "fixture",
                "notes": "",
            },
            {
                "Ensembl_Gene_ID": "ENSG00000141510",
                "Symbol": "TP53",
                "cancer_code": "AAA",
                "source_cohort": "S1",
                "source_project": "fixture",
                "source_version": "fixture-v2",
                "TPM_median": 2.0,
                "TPM_q1": 2.0,
                "TPM_q3": 2.0,
                "TPM_clean_median": 20.0,
                "TPM_clean_q1": 20.0,
                "TPM_clean_q3": 20.0,
                "n_samples": 1,
                "n_detected": 1,
                "processing_pipeline": "fixture",
                "notes": "",
            },
        ]
    )
    accessors._REFERENCE_VIEW_CACHE.clear()
    monkeypatch.setattr(accessors, "_load_cancer_reference_expression", lambda: fake)
    _disable_precomputed_views(monkeypatch, tmp_path)

    v = cohort_expression_views()

    assert v.tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]
    assert v.tpm["AAA"].iloc[0] == 3.0
    assert v.clean_tpm["AAA"].iloc[0] == 30.0


def _fixture_row(ensg, code, version, tpm):
    return {
        "Ensembl_Gene_ID": ensg, "Symbol": ensg, "cancer_code": code,
        "source_cohort": "S1", "source_project": "fixture",
        "source_version": version, "TPM_median": tpm, "TPM_q1": tpm,
        "TPM_q3": tpm, "TPM_clean_median": tpm, "TPM_clean_q1": tpm,
        "TPM_clean_q3": tpm, "n_samples": 1, "n_detected": 1,
        "processing_pipeline": "fixture", "notes": "",
    }


def test_views_protein_coding_and_coverage_filters(monkeypatch, tmp_path):
    import pandas as pd

    # TP53 (protein_coding) in both cohorts; MALAT1 (lncRNA) in one cohort only.
    fake = pd.DataFrame([
        _fixture_row("ENSG00000141510", "AAA", "v1", 1.0),
        _fixture_row("ENSG00000141510", "BBB", "v1", 1.0),
        _fixture_row("ENSG00000251562", "AAA", "v1", 5.0),
    ])
    monkeypatch.setattr(
        accessors, "_load_cancer_reference_expression", lambda: fake
    )

    accessors._REFERENCE_VIEW_CACHE.clear()
    _disable_precomputed_views(monkeypatch, tmp_path)
    pc = cohort_expression_views(protein_coding=True)
    assert pc.clean_tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]

    accessors._REFERENCE_VIEW_CACHE.clear()
    _disable_precomputed_views(monkeypatch, tmp_path)
    cov = cohort_expression_views(min_cohort_coverage=1.0)
    # only TP53 is measured in every cohort
    assert cov.clean_tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]


def test_views_reject_invalid_min_cohort_coverage():
    import pytest

    with pytest.raises(ValueError, match="min_cohort_coverage"):
        cohort_expression_views("CLL", min_cohort_coverage=-0.1)
    with pytest.raises(ValueError, match="min_cohort_coverage"):
        cohort_expression_views("CLL", min_cohort_coverage=1.1)


def test_views_precomputed_artifact_fast_path(tmp_path, monkeypatch):
    import pandas as pd

    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000251562"],
            "Symbol": ["TP53", "MALAT1"],
            "CLL": [10.0, 20.0],
            "PRAD": [30.0, None],
        }
    ).to_parquet(tmp_path / "tpm.parquet", index=False)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000251562"],
            "Symbol": ["TP53", "MALAT1"],
            "CLL": [1.0, 2.0],
            "PRAD": [3.0, None],
        }
    ).to_parquet(tmp_path / "clean_tpm.parquet", index=False)
    pd.DataFrame(
        [
            {
                "cancer_code": "CLL",
                "source_cohort": "S1",
                "processing_pipeline": "fixture",
                "n_samples": 2,
            },
            {
                "cancer_code": "PRAD",
                "source_cohort": "S2",
                "processing_pipeline": "fixture",
                "n_samples": 3,
            },
        ]
    ).to_parquet(tmp_path / "provenance.parquet", index=False)
    (tmp_path / "_manifest.json").write_text('{"canonical_gene_ids": true}\n')
    accessors._load_precomputed_cohort_views.cache_clear()
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: tmp_path)
    # When the artifact is usable, NEITHER fallback may run: not the canonical
    # rebuild, not the from-reference builder.
    for name in (
        "_rebuild_full_canonical_views",
        "_cohort_expression_views_from_reference",
    ):
        monkeypatch.setattr(
            accessors, name,
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("slow path used")),
        )

    v = cohort_expression_views("CLL", genes=["TP53"])

    assert v.tpm.columns.tolist() == ["Ensembl_Gene_ID", "Symbol", "CLL"]
    assert v.tpm["Ensembl_Gene_ID"].tolist() == ["ENSG00000141510"]
    assert v.clean_tpm["CLL"].tolist() == [1.0]
    assert v.provenance["source_cohort"].tolist() == ["S1"]


def test_views_precomputed_gene_filter_drops_empty_cohorts(tmp_path, monkeypatch):
    import pandas as pd

    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "Symbol": ["TP53"],
            "CLL": [10.0],
            "PRAD": [None],
        }
    ).to_parquet(tmp_path / "tpm.parquet", index=False)
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510"],
            "Symbol": ["TP53"],
            "CLL": [1.0],
            "PRAD": [None],
        }
    ).to_parquet(tmp_path / "clean_tpm.parquet", index=False)
    pd.DataFrame(
        [
            {"cancer_code": "CLL", "source_cohort": "S1",
             "processing_pipeline": "fixture", "n_samples": 2},
            {"cancer_code": "PRAD", "source_cohort": "S2",
             "processing_pipeline": "fixture", "n_samples": 3},
        ]
    ).to_parquet(tmp_path / "provenance.parquet", index=False)
    (tmp_path / "_manifest.json").write_text('{"canonical_gene_ids": true}\n')
    accessors._load_precomputed_cohort_views.cache_clear()
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: tmp_path)

    v = cohort_expression_views(genes=["TP53"])

    assert v.tpm.columns.tolist() == ["Ensembl_Gene_ID", "Symbol", "CLL"]
    assert v.provenance["source_cohort"].tolist() == ["S1"]


# ---------------------------------------------------------------------------
# Unified-path robustness + equivalence
#
# The canonical path slices a request out of the *full* canonical matrices.
# That full matrix comes from the precomputed artifact when present, and is
# otherwise rebuilt from the reference; both feed the *same* filter, so the two
# must return identical results. The tests below pin that equivalence, exercise
# every fallback/corruption branch, and lock the gene-filter edge cases.
# ---------------------------------------------------------------------------

# Real ENSG IDs so canonicalization / biotype lookups resolve against the
# offline authority. TP53/ACTB are protein_coding; MALAT1 is a lncRNA.
TP53 = "ENSG00000141510"
ACTB = "ENSG00000075624"
MALAT1 = "ENSG00000251562"


def _ref_row(ensg, symbol, code, version, tpm, clean):
    return {
        "Ensembl_Gene_ID": ensg, "Symbol": symbol, "cancer_code": code,
        "source_cohort": f"{code}_SRC", "source_project": "fixture",
        "source_version": version, "TPM_median": tpm, "TPM_q1": tpm,
        "TPM_q3": tpm, "TPM_clean_median": clean, "TPM_clean_q1": clean,
        "TPM_clean_q3": clean, "n_samples": 4, "n_detected": 4,
        "processing_pipeline": "fixture", "notes": "",
    }


# Use real registry codes so explicit cancer_types resolve; the monkeypatched
# reference makes these the only data, independent of the bundled cohorts.
COHORT_A = "CLL"
COHORT_B = "PRAD"


def _synthetic_reference():
    """A reference frame with multiple cohorts, cross-version symbol drift on
    one ENSG, a coding + non-coding gene, and a gene present in only one
    cohort — enough surface for canonicalization, coverage, and biotype paths."""
    import pandas as pd

    return pd.DataFrame([
        # TP53: same ENSG, two symbols across versions in COHORT_A (symbol
        # drift), also present in COHORT_B. Canonicalization collapses the drift.
        _ref_row(TP53, "old_tp53_alias", COHORT_A, "v1", 1.0, 11.0),
        _ref_row(TP53, "TP53", COHORT_A, "v2", 2.0, 12.0),
        _ref_row(TP53, "TP53", COHORT_B, "v2", 4.0, 14.0),
        # ACTB: coding, both cohorts.
        _ref_row(ACTB, "ACTB", COHORT_A, "v2", 5.0, 15.0),
        _ref_row(ACTB, "ACTB", COHORT_B, "v2", 6.0, 16.0),
        # MALAT1: non-coding, COHORT_A only (drives coverage + biology filters).
        _ref_row(MALAT1, "MALAT1", COHORT_A, "v2", 7.0, 17.0),
    ])


def _write_artifact_from_rebuild(root, monkeypatch, fake):
    """Materialize the precomputed artifact exactly as the generator does — as
    the serialized output of ``_rebuild_full_canonical_views`` over ``fake`` —
    so artifact and rebuild are the same data through two code paths."""
    monkeypatch.setattr(accessors, "_load_cancer_reference_expression", lambda: fake)
    accessors._REFERENCE_VIEW_CACHE.clear()
    accessors._load_precomputed_cohort_views.cache_clear()
    tpm, clean, prov = accessors._rebuild_full_canonical_views()
    root.mkdir(parents=True, exist_ok=True)
    tpm.to_parquet(root / "tpm.parquet", index=False)
    clean.to_parquet(root / "clean_tpm.parquet", index=False)
    prov.to_parquet(root / "provenance.parquet", index=False)
    (root / "_manifest.json").write_text('{"canonical_gene_ids": true}\n')
    return tpm, clean, prov


def _normalize(df):
    import pandas as pd

    cols = sorted(df.columns)
    keys = [c for c in ("Ensembl_Gene_ID", "Symbol") if c in cols]
    out = df[cols]
    if keys:
        out = out.sort_values(keys)
    return out.reset_index(drop=True)


def _assert_views_equal(a, b):
    import pandas as pd

    for attr in ("tpm", "clean_tpm", "clean_tpm_biological"):
        pd.testing.assert_frame_equal(
            _normalize(getattr(a, attr)), _normalize(getattr(b, attr)),
            check_dtype=False, check_like=True, rtol=1e-9, atol=1e-9,
        )
    pa = a.provenance.sort_values(list(a.provenance.columns)).reset_index(drop=True)
    pb = b.provenance.sort_values(list(b.provenance.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(pa, pb, check_dtype=False, check_like=True)


_FILTER_CASES = [
    dict(),
    dict(cancer_types=COHORT_A),
    dict(cancer_types=[COHORT_A, COHORT_B]),
    dict(genes=["TP53"]),
    dict(genes=["TP53", "ACTB", "MALAT1"]),
    dict(cancer_types=COHORT_A, genes=["TP53", "MALAT1"]),
    dict(cancer_types=COHORT_B, genes=["MALAT1"]),       # gene absent in COHORT_B
    dict(protein_coding=True),
    dict(genes=["TP53", "MALAT1"], protein_coding=True),
    dict(min_cohort_coverage=1.0),
    dict(min_cohort_coverage=0.5),
    dict(cancer_types=[COHORT_A, COHORT_B], min_cohort_coverage=1.0),
]


@pytest.mark.parametrize("kwargs", _FILTER_CASES)
def test_artifact_path_equals_rebuild_path(tmp_path, monkeypatch, kwargs):
    """The gold invariant: for any filter combination, slicing the precomputed
    artifact returns exactly what rebuilding from the reference returns."""
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)

    # Artifact present → fast path.
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    accessors._load_precomputed_cohort_views.cache_clear()
    from_artifact = cohort_expression_views(**kwargs)

    # Artifact absent → rebuild fallback (same reference + same filter).
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root / "absent")
    accessors._REFERENCE_VIEW_CACHE.clear()
    from_rebuild = cohort_expression_views(**kwargs)

    _assert_views_equal(from_artifact, from_rebuild)


@pytest.mark.parametrize("cancer_types,genes", [
    ("CLL", ["MS4A1"]),
    ("CLL", ["MS4A1", "MALAT1", "RPL13A"]),
    ("PRAD", ["KLK3", "AR", "FOXA1"]),
    ("SARC", ["TP53"]),
])
def test_artifact_matches_from_reference_oracle_current_symbols(cancer_types, genes):
    """For current (non-retired) symbols, the artifact slice agrees with the
    independent from-reference builder on the real bundle — cross-checking the
    'slice the full matrix' path against 'filter during the pivot'."""
    fast = cohort_expression_views(cancer_types, genes=genes)
    oracle = accessors._cohort_expression_views_from_reference(
        cancer_types, genes=genes, canonicalize_genes=True)
    _assert_views_equal(fast, oracle)


def test_canonical_path_resolves_retired_synonym():
    """The canonical path resolves a retired NCBI synonym to its current gene
    (GNB2L1 → RACK1, ENSG00000204628). This is intentionally richer than the
    plain ``filter_to_genes`` symbol match."""
    v = cohort_expression_views("CLL", genes=["GNB2L1"])
    assert "ENSG00000204628" in set(v.tpm["Ensembl_Gene_ID"])


@pytest.mark.parametrize("cancer_types", ["CLL", "PRAD", ["CLL", "PRAD"]])
def test_cohort_only_view_drops_unmeasured_genes(cancer_types):
    """Narrowing to a cohort (no gene filter) must return only genes measured in
    that cohort — matching the from-reference pivot — not the full all-cohort
    gene union padded with NaN (#474 review, P2a)."""
    fast = cohort_expression_views(cancer_types)
    oracle = accessors._cohort_expression_views_from_reference(
        cancer_types, canonicalize_genes=True)
    _assert_views_equal(fast, oracle)
    # Every returned gene is measured in at least one selected cohort.
    cohort_cols = [c for c in fast.tpm.columns
                   if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert fast.tpm[cohort_cols].notna().any(axis=1).all()


def test_full_view_keeps_entire_gene_union(tmp_path, monkeypatch):
    """With no cancer_types and no genes, the view is the full canonical matrix
    (no row pruning) — the narrowing prune must not fire on the unfiltered
    request."""
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    accessors._load_precomputed_cohort_views.cache_clear()
    full = cohort_expression_views()
    # All three synthetic genes survive (TP53, ACTB, MALAT1), both cohorts.
    assert set(full.tpm["Ensembl_Gene_ID"]) == {TP53, ACTB, MALAT1}
    assert sorted(c for c in full.tpm.columns
                  if c not in ("Ensembl_Gene_ID", "Symbol")) == [COHORT_A, COHORT_B]


def test_cohort_only_view_excludes_single_cohort_gene(tmp_path, monkeypatch):
    """MALAT1 lives only in COHORT_A; a COHORT_B-only view must not carry it."""
    v = _fast_views(tmp_path, monkeypatch, cancer_types=COHORT_B)
    assert MALAT1 not in set(v.tpm["Ensembl_Gene_ID"])
    assert {TP53, ACTB} <= set(v.tpm["Ensembl_Gene_ID"])


# ---------- fallback / corruption robustness ----------

def _install_fake_reference(monkeypatch, fake):
    monkeypatch.setattr(accessors, "_load_cancer_reference_expression", lambda: fake)
    accessors._REFERENCE_VIEW_CACHE.clear()
    accessors._load_precomputed_cohort_views.cache_clear()


def test_missing_one_parquet_falls_back_to_rebuild(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    (root / "clean_tpm.parquet").unlink()  # incomplete artifact

    _install_fake_reference(monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    assert accessors._cohort_views_usable(root) is False
    v = cohort_expression_views(genes=["TP53"])      # must not raise
    assert "ENSG00000141510" in set(v.tpm["Ensembl_Gene_ID"])


def test_corrupt_parquet_falls_back_to_rebuild(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    (root / "tpm.parquet").write_bytes(b"not a parquet file")  # corrupt

    _install_fake_reference(monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    # Corrupt artifact must not raise — and must warn so the slow fallback is
    # observable rather than a silent performance cliff.
    with pytest.warns(RuntimeWarning, match="could not be read"):
        v = cohort_expression_views(genes=["TP53"])
    assert "ENSG00000141510" in set(v.tpm["Ensembl_Gene_ID"])


def test_schema_invalid_artifact_falls_back_to_rebuild(tmp_path, monkeypatch):
    import pandas as pd

    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    # Drop the id columns from tpm.parquet → schema-invalid.
    bad = pd.DataFrame({"AAA": [1.0], "BBB": [2.0]})
    bad.to_parquet(root / "tpm.parquet", index=False)

    _install_fake_reference(monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    accessors._load_precomputed_cohort_views.cache_clear()
    with pytest.warns(RuntimeWarning, match="schema"):
        v = cohort_expression_views(genes=["TP53"])      # must not raise
    assert "ENSG00000141510" in set(v.tpm["Ensembl_Gene_ID"])


def test_non_canonical_manifest_rejected(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    (root / "_manifest.json").write_text('{"canonical_gene_ids": false}\n')
    assert accessors._cohort_views_usable(root) is False


def test_malformed_manifest_treated_as_canonical(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    (root / "_manifest.json").write_text("{ not valid json")
    # Unreadable manifest → no claim either way → present artifact still usable.
    assert accessors._cohort_views_usable(root) is True


def test_absent_manifest_treated_as_canonical(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    (root / "_manifest.json").unlink()
    assert accessors._cohort_views_usable(root) is True


# ---------- gene-filter edge cases (on the fast path) ----------

def _fast_views(tmp_path, monkeypatch, **kwargs):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    accessors._load_precomputed_cohort_views.cache_clear()
    return cohort_expression_views(**kwargs)


def test_empty_gene_list_yields_no_rows(tmp_path, monkeypatch):
    v = _fast_views(tmp_path, monkeypatch, genes=[])
    assert v.tpm.empty
    assert v.clean_tpm.empty
    assert v.clean_tpm_biological.empty


def test_unknown_gene_yields_no_rows(tmp_path, monkeypatch):
    v = _fast_views(tmp_path, monkeypatch, genes=["NOT_A_REAL_GENE_XYZ"])
    assert v.tpm.empty


def test_duplicate_and_whitespace_genes_dedupe(tmp_path, monkeypatch):
    v = _fast_views(tmp_path, monkeypatch, genes=["TP53", " tp53 ", "TP53"])
    assert v.tpm["Ensembl_Gene_ID"].tolist() == [TP53]


def test_gene_filter_drops_cohorts_with_no_data(tmp_path, monkeypatch):
    # MALAT1 only exists in COHORT_A, so selecting it must drop COHORT_B's column.
    v = _fast_views(tmp_path, monkeypatch, genes=["MALAT1"])
    cohort_cols = [c for c in v.tpm.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    assert cohort_cols == [COHORT_A]
    assert set(v.provenance["source_cohort"]) == {f"{COHORT_A}_SRC"}


# ---------- provenance / column hygiene ----------

def test_provenance_never_exposes_cancer_code(tmp_path, monkeypatch):
    v = _fast_views(tmp_path, monkeypatch)
    assert "cancer_code" not in v.provenance.columns
    assert list(v.provenance.columns) == [
        "source_cohort", "processing_pipeline", "n_samples"]


def test_canonicalize_collapses_symbol_drift_on_fast_path(tmp_path, monkeypatch):
    # COHORT_A carried TP53 under two symbols across versions; the baked artifact
    # must hold a single canonical row summing both (1.0 + 2.0 = 3.0).
    v = _fast_views(tmp_path, monkeypatch, cancer_types=COHORT_A, genes=["TP53"])
    assert v.tpm["Ensembl_Gene_ID"].tolist() == [TP53]
    assert v.tpm[COHORT_A].iloc[0] == 3.0


# ---------- caching / invalidation ----------

def test_rebuild_memoized_on_reference_identity(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    _install_fake_reference(monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: tmp_path / "absent")

    calls = {"n": 0}
    real = accessors.cancer_reference_expression

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(accessors, "cancer_reference_expression", counting)
    accessors._REFERENCE_VIEW_CACHE.clear()

    cohort_expression_views(genes=["TP53"])
    first = calls["n"]
    cohort_expression_views(genes=["ACTB"])          # same reference frame
    assert calls["n"] == first                       # rebuild served from memo


# ---------- canonicalize_genes=False opt-out ----------

def test_canonicalize_false_uses_reference_not_artifact(tmp_path, monkeypatch):
    fake = _synthetic_reference()
    root = tmp_path / "views"
    _write_artifact_from_rebuild(root, monkeypatch, fake)
    _install_fake_reference(monkeypatch, fake)
    monkeypatch.setattr(accessors, "_cohort_views_root", lambda: root)
    # The artifact must NOT be consulted for the non-canonical path.
    monkeypatch.setattr(
        accessors, "_full_canonical_views",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("artifact used")),
    )
    v = cohort_expression_views(COHORT_A, genes=["TP53"], canonicalize_genes=False)
    assert TP53 in set(v.tpm["Ensembl_Gene_ID"])
