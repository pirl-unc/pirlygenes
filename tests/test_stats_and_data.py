"""Tests for the extended cancer-reference-expression schema, the
shared :mod:`pirlygenes.expression.stats` helpers, and the
``pirlygenes data list`` CLI surface."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

from pirlygenes import cli, data_inventory
from pirlygenes.expression.stats import (
    CLEAN_STAT_COLUMNS,
    COUNT_COLUMNS,
    REFERENCE_COLUMNS,
    STAT_COLUMNS,
    TUMOR_ORIGIN_VALUES,
    assign_stats,
    compute_cohort_stats,
    compute_count_columns,
    round_stat_columns,
    upsert_to_shard,
)
from pirlygenes.load_dataset import get_data


# ---------- schema ----------


def test_reference_columns_starts_with_legacy_order():
    legacy = (
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "TPM_median",
        "TPM_q1",
        "TPM_q3",
        "TPM_mean",
        "TPM_clean_median",
        "TPM_clean_q1",
        "TPM_clean_q3",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
    )
    assert REFERENCE_COLUMNS[: len(legacy)] == legacy


def test_reference_columns_appends_v53_extension():
    # The v5.3 extension is followed by the v5.4 cohort-annotation
    # extension (tumor_origin / metastasis_site). Locate the v5.3 block
    # by its first column ("TPM_std") and verify the exact 15-column
    # extension lives at that position.
    extension = (
        "TPM_std",
        "TPM_min",
        "TPM_max",
        "TPM_p5",
        "TPM_p10",
        "TPM_p90",
        "TPM_p95",
        "TPM_clean_mean",
        "TPM_clean_std",
        "TPM_clean_min",
        "TPM_clean_max",
        "TPM_clean_p5",
        "TPM_clean_p10",
        "TPM_clean_p90",
        "TPM_clean_p95",
    )
    start = REFERENCE_COLUMNS.index("TPM_std")
    assert REFERENCE_COLUMNS[start : start + len(extension)] == extension


def test_reference_columns_appends_v54_cohort_annotation():
    # The v5.4 extension carries tumor_origin / metastasis_site at the
    # very end of REFERENCE_COLUMNS so existing positional consumers
    # keep working unchanged.
    annotation = ("tumor_origin", "metastasis_site")
    assert REFERENCE_COLUMNS[-len(annotation):] == annotation


def test_stat_columns_have_raw_and_clean_parity():
    # The clean column tuple is generated from STAT_COLUMNS; they must
    # stay paired so any future addition lands on both sides.
    assert len(STAT_COLUMNS) == len(CLEAN_STAT_COLUMNS)
    for raw, clean in zip(STAT_COLUMNS, CLEAN_STAT_COLUMNS):
        assert clean == "TPM_clean_" + raw.removeprefix("TPM_")


def test_bundled_csv_has_full_schema():
    df = get_data("cancer-reference-expression")
    for col in REFERENCE_COLUMNS:
        assert col in df.columns, f"missing column {col!r}"


def test_every_bundled_shard_has_tumor_origin_set():
    """Catch any shard that ships without tumor_origin set —
    upsert_to_shard now rejects writes that violate this, but pre-v5.4
    files could still slip through if a builder is updated incorrectly."""
    df = get_data("cancer-reference-expression")
    bad = df[df["tumor_origin"].isna()]
    assert bad.empty, (
        f"{len(bad)} bundled rows have null tumor_origin; offending "
        f"source_cohorts: {sorted(bad['source_cohort'].unique())}"
    )


def test_every_bundled_tumor_origin_is_in_enum():
    """Catch typos like 'metastatic' that would otherwise slip past
    upsert_to_shard's validation if a legacy shard was ever
    hand-edited."""
    df = get_data("cancer-reference-expression")
    observed = set(df["tumor_origin"].dropna().astype(str).unique())
    invalid = observed - TUMOR_ORIGIN_VALUES
    assert not invalid, (
        f"unrecognised tumor_origin values in bundled data: {invalid}; "
        f"allowed are {sorted(TUMOR_ORIGIN_VALUES)}"
    )


# ---------- upsert_to_shard validation ----------


def _stat_kwargs(n_genes: int, n_samples: int) -> dict:
    """Build a minimal valid stat-columns block for upsert_to_shard tests."""
    import numpy as np
    cols = {c: np.zeros(n_genes, dtype=float) for c in STAT_COLUMNS}
    cols.update({c: np.zeros(n_genes, dtype=float) for c in CLEAN_STAT_COLUMNS})
    cols["n_samples"] = np.full(n_genes, n_samples, dtype=int)
    cols["n_detected"] = np.zeros(n_genes, dtype=int)
    return cols


def _minimal_rows(*, cancer_code: str, tumor_origin=None) -> pd.DataFrame:
    n = 2
    base = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG00000000001", "ENSG00000000002"],
        "Symbol": ["FAKEA", "FAKEB"],
        "cancer_code": cancer_code,
        "source_cohort": "TEST_SOURCE",
        "source_project": "test",
        "source_version": "test_v1",
        "processing_pipeline": "test_pipeline",
        "notes": "fixture",
    })
    for k, v in _stat_kwargs(n, n_samples=5).items():
        base[k] = v
    if tumor_origin is not None:
        base["tumor_origin"] = tumor_origin
    base["metastasis_site"] = pd.NA
    return base


def test_upsert_to_shard_rejects_missing_tumor_origin(tmp_path):
    """A builder that forgets to set tumor_origin must fail at write time."""
    import pytest as _pt
    rows = _minimal_rows(cancer_code="FAKE")
    # Drop the column entirely; simulates a builder that never set it.
    rows = rows.drop(columns=["tumor_origin"], errors="ignore")
    with _pt.raises(ValueError, match="tumor_origin"):
        upsert_to_shard(
            tmp_path, rows,
            source_cohort="TEST_SOURCE", cancer_codes=["FAKE"],
        )


def test_upsert_to_shard_rejects_invalid_tumor_origin(tmp_path):
    """Typos like 'metastatic' (vs 'metastasis') must fail at write time."""
    import pytest as _pt
    rows = _minimal_rows(cancer_code="FAKE", tumor_origin="metastatic")
    with _pt.raises(ValueError, match="unrecognised tumor_origin"):
        upsert_to_shard(
            tmp_path, rows,
            source_cohort="TEST_SOURCE", cancer_codes=["FAKE"],
        )


def test_upsert_to_shard_accepts_valid_tumor_origin(tmp_path):
    """Sanity: a correctly-set tumor_origin passes validation and lands
    on disk via the regular shard path."""
    rows = _minimal_rows(cancer_code="FAKE", tumor_origin="primary")
    written = upsert_to_shard(
        tmp_path, rows,
        source_cohort="TEST_SOURCE", cancer_codes=["FAKE"],
    )
    assert (tmp_path / "TEST_SOURCE.csv.gz").exists()
    assert set(written["tumor_origin"]) == {"primary"}


def test_upsert_to_shard_canonicalizes_renamed_codes(tmp_path):
    """A builder still emitting a pre-rename code (e.g. recount3 routing →
    'MID_NET') must land under the current registry code ('NET_MIDGUT'),
    and the cross-code upsert must REPLACE any existing canonical-name rows
    rather than leaving a stale rename-orphan copy alongside it."""
    from pirlygenes.gene_sets_cancer import canonical_cancer_code
    assert canonical_cancer_code("MID_NET") == "NET_MIDGUT"
    assert canonical_cancer_code("PRAD") == "PRAD"  # untouched

    # Seed the shard with a stale canonical-name row (as a prior in-place
    # rename migration would have left it).
    stale = _minimal_rows(cancer_code="NET_MIDGUT", tumor_origin="primary")
    stale["source_version"] = "stale_v1"
    upsert_to_shard(
        tmp_path, stale, source_cohort="TEST_SOURCE", cancer_codes=["NET_MIDGUT"],
    )
    # New build emits the OLD code name; cancer_codes also uses the old name.
    fresh = _minimal_rows(cancer_code="MID_NET", tumor_origin="primary")
    fresh["source_version"] = "fresh_v4"
    merged = upsert_to_shard(
        tmp_path, fresh, source_cohort="TEST_SOURCE", cancer_codes=["MID_NET"],
    )
    # Result: only the canonical code survives, carrying the fresh data.
    assert set(merged["cancer_code"]) == {"NET_MIDGUT"}
    assert set(merged["source_version"]) == {"fresh_v4"}
    on_disk = pd.read_csv(tmp_path / "TEST_SOURCE.csv.gz")
    assert set(on_disk["cancer_code"]) == {"NET_MIDGUT"}
    assert "MID_NET" not in set(on_disk["cancer_code"])


def test_upsert_samples_manifest_preserves_other_cohorts_rows_and_columns(tmp_path):
    """A builder rebuilding cohort A must not drop cohort B's rows — nor strip
    columns B carries that A's manifest lacks (the v5.20.0 truncation/column-
    stripping bug). Replacement is keyed on the source_cohorts present in the
    new rows; everything else is preserved verbatim."""
    from pirlygenes.expression.stats import upsert_samples_manifest

    path = tmp_path / "samples.csv.gz"
    # Cohort B carries a 'lineage_label' column that cohort A's builder omits.
    cohort_b = pd.DataFrame({
        "cancer_code": ["B", "B"],
        "source_cohort": ["COHORT_B", "COHORT_B"],
        "sample_id": ["b1", "b2"],
        "included": [True, True],
        "lineage_label": ["lin_b", "lin_b"],
    })
    upsert_samples_manifest(path, cohort_b)

    # Rebuild cohort A with a NARROWER column set (no lineage_label) + stale A row
    # already present to prove replacement.
    cohort_a_v1 = pd.DataFrame({
        "cancer_code": ["A"], "source_cohort": ["COHORT_A"],
        "sample_id": ["a_old"], "included": [False],
    })
    upsert_samples_manifest(path, cohort_a_v1)
    cohort_a_v2 = pd.DataFrame({
        "cancer_code": ["A", "A"], "source_cohort": ["COHORT_A", "COHORT_A"],
        "sample_id": ["a1", "a2"], "included": [True, True],
    })
    out = upsert_samples_manifest(path, cohort_a_v2)

    on_disk = pd.read_csv(path)
    # Cohort B fully preserved, including its lineage_label values.
    b = on_disk[on_disk["source_cohort"] == "COHORT_B"]
    assert len(b) == 2
    assert set(b["lineage_label"]) == {"lin_b"}
    # Cohort A replaced (stale a_old gone, a1/a2 present); union columns kept.
    a = on_disk[on_disk["source_cohort"] == "COHORT_A"]
    assert set(a["sample_id"]) == {"a1", "a2"}
    assert "a_old" not in set(on_disk["sample_id"])
    assert "lineage_label" in on_disk.columns
    assert out["source_cohort"].value_counts().to_dict() == {"COHORT_A": 2, "COHORT_B": 2}


def test_upsert_to_shard_per_cancer_code_shards_writes_one_file_per_code(tmp_path):
    """When per_cancer_code_shards=True, write `<source>__<code>.csv.gz`
    per code so a multi-code source can stay under GitHub's 100 MiB
    hard limit even after the schema grows."""
    rows_a = _minimal_rows(cancer_code="CODE_A", tumor_origin="primary")
    rows_b = _minimal_rows(cancer_code="CODE_B", tumor_origin="primary")
    rows = pd.concat([rows_a, rows_b], ignore_index=True)
    upsert_to_shard(
        tmp_path, rows,
        source_cohort="TEST_SPLIT",
        cancer_codes=["CODE_A", "CODE_B"],
        per_cancer_code_shards=True,
    )
    files = sorted(p.name for p in tmp_path.glob("TEST_SPLIT*.csv.gz"))
    assert files == ["TEST_SPLIT__CODE_A.csv.gz", "TEST_SPLIT__CODE_B.csv.gz"]


def test_per_cancer_code_shards_concat_back_via_loader(tmp_path):
    """Round-trip: when ``per_cancer_code_shards=True`` splits a
    cohort across per-code files, the shard loader
    (:func:`pirlygenes.load_dataset._load_shard_directory`) must
    transparently concat them so downstream consumers see the same
    logical dataset as a single combined shard would have produced.

    This is the safety net behind the TCGA-subset re-shard: without
    this guarantee, splitting any source past the GitHub size limit
    silently breaks every cancer-reference-expression reader.
    """
    from pirlygenes.load_dataset import _load_shard_directory

    rows_a = _minimal_rows(cancer_code="CODE_A", tumor_origin="primary")
    rows_b = _minimal_rows(cancer_code="CODE_B", tumor_origin="mixed")
    rows = pd.concat([rows_a, rows_b], ignore_index=True)
    upsert_to_shard(
        tmp_path, rows,
        source_cohort="ROUNDTRIP",
        cancer_codes=["CODE_A", "CODE_B"],
        per_cancer_code_shards=True,
    )
    # Two files on disk, one per code
    files = sorted(p.name for p in tmp_path.glob("ROUNDTRIP*.csv.gz"))
    assert files == ["ROUNDTRIP__CODE_A.csv.gz", "ROUNDTRIP__CODE_B.csv.gz"]

    # Loader concats both into a single logical frame, preserving the
    # per-code tumor_origin annotations
    loaded = _load_shard_directory(tmp_path)
    assert set(loaded["cancer_code"]) == {"CODE_A", "CODE_B"}
    by_code = loaded.set_index("cancer_code")["tumor_origin"].to_dict()
    assert by_code == {"CODE_A": "primary", "CODE_B": "mixed"}
    # Row count = sum of per-code shards (2 genes × 2 codes = 4 rows)
    assert len(loaded) == len(rows)


def test_upsert_to_shard_per_cancer_code_warns_on_unexpected_codes(tmp_path):
    """Codes present in new_rows but missing from the cancer_codes list
    usually indicate accidental cross-contamination in the input
    frame — surface a warning so the builder author notices."""
    import pytest as _pt
    rows_a = _minimal_rows(cancer_code="CODE_A", tumor_origin="primary")
    rows_b = _minimal_rows(cancer_code="STRAY", tumor_origin="primary")
    rows = pd.concat([rows_a, rows_b], ignore_index=True)
    with _pt.warns(UserWarning, match="STRAY"):
        upsert_to_shard(
            tmp_path, rows,
            source_cohort="TEST_SPLIT_WARN",
            cancer_codes=["CODE_A"],   # 'STRAY' not listed
            per_cancer_code_shards=True,
        )
    # Both files written — the warning surfaces the surprise but
    # doesn't block the data, matching the docstring's "writing them
    # anyway" promise.
    files = sorted(p.name for p in tmp_path.glob("TEST_SPLIT_WARN*.csv.gz"))
    assert "TEST_SPLIT_WARN__CODE_A.csv.gz" in files
    assert "TEST_SPLIT_WARN__STRAY.csv.gz" in files


def test_data_bundle_prune_lists_and_deletes_stale_dirs(tmp_path, monkeypatch):
    """``pirlygenes data prune`` should list every cache dir, keep the
    current version's, and delete the rest."""
    from pirlygenes import data_bundle

    # Build a fake cache root with two stale version dirs + the
    # current-version dir.
    monkeypatch.setenv(
        "PIRLYGENES_BUNDLED_DATA", str(tmp_path / f"v{data_bundle.DATA_VERSION}"),
    )
    for v in ["v5.0.0", "v5.1.0", f"v{data_bundle.DATA_VERSION}"]:
        d = tmp_path / v
        d.mkdir()
        (d / "marker.csv").write_text("x")

    versions = data_bundle.list_cache_versions()
    by_v = {e["version"]: e for e in versions}
    assert {"v5.0.0", "v5.1.0", f"v{data_bundle.DATA_VERSION}"} <= set(by_v)
    assert by_v[f"v{data_bundle.DATA_VERSION}"]["is_current"] is True
    assert by_v["v5.0.0"]["is_current"] is False

    # Dry-run: returns candidates but leaves disk alone
    candidates = data_bundle.prune_cache(keep_current=True, dry_run=True)
    candidate_versions = {c["version"] for c in candidates}
    assert candidate_versions == {"v5.0.0", "v5.1.0"}
    assert (tmp_path / "v5.0.0").exists()  # still there

    # Real prune
    data_bundle.prune_cache(keep_current=True, dry_run=False)
    assert not (tmp_path / "v5.0.0").exists()
    assert not (tmp_path / "v5.1.0").exists()
    assert (tmp_path / f"v{data_bundle.DATA_VERSION}").exists()


def test_upsert_to_shard_allow_unset_for_legacy_backfill(tmp_path):
    """The v5.4 migration backfill rewrites legacy rows; ``allow_unset_tumor_origin``
    lets it pass NaN through during that one-time migration."""
    rows = _minimal_rows(cancer_code="FAKE", tumor_origin=None)
    rows = rows.drop(columns=["tumor_origin"], errors="ignore")
    upsert_to_shard(
        tmp_path, rows,
        source_cohort="TEST_LEGACY", cancer_codes=["FAKE"],
        allow_unset_tumor_origin=True,
    )
    assert (tmp_path / "TEST_LEGACY.csv.gz").exists()


# ---------- compute_cohort_stats ----------


def test_compute_cohort_stats_against_known_values():
    # Two genes, five samples. Picked so every stat is hand-checkable.
    values = pd.DataFrame(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],   # mean 2, std≈1.581, q1=1, median=2, q3=3
            [10.0, 10.0, 10.0, 10.0, 10.0],  # all-10
        ],
        index=["g1", "g2"],
    )
    stats = compute_cohort_stats(values)
    assert stats["TPM_median"].tolist() == [2.0, 10.0]
    assert stats["TPM_mean"].tolist() == [2.0, 10.0]
    assert stats["TPM_q1"].tolist() == [1.0, 10.0]
    assert stats["TPM_q3"].tolist() == [3.0, 10.0]
    assert stats["TPM_min"].tolist() == [0.0, 10.0]
    assert stats["TPM_max"].tolist() == [4.0, 10.0]
    # std for {0,1,2,3,4} with ddof=1 is sqrt(2.5) ≈ 1.5811
    assert stats["TPM_std"][0] == np.sqrt(2.5)
    assert stats["TPM_std"][1] == 0.0
    # p5/p10/p90/p95 on a 5-vector use linear interpolation
    assert stats["TPM_p10"][0] == 0.4
    assert stats["TPM_p90"][0] == 3.6


def test_compute_cohort_stats_clean_prefix():
    values = pd.DataFrame([[1.0, 2.0, 3.0]], index=["g1"])
    stats = compute_cohort_stats(values, prefix="TPM_clean_")
    assert set(stats.keys()) == set(CLEAN_STAT_COLUMNS)


def test_compute_cohort_stats_single_sample_has_nan_std():
    values = pd.DataFrame([[7.0]], index=["g1"])
    stats = compute_cohort_stats(values)
    assert np.isnan(stats["TPM_std"][0])
    assert stats["TPM_mean"][0] == 7.0


def test_compute_count_columns():
    values = pd.DataFrame(
        [
            [0.0, 0.0, 1.0],   # 1 of 3 detected
            [5.0, 5.0, 5.0],   # 3 of 3 detected
        ],
        index=["g1", "g2"],
    )
    counts = compute_count_columns(values)
    assert counts["n_samples"].tolist() == [3, 3]
    assert counts["n_detected"].tolist() == [1, 3]


def test_assign_stats_populates_full_suite():
    raw = pd.DataFrame([[0.0, 2.0, 4.0]], index=["g1"])
    clean = pd.DataFrame([[0.0, 2.0, 4.0]], index=["g1"])
    out = pd.DataFrame({"Ensembl_Gene_ID": ["ENSG1"], "Symbol": ["S1"]})
    assign_stats(out, raw, clean)
    for col in STAT_COLUMNS + CLEAN_STAT_COLUMNS + COUNT_COLUMNS:
        assert col in out.columns, f"assign_stats failed to populate {col!r}"


def test_round_stat_columns_only_touches_known_columns():
    out = pd.DataFrame({"TPM_median": [1.234567891], "other": [9.999999]})
    rounded = round_stat_columns(out)
    assert rounded["TPM_median"].iloc[0] == round(1.234567891, 6)
    assert rounded["other"].iloc[0] == 9.999999


# ---------- data inventory + CLI ----------


def test_summarize_inventory_smoke():
    snapshot = data_inventory.summarize_inventory()
    assert snapshot.total_rows > 0
    assert snapshot.unique_genes > 0
    assert snapshot.cohort_rows
    assert snapshot.registered_sources > 0
    cohort_codes = {row.cancer_code for row in snapshot.cohort_rows}
    assert "BL" in cohort_codes
    assert "MM" in cohort_codes


def test_render_inventory_contains_expected_lines():
    snapshot = data_inventory.summarize_inventory()
    rendered = data_inventory.render_inventory(snapshot)
    assert "cancer-reference-expression" in rendered
    assert "size on disk" in rendered
    assert "samples:" in rendered                # totals include a sample count
    # flat columnar view is one row per cohort sorted by samples, with
    # distinct tool/unit/derivation/source columns
    flat = data_inventory.render_inventory(snapshot, flat=True)
    for col in ("tool", "unit", "stratified by", "source"):
        assert col in flat
    assert "normalized to clean TPM" in rendered  # the native-unit note
    assert "genes" in rendered                   # counts labelled genes, not "rows"
    assert "Burkitt Lymphoma" in rendered        # cancer-type name shown per code
    assert "BL" in rendered


def _run_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = cli.main(args)
    return rc, stdout.getvalue(), stderr.getvalue()


def test_cli_data_list_smoke():
    rc, out, _ = _run_cli(["data", "list"])
    assert rc == 0
    assert "cancer-reference-expression" in out
    assert "BL" in out


def test_cli_data_with_no_action_usage():
    rc, _, err = _run_cli(["data"])
    assert rc == 2
    assert "data" in err
