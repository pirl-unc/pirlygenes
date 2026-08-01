"""Tests for the delegated reference schema and pure statistic helpers."""

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
)
from pirlygenes.load_dataset import get_data


# ---------- schema ----------


def test_data_bundle_excludes_delegated_reference_expression_shards():
    from pirlygenes import data_bundle

    assert not any(
        path == "cancer-reference-expression"
        or path.startswith("cancer-reference-expression/")
        for path in data_bundle.DOWNLOADABLE_PATHS
    )


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


def test_every_delegated_row_has_tumor_origin_set():
    """The owner dataset must provide a source-origin annotation."""
    df = get_data("cancer-reference-expression")
    bad = df[df["tumor_origin"].isna()]
    assert bad.empty, (
        f"{len(bad)} bundled rows have null tumor_origin; offending "
        f"source_cohorts: {sorted(bad['source_cohort'].unique())}"
    )


def test_every_delegated_tumor_origin_is_in_enum():
    """The owner dataset must use the compatibility origin vocabulary."""
    df = get_data("cancer-reference-expression")
    observed = set(df["tumor_origin"].dropna().astype(str).unique())
    invalid = observed - TUMOR_ORIGIN_VALUES
    assert not invalid, (
        f"unrecognised tumor_origin values in bundled data: {invalid}; "
        f"allowed are {sorted(TUMOR_ORIGIN_VALUES)}"
    )


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


def test_inventory_preserves_oncorefs_canonical_source_cohort_labels(
    tmp_path, monkeypatch,
):
    import pirlygenes.expression as expression

    storage = "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES"
    canonical = "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["E1", "E1", "E1"],
            "cancer_code": ["SARC_DDLPS", "SARC_WDLPS", "SARC_PLEOLPS"],
            "source_cohort": [canonical, canonical, storage],
            "source_project": ["Treehouse"] * 3,
            "n_samples": [48, 5, 4],
            "processing_pipeline": ["treehouse_polya"] * 3,
            "tumor_origin": ["primary"] * 3,
        }
    ).to_csv(tmp_path / "sarc.csv", index=False)
    monkeypatch.setattr(data_inventory, "_active_reference_dir", lambda: tmp_path)
    monkeypatch.setattr(
        data_inventory,
        "_SUMMARY_CACHE",
        tmp_path / "inventory_summary.json",
    )
    monkeypatch.setattr(data_inventory, "load_registry", lambda: [])
    monkeypatch.setattr(
        expression,
        "available_cancer_expression_references",
        lambda: pd.DataFrame(columns=[
            "cancer_code",
            "source_cohort",
            "source_project",
            "n_samples",
            "processing_pipeline",
            "tumor_origin",
        ]),
    )

    snapshot = data_inventory.summarize_inventory(progress=False)
    cohort_for = {row.cancer_code: row.source_cohort for row in snapshot.cohort_rows}

    assert cohort_for["SARC_DDLPS"] == canonical
    assert cohort_for["SARC_WDLPS"] == canonical
    assert cohort_for["SARC_PLEOLPS"] == storage


def test_summarize_inventory_smoke():
    snapshot = data_inventory.summarize_inventory()
    assert snapshot.total_rows > 0
    assert snapshot.unique_genes > 0
    assert snapshot.cohort_rows
    assert snapshot.registered_sources > 0
    cohort_codes = {row.cancer_code for row in snapshot.cohort_rows}
    assert "BL" in cohort_codes
    assert "MM" in cohort_codes


def test_inventory_keys_match_public_reference_manifest():
    from pirlygenes.expression import available_cancer_expression_references

    snapshot = data_inventory.summarize_inventory(progress=False)
    inventory_keys = {
        (row.cancer_code, row.source_cohort) for row in snapshot.cohort_rows
    }
    manifest = available_cancer_expression_references()
    manifest_keys = set(
        manifest[["cancer_code", "source_cohort"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )

    assert inventory_keys == manifest_keys
    ess = {
        row.cancer_code: row
        for row in snapshot.cohort_rows
        if row.cancer_code in {"SARC_ESS_HG", "SARC_ESS_LG"}
    }
    assert ess["SARC_ESS_HG"].n_rows is None
    assert ess["SARC_ESS_HG"].n_samples == 4
    assert ess["SARC_ESS_LG"].n_rows is None
    assert ess["SARC_ESS_LG"].n_samples == 9


def test_inventory_cache_signature_tracks_owner_data_version(tmp_path):
    shard = tmp_path / "reference.csv.gz"
    shard.write_bytes(b"fixture")

    before = data_inventory._shard_signature(
        [shard], owner_data_version="5.23.7"
    )
    after = data_inventory._shard_signature(
        [shard], owner_data_version="5.23.8"
    )

    assert before != after


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
