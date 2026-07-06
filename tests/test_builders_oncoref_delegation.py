"""End-to-end fixture tests for the oncoref-delegated source-matrix builders.

These exercise the real pipeline (read → normalize → oncoref canonicalize →
oncoref sample-QC → clean-TPM stats → shard upsert) on a synthetic matrix with
no network: the source file is pre-placed so ``_download`` skips, and the
per-sample parquet cache is redirected to the tmp dir.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pirlygenes.builders import oncoref_source as osrc


# ─── shared helper (unit) ─────────────────────────────────────────────────


def test_canonicalize_source_resolves_entrez_and_sums_duplicate_genes():
    """A source matrix mixing versioned ENSG, an Entrez id for the same gene,
    a HUGO symbol, and a junk row: Entrez resolves, the duplicate gene sums in
    linear space, and the junk row lands in the audit as high-expression
    unresolved."""
    tpm = pd.DataFrame(
        {"a": [100.0, 20.0, 30.0, 5.0], "b": [80.0, 10.0, 0.0, 0.0]},
        index=["ENSG00000141510.11", "7157", "EGFR", "NOTAGENE"],  # 7157 == TP53
    )
    tpm.index.name = "gene_id"
    res = osrc.canonicalize_source(tpm)

    genes = set(res.gene_table["Ensembl_Gene_ID"])
    assert "ENSG00000141510" in genes  # TP53 (from ENSG + Entrez rows)
    assert "ENSG00000146648" in genes  # EGFR (symbol)
    # ENSG + Entrez rows both hit TP53 → one summed row (100+20, 80+10).
    assert res.values.loc["ENSG00000141510", "a"] == pytest.approx(120.0)
    assert res.values.loc["ENSG00000141510", "b"] == pytest.approx(90.0)
    assert res.mapping_stats["n_resolved_rows"] == 3
    assert res.mapping_stats["n_unresolved_rows"] == 1
    unresolved = res.audit[res.audit["mapping_status"] != "resolved"]
    assert list(unresolved["source_row_id"]) == ["NOTAGENE"]
    assert bool(unresolved["high_expression_unresolved"].iloc[0])


def test_source_metadata_proxy_scale_is_warn_only():
    """A declared non-linear proxy scale resolves ``linear_tpm_comparable`` False
    so oncoref skips the RNA-seq fail gates (warn-only)."""
    md = osrc.source_metadata(source_scale_class="microarray_tpm_proxy",
                              linear_tpm_comparable=False)
    assert md["linear_tpm_comparable"] is False and md["tpm_proxy"] is True
    # Default RNA-seq TPM stays comparable.
    assert osrc.source_metadata(unit="TPM")["linear_tpm_comparable"] is True
    assert osrc.source_metadata(unit="raw_counts")["source_scale_class"] == "count_derived_tpm"


# ─── geo_matrix.build_source end-to-end ───────────────────────────────────


_MATRIX = (
    "gene_id\tSymbol\ts1\ts2\ts3\n"
    "ENSG00000141510.11\tTP53\t120\t110\t100\n"    # TP53
    "7157\tTP53_ENTREZ\t30\t20\t10\n"               # Entrez TP53 → sums with above
    "ENSG00000075624\tACTB\t500\t480\t520\n"        # ACTB
    "GAPDH\tGAPDH\t450\tbad\t470\n"                 # symbol row + one parse-fail cell
    "ENSG00000111640\tGAPDH2\t\t5\t5\n"             # GAPDH ensg + one blank (input-missing)
    "NOTAGENE\tJUNK\t9\t9\t9\n"                     # unresolved
)


def _make_source(**overrides):
    from pirlygenes.builders.geo_matrix import GeoMatrixSource

    kwargs = dict(
        cancer_code="LUAD",
        source_cohort="TEST_GEO_ONCOREF",
        source_project="GEO",
        citation="PMID:0000000",
        file_url="file://unused",
        file_name="matrix.tsv",
        unit="TPM",
        gene_id_col="gene_id",
        drop_cols=("Symbol",),
        sample_qc_mode="all",   # keep every sample (tiny fixture fails QC gates)
    )
    kwargs.update(overrides)
    return GeoMatrixSource(**kwargs)


def test_build_source_delegates_to_oncoref_end_to_end(tmp_path, monkeypatch):
    from pirlygenes import cohorts
    from pirlygenes.builders import geo_matrix

    cache_dir = tmp_path / "test-geo-oncoref"
    cache_dir.mkdir()
    (cache_dir / "matrix.tsv").write_text(_MATRIX, encoding="utf-8")
    # Redirect the per-sample parquet cache to our tmp cache dir.
    monkeypatch.setattr(cohorts.downloads, "source_cache_dir", lambda source_id: cache_dir)

    shard_dir = tmp_path / "cancer-reference-expression"
    counts = geo_matrix.build_source(
        _make_source(), cache_dir=cache_dir, summary_output=shard_dir,
    )
    assert counts == {"LUAD": 3}

    # Shard written with the LUAD rows.
    shard = pd.read_csv(shard_dir / "TEST_GEO_ONCOREF.csv.gz")
    assert set(shard["cancer_code"]) == {"LUAD"}
    genes = set(shard["Ensembl_Gene_ID"])
    assert {"ENSG00000141510", "ENSG00000075624", "ENSG00000111640"} <= genes
    assert "NOTAGENE" not in genes  # unresolved dropped
    # Entrez TP53 summed into the canonical TP53 row (one row, not two).
    assert (shard["Ensembl_Gene_ID"] == "ENSG00000141510").sum() == 1

    # Derived artifacts: mapping audit, parse diagnostics, per-sample QC.
    derived = cache_dir / "derived"
    audit = pd.read_csv(derived / "TEST_GEO_ONCOREF_mapping_audit.csv")
    assert (audit["source_row_id"].astype(str) == "7157").any()
    assert (audit["mapping_status"] == "resolved").sum() >= 4  # ENSGx3 + entrez + symbol
    parse = pd.read_csv(derived / "TEST_GEO_ONCOREF_parse_diagnostics.csv")
    # s2 had a non-numeric "bad" cell (parse-missing); s1 had a blank (input-missing).
    s2 = parse.loc[parse["value_col"] == "s2"].iloc[0]
    assert int(s2["n_parse_missing"]) >= 1
    qc = pd.read_csv(derived / "LUAD_sample_qc.csv")
    assert set(qc["sample_id"]) == {"s1", "s2", "s3"}
    assert {"sample_qc_status", "housekeeping_detection_floor_tpm"} <= set(qc.columns)

    # Per-sample parquet persisted under the code stem.
    assert (derived / "LUAD_per_sample_tpm.parquet").exists()


def test_build_source_qc_mode_pass_filters_and_removes_stale_parquet(tmp_path, monkeypatch):
    """With a fail-inducing QC mode and a tiny matrix, LUAD samples all fail →
    no summary rows, and any stale parquet is removed (no crash, clear error)."""
    from pirlygenes import cohorts
    from pirlygenes.builders import geo_matrix

    cache_dir = tmp_path / "test-geo-oncoref2"
    cache_dir.mkdir()
    (cache_dir / "matrix.tsv").write_text(_MATRIX, encoding="utf-8")
    monkeypatch.setattr(cohorts.downloads, "source_cache_dir", lambda source_id: cache_dir)
    # Pre-place a stale parquet so we can assert it gets removed.
    (cache_dir / "derived").mkdir()
    (cache_dir / "derived" / "LUAD_per_sample_tpm.parquet").write_bytes(b"stale")

    shard_dir = tmp_path / "cancer-reference-expression"
    with pytest.raises(RuntimeError, match="no cohort had samples"):
        geo_matrix.build_source(
            _make_source(sample_qc_mode="pass"),
            cache_dir=cache_dir, summary_output=shard_dir,
        )
    # QC manifest still written (covers every sample); stale parquet removed.
    assert (cache_dir / "derived" / "LUAD_sample_qc.csv").exists()
    assert not (cache_dir / "derived" / "LUAD_per_sample_tpm.parquet").exists()


def test_build_source_canonicalizes_cohort_code_stems(tmp_path, monkeypatch):
    """A sample_to_cancer_code rule that emits a pre-rename alias (MID_NET) must
    still land the QC manifest AND the per-sample parquet under the canonical
    code (NET_MIDGUT) — one stem for every artifact, so nothing is orphaned."""
    from pirlygenes import cohorts
    from pirlygenes.builders import geo_matrix

    cache_dir = tmp_path / "test-geo-alias"
    cache_dir.mkdir()
    (cache_dir / "matrix.tsv").write_text(_MATRIX, encoding="utf-8")
    monkeypatch.setattr(cohorts.downloads, "source_cache_dir", lambda source_id: cache_dir)

    shard_dir = tmp_path / "cancer-reference-expression"
    counts = geo_matrix.build_source(
        _make_source(sample_to_cancer_code=lambda s: "MID_NET"),  # alias for every sample
        cache_dir=cache_dir, summary_output=shard_dir,
    )
    assert counts == {"NET_MIDGUT": 3}  # counts keyed by canonical code, not the alias
    derived = cache_dir / "derived"
    assert (derived / "NET_MIDGUT_sample_qc.csv").exists()
    assert (derived / "NET_MIDGUT_per_sample_tpm.parquet").exists()
    assert not (derived / "MID_NET_sample_qc.csv").exists()  # alias stem never written
    shard = pd.read_csv(shard_dir / "TEST_GEO_ONCOREF.csv.gz")
    assert set(shard["cancer_code"]) == {"NET_MIDGUT"}


def _load_build_geo_matrix():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "build_geo_matrix.py"
    spec = importlib.util.spec_from_file_location("build_geo_matrix_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_geo_source_threads_qc_scale_knobs_from_yaml():
    """_build_geo_source must forward sample_qc_mode / source_scale_class /
    linear_tpm_comparable from the YAML entry — otherwise a proxy/microarray GEO
    source can't be marked non-linear and oncoref's warn-only path is unreachable
    (it would wrongly hit the RNA-seq hard-fail gates)."""
    mod = _load_build_geo_matrix()
    base = {"cancer_codes": ["BRCA"], "source_cohort": "X",
            "file_url": "u", "file_name": "f", "unit": "TPM"}

    proxy = mod._build_geo_source({**base,
        "source_scale_class": "microarray_tpm_proxy",
        "linear_tpm_comparable": False, "sample_qc_mode": "all"})
    assert proxy.source_scale_class == "microarray_tpm_proxy"
    assert proxy.linear_tpm_comparable is False
    assert proxy.sample_qc_mode == "all"

    # Defaults stay RNA-seq-linear / pass_or_warn for an ordinary TPM cohort.
    default = mod._build_geo_source(base)
    assert default.sample_qc_mode == "pass_or_warn"
    assert default.source_scale_class == ""
    assert default.linear_tpm_comparable is None
