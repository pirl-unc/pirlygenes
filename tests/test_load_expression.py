import warnings

import pandas as pd
import pytest

import pirlygenes.gene_ids as gene_ids_mod
import pirlygenes.load_expression as le


def test_get_canonical_gene_name_from_gene_ids_string(monkeypatch):
    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": None}.get(gid),
    )
    assert le.get_canonical_gene_name_from_gene_ids_string("ENSG1;ENSG2") == "A"
    assert le.get_canonical_gene_name_from_gene_ids_string(float("nan")) == ""


def test_load_expression_data_with_existing_ids(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "Gene": ["B7-H3", "TP53"],
            "Ensembl Gene ID": ["ENSG00000103855", "ENSG00000141510"],
            "TPM": [1.0, 2.0],
        }
    ).to_csv(p, index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG00000103855": "CD276", "ENSG00000141510": "TP53"}[gid],
    )

    out = le.load_expression_data(str(p), verbose=False, progress=False)
    assert "canonical_gene_name" in out.columns
    assert "gene_display_name" in out.columns
    assert list(out["canonical_gene_name"]) == ["CD276", "TP53"]
    assert list(out["gene_display_name"]) == ["B7-H3", "p53"]


def test_load_expression_data_without_ensembl_ids(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame({"Gene Symbol": ["A", "B", "C", "D"], "TPM": [1, 2, 3, 4]}).to_csv(
        p, index=False
    )

    monkeypatch.setattr(
        le,
        "find_canonical_gene_ids_and_names",
        lambda genes: (
            ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
            ["A1", ["B1", "B2"], None, 123],
        ),
    )
    monkeypatch.setattr(le, "find_gene_name_from_ensembl_gene_id", lambda gid: gid)

    out = le.load_expression_data(str(p), verbose=False, progress=False)
    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    assert list(out["canonical_gene_name"]) == ["A1", "B1;B2", "", "?"]


def test_load_expression_aggregate_and_save(tmp_path, monkeypatch):
    p = tmp_path / "tx.csv"
    out_csv = tmp_path / "rolled.csv"
    pd.DataFrame({"transcript": ["tx1"], "tpm": [1.0]}).to_csv(p, index=False)

    monkeypatch.setattr(
        le,
        "tx2gene",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "gene": ["GENE1"],
                "TPM": [1.0],
                "gene_id": ["ENSG1"],
                "ensembl_release": [112],
            }
        ),
    )
    monkeypatch.setattr(le, "find_gene_name_from_ensembl_gene_id", lambda gid: "GENE1")

    out = le.load_expression_data(
        str(p),
        aggregate_gene_expression=True,
        aggregated_output_path=str(out_csv),
        verbose=False,
        progress=False,
    )
    assert out_csv.exists()
    assert list(out["ensembl_gene_id"]) == ["ENSG1"]


def test_load_expression_aggregate_raises_on_large_unresolved_fraction(tmp_path, monkeypatch):
    p = tmp_path / "tx.csv"
    pd.DataFrame({"transcript": ["tx1"], "tpm": [1.0]}).to_csv(p, index=False)

    def _fake_tx2gene(*_args, **_kwargs):
        out = pd.DataFrame(
            {
                "gene": ["GENE1"],
                "TPM": [1.0],
                "gene_id": ["ENSG1"],
                "ensembl_release": [112],
            }
        )
        out.attrs["transcript_aggregation_stats"] = {
            "known_tpm": 910000.0,
            "unknown_tpm": 90000.0,
            "unknown_fraction": 0.09,
            "unresolved_unique_count": 123,
            "unresolved_high_tpm": [{"tx": "ENSTOLD1", "TPM": 1200.0}],
        }
        return out

    monkeypatch.setattr(le, "tx2gene", _fake_tx2gene)

    with pytest.raises(ValueError, match="silently treat those genes as 0 TPM"):
        le.load_expression_data(
            str(p),
            aggregate_gene_expression=True,
            verbose=False,
            progress=False,
        )


def test_load_expression_with_gene_sidecar_skips_aggregation(tmp_path, monkeypatch):
    p = tmp_path / "sample.quant.sf.csv"
    sidecar = tmp_path / "Gene.csv"
    pd.DataFrame({"TPM": [1.0, 2.0]}).to_csv(p, index=False)
    pd.DataFrame({"TPM": ["A", "B"]}).to_csv(sidecar, index=False)

    monkeypatch.setattr(le, "tx2gene", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not aggregate")))
    monkeypatch.setattr(
        le,
        "find_canonical_gene_ids_and_names",
        lambda genes: (["ENSG1", "ENSG2"], ["A", "B"]),
    )
    monkeypatch.setattr(le, "find_gene_name_from_ensembl_gene_id", lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid])

    out = le.load_expression_data(
        str(p),
        aggregate_gene_expression=True,
        verbose=False,
        progress=False,
    )
    assert list(out["gene"]) == ["A", "B"]
    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2"]


def test_load_expression_gene_id_only_with_tpm_alias(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "ensembl_gene": ["ENSG1", "ENSG2"],
            "gene_tpm_cognizant_corrector": [1.5, 2.5],
        }
    ).to_csv(p, index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid],
    )

    out = le.load_expression_data(str(p), verbose=False, progress=False)
    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2"]
    assert list(out["canonical_gene_name"]) == ["A", "B"]
    assert list(out["gene"]) == ["A", "B"]
    assert list(out["TPM"]) == [1.5, 2.5]


def test_load_expression_accepts_kallisto_gene_abundance_column(tmp_path, monkeypatch):
    p = tmp_path / "gene_abundance.tsv"
    pd.DataFrame(
        {
            "gene_name": ["A", "B"],
            "gene": ["ENSG1", "ENSG2"],
            "abundance": [4.0, 5.0],
            "counts": [40, 50],
            "length": [1000, 1200],
        }
    ).to_csv(p, sep="\t", index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid],
    )

    out = le.load_expression_data(str(p), verbose=False, progress=False)

    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2"]
    assert list(out["TPM"]) == [4.0, 5.0]


def test_load_expression_prebuilds_indexes_before_canonical_name_progress(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "Ensembl Gene ID": ["ENSG1", "ENSG2"],
            "TPM": [1.0, 2.0],
        }
    ).to_csv(p, index=False)

    events = []

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid],
    )
    monkeypatch.setattr(
        gene_ids_mod,
        "_build_indexes",
        lambda: events.append("build"),
    )
    monkeypatch.setattr(
        le,
        "tqdm",
        lambda iterator, **_kwargs: events.append("tqdm") or iterator,
    )

    out = le.load_expression_data(str(p), verbose=False, progress=True)

    assert list(out["canonical_gene_name"]) == ["A", "B"]
    assert events[:2] == ["build", "tqdm"]


def test_load_expression_select_sample_rows(tmp_path, monkeypatch):
    p = tmp_path / "expr.csv"
    pd.DataFrame(
        {
            "analysis_id": ["S1", "S1", "S2"],
            "ensembl_gene": ["ENSG1", "ENSG2", "ENSG1"],
            "gene_tpm_cognizant_corrector": [5.0, 6.0, 99.0],
        }
    ).to_csv(p, index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid],
    )

    out = le.load_expression_data(
        str(p),
        sample_id_col="analysis_id",
        sample_id_value="S1",
        verbose=False,
        progress=False,
    )
    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2"]
    assert list(out["TPM"]) == [5.0, 6.0]


def test_load_expression_selects_wide_sample_column(tmp_path, monkeypatch):
    p = tmp_path / "wide.tsv"
    pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_name": ["A", "B"],
            "sample_a": [7.0, 8.0],
            "sample_b": [99.0, 100.0],
        }
    ).to_csv(p, sep="\t", index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid],
    )

    out = le.load_expression_data(
        str(p),
        sample_id_value="sample_a",
        verbose=False,
        progress=False,
    )

    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2"]
    assert list(out["gene"]) == ["A", "B"]
    assert list(out["TPM"]) == [7.0, 8.0]


def test_load_expression_wide_sample_preserves_explicit_gene_columns(tmp_path, monkeypatch):
    p = tmp_path / "wide_custom_gene_cols.tsv"
    pd.DataFrame(
        {
            "Stable ID": ["ENSG1", "ENSG2"],
            "Hugo_Symbol": ["A", "B"],
            "sample_a": [7.0, 8.0],
            "sample_b": [99.0, 100.0],
        }
    ).to_csv(p, sep="\t", index=False)

    monkeypatch.setattr(
        le,
        "find_gene_name_from_ensembl_gene_id",
        lambda gid: {"ENSG1": "A", "ENSG2": "B"}[gid],
    )

    out = le.load_expression_data(
        str(p),
        sample_id_value="sample_a",
        gene_name_col="Hugo_Symbol",
        gene_id_col="Stable ID",
        verbose=False,
        progress=False,
    )

    assert list(out["ensembl_gene_id"]) == ["ENSG1", "ENSG2"]
    assert list(out["gene"]) == ["A", "B"]
    assert list(out["TPM"]) == [7.0, 8.0]


def test_load_expression_error_paths(tmp_path):
    p = tmp_path / "expr.csv"
    pd.DataFrame({"TPM": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        le.load_expression_data(str(p), verbose=False, progress=False)

    with pytest.raises(ValueError):
        le.load_expression_data(str(tmp_path / "expr.bad"), verbose=False, progress=False)

    p2 = tmp_path / "expr2.csv"
    pd.DataFrame(
        {
            "analysis_id": ["S1"],
            "ensembl_gene": ["ENSG1"],
            "gene_tpm_cognizant_corrector": [1.0],
        }
    ).to_csv(p2, index=False)
    with pytest.raises(ValueError):
        le.load_expression_data(
            str(p2),
            sample_id_col="analysis_id",
            sample_id_value="missing",
            verbose=False,
            progress=False,
        )


# ── FPKM → TPM conversion ───────────────────────────────────────────────

def test_detect_and_convert_to_tpm_converts_fpkm_column():
    """FPKM column gets rescaled so values sum to 1e6, and a warning fires."""
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3"],
        "FPKM": [100.0, 200.0, 300.0],  # total = 600
    })
    with pytest.warns(UserWarning, match="converted to TPM"):
        out = le._detect_and_convert_to_tpm(df, verbose=False)
    assert "TPM" in out.columns
    assert "FPKM" not in out.columns
    assert abs(out["TPM"].sum() - 1e6) < 1.0
    # Within-sample ratios preserved
    assert out.loc[0, "TPM"] == pytest.approx(1e6 * 100 / 600)


def test_detect_and_convert_to_tpm_leaves_tpm_column_alone():
    """When a TPM column is already present, no conversion happens."""
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG1", "ENSG2"],
        "TPM": [50.0, 150.0],
    })
    # No warning expected
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = le._detect_and_convert_to_tpm(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


def test_detect_and_convert_to_tpm_noop_when_both_fpkm_and_tpm_present():
    """If both columns exist, TPM is already there; leave FPKM column untouched."""
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG1", "ENSG2"],
        "FPKM": [100.0, 200.0],
        "TPM": [60.0, 140.0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = le._detect_and_convert_to_tpm(df, verbose=False)
    assert list(out["FPKM"]) == [100.0, 200.0]
    assert list(out["TPM"]) == [60.0, 140.0]


def test_detect_and_convert_to_tpm_noop_when_no_fpkm_column():
    """No FPKM column → no-op, no warning."""
    df = pd.DataFrame({"ensembl_gene_id": ["ENSG1"], "other": [1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = le._detect_and_convert_to_tpm(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize("derived_col", [
    "log2_fpkm",
    "log_fpkm",
    "FPKM_zscore",
    "fpkm_adjusted",
    "FPKM_rank",
    "FPKM_COAD",  # TCGA reference column
    "FPKM_BRCA",
    "fpkm_per_million",
])
def test_detect_and_convert_to_tpm_ignores_derived_fpkm_columns(derived_col):
    """Derived or per-cohort FPKM columns must NOT trigger conversion."""
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG1", "ENSG2"],
        derived_col: [2.5, 3.0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = le._detect_and_convert_to_tpm(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize("raw_col", ["FPKM", "fpkm", "gene_fpkm", "gene FPKM", "rna_fpkm"])
def test_detect_and_convert_to_tpm_recognises_raw_fpkm_variants(raw_col):
    """Raw-FPKM variants (exact match) should all trigger conversion."""
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG1", "ENSG2"],
        raw_col: [100.0, 200.0],
    })
    with pytest.warns(UserWarning, match="converted to TPM"):
        out = le._detect_and_convert_to_tpm(df, verbose=False)
    assert "TPM" in out.columns
    assert raw_col not in out.columns
    assert abs(out["TPM"].sum() - 1e6) < 1.0


# ── Alt-haplotype ID aliasing + TPM summing ─────────────────────────────

def test_apply_id_aliases_sums_alt_haplotype_tpm(monkeypatch):
    """Alt-haplotype + primary rows for the same gene collapse to a single
    canonical row with summed TPM."""
    monkeypatch.setattr(
        le,
        "_load_ensembl_id_aliases",
        lambda: {"ENSG00000235657": "ENSG00000206503"},  # HLA-A alt → primary
    )
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000206503", "ENSG00000235657"],
        "gene": ["HLA-A", ""],
        "TPM": [42.0, 8.0],
    })
    out = le._apply_id_aliases_and_sum(df, verbose=False)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["ensembl_gene_id"] == "ENSG00000206503"
    assert row["TPM"] == 50.0  # 42 + 8
    # Non-empty symbol is preferred over the empty alt-row symbol
    assert row["gene"] == "HLA-A"


def test_apply_id_aliases_prefers_non_empty_symbol_regardless_of_order(monkeypatch):
    """Even if the empty-symbol row comes first, the populated value wins."""
    monkeypatch.setattr(
        le,
        "_load_ensembl_id_aliases",
        lambda: {"ENSG00000235657": "ENSG00000206503"},
    )
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000235657", "ENSG00000206503"],
        "gene": ["", "HLA-A"],
        "TPM": [8.0, 42.0],
    })
    out = le._apply_id_aliases_and_sum(df, verbose=False)
    assert len(out) == 1
    assert out.iloc[0]["gene"] == "HLA-A"


def test_apply_id_aliases_noop_when_no_alt_haplotype_ids(monkeypatch):
    """Clean data with no alt-haplotype IDs is unchanged."""
    monkeypatch.setattr(
        le,
        "_load_ensembl_id_aliases",
        lambda: {"ENSG00000235657": "ENSG00000206503"},
    )
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000000003", "ENSG00000000419"],
        "gene": ["TSPAN6", "DPM1"],
        "TPM": [10.0, 20.0],
    })
    out = le._apply_id_aliases_and_sum(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


def test_apply_id_aliases_noop_when_no_ensembl_gene_id_column():
    """Without an ensembl_gene_id column, function is a pass-through."""
    df = pd.DataFrame({"gene": ["X"], "TPM": [1.0]})
    out = le._apply_id_aliases_and_sum(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


def test_apply_id_aliases_noop_when_aliases_file_missing(monkeypatch):
    """If the aliases CSV isn't available, function is a pass-through."""
    monkeypatch.setattr(le, "_load_ensembl_id_aliases", lambda: {})
    df = pd.DataFrame({
        "ensembl_gene_id": ["ENSG00000235657"],
        "TPM": [5.0],
    })
    out = le._apply_id_aliases_and_sum(df, verbose=False)
    pd.testing.assert_frame_equal(out, df)


def test_load_ensembl_id_aliases_resolves_chains(monkeypatch, tmp_path):
    """A→B→C in the bundled data should be flattened to A→C, B→C."""
    chained = pd.DataFrame({
        "alt_haplotype_id": ["ENSG_A", "ENSG_B"],
        "primary_contig_id": ["ENSG_B", "ENSG_C"],
        "symbol": ["", ""],
        "source": ["test", "test"],
    })
    monkeypatch.setattr(le, "get_data", lambda name: chained, raising=False)
    import pirlygenes.load_dataset as ld
    monkeypatch.setattr(ld, "get_data", lambda name: chained)
    out = le._load_ensembl_id_aliases()
    assert out["ENSG_A"] == "ENSG_C"
    assert out["ENSG_B"] == "ENSG_C"


def test_load_ensembl_id_aliases_detects_cycles(monkeypatch):
    """A→B→A in the data should raise ValueError."""
    cyclic = pd.DataFrame({
        "alt_haplotype_id": ["ENSG_A", "ENSG_B"],
        "primary_contig_id": ["ENSG_B", "ENSG_A"],
        "symbol": ["", ""],
        "source": ["test", "test"],
    })
    import pirlygenes.load_dataset as ld
    monkeypatch.setattr(ld, "get_data", lambda name: cyclic)
    with pytest.raises(ValueError, match="Cycle"):
        le._load_ensembl_id_aliases()


# ── Ensembl release version check ───────────────────────────────────────

def _reset_release_check():
    """Reset the once-per-process flag so tests can exercise the warning."""
    le._ensembl_release_check_done = False


def test_ensembl_release_check_warns_when_no_releases_installed(monkeypatch):
    _reset_release_check()
    from pyensembl import shell
    monkeypatch.setattr(shell, "collect_all_installed_ensembl_releases", lambda: [])
    with pytest.warns(UserWarning, match="No human Ensembl releases"):
        le._check_installed_ensembl_releases()


def test_ensembl_release_check_warns_when_only_old_releases(monkeypatch):
    _reset_release_check()
    from types import SimpleNamespace
    from pyensembl import shell
    old_release = SimpleNamespace(
        species=SimpleNamespace(latin_name="homo_sapiens"),
        release=93,
    )
    monkeypatch.setattr(
        shell,
        "collect_all_installed_ensembl_releases",
        lambda: [old_release],
    )
    with pytest.warns(UserWarning, match="older than the recommended minimum"):
        le._check_installed_ensembl_releases()


def test_ensembl_release_check_silent_when_recent_release_installed(monkeypatch):
    _reset_release_check()
    from types import SimpleNamespace
    from pyensembl import shell
    recent = SimpleNamespace(
        species=SimpleNamespace(latin_name="homo_sapiens"),
        release=112,
    )
    monkeypatch.setattr(
        shell,
        "collect_all_installed_ensembl_releases",
        lambda: [recent],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        le._check_installed_ensembl_releases()


def test_ensembl_release_check_fires_only_once_per_process(monkeypatch):
    _reset_release_check()
    from pyensembl import shell
    monkeypatch.setattr(shell, "collect_all_installed_ensembl_releases", lambda: [])
    # First call warns
    with pytest.warns(UserWarning):
        le._check_installed_ensembl_releases()
    # Second call is silent (the flag persists)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        le._check_installed_ensembl_releases()
