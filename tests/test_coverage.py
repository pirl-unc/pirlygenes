"""Tests for pirlygenes.coverage (cohort-level patient coverage of a gene set).

Uses a synthetic per-sample matrix written to a tmp cache — never real
patient/clinical data."""

import pandas as pd
import pytest

from pirlygenes import cohorts, coverage
from pirlygenes.cli import main as cli_main


# A 3-gene × 4-sample synthetic cohort with hand-chosen TPM so coverage is
# deterministic at threshold 25:
#   GENEA > 25 in s1, s2          (covers 2/4)
#   GENEB > 25 in s2, s3          (adds s3 -> 3/4)
#   GENEC > 25 in s1              (adds nothing new)
# greedy plateau = 3/4 = 75%.
_SYNTH = pd.DataFrame(
    {
        "Ensembl_Gene_ID": ["ENSG00000001", "ENSG00000002", "ENSG00000003"],
        "Symbol": ["GENEA", "GENEB", "GENEC"],
        "s1": [100.0, 1.0, 80.0],
        "s2": [60.0, 90.0, 2.0],
        "s3": [3.0, 40.0, 1.0],
        "s4": [0.0, 0.0, 0.0],
    }
)


@pytest.fixture
def synth_source(tmp_path, monkeypatch):
    """Register a synthetic cohort 'SYNTH' backed by a tmp-dir parquet."""
    derived = tmp_path / "derived"
    derived.mkdir()
    _SYNTH.to_parquet(derived / "SYNTH_per_sample_tpm.parquet", index=False)

    monkeypatch.setattr(coverage._cohorts.downloads, "source_cache_dir",
                        lambda source_id, **k: tmp_path)
    monkeypatch.setitem(
        cohorts._REGISTRY, "synth",
        {"SYNTH": cohorts.Cohort("SYNTH", "SYNTH", "synth")},
    )
    return tmp_path


def _symbol_csv(tmp_path):
    p = tmp_path / "panel.csv"
    p.write_text("Symbol\nGENEA\nGENEB\nGENEC\n")
    return p


# --- resolve_gene_set ------------------------------------------------------

def test_resolve_named_panels():
    label, ensgs, symbols = coverage.resolve_gene_set("cta")
    assert label == "CTA" and len(ensgs) > 50 and not symbols

    label, ensgs, symbols = coverage.resolve_gene_set("lineage:PRAD")
    assert label == "lineage:PRAD"
    assert ensgs or symbols  # lineage panel has ENSG and/or symbols


def test_resolve_csv_path(tmp_path):
    label, ensgs, symbols = coverage.resolve_gene_set(str(_symbol_csv(tmp_path)))
    assert symbols == {"GENEA", "GENEB", "GENEC"}


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        coverage.resolve_gene_set("definitely-not-a-panel")


# --- matrix + greedy + counts ----------------------------------------------

def test_cohort_matrix_filters_to_panel(synth_source):
    cohort = cohorts.cohorts_for_source("synth")["SYNTH"]
    mat = coverage.cohort_matrix(cohort, symbols={"GENEA", "GENEB"})
    assert set(mat.index) == {"GENEA", "GENEB"}
    assert mat.shape == (2, 4)


def test_greedy_coverage_plateau(synth_source):
    cohort = cohorts.cohorts_for_source("synth")["SYNTH"]
    mat = coverage.cohort_matrix(cohort, symbols={"GENEA", "GENEB", "GENEC"})
    order, cum, n = coverage.greedy_coverage(mat, threshold=25)
    assert n == 4
    assert cum[-1] == pytest.approx(0.75)  # 3 of 4 patients
    assert mat.index[order[0]] == "GENEA"  # most-covering gene first


def test_patient_coverage_counts(synth_source, tmp_path):
    df = coverage.patient_coverage(str(_symbol_csv(tmp_path)), source_id="synth",
                                   thresholds=(25,))
    a = df[df.Symbol == "GENEA"].iloc[0]
    assert a.n_samples == 4 and a.n_gt25 == 2 and a.pct_gt25 == 50.0
    # GENEC is >25 in one sample, so it is retained (any_hit), not dropped.
    assert "GENEC" in set(df.Symbol)


# --- CLI dispatch ----------------------------------------------------------

def test_cli_patient_coverage(synth_source, tmp_path):
    out = tmp_path / "out"
    rc = cli_main([
        "plot", "patient-coverage", "--gene-set", str(_symbol_csv(tmp_path)),
        "--source", "synth", "--threshold", "25", "--out", str(out),
    ])
    assert rc == 0
    assert (out / "panel_csv_patient_counts.csv").exists()
    assert (out / "panel_csv_stacked_coverage_t25.png").exists()


def test_cli_bad_gene_set_returns_2(tmp_path):
    rc = cli_main([
        "plot", "patient-coverage", "--gene-set", "nope", "--out", str(tmp_path),
    ])
    assert rc == 2


def test_cli_plot_no_action_returns_2():
    assert cli_main(["plot"]) == 2
