"""Tests for pirlygenes.coverage (cohort-level patient coverage of a gene set).

Uses a synthetic per-sample matrix written to a tmp cache — never real
patient/clinical data.

The synthetic cohort uses **real** Ensembl gene ids / symbols (TP53, EGFR,
MYC) so the symbol→ENSG resolution path is exercised end-to-end: a gene-set
given as symbols is resolved to ENSGs, and the cohort matrix is matched on
ENSG only (symbols are never a join key)."""

import pandas as pd
import pytest

from pirlygenes import cohorts, coverage

# Real ENSGs so a symbol-only panel resolves through pyensembl and then matches
# the matrix on ENSG. TPM hand-chosen so coverage is deterministic at 25:
#   TP53 > 25 in s1, s2     (covers 2/4)
#   EGFR > 25 in s2, s3     (adds s3 -> 3/4)
#   MYC  > 25 in s1         (adds nothing new)
# greedy plateau = 3/4 = 75%.
_TP53 = "ENSG00000141510"
_EGFR = "ENSG00000146648"
_MYC = "ENSG00000136997"
_SYNTH = pd.DataFrame(
    {
        "Ensembl_Gene_ID": [_TP53, _EGFR, _MYC],
        "Symbol": ["TP53", "EGFR", "MYC"],
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
    p.write_text("Symbol\nTP53\nEGFR\nMYC\n")
    return p


def _ensg_csv(tmp_path):
    p = tmp_path / "panel_ensg.csv"
    p.write_text(f"Ensembl_Gene_ID\n{_TP53}\n{_EGFR}\n{_MYC}\n")
    return p


# --- resolve_gene_set ------------------------------------------------------

def test_resolve_named_panels():
    label, ensgs = coverage.resolve_gene_set("cta")
    assert label == "CTA" and len(ensgs) > 50
    assert all(str(e).startswith("ENSG") for e in ensgs)

    label, ensgs = coverage.resolve_gene_set("lineage:PRAD")
    assert label == "lineage:PRAD"
    assert ensgs and all(str(e).startswith("ENSG") for e in ensgs)


def test_resolve_csv_path_ensg(tmp_path):
    """An ENSG-column CSV resolves to exactly those (unversioned) ids."""
    label, ensgs = coverage.resolve_gene_set(str(_ensg_csv(tmp_path)))
    assert ensgs == {_TP53, _EGFR, _MYC}


def test_resolve_csv_path_symbols(tmp_path):
    """A symbol-column CSV resolves each symbol to its ENSG (lookup only)."""
    label, ensgs = coverage.resolve_gene_set(str(_symbol_csv(tmp_path)))
    assert ensgs == {_TP53, _EGFR, _MYC}


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        coverage.resolve_gene_set("definitely-not-a-panel")


# --- matrix + greedy + counts ----------------------------------------------

def test_cohort_matrix_filters_to_panel(synth_source):
    cohort = cohorts.cohorts_for_source("synth")["SYNTH"]
    mat = coverage.cohort_matrix(cohort, ensgs={_TP53, _EGFR})
    assert set(mat.index) == {_TP53, _EGFR}  # ENSG-indexed
    assert mat.shape == (2, 4)
    # symbol display map carried in attrs, never used as a join key
    assert mat.attrs["symbols"][_TP53] == "TP53"


def test_greedy_coverage_plateau(synth_source):
    cohort = cohorts.cohorts_for_source("synth")["SYNTH"]
    mat = coverage.cohort_matrix(cohort, ensgs={_TP53, _EGFR, _MYC})
    order, cum, n = coverage.greedy_coverage(mat, threshold=25)
    assert n == 4
    assert cum[-1] == pytest.approx(0.75)  # 3 of 4 patients
    assert mat.index[order[0]] == _TP53  # most-covering gene first


def test_patient_coverage_counts(synth_source, tmp_path):
    df = coverage.patient_coverage(str(_symbol_csv(tmp_path)), source_id="synth",
                                   thresholds=(25,))
    a = df[df.Ensembl_Gene_ID == _TP53].iloc[0]
    assert a.n_samples == 4 and a.n_gt25 == 2 and a.pct_gt25 == 50.0
    assert a.Symbol == "TP53"  # symbol carried for display
    # MYC is >25 in one sample, so it is retained (any_hit), not dropped.
    assert _MYC in set(df.Ensembl_Gene_ID)


# --- CLI dispatch ----------------------------------------------------------

def test_cli_patient_coverage(synth_source, tmp_path):
    from pirlygenes.cli import main as cli_main
    out = tmp_path / "out"
    rc = cli_main([
        "plot", "patient-coverage", "--gene-set", str(_symbol_csv(tmp_path)),
        "--source", "synth", "--threshold", "25", "--out", str(out),
    ])
    assert rc == 0
    assert (out / "panel_csv_patient_counts.csv").exists()
    assert (out / "panel_csv_stacked_coverage_t25.png").exists()


def test_cli_bad_gene_set_returns_2(tmp_path):
    from pirlygenes.cli import main as cli_main
    rc = cli_main([
        "plot", "patient-coverage", "--gene-set", "nope", "--out", str(tmp_path),
    ])
    assert rc == 2


def test_cli_plot_no_action_returns_2():
    from pirlygenes.cli import main as cli_main
    assert cli_main(["plot"]) == 2
