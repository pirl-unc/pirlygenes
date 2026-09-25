"""Covering-set panel design: burden weighting and greedy cover contracts."""
from __future__ import annotations

import pandas as pd
import pytest

from pirlygenes import coverage
from pirlygenes import gene_sets_cancer as gsc


def test_stat_choices_match_the_cli():
    """The CLI hard-codes the stat names to keep pandas out of `--help`; if the
    module grows a statistic, the flag has to grow with it."""
    from pirlygenes.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["plot", "covering-set", "--stat", "median"])
    assert args.stat == "median"
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "covering-set", "--stat", "nonesuch"])
    assert set(coverage.COVERING_SET_STATS) == {"median", "q3"}


def test_every_registry_code_resolves_to_a_burden_category():
    """A code with no burden category is a silent hole in the patient-weighted
    curve, so the gap has to be visible rather than discovered later.

    This pins the *mechanism*, not oncoref's numbers: whatever the registry
    holds, `burden_weights` must account for every code it is handed — either
    with a weight or by naming it in `unmapped`.
    """
    codes = list(gsc.cancer_type_registry()["code"].astype(str))
    weights, unmapped = coverage.burden_weights(codes)
    assert set(weights.index) == set(codes)
    assert set(unmapped) <= set(codes)
    # every code is either weighted or explicitly reported as unmapped
    weighted = {c for c in codes if weights[c] > 0}
    assert weighted | set(unmapped) | {c for c in codes if weights[c] == 0} == set(codes)
    assert not (weighted & set(unmapped))


def test_burden_weights_split_a_shared_category():
    """COAD and READ share the `colorectal` burden category. Handing each the
    full share would double-count colorectal patients."""
    both, _ = coverage.burden_weights(["COAD", "READ"])
    alone, _ = coverage.burden_weights(["COAD"])
    assert both["COAD"] == pytest.approx(both["READ"])
    assert both["COAD"] == pytest.approx(alone["COAD"] / 2)


def test_greedy_cover_orders_by_weighted_gain_and_reaches_the_ceiling():
    matrix = pd.DataFrame(
        # BIG covers two heavy types, NICHE one light type, DEAD nothing
        {"BIG": [100.0, 100.0, 0.0, 0.0],
         "NICHE": [0.0, 0.0, 100.0, 0.0],
         "DEAD": [0.0, 0.0, 0.0, 0.0]},
        index=["A", "B", "C", "D"],
    )
    weights = pd.Series({"A": 10.0, "B": 10.0, "C": 1.0, "D": 5.0})
    steps, coverable = coverage.greedy_covering_set(matrix, 30.0, weights)
    assert [s.gene for s in steps] == ["BIG", "NICHE"]
    assert steps[0].new_codes == ("A", "B")
    assert steps[0].cum_weight == pytest.approx(20.0)
    assert steps[-1].cum_codes == 3
    # D has weight but no gene over threshold: it is not coverable at all
    assert set(coverable) == {"A", "B", "C"}


def test_greedy_cover_still_picks_up_zero_weight_types():
    """An unmapped (zero-weight) cancer type must not be stranded once every
    weighted type is covered — it still counts on the cancer-type line."""
    matrix = pd.DataFrame(
        {"HEAVY": [100.0, 0.0], "ORPHAN": [0.0, 100.0]}, index=["A", "Z"])
    weights = pd.Series({"A": 10.0, "Z": 0.0})
    steps, coverable = coverage.greedy_covering_set(matrix, 30.0, weights)
    assert [s.gene for s in steps] == ["HEAVY", "ORPHAN"]
    assert steps[-1].cum_codes == 2
    assert set(coverable) == {"A", "Z"}


@pytest.mark.parametrize("metric,metric_label", [
    ("us_incidence_pct", "US annual incidence"),
    ("us_mortality_pct", "US annual mortality"),
    ("world_mortality_pct", "world annual mortality"),
])
def test_render_keeps_uncoverable_types_and_labels_selected_panel_and_metric(
    tmp_path, monkeypatch, metric, metric_label,
):
    from matplotlib.figure import Figure

    matrix = pd.DataFrame({"G1": [100, 0, 0, 0], "G2": [0, 100, 0, 0]},
                          index=["A", "B", "C", "D"])
    monkeypatch.setattr(coverage, "resolve_gene_set", lambda _: ("surfaceome", {"G1", "G2"}))
    monkeypatch.setattr(coverage, "covering_set_matrix", lambda *a, **k: matrix)
    seen = {}

    def weights(codes, *, metric):
        seen["metric"] = metric
        return pd.Series([4.0, 2.0, 1.0, 1.0], index=codes), []

    def capture(fig, *_args, **_kwargs):
        ax = fig.axes[0]
        seen["title"] = ax.get_title()
        seen["curves"] = [list(line.get_ydata()) for line in ax.lines[:2]]
        seen["legend"] = [t.get_text() for t in ax.get_legend().get_texts()]

    monkeypatch.setattr(coverage, "burden_weights", weights)
    monkeypatch.setattr(Figure, "savefig", capture)
    result = coverage.render_covering_set("surfaceome", metric=metric, out_dir=tmp_path)
    table = pd.read_csv(result["paths"]["covering_set_csv"])
    assert table["cum_pct_cancer_types"].tolist() == [25.0, 50.0]
    assert table["cum_pct_patients"].tolist() == [50.0, 75.0]
    assert seen["curves"] == [[50.0, 75.0], [25.0, 50.0]]
    assert result["n_cancer_types"] == 4 and result["n_coverable"] == 2
    assert seen["metric"] == result["metric"] == metric
    assert "surfaceome covering set" in seen["title"]
    assert "2 of 4 cancer types coverable" in seen["title"]
    assert metric_label in seen["title"]
    assert seen["legend"] == [f"{metric_label} burden", "cancer types"]


@pytest.mark.parametrize("stat,value_col", [("q3", "q3"), ("median", "expression")])
@pytest.mark.parametrize("preloaded", [False, True])
def test_matrix_preserves_every_collapsed_cta_group(monkeypatch, stat, value_col,
                                                   preloaded):
    """A cohort reached only by a grouped target must remain coverable."""
    from pirlygenes.expression import accessors
    from pirlygenes.expression.protein_groups import collapse_cdna_identical_loci_long

    ids = sorted(gsc.CTA_gene_ids())
    raw = pd.DataFrame({
        "Ensembl_Gene_ID": ids, "Symbol": ids,
        "cancer_code": "BRCA", "source_cohort": "test", "n_samples": 20,
        "expression": 0.0, "q3": 0.0,
    })
    collapsed = collapse_cdna_identical_loci_long(
        raw, group_keys=["cancer_code", "source_cohort"],
        sum_cols=["expression", "q3"])
    groups = set(collapsed.loc[collapsed["Ensembl_Gene_ID"].str.contains("/"),
                               "Ensembl_Gene_ID"])
    assert len(groups) == 16
    assert {"CTAG1A/B", "XAGE1A/B"} <= groups
    collapsed.loc[collapsed["Ensembl_Gene_ID"] == "CTAG1A/B", value_col] = 80.0

    def load(*, collapse_cdna_identical):
        assert collapse_cdna_identical is True
        assert not preloaded, "a supplied frame must not trigger another load"
        return collapsed

    monkeypatch.setattr(accessors, "cancer_reference_expression", load)
    matrix = coverage.covering_set_matrix(
        stat, df=collapsed if preloaded else None)
    assert set(matrix.columns) == set(collapsed["Symbol"])
    assert groups <= set(matrix.columns)
    assert matrix.loc["BRCA", "CTAG1A/B"] == 80.0
    steps, coverable = coverage.greedy_covering_set(
        matrix, 30.0, pd.Series({"BRCA": 10.0}))
    assert coverable == ["BRCA"]
    assert [step.gene for step in steps] == ["CTAG1A/B"]
    assert steps[-1].cum_weight == 10.0


def test_matrix_folds_non_cta_panels_and_keeps_single_loci(monkeypatch):
    """Fold any resolved panel, including versioned noncanonical member IDs."""
    panel = {"ENSG00000268651.1", "ENSG00000204382", "ENSG00000141510"}
    monkeypatch.setattr(coverage, "resolve_gene_set", lambda _: ("custom", panel))
    frame = pd.DataFrame({
        "Ensembl_Gene_ID": ["CTAG1A/B", "XAGE1A/B", "ENSG00000141510", "ENSG_OTHER"],
        "Symbol": ["CTAG1A/B", "XAGE1A/B", "TP53", "OTHER"],
        "cancer_code": "BRCA", "source_cohort": "test", "n_samples": 20,
        "expression": [10.0, 20.0, 30.0, 999.0],
        "q3": [40.0, 50.0, 60.0, 999.0],
    })
    matrix = coverage.covering_set_matrix(gene_set="custom", df=frame)
    assert matrix.loc["BRCA"].to_dict() == {
        "CTAG1A/B": 40.0, "XAGE1A/B": 50.0, "TP53": 60.0,
    }


@pytest.mark.parametrize("selected", [{"ENSG_G1"}, {"ENSG_ABSENT"}])
def test_matrix_keeps_eligible_types_without_panel_measurements(monkeypatch, selected):
    monkeypatch.setattr(coverage, "resolve_gene_set", lambda _: ("custom", selected))
    frame = pd.DataFrame({
        "Ensembl_Gene_ID": ["ENSG_G1", "ENSG_OTHER"],
        "Symbol": ["G1", "OTHER"], "cancer_code": ["BRCA", "COAD"],
        "source_cohort": "test", "n_samples": 20,
        "expression": 100.0, "q3": 100.0,
    })
    matrix = coverage.covering_set_matrix(gene_set="custom", df=frame)
    assert list(matrix.index) == ["BRCA", "COAD"]
    assert matrix.loc["COAD"].isna().all()
    restricted = coverage.covering_set_matrix(gene_set="custom", df=frame, codes=["COAD"])
    assert list(restricted.index) == ["COAD"]
