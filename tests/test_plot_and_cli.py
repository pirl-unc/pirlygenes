from types import SimpleNamespace

import pandas as pd
import pytest

import pirlygenes.plot as plot_mod
import pirlygenes.cli as cli_mod


def test_guess_gene_cols_and_pick_genes():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
            "gene_display_name": ["A", "B", "C"],
            "TPM": [1.0, 3.0, 2.0],
            "category": ["x", "x", "y"],
        }
    )
    gid_col, gname_col = plot_mod._guess_gene_cols(df)
    assert gid_col == "gene_id"
    assert gname_col == "gene_display_name"

    selected = plot_mod.pick_genes_to_annotate(df, num_per_category=1)
    assert selected == {"ENSG2", "ENSG3"}

    with pytest.raises(KeyError):
        plot_mod._guess_gene_cols(pd.DataFrame({"TPM": [1.0]}))


def test_resolve_always_label_gene_ids(monkeypatch):
    df = pd.DataFrame(
        {"gene_id": ["ENSG1", "ENSG2"], "gene_display_name": ["GENE1", "B7-H3"]}
    )
    monkeypatch.setattr(
        plot_mod,
        "find_canonical_gene_ids_and_names",
        lambda tokens: (["ENSG2"], ["CD276"]),
    )
    out = plot_mod.resolve_always_label_gene_ids(df, {"GENE1", "CD276"})
    assert out == {"ENSG1", "ENSG2"}


def test_plot_gene_expression_smoke(monkeypatch, tmp_path):
    prepared = pd.DataFrame(
        {
            "gene_id": ["ENSG1", "ENSG2"],
            "gene_display_name": ["GENE1", "GENE2"],
            "TPM": [0.05, 2.0],
            "category": ["A", "A"],
        }
    )
    monkeypatch.setattr(plot_mod, "prepare_gene_expr_df", lambda *a, **k: prepared.copy())
    monkeypatch.setattr(plot_mod, "adjust_text", lambda *a, **k: None)

    class FakeAx:
        def __init__(self):
            self.text_calls = []
            self.collections = []

        def text(self, *args, **kwargs):
            self.text_calls.append((args, kwargs))
            return SimpleNamespace()

        def scatter(self, *args, **kwargs):
            pass

        def axhline(self, *args, **kwargs):
            pass

        def annotate(self, *args, **kwargs):
            pass

    class FakeFigure:
        def __init__(self):
            self.saved = None

        def savefig(self, *args, **kwargs):
            self.saved = (args, kwargs)

    class FakeCat:
        def __init__(self):
            self.ax = FakeAx()
            self.figure = FakeFigure()

    fake_cat = FakeCat()
    monkeypatch.setattr(plot_mod.sns, "catplot", lambda **kwargs: fake_cat)

    out_path = tmp_path / "plot.png"
    result = plot_mod.plot_gene_expression(
        pd.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2"],
                "gene_display_name": ["GENE1", "GENE2"],
                "TPM": [0.05, 2.0],
            }
        ),
        gene_sets={"A": {"GENE1", "GENE2"}},
        save_to_filename=str(out_path),
        always_label_genes={"GENE1"},
        save_dpi=123,
    )
    assert result is fake_cat
    assert fake_cat.figure.saved is not None
    _, kwargs = fake_cat.figure.saved
    assert kwargs["dpi"] == 123


def test_cli_plot_expression_and_main(monkeypatch):
    calls = []
    scatter_calls = []
    cancer_gene_calls = []
    pca_calls = []
    monkeypatch.setattr(cli_mod, "load_expression_data", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(cli_mod, "plot_gene_expression", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_sample_vs_cancer", lambda *a, **k: scatter_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_cancer_type_genes", lambda *a, **k: cancer_gene_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_cancer_type_pca", lambda *a, **k: pca_calls.append(k))
    monkeypatch.setattr(cli_mod, "therapy_target_gene_id_to_name", lambda t: {"ENSG_MOCK": t})
    monkeypatch.setattr(cli_mod, "pMHC_TCE_target_gene_names", lambda: {"PMHC"})
    monkeypatch.setattr(cli_mod, "surface_TCE_target_gene_names", lambda: {"SURF"})

    cli_mod.plot_expression(
        "input.csv",
        output_image_prefix="out",
        aggregate_gene_expression=True,
        label_genes="FAP,CD276",
        output_dpi=200,
    )
    assert len(calls) == 2
    assert calls[0]["save_to_filename"] == "out-summary.png"
    assert calls[1]["save_to_filename"] == "out-treatments.png"
    assert calls[1]["gene_sets"]["Radio"] == {"ENSG_MOCK": "radioligand"}
    assert calls[1]["always_label_genes"] == {"FAP", "CD276"}
    assert len(scatter_calls) == 1
    assert scatter_calls[0]["save_to_filename"] == "out-vs-cancer.pdf"
    assert len(cancer_gene_calls) == 1
    assert len(pca_calls) == 1

    printed = []
    monkeypatch.setattr(cli_mod, "print_name_and_version", lambda: printed.append("v"))
    monkeypatch.setattr(cli_mod, "dispatch_commands", lambda cmds: printed.append(cmds))
    cli_mod.main()
    assert printed and printed[0] == "v"
