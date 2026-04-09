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
    mds_calls = []
    tissue_calls = []
    safety_calls = []
    monkeypatch.setattr(cli_mod, "load_expression_data", lambda *a, **k: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(cli_mod, "plot_gene_expression", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_sample_vs_cancer", lambda *a, **k: scatter_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_therapy_target_tissues", lambda *a, **k: tissue_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_therapy_target_safety", lambda *a, **k: safety_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_cancer_type_genes", lambda *a, **k: cancer_gene_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_cancer_type_disjoint_genes", lambda *a, **k: cancer_gene_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_cancer_type_pca", lambda *a, **k: pca_calls.append(k))
    monkeypatch.setattr(cli_mod, "plot_cancer_type_mds", lambda *a, **k: mds_calls.append(k))
    monkeypatch.setattr(cli_mod, "therapy_target_gene_id_to_name", lambda t: {"ENSG_MOCK": t})
    monkeypatch.setattr(cli_mod, "pMHC_TCE_target_gene_id_to_name", lambda: {"ENSG_PMHC": "PMHC"})
    monkeypatch.setattr(cli_mod, "surface_TCE_target_gene_id_to_name", lambda: {"ENSG_SURF": "SURF"})
    mock_analysis = {
        "cancer_type": "PRAD", "cancer_name": "Prostate", "cancer_score": 0.9,
        "top_cancers": [("PRAD", 0.9)],
        "purity": {
            "overall_estimate": 0.1, "overall_lower": 0.05, "overall_upper": 0.15,
            "components": {"stromal": {"enrichment": 4.0}, "immune": {"enrichment": 2.0}},
        },
        "tissue_scores": [("prostate", 0.9, 20)],
        "mhc1": {"HLA-A": 100, "HLA-B": 200, "HLA-C": 150, "B2M": 3000},
        "mhc2": {},
    }
    monkeypatch.setattr(cli_mod, "analyze_sample", lambda *a, **k: mock_analysis)
    monkeypatch.setattr(cli_mod, "plot_sample_summary", lambda *a, **k: (None, mock_analysis))
    monkeypatch.setattr(cli_mod, "plot_tumor_purity", lambda *a, **k: (None, mock_analysis["purity"]))

    report_calls = []
    target_report_calls = []
    monkeypatch.setattr(cli_mod, "_generate_text_reports", lambda *a, **k: report_calls.append(True))
    monkeypatch.setattr(cli_mod, "_generate_target_report", lambda *a, **k: target_report_calls.append(True))
    monkeypatch.setattr(cli_mod, "_select_embedding_genes_bottleneck", lambda **k: (None, {
        "per_type": {}, "n_genes": 0, "n_types": 0, "method": "bottleneck", "tme_tissues": [],
    }))
    monkeypatch.setattr(cli_mod, "plot_purity_adjusted_targets", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "estimate_tumor_expression", lambda *a, **k: pd.DataFrame())

    cli_mod.analyze(
        "input.csv",
        output_image_prefix="out",
        aggregate_gene_expression=True,
        label_genes="FAP,CD276",
        output_dpi=200,
        therapy_target_top_k=12,
        therapy_target_tpm_threshold=18,
    )
    assert len(calls) == 4  # immune, tumor, antigens, treatments
    assert calls[0]["save_to_filename"] == "out-immune.png"
    assert calls[1]["save_to_filename"] == "out-tumor.png"
    assert calls[2]["save_to_filename"] == "out-antigens.png"
    assert calls[3]["save_to_filename"] == "out-treatments.png"
    assert calls[3]["gene_sets"]["Radio"] == {"ENSG_MOCK": "radioligand"}
    assert calls[3]["always_label_genes"] == {"FAP", "CD276"}
    assert len(scatter_calls) == 1
    assert scatter_calls[0]["save_to_filename"] == "out-vs-cancer.pdf"
    assert tissue_calls[0]["top_k"] == 12
    assert tissue_calls[0]["tpm_threshold"] == 18
    assert safety_calls[0]["top_k"] == 12
    assert safety_calls[0]["tpm_threshold"] == 18
    assert len(cancer_gene_calls) == 2  # genes + disjoint
    assert len(pca_calls) == 1  # tme
    assert len(mds_calls) == 1
    assert len(report_calls) == 1

    printed = []
    monkeypatch.setattr(cli_mod, "print_name_and_version", lambda: printed.append("v"))
    monkeypatch.setattr(cli_mod, "dispatch_commands", lambda cmds: printed.append(cmds))
    cli_mod.main()
    assert printed and printed[0] == "v"


def test_collect_ranked_therapy_targets_tracks_multicategory_and_approval(monkeypatch):
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG_A", "ENSG_B", "ENSG_C"],
            "gene_display_name": ["GENEA", "GENEB", "GENEC"],
            "TPM": [120.0, 90.0, 60.0],
        }
    )

    therapy_maps = {
        "ADC": {"ENSG_B": "GENEB"},
        "ADC-approved": {"ENSG_B": "GENEB"},
        "CAR-T": {"ENSG_A": "GENEA"},
        "CAR-T-approved": {},
        "TCR-T": {},
        "TCR-T-approved": {},
        "bispecific-antibodies": {"ENSG_A": "GENEA"},
        "bispecific-antibodies-approved": {},
        "radioligand": {"ENSG_C": "GENEC"},
    }

    monkeypatch.setattr(
        plot_mod,
        "therapy_target_gene_id_to_name",
        lambda therapy: therapy_maps.get(therapy, {}),
    )
    monkeypatch.setattr(
        plot_mod,
        "get_data",
        lambda name: pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG_C"],
                "Status_Bucket": ["FDA_approved"],
            }
        )
        if name == "radioligand-targets"
        else pd.DataFrame(),
    )

    out = plot_mod._collect_ranked_therapy_targets(df, top_k=1, tpm_threshold=10)

    assert [row["gene_id"] for row in out] == ["ENSG_A", "ENSG_B", "ENSG_C"]
    assert out[0]["therapies"] == ("CAR-T", "bispecific-antibodies")
    assert out[0]["has_approved"] is True
    assert out[0]["approved_therapies"] == ("CAR-T",)
    assert out[1]["therapies"] == ("ADC",)
    assert out[1]["approved_therapies"] == ("ADC",)
    assert out[2]["therapies"] == ("radioligand",)
    assert out[2]["approved_therapies"] == ("radioligand",)
