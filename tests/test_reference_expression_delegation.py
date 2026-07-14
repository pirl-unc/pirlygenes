"""Contract tests for the oncoref cancer-reference-expression cutover (#557)."""

import numpy as np
import pandas as pd

import oncoref

from pirlygenes.expression import accessors


def test_delegated_parity_across_reference_classes():
    """The adapter must not change source rows or values across key cohorts."""
    cases = [
        ("LUAD", ["TP53"]),          # common TCGA project
        ("CLL", ["MS4A1"]),         # non-TCGA heme reference
        ("MTC", ["CALCA"]),         # microarray TPM proxy
        ("BRCA_Basal", ["KRT5"]),   # molecular subtype
        ("SARC", ["TP53"]),         # computed source union
    ]
    compare_columns = [
        "Ensembl_Gene_ID",
        "Symbol",
        "cancer_code",
        "source_cohort",
        "source_project",
        "source_version",
        "n_samples",
        "n_detected",
        "processing_pipeline",
        "notes",
        "normalization",
        "expression",
        "q1",
        "q3",
    ]
    sort_columns = ["cancer_code", "source_cohort", "Ensembl_Gene_ID"]

    for code, genes in cases:
        actual = accessors.cancer_reference_expression(
            cancer_types=code,
            genes=genes,
            normalize="tpm_clean",
        )
        delegated = oncoref.cancer_reference_expression(
            cancer_types=code,
            genes=genes,
            normalize="tpm_clean",
            format="long",
            include_provenance=True,
            on_missing="empty",
            auto_fetch=False,
            sample_qc="all",
            reference_source="summary_rows_all",
            gene_id_style="pirlygenes",
            gene_universe="pirlygenes",
        ).copy()
        delegated, _ = accessors._normalize_reference_source_cohort_labels(
            delegated
        )
        delegated["normalization"] = "TPM_clean"

        actual_cmp = actual[compare_columns].sort_values(sort_columns).reset_index(drop=True)
        delegated_cmp = (
            delegated[compare_columns].sort_values(sort_columns).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(actual_cmp, delegated_cmp, check_dtype=False)
        assert actual.attrs["delegated_to"] == "oncoref.cancer_reference_expression"
        assert actual.attrs["reference_source"] == "summary_rows_all"


def test_adapter_expands_legacy_aliases_and_derives_raw_log_without_fallback(
    monkeypatch,
):
    calls = []

    def fake_oncoref(**kwargs):
        calls.append(kwargs)
        out = pd.DataFrame(
            {
                "Ensembl_Gene_ID": ["ENSG00000163534"],
                "Symbol": ["FCRL5"],
                "Proteoform_ID": ["ENSG00000163534"],
                "Member_Ensembl_Gene_IDs": ["ENSG00000163534"],
                "cancer_code": ["MM"],
                "normalization": ["tpm_raw"],
                "source_cohort": ["MMRF_COMMPASS_IA21"],
                "source_project": ["MMRF CoMMpass"],
                "source_version": ["test"],
                "n_samples": [10],
                "n_detected": [8.0],
                "processing_pipeline": ["test_tpm"],
                "notes": ["test"],
                "expression": [9.0],
                "q1": [3.0],
                "q3": [15.0],
            }
        )
        out.attrs.update(
            {
                "reference_source": "summary_rows_all",
                "availability": [
                    {
                        "cancer_code": "MM",
                        "normalization": "tpm_raw",
                        "available": True,
                    }
                ],
                "missing_requests": [],
            }
        )
        return out

    monkeypatch.setattr(oncoref, "cancer_reference_expression", fake_oncoref)
    monkeypatch.setattr(
        accessors,
        "_load_cancer_reference_expression",
        lambda: (_ for _ in ()).throw(AssertionError("local fallback used")),
    )

    out = accessors.cancer_reference_expression(
        cancer_types="MM",
        genes=["FCRH5"],
        normalize="tpm_log1p",
    )

    assert calls[0]["normalize"] == "tpm"
    assert calls[0]["sample_qc"] == "all"
    assert calls[0]["reference_source"] == "summary_rows_all"
    assert calls[0]["gene_id_style"] == "pirlygenes"
    assert calls[0]["gene_universe"] == "pirlygenes"
    assert {"FCRH5", "FCRL5"} <= set(calls[0]["genes"])
    assert out["normalization"].tolist() == ["TPM_log1p"]
    assert out["expression"].iloc[0] == np.log1p(9.0)
    assert out.attrs["missing_requests"] == []
    assert out.attrs["compatibility_transforms"] == [
        "legacy gene aliases expanded before delegated filtering",
        "tpm_log1p derived with numpy.log1p from delegated raw TPM",
    ]


def test_delegated_filter_preserves_legacy_ensembl_queries():
    out = accessors.cancer_reference_expression(
        cancer_types="CLL",
        genes=["ENSG00000148362.9", "ENSG00000280987"],
    )
    assert dict(zip(out["Symbol"], out["Ensembl_Gene_ID"])) == {
        "PAXX": "ENSG00000148362",
        "MATR3": "ENSG00000015479",
    }
    assert out["expression"].notna().all()


def test_sarc_histology_source_label_and_filter_are_canonicalized():
    canonical = "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY"
    out = accessors.cancer_reference_expression(
        cancer_types="SARC",
        genes=["TP53"],
        source_cohort=(cohort for cohort in [canonical]),
    )
    assert not out.empty
    assert set(out["source_cohort"]) == {canonical}
    assert set(out["cancer_code"]) == {"SARC_DDLPS", "SARC_WDLPS"}
    assert (
        "SARC DDLPS/WDLPS source cohort normalized to registry label"
        in out.attrs["compatibility_transforms"]
    )


def test_identical_locus_collapse_precedes_raw_log_transform():
    for linear_mode, log_mode in (
        ("tpm", "tpm_log1p"),
        ("tpm_clean", "tpm_clean_log1p"),
    ):
        linear = accessors.cancer_reference_expression(
            cancer_types="SKCM",
            genes=["CTAG1A", "CTAG1B"],
            normalize=linear_mode,
        )
        collapsed_log = accessors.cancer_reference_expression(
            cancer_types="SKCM",
            genes=["CTAG1A", "CTAG1B"],
            normalize=log_mode,
            collapse_cdna_identical=True,
        )
        expected = np.log1p(linear["expression"].sum())
        actual = collapsed_log.loc[
            collapsed_log["Symbol"].eq("CTAG1A/B"), "expression"
        ]
        assert len(actual) == 1
        assert np.isclose(actual.iloc[0], expected)
