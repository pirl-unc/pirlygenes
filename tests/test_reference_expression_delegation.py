"""Contract tests for the oncoref cancer-reference-expression cutover (#557)."""

import numpy as np
import pandas as pd
import pytest

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


def test_delegated_filter_preserves_case_insensitive_trimmed_symbols():
    for gene in ("ms4a1", " MS4A1 "):
        out = accessors.cancer_reference_expression(
            cancer_types="CLL",
            genes=[gene],
        )
        assert out["Symbol"].tolist() == ["MS4A1"]

        compact = accessors.cancer_expression("CLL", genes=[gene])
        assert compact["Symbol"].tolist() == ["MS4A1"]


@pytest.mark.parametrize(
    ("code", "display_alias", "official_symbol"),
    [
        ("LUAD", "p53", "TP53"),
        ("LUAD", "P53", "TP53"),
        ("SKCM", "gp100", "PMEL"),
        ("SKCM", "GP100", "PMEL"),
        ("OV", "FRα", "FOLR1"),
        ("OV", "frΑ", "FOLR1"),
    ],
)
def test_display_aliases_are_unicode_case_insensitive(
    code, display_alias, official_symbol,
):
    out = accessors.cancer_reference_expression(
        cancer_types=code,
        genes=[display_alias],
    )
    assert not out.empty
    assert set(out["Symbol"]) == {official_symbol}

    compact = accessors.cancer_expression(code, genes=[display_alias])
    assert not compact.empty
    assert set(compact["Symbol"]) == {official_symbol}


def test_explicit_empty_source_kind_returns_no_rows():
    out = accessors.cancer_reference_expression(
        cancer_types="CLL",
        genes=["MS4A1"],
        source_kind=[],
    )
    assert out.empty
    assert out.attrs["availability"][0]["available"] is False


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("source_cohort", [[], (), ""])
def test_explicit_empty_source_cohort_returns_no_rows(source_cohort, pool):
    out = accessors.cancer_reference_expression(
        cancer_types="CLL",
        genes=["MS4A1"],
        source_cohort=source_cohort,
        pool=pool,
    )
    assert out.empty
    assert out.attrs["availability"][0]["available"] is False


def test_source_filtered_empty_wide_result_keeps_requested_columns():
    out = accessors.cancer_reference_expression(
        cancer_types="CLL",
        genes=["MS4A1"],
        normalize=["tpm", "tpm_clean"],
        format="wide",
        source_kind="tcga",
    )
    assert out.empty
    assert list(out.columns) == [
        "Ensembl_Gene_ID",
        "Symbol",
        "CLL_TPM",
        "CLL_TPM_clean",
    ]


@pytest.mark.parametrize(
    "code",
    ["CRC_MSI", "NEC", "NEC_LUNG", "NEN", "NET", "RCC", "THYM_EPITHELIAL"],
)
def test_oncoref_only_groupings_remain_exact_unavailable_requests(code):
    long = accessors.cancer_reference_expression(
        cancer_types=code,
        genes=["TP53"],
    )
    assert long.empty

    wide = accessors.cancer_reference_expression(
        cancer_types=code,
        genes=["TP53"],
        format="wide",
    )
    assert wide.empty
    assert list(wide.columns) == ["Ensembl_Gene_ID", "Symbol"]


def test_merkel_geo_kind_uses_pirlygenes_cohort_registry():
    out = accessors.cancer_reference_expression(
        cancer_types="NEC_MERKEL",
        genes=["TP53"],
        source_kind="geo",
    )
    assert not out.empty
    assert set(out["cancer_code"]) == {"NEC_MERKEL"}
    assert set(out["source_cohort"]) == {"GSE235092_MERKEL_2024"}


def test_exact_cohort_filter_is_applied_before_pooling():
    out = accessors.cancer_reference_expression(
        cancer_types="CLL",
        genes=["MS4A1"],
        source_cohort="CLLMAP_2022",
        pool=True,
    )
    assert not out.empty
    assert set(out["source_cohort"]) == {"POOLED"}
    assert out["expression"].notna().all()


def test_nutm_exact_cohort_filter_is_applied_before_pooling():
    out = accessors.cancer_reference_expression(
        cancer_types="NUTM",
        genes=["TP53"],
        source_cohort="UNC_NUTM1",
        pool=True,
    )
    assert not out.empty
    assert set(out["source_cohort"]) == {"POOLED"}
    assert out["expression"].notna().all()
    assert any(
        record["cancer_code"] == "NUTM" and record["available"]
        for record in out.attrs["availability"]
    )


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
    availability = {
        record["cancer_code"]: record for record in out.attrs["availability"]
    }
    assert availability["SARC_DDLPS"]["available"] is True
    assert availability["SARC_WDLPS"]["available"] is True
    assert availability["SARC_PLEOLPS"]["available"] is False
    assert (
        availability["SARC_PLEOLPS"]["missing_reason"]
        == "no_reference_summary_rows"
    )
    assert any(
        record["cancer_code"] == "SARC_PLEOLPS"
        for record in out.attrs["missing_requests"]
    )
    assert (
        "source-cohort labels normalized to public registry identities"
        in out.attrs["compatibility_transforms"]
    )
    assert (
        "availability reconciled with public source-cohort filter"
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


def test_one_shot_filters_are_reused_across_normalization_modes():
    out = accessors.cancer_reference_expression(
        cancer_types=(code for code in ["CLL"]),
        genes=(gene for gene in ["MS4A1"]),
        normalize=["tpm", "tpm_clean"],
        source_kind=(kind for kind in ["cllmap"]),
        source_cohort=(cohort for cohort in ["CLLMAP_2022"]),
    )
    assert set(out["normalization"]) == {"TPM", "TPM_clean"}
    assert out.groupby("normalization").size().nunique() == 1
    assert set(out["Symbol"]) == {"MS4A1"}
