"""Tests for the prefix-agnostic sarcoma-lineage membership (Phase C foundation)."""

from pirlygenes.gene_sets_cancer import sarcoma_lineage_codes


def test_spans_soft_tissue_bone_and_pediatric():
    codes = set(sarcoma_lineage_codes())
    # soft-tissue SARC_* + the umbrella + chondrosarcoma
    assert {"SARC", "SARC_LMS", "SARC_DDLPS", "SARC_SYN", "CHON"} <= codes
    # bone/pediatric sarcomas keep their (non-SARC_) codes but are members
    assert {"OS", "EWS"} <= codes
    # pediatric rhabdomyosarcomas
    assert {"RMS_ERMS", "RMS_ARMS"} <= codes
    # chordoma (notochordal, family="rare") included via the extra-codes set
    assert "CHOR" in codes


def test_endometrial_stromal_sarcomas_included():
    # ESS_LG/HG are family=sarcoma (curated, no per-sample data) — they belong
    # to the taxonomy even though they contribute no samples to the aggregate.
    codes = set(sarcoma_lineage_codes())
    assert {"ESS_LG", "ESS_HG"} <= codes


def test_excludes_non_sarcomas():
    codes = set(sarcoma_lineage_codes())
    for non_sarc in ("BRCA", "SKCM", "GBM", "PRAD", "NBL", "MTC"):
        assert non_sarc not in codes


def test_with_expression_only_drops_curated_entries():
    all_codes = set(sarcoma_lineage_codes())
    data_codes = set(sarcoma_lineage_codes(with_expression_only=True))
    assert data_codes <= all_codes
    # curated, no per-sample expression -> dropped from the expression subset
    assert "ESS_LG" in all_codes and "ESS_LG" not in data_codes
    # a Treehouse-backed code survives
    assert "SARC_LMS" in data_codes
    assert "OS" in data_codes
