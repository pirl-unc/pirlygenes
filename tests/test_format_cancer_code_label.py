"""Plot-label formatting for cancer-type codes: pos/neg molecular-status
suffixes render as Unicode superscripts."""

from pirlygenes.gene_sets_cancer import format_cancer_code_label as fmt


def test_hpv_pos_neg_become_superscripts():
    assert fmt("HNSC_HPVpos") == "HNSC_HPV⁺"   # HNSC_HPV⁺
    assert fmt("HNSC_HPVneg") == "HNSC_HPV⁻"   # HNSC_HPV⁻


def test_other_codes_unchanged():
    for code in ("PRAD", "SARC_LMS", "NET_PANCREAS", "BRCA_LumA", "SCLC"):
        assert fmt(code) == code


def test_non_string_coerced():
    assert fmt(123) == "123"
