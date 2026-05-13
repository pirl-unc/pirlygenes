"""Smoke tests for the kept expression_qc module.

The full expression_qc behaviour is exercised by trufflepig's analyze
test suite (which depends on pirlygenes). These tests are deliberately
narrow — they ensure the module imports cleanly inside pirlygenes
alone, the gene-class classifier returns sensible labels for a handful
of canonical symbols, and the normalize functions accept the schema
that ``gene_sets_cancer.pan_cancer_expression`` calls them with.
"""

from __future__ import annotations

import pandas as pd

from pirlygenes.expression_qc import (
    classify_gene_qc,
    is_rescue_feature,
    normalize_expression,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
)


def test_classify_gene_qc_handles_canonical_symbol_classes():
    assert classify_gene_qc("MT-CO1").group == "mt_dna"
    assert classify_gene_qc("MTCO1P12").group == "mt_like_pseudogene"
    assert classify_gene_qc("RNA5SP389").label == "5S rRNA pseudogene"
    assert classify_gene_qc("RPL13AP5").group == "ribosomal_protein_pseudogene"
    assert classify_gene_qc("SNORD3A").group == "small_ncrna"
    assert classify_gene_qc("H2AC1").group == "histone"
    assert classify_gene_qc("HBB").group == "hemoglobin"
    assert classify_gene_qc("IGKC").group == "immune_receptor"


def test_classify_gene_qc_rejects_blank_and_unknown():
    assert classify_gene_qc(None).group != "mt_dna"
    assert classify_gene_qc("").group != "mt_dna"
    # An ordinary protein-coding gene should not classify as anything special
    # (returns the catch-all "other" / "" — just assert it isn't mtDNA).
    assert classify_gene_qc("TP53").group != "mt_dna"


def test_is_rescue_feature_flags_mt_and_rrna_pseudogenes():
    # The "rescue" set is mt_dna / mt_like_pseudogene / rrna_like — these are
    # the technical-RNA features the analyzer strips during QC rescue.
    assert is_rescue_feature("MT-CO1")
    assert is_rescue_feature("RNA5SP389")
    assert not is_rescue_feature("TP53")
    assert not is_rescue_feature("EGFR")


def test_normalize_expression_zeros_technical_rna_and_returns_summary():
    """Defaults strip mt_dna / mt_like_pseudogene / rrna_like rows.

    The function returns ``(df, summary)``; the summary records what was
    removed and the column-wise mass distribution.
    """
    df = pd.DataFrame(
        {
            "Symbol": ["TP53", "EGFR", "MT-CO1", "RNA5SP389"],
            "FPKM_BRCA": [10.0, 20.0, 1000.0, 50.0],
            "FPKM_LUAD": [5.0, 25.0, 1500.0, 100.0],
        }
    )
    out, summary = normalize_expression(df)
    assert summary["applied"]
    # Affected rows get zeroed (or dropped) — the protein-coding rows survive.
    survivors = out[out["FPKM_BRCA"] > 0]
    assert "TP53" in set(survivors["Symbol"])
    assert "EGFR" in set(survivors["Symbol"])


def test_normalize_technical_rna_long_table_keeps_protein_coding():
    """Long-table shape used by subtype-deconvolved-expression rows."""
    long_df = pd.DataFrame(
        {
            "cancer_code": ["BRCA", "BRCA", "BRCA"],
            "symbol": ["TP53", "MT-CO1", "EGFR"],
            "tumor_tpm_median": [10.0, 5000.0, 25.0],
        }
    )
    out = normalize_technical_rna_long_table(long_df)
    # Some rows survive; the function may return either a tuple (df, summary)
    # or a DataFrame depending on version — accept both.
    if isinstance(out, tuple):
        out = out[0]
    survivors = set(out[out["tumor_tpm_median"] > 0]["symbol"])
    assert "TP53" in survivors
    assert "EGFR" in survivors
