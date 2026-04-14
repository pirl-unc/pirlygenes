import pandas as pd

from pirlygenes.gene_sets_cancer import pan_cancer_expression, degradation_gene_pairs
from pirlygenes.sample_context import (
    SampleContext,
    infer_sample_context,
    _HISTONE_SYMBOL_PREFIXES,
    _MT_MRNA_SYMBOLS,
    _MT_RRNA_SYMBOLS,
)


def _frame_from_pairs(pairs):
    """Build a minimal (gene_symbol, TPM) expression frame from [(sym, tpm)]."""
    return pd.DataFrame([{"gene_symbol": s, "TPM": float(t)} for s, t in pairs])


def _pan_cancer_ntpm_sample(tissue):
    """Synthetic sample equal to a reference normal tissue column."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame(
        {
            "ensembl_gene_id": ref["Ensembl_Gene_ID"],
            "gene_symbol": ref["Symbol"],
            "TPM": ref[f"nTPM_{tissue}"].astype(float),
        }
    )


def _fresh_degradation_pair_expectations():
    """Return list of (short, long, expected) pairs whose ``expected`` ratio
    is used as the reference 'fresh' value."""
    return list(degradation_gene_pairs())


# ── Library-prep inference ────────────────────────────────────────────────


def test_library_prep_polya_signature_is_detected():
    """Histones absent, MT rRNAs absent, MT mRNAs present → poly-A."""
    # Background expression to make histone + MT fractions small/large
    # relative to total.
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # MT mRNAs present (typical poly-A capture keeps them)
    rows += [(sym, 50) for sym in _MT_MRNA_SYMBOLS]
    # MT rRNAs absent. Histones absent.
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "poly_a"
    assert ctx.library_prep_confidence >= 0.5


def test_library_prep_total_rna_signature_is_detected():
    """Histones high, MT rRNAs dominant → total RNA."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # High histone mass — use modern HGNC names so prefix detection
    # catches H2AC* / H2BC* / H3C* / H4C*.
    rows += [
        ("H2BC1", 80), ("H2BC3", 70), ("H2BC4", 65), ("H2AC1", 60),
        ("H3C1", 55), ("H4C1", 50),
    ]
    # MT rRNAs dominate MT signal (characteristic of total RNA libraries)
    rows += [(sym, 200) for sym in _MT_RRNA_SYMBOLS]
    rows += [(sym, 20) for sym in list(_MT_MRNA_SYMBOLS)[:5]]
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "total_rna"
    assert ctx.library_prep_confidence >= 0.7


def test_library_prep_ribo_depleted_signature_is_detected():
    """Histones present, MT rRNAs depleted → ribo-depleted."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    rows += [
        ("H2BC1", 80), ("H2BC3", 70), ("H2AC1", 60), ("H3C1", 55), ("H4C1", 50),
    ]
    # MT rRNAs absent or very low; MT mRNAs present
    rows += [(sym, 40) for sym in _MT_MRNA_SYMBOLS]
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "ribo_depleted"


def test_library_prep_exome_capture_signature_is_detected():
    """MT absent AND histones absent → exome capture."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300), ("TP53", 20), ("BRCA1", 10)]
    # Nothing else — no MT, no histones.
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.library_prep == "exome_capture"


# ── Preservation + degradation ────────────────────────────────────────────


def test_fresh_sample_not_flagged_as_degraded():
    """A synthetic sample equal to a reference tissue column has
    degradation-pair ratios equal to the expected values, so the
    inference should land on fresh_frozen."""
    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    assert ctx.preservation == "fresh_frozen"
    assert ctx.degradation_severity == "none"
    assert ctx.degradation_index is not None
    assert 0.7 < ctx.degradation_index < 1.5


def test_ffpe_sample_triggers_ffpe_preservation():
    """Construct a sample where long transcripts are systematically
    depressed relative to their matched short transcripts — the
    canonical FFPE signature.
    """
    import random
    random.seed(0)
    rows = []
    # Background
    rows += [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    # MT genes present (poly-A-like) for library_prep side signal
    rows += [(sym, 30) for sym in _MT_MRNA_SYMBOLS]
    # For each degradation pair, short gets normal TPM but long gets
    # 10x lower than the fresh expected ratio would predict.
    for short_sym, long_sym, expected in _fresh_degradation_pair_expectations()[:18]:
        short_tpm = 50.0 + random.random() * 20
        long_tpm = short_tpm * float(expected) * 0.10   # 10× depletion
        rows.append((short_sym, short_tpm))
        rows.append((long_sym, long_tpm))
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.preservation == "ffpe"
    assert ctx.degradation_severity in ("moderate", "severe")
    assert ctx.degradation_index is not None and ctx.degradation_index < 0.3
    assert ctx.long_transcript_weight_factor() < 1.0
    assert ctx.purity_ci_widening_factor() > 1.0


# ── Missing-MT ────────────────────────────────────────────────────────────


def test_missing_mt_flag_set_when_mt_genes_absent():
    """Expression frames without any MT gene should set missing_mt."""
    rows = [("ACTB", 500), ("GAPDH", 400), ("TUBB", 300)]
    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)
    assert ctx.missing_mt is True
    assert any("Mitochondrial genes missing" in f for f in ctx.flags)


# ── Downstream adjustment factors ─────────────────────────────────────────


def test_long_transcript_weight_factor_monotone_with_severity():
    base = SampleContext(degradation_severity="none")
    mild = SampleContext(degradation_severity="mild")
    moderate = SampleContext(degradation_severity="moderate")
    severe = SampleContext(degradation_severity="severe")
    assert base.long_transcript_weight_factor() == 1.0
    assert (
        severe.long_transcript_weight_factor()
        < moderate.long_transcript_weight_factor()
        < mild.long_transcript_weight_factor()
        <= 1.0
    )


def test_purity_ci_widening_factor_monotone_with_severity():
    base = SampleContext(degradation_severity="none")
    mild = SampleContext(degradation_severity="mild")
    moderate = SampleContext(degradation_severity="moderate")
    severe = SampleContext(degradation_severity="severe")
    assert base.purity_ci_widening_factor() == 1.0
    assert (
        severe.purity_ci_widening_factor()
        > moderate.purity_ci_widening_factor()
        > mild.purity_ci_widening_factor()
        > 1.0
    )


def test_summary_line_has_prep_and_preservation():
    ctx = SampleContext(
        library_prep="poly_a",
        preservation="ffpe",
        degradation_severity="severe",
    )
    line = ctx.summary_line()
    assert "poly-A" in line
    assert "FFPE" in line
    assert "severe" in line


def test_expression_distribution_signals_populated():
    """The full-range expression summary should capture detection
    breadth and dynamic range for every sample."""
    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    s = ctx.signals
    for key in (
        "n_genes_with_any_expression",
        "genes_detected_above_0p5_tpm",
        "genes_detected_above_1_tpm",
        "genes_detected_above_10_tpm",
        "log2_tpm_median",
        "log2_tpm_iqr",
        "log2_tpm_p95",
        "top_1pct_share_of_total_tpm",
    ):
        assert key in s, f"{key} missing from signals"
    assert s["n_genes_with_any_expression"] > 5000
    assert 0 < s["top_1pct_share_of_total_tpm"] <= 1.0


def test_tempus_like_exome_capture_ffpe_detected_correctly():
    """Bug 2026-04-14: Tempus FFPE exome-capture samples got labeled
    'unknown library prep, fresh_frozen'. Simulate the signal profile
    from that sample (MT-rRNA absent, MT-mRNA present at vanishing
    fractions, histone in the 0.1–0.5% 'limbo' band, and a long-pair
    index >> 1 from capture enrichment) and verify the detector now
    routes to ``exome_capture`` + ``preservation=unknown``.
    """
    # Background that dominates total TPM.
    rows = [("ACTB", 1000), ("GAPDH", 800), ("TUBB", 600), ("EEF1A1", 500)]
    # Fill out the gene universe enough that histones land in the
    # "limbo" band (~0.3% of total). Use 8 bulk genes at ~100 TPM each.
    for i in range(8):
        rows.append((f"FILLER{i}", 200))
    # Very small amounts of a few MT-mRNA genes (probe bleed-through).
    rows += [("MT-CO1", 0.1), ("MT-ND1", 0.1)]
    # Histones in the "limbo" band — ~0.3% of total.
    total_bg = 1000 + 800 + 600 + 500 + 8 * 200 + 0.2
    target_histone = total_bg * 0.003 / (1 - 0.003)  # ≈ 0.3% of final total
    rows += [("H2BC1", target_histone / 6)] * 6
    # Degradation pairs: long transcripts over-represented ~3x.
    from pirlygenes.gene_sets_cancer import degradation_gene_pairs
    for short_sym, long_sym, expected in list(degradation_gene_pairs())[:15]:
        rows.append((short_sym, 20.0))
        rows.append((long_sym, 20.0 * float(expected) * 3.0))

    frame = _frame_from_pairs(rows)
    ctx = infer_sample_context(frame)

    # Library prep: exome_capture high confidence (MT-rRNA absent,
    # MT-total tiny).
    assert ctx.library_prep == "exome_capture", ctx.library_prep
    assert ctx.library_prep_confidence >= 0.85

    # Preservation: must NOT be labeled fresh_frozen on a capture-
    # biased sample — the length-pair index is inflated artefactually.
    assert ctx.preservation != "fresh_frozen"
    assert ctx.degradation_index is not None
    assert ctx.degradation_index > 1.4
    # The flag must explain the capture-bias interpretation.
    assert any(
        "exon-capture" in f or "capture enrichment" in f
        for f in ctx.flags
    ), ctx.flags


def test_degradation_index_above_upper_bound_yields_unknown_preservation():
    """Regression: the index > 1.4 path must route to
    ``preservation=unknown`` rather than ``fresh_frozen``."""
    rows = [("ACTB", 500), ("GAPDH", 400)]
    # Every pair has long >> short, index ≈ 5× expected.
    from pirlygenes.gene_sets_cancer import degradation_gene_pairs
    for short_sym, long_sym, expected in list(degradation_gene_pairs())[:12]:
        rows.append((short_sym, 10.0))
        rows.append((long_sym, 10.0 * float(expected) * 5.0))
    rows += [(s, 30) for s in _MT_MRNA_SYMBOLS]  # Present — not exome capture
    frame = _frame_from_pairs(rows)

    ctx = infer_sample_context(frame)
    assert ctx.preservation == "unknown"
    assert ctx.degradation_index is not None and ctx.degradation_index > 1.4


def test_plot_sample_context_writes_png(tmp_path):
    from pirlygenes.sample_context import plot_sample_context

    df = _pan_cancer_ntpm_sample("liver")
    ctx = infer_sample_context(df)
    out = tmp_path / "sample-context.png"
    plot_sample_context(ctx, save_to_filename=str(out), save_dpi=80)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_histone_prefixes_include_both_hgnc_old_and_new():
    # Guard against accidental deletion: we detect histones whether
    # annotations use old HIST1H* names or modern HGNC H2AC*/H3C*/H4C*.
    prefixes = _HISTONE_SYMBOL_PREFIXES
    assert "H2BC" in prefixes
    assert "H3C" in prefixes
    assert "HIST1H" in prefixes
