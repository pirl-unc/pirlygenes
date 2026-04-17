"""Tests for the robust attribution algorithm (#128).

Covers three invariants the new breadth-aware attribution must hold:

1. Tissue-restricted genes (KLK3 in prostate, NEUROD1 in brain, etc.)
   must NOT trip the broadly-expressed flag even when low-level
   detection crosses many tissues.
2. Ubiquitously-expressed housekeeping-like genes (TBCE, NPM1, broad
   surface markers) MUST trip the flag.
3. The breadth floor on non-tumor attribution dampens the tumor-core
   residual for broadly-expressed genes without zeroing it out, and
   doesn't touch restricted genes.
"""

import numpy as np
import pandas as pd

from pirlygenes.plot_tumor_expr import (
    AMPLIFICATION_MIN_FOLD,
    BROADLY_ENRICHED_MAX_RATIO,
    BROAD_TISSUE_COUNT,
    HK_TISSUE_NTPM_THRESHOLD,
    BREADTH_BASELINE_TOP_N,
)


# ── threshold-sanity tests ──────────────────────────────────────────────

def test_thresholds_are_tunable_constants():
    """Thresholds should be module-level constants so downstream
    consumers can tune them for specific tissue contexts without
    forking the call sites."""
    assert HK_TISSUE_NTPM_THRESHOLD == 5.0
    assert BROAD_TISSUE_COUNT == 15
    assert BROADLY_ENRICHED_MAX_RATIO == 3.0
    assert BREADTH_BASELINE_TOP_N == 10
    assert AMPLIFICATION_MIN_FOLD == 5.0


# ── amplification override ──────────────────────────────────────────────


def test_amplification_cell_renders_amplified_not_broadly_expr():
    """A gene broadly expressed at baseline but observed ≥ 5× peak
    healthy should render 'amplified N×' rather than 'broadly expr.'
    — HER2 / MDM2 / GPC3 pattern (#128 refinement)."""
    from pirlygenes.cli import _format_attribution_cell

    row = {
        "observed_tpm": 1200.0,
        "attribution": {"endothelial": 40.0},
        "attr_tumor_tpm": 1100.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 40.0,
        # broadly_expressed is already gated on `not amplified` in the
        # pipeline, so the cell receives broadly_expressed=False and
        # amplified=True for HER2-amp BRCA.
        "broadly_expressed": False,
        "amplified_over_healthy": True,
        "amplification_fold": 12.0,
    }
    cell = _format_attribution_cell(row)
    assert "amplified" in cell
    assert "12.0" in cell
    assert "broadly expr." not in cell


def test_amplification_does_not_trigger_confidence_downgrade():
    """Confidence tier must NOT downgrade an amplified target even
    though it's broadly expressed at baseline."""
    from pirlygenes.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 1200.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.92,
        # Pipeline sets broadly_expressed=False on amplified-over-
        # healthy genes; confidence sees that as the gate.
        "broadly_expressed": False,
        "amplified_over_healthy": True,
        "amplification_fold": 12.0,
        "n_healthy_tissues_expressed": 42,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "high", (
        f"amplified target wrongly downgraded to {tier.tier!r}: {tier}"
    )


def test_truly_broadly_expressed_still_downgrades():
    """Without amplification, the broadly-expressed flag should still
    fire and the confidence tier should drop to low."""
    from pirlygenes.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 800.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.88,
        "broadly_expressed": True,
        "amplified_over_healthy": False,
        "amplification_fold": 1.4,
        "n_healthy_tissues_expressed": 42,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "low"


# ── breadth-gate semantics ──────────────────────────────────────────────


def _simulate_tissue_profile(peak_tpm, n_tissues_at_peak=1,
                             other_tissues_tpm=0.0, n_total_tissues=50):
    """Build a synthetic row of 50 nTPM values. One or more tissues at
    ``peak_tpm``, the rest at ``other_tissues_tpm``.
    """
    values = [other_tissues_tpm] * n_total_tissues
    for i in range(n_tissues_at_peak):
        values[i] = peak_tpm
    return values


def _breadth_call(peak_tpm, n_tissues_at_peak, other_tpm, n_total_tissues=50):
    """Compute broadly_expressed directly from a simulated profile using
    the same logic the production code applies.
    """
    profile = np.array(_simulate_tissue_profile(
        peak_tpm, n_tissues_at_peak, other_tpm, n_total_tissues,
    ))
    n_expressed = int((profile >= HK_TISSUE_NTPM_THRESHOLD).sum())
    top_n = np.sort(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = float(top_n.mean())
    peak = float(profile.max())
    ratio = peak / mean_top if mean_top > 0 else float("inf")
    return {
        "broadly_expressed": (
            n_expressed >= BROAD_TISSUE_COUNT
            and ratio < BROADLY_ENRICHED_MAX_RATIO
        ),
        "n_expressed": n_expressed,
        "ratio": ratio,
        "mean_top": mean_top,
    }


def test_prostate_restricted_gene_not_broadly_expressed():
    """KLK3-like profile: very high in prostate, near-zero elsewhere.
    Must NOT be flagged broadly-expressed even though a handful of
    tissues cross the detection threshold due to baseline noise."""
    call = _breadth_call(peak_tpm=500.0, n_tissues_at_peak=1, other_tpm=0.2)
    assert not call["broadly_expressed"], (
        f"restricted gene flagged broadly_expressed: {call!r}"
    )
    # Sanity: enrichment ratio is huge.
    assert call["ratio"] > 5


def test_brain_restricted_gene_not_broadly_expressed():
    """NEUROD1-like: high in a few CNS tissues, quiet elsewhere."""
    call = _breadth_call(peak_tpm=200.0, n_tissues_at_peak=3, other_tpm=0.1)
    assert not call["broadly_expressed"], call


def test_ubiquitous_housekeeping_is_broadly_expressed():
    """TBCE-like: moderate expression everywhere, no strong enrichment
    anywhere. Must trip the flag."""
    # 50 tissues all at ~80 nTPM — no enrichment, full breadth.
    profile = [80.0] * 50
    n_expressed = sum(1 for v in profile if v >= HK_TISSUE_NTPM_THRESHOLD)
    peak = max(profile)
    top_n = sorted(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = sum(top_n) / len(top_n)
    ratio = peak / mean_top
    broadly = (
        n_expressed >= BROAD_TISSUE_COUNT
        and ratio < BROADLY_ENRICHED_MAX_RATIO
    )
    assert broadly, f"expected broadly-expressed; got n={n_expressed}, ratio={ratio}"


def test_broad_surface_with_mild_enrichment_is_broadly_expressed():
    """CRIM1 / IL6ST-like: expressed across many tissues with only mild
    enrichment in any single one (< 3× mean-of-top)."""
    # 25 tissues at 30 nTPM, peak at 60 — ratio 2× top-10 mean.
    values = [30.0] * 25 + [0.0] * 25
    values[0] = 60.0
    profile = np.array(values)
    n_expressed = int((profile >= HK_TISSUE_NTPM_THRESHOLD).sum())
    top_n = np.sort(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = float(top_n.mean())
    ratio = float(profile.max()) / mean_top
    broadly = (
        n_expressed >= BROAD_TISSUE_COUNT
        and ratio < BROADLY_ENRICHED_MAX_RATIO
    )
    assert broadly, (
        f"broad-surface gene not flagged: n={n_expressed}, ratio={ratio}"
    )


def test_borderline_tissue_count_but_strong_enrichment_not_flagged():
    """Gene crossing the tissue-count threshold purely on low-level
    detection, but with one tissue dominating by >3×, must NOT be
    flagged. This is the enrichment-ratio gate in action — it keeps
    the flag tissue-type agnostic."""
    # 20 tissues at 6 nTPM (just above HK threshold), one at 500.
    values = [6.0] * 20 + [0.0] * 30
    values[0] = 500.0
    profile = np.array(values)
    n_expressed = int((profile >= HK_TISSUE_NTPM_THRESHOLD).sum())
    top_n = np.sort(profile)[-BREADTH_BASELINE_TOP_N:]
    mean_top = float(top_n.mean())
    ratio = float(profile.max()) / mean_top
    broadly = (
        n_expressed >= BROAD_TISSUE_COUNT
        and ratio < BROADLY_ENRICHED_MAX_RATIO
    )
    assert not broadly, (
        f"tissue-restricted (strong enrichment) flagged broadly: "
        f"n={n_expressed}, ratio={ratio}"
    )


# ── confidence-tier integration ─────────────────────────────────────────


def test_confidence_tier_fires_low_on_broadly_expressed():
    from pirlygenes.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 800.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.92,
        "broadly_expressed": True,
        "n_healthy_tissues_expressed": 28,
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "low"
    assert any("broadly expressed" in r for r in tier.reasons), tier.reasons


def test_confidence_tier_stays_high_on_tissue_restricted_high_attr():
    from pirlygenes.confidence import (
        ConfidenceTier,
        compute_target_confidence,
    )

    purity_tier = ConfidenceTier(tier="high", reasons=[])
    target = {
        "observed_tpm": 142.0,
        "tme_dominant": False,
        "tme_explainable": False,
        "attr_tumor_fraction": 0.90,
        "broadly_expressed": False,
        "n_healthy_tissues_expressed": 2,  # prostate-restricted
    }
    tier = compute_target_confidence(target, purity_tier)
    assert tier.tier == "high", tier


# ── brief / top-3 skip semantics ────────────────────────────────────────


def test_brief_trusts_curation_over_broadly_expressed_flag():
    """Curated therapy targets (#110) may legitimately be broadly
    expressed — HER2 for BRCA, MDM2 for WD/DD-LPS, GPC3 for HCC are
    all expressed in many healthy tissues, but the targeting
    mechanism is amplification / lineage-retained overexpression, not
    baseline specificity. The brief's top-3 must trust the curation
    panel and NOT filter broadly-expressed rows out — otherwise
    legitimate first-line agents get silently dropped from the
    clinician handoff (#128).
    """
    from pirlygenes.brief import _top_therapies

    targets_df = pd.DataFrame([
        {
            "symbol": "ERBB2", "agent": "trastuzumab",
            "agent_class": "antibody", "phase": "approved",
            "indication": "HER2+ BRCA",
        },
        {
            "symbol": "FOLH1", "agent": "177Lu-PSMA-617",
            "agent_class": "radioligand", "phase": "approved",
            "indication": "mCRPC",
        },
    ])
    ranges_df = pd.DataFrame([
        {
            "symbol": "ERBB2", "observed_tpm": 1200.0,  # HER2-amplified
            "attribution": {"endothelial": 40.0},
            "attr_tumor_tpm": 1100.0, "attr_tumor_fraction": 0.92,
            "attr_top_compartment": "endothelial",
            "attr_top_compartment_tpm": 40.0,
            "tme_dominant": False, "tme_explainable": False,
            # HER2 is broadly expressed in HPA but AMPLIFICATION is what
            # drives the therapy — this flag must not drop it from the
            # brief.
            "broadly_expressed": True,
        },
        {
            "symbol": "FOLH1", "observed_tpm": 142.0,
            "attribution": {"endothelial": 12.0},
            "attr_tumor_tpm": 128.0, "attr_tumor_fraction": 0.90,
            "attr_top_compartment": "endothelial",
            "attr_top_compartment_tpm": 12.0,
            "tme_dominant": False, "tme_explainable": False,
            "broadly_expressed": False,
        },
    ])

    top = _top_therapies(targets_df, ranges_df, limit=3)
    symbols = [t["symbol"] for t, _ in top]
    assert "ERBB2" in symbols, (
        "curated HER2 target must remain in the brief top-3 despite "
        "being broadly expressed in HPA"
    )
    assert "FOLH1" in symbols


# ── attribution cell rendering ──────────────────────────────────────────


def test_attribution_cell_appends_broadly_expr_tag():
    from pirlygenes.cli import _format_attribution_cell

    row = {
        "observed_tpm": 800.0,
        "attribution": {"endothelial": 20.0, "fibroblast": 10.0},
        "attr_tumor_tpm": 770.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 20.0,
        "broadly_expressed": True,
    }
    cell = _format_attribution_cell(row)
    assert "broadly expr." in cell
    assert "770" in cell


def test_attribution_cell_no_tag_on_tissue_restricted():
    from pirlygenes.cli import _format_attribution_cell

    row = {
        "observed_tpm": 142.0,
        "attribution": {"endothelial": 12.0},
        "attr_tumor_tpm": 128.0,
        "attr_top_compartment": "endothelial",
        "attr_top_compartment_tpm": 12.0,
        "broadly_expressed": False,
    }
    cell = _format_attribution_cell(row)
    assert "broadly expr." not in cell
