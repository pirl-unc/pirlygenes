"""Regression test for issue #160 — family-factor penalty miscalls
orphan-family cancer types on their own cohort medians.

Before the fix, the classifier's geomean multiplied a `family_factor`
that dropped orphan cancer types (BLCA, PAAD — no family grouping) to
~0.15 whenever a family-matched competitor sat near the top family.
That made the TCGA-median of BLCA itself classify as ESCA (squamous
family) despite BLCA having the highest signature, purity, and
lineage scores.

The fix adds a raw-signal dominance override: when an orphan's
`signature × purity × lineage_support` product exceeds the top
family-matched competitor's by ≥ 1.5×, the family penalty is
suspended for that orphan.
"""

import pandas as pd
import pytest

from pirlygenes.gene_sets_cancer import pan_cancer_expression
from pirlygenes.tumor_purity import rank_cancer_type_candidates


def _cohort_median_sample(code: str) -> pd.DataFrame:
    """Synthesize a pseudo-sample whose TPM column is the TCGA cohort
    median for ``code``. Equivalent to what the median battery uses."""
    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    return pd.DataFrame({
        "ensembl_gene_id": ref["Ensembl_Gene_ID"],
        "gene_symbol": ref["Symbol"],
        "TPM": ref[f"FPKM_{code}"].astype(float),
    })


@pytest.mark.parametrize("code", ["BLCA"])
def test_orphan_family_cohort_median_classifies_as_itself(code):
    """BLCA median → top candidate must be BLCA, not the family
    competitor that was winning before the fix (ESCA)."""
    df = _cohort_median_sample(code)
    ranked = rank_cancer_type_candidates(df)
    assert ranked, "classifier returned no candidates"
    top = ranked[0]
    assert top["code"] == code, (
        f"{code} median miscalled as {top['code']}. Rows: "
        + ", ".join(f"{r['code']}({r['support_geomean']:.3f})" for r in ranked[:5])
    )


def test_orphan_dominance_override_raises_orphan_family_factor():
    """When an orphan's raw signal dominates the family winner's by
    ≥ 1.5×, the override resets its family_factor to 1.0 and
    recomputes support_score / support_geomean."""
    df = _cohort_median_sample("BLCA")
    ranked = rank_cancer_type_candidates(df)
    blca_row = next((r for r in ranked if r["code"] == "BLCA"), None)
    assert blca_row is not None, "BLCA missing from candidate trace"
    # Post-override BLCA (orphan) should not be down-weighted.
    assert blca_row["family_factor"] >= 0.9, (
        f"BLCA family_factor={blca_row['family_factor']} — orphan "
        "dominance override did not fire; expected ≥ 0.9"
    )


def test_non_orphan_family_matched_candidates_keep_family_factor():
    """The override only fires for orphans. A family-matched candidate
    (e.g. COAD → CRC family) should NOT have its family_factor
    changed by this code path — the within-family boost/penalty logic
    is independent."""
    df = _cohort_median_sample("COAD")
    ranked = rank_cancer_type_candidates(df)
    coad_row = next((r for r in ranked if r["code"] == "COAD"), None)
    assert coad_row is not None
    # COAD has a family label — override shouldn't touch it.
    assert coad_row["family_label"] is not None
    # And classifier still picks COAD as top.
    assert ranked[0]["code"] == "COAD"


def test_override_rejects_orphan_with_tme_driven_raw_signal():
    """Guard: the override must NOT fire when an orphan's apparent raw
    signal is TME-driven rather than tumor-driven. On a COAD + lymph-
    node 30/70 mix, DLBC's raw_signal (sig × purity × lineage) can
    exceed COAD's because the lymph-node TME matches DLBC's panel —
    but DLBC's purity is ~0.33 and signature is ~0.60, below the
    override's signature + purity gates. Classifier should keep COAD
    (or READ) on top, not DLBC."""
    from pirlygenes.gene_sets_cancer import pan_cancer_expression

    ref = pan_cancer_expression().drop_duplicates(subset="Ensembl_Gene_ID")
    coad = pd.Series(
        ref["FPKM_COAD"].astype(float).values,
        index=ref["Ensembl_Gene_ID"].values,
    )
    lymph = pd.Series(
        ref["nTPM_lymph_node"].astype(float).values,
        index=ref["Ensembl_Gene_ID"].values,
    )
    mix = 0.3 * coad + 0.7 * lymph
    df_mix = pd.DataFrame({
        "ensembl_gene_id": mix.index,
        "gene_symbol": ref.set_index("Ensembl_Gene_ID").loc[mix.index, "Symbol"].values,
        "TPM": mix.values,
    })
    ranked = rank_cancer_type_candidates(
        df_mix,
        candidate_codes=["COAD", "READ", "DLBC", "THYM", "SARC", "STAD"],
        top_k=6,
    )
    top_code = ranked[0]["code"]
    assert top_code in {"COAD", "READ"}, (
        f"COAD+lymph mix miscalled as {top_code} — override fired "
        f"despite TME bleed-through. Rows: "
        + ", ".join(
            f"{r['code']}(sig={r['signature_score']:.2f},pur={r['purity_estimate']:.2f},"
            f"ff={r['family_factor']:.2f},gm={r['support_geomean']:.2f})"
            for r in ranked[:4]
        )
    )
