"""clean_tpm_v5 cap-only technical normalization (#304).

v4 (censored_fill='fixed_fraction') FORCES the technical block to 25%, inflating
technical genes UP in technically-light cohorts (CD138-sorted MM, T_ALL). v5
(censored_fill='fixed_fraction_cap') CAPS at 25% — compress when above, never
inflate above the natural level.
"""

import pandas as pd

from pirlygenes.expression.normalize import clean_tpm_matrix

_IDX = ["MT-CO1", "MT-ND1", "GAPDH", "ACTB"]
_REM = pd.Series([True, True, False, False], index=_IDX)
# sample LIGHT: technical = 10% of budget; HEAVY: technical = 50%
_VALS = pd.DataFrame(
    {"LIGHT": [50000, 50000, 450000, 450000],
     "HEAVY": [250000, 250000, 250000, 250000]}, index=_IDX)


def _tech_frac(clean, col):
    return clean[col].loc[["MT-CO1", "MT-ND1"]].sum() / 1_000_000.0


def test_v4_forces_to_fraction_inflating_light():
    c = clean_tpm_matrix(_VALS, removable=_REM, censored_fill="fixed_fraction")
    assert abs(_tech_frac(c, "LIGHT") - 0.25) < 1e-6   # 10% -> 25% (inflated)
    assert abs(_tech_frac(c, "HEAVY") - 0.25) < 1e-6


def test_v5_caps_only_never_inflates():
    c = clean_tpm_matrix(_VALS, removable=_REM,
                         censored_fill="fixed_fraction_cap")
    assert abs(_tech_frac(c, "LIGHT") - 0.10) < 1e-6   # stays 10% (not inflated)
    assert abs(_tech_frac(c, "HEAVY") - 0.25) < 1e-6   # 50% capped to 25%
    # each compartment renormalises to a full 1e6 budget
    assert abs(c["LIGHT"].sum() - 1_000_000.0) < 1.0
    assert abs(c["HEAVY"].sum() - 1_000_000.0) < 1.0
