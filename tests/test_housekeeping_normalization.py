from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pirlygenes.expression.normalize import tpm_to_housekeeping_normalized


HK_IDS = [f"ENSGHK{i:04d}" for i in range(1, 7)]


def _housekeeping_frame(hk_values: list[float], target_value: float = 100.0) -> pd.DataFrame:
    rows = [
        {
            "Symbol": f"HK{i}",
            "Ensembl_Gene_ID": ensg,
            "TPM_S1": value,
        }
        for i, (ensg, value) in enumerate(zip(HK_IDS, hk_values), start=1)
    ]
    rows.append({
        "Symbol": "TARGET",
        "Ensembl_Gene_ID": "ENSGTARGET",
        "TPM_S1": target_value,
    })
    return pd.DataFrame(rows)


def test_housekeeping_normalization_skips_sparse_hk_columns():
    df = _housekeeping_frame([0.0, 0.0, 0.0, 0.0, 10.0, 20.0])

    out, record = tpm_to_housekeeping_normalized(
        df,
        value_cols=["TPM_S1"],
        panel_ids=HK_IDS,
    )

    pd.testing.assert_series_equal(out["TPM_S1"], df["TPM_S1"])
    assert record["applied"] is False
    col = record["columns"]["TPM_S1"]
    assert col["applied"] is False
    assert col["skip_reason"] == "insufficient positive HK genes"
    assert col["n_hk_positive"] == 2
    assert col["n_hk_nonpositive"] == 4
    assert col["min_hk_positive_required"] == 5


def test_housekeeping_normalization_uses_only_positive_hk_values():
    hk_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    df = _housekeeping_frame(hk_values)

    out, record = tpm_to_housekeeping_normalized(
        df,
        value_cols=["TPM_S1"],
        panel_ids=HK_IDS,
    )

    expected_geomean = float(np.exp(np.log(np.array(hk_values[1:]) + 0.1).mean()) - 0.1)
    assert record["applied"] is True
    assert record["columns"]["TPM_S1"]["n_hk_used"] == 5
    assert record["columns"]["TPM_S1"]["n_hk_nonpositive"] == 1
    assert record["columns"]["TPM_S1"]["hk_geomean"] == pytest.approx(expected_geomean)
    assert out.loc[df["Symbol"].eq("TARGET"), "TPM_S1"].iloc[0] == pytest.approx(
        100.0 / expected_geomean
    )
