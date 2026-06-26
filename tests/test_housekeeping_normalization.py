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


def test_housekeeping_normalization_delegates_to_oncoref(monkeypatch):
    df = _housekeeping_frame([0.0, 0.0, 0.0, 0.0, 10.0, 20.0])
    calls = []

    def fake_oncoref(df_arg, **kwargs):
        calls.append((df_arg, kwargs))
        out = df_arg.copy()
        out["TPM_S1"] = out["TPM_S1"] / 2.0
        return out, {"applied": True, "panel": kwargs["panel_name"], "columns": {}}

    import pirlygenes.expression.normalize as norm

    monkeypatch.setattr(norm, "_oncoref_tpm_to_housekeeping_normalized", fake_oncoref)

    out, record = tpm_to_housekeeping_normalized(
        df,
        value_cols=["TPM_S1"],
        panel_ids=HK_IDS,
        panel_name="test_panel",
    )

    assert calls
    assert calls[0][1]["panel_ids"] == {s.split(".", 1)[0] for s in HK_IDS}
    assert calls[0][1]["panel_name"] == "test_panel"
    assert record["applied"] is True
    pd.testing.assert_series_equal(out["TPM_S1"], df["TPM_S1"] / 2.0)


def test_housekeeping_normalization_uses_oncoref_denominator():
    hk_values = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    df = _housekeeping_frame(hk_values)

    out, record = tpm_to_housekeeping_normalized(
        df,
        value_cols=["TPM_S1"],
        panel_ids=HK_IDS,
    )

    assert record["applied"] is True
    expected_denominator = record["columns"]["TPM_S1"]["denominator"]
    assert record["panel"] == "custom"
    assert record["panel_genes_present"] == len(HK_IDS)
    assert out.loc[df["Symbol"].eq("TARGET"), "TPM_S1"].iloc[0] == pytest.approx(
        100.0 / expected_denominator
    )


def test_housekeeping_normalization_rejects_symbol_panel():
    df = _housekeeping_frame([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    with pytest.raises(ValueError, match="panel_ids"):
        tpm_to_housekeeping_normalized(df, value_cols=["TPM_S1"], panel=["ACTB"])
