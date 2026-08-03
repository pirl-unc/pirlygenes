"""Regeneration contracts for the persisted pan-cancer rollup artifact."""

from pathlib import Path

import pandas as pd
from oncoref import data_bundle as oncoref_data_bundle

from scripts.generate_pan_cancer_expression_rollups import (
    SELECTED_SOURCE_SHARDS,
    _read_selected_source_rows,
)


def test_selected_rollup_shards_exist_in_pinned_oncoref_bundle():
    root = oncoref_data_bundle.find("cancer-reference-expression")
    assert root is not None

    missing = sorted(
        filename
        for filename in set(SELECTED_SOURCE_SHARDS.values())
        if not (Path(root) / filename).is_file()
    )
    assert missing == []


def test_consolidated_source_reader_keeps_only_requested_codes(tmp_path):
    path = tmp_path / "consolidated.csv.gz"
    pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG1", "ENSG2", "ENSG3"],
            "Symbol": ["A", "B", "C"],
            "cancer_code": ["COAD", "READ", "LUAD"],
            "TPM_median": [1.0, 2.0, 3.0],
            "n_samples": [3, 1, 515],
        }
    ).to_csv(path, index=False)

    out = _read_selected_source_rows(path, {"COAD", "READ"})

    assert out["cancer_code"].tolist() == ["COAD", "READ"]
