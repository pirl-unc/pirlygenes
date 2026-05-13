from pathlib import Path

from pirlygenes.load_dataset import get_data
from pirlygenes.qc_feature_groups import (
    QC_FEATURE_FILES,
    qc_class_for_ensembl_id,
    qc_class_for_symbol,
    qc_feature_ensembl_ids,
    qc_feature_groups,
    qc_feature_symbols,
    qc_feature_table,
)


EXPECTED_COLUMNS = {
    "Ensembl_Gene_ID",
    "Symbol",
    "qc_group",
    "qc_label",
    "ensembl_releases",
    "biotypes",
}


def test_qc_csvs_are_loadable_datasets():
    for group, filename in QC_FEATURE_FILES:
        dataset = Path(filename).stem
        df = get_data(dataset)
        assert EXPECTED_COLUMNS.issubset(df.columns)
        assert set(df["qc_group"]) == {group}
        assert df["Ensembl_Gene_ID"].astype(str).str.startswith("ENSG").all()


def test_qc_feature_table_combines_all_groups():
    table = qc_feature_table()
    groups = qc_feature_groups()

    expected_groups = {group for group, _ in QC_FEATURE_FILES}
    assert expected_groups.issubset(set(table["qc_group"]))
    assert expected_groups == set(groups)
    assert len(table) > 10_000


def test_qc_lookup_by_ensembl_id_strips_version():
    hit = qc_class_for_ensembl_id("ENSG00000251562.5")

    assert hit is not None
    assert hit.ensembl_gene_id == "ENSG00000251562"
    assert hit.symbol == "MALAT1"
    assert hit.group == "polyadenylation_bias_lncrna"


def test_qc_lookup_by_symbol_and_group_accessors():
    mt = qc_class_for_symbol("MT-CO1")
    hb = qc_class_for_symbol("HBB")

    assert mt is not None
    assert mt.group == "mt_dna"
    assert hb is not None
    assert hb.group == "hemoglobin"
    assert "ENSG00000244734" in qc_feature_ensembl_ids("hemoglobin")
    assert "HBA1" in qc_feature_symbols("hemoglobin")
