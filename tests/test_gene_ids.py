#!/usr/bin/env python3

from pyensembl import ensembl_grch38

from pirlygenes import load_all_dataframes


def _split_ids(value):
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(";") if token.strip()]


def _collect_ensembl_gene_ids(df):
    id_columns = [
        c for c in df.columns if ("Ensembl_Gene_ID" in c or c.endswith("_Gene_IDs"))
    ]
    result = set()
    for col in id_columns:
        for value in df[col]:
            for token in _split_ids(value):
                if token.startswith("ENSG"):
                    result.add(token)
    return result


def test_all_gene_ids_resolve_in_ensembl():
    """
    Validate that all Ensembl gene IDs in packaged CSVs resolve via pyensembl.

    This intentionally supports multiple schema styles (singular/plural target columns).
    """
    all_ids = set()
    for _, df in load_all_dataframes():
        all_ids.update(_collect_ensembl_gene_ids(df))

    assert all_ids, "No Ensembl gene IDs were found in packaged CSVs."

    missing = []
    for gene_id in sorted(all_ids):
        try:
            gene = ensembl_grch38.gene_by_id(gene_id)
        except Exception:
            gene = None
        if gene is None:
            missing.append(gene_id)

    assert not missing, "Unresolvable Ensembl gene IDs: %s" % ", ".join(missing)
