#!/usr/bin/env python3

from pyensembl.shell import collect_all_installed_ensembl_releases

from pirlygenes import load_all_dataframes

_human_genomes = sorted(
    [g for g in collect_all_installed_ensembl_releases()
     if g.species.latin_name == "homo_sapiens"],
    reverse=True,
    key=lambda g: g.release,
)


def _resolve_in_any_release(gene_id):
    for genome in _human_genomes:
        try:
            gene = genome.gene_by_id(gene_id)
            if gene is not None:
                return True
        except Exception:
            pass
    return False


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


# Large external datasets contain IDs from newer Ensembl releases that may
# not be installed in CI.  The HPA cell-type atlas uses Ensembl 112+ gene
# IDs (e.g. ENSG00000283886), which don't resolve in older pyensembl releases.
_SKIP_DATASETS = {"surface-proteins", "pan-cancer-expression", "hpa-cell-type-expression"}


def test_all_gene_ids_resolve_in_ensembl():
    """
    Validate that all Ensembl gene IDs in curated packaged CSVs resolve via pyensembl.

    This intentionally supports multiple schema styles (singular/plural target columns).
    Checks all installed human Ensembl releases, not just the default GRCh38.
    """
    all_ids = set()
    for name, df in load_all_dataframes():
        if any(skip in name for skip in _SKIP_DATASETS):
            continue
        all_ids.update(_collect_ensembl_gene_ids(df))

    assert all_ids, "No Ensembl gene IDs were found in packaged CSVs."

    missing = []
    for gene_id in sorted(all_ids):
        if not _resolve_in_any_release(gene_id):
            missing.append(gene_id)

    assert not missing, "Unresolvable Ensembl gene IDs: %s" % ", ".join(missing)
