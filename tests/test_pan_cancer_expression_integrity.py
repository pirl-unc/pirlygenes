"""Integrity guards for pan-cancer-expression.csv (issue #379).

The HPA(nTPM) x TCGA(FPKM) merge can leave *phantom* rows: a gene under a
recently-reassigned Ensembl ID that has no match in the older GDC/STAR TCGA
annotation (all-FPKM-NaN), coexisting with the gene's canonical row that DOES
carry TCGA data. In a symbol-keyed aggregation the NaN phantom can wipe out the
canonical value (real + NaN -> NaN -> silently coerced to 0). These guards lock
out that collision class.

Note: a gene legitimately measured by HPA but absent from the TCGA annotation
keeps NaN FPKM on purpose (missing != zero) -- we do NOT coerce those to 0. The
invariant is only that no *canonical, TCGA-valued* gene is shadowed by an
all-NaN duplicate under the same symbol.
"""

import pandas as pd

from pirlygenes.load_dataset import get_data


def _df():
    return get_data("pan-cancer-expression.csv")


def test_no_duplicate_ensembl_ids():
    d = _df()
    dups = d["Ensembl_Gene_ID"][d["Ensembl_Gene_ID"].duplicated()].tolist()
    assert not dups, f"duplicate Ensembl IDs: {dups}"


def test_no_phantom_symbol_collision():
    """No symbol may have BOTH a TCGA-valued row and an all-TCGA-NaN duplicate
    row (the #379 collision: the NaN row corrupts symbol-keyed aggregation)."""
    d = _df()
    fpkm = [c for c in d.columns if c.startswith("FPKM_")]
    assert fpkm, "expected FPKM_<cohort> TCGA columns"
    all_nan = d[fpkm].isna().all(axis=1)
    valued_symbols = set(d.loc[~all_nan, "Symbol"].dropna())
    offenders = sorted(
        set(d.loc[all_nan, "Symbol"].dropna()) & valued_symbols
    )
    assert not offenders, (
        "symbols with a canonical TCGA-valued row shadowed by an all-NaN "
        f"duplicate (phantom collision): {offenders}"
    )
