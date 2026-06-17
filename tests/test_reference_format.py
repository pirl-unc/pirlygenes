"""Format guard for the structured citation columns (regression guard for #456).

#456 found ~30% of citations were real-but-wrong-paper or otherwise malformed
(machine-generated, never verified). A topic-match audit can't run in CI (no
PubMed network), but this catches the cheap, common corruption: a citation cell
that isn't a well-formed reference token. Each non-empty value in the structured
citation columns below must be one or more of:

    PMID:<digits>  |  DOI:<...>  |  10.<...> (bare DOI)  |  GSE<digits> (GEO source)

joined by ';' when a row cites more than one. Free-text bibliographic columns
(housekeeping-genes.Reference, TCR-T-*.Reference) use a different, legitimate
prose style and are intentionally out of scope.
"""

import re

import pandas as pd

from pirlygenes.load_dataset import get_data

# (dataset name, column) for the columns remediated under #456.
STRUCTURED_CITATION_COLUMNS = [
    ("cancer-fusions", "pmid"),
    ("cancer-type-registry", "source_pmid"),
    ("cancer-tmb", "pmid_doi"),
    ("cancer-apd1-response", "pmid_doi"),
    ("cancer-frameshift-burden", "pmid_doi"),
    ("cancer-family-panels", "reference"),
    ("cancer-compartment-panels", "reference"),
    ("cancer-supertype-panels", "reference"),
    ("cancer-type-discriminators", "source"),
]

_TOKEN = re.compile(r"^(PMID:\d{6,9}|DOI:\S+|10\.\d{4,}/\S+|GSE\d+)$")


def _well_formed(value: str) -> bool:
    value = str(value).strip()
    if not value:
        return True  # empty = uncited; allowed (a wrong citation is the bug, not none)
    return all(_TOKEN.match(tok.strip()) for tok in value.split(";") if tok.strip())


def test_structured_citation_columns_are_well_formed():
    """Every non-empty citation cell is a PMID/DOI/GSE token (or ';'-joined set).
    Catches a fabricated bare number, a stray non-citation string, or a PMC id
    pasted in place of a PMID — the corruption #456 fixed."""
    failures = []
    for name, col in STRUCTURED_CITATION_COLUMNS:
        df = get_data(name, copy=False)
        assert col in df.columns, f"{name}: missing column {col!r}"
        for value in df[col].dropna().astype(str).unique():
            if not _well_formed(value):
                failures.append(f"{name}::{col} = {value!r}")
    assert not failures, "malformed citation cells:\n  " + "\n  ".join(failures)
