"""Offline guardrail for citation columns in the gene-set CSVs (#456/#469/#470).

The wrong-paper-PMID corruption found in #456 was remediated; this test keeps it
from regressing without needing the network. It checks the *form* of every
citation cell (a real PMID / PMCID / DOI / data accession / sanctioned
``curated_literature`` placeholder — never bare prose or a TODO/placeholder) and
that "Surname YYYY" source labels carry no impossible future year (the bug that
left FL labelled "...2026").

The authoritative topic-match audit against PubMed lives in
``scripts/audit_citations.py`` (networked, maintainer-run)."""
from __future__ import annotations

import csv
import datetime
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"

# {csv stem: citation column} — the columns whose values must be real citations.
CITATION_COLUMNS = {
    "cancer-fusions": "pmid",
    "cancer-type-registry": "source_pmid",
    "cancer-tmb": "pmid_doi",
    "cancer-apd1-response": "pmid_doi",
    "cancer-frameshift-burden": "pmid_doi",
    "therapy-response-signatures": "refs",
    "ffpe-sensitive-markers": "refs",
    "cancer-family-panels": "reference",
    "cancer-compartment-panels": "reference",
    "cancer-supertype-panels": "reference",
    "cancer-type-discriminators": "source",
    "rare-cancer-fusion-rules": "source",
}

# one allowed citation token
_TOKEN = re.compile(
    r"^(PMID:\d{6,9}"
    r"|PMCID:PMC\d+"
    r"|(DOI:)?10\.\d{4,9}/\S+"
    r"|(GSE|GSM|SRP|SRR|PRJNA)\d+"
    r"|E-MTAB-\d+"
    r"|curated_literature)$",
    re.IGNORECASE,
)
_FORBIDDEN = re.compile(r"\b(tbd|todo|placeholder|xxx|fixme|tba|n/?a)\b", re.I)
_THIS_YEAR = datetime.date.today().year


def _rows(stem):
    path = DATA / f"{stem}.csv"
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("stem,col", CITATION_COLUMNS.items())
def test_citation_cells_are_well_formed(stem, col):
    bad = []
    for i, row in enumerate(_rows(stem), start=2):
        cell = (row.get(col) or "").strip()
        if not cell:
            continue  # blank citation is allowed (not every row has a source)
        if _FORBIDDEN.search(cell):
            bad.append((i, cell, "forbidden placeholder"))
            continue
        for tok in re.split(r"[;\s]+", cell):
            if tok and not _TOKEN.match(tok):
                bad.append((i, cell, f"unparseable token {tok!r}"))
                break
    assert not bad, f"{stem}.{col} has malformed citations: {bad[:8]}"


def test_cancer_tmb_source_years_not_in_the_future():
    """A 'Surname YYYY' source label must not name a future year (the FL '2026'
    bug). Catches fabricated/typo year labels offline."""
    offenders = []
    for row in _rows("cancer-tmb"):
        m = re.search(r"((?:19|20)\d{2})", row.get("source", ""))
        if m and int(m.group(1)) > _THIS_YEAR:
            offenders.append((row.get("cancer_code"), row.get("source")))
    assert not offenders, f"future-year source labels: {offenders}"
