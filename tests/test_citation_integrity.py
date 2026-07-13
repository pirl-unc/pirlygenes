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
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"

# {csv stem: citation column(s)} — the columns whose values must be real citations.
CITATION_COLUMNS = {
    "cancer-fusions": "pmid",
    # cancer-type-registry, cancer-tmb, and cancer-apd1-response are re-exported
    # from oncoref (pirlygenes#523/#541); their citation format is validated
    # through the get_data accessor in test_reference_format, and oncoref owns
    # the content/citation validation for the tables it curates.
    "cancer-frameshift-burden": "pmid_doi",
    "therapy-response-signatures": "refs",
    "ffpe-sensitive-markers": "refs",
    "cancer-family-panels": "reference",
    "cancer-compartment-panels": "reference",
    "cancer-supertype-panels": "reference",
    "cancer-type-discriminators": "source",
    "cancer-viral-antigens": [
        "source",
        "association_source",
        "integration_source",
        "antigen_expression_source",
        "targetability_source",
    ],
    "degenerate-subtype-pairs": "refs",
    "fusion-surrogate-expression": "refs",
    "housekeeping-genes": "Reference",
    "lineage-genes": "reference",
    "cancer-key-genes": "source",
    "rare-cancer-fusion-rules": "source",
    "surface-proteins": "Source",
    "TCR-T-approved": "pmid_doi",
    "TCR-T-trials": "pmid_doi",
}

# one allowed citation token
_TOKEN = re.compile(
    r"^(PMID:\d{6,9}"
    r"|PMCID:PMC\d+"
    r"|(DOI:)?10\.\d{4,9}/\S+"
    r"|NCT\d{8}"
    r"|UMIN\d{9}"
    r"|(GSE|GSM|SRP|SRR|PRJNA)\d+"
    r"|E-MTAB-\d+"
    r"|DAILYMED_SETID:[0-9a-f-]+"
    r"|FDA_LABEL:[A-Z0-9_]+"
    r"|OPENFDA_FAERS"
    r"|ENSEMBL_RELEASE:\d+"
    r"|curated_literature)$",
    re.IGNORECASE,
)
_FORBIDDEN = re.compile(r"\b(tbd|todo|placeholder|xxx|fixme|tba|n/?a)\b", re.I)


def _rows(stem):
    path = DATA / f"{stem}.csv"
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("stem,col", CITATION_COLUMNS.items())
def test_citation_cells_are_well_formed(stem, col):
    bad = []
    cols = [col] if isinstance(col, str) else col
    for i, row in enumerate(_rows(stem), start=2):
        for one_col in cols:
            cell = (row.get(one_col) or "").strip()
            if not cell:
                continue  # blank citation is allowed (not every row has a source)
            if _FORBIDDEN.search(cell):
                bad.append((i, one_col, cell, "forbidden placeholder"))
                continue
            for tok in re.split(r"[;\s]+", cell):
                if tok and not _TOKEN.match(tok):
                    bad.append((i, one_col, cell, f"unparseable token {tok!r}"))
                    break
    assert not bad, f"{stem}.{col} has malformed citations: {bad[:8]}"


# NOTE: the "Surname YYYY source label must not name a future year" guard (the
# FL '2026' bug) moved to oncoref alongside cancer-tmb (pirlygenes#541); oncoref
# owns the year-sanity check for the TMB table it now curates.


def test_gene_list_provenance_manifest_covers_uncited_tables():
    required = {
        "cancer-type-genes",
        "cancer-driver-genes",
        "cancer-driver-variants",
        "gene-sets",
        "narrative-gene-sets",
        "stem-cell-marker-panels",
        "tme-markers",
        "lineage-genes",
        "ADC-approved",
        "CAR-T-approved",
        "bispecific-antibodies-approved",
        "multispecific-tcell-engager-trials",
        "radioligand-targets",
        "histone-genes",
        "hemoglobin-genes",
        "immune-receptor-segments",
        "ribosomal-protein-genes",
        "ribosomal-protein-pseudogenes",
        "rrna-and-pseudogenes",
        "numt-pseudogenes",
        "small-noncoding-rnas",
    }
    rows = _rows("gene-list-provenance")
    seen = {row["table_name"] for row in rows}
    assert required <= seen
    for row in rows:
        assert row["provenance_type"].strip()
        source = row.get("source_id", "").strip()
        if source:
            for tok in source.split(";"):
                assert _TOKEN.match(tok.strip()), (row["table_name"], source)


_GSE = re.compile(r"GSE\d+")


def test_expression_source_gse_accessions_are_structured():
    """Every expression source that names a GSE in its citation / URL / file
    metadata must expose that accession in the structured ``accession:`` field,
    and the field must match one of the GSEs it references (#470).

    Guards against GEO provenance that is only recoverable by ad hoc text
    extraction from ``citation`` / ``file_url`` / ``file_name``.
    """
    import yaml

    payload = yaml.safe_load((DATA / "expression_sources.yaml").read_text())
    bad = []
    for src in payload.get("sources") or []:
        text = " ".join(
            str(src.get(k, "") or "")
            for k in ("citation", "url", "file_url", "file_name")
        )
        referenced = set(_GSE.findall(text))
        if not referenced:
            continue
        accession = str(src.get("accession", "") or "").strip()
        if not accession:
            bad.append((src.get("id"), "GSE in text but no accession:", sorted(referenced)))
        elif not (set(_GSE.findall(accession)) & referenced):
            bad.append((src.get("id"), f"accession {accession!r} not among {sorted(referenced)}"))
    assert not bad, f"expression_sources.yaml GEO accessions not structured: {bad}"
