"""Audit citation PMIDs in the gene-set CSVs against PubMed (issue #456).

For every row's citation PMID, fetch the PubMed record (NCBI esummary) and check
it against the row's claim:

  * EXISTENCE   — does the PMID resolve at all?
  * AUTHOR/YEAR — when the row names a source as "Surname YYYY" (cancer-tmb),
                  does the PubMed first-author surname + year match? Accent- and
                  consortium-aware (TCGA/Network papers skip the author check).
                  This is the high-precision signal.
  * TOPIC       — does the title share a token with the row's gene / fusion /
                  cancer / oncology vocabulary? A *review* signal: it also flags
                  legitimately-shared pan-cancer landmark papers (TCGA copy-number,
                  Turajlic indel), so TOPIC_NONE means "eyeball it", not "wrong".

Emits ``citation-audit.csv`` and prints flagged rows grouped by severity. This is
the reproducible method behind #456; re-run after edits. Offline-friendly: the
only outbound calls are batched esummary.

    python scripts/audit_citations.py
    python scripts/audit_citations.py --file cancer-tmb
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# {stem: citation_column}.  ``source_col`` (optional) names a "Surname YYYY"
# label column to cross-check the cited paper's author/year against.
# cancer-type-registry, cancer-tmb, and cancer-apd1-response are re-exported
# from oncoref (pirlygenes#523/#541); they have no local CSV under
# pirlygenes/data/ and oncoref runs the citation audit for the tables it curates.
FILES = {
    "cancer-fusions": dict(col="pmid"),
    "cancer-frameshift-burden": dict(col="pmid_doi"),
    "therapy-response-signatures": dict(col="refs"),
    "ffpe-sensitive-markers": dict(col="refs"),
    "housekeeping-genes": dict(col="Reference"),
    "surface-proteins": dict(col="Source"),
    "cancer-family-panels": dict(col="reference"),
    "cancer-compartment-panels": dict(col="reference"),
    "cancer-supertype-panels": dict(col="reference"),
    "cancer-type-discriminators": dict(col="source"),
    "cancer-viral-antigens": dict(col=[
        "source",
        "association_source",
        "integration_source",
        "antigen_expression_source",
        "targetability_source",
    ]),
    "degenerate-subtype-pairs": dict(col="refs"),
    "fusion-surrogate-expression": dict(col="refs"),
    "fusion-expression-effects": dict(col="source"),
    "mutation-expression-effects": dict(col="source"),
    "rare-cancer-rna-surrogates": dict(col="source"),
    "rare-cancer-fusion-rules": dict(col="source"),
    "TCR-T-approved": dict(col="pmid_doi"),
    "TCR-T-trials": dict(col="pmid_doi"),
}

# row-identifying columns, first match wins
KEY_COLS = ["Symbol", "symbol", "cancer_code", "code", "contrast", "Compartment",
            "Supertype", "agent"]

ONCO_TERMS = {
    "cancer", "carcinoma", "sarcoma", "tumor", "tumour", "tumors", "tumours",
    "leukemia", "leukaemia", "lymphoma", "melanoma", "glioma", "glioblastoma",
    "medulloblastoma", "blastoma", "myeloma", "neoplasm", "neoplasms", "malignant",
    "oncology", "metastatic", "mutational", "mutation", "mutations", "burden",
    "neoantigen", "neoantigens", "immunotherapy", "checkpoint", "pd-1", "pd-l1",
    "fusion", "translocation", "rearranged", "rearrangement", "subgroup",
    "subgroups", "subtype", "subtypes", "genomic", "genome", "expression",
    "transcriptome", "transcriptomic", "atlas", "sequencing", "indel", "indels",
    "insertion", "deletion", "frameshift", "response", "objective", "adenocarcinoma",
    "squamous", "antigen", "receptor", "signaling", "signalling", "ffpe",
    "paraffin", "preservation",
}
CONSORTIA = ("tcga", "icgc", "pcawg", "genome atlas", "consortium", "network",
             "group", "initiative")
_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def _pmid(value: str) -> str | None:
    """First PMID in a citation cell. A DOI without an explicit PMID is not one."""
    s = str(value)
    if re.search(r"10\.\d{4}", s) and "PMID" not in s.upper():
        return None
    m = re.search(r"\b(\d{6,9})\b", s)
    return m.group(1) if m else None


def _clean_surname_year(source: str):
    """('Chalmers 2017')->('chalmers','2017'); ('Chordoma WGS')/('TCGA')->(None,None)."""
    s = source.strip()
    m = re.match(r"^([A-Za-zÀ-ÿ'’.\- ]+?)\s+((?:19|20)\d{2})$", s)
    if not m:
        return None, None
    return _strip_accents(m.group(1).lower().strip()), m.group(2)


def esummary(pmids: list[str]) -> dict:
    out = {}
    for i in range(0, len(pmids), 180):
        batch = pmids[i:i + 180]
        url = f"{ESUMMARY}?" + urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "json"})
        try:
            data = json.load(urllib.request.urlopen(url, timeout=30))["result"]
        except Exception as exc:  # noqa: BLE001
            print(f"  esummary batch failed: {exc}", file=sys.stderr)
            data = {}
        for pid in batch:
            rec = data.get(pid)
            if not rec or "error" in (rec or {}):
                out[pid] = None
                continue
            authors = [a.get("name", "") for a in rec.get("authors", [])]
            out[pid] = {
                "title": rec.get("title", ""),
                "authors_l": _strip_accents(" ".join(authors).lower()),
                "year": rec.get("pubdate", "")[:4],
            }
        time.sleep(0.34)
    return out


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(_strip_accents(text.lower())))


def audit_file(stem: str, spec: dict) -> list[dict]:
    path = DATA / f"{stem}.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    cols = rows[0].keys() if rows else []
    keycol = next((k for k in KEY_COLS if k in cols), None)
    citecols = spec["col"] if isinstance(spec["col"], list) else [spec["col"]]
    citecol_set = set(citecols)
    pmids = []
    for r in rows:
        r["_pmids"] = {}
        for citecol in citecols:
            pid = _pmid(r.get(citecol, ""))
            r["_pmids"][citecol] = pid
            if pid:
                pmids.append(pid)
    titles = esummary(sorted(set(pmids))) if pmids else {}

    results = []
    for r in rows:
        key = (r.get(keycol, "") if keycol
               else f"{r.get('gene_5prime','')}-{r.get('gene_3prime','')}")
        claim = " ".join(str(v) for k, v in r.items()
                         if k not in citecol_set and not k.startswith("_"))
        for citecol in citecols:
            result_file = stem if len(citecols) == 1 else f"{stem}.{citecol}"
            pid = r["_pmids"][citecol]
            raw = str(r.get(citecol, "")).strip()
            if not pid:
                # legitimate non-PMID citation forms: DOI, PMCID, a GEO/SRA/array
                # accession (data source), or an explicit curation placeholder.
                non_pmid = re.search(
                    r"10\.\d{4}|PMC\d|GSE\d|GSM\d|SRP\d|E-MTAB|curated_literature",
                    raw, re.I)
                verdict = "OK_NO_PMID" if (not raw or non_pmid) else "UNPARSEABLE"
                results.append(dict(file=result_file, key=key, pmid=raw,
                                    verdict=verdict, title="", claim=""))
                continue
            rec = titles.get(pid)
            if rec is None:
                results.append(dict(file=result_file, key=key, pmid=pid,
                                    verdict="NONEXISTENT", title="", claim=claim[:80]))
                continue
            verdicts = []
            exp_a, exp_y = _clean_surname_year(r.get(spec.get("source_col", ""), ""))
            consortium = any(c in (r.get(spec.get("source_col", ""), "") + " "
                                   + rec["authors_l"]).lower() for c in CONSORTIA)
            if not consortium:
                if exp_y and rec["year"] and exp_y != rec["year"]:
                    verdicts.append(f"YEAR {rec['year']}≠{exp_y}")
                if exp_a and exp_a not in rec["authors_l"]:
                    verdicts.append(f"AUTHOR≠{exp_a}")
            shared = (_tokens(claim) & _tokens(rec["title"])) | (
                _tokens(rec["title"]) & ONCO_TERMS)
            if not shared:
                verdicts.append("TOPIC_NONE")
            results.append(dict(file=result_file, key=key, pmid=pid,
                                verdict=";".join(verdicts) if verdicts else "ok",
                                title=rec["title"], claim=claim[:80]))
    return results


# high-precision = a near-certain wrong/dead citation; topic-only = review
def _is_hard(verdict: str) -> bool:
    return ("NONEXISTENT" in verdict or "UNPARSEABLE" in verdict
            or "AUTHOR" in verdict or "YEAR" in verdict)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", choices=list(FILES))
    args = ap.parse_args()
    targets = {args.file: FILES[args.file]} if args.file else FILES

    results = []
    for stem, spec in targets.items():
        print(f"auditing {stem} ...", flush=True)
        results.extend(audit_file(stem, spec))

    out = DATA.parent.parent / "citation-audit.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "key", "pmid", "verdict",
                                           "title", "claim"])
        w.writeheader()
        w.writerows(results)

    hard = [r for r in results if _is_hard(r["verdict"])]
    topic = [r for r in results if r["verdict"] == "TOPIC_NONE"]
    print(f"\n{len(results)} rows | {len(hard)} high-precision flags | "
          f"{len(topic)} topic-review flags -> {out}\n")
    print("== HIGH-PRECISION (likely wrong/dead — fix these) ==")
    for r in sorted(hard, key=lambda x: (x["file"], x["key"])):
        print(f"  [{r['verdict']}] {r['file']} {r['key']} PMID:{r['pmid']}")
        if r["title"]:
            print(f"      {r['title'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
