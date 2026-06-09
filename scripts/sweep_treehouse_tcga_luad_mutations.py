#!/usr/bin/env python
"""Split TCGA-LUAD by driver-mutation class: LUAD_EGFR / KRAS / STK11.

Uses cBioPortal study ``luad_tcga_pan_can_atlas_2018`` mutation
profile (TCGA Pan-Cancer Atlas MAF) for the three driver classes.
LUAD_STK11 includes samples with STK11 OR KEAP1 mutation (they're
the canonical co-altered "K/L" axis with distinct immune-cold
expression signatures).

Each mutation-positive sample → corresponding subtype cohort. A
sample can be in multiple cohorts (e.g. KRAS + STK11 co-mutated);
that's intentional — these are not mutually-exclusive subtypes,
they're driver-defined cohorts.

Three cohorts under
``source_cohort=TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT``.

This script uses "any coding mutation in the gene" as the positivity
call — overcounts slightly vs "hotspot mutations only" but matches
cBioPortal's default. For more rigorous calls (e.g. EGFR L858R /
ex19del only, KRAS G12X only), overlay the MAF directly.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from pirlygenes.builders.treehouse import (
    TreehouseCohort,
    TreehouseRelease,
    run_sweep,
    tcga_case_predicate,
)
from pirlygenes.cohorts import cohorts_for_group


CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"
RELEASE_DIR = CACHE_ROOT / "treehouse-polya-25-01"

RELEASE = TreehouseRelease(
    source_id="treehouse-polya-25-01",
    source_cohort="TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
    source_project="Treehouse (TCGA-LUAD) × cBioPortal MAF",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA TCGA-LUAD samples "
        "× cBioPortal luad_tcga_pan_can_atlas_2018 mutation calls"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_tcga_luad_mut_log2tpm_to_tpm",
)


GENE_ID_BY_SYMBOL = {"EGFR": 1956, "KRAS": 3845, "STK11": 6794, "KEAP1": 9817}


def _fetch_cbioportal_luad_mutations(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    study = "luad_tcga_pan_can_atlas_2018"
    profile_id = f"{study}_mutations"
    url = (
        f"https://www.cbioportal.org/api/molecular-profiles/{profile_id}"
        f"/mutations/fetch?sampleListId={study}_all"
    )
    body = json.dumps(
        {
            "entrezGeneIds": list(GENE_ID_BY_SYMBOL.values()),
            "sampleListId": f"{study}_all",
        }
    )
    req = urllib.request.Request(
        url, data=body.encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        mutations = json.load(r)
    sym_by_id = {v: k for k, v in GENE_ID_BY_SYMBOL.items()}
    rows = []
    for m in mutations:
        gene = sym_by_id.get(m.get("entrezGeneId"))
        if gene:
            rows.append({"sampleId": m["sampleId"], "gene": gene})
    df = pd.DataFrame(rows).drop_duplicates()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _cases_with_any_mutation(mut: pd.DataFrame, genes: list[str]) -> set[str]:
    """Return TCGA case submitter_ids with any mutation in the listed genes."""
    sub = mut[mut["gene"].isin(genes)]
    # sampleId like TCGA-05-4244-01 → case_id = first 3 fields
    case_ids = sub["sampleId"].astype(str).str.split("-").str[:3].str.join("-")
    return set(case_ids)


# Cohort definitions (code, stem, selection="mutation:<gene[,gene]>") come from
# the single registry in pirlygenes.cohorts (group "tcga_luad_mut"). Notes are
# build-specific prose, kept here keyed by code.
_NOTE_BY_CODE = {
    "LUAD_EGFR": "EGFR-mutant: any cBioPortal-called EGFR coding mutation",
    "LUAD_KRAS": "KRAS-mutant: any cBioPortal-called KRAS coding mutation",
    "LUAD_STK11": (
        "STK11/KEAP1-mutant: any cBioPortal-called STK11 OR KEAP1 coding "
        "mutation (canonical K/L axis; immune-cold subset)."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    cache = RELEASE_DIR / "derived" / "cbioportal_luad_mutations.csv"
    mut = _fetch_cbioportal_luad_mutations(cache)
    counts_by_gene = mut.groupby("gene")["sampleId"].nunique().to_dict()
    print(f"cBioPortal LUAD mutation counts: {counts_by_gene}")

    cohorts = []
    for c in cohorts_for_group("tcga_luad_mut"):
        genes = c.selection.split(":", 1)[1].split(",")
        cases = _cases_with_any_mutation(mut, genes)
        if not cases:
            print(f"skipping {c.code}: no cases")
            continue
        cohorts.append(
            TreehouseCohort(
                cancer_code=c.code,
                disease_label=c.disease_label,
                sample_predicate=tcga_case_predicate(cases),
                extra_notes=_NOTE_BY_CODE[c.code],
                cache_stem=c.stem,
            )
        )

    run_sweep(
        RELEASE,
        cohorts,
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        refresh_cache=args.refresh_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
