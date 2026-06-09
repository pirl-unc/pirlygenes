#!/usr/bin/env python
"""Split Treehouse TCGA "glioma" bucket into TCGA-GBM and TCGA-LGG.

Treehouse 25.01 PolyA labels all GBM and LGG samples with the single
disease string "glioma" (689 samples). The TCGA project assignment
lives at the case level in the GDC, not at the sample level in
Treehouse — so this script queries the GDC for the GBM/LGG case
membership and routes each Treehouse glioma sample to the right
cohort by joining on the TCGA case submitter_id (first three fields
of the aliquot barcode).

GDC API call is cached at
``~/.cache/pirlygenes/expression/treehouse-polya-25-01/derived/tcga_glioma_case_to_project.csv``
so the script is idempotent after the first run.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
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
    source_cohort="TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
    source_project="Treehouse (TCGA samples)",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA, TCGA-GBM/LGG split "
        "via GDC case→project lookup; "
        "downloaded from public.gi.ucsc.edu/~ekephart/public-data/"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=RELEASE_DIR,
    pipeline_prefix="treehouse_polya_25_01_tcga_glioma_split_log2tpm_to_tpm",
    # Shares source_cohort with the per-code TCGA_SUBSET sweep, so it must
    # shard per-code too — otherwise GBM/LGG land in a combined shard that
    # duplicates the per-code __GBM/__LGG shards.
    per_cancer_code_shards=True,
)


def _fetch_case_to_project_map(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-GBM", "TCGA-LGG"],
        },
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "submitter_id,project.project_id",
        "format": "JSON",
        "size": "2000",
    }
    url = "https://api.gdc.cancer.gov/cases?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as r:
        hits = json.load(r)["data"]["hits"]
    df = pd.DataFrame(
        [
            {"submitter_id": h["submitter_id"], "project_id": h["project"]["project_id"]}
            for h in hits
        ]
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def _project_predicate(project_id: str, case_to_project: dict[str, str]):
    """Predicate for the TCGA samples whose case maps to ``project_id``."""
    cases = {cid for cid, proj in case_to_project.items() if proj == project_id}
    return tcga_case_predicate(cases)


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

    map_cache = RELEASE_DIR / "derived" / "tcga_glioma_case_to_project.csv"
    case_map_df = _fetch_case_to_project_map(map_cache)
    case_to_project = dict(zip(case_map_df["submitter_id"], case_map_df["project_id"]))
    print(
        f"loaded GDC case→project map: "
        f"{(case_map_df['project_id']=='TCGA-GBM').sum()} GBM, "
        f"{(case_map_df['project_id']=='TCGA-LGG').sum()} LGG"
    )

    # Cohort definitions (code, stem, selection="gdc_project:TCGA-*") come from
    # the single registry in pirlygenes.cohorts (group "tcga_glioma").
    cohorts = []
    for c in cohorts_for_group("tcga_glioma"):
        project_id = c.selection.split(":", 1)[1]
        cohorts.append(
            TreehouseCohort(
                cancer_code=c.code,
                disease_label=c.disease_label,
                sample_predicate=_project_predicate(project_id, case_to_project),
                extra_notes=(
                    f"{project_id} split from Treehouse's compendium-wide "
                    "'glioma' bucket via GDC case→project lookup on the TCGA "
                    "submitter ID prefix."
                ),
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
