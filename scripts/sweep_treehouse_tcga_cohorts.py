#!/usr/bin/env python
"""TCGA-via-Treehouse sweep: every TCGA cohort in the PolyA compendium.

The Treehouse 25.01 PolyA compendium re-processes ~9,800 TCGA RNA-seq
samples through its own pipeline. Each cohort listed here is the
TCGA subset of that compendium for the matching disease label —
filtered with ``th_dataset_id.startswith("TCGA")`` so Treehouse-
internal samples for the same disease don't contaminate the cohort.

source_cohort tag is ``TREEHOUSE_POLYA_25_01_TCGA_SUBSET`` so a future
authoritative GDC STAR-counts build (plan milestone 3) can land
alongside under a different source_cohort tag.

Special handling not yet implemented (see audit's "Open questions"):
- GBM vs LGG split — Treehouse uses one "glioma" bucket. Both codes
  are skipped here pending a TSS-code-based split (or a join with
  TCGA's MAF / molecular-subtype table).
- SARC sub-histology (GIST, MYXLPS, WDLPS) — needs TCGA-SARC
  biospecimen sub-histology lookup; skipped here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pirlygenes.builders.treehouse import (
    TreehouseCohort,
    TreehouseRelease,
    run_sweep,
)


CACHE_ROOT = Path.home() / ".cache" / "pirlygenes" / "expression"

RELEASE = TreehouseRelease(
    source_id="treehouse-polya-25-01",
    source_cohort="TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
    source_project="Treehouse (TCGA samples)",
    release_label=(
        "Treehouse Tumor Compendium 25.01 PolyA, TCGA subset "
        "(hugo_log2tpm_58581genes_2025-02-27); downloaded from "
        "public.gi.ucsc.edu/~ekephart/public-data/"
    ),
    tpm_filename="Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv",
    clinical_filename="clinical_Treehouse-Tumor-Compendium-25.01-PolyA_20250131v1.tsv",
    cache_dir=CACHE_ROOT / "treehouse-polya-25-01",
    pipeline_prefix="treehouse_polya_25_01_tcga_subset_log2tpm_to_tpm",
    # 36 TCGA cohorts in one source — combined shard hit 99.47 MiB
    # after the v5.4 schema and was re-sharded per cancer_code via
    # scripts/reshard_tcga_subset.py. Keep this flag true so re-runs
    # of the sweep preserve the split.
    per_cancer_code_shards=True,
)


def _is_tcga(row: dict) -> bool:
    return str(row.get("th_dataset_id", "")).startswith("TCGA")


# (cancer_code, Treehouse disease label). One row per direct TCGA
# project mapping. GBM/LGG and SARC sub-histology splits are deferred.
TCGA_MAPPING: list[tuple[str, str]] = [
    ("ACC", "adrenocortical carcinoma"),
    ("BLCA", "bladder urothelial carcinoma"),
    ("BRCA", "breast invasive carcinoma"),
    ("CESC", "cervical & endocervical cancer"),
    ("CHOL", "cholangiocarcinoma"),
    ("COAD", "colon adenocarcinoma"),
    ("DLBC", "diffuse large B-cell lymphoma"),
    ("ESCA", "esophageal carcinoma"),
    ("HNSC", "head & neck squamous cell carcinoma"),
    ("KICH", "kidney chromophobe"),
    ("KIRC", "kidney clear cell carcinoma"),
    ("KIRP", "kidney papillary cell carcinoma"),
    ("LAML", "acute myeloid leukemia"),
    ("LIHC", "hepatocellular carcinoma"),
    ("LUAD", "lung adenocarcinoma"),
    ("LUSC", "lung squamous cell carcinoma"),
    ("MESO", "mesothelioma"),
    ("OV", "ovarian serous cystadenocarcinoma"),
    ("PAAD", "pancreatic adenocarcinoma"),
    ("PCPG", "pheochromocytoma & paraganglioma"),
    ("PRAD", "prostate adenocarcinoma"),
    ("READ", "rectum adenocarcinoma"),
    ("SARC", "leiomyosarcoma"),  # Note: TCGA-SARC includes multiple histologies; this
                                  # is the largest histology in the TCGA-SARC project. Real
                                  # TCGA-SARC umbrella build needs all histologies merged —
                                  # see audit "Open questions". Tagged here as a placeholder.
    ("SKCM", "skin cutaneous melanoma"),
    ("STAD", "stomach adenocarcinoma"),
    ("TGCT", "testicular germ cell tumor"),
    ("THCA", "thyroid carcinoma"),
    ("THYM", "thymoma"),
    ("UCEC", "uterine corpus endometrioid carcinoma"),
    ("UCS", "uterine carcinosarcoma"),
    ("UVM", "uveal melanoma"),
]


COHORTS = [
    TreehouseCohort(
        cancer_code=code,
        disease_label=label,
        sample_predicate=_is_tcga,
        extra_notes=(
            "TCGA subset only: filtered via "
            "`th_dataset_id.startswith('TCGA')` against Treehouse's "
            "compendium-wide disease label. Treehouse-internal "
            "samples with the same disease are excluded from this row."
        ),
        cache_stem=f"tcga_{code.lower()}",
    )
    for code, label in TCGA_MAPPING
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    cohorts = COHORTS
    if args.only:
        wanted = {c.strip() for c in args.only.split(",") if c.strip()}
        cohorts = [c for c in COHORTS if c.cancer_code in wanted]
        if not cohorts:
            raise SystemExit(f"--only={args.only!r} matched no cohorts")

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
