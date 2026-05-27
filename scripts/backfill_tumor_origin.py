#!/usr/bin/env python
"""Backfill ``tumor_origin`` / ``metastasis_site`` on existing shards.

The v5.4 schema adds primary-vs-metastasis annotation. Every shard in
``pirlygenes/data/cancer-reference-expression/`` already has the two
new columns (via the upsert helper's reindex), but legacy shards have
NaN for both. This script walks each shard, looks up its
``source_cohort`` in a curated table below, and writes the right
values.

Rules baked in here (audited 2026-05-27):

  primary:
    BEATAML_OHSU_2022          AML BM/PB diagnostic
    CGCI_BLGSP                 Burkitt primary tumor
    CLLMAP_2022                CLL PBMC
    DRMETRICS_ALCALA_2019_LNEN LCNEC / carcinoid primary
    GSE100026_DING_2017        CML PBMC
    GSE114922_SHIOZAWA_2018    MDS BM
    GSE118014_ALVAREZ_2018     PANNET primary
    GSE120328_LAMPRECHT_2018   HL LCM tumor cells
    GSE142334_FL_TFL_2021      FL primary biopsy
    GSE171811_ECCITE_CTCL      CTCL skin lesions
    GSE241095_KS_SKIN_2023     KS skin lesions
    GSE248751_HUMAN_CCS_2023   CCS primary tumors
    GSE271664_BODOR_2025       MCL primary lymph
    GSE283710_WASHU_2024       MPN BM
    GSE294016_BARTL_2025_SGC   salivary primary
    GSE299759_MEIJER_2026      chondrosarcoma primary
    GSE328026_PECOMA_2026      PEComa primary
    GSE75885_DELESPAUL_2017    STS primary
    MMRF_COMMPASS              MM BM CD138+
    SCLC_UCOLOGNE_2015         SCLC primary
    SCLC_UCOLOGNE_2015_TF_DOMINANCE  subtype split of SCLC primary
    TARGET_ALL_2018            pediatric ALL diagnostic
    TARGET_RT_2017             rhabdoid primary
    TARGET_WT_2015             Wilms primary

  metastasis (already set, skipped here):
    GSE98894_ALVAREZ_2018_NET  liver metastases of GEP-NET

  mixed (primary + recurrence + metastasis without per-sample split):
    TARGET_NBL_2018            includes BM-metastatic NBL samples
    TREEHOUSE_POLYA_25_01      pediatric polyA compendium (mixed stages)
    TREEHOUSE_RIBOD_25_01      pediatric ribodepleted compendium
    TREEHOUSE_POLYA_25_01_TCGA_SUBSET           TCGA-via-Treehouse (mostly
                                                primary but unaudited)
    TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50       BRCA subtype split — TCGA
                                                BRCA includes ~7 mets
    TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV         HNSC subtype split (primary)
    TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT         LUAD mutation split (primary)
    TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS  MBL subgroups (primary)

  The TCGA-derived Treehouse subsets are *largely* primary (TCGA
  policy is one primary tumor per case, with a handful of metastasis
  samples in SKCM / BRCA / TGCT / SARC etc.). We mark them
  ``mixed`` because we haven't done the per-sample audit; a follow-up
  pass can downgrade well-curated ones to ``primary``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pirlygenes.expression.stats import REFERENCE_COLUMNS

PRIMARY_SOURCES: frozenset[str] = frozenset({
    "BEATAML_OHSU_2022",
    "CGCI_BLGSP",
    "CLLMAP_2022",
    "DRMETRICS_ALCALA_2019_LNEN",
    "GSE100026_DING_2017",
    "GSE114922_SHIOZAWA_2018",
    "GSE118014_ALVAREZ_2018",
    "GSE120328_LAMPRECHT_2018",
    "GSE142334_FL_TFL_2021",
    "GSE171811_ECCITE_CTCL",
    "GSE241095_KS_SKIN_2023",
    "GSE248751_HUMAN_CCS_2023",
    "GSE271664_BODOR_2025",
    "GSE283710_WASHU_2024",
    "GSE294016_BARTL_2025_SGC",
    "GSE299759_MEIJER_2026",
    "GSE328026_PECOMA_2026",
    "GSE75885_DELESPAUL_2017",
    "MMRF_COMMPASS",
    "SCLC_UCOLOGNE_2015",
    "SCLC_UCOLOGNE_2015_TF_DOMINANCE",
    "TARGET_ALL_2018",
    "TARGET_RT_2017",
    "TARGET_WT_2015",
})

MIXED_SOURCES: frozenset[str] = frozenset({
    "TARGET_NBL_2018",
    "TREEHOUSE_POLYA_25_01",
    "TREEHOUSE_RIBOD_25_01",
    "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
    "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
    "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV",
    "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
    "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS",
})

# Sources we EXPLICITLY skip — they set their own tumor_origin in the
# builder and any rewrite here would clobber it.
SELF_ANNOTATED_SOURCES: frozenset[str] = frozenset({
    "GSE98894_ALVAREZ_2018_NET",
})


def _classify(source_cohort: str) -> tuple[str | None, str | None]:
    """Return (tumor_origin, metastasis_site) for the source_cohort."""
    if source_cohort in PRIMARY_SOURCES:
        return ("primary", None)
    if source_cohort in MIXED_SOURCES:
        return ("mixed", None)
    if source_cohort in SELF_ANNOTATED_SOURCES:
        return (None, None)
    return (None, None)


def backfill_shard(path: Path, *, dry_run: bool = False) -> dict[str, int]:
    df = pd.read_csv(path, low_memory=False)
    if "source_cohort" not in df.columns:
        return {"skipped_no_source_cohort": 1}
    # Each shard holds one source_cohort by construction.
    sc_vals = df["source_cohort"].dropna().unique()
    if len(sc_vals) != 1:
        return {"skipped_multi_source_cohort": 1}
    source_cohort = str(sc_vals[0])
    new_origin, new_metsite = _classify(source_cohort)

    if new_origin is None:
        if source_cohort in SELF_ANNOTATED_SOURCES:
            return {
                "preserved_self_annotated": 1,
                "source_cohort": source_cohort,
            }
        return {"skipped_unclassified": 1, "source_cohort": source_cohort}

    # Don't overwrite values that the builder already set explicitly.
    has_origin = (
        "tumor_origin" in df.columns
        and df["tumor_origin"].notna().any()
    )
    if has_origin:
        return {
            "skipped_already_set": 1,
            "source_cohort": source_cohort,
            "existing_origin": str(df["tumor_origin"].dropna().iloc[0]),
        }

    df["tumor_origin"] = new_origin
    df["metastasis_site"] = new_metsite if new_metsite is not None else pd.NA
    df = df.reindex(columns=list(REFERENCE_COLUMNS))

    if not dry_run:
        df.to_csv(path, index=False, compression="gzip")
    return {
        "updated": len(df),
        "source_cohort": source_cohort,
        "tumor_origin": new_origin,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shards = sorted(args.shard_dir.glob("*.csv.gz"))
    print(f"backfilling {len(shards)} shard(s) in {args.shard_dir}...")
    updated = preserved = skipped = unclassified = 0
    for shard in shards:
        result = backfill_shard(shard, dry_run=args.dry_run)
        tag = ", ".join(f"{k}={v}" for k, v in result.items())
        print(f"  {shard.name}: {tag}")
        if "updated" in result:
            updated += 1
        elif "preserved_self_annotated" in result:
            preserved += 1
        elif "skipped_unclassified" in result:
            unclassified += 1
        else:
            skipped += 1
    print(
        f"done. updated={updated}, preserved={preserved}, "
        f"skipped={skipped}, unclassified={unclassified} "
        f"(dry_run={args.dry_run})"
    )
    if unclassified:
        print(
            "  unclassified shards need a manual entry in "
            "PRIMARY_SOURCES / MIXED_SOURCES / SELF_ANNOTATED_SOURCES."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
