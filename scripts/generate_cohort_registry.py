#!/usr/bin/env python
"""Generate ``pirlygenes/data/cohort-registry.csv`` (#296).

The first-class cohort vocabulary: one row per ``cohort_id``, the single
authority a consumer validates ``source_cohort`` against — including the
computed aggregates (``COMPUTED_PAN_SARCOMA``) and the literature-curated
placeholder, which are NOT in ``available_cancer_expression_references()``.

Derived (re-runnable) from:
  - the packaged reference manifest (per-cohort sample counts, # codes,
    source_version provenance) — bundled shards;
  - oncoref's cancer-type registry ``source_cohort`` column (adds the
    computed/curated cohorts that have no shard) — read via
    ``cancer_type_registry()``, not a packaged CSV;
  - the computed aggregates (``cohort_aggregates``) for the member lists +
    provenance of the shardless computed unions (COMPUTED_PAN_SARCOMA,
    COMPUTED_COLORECTAL).

Columns: cohort_id, prefix, kind, source_project, assay, n_samples, n_codes,
is_computed, member_cohorts, provenance.

    python scripts/generate_cohort_registry.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

DATA = Path("pirlygenes/data")

# kind / source_project / default assay by cohort_id prefix. The literal
# underscore-prefix token is the `prefix`; `kind` is the high-level pipeline
# family (what the cross-source combining rule keys off).
def _classify(cid: str):
    if cid.startswith("COMPUTED_"):
        return "computed", "Computed aggregate", "computed-aggregate"
    if cid == "LITERATURE_CURATED":
        return "curated", "Literature-curated", "literature-curated"
    if cid.startswith("TREEHOUSE_"):
        proj = ("TCGA (Treehouse-reprocessed)" if "_TCGA_" in cid
                else "Treehouse")
        return "treehouse", proj, "bulk RNA-seq"
    if cid.startswith("TARGET_"):
        return "target", "TARGET", "bulk RNA-seq"
    if cid.startswith("SCLC_UCOLOGNE"):
        return "ucologne", "University of Cologne", "bulk RNA-seq"
    if cid.startswith("DRMETRICS"):
        return "geo", "GEO (DR-metrics / Alcala LNEN)", "bulk RNA-seq"
    if cid.startswith("GSE"):
        return "geo", "GEO", "bulk RNA-seq"
    if cid.startswith("BEATAML"):
        return "beataml", "BeatAML (OHSU)", "bulk RNA-seq"
    if cid.startswith("MMRF"):
        return "mmrf", "MMRF CoMMpass", "bulk RNA-seq"
    if cid.startswith("CLLMAP"):
        return "cllmap", "CLL-map", "bulk RNA-seq"
    if cid.startswith("CGCI"):
        return "cgci", "CGCI", "bulk RNA-seq"
    if cid.startswith("UNC"):
        return "unc", "UNC NUTM1 case series", "bulk RNA-seq"
    return "other", "", "bulk RNA-seq"


# Cohorts whose assay is NOT bulk RNA-seq (override the default).
_MICROARRAY = {"GSE32662_PRINGLE_2012_MTC", "GSE30929_SINGER_2007_LPS"}


def main():
    # 1. per-cohort stats from the bundled shards
    frames = []
    for f in glob.glob(str(DATA / "cancer-reference-expression" / "*.csv.gz")):
        cols = ["cancer_code", "source_cohort", "n_samples", "source_project",
                "source_version"]
        df = pd.read_csv(f, dtype=str)
        frames.append(df[[c for c in cols if c in df.columns]].drop_duplicates())
    man = pd.concat(frames, ignore_index=True).drop_duplicates()
    man["n_samples"] = pd.to_numeric(man["n_samples"], errors="coerce").fillna(0)

    # one (cohort, code) sample count is constant across genes; dedup to it
    per = man.drop_duplicates(["source_cohort", "cancer_code"])
    by_cohort = per.groupby("source_cohort").agg(
        n_samples=("n_samples", "sum"),
        n_codes=("cancer_code", "nunique"),
        source_version=("source_version", "first"),
    )

    # 2. full cohort universe = shards ∪ registry source_cohort column. The
    # cancer-type registry is owned by oncoref now (the packaged
    # cancer-type-registry.csv was removed in the oncoref migration), so read it
    # through the accessor rather than a deleted CSV.
    import sys
    sys.path.insert(0, ".")
    from pirlygenes.gene_sets_cancer import cancer_type_registry, cohort_aggregates
    reg = cancer_type_registry()
    universe = set(by_cohort.index) | set(reg["source_cohort"].dropna().astype(str))

    # 3. computed-aggregate members, keyed by the source_cohort id the registry
    # assigns to each computed union (CRC -> COMPUTED_COLORECTAL, SARC ->
    # COMPUTED_PAN_SARCOMA). Members + provenance come from cohort_aggregates so
    # these shardless rows regenerate instead of being hand-maintained.
    agg = cohort_aggregates()
    computed_members = {
        "COMPUTED_PAN_SARCOMA": (
            agg.get("SARC", []),
            "pan-sarcoma grand union of all SARC_* histology atoms",
        ),
        "COMPUTED_COLORECTAL": (
            agg.get("CRC", []),
            "colorectal union of COAD+READ (rectum curated as colorectal)",
        ),
    }

    rows = []
    for cid in sorted(universe):
        prefix = cid.split("_")[0]
        kind, proj, assay = _classify(cid)
        if cid in _MICROARRAY:
            assay = "microarray"
        is_computed = kind == "computed"
        if cid in by_cohort.index:
            n_samples = int(by_cohort.loc[cid, "n_samples"])
            n_codes = int(by_cohort.loc[cid, "n_codes"])
            prov = str(by_cohort.loc[cid, "source_version"] or "")
        else:
            n_samples, n_codes, prov = 0, 0, ""
        member_cohorts = ""
        if cid in computed_members:
            members, prov = computed_members[cid]
            member_cohorts = ";".join(members)
            n_codes = len(members)
        elif cid == "LITERATURE_CURATED":
            prov = "registry entry without a built expression matrix"
        rows.append({
            "cohort_id": cid, "prefix": prefix, "kind": kind,
            "source_project": proj, "assay": assay,
            "n_samples": n_samples, "n_codes": n_codes,
            "is_computed": is_computed, "member_cohorts": member_cohorts,
            "provenance": prov,
        })

    out = pd.DataFrame(rows).sort_values(["kind", "cohort_id"]).reset_index(drop=True)
    out.to_csv(DATA / "cohort-registry.csv", index=False)
    print(f"wrote cohort-registry.csv: {len(out)} cohorts "
          f"({out['kind'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
