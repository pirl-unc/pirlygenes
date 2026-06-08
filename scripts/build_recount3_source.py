#!/usr/bin/env python
"""Rebuild a spotty GEO cohort from recount3's coverage-complete gene sums.

recount3 (Gencode v26) re-quantified these SRA studies with the full gene
universe, recovering genes the authors' processed matrices dropped (e.g.
XAGE1A/XAGE1B in the GEP-NET cohort). Each source here keeps the SAME
``source_cohort`` tag as the original shard, so it replaces it in place.

See ``docs/recount3-integration.md`` and the
``pirlygenes/builders/recount3.py`` module docstring for the normalization
(coverage gene-sums → length-normalized TPM → shared clean-TPM).

    python scripts/build_recount3_source.py <source-id>
    python scripts/build_recount3_source.py --all
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from pirlygenes.builders import recount3 as rc3
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)

CACHE_DIR = Path.home() / ".cache" / "pirlygenes" / "recount3"


@dataclass
class Recount3Source:
    source_id: str
    srp: str
    source_cohort: str          # MUST match the existing shard tag (replace in place)
    source_project: str
    citation: str
    # (attrs, title) → cancer_code or None. None drops the sample.
    route: Callable[[dict, str], str | None]
    tumor_origin: str           # see TUMOR_ORIGIN_VALUES
    note: str                   # cohort-level provenance note (per code, n filled in)
    metastasis_site: str | None = None
    codes: list[str] = field(default_factory=list)
    # expected per-code sample count; a mismatch means recount3 metadata
    # strings drifted and routing silently dropped/added samples -> warn loudly.
    expected_n: dict[str, int] = field(default_factory=dict)


def _net_route(a: dict, title: str) -> str | None:
    origin = a.get("origin", "").lower()
    for needle, code in (
        ("small intest", "NET_MIDGUT"), ("ileum", "NET_MIDGUT"),
        ("jejunum", "NET_MIDGUT"), ("duodenum", "NET_MIDGUT"),
        ("pancrea", "NET_PANCREAS"), ("rect", "NET_RECTAL"),
    ):
        if needle in origin:
            return code
    return None


SOURCES: dict[str, Recount3Source] = {
    "gse98894-midnet": Recount3Source(
        source_id="gse98894-midnet", srp="SRP107025",
        source_cohort="GSE98894_ALVAREZ_2018_NET",
        source_project="Alvarez 2018 GEP-NET (GSE98894) — recount3 Gencode v26",
        citation="PMID 30013182 (Alvarez 2018)",
        route=_net_route, tumor_origin="metastasis", metastasis_site="liver",
        codes=["NET_MIDGUT", "NET_PANCREAS", "NET_RECTAL"],
        expected_n={"NET_MIDGUT": 81, "NET_PANCREAS": 113, "NET_RECTAL": 18},
        note="liver metastases of GEP-NET; primary site routed from recount3 "
             "sample origin attribute",
    ),
    "gse114922-mds": Recount3Source(
        source_id="gse114922-mds", srp="SRP149374",
        source_cohort="GSE114922_SHIOZAWA_2018",
        source_project="Shiozawa 2018 MDS (GSE114922) — recount3 Gencode v26",
        citation="GSE114922 (Shiozawa 2018)",
        route=lambda a, t: "MDS" if (
            a.get("cell type", "").startswith("CD34+")
            and a.get("disease status") == "Myelodysplastic Syndrome"
        ) else None,
        tumor_origin="primary", codes=["MDS"], expected_n={"MDS": 82},
        note="bone-marrow CD34+ HSPCs, disease status == Myelodysplastic "
             "Syndrome (healthy controls and non-CD34+ precursors dropped)",
    ),
    "gse118014-pannet": Recount3Source(
        source_id="gse118014-pannet", srp="SRP156049",
        source_cohort="GSE118014_ALVAREZ_2018",
        source_project="Alvarez 2018 primary PanNET (GSE118014) — recount3 Gencode v26",
        citation="GSE118014 (Alvarez 2018)",
        route=lambda a, t: "NET_PANCREAS", tumor_origin="primary", codes=["NET_PANCREAS"],
        expected_n={"NET_PANCREAS": 33},
        note="well-differentiated primary pancreatic neuroendocrine tumors",
    ),
    "gse120328-hl": Recount3Source(
        source_id="gse120328-hl", srp="SRP162356",
        source_cohort="GSE120328_LAMPRECHT_2018",
        source_project="Lamprecht 2018 cHL LCM (GSE120328) — recount3 Gencode v26",
        citation="PMID 30546079 (Lamprecht 2018)",
        route=lambda a, t: "HL" if str(t).endswith("_TU") else None,
        tumor_origin="primary", codes=["HL"], expected_n={"HL": 5},
        note="LCM-enriched HRS tumour cells (sample titles _TU; non-tumour "
             "_NTC controls dropped) — HRS-cell-specific, not whole-tumour",
    ),
}


def build(src: Recount3Source, summary_output: Path) -> int:
    print(f"[{src.source_id}] recount3 {src.srp} → {src.source_cohort}")
    annotation = rc3.fetch_gene_annotation(CACHE_DIR)
    gene_sums = rc3.fetch_gene_sums(src.srp, CACHE_DIR)
    meta = rc3.fetch_sample_metadata(src.srp, CACHE_DIR)

    # route every run, keep only the routable ones, aggregate runs→samples
    attrs = {
        str(r): rc3.parse_sample_attributes(s)
        for r, s in zip(meta["external_id"], meta["sample_attributes"])
    }
    title = dict(zip(meta["external_id"].astype(str), meta["sample_title"].astype(str)))
    run_code = {
        r: src.route(attrs[r], title[r]) for r in attrs
    }
    keep = {r for r, c in run_code.items() if c is not None}
    dropped = len(run_code) - len(keep)
    print(f"  routed {len(keep)}/{len(run_code)} runs ({dropped} unroutable, dropped)")
    sample_gs, sample_meta = rc3.aggregate_runs_to_samples(
        gene_sums, meta, keep_runs=keep,
    )
    # sample → code (all runs of a sample share a code; take the sample's first run)
    sample_code = {}
    for sample_id, row in sample_meta.iterrows():
        sample_code[sample_id] = run_code[str(row["external_id"])]
    counts = pd.Series(list(sample_code.values())).value_counts().to_dict()
    print(f"  samples by code: {counts}")
    # Guard against silent routing drift (recount3 metadata strings changing).
    for code, want in src.expected_n.items():
        got = counts.get(code, 0)
        if got != want:
            print(
                f"  WARNING: {code} routed n={got} but expected {want} — "
                "recount3 sample attributes may have changed; check routing."
            )

    tpm = rc3.gene_sums_to_tpm(sample_gs, annotation["bp_length"])
    clean = rc3.to_clean_tpm(tpm, annotation)
    symbol = annotation["Symbol"].reindex(tpm.index).fillna(
        pd.Series(tpm.index, index=tpm.index)
    )

    summaries, written = [], []
    for code in src.codes:
        cols = [s for s, c in sample_code.items() if c == code]
        if not cols:
            print(f"  {code}: no samples — skipping")
            continue
        out = pd.DataFrame({
            "Ensembl_Gene_ID": tpm.index, "Symbol": symbol.to_numpy(),
        })
        # Persist the per-code per-sample matrix (raw TPM) for medoids +
        # percentiles (uniform hook; write_per_sample canonicalises the stem).
        from pirlygenes import cohorts as _cohorts
        _cohorts.write_per_sample(out[["Ensembl_Gene_ID", "Symbol"]], tpm[cols],
                                  src.source_id, code)
        out["cancer_code"] = code
        out["source_cohort"] = src.source_cohort
        out["source_project"] = src.source_project
        out["source_version"] = (
            f"recount3 {src.srp} gene_sums (Gencode v26 coverage) → "
            f"exonic-length-normalized TPM → per-sample sum-to-1e6 → "
            f"tech-RNA zero; routed from recount3 sample attributes."
        )
        assign_stats(out, tpm[cols], clean[cols])
        out["processing_pipeline"] = (
            f"recount3_{src.srp.lower()}_gencode_v26_gene_sums_to_clean_tpm_v4"
        )
        out["tumor_origin"] = src.tumor_origin
        if src.metastasis_site:
            out["metastasis_site"] = src.metastasis_site
        out["notes"] = (
            f"{code} from {src.source_cohort} via recount3 {src.srp} "
            f"(Gencode v26, n={len(cols)}): {src.note}. {src.citation}."
        )
        out = round_stat_columns(out).reindex(columns=list(REFERENCE_COLUMNS))
        summaries.append(out)
        written.append(code)
        print(f"  {code}: n={len(cols)} → {len(out)} gene rows")

    if not summaries:
        print("  nothing to write")
        return 1
    upsert_to_shard(
        summary_output, pd.concat(summaries, ignore_index=True),
        source_cohort=src.source_cohort, cancer_codes=written,
    )
    print(f"  upserted shard {src.source_cohort}.csv.gz")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source_id", nargs="?", choices=sorted(SOURCES))
    p.add_argument("--all", action="store_true", help="build every recount3 source")
    p.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    args = p.parse_args()
    if not args.all and not args.source_id:
        p.error("give a source_id or --all")
    targets = list(SOURCES.values()) if args.all else [SOURCES[args.source_id]]
    rc = 0
    for src in targets:
        rc |= build(src, args.summary_output)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
