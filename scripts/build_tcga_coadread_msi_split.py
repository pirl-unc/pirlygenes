#!/usr/bin/env python
"""Split the TCGA-COAD / TCGA-READ per-sample cohorts into MSI-H vs MSS.

Microsatellite-instability status is the colorectal immune-hot/cold axis with
the largest known anti-PD-1 differential (MSI-H ORR ~45% vs MSS ~0%). This joins
cBioPortal's per-sample MSIsensor score (study ``coadread_tcga_pan_can_atlas_2018``,
SAMPLE attribute ``MSI_SENSOR_SCORE``; >= 10 = MSI-H, the Niu 2014 / TCGA cutoff)
onto the TCGA COAD/READ samples already in the Treehouse PolyA per-sample
parquets, by case submitter-id — the same per-case-join pattern as the
HNSC-HPV and BRCA-PAM50 splits.

Reuses the existing ``tcga_coad`` / ``tcga_read`` per-sample TPM (no need to
re-run the 5.8 GB compendium sweep); writes ``tcga_coad_msi`` / ``tcga_coad_mss``
/ ``tcga_read_msi`` / ``tcga_read_mss`` per-sample parquets. COAD vs READ comes
from the originating Treehouse cohort (not cBioPortal CANCER_TYPE_DETAILED, whose
"mucinous" category is anatomically ambiguous).

Run:  python scripts/build_tcga_coadread_msi_split.py
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

from pirlygenes import cohorts as _cohorts
from pirlygenes.downloads import source_cache_dir
from pirlygenes.expression.stats import build_reference_rows, upsert_to_shard

SOURCE_ID = "treehouse-polya-25-01"
SUMMARY_COHORT = "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI"
MSI_H_CUTOFF = 10.0  # MSIsensor score >= 10 = MSI-H (Niu 2014 / TCGA convention)


def _summary_row(gene_table, values, code: str):
    return build_reference_rows(
        gene_table, values,
        cancer_code=code,
        source_cohort=SUMMARY_COHORT,
        source_project="Treehouse (TCGA-COADREAD) × cBioPortal MSI",
        source_version=(
            "Treehouse 25.01 PolyA TCGA-COAD/READ × cBioPortal "
            "coadread_tcga_pan_can_atlas_2018 MSIsensor score (>=10 = MSI-H)"),
        processing_pipeline=(
            "treehouse_polya_25_01_tcga_coadread_msi_log2tpm_to_tpm_clean_tpm_v4"),
        notes=(
            f"{code}: TCGA {code.split('_')[0]} samples in Treehouse 25.01 "
            "PolyA split by cBioPortal MSIsensor score (>=10 = MSI-H)."),
        tumor_origin="primary",
    )


def _fetch_msi(cache_path: Path) -> dict[str, float]:
    """{case submitter_id -> MSIsensor score} from cBioPortal (cached)."""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        url = (
            "https://www.cbioportal.org/api/studies/"
            "coadread_tcga_pan_can_atlas_2018/clinical-data"
            "?clinicalDataType=SAMPLE&attributeId=MSI_SENSOR_SCORE"
            "&projection=SUMMARY&pageSize=4000"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        df = pd.DataFrame([{"sampleId": d["sampleId"], "score": d["value"]}
                           for d in data])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    out: dict[str, float] = {}
    for sid, score in zip(df["sampleId"].astype(str), df["score"]):
        case = "-".join(str(sid).split("-")[:3])
        try:
            out[case] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-output", type=Path,
                    default=Path("pirlygenes/data/cancer-reference-expression"))
    args = ap.parse_args()
    derived = source_cache_dir(SOURCE_ID) / "derived"
    msi = _fetch_msi(derived / "cbioportal_coadread_msi.csv")
    n_h = sum(1 for v in msi.values() if v >= MSI_H_CUTOFF)
    print(f"cBioPortal MSI: {len(msi)} cases scored; {n_h} MSI-H / "
          f"{len(msi) - n_h} MSS (cutoff {MSI_H_CUTOFF})", flush=True)

    targets = {c.code: c for c in _cohorts.cohorts_for_group("tcga_coadread_msi")}
    for parent in ("COAD", "READ"):
        src = _cohorts.cohorts_for_source(SOURCE_ID)[parent]
        df = _cohorts.read_per_sample(src)
        gene_table = df[list(_cohorts.ID_COLS)]
        sample_cols = _cohorts.sample_columns(df)

        def _status(col: str) -> str | None:
            score = msi.get("-".join(str(col).split("-")[:3]))
            if score is None:
                return None
            return "MSI" if score >= MSI_H_CUTOFF else "MSS"

        by_status: dict[str, list[str]] = {}
        for col in sample_cols:
            s = _status(col)
            if s:
                by_status.setdefault(s, []).append(col)
        summaries, written = [], []
        for s, cols in by_status.items():
            code = f"{parent}_{s}"
            cohort = targets[code]
            _cohorts.write_per_sample(gene_table, df[cols], SOURCE_ID, cohort.code)
            summaries.append(_summary_row(gene_table, df[cols], code))
            written.append(code)
            print(f"  {code}: {len(cols)} samples", flush=True)
        # Every registered {parent}_MSI / {parent}_MSS cohort must get data; a
        # status with zero matched samples would silently leave a registered
        # cohort unbuilt (a dangling registry/medoid entry). Surface it loudly.
        missing = sorted({c.code for c in targets.values()
                          if c.code.startswith(f"{parent}_")} - set(written))
        if missing:
            print(f"  WARNING: no samples matched for {missing} (parent "
                  f"{parent}); registered cohort(s) left unbuilt", flush=True)
        if summaries:
            upsert_to_shard(args.summary_output, pd.concat(summaries, ignore_index=True),
                            source_cohort=SUMMARY_COHORT, cancer_codes=written,
                            per_cancer_code_shards=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
