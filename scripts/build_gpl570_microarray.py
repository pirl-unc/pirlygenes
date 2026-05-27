#!/usr/bin/env python
"""CLI: dispatch to pirlygenes.builders.affy_gpl570.build_gpl570_source.

Looks up source config from ``pirlygenes/data/expression_sources.yaml``
by ``--source-id`` (or the first positional arg, for compatibility
with the ``pirlygenes build`` dispatcher). Source entries for this
builder use the ``source_type: geo-gpl570-microarray`` tag and the
following fields:

  cancer_codes: [<CODE>]
  builder: scripts/build_gpl570_microarray.py
  series_matrix_url: <direct download URL to series_matrix.txt.gz>
  series_matrix_filename: <local filename in cache_dir>
  source_cohort: <e.g. GSE63790_LANG_2016_HCL>
  source_project: <e.g. GEO>
  citation: <PMID or DOI>
  notes: <free text — appended to the 'notes' column>
  sample_filter:
    include_match: <regex on any series_matrix sample field>
    exclude_match: <regex>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pirlygenes.builders.affy_gpl570 import build_gpl570_source
from pirlygenes.downloads import source_cache_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-id", default=None,
        help="Source id from expression_sources.yaml.",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument(
        "--samples-output", type=Path, default=None,
        help="(Accepted for dispatcher compatibility; not used here.)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override cache dir (default: ~/.cache/pirlygenes/expression/<source-id>/)",
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args, extra = parser.parse_known_args()

    source_id = args.source_id
    if source_id is None and extra:
        source_id = extra[0]
    if source_id is None:
        raise SystemExit("provide --source-id <id>")

    registry_path = (
        Path(__file__).resolve().parents[1]
        / "pirlygenes" / "data" / "expression_sources.yaml"
    )
    with registry_path.open() as f:
        payload = yaml.safe_load(f) or {}
    entry = next(
        (e for e in payload.get("sources", []) if e.get("id") == source_id),
        None,
    )
    if entry is None:
        raise SystemExit(f"source id {source_id!r} not found in registry")
    if entry.get("source_type") != "geo-gpl570-microarray":
        raise SystemExit(
            f"source {source_id!r} has source_type="
            f"{entry.get('source_type')!r}, not 'geo-gpl570-microarray'"
        )

    cache_dir = args.cache_dir or source_cache_dir(
        source_id, category="expression",
    )
    sample_filter = entry.get("sample_filter") or {}
    n = build_gpl570_source(
        series_matrix_url=entry["series_matrix_url"],
        series_matrix_filename=entry["series_matrix_filename"],
        cache_dir=Path(cache_dir),
        cancer_code=entry["cancer_codes"][0],
        source_cohort=entry["source_cohort"],
        source_project=entry.get("source_project", "GEO"),
        citation=entry.get("citation", ""),
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
        sample_include_regex=sample_filter.get("include_match"),
        sample_exclude_regex=sample_filter.get("exclude_match"),
        extra_notes=entry.get("notes", ""),
    )
    print(f"done: {source_id} ({n} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
