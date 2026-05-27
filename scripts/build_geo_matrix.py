#!/usr/bin/env python
"""CLI: dispatch to pirlygenes.builders.geo_matrix.build_source.

Looks up source config from ``pirlygenes/data/expression_sources.yaml``
by ``--source-id`` (or the first positional arg, for compatibility
with the ``pirlygenes build`` dispatcher which already passes the
source id). Source entries in the YAML for this builder use the
``source_type: geo-matrix`` tag and carry the following fields:

  cancer_codes: [<CODE>] | [<CODE>, <CODE>, ...]  # latter when
                                                     # one matrix
                                                     # splits across
                                                     # histologies
  builder: scripts/build_geo_matrix.py
  file_url: <direct download URL>
  file_name: <local filename in cache_dir>
  unit: TPM | FPKM | RPKM | log2(TPM+1) | raw_counts
  gene_id_col: ""              # "" = first column (most common)
  gene_id_type: ensembl | hugo | entrez | auto
  drop_cols: ["Entrez_Gene_Id"]   # optional non-sample columns to drop
  sep: "\t" | ","                 # optional, default TSV
  source_cohort: <e.g. GSE120328_LAMPRECHT_2018>
  source_project: <e.g. GEO>
  citation: <PMID or DOI>
  notes: <free text for the 'notes' column>
  pipeline_stem: <optional override for processing_pipeline prefix>

For sources where one matrix needs to split across multiple
cancer_codes (e.g. an LCNEC vs typical-carcinoid table that has both
in one file), the config also carries:

  sample_to_cancer_code:
    rules:
      - match: <regex on sample id>
        cancer_code: <CODE>
      - match: ...
        cancer_code: <CODE>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

import yaml

from pirlygenes.builders.geo_matrix import GeoMatrixSource, build_source
from pirlygenes.downloads import load_registry, source_cache_dir


def _build_sample_to_cancer_code(rules: list[dict]) -> Callable[[str], str | None]:
    compiled = [(re.compile(r["match"]), r["cancer_code"]) for r in rules]

    def _dispatch(sample_id: str) -> str | None:
        for regex, code in compiled:
            if regex.search(sample_id):
                return code
        return None
    return _dispatch


def _build_sample_filter(spec: dict) -> Callable[[list[str]], list[str]] | None:
    """Build a sample-keep filter from YAML spec.

    Supports two rule kinds (combinable):
      include_match: regex — only keep samples matching this
      exclude_match: regex — drop samples matching this
    """
    include_re = re.compile(spec["include_match"]) if "include_match" in spec else None
    exclude_re = re.compile(spec["exclude_match"]) if "exclude_match" in spec else None

    def _filter(samples: list[str]) -> list[str]:
        out = []
        for s in samples:
            if include_re is not None and not include_re.search(s):
                continue
            if exclude_re is not None and exclude_re.search(s):
                continue
            out.append(s)
        return out
    return _filter


def _build_geo_source(entry: dict) -> GeoMatrixSource:
    cancer_codes = entry["cancer_codes"]
    cancer_code = cancer_codes if len(cancer_codes) > 1 else cancer_codes[0]
    sample_to_cancer_code = None
    rules = entry.get("sample_to_cancer_code") or {}
    if rules and rules.get("rules"):
        sample_to_cancer_code = _build_sample_to_cancer_code(rules["rules"])
    sample_filter = None
    if entry.get("sample_filter"):
        sample_filter = _build_sample_filter(entry["sample_filter"])
    return GeoMatrixSource(
        cancer_code=cancer_code,
        source_cohort=entry["source_cohort"],
        source_project=entry.get("source_project", "GEO"),
        citation=entry.get("citation", ""),
        file_url=entry["file_url"],
        file_name=entry["file_name"],
        unit=entry["unit"],
        gene_id_col=entry.get("gene_id_col", ""),
        gene_id_type=entry.get("gene_id_type", "auto"),
        drop_cols=tuple(entry.get("drop_cols", [])),
        sep=entry.get("sep", "\t"),
        transposed=bool(entry.get("transposed", False)),
        sample_filter=sample_filter,
        sample_to_cancer_code=sample_to_cancer_code,
        notes=entry.get("notes", ""),
        pipeline_stem=entry.get("pipeline_stem", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-id", default=None,
        help="Source id from expression_sources.yaml. If omitted, "
             "uses the first positional arg from `pirlygenes build`.",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument(
        "--samples-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression-samples.csv.gz"),
        help="(Not used by this builder — accepted for dispatcher compatibility.)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override cache dir (default: ~/.cache/pirlygenes/expression/<source-id>/)",
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    args, extra = parser.parse_known_args()

    # Find source id either from --source-id or the first positional
    source_id = args.source_id
    if source_id is None and extra:
        source_id = extra[0]
    if source_id is None:
        raise SystemExit("provide --source-id <id>")

    # Read the YAML config raw (load_registry doesn't include all fields)
    import yaml as _yaml
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "pirlygenes" / "data" / "expression_sources.yaml"
    )
    with registry_path.open() as f:
        payload = _yaml.safe_load(f) or {}
    entry = next(
        (e for e in payload.get("sources", []) if e.get("id") == source_id),
        None,
    )
    if entry is None:
        raise SystemExit(f"source id {source_id!r} not found in registry")
    if entry.get("source_type") != "geo-matrix":
        raise SystemExit(
            f"source {source_id!r} has source_type={entry.get('source_type')!r}, "
            "not 'geo-matrix' — wrong builder for this entry"
        )

    source = _build_geo_source(entry)
    cache_dir = args.cache_dir or source_cache_dir(source_id, category="expression")
    counts = build_source(
        source,
        cache_dir=Path(cache_dir),
        summary_output=args.summary_output,
        ensembl_release=args.ensembl_release,
    )
    print(f"done. sample counts by cancer_code: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
