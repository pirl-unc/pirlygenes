#!/usr/bin/env python
"""MID_NET reference shard from GSE98894 (Alvarez 2018 PMID 30013182).

GSE98894 ships its expression data as a single ``GSE98894_RAW.tar``
holding one ``GSM*_<sample-name>_counts.txt.gz`` per sample (212 in
all): 81 small intestinal NET, 113 pancreatic NET, 18 rectal NET.
Each per-sample file is a two-column ``gene_id\tcount`` (HTSeq).

We pull primary-site labels from the series_matrix metadata
(Sample_characteristics_ch1 = ``tissue:`` or ``primary site:``) and
split:

  - MID_NET  ← small intestinal
  - PNET     ← pancreatic
  - REC_NET  ← rectal  (registered only if present)

Only MID_NET is the primary deliverable here; PNET / REC_NET shards
are written if their cancer_codes exist in the registry. We never
silently mix histologies into a single bucket.

Source-id ``gse98894-midnet`` in ``expression_sources.yaml``.
"""

from __future__ import annotations

import argparse
import gzip
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from pirlygenes.builders.affy_gpl570 import parse_series_matrix
from pirlygenes.builders.geo_matrix import (
    _clean_tpm,
    _gene_lengths_kb_for_index,
    _technical_mask,
    normalize_to_tpm,
)
from pirlygenes.builders.ncbi_gene_info import harmonize_entrez_via_ncbi
from pirlygenes.expression.stats import (
    assign_stats,
    finalize_reference_rows,
    upsert_to_shard,
)


RAW_TAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE98nnn/GSE98894/suppl/"
    "GSE98894_RAW.tar"
)
SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE98nnn/GSE98894/matrix/"
    "GSE98894_series_matrix.txt.gz"
)
SOURCE_COHORT = "GSE98894_ALVAREZ_2018_NET"
SOURCE_PROJECT = "Alvarez 2018 small-intestinal & pancreatic NET (GSE98894)"
CITATION = "PMID 30013182 (Alvarez 2018)"

# Map series_matrix characteristic values → pirlygenes cancer code.
# GSE98894 fields: char_type is constant 'liver metastasis'; char_origin
# carries primary-site words (pancreas / small intestine / rectum). All
# 212 samples are LIVER METASTASES of GEP-NET — none are primary-site
# tumors. We attribute the metastasis-of-X to the X primary cancer_code
# but flag tumor_origin='metastasis' / metastasis_site='liver' in the
# shard so consumers can hold the caveat.
ORIGIN_TO_CODE = {
    "small intestine": "MID_NET",
    "small intestinal": "MID_NET",
    "ileum": "MID_NET",
    "jejunum": "MID_NET",
    "duodenum": "MID_NET",
    "pancreas": "PANNET",
    "pancreatic": "PANNET",
    "rectum": "REC_NET",
    "rectal": "REC_NET",
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        shutil.copyfileobj(r, h)
    tmp.replace(dest)
    return dest


def _extract_counts_matrix(tar_path: Path) -> pd.DataFrame:
    """Return counts matrix: rows = gene_id (Entrez), cols = GSM accession."""
    per_sample: dict[str, pd.Series] = {}
    print(f"opening tar {tar_path.name}...")
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.isfile()
                   and m.name.endswith("_counts.txt.gz")]
        print(f"  {len(members)} per-sample files")
        for m in members:
            gsm = m.name.split("_", 1)[0]
            fh = tf.extractfile(m)
            if fh is None:
                continue
            with gzip.open(io.BytesIO(fh.read()), "rt") as h:
                # Files have a header row: "GeneID\tCounts" then
                # tab-separated rows of Entrez_id + count.
                df = pd.read_csv(h, sep="\t", header=0)
            # Defensive: normalize column names then keep first two.
            df.columns = [str(c).strip() for c in df.columns]
            id_col, count_col = df.columns[0], df.columns[1]
            df = df[[id_col, count_col]].rename(
                columns={id_col: "gene_id", count_col: "count"}
            )
            df["gene_id"] = df["gene_id"].astype(str).str.strip()
            df = df[~df["gene_id"].str.startswith("__")]
            df["count"] = pd.to_numeric(df["count"], errors="coerce")
            series = df.dropna(subset=["count"]).set_index("gene_id")["count"]
            per_sample[gsm] = series
    matrix = pd.DataFrame(per_sample)
    matrix.index.name = "gene_id"
    matrix = matrix.fillna(0.0)
    print(f"  combined counts matrix: {matrix.shape}")
    return matrix


# Match the ``char_<key>`` portion exactly (after stripping the
# ``char_`` prefix) against any of these tokens. Whole-token match
# avoids surprises like ``char_originator_email`` accidentally being
# used as a primary-site field.
_ORIGIN_FIELD_TOKENS: frozenset[str] = frozenset({
    "origin", "primary", "primary_site", "site",
    "tissue", "tissue_of_origin", "primary_tissue",
    "histology", "diagnosis", "anatomic", "anatomic_site",
})


def _route_samples(
    sample_meta: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Return {gsm: cancer_code} based on primary-site fields.

    For GSE98894 the relevant char_* field is ``char_origin`` (the
    primary tumor site); ``char_type`` is constant ``liver metastasis``.
    We only consult ``char_<key>`` fields whose key (after the
    ``char_`` prefix) is in :data:`_ORIGIN_FIELD_TOKENS` — that way
    something like ``char_originator_email`` (a hypothetical contact
    field) wouldn't accidentally feed into the routing.
    """
    routing: dict[str, str] = {}
    unmatched: list[tuple[str, dict[str, str]]] = []
    for gsm, fields in sample_meta.items():
        origin_values = [
            v for k, v in fields.items()
            if k.startswith("char_")
            and k.removeprefix("char_") in _ORIGIN_FIELD_TOKENS
        ]
        code: str | None = None
        for raw in origin_values:
            low = raw.lower().strip()
            for needle, c in ORIGIN_TO_CODE.items():
                if needle in low:
                    code = c
                    break
            if code is not None:
                break
        if code is None:
            unmatched.append((gsm, fields))
        else:
            routing[gsm] = code
    if unmatched:
        print(f"  WARN: {len(unmatched)} samples could not be tissue-routed")
        for gsm, fields in unmatched[:3]:
            print(f"    {gsm}: {fields}")
    return routing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path.home() / ".cache" / "pirlygenes" / "expression"
                / "gse98894-net",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=Path("pirlygenes/data/cancer-reference-expression"),
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--codes", nargs="*", default=("MID_NET", "PANNET", "REC_NET"),
        help="cancer codes to upsert; default all three GEP-NET primary "
             "sites that the GSE98894 cohort spans (each shard is "
             "tagged tumor_origin=metastasis, metastasis_site=liver). "
             "PANNET is the existing pancreatic-NET registry code; "
             "REC_NET was added with this build.",
    )
    parser.add_argument(
        "--samples-output", type=Path, default=None,
        help="(Accepted for dispatcher compatibility; not used here.)",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.cache_dir / "GSE98894_RAW.tar"
    series_path = args.cache_dir / "GSE98894_series_matrix.txt.gz"

    print("downloading RAW.tar...")
    _download(RAW_TAR_URL, tar_path)
    print("downloading series_matrix...")
    _download(SERIES_MATRIX_URL, series_path)

    print("parsing series_matrix (metadata only)...")
    _, sample_meta = parse_series_matrix(series_path)
    print(f"  metadata rows: {len(sample_meta)}")

    print("extracting per-sample counts from RAW.tar...")
    counts = _extract_counts_matrix(tar_path)

    # The per-sample files are keyed by GSM in the filename; series_matrix
    # is also keyed by GSM. Align.
    shared = sorted(set(counts.columns) & set(sample_meta))
    print(f"  GSMs in both counts and metadata: {len(shared)}")
    counts = counts[shared]

    print("routing samples to cancer codes via tissue/primary-site...")
    routing = _route_samples({g: sample_meta[g] for g in shared})
    code_counts = pd.Series(list(routing.values())).value_counts().to_dict()
    print(f"  routing: {code_counts}")

    # GSE98894 per-sample files use Entrez Gene IDs (numeric). Map via
    # NCBI gene_info → HUGO → ENSG (pyensembl), then sum-aggregate per
    # ENSG before length-normalizing to TPM.
    print(f"harmonizing Entrez → Ensembl release {args.ensembl_release} "
          "via NCBI gene_info...")
    mapping, values = harmonize_entrez_via_ncbi(
        counts, ensembl_release=args.ensembl_release,
    )
    gene_table = (
        mapping.drop_duplicates("Ensembl_Gene_ID")[
            ["Ensembl_Gene_ID", "Symbol"]
        ]
        .reset_index(drop=True)
    )
    values = values.reindex(gene_table["Ensembl_Gene_ID"]).fillna(0.0)
    print(f"  canonical genes: {len(gene_table)}")

    lengths_kb = _gene_lengths_kb_for_index(
        values.index, gene_id_type="ensembl",
        ensembl_release=args.ensembl_release,
    )
    tpm = normalize_to_tpm(values, unit="raw_counts", gene_lengths_kb=lengths_kb)
    print(f"  TPM shape: {tpm.shape}")

    summaries: list[pd.DataFrame] = []
    written_codes: list[str] = []
    for code in args.codes:
        gsms = [g for g in tpm.columns if routing.get(g) == code]
        if not gsms:
            print(f"  {code}: no samples — skipping")
            continue
        sub = tpm[gsms]
        clean = _clean_tpm(sub, gene_table=gene_table)
        out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
        out["cancer_code"] = code
        out["source_cohort"] = SOURCE_COHORT
        out["source_project"] = SOURCE_PROJECT
        out["source_version"] = (
            f"GSE98894 RAW.tar HTSeq counts → Entrez→ENSG (NCBI "
            f"gene_info → pyensembl release {args.ensembl_release}) → "
            f"length-norm TPM (Ensembl {args.ensembl_release} gene lengths) "
            f"→ per-sample sum-to-1e6; primary-site routed via "
            f"series_matrix char_origin."
        )
        assign_stats(out, sub, clean)
        out["processing_pipeline"] = (
            f"gse98894_alvarez_2018_net_raw_counts_to_tpm_ensembl"
            f"{args.ensembl_release}_clean_tpm_v4"
        )
        out["metastasis_site"] = "liver"
        out["notes"] = (
            f"{code} from Alvarez 2018 GEP-NET cohort GSE98894 "
            f"(n={len(gsms)}). ALL samples are liver metastases "
            f"(not primary-site tumors); primary site routed from "
            f"series_matrix char_origin. HTSeq counts (Entrez) → NCBI "
            f"Symbol → pyensembl ENSG → length-norm TPM → tech-RNA "
            f"zero. {CITATION}."
        )
        out = finalize_reference_rows(out, tumor_origin="metastasis")
        summaries.append(out)
        written_codes.append(code)
        print(f"  {code}: n={len(gsms)} → {len(out)} gene rows")

    if not summaries:
        print("no cohort matched the requested codes — nothing to write")
        return 1
    combined = pd.concat(summaries, ignore_index=True)
    upsert_to_shard(
        args.summary_output,
        combined,
        source_cohort=SOURCE_COHORT,
        cancer_codes=written_codes,
    )
    print(f"upserted {len(combined)} rows into shard {SOURCE_COHORT}.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
