#!/usr/bin/env python
"""Build packaged GEO heme expression reference summaries.

Inputs are open GEO supplementary expression matrices for heme cohorts that
have enough sample-level metadata to produce a defensible direct tumor
reference. The raw matrices are not packaged; this script writes cohort
summaries and sample-provenance rows into the shared
``cancer-reference-expression`` tables.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc
from pirlygenes.expression.stats import (
    REFERENCE_COLUMNS,
    assign_stats,
    round_stat_columns,
    upsert_to_shard,
)
SAMPLE_COLUMNS = [
    "cancer_code",
    "source_cohort",
    "source_project",
    "case_id",
    "sample_id",
    "source_file_id",
    "source_file_name",
    "source_project_id",
    "sample_type",
    "primary_diagnosis",
    "md5sum",
    "file_size",
    "workflow_type",
    "raw_unit",
    "processing_pipeline",
    "source_url",
    "lineage_evidence_source",
    "included",
    "exclusion_reason",
    "lineage_label",
]
GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"
PIPELINE_PREFIX = "geo_heme_expression_ensembl112_clean_tpm_v1"


@dataclass(frozen=True)
class GeoSource:
    accession: str
    source_file: str
    cancer_code: str
    source_cohort: str
    source_project: str
    gene_col: str
    sep: str
    raw_unit: str
    sample_predicate: Callable[[str, dict[str, str]], bool]
    exclusion_reason: Callable[[str, dict[str, str]], str]
    notes: str


def _char(sample: dict[str, str], key: str) -> str:
    return sample.get(key, "")


def _cml_included(title: str, _sample: dict[str, str]) -> bool:
    return title.startswith("CML-CP")


def _cml_exclusion(title: str, sample: dict[str, str]) -> str:
    if _cml_included(title, sample):
        return ""
    disease = _char(sample, "disease state").lower()
    if "blast crisis" in disease:
        return "blast_crisis_not_chronic_phase"
    if "healthy" in disease:
        return "healthy_control"
    return "not_chronic_phase_cml"


def _mds_included(_title: str, sample: dict[str, str]) -> bool:
    return _char(sample, "disease status") == "Myelodysplastic Syndrome"


def _mds_exclusion(title: str, sample: dict[str, str]) -> str:
    return "" if _mds_included(title, sample) else "not_mds"


def _all_included(_title: str, _sample: dict[str, str]) -> bool:
    return True


def _not_excluded(_title: str, _sample: dict[str, str]) -> str:
    return ""


_MPN_STATES = {
    "Polycythemia vera",
    "Essential thrombocythemia",
    "Myelofibrosis",
}


def _mpn_included(_title: str, sample: dict[str, str]) -> bool:
    return _char(sample, "disease state") in _MPN_STATES


def _mpn_exclusion(title: str, sample: dict[str, str]) -> str:
    if _mpn_included(title, sample):
        return ""
    disease = _char(sample, "disease state")
    if disease == "Post-MPN secondary acute myeloid leukemia":
        return "secondary_aml_not_chronic_phase_mpn"
    if disease:
        return "not_chronic_phase_mpn"
    return "not_mpn"


GEO_SOURCES = [
    GeoSource(
        accession="GSE100026",
        source_file="GSE100026_expressed_gene_RPKM.txt.gz",
        cancer_code="CML",
        source_cohort="GSE100026_DING_2017",
        source_project="GEO",
        gene_col="Gene",
        sep="\t",
        raw_unit="RPKM",
        sample_predicate=_cml_included,
        exclusion_reason=_cml_exclusion,
        notes=(
            "GEO GSE100026 peripheral-blood mononuclear-cell RPKM matrix; "
            "chronic-phase CML samples only; RPKM values converted to TPM "
            "within retained, uniquely Ensembl-harmonized genes."
        ),
    ),
    GeoSource(
        accession="GSE114922",
        source_file="GSE114922_TPM_table.txt.gz",
        cancer_code="MDS",
        source_cohort="GSE114922_SHIOZAWA_2018",
        source_project="GEO",
        gene_col="ensembl_ID",
        sep="\t",
        raw_unit="TPM",
        sample_predicate=_mds_included,
        exclusion_reason=_mds_exclusion,
        notes=(
            "GEO GSE114922 bone-marrow CD34+ HSPC TPM matrix; "
            "myelodysplastic-syndrome samples only; Ensembl IDs harmonized "
            "to Ensembl release 112."
        ),
    ),
    GeoSource(
        accession="GSE271664",
        source_file="GSE271664_HTSeq_counts.csv.gz",
        cancer_code="MCL",
        source_cohort="GSE271664_BODOR_2025",
        source_project="GEO",
        gene_col="Gene",
        sep=",",
        raw_unit="HTSeq counts",
        sample_predicate=_all_included,
        exclusion_reason=_not_excluded,
        notes=(
            "GEO GSE271664 mantle-cell-lymphoma FFPE lymph-biopsy HTSeq "
            "count matrix; raw counts length-normalized with Ensembl release "
            "112 gene spans to TPM."
        ),
    ),
    GeoSource(
        accession="GSE283710",
        source_file="GSE283710_CD34_rawcounts.csv.gz",
        cancer_code="MPN",
        source_cohort="GSE283710_WASHU_2024",
        source_project="GEO",
        gene_col="external_gene_name",
        sep=",",
        raw_unit="raw counts",
        sample_predicate=_mpn_included,
        exclusion_reason=_mpn_exclusion,
        notes=(
            "GEO GSE283710 sorted CD34+ count matrix; chronic-phase PV, ET, "
            "and MF samples only; raw counts length-normalized with Ensembl "
            "release 112 gene spans to TPM."
        ),
    ),
]


def _series_bucket(accession: str) -> str:
    return f"{accession[:3]}{accession[3:-3]}nnn"


def _geo_url(source: GeoSource, kind: str) -> str:
    base = f"{GEO_FTP}/{_series_bucket(source.accession)}/{source.accession}"
    if kind == "soft":
        return f"{base}/soft/{source.accession}_family.soft.gz"
    if kind == "supp":
        return f"{base}/suppl/{source.source_file}"
    raise ValueError(kind)


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, tmp_path.open(
        "wb",
    ) as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(path)
    return path


def _parse_soft_samples(path: Path) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                current = {"geo_accession": line.split(" = ", 1)[1]}
                continue
            if current is None:
                continue
            if line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
                samples[current["title"]] = current
                continue
            if line.startswith("!Sample_geo_accession = "):
                current["geo_accession"] = line.split(" = ", 1)[1]
                continue
            if line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1]
                if ":" in value:
                    key, val = value.split(":", 1)
                    current[key.strip().lower()] = val.strip()
    return samples


def _gene_by_id(genome: EnsemblRelease, gene_id: str):
    try:
        return genome.gene_by_id(gene_id.split(".", 1)[0])
    except Exception:
        return None


def _unique_gene_by_symbol(genome: EnsemblRelease, symbol: str):
    if not symbol:
        return None
    try:
        genes = genome.genes_by_name(symbol)
    except Exception:
        return None
    gene_ids = {gene.gene_id.split(".", 1)[0] for gene in genes}
    if len(gene_ids) != 1:
        return None
    return genes[0]


def _gene_length(gene) -> int:
    return int(gene.end) - int(gene.start) + 1


def _harmonize_source_genes(
    source_genes: pd.Series,
    *,
    ensembl_release: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    genome = EnsemblRelease(ensembl_release)
    rows = []
    counts = {"source_id": 0, "symbol": 0, "dropped": 0}
    for idx, value in source_genes.items():
        token = str(value).split(".", 1)[0].strip()
        gene = _gene_by_id(genome, token) if token.startswith("ENSG") else None
        if gene is not None:
            counts["source_id"] += 1
        else:
            gene = _unique_gene_by_symbol(genome, token)
            if gene is not None:
                counts["symbol"] += 1
        if gene is None:
            counts["dropped"] += 1
            continue
        rows.append({
            "row_index": idx,
            "source_gene": value,
            "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
            "Symbol": gene.gene_name or token,
            "gene_length": _gene_length(gene),
        })
    mapping = pd.DataFrame(rows)
    return mapping, counts | {"canonical_genes": mapping["Ensembl_Gene_ID"].nunique()}


def _collapse_values(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    sample_cols: list[str],
    *,
    raw_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_matrix = df.loc[mapping["row_index"], sample_cols].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    value_matrix.index = mapping["row_index"].to_numpy()
    mapped = mapping.set_index("row_index")
    values = value_matrix.copy()
    values["Ensembl_Gene_ID"] = mapped["Ensembl_Gene_ID"]

    if raw_unit.lower() in {"raw counts", "htseq counts"}:
        lengths_kb = mapped["gene_length"].astype(float) / 1000.0
        rpk = value_matrix.div(lengths_kb, axis=0)
        rpk["Ensembl_Gene_ID"] = mapped["Ensembl_Gene_ID"]
        collapsed = rpk.groupby("Ensembl_Gene_ID", sort=False).sum()
        sums = collapsed.sum(axis=0)
        tpm = collapsed.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0
    else:
        collapsed = values.groupby("Ensembl_Gene_ID", sort=False).sum()
        if raw_unit == "RPKM":
            sums = collapsed.sum(axis=0)
            tpm = collapsed.div(sums.where(sums > 0), axis=1).fillna(0.0) * 1_000_000.0
        else:
            tpm = collapsed

    gene_table = (
        mapping[["Ensembl_Gene_ID", "Symbol"]]
        .drop_duplicates("Ensembl_Gene_ID")
        .set_index("Ensembl_Gene_ID")
        .reindex(tpm.index)
        .reset_index()
    )
    return gene_table, tpm


def _technical_mask(gene_table: pd.DataFrame) -> pd.Series:
    remove_groups = {str(group) for group in _TECHNICAL_RNA_GROUPS}
    qc = [
        classify_gene_qc(symbol, ensembl_id=ensg)
        for symbol, ensg in zip(
            gene_table["Symbol"],
            gene_table["Ensembl_Gene_ID"],
        )
    ]
    return pd.Series(
        [klass.group in remove_groups for klass in qc],
        index=gene_table.index,
    )


def _clean_tpm(values: pd.DataFrame, removable: pd.Series) -> pd.DataFrame:
    clean = values.copy()
    clean.loc[removable.to_numpy(), :] = 0.0
    remaining = clean.sum(axis=0)
    scale = pd.Series(np.nan, index=remaining.index, dtype=float)
    positive = remaining > 0
    scale.loc[positive] = 1_000_000.0 / remaining.loc[positive]
    return clean.mul(scale, axis=1).fillna(0.0)


def _build_sample_manifest(
    source: GeoSource,
    samples: dict[str, dict[str, str]],
    matrix_sample_cols: list[str],
) -> pd.DataFrame:
    rows = []
    matrix_set = set(matrix_sample_cols)
    for title, sample in sorted(samples.items()):
        if title not in matrix_set:
            continue
        included = source.sample_predicate(title, sample)
        rows.append({
            "cancer_code": source.cancer_code,
            "source_cohort": source.source_cohort,
            "source_project": source.source_project,
            "case_id": sample.get("patient id", title),
            "sample_id": title,
            "source_file_id": sample.get("geo_accession", ""),
            "source_file_name": source.source_file,
            "source_project_id": source.accession,
            "sample_type": sample.get("tissue", ""),
            "primary_diagnosis": sample.get("disease status")
            or sample.get("disease state", ""),
            "md5sum": "",
            "file_size": "",
            "workflow_type": "GEO supplementary matrix",
            "raw_unit": source.raw_unit,
            "processing_pipeline": PIPELINE_PREFIX,
            "source_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={source.accession}",
            "lineage_evidence_source": (
                f"{source.accession} GEO Sample characteristics"
            ),
            "included": included,
            "exclusion_reason": source.exclusion_reason(title, sample),
            "lineage_label": source.cancer_code if included else "",
        })
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def _summarize(
    source: GeoSource,
    gene_table: pd.DataFrame,
    values: pd.DataFrame,
) -> pd.DataFrame:
    clean = _clean_tpm(values, _technical_mask(gene_table))
    out = gene_table[["Ensembl_Gene_ID", "Symbol"]].copy()
    out["cancer_code"] = source.cancer_code
    out["source_cohort"] = source.source_cohort
    out["source_project"] = source.source_project
    out["source_version"] = (
        f"{source.accession} {source.source_file}; Ensembl IDs harmonized "
        "to Ensembl release 112; downloaded 2026-05-19"
    )
    assign_stats(out, values, clean)
    out["processing_pipeline"] = PIPELINE_PREFIX
    out["notes"] = source.notes
    return round_stat_columns(out)[list(REFERENCE_COLUMNS)]


def _build_source(
    source: GeoSource,
    *,
    cache_dir: Path,
    ensembl_release: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    source_dir = cache_dir / source.accession
    soft_path = _download(
        _geo_url(source, "soft"),
        source_dir / f"{source.accession}_family.soft.gz",
    )
    matrix_path = _download(_geo_url(source, "supp"), source_dir / source.source_file)
    samples = _parse_soft_samples(soft_path)
    df = pd.read_csv(
        matrix_path,
        sep=source.sep,
        compression="gzip",
        low_memory=False,
    )
    metadata_sample_cols = [col for col in df.columns if col in samples]
    included_cols = [
        col
        for col in metadata_sample_cols
        if source.sample_predicate(col, samples[col])
    ]
    if not included_cols:
        raise RuntimeError(f"No included samples for {source.accession}")
    mapping, counts = _harmonize_source_genes(
        df[source.gene_col],
        ensembl_release=ensembl_release,
    )
    gene_table, values = _collapse_values(
        df,
        mapping,
        included_cols,
        raw_unit=source.raw_unit,
    )
    summary = _summarize(source, gene_table, values)
    manifest = _build_sample_manifest(source, samples, metadata_sample_cols)
    print(
        f"{source.cancer_code}: {len(summary)} genes, "
        f"{len(included_cols)} included samples, {counts}"
    )
    return summary, manifest, counts


def _upsert_reference(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Write each source_cohort group to its own shard under ``path``.

    ``path`` is the sharded data directory
    (``pirlygenes/data/cancer-reference-expression/``); each unique
    ``source_cohort`` in ``new_rows`` becomes one shard.
    """
    written = []
    for source_cohort, group in new_rows.groupby("source_cohort", sort=False):
        codes = sorted(group["cancer_code"].astype(str).unique())
        shard = upsert_to_shard(
            path,
            group.reset_index(drop=True),
            source_cohort=str(source_cohort),
            cancer_codes=codes,
        )
        written.append(shard)
    return pd.concat(written, ignore_index=True) if written else new_rows.iloc[0:0]


def _upsert_samples(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    if existing.empty:
        out = new_rows.copy()
    else:
        keys = set(
            zip(
                new_rows["cancer_code"].astype(str),
                new_rows["source_cohort"].astype(str),
            )
        )
        keep = ~existing[["cancer_code", "source_cohort"]].apply(
            lambda row: (str(row["cancer_code"]), str(row["source_cohort"])) in keys,
            axis=1,
        )
        out = pd.concat([existing[keep], new_rows], ignore_index=True)
    out = out[SAMPLE_COLUMNS].sort_values(
        ["cancer_code", "source_cohort", "sample_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[source.cancer_code for source in GEO_SOURCES],
        help="Cancer codes to build; defaults to all GEO heme sources.",
    )
    args = parser.parse_args()

    requested = {code.upper() for code in args.sources}
    sources = [source for source in GEO_SOURCES if source.cancer_code in requested]
    if not sources:
        raise SystemExit(f"No matching sources for {sorted(requested)}")

    summaries = []
    manifests = []
    for source in sources:
        summary, manifest, _counts = _build_source(
            source,
            cache_dir=args.cache_dir,
            ensembl_release=args.ensembl_release,
        )
        summaries.append(summary)
        manifests.append(manifest)
    combined_summary = _upsert_reference(args.summary_output, pd.concat(summaries))
    combined_samples = _upsert_samples(args.samples_output, pd.concat(manifests))
    print(
        f"Wrote {len(combined_summary)} reference rows and "
        f"{len(combined_samples)} sample provenance rows."
    )


if __name__ == "__main__":
    main()
