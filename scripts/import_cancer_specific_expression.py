#!/usr/bin/env python
"""Import cancer-specific cohort expression summaries.

The input summary is expected to be a neutral cohort-level table with one
row per ``(symbol, cancer_code, subtype, source_cohort)`` and median/Q1/Q3
tumor TPM columns. Pirlygenes packages the selected rows as ID-keyed
reference expression, not as analysis artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes.expression.qc import _TECHNICAL_RNA_GROUPS, classify_gene_qc
from pirlygenes.expression.stats import REFERENCE_COLUMNS


DEFAULT_SUMMARY_INPUT = "cancer-specific-expression-summary.csv.gz"
SOLID_MARKERS = "tumor-up-vs-matched-normal.csv"
HEME_MARKERS = "heme-tumor-up-vs-matched-normal.csv"
PIPELINE = "cohort_summary_tpm_symbol_to_ensembl112_clean_tpm_v1"
SOURCE_VERSION = (
    "Imported from cohort-level tumor-TPM summaries; source rows are "
    "symbol-only median/Q1/Q3 summaries; symbols harmonized to current "
    "Ensembl release 112 IDs by unique current gene name or conservative "
    "unique historical Ensembl-name rescue; imported 2026-05-19"
)
NOTES = (
    "Cohort-level tumor TPM median/Q1/Q3 summary; "
    "TPM_* preserves source values for mapped genes without redistributing "
    "omitted symbol-only rows; "
    "raw per-sample values are not bundled, so TPM_clean is computed by "
    "summary-level technical-RNA zeroing with per-summary denominator "
    "rescaling. Symbol-only genes unresolved or ambiguous after current and "
    "historical Ensembl lookup are omitted."
)
HISTORICAL_ENSEMBL_RELEASES = [75, 77, 80, 107, 109, 110, 111]


@dataclass(frozen=True)
class ImportGroup:
    source_cohort: str
    input_cancer_code: str
    output_cancer_code: str
    source_project: str
    input_subtype: str | None = None


IMPORT_GROUPS = [
    ImportGroup("BEATAML_OHSU_2022", "BEATAML", "LAML_APL", "BeatAML", "BEATAML_APL"),
    ImportGroup(
        "BEATAML_OHSU_2022",
        "BEATAML",
        "LAML_ELN_Adv",
        "BeatAML",
        "BEATAML_ELN_Adverse",
    ),
    ImportGroup(
        "BEATAML_OHSU_2022",
        "BEATAML",
        "LAML_ELN_Fav",
        "BeatAML",
        "BEATAML_ELN_Favorable",
    ),
    ImportGroup(
        "BEATAML_OHSU_2022",
        "BEATAML",
        "LAML_ELN_Int",
        "BeatAML",
        "BEATAML_ELN_Intermediate",
    ),
    ImportGroup("GSE118014_ALVAREZ_2018", "PANNET", "PANNET", "GEO"),
    ImportGroup("GSE299759_MEIJER_2026", "CHON", "CHON", "GEO"),
    ImportGroup("GSE75885_DELESPAUL_2017", "SARC_DDLPS", "SARC_DDLPS", "GEO"),
    ImportGroup("GSE75885_DELESPAUL_2017", "SARC_LGFMS", "SARC_LGFMS", "GEO"),
    ImportGroup("GSE75885_DELESPAUL_2017", "SARC_PLEOLPS", "SARC_PLEOLPS", "GEO"),
    ImportGroup("SCLC_UCOLOGNE_2015", "SCLC", "SCLC", "University of Cologne"),
    ImportGroup(
        "TARGET_NBL_2018",
        "TARGET_NBL",
        "NBL_MYCN_amp",
        "TARGET",
        "NBL_MYCN_amp",
    ),
    ImportGroup(
        "TARGET_NBL_2018",
        "TARGET_NBL",
        "NBL_MYCN_nonamp",
        "TARGET",
        "NBL_MYCN_nonamp",
    ),
    ImportGroup("TARGET_RT_2017", "TARGET_RT", "RT", "TARGET"),
    ImportGroup("TARGET_WT_2015", "TARGET_WT", "WILMS", "TARGET"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "ATRT", "ATRT", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "EWS", "EWS", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "HEPB", "HEPB", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "MBL", "MBL", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "NUTM", "NUTM", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "OS", "OS", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "RMS_ARMS", "RMS_ARMS", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "RMS_ERMS", "RMS_ERMS", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "RMS_PRMS", "RMS_PRMS", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "RMS_SSRMS", "RMS_SSRMS", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "SARC_LMS", "SARC_LMS", "Treehouse"),
    ImportGroup(
        "TREEHOUSE_POLYA_25_01",
        "SARC_LPS_UNSPEC",
        "SARC_LPS_UNSPEC",
        "Treehouse",
    ),
    ImportGroup("TREEHOUSE_POLYA_25_01", "SARC_MYXFIB", "SARC_MYXFIB", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "SARC_SYN", "SARC_SYN", "Treehouse"),
    ImportGroup("TREEHOUSE_POLYA_25_01", "SARC_UPS", "SARC_UPS", "Treehouse"),
    ImportGroup("TREEHOUSE_RIBOD_25_01", "CHOR", "CHOR", "Treehouse/RiboD"),
    ImportGroup("TREEHOUSE_RIBOD_25_01", "RB", "RB", "Treehouse/RiboD"),
]
IMPORT_SOURCE_COHORTS = sorted({group.source_cohort for group in IMPORT_GROUPS})
IMPORT_OUTPUT_CODES = sorted({group.output_cancer_code for group in IMPORT_GROUPS})


def _select_group(source: pd.DataFrame, group: ImportGroup) -> pd.DataFrame:
    mask = (
        source["source_cohort"].astype(str).eq(group.source_cohort)
        & source["cancer_code"].astype(str).eq(group.input_cancer_code)
    )
    if group.input_subtype is None:
        mask &= source["subtype"].isna() | source["subtype"].astype(str).eq("")
    else:
        mask &= source["subtype"].astype(str).eq(group.input_subtype)
    selected = source.loc[mask].copy()
    if selected.empty:
        raise RuntimeError(f"No rows matched import group {group}")
    duplicated = selected["symbol"].astype(str).duplicated()
    if duplicated.any():
        examples = ", ".join(sorted(selected.loc[duplicated, "symbol"].astype(str))[:8])
        raise RuntimeError(f"Duplicate symbols in import group {group}: {examples}")
    selected["cancer_code"] = group.output_cancer_code
    selected["source_project"] = group.source_project
    return selected


def _unique_gene_by_symbol(genome: EnsemblRelease, symbol: str):
    if not symbol:
        return None
    candidates = [symbol]
    upper = symbol.upper()
    if upper != symbol:
        candidates.append(upper)
    for candidate in candidates:
        try:
            genes = genome.genes_by_name(candidate)
        except Exception:
            genes = []
        gene_ids = {gene.gene_id.split(".", 1)[0] for gene in genes}
        if len(gene_ids) == 1:
            return genes[0]
    return None


def _historical_symbol_rescue(
    symbol: str,
    *,
    current: EnsemblRelease,
    historical: list[EnsemblRelease],
):
    candidates: dict[str, str] = {}
    for genome in historical:
        gene = _unique_gene_by_symbol(genome, symbol)
        if gene is None:
            continue
        gene_id = gene.gene_id.split(".", 1)[0]
        candidates[gene_id] = gene.gene_name or symbol

    current_candidates = {}
    for gene_id in candidates:
        try:
            gene = current.gene_by_id(gene_id)
        except Exception:
            continue
        current_candidates[gene_id] = gene
    if len(current_candidates) != 1:
        if current_candidates:
            return None, "historical_ambiguous"
        if candidates:
            return None, "historical_retired"
        return None, "unresolved"
    return next(iter(current_candidates.values())), "historical"


def _symbol_mapping(
    symbols: pd.Series,
    ensembl_release: int,
    historical_ensembl_releases: list[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    genome = EnsemblRelease(ensembl_release)
    historical = [EnsemblRelease(release) for release in historical_ensembl_releases]
    rows = []
    counts = {
        "resolved_current": 0,
        "resolved_historical": 0,
        "unresolved": 0,
        "historical_retired": 0,
        "historical_ambiguous": 0,
    }
    for symbol in sorted(set(symbols.fillna("").astype(str))):
        gene = _unique_gene_by_symbol(genome, symbol)
        source = "current"
        if gene is None:
            gene, source = _historical_symbol_rescue(
                symbol,
                current=genome,
                historical=historical,
            )
            if gene is None:
                counts[source] += 1
                continue
        rows.append({
            "source_symbol": symbol,
            "Ensembl_Gene_ID": gene.gene_id.split(".", 1)[0],
            "Symbol": gene.gene_name or symbol,
        })
        counts[f"resolved_{source}"] += 1
    mapping = pd.DataFrame(rows)
    return mapping, counts


def _technical_mask(df: pd.DataFrame) -> pd.Series:
    remove_groups = {str(group) for group in _TECHNICAL_RNA_GROUPS}
    qc = [
        classify_gene_qc(symbol, ensembl_id=ensg)
        for symbol, ensg in zip(df["Symbol"], df["Ensembl_Gene_ID"])
    ]
    return pd.Series([klass.group in remove_groups for klass in qc], index=df.index)


def _add_summary_clean_tpm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("TPM_median", "TPM_clean_median"),
        ("TPM_q1", "TPM_clean_q1"),
        ("TPM_q3", "TPM_clean_q3"),
    ]
    grouped = out.groupby(["cancer_code", "source_cohort"], sort=False)
    for _key, index in grouped.groups.items():
        group = out.loc[index]
        removable = _technical_mask(group)
        for raw_col, clean_col in pairs:
            raw = pd.to_numeric(group[raw_col], errors="coerce").fillna(0.0)
            clean = raw.copy()
            clean.loc[removable.to_numpy()] = 0.0
            raw_total = float(raw.sum())
            clean_total = float(clean.sum())
            if raw_total > 0 and clean_total > 0:
                clean *= raw_total / clean_total
            out.loc[index, clean_col] = clean.to_numpy()
    return out


def build_reference_rows(
    summary_input: Path,
    *,
    ensembl_release: int,
    historical_ensembl_releases: list[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_csv(summary_input, low_memory=False)
    frames = [_select_group(source, group) for group in IMPORT_GROUPS]
    selected = pd.concat(frames, ignore_index=True)
    mapping, counts = _symbol_mapping(
        selected["symbol"],
        ensembl_release,
        historical_ensembl_releases,
    )
    merged = selected.merge(
        mapping,
        left_on="symbol",
        right_on="source_symbol",
        how="inner",
    )
    merged = merged.rename(
        columns={
            "tumor_tpm_median": "TPM_median",
            "tumor_tpm_q1": "TPM_q1",
            "tumor_tpm_q3": "TPM_q3",
        }
    )
    agg = {
        "Symbol": "first",
        "source_project": "first",
        "TPM_median": "sum",
        "TPM_q1": "sum",
        "TPM_q3": "sum",
        "n_samples": "max",
    }
    out = (
        merged.groupby(
            ["cancer_code", "source_cohort", "Ensembl_Gene_ID"],
            as_index=False,
            sort=False,
        )
        .agg(agg)
        .reset_index(drop=True)
    )
    out["source_version"] = SOURCE_VERSION
    out["TPM_mean"] = np.nan
    out["n_detected"] = np.nan
    out["processing_pipeline"] = PIPELINE
    out["notes"] = NOTES
    out = _add_summary_clean_tpm(out)
    # Summary-only imports don't carry per-sample data, so the v5.3
    # extended stats (std/min/max/p5/p10/p90/p95 + clean companions
    # incl. clean_mean) stay NaN. Schema parity with builder rows is
    # enforced by reindex against REFERENCE_COLUMNS below.
    numeric_cols = [
        "TPM_median",
        "TPM_q1",
        "TPM_q3",
        "TPM_mean",
        "TPM_clean_median",
        "TPM_clean_q1",
        "TPM_clean_q3",
        "n_samples",
        "n_detected",
    ]
    out[numeric_cols] = out[numeric_cols].round(6)
    out = out.reindex(columns=list(REFERENCE_COLUMNS)).sort_values(
        ["cancer_code", "source_cohort", "Ensembl_Gene_ID"]
    )
    counts["imported_rows"] = len(out)
    counts["imported_codes"] = out["cancer_code"].nunique()
    return out.reset_index(drop=True), counts


def upsert_reference_rows(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()
    if existing.empty:
        out = new_rows.copy()
    else:
        is_imported = existing["source_cohort"].astype(str).isin(IMPORT_SOURCE_COHORTS)
        is_imported &= existing["cancer_code"].astype(str).isin(IMPORT_OUTPUT_CODES)
        out = pd.concat(
            [existing.loc[~is_imported].reindex(columns=list(REFERENCE_COLUMNS)), new_rows],
            ignore_index=True,
        )
    out = out.reindex(columns=list(REFERENCE_COLUMNS)).sort_values(
        ["cancer_code", "source_cohort", "Ensembl_Gene_ID"],
        na_position="last",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def copy_marker_table(source_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(source_path, low_memory=False)
    df = df.sort_values(
        ["cancer_code", "fold_change_vs_matched_normal", "symbol"],
        ascending=[True, False, True],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-data-dir",
        default=Path("."),
        type=Path,
    )
    parser.add_argument(
        "--summary-input",
        default=None,
        type=Path,
        help=(
            "Neutral cohort-level expression summary CSV/CSV.GZ. Defaults to "
            f"<source-data-dir>/{DEFAULT_SUMMARY_INPUT}."
        ),
    )
    parser.add_argument("--solid-marker-input", default=None, type=Path)
    parser.add_argument("--heme-marker-input", default=None, type=Path)
    parser.add_argument(
        "--summary-output",
        default=Path("pirlygenes/data/cancer-reference-expression.csv.gz"),
        type=Path,
    )
    parser.add_argument(
        "--solid-marker-output",
        default=Path("pirlygenes/data/tumor-up-vs-matched-normal.csv"),
        type=Path,
    )
    parser.add_argument(
        "--heme-marker-output",
        default=Path("pirlygenes/data/heme-tumor-up-vs-matched-normal.csv"),
        type=Path,
    )
    parser.add_argument("--ensembl-release", default=112, type=int)
    parser.add_argument(
        "--historical-ensembl-releases",
        default=",".join(str(release) for release in HISTORICAL_ENSEMBL_RELEASES),
        help=(
            "Comma-separated Ensembl releases used only to rescue old "
            "symbol names whose gene IDs still resolve in the current release."
        ),
    )
    args = parser.parse_args()
    summary_input = args.summary_input or args.source_data_dir / DEFAULT_SUMMARY_INPUT
    solid_marker_input = (
        args.solid_marker_input or args.source_data_dir / SOLID_MARKERS
    )
    heme_marker_input = args.heme_marker_input or args.source_data_dir / HEME_MARKERS
    historical_releases = [
        int(part)
        for part in str(args.historical_ensembl_releases).split(",")
        if part.strip()
    ]

    new_rows, counts = build_reference_rows(
        summary_input,
        ensembl_release=args.ensembl_release,
        historical_ensembl_releases=historical_releases,
    )
    combined = upsert_reference_rows(args.summary_output, new_rows)
    solid = copy_marker_table(
        solid_marker_input,
        args.solid_marker_output,
    )
    heme = copy_marker_table(
        heme_marker_input,
        args.heme_marker_output,
    )
    print(
        f"Wrote {len(new_rows)} imported expression rows across "
        f"{counts['imported_codes']} cancer codes into {args.summary_output} "
        f"({len(combined)} total rows)."
    )
    print(
        "Symbol harmonization: "
        f"{counts['resolved_current']} current symbols resolved, "
        f"{counts['resolved_historical']} historical symbols rescued, "
        f"{counts['unresolved']} unresolved, "
        f"{counts['historical_retired']} retired-only, "
        f"{counts['historical_ambiguous']} ambiguous omitted."
    )
    print(
        f"Wrote {len(solid)} solid matched-normal marker rows and "
        f"{len(heme)} heme matched-normal marker rows."
    )


if __name__ == "__main__":
    main()
