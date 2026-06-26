#!/usr/bin/env python
"""Compare pirlygenes and oncoref reference-data behavior.

This is a migration aid, not a test. It imports both libraries, runs matched
public accessors and small deterministic normalization fixtures, then writes
machine-readable differences under ``analyses/outputs/run_oncoref_parity`` by
default. Expression comparisons are skipped unless the oncoref data bundle is
already local or ``--allow-fetch`` is supplied.

Typical use from the pirlygenes checkout::

    python scripts/compare_oncoref_pirlygenes.py --oncoref-path ../oncoref

Use ``--skip-expression`` for a fast small-table + normalization pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "analyses" / "outputs" / "run_oncoref_parity"
DEFAULT_CODES = (
    "LUAD",
    "LUAD_KRAS",
    "LUAD_STK11",
    "COAD",
    "READ",
    "COAD_MSI",
    "READ_MSI",
    "CRC_MSI",
    "SCLC_ASCL1",
    "SARC_OS",
    "SARC_UPS",
    "SARC_DDLPS",
    "ASTB",
    "MPN",
    "THYM",
)
DEFAULT_COHORTS = ("PRAD", "CLL", "COAD_MSI", "READ_MSI")
DEFAULT_GENES = ("TP53", "MS4A1", "KLK3", "MALAT1", "RPL13A")
BURDEN_METRICS = (
    "us_incidence_pct",
    "us_mortality_pct",
    "world_incidence_pct",
    "world_mortality_pct",
)


@dataclass
class Libraries:
    pg: Any
    pg_gsc: Any
    pg_norm: Any
    pg_expr: Any
    oc: Any
    oc_norm: Any
    oc_bundle: Any


def _csv_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    return [p.strip() for p in value.split(",") if p.strip()]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def _safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _values_equal(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    try:
        left_na = pd.isna(left)
    except (TypeError, ValueError):
        left_na = False
    try:
        right_na = pd.isna(right)
    except (TypeError, ValueError):
        right_na = False
    if bool(left_na) and bool(right_na):
        return True
    if bool(left_na) != bool(right_na):
        return False

    left_num = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_num = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if not pd.isna(left_num) and not pd.isna(right_num):
        return bool(math.isclose(float(left_num), float(right_num), rel_tol=rtol, abs_tol=atol))
    return str(left) == str(right)


def _readable_key(index_value: Any, key_cols: list[str]) -> dict[str, Any]:
    if len(key_cols) == 1:
        return {key_cols[0]: index_value}
    if not isinstance(index_value, tuple):
        index_value = (index_value,)
    return dict(zip(key_cols, index_value))


def _prepare_for_key(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in key_cols:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compare_frames(
    *,
    name: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_label: str,
    right_label: str,
    out_dir: Path,
    key_cols: list[str],
    rtol: float,
    atol: float,
    max_diff_rows: int | None,
) -> dict[str, Any]:
    """Compare two DataFrames by key and write schema/row/value diffs."""
    comp_dir = out_dir / "tables"
    safe = _safe_name(name)
    summary: dict[str, Any] = {
        "section": "table",
        "name": name,
        "left_rows": len(left),
        "right_rows": len(right),
        "left_cols": len(left.columns),
        "right_cols": len(right.columns),
        "column_order_equal": list(left.columns) == list(right.columns),
        "only_left_cols": sorted(set(left.columns) - set(right.columns)),
        "only_right_cols": sorted(set(right.columns) - set(left.columns)),
        "key_cols": key_cols,
        "status": "ok",
    }
    _write_json(comp_dir / f"{safe}__schema.json", summary)

    if not key_cols or any(c not in left.columns for c in key_cols) or any(
        c not in right.columns for c in key_cols
    ):
        summary["status"] = "schema_mismatch"
        summary["error"] = "missing key columns"
        return summary

    left_keyed = _prepare_for_key(left, key_cols)
    right_keyed = _prepare_for_key(right, key_cols)
    if left_keyed.duplicated(key_cols).any() or right_keyed.duplicated(key_cols).any():
        summary["status"] = "duplicate_keys"
        summary["left_duplicate_key_count"] = int(left_keyed.duplicated(key_cols).sum())
        summary["right_duplicate_key_count"] = int(right_keyed.duplicated(key_cols).sum())
        return summary

    left_idx = left_keyed.set_index(key_cols, drop=False)
    right_idx = right_keyed.set_index(key_cols, drop=False)
    left_keys = set(left_idx.index)
    right_keys = set(right_idx.index)
    only_left = sorted(left_keys - right_keys)
    only_right = sorted(right_keys - left_keys)
    common = sorted(left_keys & right_keys)

    summary["only_left_rows"] = len(only_left)
    summary["only_right_rows"] = len(only_right)
    summary["common_rows"] = len(common)

    if only_left:
        rows = left_idx.loc[only_left].reset_index(drop=True)
        _write_df(comp_dir / f"{safe}__only_{left_label}.csv", rows)
    if only_right:
        rows = right_idx.loc[only_right].reset_index(drop=True)
        _write_df(comp_dir / f"{safe}__only_{right_label}.csv", rows)

    common_cols = [c for c in left.columns if c in right.columns and c not in key_cols]
    diffs: list[dict[str, Any]] = []
    total_diffs = 0
    for key in common:
        lrow = left_idx.loc[key]
        rrow = right_idx.loc[key]
        for col in common_cols:
            lv = lrow[col]
            rv = rrow[col]
            if _values_equal(lv, rv, rtol=rtol, atol=atol):
                continue
            total_diffs += 1
            if max_diff_rows is None or len(diffs) < max_diff_rows:
                rec = _readable_key(key, key_cols)
                rec.update(
                    {
                        "column": col,
                        left_label: lv,
                        right_label: rv,
                    }
                )
                diffs.append(rec)

    summary["value_diff_cells"] = total_diffs
    summary["value_diff_cells_written"] = len(diffs)
    if diffs:
        _write_df(comp_dir / f"{safe}__value_diffs.csv", pd.DataFrame(diffs))
    if total_diffs or only_left or only_right or summary["only_left_cols"] or summary["only_right_cols"]:
        summary["status"] = "different"
    return summary


def compare_maps(
    *,
    name: str,
    left_map: dict[Any, Any],
    right_map: dict[Any, Any],
    left_label: str,
    right_label: str,
    out_dir: Path,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    safe = _safe_name(name)
    all_keys = sorted({str(k) for k in left_map} | {str(k) for k in right_map})
    left_norm = {str(k): v for k, v in left_map.items()}
    right_norm = {str(k): v for k, v in right_map.items()}
    rows = []
    diff_count = 0
    for key in all_keys:
        in_left = key in left_norm
        in_right = key in right_norm
        lv = left_norm.get(key)
        rv = right_norm.get(key)
        if not in_left:
            status = f"missing_in_{left_label}"
        elif not in_right:
            status = f"missing_in_{right_label}"
        elif _values_equal(lv, rv, rtol=rtol, atol=atol):
            status = "same"
        else:
            status = "different"
        if status != "same":
            diff_count += 1
        rows.append({"key": key, left_label: lv, right_label: rv, "status": status})
    df = pd.DataFrame(rows)
    _write_df(out_dir / "maps" / f"{safe}.csv", df[df["status"] != "same"])
    return {
        "section": "map",
        "name": name,
        "left_rows": len(left_map),
        "right_rows": len(right_map),
        "diff_rows": diff_count,
        "status": "different" if diff_count else "ok",
    }


def compare_probe_values(
    *,
    name: str,
    codes: list[str],
    left_fn: Callable[[str], Any],
    right_fn: Callable[[str], Any],
    left_label: str,
    right_label: str,
    out_dir: Path,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    rows = []
    diff_count = 0
    for code in codes:
        try:
            lv = left_fn(code)
            left_error = ""
        except Exception as e:  # noqa: BLE001 - comparison script captures failures.
            lv = None
            left_error = f"{type(e).__name__}: {e}"
        try:
            rv = right_fn(code)
            right_error = ""
        except Exception as e:  # noqa: BLE001
            rv = None
            right_error = f"{type(e).__name__}: {e}"
        same = not left_error and not right_error and _values_equal(lv, rv, rtol=rtol, atol=atol)
        if not same:
            diff_count += 1
        rows.append(
            {
                "probe": code,
                left_label: lv,
                right_label: rv,
                f"{left_label}_error": left_error,
                f"{right_label}_error": right_error,
                "status": "same" if same else "different",
            }
        )
    df = pd.DataFrame(rows)
    _write_df(out_dir / "probes" / f"{_safe_name(name)}.csv", df[df["status"] != "same"])
    return {
        "section": "probe",
        "name": name,
        "probes": len(codes),
        "diff_rows": diff_count,
        "status": "different" if diff_count else "ok",
    }


def import_libraries(oncoref_path: Path | None) -> Libraries:
    sys.path.insert(0, str(REPO_ROOT))
    if oncoref_path is not None:
        sys.path.insert(0, str(oncoref_path.resolve()))
    pg = importlib.import_module("pirlygenes")
    pg_gsc = importlib.import_module("pirlygenes.gene_sets_cancer")
    pg_norm = importlib.import_module("pirlygenes.expression.normalize")
    pg_expr = importlib.import_module("pirlygenes.expression.accessors")
    oc = importlib.import_module("oncoref")
    oc_norm = importlib.import_module("oncoref.normalization")
    oc_bundle = importlib.import_module("oncoref.data_bundle")
    return Libraries(
        pg=pg,
        pg_gsc=pg_gsc,
        pg_norm=pg_norm,
        pg_expr=pg_expr,
        oc=oc,
        oc_norm=oc_norm,
        oc_bundle=oc_bundle,
    )


def compare_clinical(
    libs: Libraries,
    *,
    out_dir: Path,
    codes: list[str],
    rtol: float,
    atol: float,
    max_diff_rows: int | None,
) -> list[dict[str, Any]]:
    pg = libs.pg_gsc
    oc = libs.oc
    summaries: list[dict[str, Any]] = []
    table_specs = [
        ("cancer_tmb_df", pg.cancer_tmb_df, oc.cancer_tmb_df, ["cancer_code"]),
        (
            "cancer_apd1_response_df",
            pg.cancer_apd1_response_df,
            oc.cancer_apd1_response_df,
            ["cancer_code"],
        ),
        ("cancer_burden_df", pg.cancer_burden_df, oc.cancer_burden_df, ["burden_category"]),
    ]
    for name, left_fn, right_fn, key_cols in table_specs:
        summaries.append(
            compare_frames(
                name=name,
                left=left_fn(),
                right=right_fn(),
                left_label="pirlygenes",
                right_label="oncoref",
                out_dir=out_dir,
                key_cols=key_cols,
                rtol=rtol,
                atol=atol,
                max_diff_rows=max_diff_rows,
            )
        )

    if hasattr(oc, "cancer_ici_response_df"):
        ici_df = oc.cancer_ici_response_df()
        _write_df(
            out_dir / "tables" / "cancer_ici_response_df__oncoref_only.csv",
            ici_df,
        )
        summaries.append(
            {
                "section": "table",
                "name": "cancer_ici_response_df__oncoref_only",
                "status": "missing_in_pirlygenes",
                "left_rows": 0,
                "right_rows": len(ici_df),
                "left_cols": 0,
                "right_cols": len(ici_df.columns),
                "reason": "oncoref exposes broad ICI long table; pirlygenes "
                "only has cancer_apd1_response_df",
            }
        )

    summaries.append(
        compare_maps(
            name="cancer_tmb",
            left_map=pg.cancer_tmb(),
            right_map=oc.cancer_tmb(),
            left_label="pirlygenes",
            right_label="oncoref",
            out_dir=out_dir,
            rtol=rtol,
            atol=atol,
        )
    )
    summaries.append(
        compare_maps(
            name="cancer_apd1_response",
            left_map=pg.cancer_apd1_response(),
            right_map=oc.cancer_apd1_response(),
            left_label="pirlygenes",
            right_label="oncoref",
            out_dir=out_dir,
            rtol=rtol,
            atol=atol,
        )
    )
    for metric in BURDEN_METRICS:
        summaries.append(
            compare_maps(
                name=f"cancer_burden__{metric}",
                left_map=pg.cancer_burden(metric=metric),
                right_map=oc.cancer_burden(metric=metric),
                left_label="pirlygenes",
                right_label="oncoref",
                out_dir=out_dir,
                rtol=rtol,
                atol=atol,
            )
        )

    probe_specs = [
        ("cancer_tmb_probe", pg.cancer_tmb, oc.cancer_tmb),
        ("cancer_apd1_response_probe", pg.cancer_apd1_response, oc.cancer_apd1_response),
        ("burden_category_probe", pg.burden_category, oc.burden_category),
    ]
    for name, left_fn, right_fn in probe_specs:
        summaries.append(
            compare_probe_values(
                name=name,
                codes=codes,
                left_fn=left_fn,
                right_fn=right_fn,
                left_label="pirlygenes",
                right_label="oncoref",
                out_dir=out_dir,
                rtol=rtol,
                atol=atol,
            )
        )

    if hasattr(oc, "cancer_ici_response"):
        summaries.append(
            compare_probe_values(
                name="cancer_ici_response__oncoref_only_probe",
                codes=codes,
                left_fn=lambda _code: None,
                right_fn=oc.cancer_ici_response,
                left_label="pirlygenes",
                right_label="oncoref",
                out_dir=out_dir,
                rtol=rtol,
                atol=atol,
            )
        )
    return summaries


def _normalization_fixtures() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    base = pd.DataFrame(
        {
            "Symbol": ["MT-CO1", "MT-RNR1", "MALAT1", "RPL13A", "TP53", "ACTB"],
            "Ensembl_Gene_ID": [
                "ENSG00000198804",
                "ENSG00000211459",
                "ENSG00000251562",
                "ENSG00000142541",
                "ENSG00000141510",
                "ENSG00000075624",
            ],
        }
    )
    values = pd.DataFrame(
        {
            "S1": [500_000.0, 100_000.0, 20_000.0, 300_000.0, 50_000.0, 200_000.0],
            "S2": [800_000.0, 50_000.0, 30_000.0, 200_000.0, 60_000.0, 100_000.0],
        },
        index=base.index,
    )
    no_ribo = base[base["Symbol"] != "RPL13A"].reset_index(drop=True)
    no_ribo_values = values.loc[base["Symbol"] != "RPL13A"].reset_index(drop=True)
    return {
        "clean_tpm_with_ribo": (base, values),
        "clean_tpm_no_ribo": (no_ribo, no_ribo_values),
    }


def compare_normalization(
    libs: Libraries,
    *,
    out_dir: Path,
    rtol: float,
    atol: float,
    max_diff_rows: int | None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    pg_norm = libs.pg_norm
    oc_norm = libs.oc_norm
    for name, (gene_table, values) in _normalization_fixtures().items():
        pg_clean = pg_norm.clean_tpm_matrix(values, gene_table=gene_table)
        oc_clean = oc_norm.clean_tpm(values, gene_table)
        left = pd.concat([gene_table.reset_index(drop=True), pg_clean.reset_index(drop=True)], axis=1)
        right = pd.concat([gene_table.reset_index(drop=True), oc_clean.reset_index(drop=True)], axis=1)
        summaries.append(
            compare_frames(
                name=name,
                left=left,
                right=right,
                left_label="pirlygenes",
                right_label="oncoref",
                out_dir=out_dir,
                key_cols=["Ensembl_Gene_ID"],
                rtol=rtol,
                atol=atol,
                max_diff_rows=max_diff_rows,
            )
        )

    wide = pd.concat(
        [
            _normalization_fixtures()["clean_tpm_with_ribo"][0],
            _normalization_fixtures()["clean_tpm_with_ribo"][1].rename(
                columns={"S1": "TPM_S1", "S2": "TPM_S2"}
            ),
        ],
        axis=1,
    )
    pg_zero, pg_stats = pg_norm.normalize_expression(wide, value_cols=["TPM_S1", "TPM_S2"])
    oc_zero, oc_stats = oc_norm.normalize_expression(wide, value_cols=["TPM_S1", "TPM_S2"])
    summaries.append(
        compare_frames(
            name="normalize_expression_zero_path",
            left=pg_zero,
            right=oc_zero,
            left_label="pirlygenes",
            right_label="oncoref",
            out_dir=out_dir,
            key_cols=["Ensembl_Gene_ID"],
            rtol=rtol,
            atol=atol,
            max_diff_rows=max_diff_rows,
        )
    )
    _write_json(
        out_dir / "normalization" / "normalize_expression_zero_path__stats.json",
        {"pirlygenes": pg_stats, "oncoref": oc_stats},
    )
    return summaries


def _rename_oncoref_pan_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col.startswith("nTPM_"):
            rename[col] = f"{col[len('nTPM_') :]}_nTPM"
        elif col.startswith("TPM_"):
            rename[col] = f"{col[len('TPM_') :]}_TPM"
        elif col.startswith("FPKM_"):
            rename[col] = f"{col[len('FPKM_') :]}_FPKM"
    return df.rename(columns=rename)


def compare_expression(
    libs: Libraries,
    *,
    out_dir: Path,
    genes: list[str],
    cohorts: list[str],
    allow_fetch: bool,
    rtol: float,
    atol: float,
    max_diff_rows: int | None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not allow_fetch and not libs.oc_bundle.is_local():
        return [
            {
                "section": "expression",
                "name": "expression_bundle",
                "status": "skipped",
                "reason": "oncoref data bundle is not local; rerun with --allow-fetch",
            }
        ]

    pg_expr = libs.pg_expr
    oc = libs.oc
    try:
        pg_pan = pg_expr.pan_cancer_expression(genes=genes, normalize="tpm")
        oc_pan = _rename_oncoref_pan_columns(oc.pan_cancer_expression(genes=genes, to_tpm=True))
        common_cols = [
            c
            for c in pg_pan.columns
            if c in oc_pan.columns and c not in ("Proteoform_ID", "Member_Ensembl_Gene_IDs")
        ]
        summaries.append(
            compare_frames(
                name="pan_cancer_expression__tpm_probe",
                left=pg_pan[common_cols],
                right=oc_pan[common_cols],
                left_label="pirlygenes",
                right_label="oncoref",
                out_dir=out_dir,
                key_cols=["Ensembl_Gene_ID"],
                rtol=rtol,
                atol=atol,
                max_diff_rows=max_diff_rows,
            )
        )
        _write_json(
            out_dir / "expression" / "pan_cancer_expression__schema.json",
            {
                "pirlygenes_shape": pg_pan.shape,
                "oncoref_shape": oc_pan.shape,
                "only_pirlygenes_cols": sorted(set(pg_pan.columns) - set(oc_pan.columns)),
                "only_oncoref_cols": sorted(set(oc_pan.columns) - set(pg_pan.columns)),
                "common_cols_compared": common_cols,
            },
        )
    except Exception as e:  # noqa: BLE001
        summaries.append(
            {
                "section": "expression",
                "name": "pan_cancer_expression__tpm_probe",
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }
        )

    for cohort in cohorts:
        try:
            pg_pct = pg_expr.cohort_gene_percentiles(cohort)
            oc_pct = oc.cohort_gene_percentiles(cohort)
            summaries.append(
                compare_frames(
                    name=f"cohort_gene_percentiles__{cohort}",
                    left=pg_pct,
                    right=oc_pct,
                    left_label="pirlygenes",
                    right_label="oncoref",
                    out_dir=out_dir,
                    key_cols=["Ensembl_Gene_ID"],
                    rtol=rtol,
                    atol=atol,
                    max_diff_rows=max_diff_rows,
                )
            )
        except Exception as e:  # noqa: BLE001
            summaries.append(
                {
                    "section": "expression",
                    "name": f"cohort_gene_percentiles__{cohort}",
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )

        try:
            pg_rep = pg_expr.representative_cohort_samples(cohort, k=1)
            oc_rep = oc.representative_cohort_samples(cohort, k=1)
            summaries.append(
                compare_frames(
                    name=f"representative_cohort_samples__{cohort}__k1",
                    left=pg_rep,
                    right=oc_rep,
                    left_label="pirlygenes",
                    right_label="oncoref",
                    out_dir=out_dir,
                    key_cols=["Ensembl_Gene_ID"],
                    rtol=rtol,
                    atol=atol,
                    max_diff_rows=max_diff_rows,
                )
            )
        except Exception as e:  # noqa: BLE001
            summaries.append(
                {
                    "section": "expression",
                    "name": f"representative_cohort_samples__{cohort}__k1",
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
            )

    if not hasattr(oc, "cancer_reference_expression"):
        summaries.append(
            {
                "section": "expression",
                "name": "cancer_reference_expression",
                "status": "missing_in_oncoref",
                "reason": "oncoref has lower-level expression accessors but no "
                "pirlygenes-compatible cancer_reference_expression API",
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oncoref-path",
        type=Path,
        default=None,
        help="Path to an oncoref checkout to put on PYTHONPATH before import. "
        "Defaults to ONCOREF_PATH, then ../oncoref when present.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument("--codes", default=None, help="Comma-separated cancer-code probes.")
    parser.add_argument("--cohorts", default=None, help="Comma-separated expression cohorts.")
    parser.add_argument("--genes", default=None, help="Comma-separated expression gene probes.")
    parser.add_argument(
        "--skip-expression",
        action="store_true",
        help="Skip expression data comparisons.",
    )
    parser.add_argument(
        "--allow-fetch",
        action="store_true",
        help="Allow oncoref expression accessors to fetch the data bundle if missing.",
    )
    parser.add_argument(
        "--max-diff-rows",
        type=int,
        default=100_000,
        help="Maximum value-diff rows to write per comparison; use 0 for unlimited.",
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_diff_rows = None if args.max_diff_rows == 0 else args.max_diff_rows
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    oncoref_path = args.oncoref_path
    if oncoref_path is None and os.environ.get("ONCOREF_PATH"):
        oncoref_path = Path(os.environ["ONCOREF_PATH"])
    if oncoref_path is None:
        sibling = REPO_ROOT.parent / "oncoref"
        if sibling.exists():
            oncoref_path = sibling

    libs = import_libraries(oncoref_path)
    codes = _csv_list(args.codes, DEFAULT_CODES)
    cohorts = _csv_list(args.cohorts, DEFAULT_COHORTS)
    genes = _csv_list(args.genes, DEFAULT_GENES)

    metadata = {
        "pirlygenes_version": getattr(importlib.import_module("pirlygenes.version"), "__version__", None),
        "pirlygenes_data_version": getattr(
            importlib.import_module("pirlygenes.version"), "DATA_VERSION", None
        ),
        "oncoref_version": getattr(importlib.import_module("oncoref.version"), "__version__", None),
        "oncoref_data_version": getattr(importlib.import_module("oncoref.version"), "DATA_VERSION", None),
        "oncoref_path": str(oncoref_path.resolve()) if oncoref_path else None,
        "out_dir": str(out_dir),
        "codes": codes,
        "cohorts": cohorts,
        "genes": genes,
        "skip_expression": bool(args.skip_expression),
        "allow_fetch": bool(args.allow_fetch),
        "rtol": args.rtol,
        "atol": args.atol,
        "max_diff_rows": max_diff_rows,
        "oncoref_bundle_status": libs.oc_bundle.status(),
    }
    _write_json(out_dir / "metadata.json", metadata)

    summaries: list[dict[str, Any]] = []
    summaries.extend(
        compare_clinical(
            libs,
            out_dir=out_dir,
            codes=codes,
            rtol=args.rtol,
            atol=args.atol,
            max_diff_rows=max_diff_rows,
        )
    )
    summaries.extend(
        compare_normalization(
            libs,
            out_dir=out_dir,
            rtol=args.rtol,
            atol=args.atol,
            max_diff_rows=max_diff_rows,
        )
    )
    if not args.skip_expression:
        summaries.extend(
            compare_expression(
                libs,
                out_dir=out_dir,
                genes=genes,
                cohorts=cohorts,
                allow_fetch=args.allow_fetch,
                rtol=args.rtol,
                atol=args.atol,
                max_diff_rows=max_diff_rows,
            )
        )

    summary_df = pd.DataFrame(summaries)
    _write_df(out_dir / "summary.csv", summary_df)
    _write_json(out_dir / "summary.json", summaries)

    n_diff = int((summary_df["status"] != "ok").sum()) if "status" in summary_df else 0
    print(f"wrote comparison outputs to {out_dir}")
    print(f"comparisons: {len(summary_df)} total, {n_diff} non-ok")
    if n_diff:
        cols = [c for c in ("section", "name", "status", "diff_rows", "value_diff_cells") if c in summary_df]
        print(summary_df.loc[summary_df["status"] != "ok", cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
