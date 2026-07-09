"""Split a parent TCGA cohort's per-sample TPM into molecular-subtype cohorts.

Generalizes the COAD/READ MSI split (``build_tcga_coadread_msi_split.py``) so the
UCEC (POLE / MSI / CN-low / CN-high) and STAD (EBV / MSI / GS / CIN) subtype
builders share one path: join a per-case cBioPortal molecular-subtype label onto
the Treehouse per-sample parquet that already exists (no compendium re-sweep),
partition the sample columns by target cancer code, write one per-sample parquet
per subtype, and upsert per-cancer-code summary shards.

The label map (``{case_id: cbioportal_subtype_value}``) is produced by the
per-tissue ``sweep_treehouse_tcga_*_subtype.py`` fetchers; ``code_by_label`` maps
each cBioPortal SUBTYPE string to a pirlygenes cancer code. A sample whose case
is unlabeled, or whose label isn't in ``code_by_label`` (an excluded subtype), is
dropped — the same drop-unclassified behavior the MSI split uses.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from pirlygenes import cohorts as _cohorts
from pirlygenes.expression.stats import write_reference_rows


def case_id(sample_col: str) -> str:
    """TCGA case submitter-id — the first three barcode fields of a sample col.

    cBioPortal label maps are keyed by patient/case id (``TCGA-XX-XXXX``);
    Treehouse sample columns carry the fuller aliquot barcode, so both sides
    join on this prefix (the same key the COAD/READ MSI split uses)."""
    return "-".join(str(sample_col).split("-")[:3])


def group_samples_by_code(
    df: pd.DataFrame,
    *,
    label_by_case: dict[str, str],
    code_by_label: dict[str, str],
) -> dict[str, list[str]]:
    """Partition a per-sample frame's sample columns into target cancer codes.

    A sample whose case has no label, or whose label isn't in ``code_by_label``,
    is dropped (unclassified case / deliberately-excluded subtype)."""
    by_code: dict[str, list[str]] = {}
    for col in _cohorts.sample_columns(df):
        label = label_by_case.get(case_id(col))
        code = code_by_label.get(label) if label is not None else None
        if code:
            by_code.setdefault(code, []).append(col)
    return by_code


def build_subtype_split(
    *,
    source_id: str,
    parent_code: str,
    label_by_case: dict[str, str],
    code_by_label: dict[str, str],
    summary_cohort: str,
    summary_output,
    make_summary_row: Callable[[pd.DataFrame, pd.DataFrame, str], pd.DataFrame],
    expected_codes: set[str] | None = None,
) -> list[str]:
    """Read the parent parquet, split by label, write per-sample parquets +
    per-cancer-code summary shards. Returns the cancer codes written.

    ``make_summary_row(gene_table, values, code)`` builds one cohort's reference
    rows (typically via :func:`pirlygenes.expression.stats.build_reference_rows`).
    ``expected_codes`` (default: every value of ``code_by_label``) drives a loud
    warning for any target code that matched zero samples, so a dangling
    registry/medoid entry can't slip through silently (mirrors the MSI split).
    """
    cohort = _cohorts.cohorts_for_source(source_id)[parent_code]
    df = _cohorts.read_per_sample(cohort)
    gene_table = df[list(_cohorts.ID_COLS)]
    by_code = group_samples_by_code(
        df, label_by_case=label_by_case, code_by_label=code_by_label)

    summaries: list[pd.DataFrame] = []
    written: list[str] = []
    for code, cols in by_code.items():
        _cohorts.write_per_sample(gene_table, df[cols], source_id, code)
        summaries.append(make_summary_row(gene_table, df[cols], code))
        written.append(code)
        print(f"  {code}: {len(cols)} samples", flush=True)

    missing = sorted((expected_codes or set(code_by_label.values())) - set(written))
    if missing:
        print(f"  WARNING: no samples matched for {missing} (parent "
              f"{parent_code}); target cohort(s) left unbuilt", flush=True)
    if summaries:
        write_reference_rows(summary_output, pd.concat(summaries, ignore_index=True),
                        source_cohort=summary_cohort, cancer_codes=written,
                        per_cancer_code_shards=True)
    return written
