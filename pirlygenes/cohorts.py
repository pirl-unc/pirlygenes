"""Compatibility access to oncoref-owned per-sample expression matrices.

Pirlygenes used to own a second registry, cache layout, and writer for the
per-cohort matrices used by patient-coverage analyses.  Oncoref now publishes
the canonical matrix registry and individually fetchable parquet artifacts.
This module preserves pirlygenes' small ``Cohort``/reader surface while routing
all real cohort discovery and I/O through :mod:`oncoref.source_matrices`.

No function in this module writes reference data.  Source regeneration belongs
to :mod:`oncoref.expression_builders`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

PER_SAMPLE_SUFFIX = "_per_sample_tpm.parquet"
ID_COLS = ("Ensembl_Gene_ID", "Symbol")


@dataclass(frozen=True)
class Cohort:
    """One oncoref-owned per-sample cohort.

    ``stem`` and the build-definition fields remain for source compatibility;
    owner artifacts are addressed by ``code`` and do not depend on them.
    """

    code: str
    stem: str
    source_id: str
    group: str = ""
    disease_label: str = ""
    selection: str = ""

    @property
    def source_kind(self) -> str:
        return self.source_id.split("-")[0]

    @property
    def atom(self) -> str:
        return f"{self.source_kind}:{self.code}"


# Source IDs retained for callers of the historical source-filtered coverage
# API.  These source-specific routes are served by oncoref's code-keyed matrix
# registry; the mapping is compatibility metadata, not a second matrix registry.
_LEGACY_SOURCE_BY_COHORT = {
    "TARGET_ALL_2018": "target-all",
    "CLLMAP_2022": "cllmap",
    "GSE171811_ECCITE_CTCL": "gse171811-ctcl",
    "BEATAML_OHSU_2022": "beataml-ohsu",
    "TARGET_NBL_2018": "target-nbl",
    "TARGET_RT_2017": "target-rt",
    "TARGET_WT_2015": "target-wt",
    "GSE299759_MEIJER_2026": "gse299759-chon",
    "GSE239531_VANOOST_2024": "gse239531-chordoma",
    "GSE75885_DELESPAUL_2017": "gse75885-sarc",
    "SCLC_UCOLOGNE_2015": "sclc-ucologne-2015",
}

_LEGACY_SOURCE_PROJECT = {
    "target-all": "TARGET",
    "cllmap": "CLL-map",
    "gse171811-ctcl": "GEO",
    "beataml-ohsu": "BeatAML",
    "target-nbl": "TARGET",
    "target-rt": "TARGET",
    "target-wt": "TARGET",
    "gse299759-chon": "GEO",
    "gse239531-chordoma": "GEO",
    "gse75885-sarc": "GEO",
    "sclc-ucologne-2015": "University of Cologne",
}


def _owner_registry() -> pd.DataFrame:
    from oncoref import source_matrices

    return source_matrices.registry()


@lru_cache(maxsize=1)
def _owner_sources():
    from oncoref.expression_registry import expression_sources

    return expression_sources()


def _source_for_row(code: str, source_cohort: str):
    matches = [
        source
        for source in _owner_sources()
        if code in source.cancer_codes and source.source_cohort == source_cohort
    ]
    return matches[0] if matches else None


def _source_id_for_row(code: str, source_cohort: str) -> str:
    source = _source_for_row(code, source_cohort)
    if source is not None:
        return source.id
    return _LEGACY_SOURCE_BY_COHORT.get(source_cohort, source_cohort.lower())


def _source_matches(requested: str, actual: str) -> bool:
    if requested == actual:
        return True
    # The historical pirlygenes Treehouse source grouped all PolyA-derived
    # selectors under one source ID. Preserve that convenient filter while
    # allowing oncoref's more precise registry IDs.
    return (
        requested == "treehouse-polya-25-01"
        and actual.startswith("treehouse-polya-25-01")
    )


def _cohort_from_row(row) -> Cohort:
    code = str(row["cancer_code"])
    source_cohort = str(row["source_cohort"])
    return Cohort(
        code=code,
        stem=code,
        source_id=_source_id_for_row(code, source_cohort),
    )


def _per_sample_sources() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for source in _owner_sources():
        if not source.cancer_codes:
            continue
        label = source.source_cohort or source.id
        project = source.source_project or ""
        out[source.id] = (label, project)
    for cohort, source_id in _LEGACY_SOURCE_BY_COHORT.items():
        out.setdefault(
            source_id,
            (cohort, _LEGACY_SOURCE_PROJECT.get(source_id, "")),
        )
    return out


# Backwards-compatible inspection mapping. It is derived from oncoref at import
# time and contains no pirlygenes-owned file or build path.
PER_SAMPLE_SOURCES: dict[str, tuple[str, str]] = _per_sample_sources()


def cohorts_for_group(group: str) -> list[Cohort]:
    """Return Treehouse build-group definitions from oncoref's registry.

    This compatibility helper is read-only; rebuilding remains an oncoref task.
    """
    from oncoref.expression_builders import (
        treehouse_source_entries,
        treehouse_source_from_entry,
    )

    out: list[Cohort] = []
    for entry in treehouse_source_entries():
        source = treehouse_source_from_entry(entry)
        for cohort in source.cohorts:
            if cohort.group != group:
                continue
            out.append(
                Cohort(
                    code=cohort.cancer_code,
                    stem=cohort.cache_stem or cohort.cancer_code,
                    source_id=source.source_id,
                    group=cohort.group,
                    disease_label=cohort.disease_label,
                    selection=cohort.selection,
                )
            )
    return out


def source_label(source_id: str) -> str | None:
    entry = PER_SAMPLE_SOURCES.get(source_id)
    return entry[0] if entry else None


def source_project(source_id: str) -> str | None:
    entry = PER_SAMPLE_SOURCES.get(source_id)
    return entry[1] if entry else None


def cohorts_for_source(
    source_id: str,
    *,
    include_related: bool = True,
) -> dict[str, Cohort]:
    """Owner cohorts belonging to a source ID.

    ``include_related`` preserves pirlygenes' historical Treehouse behavior:
    asking for the base PolyA source also includes its subtype/selection source
    IDs. Cache accounting sets it false so one physical artifact is never
    charged to two registry entries.
    """
    out: dict[str, Cohort] = {}
    for _, row in _owner_registry().iterrows():
        cohort = _cohort_from_row(row)
        matches = (
            _source_matches(source_id, cohort.source_id)
            if include_related
            else source_id == cohort.source_id
        )
        if matches:
            out[cohort.code] = cohort
    return out


def parquet_path(cohort: Cohort):
    """The oncoref cache path for ``cohort`` (it may not be fetched yet)."""
    from oncoref import source_matrices

    return source_matrices.local_path(cohort.code)


def available_cohorts(source_id: str) -> dict[str, Cohort]:
    """Cohorts already present in oncoref's local source-matrix cache."""
    from oncoref import source_matrices

    return {
        code: cohort
        for code, cohort in cohorts_for_source(source_id).items()
        if source_matrices.is_cached(code)
    }


def all_available_cohorts() -> dict[str, Cohort]:
    """Every cohort already present in oncoref's local matrix cache."""
    from oncoref import source_matrices

    return {
        str(row["cancer_code"]): _cohort_from_row(row)
        for _, row in _owner_registry().iterrows()
        if source_matrices.is_cached(str(row["cancer_code"]))
    }


def sample_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in ID_COLS]


def read_per_sample(cohort: Cohort) -> pd.DataFrame:
    """Read a cohort matrix, fetching its owner artifact on first use."""
    from oncoref import source_matrices

    return pd.read_parquet(source_matrices.ensure(cohort.code))


def _parquet_sample_count(cohort: Cohort) -> int:
    from oncoref import source_matrices

    try:
        return int(source_matrices.cohort_info(cohort.code)["n_samples"])
    except (KeyError, TypeError, ValueError):
        import pyarrow.parquet as pq

        return max(
            0,
            pq.read_metadata(parquet_path(cohort)).num_columns - len(ID_COLS),
        )


def iter_per_sample_cohorts(*, sources=None, unique_by_code=True):
    """Yield cached owner matrices as ``(Cohort, DataFrame)`` pairs.

    Oncoref publishes one selected matrix per cancer code, so
    ``unique_by_code`` is retained only for call compatibility.
    """
    wanted = None if sources is None else set(sources)
    seen: set[str] = set()
    for code, cohort in all_available_cohorts().items():
        if wanted is not None and not any(
            _source_matches(source_id, cohort.source_id)
            for source_id in wanted
        ):
            continue
        if unique_by_code and code in seen:
            continue
        seen.add(code)
        yield cohort, read_per_sample(cohort)


# Historical private snapshots retained for callers that inspected them.
_PER_SAMPLE_COHORTS: tuple[Cohort, ...] = tuple(
    _cohort_from_row(row) for _, row in _owner_registry().iterrows()
)
_TREEHOUSE_COHORTS: tuple[Cohort, ...] = tuple(
    cohort
    for cohort in _PER_SAMPLE_COHORTS
    if cohort.source_id.startswith("treehouse-")
)
