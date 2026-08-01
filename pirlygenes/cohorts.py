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
    # This historical pirlygenes ID predates oncoref's current
    # ``beataml-ohsu-2022`` acquisition ID. Current owner sources always use
    # their authoritative source_project; only the retired read alias needs a
    # compatibility display value.
    "beataml-ohsu": "BeatAML",
}

# Historical composite filters that never represented one physical dataset.
# Keep them only as read-side aliases over the independently owned oncoref
# sources; never emit the composite label as matrix provenance.
_LEGACY_COMPOSITE_SOURCES = {
    "geo-heme": (
        "gse100026-cml",
        "gse114922-mds",
        "gse271664-mcl",
        "gse283710-mpn",
    ),
}

_LEGACY_COMPOSITE_METADATA = {
    "geo-heme": ("GEO_HEME_2022", "GEO"),
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


def _cohort_from_row(row) -> Cohort:
    code = str(row["cancer_code"])
    source_cohort = str(row["source_cohort"])
    return Cohort(
        code=code,
        stem=code,
        source_id=_source_id_for_row(code, source_cohort),
    )


def _per_sample_sources() -> dict[str, tuple[str, str]]:
    from oncoref import source_matrices

    out: dict[str, tuple[str, str]] = {}
    for source in _owner_sources():
        if not source.cancer_codes:
            continue
        label = source.source_cohort
        if not label:
            try:
                resolution = source_matrices.resolution_for_source(source.id)
            except source_matrices.SourceMatrixError:
                resolution = None
            selected_labels = {
                matrix.source_cohort
                for matrix in resolution.matrices
            } if resolution is not None else set()
            if len(selected_labels) == 1:
                label = selected_labels.pop()
        label = label or source.id
        compatibility_id = _LEGACY_SOURCE_BY_COHORT.get(label, source.id)
        project = (
            source.source_project
            or _LEGACY_SOURCE_PROJECT.get(compatibility_id, "")
        )
        out[source.id] = (label, project)
    for cohort, source_id in _LEGACY_SOURCE_BY_COHORT.items():
        # Some owner registry entries intentionally omit display provenance
        # (for example TARGET ALL and CLL-map). Preserve complete owner values,
        # but fill each absent field from pirlygenes' compatibility metadata.
        # A placeholder source ID must not shadow a known public cohort label.
        label, project = out.get(source_id, ("", ""))
        if not label or label == source_id:
            label = cohort
        if not project:
            project = _LEGACY_SOURCE_PROJECT.get(source_id, "")
        out[source_id] = (label, project)
    for source_id, metadata in _LEGACY_COMPOSITE_METADATA.items():
        out.setdefault(source_id, metadata)
    return out


# Backwards-compatible inspection mapping. It is derived from oncoref at import
# time and contains no pirlygenes-owned file or build path.
PER_SAMPLE_SOURCES: dict[str, tuple[str, str]] = _per_sample_sources()


@lru_cache(maxsize=1)
def _legacy_source_routes() -> dict[str, tuple[str, ...]]:
    """Map historical source IDs to current owner routes by provenance.

    Oncoref source IDs describe acquisition/build routes, while pirlygenes'
    older IDs sometimes named the selected matrix cohort. Resolve that older
    vocabulary through oncoref's public source-to-matrix resolver instead of
    maintaining duplicate old-to-new source-ID aliases here.
    """
    from oncoref import source_matrices

    owner_ids_by_cohort: dict[str, list[str]] = {}
    for source in _owner_sources():
        try:
            resolution = source_matrices.resolution_for_source(source.id)
        except source_matrices.SourceMatrixError:
            continue
        for matrix in resolution.matrices:
            ids = owner_ids_by_cohort.setdefault(matrix.source_cohort, [])
            if source.id not in ids:
                ids.append(source.id)

    routes: dict[str, tuple[str, ...]] = {}
    for cohort, legacy_source_id in _LEGACY_SOURCE_BY_COHORT.items():
        owner_ids = owner_ids_by_cohort.get(cohort, [])
        if owner_ids:
            routes[legacy_source_id] = tuple(owner_ids)
    return routes


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
    from oncoref import source_matrices

    owner_sources = _owner_sources()
    owner_ids = {source.id for source in owner_sources}

    if source_id in _LEGACY_COMPOSITE_SOURCES:
        source_ids = (
            _LEGACY_COMPOSITE_SOURCES[source_id]
            if include_related
            else ()
        )
    elif source_id in owner_ids and include_related:
        # Historical pirlygenes filters treated a base source ID as including
        # its more specific selection/subtype routes. Preserve that general
        # prefix relationship while asking oncoref to resolve every underlying
        # owner source independently.
        source_ids = tuple(
            source.id
            for source in owner_sources
            if source.id == source_id
            or source.id.startswith(f"{source_id}-")
        )
    elif source_id in owner_ids:
        source_ids = (source_id,)
    else:
        # A few pre-oncoref source IDs named the physical cohort rather than
        # the current acquisition route. Resolve those compatibility aliases
        # through owner metadata, then use the same owner resolver as every
        # current source ID.
        source_ids = _legacy_source_routes().get(source_id, ())

    wanted_codes: list[str] = []
    seen: set[str] = set()
    for resolved_source_id in source_ids:
        try:
            codes = source_matrices.codes_for_source(resolved_source_id)
        except source_matrices.SourceMatrixError:
            continue
        for code in codes:
            if code not in seen:
                wanted_codes.append(code)
                seen.add(code)

    registry_by_code = {
        str(row["cancer_code"]): row
        for _, row in _owner_registry().iterrows()
    }
    if include_related:
        selected_rows = [
            registry_by_code[code]
            for code in wanted_codes
            if code in registry_by_code
        ]
        selected_cohorts = {
            str(row["source_cohort"]) for row in selected_rows
        }
        for code, row in registry_by_code.items():
            if (
                code not in seen
                and any(
                    code.startswith(f"{selected}_")
                    or selected.startswith(f"{code}_")
                    for selected in wanted_codes
                )
                and str(row["source_cohort"]) in selected_cohorts
            ):
                wanted_codes.append(code)
                seen.add(code)

    return {
        code: _cohort_from_row(registry_by_code[code])
        for code in wanted_codes
        if code in registry_by_code
    }


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
    if sources is None:
        wanted_codes = None
    else:
        requested = (sources,) if isinstance(sources, str) else tuple(sources)
        wanted_codes = {
            code
            for source_id in requested
            for code in cohorts_for_source(source_id)
        }
    seen: set[str] = set()
    for code, cohort in all_available_cohorts().items():
        if wanted_codes is not None and code not in wanted_codes:
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
