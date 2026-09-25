"""CTA plots and provenance owned by oncoref; this module preserves the CLI API.

Both repositories render the same complete paper union and public default mask.
The exact oncoref dependency pin is the versioned figure/data contract.
"""
from oncoref.cta_curation_plots import (
    FILENAMES,
    PRIMARY_SOURCES,
    RELIABILITY_ORDER,
    RELIABILITY_THRESHOLD,
    _bool_series,
    _evidence,
    _per_source_counts,
    _tag_sets,
    placental_source_sets,
    render,
    source_overlap_counts,
)
from oncoref.cta_curation_plots import (
    stage_counts as _owner_stage_counts,
)
from oncoref.cta_curation_plots import (
    stage_membership as _owner_stage_membership,
)

# Kept for callers predating the mandatory owner provenance tables.
PUBLICATION_FILENAMES = {}


def publication_data_available():
    return True


def stage_counts():
    return [dict(zip(("stage", "remaining", "dropped"), row)) for row in _owner_stage_counts()]


def stage_membership(df=None):
    return _owner_stage_membership(df).rename(columns={"family_eligible": "non_cta_removed"})


__all__ = [
    'FILENAMES',
    'PRIMARY_SOURCES',
    'PUBLICATION_FILENAMES',
    'RELIABILITY_ORDER',
    'RELIABILITY_THRESHOLD',
    '_bool_series',
    '_evidence',
    '_per_source_counts',
    '_tag_sets',
    'placental_source_sets',
    'publication_data_available',
    'render',
    'source_overlap_counts',
    'stage_counts',
    'stage_membership',
]
