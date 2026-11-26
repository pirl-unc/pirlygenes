"""Historical 10-paper CTA curation figures owned by oncoref; preserves the CLI API.

``render()`` and the funnel/overlap helpers wrap :mod:`oncoref.cta_curation_plots`,
which reproduces the frozen v1 HPA paper-intake curation figures (624 historical
HPA-filtered genes). Active CTA target selection and manifest provenance come
from :mod:`pirlygenes.cta_policy` (:mod:`oncoref.target_selection`); inspect
:func:`active_cta_policy_manifest` for the live OncoRef policy receipt.
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

from .cta_policy import CTA_policy_manifest as active_cta_policy_manifest

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
    'active_cta_policy_manifest',
    'placental_source_sets',
    'publication_data_available',
    'render',
    'source_overlap_counts',
    'stage_counts',
    'stage_membership',
]
