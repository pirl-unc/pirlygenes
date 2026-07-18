# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public source-cohort identities for delegated reference expression.

The physical oncoref summary artifact can temporarily retain an older cohort
label after its registry adopts a more specific public identity.  Overrides
are keyed by both cancer code and stored cohort: a shared physical cohort must
not relabel unrelated cancer types.  Every pirlygenes read path uses these
helpers so frames, filters, metadata, and inventory expose one contract.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


_PUBLIC_SOURCE_COHORT_BY_STORED_PAIR = {
    (
        "SARC_DDLPS",
        "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
    ): "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
    (
        "SARC_WDLPS",
        "TREEHOUSE_POLYA_25_01_TCGA_SUBSET",
    ): "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY",
}


def canonical_reference_source_cohort(cancer_code: object, source_cohort: object) -> str:
    """Return the public cohort label for one code/stored-label pair."""
    key = (str(cancer_code), str(source_cohort))
    return _PUBLIC_SOURCE_COHORT_BY_STORED_PAIR.get(key, key[1])


def normalize_reference_source_cohort_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Normalize known stored labels without mutating the delegated frame."""
    if not {"cancer_code", "source_cohort"} <= set(df.columns):
        return df, False

    replacement_masks: list[tuple[pd.Series, str]] = []
    for (code, stored), public in _PUBLIC_SOURCE_COHORT_BY_STORED_PAIR.items():
        mask = df["cancer_code"].eq(code) & df["source_cohort"].eq(stored)
        if mask.any():
            replacement_masks.append((mask, public))
    if not replacement_masks:
        return df, False

    out = df.copy(deep=False)
    labels = df["source_cohort"].astype(object).copy()
    for mask, public in replacement_masks:
        labels.loc[mask] = public
    out["source_cohort"] = labels
    return out, True


def normalize_reference_source_cohort_records(
    records: Iterable[dict],
) -> tuple[list[dict], bool]:
    """Apply the same pair normalization to availability-style records."""
    out: list[dict] = []
    changed = False
    for record in records:
        adapted = dict(record)
        stored = adapted.get("source_cohort")
        public = canonical_reference_source_cohort(
            adapted.get("cancer_code", ""),
            stored,
        )
        if stored is not None and str(stored) != public:
            adapted["source_cohort"] = public
            changed = True
        out.append(adapted)
    return out, changed


def reference_source_cohort_storage_filter(source_cohort):
    """Expand public/storage aliases accepted across oncoref data versions.

    oncoref <=1.8.129 filters the affected SARC summary rows under their
    historical physical label, while >=1.8.130 canonicalizes those rows before
    filtering.  Forward both equivalent labels so pirlygenes remains compatible
    with either representation.  The exact public semantics are applied after
    delegation by :func:`reference_source_cohort_public_filter`.
    """
    if source_cohort is None:
        return None
    scalar = isinstance(source_cohort, str)
    requested = [source_cohort] if scalar else list(source_cohort)
    aliases: dict[str, list[str]] = {}
    for (_, stored), public in _PUBLIC_SOURCE_COHORT_BY_STORED_PAIR.items():
        aliases.setdefault(stored, [])
        aliases.setdefault(public, [])
        if public not in aliases[stored]:
            aliases[stored].append(public)
        if stored not in aliases[public]:
            aliases[public].append(stored)

    translated: list[str] = []
    for cohort in requested:
        value = str(cohort)
        for candidate in (value, *aliases.get(value, [])):
            if candidate not in translated:
                translated.append(candidate)
    return translated[0] if scalar and len(translated) == 1 else translated


def reference_source_cohort_public_filter(source_cohort):
    """Return canonical labels accepted for an exact public source filter.

    A request using a historical storage label remains a compatibility alias
    for rows that now expose a more specific public label.  A request using the
    canonical label stays exact and does not admit unrelated rows that still
    share the physical storage cohort.
    """
    if source_cohort is None:
        return None
    requested = (
        {source_cohort}
        if isinstance(source_cohort, str)
        else set(source_cohort)
    )
    accepted = {str(cohort) for cohort in requested}
    for (_, stored), public in _PUBLIC_SOURCE_COHORT_BY_STORED_PAIR.items():
        if stored in accepted:
            accepted.add(public)
    return frozenset(accepted)


__all__ = [
    "canonical_reference_source_cohort",
    "normalize_reference_source_cohort_labels",
    "normalize_reference_source_cohort_records",
    "reference_source_cohort_public_filter",
    "reference_source_cohort_storage_filter",
]
