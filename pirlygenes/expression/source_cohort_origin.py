"""``source_cohort → (tumor_origin, metastasis_site)`` classification.

For the v5.4 schema migration the per-source-cohort classification
lives here so the backfill script, future audit scripts, and any
``pirlygenes`` consumer can share one source of truth.

Two layers, evaluated in this order:

1. **YAML overlay** — ``pirlygenes/data/expression_sources.yaml`` entries
   with an explicit ``tumor_origin:`` (and optional ``metastasis_site:``
   + ``source_cohort:``) field win. Builders that emit non-primary
   shards declare their classification here.

2. **Hardcoded fallback** — :data:`PRIMARY_SOURCES`,
   :data:`MIXED_SOURCES`, :data:`SELF_ANNOTATED_SOURCES`. Used for
   legacy shards whose YAML entries predate the v5.4 schema and don't
   carry a ``tumor_origin`` field yet.

New builders should:
- Set ``tumor_origin`` (and ``metastasis_site`` when applicable) on
  every emitted row. :func:`pirlygenes.expression.stats.write_reference_rows`
  validates this and rejects unset / unrecognised values.
- Add ``tumor_origin:`` to the YAML source entry too, so the value is
  declarative (separate from per-row data).

The backfill script is then only relevant for one-time schema-migration
passes against legacy data.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import yaml


PRIMARY_SOURCES: frozenset[str] = frozenset({
    "BEATAML_OHSU_2022",
    "CGCI_BLGSP",
    "CLLMAP_2022",
    "DRMETRICS_ALCALA_2019_LNEN",
    "GSE100026_DING_2017",
    "GSE114922_SHIOZAWA_2018",
    "GSE118014_ALVAREZ_2018",
    "GSE120328_LAMPRECHT_2018",
    "GSE142334_FL_TFL_2021",
    "GSE171811_ECCITE_CTCL",
    "GSE241095_KS_SKIN_2023",
    "GSE248751_HUMAN_CCS_2023",
    "GSE271664_BODOR_2025",
    "GSE283710_WASHU_2024",
    "GSE294016_BARTL_2025_SGC",
    "GSE299759_MEIJER_2026",
    "GSE328026_PECOMA_2026",
    "GSE75885_DELESPAUL_2017",
    "MMRF_COMMPASS",
    "SCLC_UCOLOGNE_2015",
    "SCLC_UCOLOGNE_2015_TF_DOMINANCE",
    "TARGET_ALL_2018",
    "TARGET_RT_2017",
    "TARGET_WT_2015",
})

MIXED_SOURCES: frozenset[str] = frozenset({
    "TARGET_NBL_2018",
    "TREEHOUSE_POLYA_25_01",
    "TREEHOUSE_RIBOD_25_01",
    "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES",
    "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
    "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV",
    "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
    "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS",
})

# Sources whose builder already sets ``tumor_origin`` per-row (and may
# vary it between cancer_codes inside the same shard). The classifier
# returns ``None`` for these so any caller knows to leave existing
# row-level values intact.
SELF_ANNOTATED_SOURCES: frozenset[str] = frozenset({
    "GSE98894_ALVAREZ_2018_NET",
})


_DEFAULT_YAML_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "expression_sources.yaml"
)

# Mutable so :func:`set_yaml_path` (a test hook) can redirect lookups
# at the YAML registry without monkey-patching a private constant.
# Production code should NOT mutate this — use ``set_yaml_path`` from
# a test, paired with ``clear_cache()``.
_YAML_PATH = _DEFAULT_YAML_PATH


def set_yaml_path(path: Path | None) -> None:
    """Redirect (or reset) the YAML registry path used by the classifier.

    Test hook: a test that wants to swap in a fixture YAML calls
    ``set_yaml_path(tmp_path / "fake.yaml")`` then ``clear_cache()``,
    and resets via ``set_yaml_path(None)`` (or ``clear_cache()``-only
    if relying on autouse-fixture cleanup) at teardown.
    Passing ``None`` restores the bundled default.
    """
    global _YAML_PATH
    _YAML_PATH = _DEFAULT_YAML_PATH if path is None else Path(path)
    clear_cache()


@lru_cache(maxsize=1)
def _yaml_overlay() -> dict[str, tuple[str, str | None]]:
    """Return ``{source_cohort: (tumor_origin, metastasis_site_or_None)}``
    parsed from any YAML source that declares those fields.

    Entries that declare ``tumor_origin:`` but no ``source_cohort:``
    are skipped (the classifier is source_cohort-keyed) — and a
    ``UserWarning`` is emitted listing them, so a contributor who
    half-completes the YAML schema sees an explicit signal rather
    than silent inaction.
    """
    if not _YAML_PATH.exists():
        return {}
    payload = yaml.safe_load(_YAML_PATH.read_text()) or {}
    out: dict[str, tuple[str, str | None]] = {}
    incomplete: list[str] = []
    for entry in payload.get("sources", []) or []:
        origin = entry.get("tumor_origin")
        if not origin:
            continue
        cohort = entry.get("source_cohort")
        if not cohort:
            # tumor_origin declared but source_cohort missing — the
            # classifier can't apply this, so warn the contributor.
            entry_id = str(entry.get("id", "<no id>"))
            incomplete.append(entry_id)
            continue
        out[str(cohort)] = (
            str(origin),
            (str(entry["metastasis_site"]) if entry.get("metastasis_site") else None),
        )
    if incomplete:
        # stacklevel=3 so the warning points past _yaml_overlay() AND
        # past classify_source_cohort() at the actual caller (a
        # builder, test, or notebook) — that's the audience who needs
        # to see what triggered the lookup.
        warnings.warn(
            "expression_sources.yaml: entries with tumor_origin: but "
            f"no source_cohort: are ignored by classify_source_cohort: "
            f"{incomplete}. Add `source_cohort: <SHARD_NAME>` to make "
            "the classification apply.",
            UserWarning,
            stacklevel=3,
        )
    return out


def classify_source_cohort(
    source_cohort: str,
) -> tuple[str | None, str | None]:
    """Return ``(tumor_origin, metastasis_site)`` for the given source_cohort.

    Returns ``(None, None)`` when the cohort is self-annotated (the
    builder sets per-row values) OR truly unclassified — callers
    distinguish the two via the :data:`SELF_ANNOTATED_SOURCES` set.
    """
    overlay = _yaml_overlay()
    if source_cohort in overlay:
        return overlay[source_cohort]
    if source_cohort in SELF_ANNOTATED_SOURCES:
        return (None, None)
    if source_cohort in PRIMARY_SOURCES:
        return ("primary", None)
    if source_cohort in MIXED_SOURCES:
        return ("mixed", None)
    return (None, None)


def known_source_cohorts() -> frozenset[str]:
    """Every source_cohort classified by any of the above maps."""
    return frozenset(
        _yaml_overlay()
    ) | PRIMARY_SOURCES | MIXED_SOURCES | SELF_ANNOTATED_SOURCES


def clear_cache() -> None:
    """Drop the YAML-overlay cache.

    Tests that monkey-patch ``_YAML_PATH`` (or replace the YAML file
    contents on disk) need to call this so the next
    :func:`classify_source_cohort` lookup re-reads from the new
    location. Matches the public ``clear_cache()`` pattern that
    :class:`pirlygenes.gene_sets_cancer._CancerTypeNamesView` uses
    for the same purpose.
    """
    _yaml_overlay.cache_clear()


__all__ = [
    "PRIMARY_SOURCES",
    "MIXED_SOURCES",
    "SELF_ANNOTATED_SOURCES",
    "classify_source_cohort",
    "known_source_cohorts",
    "clear_cache",
    "set_yaml_path",
]
