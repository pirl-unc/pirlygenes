"""Tests for the source_cohort → tumor_origin classifier + cache.

Covers the YAML-overlay path (where most tests' coverage was previously
absent — only the hardcoded fallback was exercised end-to-end via the
backfilled bundled shards).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pirlygenes.expression import source_cohort_origin as sco


@pytest.fixture(autouse=True)
def _reset_origin_cache():
    """Restore the bundled YAML path + drop the overlay cache around
    every test, so swapping ``_YAML_PATH`` in one test can never leak
    state into a sibling. Runs even if the test body raises."""
    sco.set_yaml_path(None)
    yield
    sco.set_yaml_path(None)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "expression_sources.yaml"
    path.write_text(body)
    sco.set_yaml_path(path)
    return path


# ─── YAML overlay ─────────────────────────────────────────────────────


def test_yaml_overlay_warns_when_tumor_origin_set_but_source_cohort_missing(
    tmp_path,
):
    """A YAML entry that sets ``tumor_origin:`` but forgets
    ``source_cohort:`` is silently un-classifiable — emit a UserWarning
    listing the offending entries so a contributor catches it."""
    _write_yaml(tmp_path,
        "sources:\n"
        "  - id: complete-entry\n"
        "    source_cohort: GOOD_COHORT\n"
        "    tumor_origin: primary\n"
        "  - id: incomplete-entry\n"
        "    tumor_origin: metastasis\n"   # missing source_cohort
        "    metastasis_site: liver\n"
    )

    with pytest.warns(UserWarning, match="incomplete-entry"):
        # Force the lookup to populate the cache (and run the warn pass).
        sco.classify_source_cohort("GOOD_COHORT")


def test_yaml_overlay_picks_up_complete_entries(tmp_path):
    """A YAML entry with both fields populates the overlay and
    classify_source_cohort returns its values (overriding any
    hardcoded fallback)."""
    _write_yaml(tmp_path,
        "sources:\n"
        "  - id: brand-new\n"
        "    source_cohort: BRAND_NEW_COHORT\n"
        "    tumor_origin: recurrence\n"
    )
    origin, met = sco.classify_source_cohort("BRAND_NEW_COHORT")
    assert origin == "recurrence"
    assert met is None


def test_yaml_overlay_caches_first_read(tmp_path):
    """Without ``clear_cache``, an on-disk YAML edit is NOT picked up —
    the lru_cache returns the first-read overlay. This locks in the
    cache semantics consumers depend on for performance."""
    yaml_path = _write_yaml(tmp_path,
        "sources:\n"
        "  - id: a\n"
        "    source_cohort: COHORT_A\n"
        "    tumor_origin: primary\n"
    )
    assert sco.classify_source_cohort("COHORT_A") == ("primary", None)

    # Overwrite the YAML — the overlay should still serve from cache.
    yaml_path.write_text(
        "sources:\n"
        "  - id: b\n"
        "    source_cohort: COHORT_B\n"
        "    tumor_origin: metastasis\n"
    )
    assert sco.classify_source_cohort("COHORT_A") == ("primary", None)


def test_clear_cache_observably_drops_overlay(tmp_path):
    """After ``clear_cache()``, the next lookup re-reads the YAML so
    on-disk edits become visible. Matches the test-friendly pattern
    used by ``_CancerTypeNamesView``."""
    yaml_path = _write_yaml(tmp_path,
        "sources:\n"
        "  - id: a\n"
        "    source_cohort: COHORT_A\n"
        "    tumor_origin: primary\n"
    )
    assert sco.classify_source_cohort("COHORT_A") == ("primary", None)

    yaml_path.write_text(
        "sources:\n"
        "  - id: b\n"
        "    source_cohort: COHORT_B\n"
        "    tumor_origin: metastasis\n"
        "    metastasis_site: bone\n"
    )
    sco.clear_cache()
    assert sco.classify_source_cohort("COHORT_A") == (None, None)
    assert sco.classify_source_cohort("COHORT_B") == ("metastasis", "bone")


def test_set_yaml_path_none_restores_default(tmp_path):
    """``set_yaml_path(None)`` flips back to the bundled
    ``expression_sources.yaml`` — confirmed by looking up a known
    PRIMARY_SOURCES entry, which the bundled YAML's classify pipeline
    should match."""
    _write_yaml(tmp_path, "sources: []\n")
    # Bundled lookup hidden by the empty override
    primary_example = next(iter(sco.PRIMARY_SOURCES))
    # Falls back through the hardcoded set even with empty YAML
    assert sco.classify_source_cohort(primary_example) == ("primary", None)

    sco.set_yaml_path(None)   # restore default
    # Still classified — but now via either YAML or hardcoded path
    assert sco.classify_source_cohort(primary_example) == ("primary", None)


# ─── Hardcoded fallback path ──────────────────────────────────────────


def test_classify_falls_back_to_primary_set(tmp_path):
    """When the YAML doesn't classify a cohort but it's in
    PRIMARY_SOURCES, return ('primary', None)."""
    _write_yaml(tmp_path, "sources: []\n")
    primary_example = next(iter(sco.PRIMARY_SOURCES))
    assert sco.classify_source_cohort(primary_example) == ("primary", None)


def test_classify_self_annotated_returns_none(tmp_path):
    """Self-annotated cohorts (the builder sets tumor_origin per-row)
    return (None, None) so the backfill leaves them alone."""
    _write_yaml(tmp_path, "sources: []\n")
    assert sco.classify_source_cohort("GSE98894_ALVAREZ_2018_NET") == (None, None)


def test_classify_unknown_returns_none(tmp_path):
    """An unknown source_cohort returns (None, None) — caller decides
    whether to log / skip / fail."""
    _write_yaml(tmp_path, "sources: []\n")
    assert sco.classify_source_cohort("ZZZ_NOTACOHORT") == (None, None)
