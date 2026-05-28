"""Tests for the source_cohort → tumor_origin classifier + cache.

Covers the YAML-overlay path (where most tests' coverage was previously
absent — only the hardcoded fallback was exercised end-to-end via the
backfilled bundled shards).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pirlygenes.expression import source_cohort_origin as sco


# ─── YAML overlay ─────────────────────────────────────────────────────


def test_yaml_overlay_warns_when_tumor_origin_set_but_source_cohort_missing(
    tmp_path, monkeypatch,
):
    """A YAML entry that sets ``tumor_origin:`` but forgets
    ``source_cohort:`` is silently un-classifiable — emit a UserWarning
    listing the offending entries so a contributor catches it."""
    yaml_path = tmp_path / "expression_sources.yaml"
    yaml_path.write_text(
        "sources:\n"
        "  - id: complete-entry\n"
        "    source_cohort: GOOD_COHORT\n"
        "    tumor_origin: primary\n"
        "  - id: incomplete-entry\n"
        "    tumor_origin: metastasis\n"   # missing source_cohort
        "    metastasis_site: liver\n"
    )
    monkeypatch.setattr(sco, "_YAML_PATH", yaml_path)
    sco.clear_cache()

    with pytest.warns(UserWarning, match="incomplete-entry"):
        # Force the lookup to populate the cache (and run the warn pass).
        sco.classify_source_cohort("GOOD_COHORT")

    sco.clear_cache()  # leave no state behind for sibling tests


def test_yaml_overlay_picks_up_complete_entries(tmp_path, monkeypatch):
    """A YAML entry with both fields populates the overlay and
    classify_source_cohort returns its values (overriding any
    hardcoded fallback)."""
    yaml_path = tmp_path / "expression_sources.yaml"
    yaml_path.write_text(
        "sources:\n"
        "  - id: brand-new\n"
        "    source_cohort: BRAND_NEW_COHORT\n"
        "    tumor_origin: recurrence\n"
    )
    monkeypatch.setattr(sco, "_YAML_PATH", yaml_path)
    sco.clear_cache()

    origin, met = sco.classify_source_cohort("BRAND_NEW_COHORT")
    assert origin == "recurrence"
    assert met is None

    sco.clear_cache()


def test_clear_cache_is_public_and_resets_overlay(tmp_path, monkeypatch):
    """``clear_cache()`` is the public API (matching the
    _CancerTypeNamesView pattern) — verify it actually drops the cache
    so swapping the YAML on disk between calls is observable."""
    yaml_path = tmp_path / "expression_sources.yaml"
    yaml_path.write_text(
        "sources:\n"
        "  - id: a\n"
        "    source_cohort: COHORT_A\n"
        "    tumor_origin: primary\n"
    )
    monkeypatch.setattr(sco, "_YAML_PATH", yaml_path)
    sco.clear_cache()

    assert sco.classify_source_cohort("COHORT_A") == ("primary", None)

    # Swap the YAML to remove COHORT_A; without clear_cache the lookup
    # would still see the cached overlay.
    yaml_path.write_text(
        "sources:\n"
        "  - id: b\n"
        "    source_cohort: COHORT_B\n"
        "    tumor_origin: metastasis\n"
        "    metastasis_site: bone\n"
    )
    # Stale cache — should still hit
    assert sco.classify_source_cohort("COHORT_A") == ("primary", None)

    sco.clear_cache()
    # Fresh read — COHORT_A no longer in YAML
    assert sco.classify_source_cohort("COHORT_A") == (None, None)
    assert sco.classify_source_cohort("COHORT_B") == ("metastasis", "bone")

    sco.clear_cache()


# ─── Hardcoded fallback path ──────────────────────────────────────────


def test_classify_falls_back_to_primary_set(tmp_path, monkeypatch):
    """When the YAML doesn't classify a cohort but it's in
    PRIMARY_SOURCES, return ('primary', None)."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("sources: []\n")
    monkeypatch.setattr(sco, "_YAML_PATH", yaml_path)
    sco.clear_cache()

    # Pick any known PRIMARY source
    primary_example = next(iter(sco.PRIMARY_SOURCES))
    assert sco.classify_source_cohort(primary_example) == ("primary", None)

    sco.clear_cache()


def test_classify_self_annotated_returns_none(tmp_path, monkeypatch):
    """Self-annotated cohorts (the builder sets tumor_origin per-row)
    return (None, None) so the backfill leaves them alone."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("sources: []\n")
    monkeypatch.setattr(sco, "_YAML_PATH", yaml_path)
    sco.clear_cache()

    assert sco.classify_source_cohort("GSE98894_ALVAREZ_2018_NET") == (None, None)

    sco.clear_cache()


def test_classify_unknown_returns_none(tmp_path, monkeypatch):
    """An unknown source_cohort returns (None, None) — caller decides
    whether to log / skip / fail."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("sources: []\n")
    monkeypatch.setattr(sco, "_YAML_PATH", yaml_path)
    sco.clear_cache()

    assert sco.classify_source_cohort("ZZZ_NOTACOHORT") == (None, None)

    sco.clear_cache()
