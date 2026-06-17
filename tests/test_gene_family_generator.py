"""Guards for scripts/generate_gene_family_sets.py — the derived gene-family
CSV generator. Two regressions are protected:

1. Release discovery must not depend on ``EnsemblRelease.gtf_path`` (which
   returns ``None`` until a download is attempted in recent pyensembl, making
   the generator silently find *zero* releases and no-op).
2. The cross-release-union CSVs must never be silently *shrunk* by a run that
   sees fewer installed releases than the committed data was built from.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_gene_family_sets.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gene_family_generator", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gfg = _load_generator()


def test_release_discovery_returns_sorted_ints_without_download():
    """Discovery returns a sorted list of release ints (empty if none built),
    and never raises — regardless of pyensembl's ``gtf_path`` behaviour."""
    rels = gfg._installed_grch38_releases()
    assert isinstance(rels, list)
    assert all(isinstance(r, int) for r in rels)
    assert rels == sorted(rels)


def test_cache_root_is_release_agnostic():
    """The cache root is found without depending on any one release being a
    known/installed pyensembl release (regression for the hardcoded 111)."""
    root = gfg._grch38_cache_root()
    # Either pyensembl knows no GRCh38 release at all (None) or we get a path
    # ending in the GRCh38 reference dir — never an exception.
    assert root is None or root.name == "GRCh38"


def test_most_recent_installed_release_matches_max():
    rels = gfg._installed_grch38_releases()
    if not rels:
        assert gfg._most_recent_installed_release() is None
    else:
        assert gfg._most_recent_installed_release() == max(rels)


def test_discovery_includes_default_release_111():
    """With the default release installed (ensured by conftest where possible),
    discovery surfaces it. Skips cleanly if 111 isn't available (offline/CI)."""
    rels = gfg._installed_grch38_releases()
    if 111 not in rels:
        pytest.skip("Ensembl release 111 not installed in this environment")
    assert 111 in rels


def test_existing_row_count(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("Symbol,Ensembl_Gene_ID\na,ENSG1\nb,ENSG2\n")
    assert gfg._existing_row_count(p) == 2
    assert gfg._existing_row_count(tmp_path / "missing.csv") is None


def test_shrink_guard_flags_only_smaller_tables(tmp_path):
    (tmp_path / "fam.csv").write_text(
        "Symbol,Ensembl_Gene_ID\nA,ENSG1\nB,ENSG2\nC,ENSG3\n")  # 3 data rows
    smaller = pd.DataFrame({"Symbol": ["A"], "Ensembl_Gene_ID": ["ENSG1"]})
    bigger = pd.DataFrame({"Symbol": list("ABCD"),
                           "Ensembl_Gene_ID": [f"ENSG{i}" for i in range(4)]})
    ks = ["fam"]  # restrict the known-slug set so the test is hermetic
    # shrink -> flagged with (slug, existing, new)
    assert gfg.shrinking_families({"fam": smaller}, tmp_path, known_slugs=ks) == [("fam", 3, 1)]
    # grew or equal -> not flagged
    assert gfg.shrinking_families({"fam": bigger}, tmp_path, known_slugs=ks) == []
    # brand-new family (no committed file) -> not flagged
    assert gfg.shrinking_families({"newfam": smaller}, tmp_path, known_slugs=["newfam"]) == []


def test_shrink_guard_flags_whole_family_drop(tmp_path):
    """A family that regenerates to ZERO rows (slug absent from the tables dict
    but a committed CSV exists) is still flagged — it would otherwise leave a
    stale file untouched."""
    (tmp_path / "fam.csv").write_text("Symbol,Ensembl_Gene_ID\nA,ENSG1\nB,ENSG2\n")
    # tables omits 'fam' entirely; known_slugs still includes it
    assert gfg.shrinking_families({}, tmp_path, known_slugs=["fam"]) == [("fam", 2, 0)]


def test_write_family_tables_emits_header_only_for_vanished(tmp_path):
    """Writing covers EVERY canonical family: a populated one gets its rows, and
    a family absent from `tables` gets a header-only CSV (not skipped → no stale
    file left behind)."""
    populated = next(iter(gfg.GROUP_TO_SLUG.values()))
    df = pd.DataFrame({"Symbol": ["A"], "Ensembl_Gene_ID": ["ENSG00000000001"]})
    written = dict(gfg.write_family_tables({populated: df}, tmp_path))

    assert written[populated] == 1
    for slug in gfg.GROUP_TO_SLUG.values():
        path = tmp_path / f"{slug}.csv"
        assert path.exists()                      # every family written, none skipped
        header = path.read_text().splitlines()[0]
        assert header == "Symbol,Ensembl_Gene_ID"
        if slug != populated:
            assert written[slug] == 0             # vanished -> header-only
            assert path.read_text().strip() == "Symbol,Ensembl_Gene_ID"
