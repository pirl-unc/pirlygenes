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
    # shrink -> flagged with (slug, existing, new)
    assert gfg.shrinking_families({"fam": smaller}, tmp_path) == [("fam", 3, 1)]
    # grew or equal -> not flagged
    assert gfg.shrinking_families({"fam": bigger}, tmp_path) == []
    # brand-new family (no committed file) -> not flagged
    assert gfg.shrinking_families({"newfam": smaller}, tmp_path) == []
