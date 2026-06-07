"""Lightweight docs-drift guard for the approximate public-count claims in
README (#315).

README advertises ``~N names in pirlygenes.__all__`` and ``~N cancer-testis
antigens``. Those are deliberately approximate (they grow as panels expand), so
this doesn't pin an exact value — it asserts the live count stays within a
tolerance band of the documented figure. When the band is breached the README
number is stale and should be nudged; small additions never trip it. This is
exactly the drift the audit-doc staleness in #315 was about, caught at the
front door rather than in archived audit snapshots."""

import re
from pathlib import Path

import pirlygenes
from pirlygenes.gene_sets_cancer import CTA_gene_names

_README = Path(__file__).resolve().parent.parent / "README.md"


def _stated(pattern: str) -> int:
    text = _README.read_text()
    m = re.search(pattern, text)
    assert m, f"README no longer contains a count matching {pattern!r}"
    return int(m.group(1))


def _assert_close(stated: int, live: int, label: str) -> None:
    # generous band: small panel growth shouldn't churn the README, but a real
    # drift (e.g. the stale 75 vs the live 86) is flagged.
    tol = max(8, round(0.12 * stated))
    assert abs(live - stated) <= tol, (
        f"README {label} claim (~{stated}) has drifted from the live value "
        f"({live}); update README.md (tolerance ±{tol})."
    )


def test_readme_all_count_current():
    _assert_close(_stated(r"~(\d+)\s+names in `pirlygenes\.__all__`"),
                  len(pirlygenes.__all__), "__all__")


def test_readme_cta_count_current():
    _assert_close(_stated(r"~(\d+)\s+cancer-testis antigens"),
                  len(CTA_gene_names()), "CTA_gene_names")
