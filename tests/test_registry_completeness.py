"""Registry-completeness contract.

Every **leaf** cancer-type code in ``cancer-type-registry.csv``
(parent_code empty) must carry a minimum package of information:

1. **Expression data** — either an ``FPKM_<code>`` column in
   ``pan-cancer-expression.csv`` (the TCGA-style pan-cancer reference)
   or a row-set in ``subtype-deconvolved-expression.csv.gz``.
2. **Lineage panel** — at least five genes registered in
   ``lineage-genes.csv`` (also enforced by #170's floor test).
3. **Biomarker** — at least one row in ``cancer-key-genes.csv``
   with ``role=biomarker``.
4. **Therapy target** — at least one row with ``role=target``.

Subtype rows (``parent_code`` set) are exempt — the parent carries the
minimum and the subtype inherits context via the mixture-cohort
mechanism (#171) or key-genes subtype_key lookup.

This test pins the contract. When a new entity is added to the
registry, this test flags the gap. The ``_TOLERATED_GAPS`` set below
enumerates codes that are allowed to miss one or more fields today —
the goal is to shrink it to empty over time. Each entry lists the
specific fields that code is allowed to be missing.
"""

import pandas as pd

from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    pan_cancer_expression,
    subtype_deconvolved_expression,
)


# ── Tolerated gaps ────────────────────────────────────────────────────
#
# Format: ``code: set of allowed-to-miss fields``. As gaps are closed,
# shrink these entries. A code with an empty set here passes every
# minimum — remove the entry entirely when that happens.
#
# Currently tolerated:
#   - Rare / pediatric / sarcoma entities that need a dedicated curation
#     pass for biomarker + therapy rows. Expression data may also be
#     missing for entities where no open cohort exists yet.
#   - Heme / NET entities whose expression cohort downloads are
#     deferred (MMRF CoMMpass, TARGET ALL, George 2018 lung NE, etc.).
#
# Re-run the audit after each gap-closure PR to see which codes can be
# removed from this allowlist.

_TOLERATED_GAPS = {
    # Heme entries still awaiting expression-data integration
    "CLL": {"expression", "lineage"},
    "MM":  {"expression"},   # MMRF CoMMpass deferred
    "HL":  {"expression"},
    "MCL": {"expression"},
    "B_ALL": {"expression", "lineage", "biomarker", "therapy"},  # TARGET ALL deferred
    "T_ALL": {"expression"},
    "BL":  {"expression"},
    "FL":  {"expression"},
    "HCL": {"expression"},
    "CTCL": {"expression"},  # biomarker + therapy + lineage filled v4.48.0
    "CML": {"expression"},
    "MDS": {"expression", "lineage", "biomarker", "therapy"},
    "MPN": {"expression", "lineage", "biomarker", "therapy"},

    # NET axis — PANNET shipped v4.47.0; rest pending
    "LUNG_NET_LC": {"expression"},        # George 2018 deferred
    "LUNG_NET_LCNEC": {"expression", "lineage", "biomarker", "therapy"},
    "MID_NET": {"expression"},            # small GEO deferred
    "MEC": {"expression"},                # Merkel cohort deferred
    "MTC": {"expression"},                # small GEO deferred

    # Pediatric entities — expression in subtype-deconvolved already
    # for OS/EWS/NBL/RMS_*/ATRT/MBL/RT via Treehouse; lineage panels
    # still need curation.
    "NBL": {"expression", "lineage"},
    "WILMS": {"expression", "lineage"},
    "HEPB": {"expression", "lineage", "biomarker", "therapy"},
    "ATRT": {"lineage"},
    "RB": {"expression", "lineage", "biomarker", "therapy"},
    "MBL": {"lineage"},
    "RT": {"lineage", "biomarker", "therapy"},
    "OS": {"lineage"},
    "EWS": {"lineage"},
    "RMS_ERMS": {"lineage"},
    "RMS_ARMS": {"lineage"},
    "RMS_SSRMS": {"lineage", "biomarker", "therapy"},
    "NUTM": {"lineage"},

    # TGCT is chemo-dominant (BEP); the ``cancer-key-genes`` curation
    # bar explicitly leaves its therapy panel empty because no
    # clinician-validated molecular-targeted therapy exists. Pinned
    # by ``test_tgct_is_biomarker_only``; mirrored here as a tolerated
    # gap so both contracts hold.
    "TGCT": {"therapy"},

    # Rare entities — need dedicated curation
    "ACINIC": {"expression", "lineage"},
    "ADCC": {"expression", "lineage"},
    "NPC": {"expression", "lineage"},
    "CHOR": {"expression", "lineage"},
    "CHON": {"lineage"},
    "SARC_IFS": {"expression", "lineage", "biomarker", "therapy"},
    "GCTB": {"expression", "lineage", "biomarker", "therapy"},
    "ESS_LG": {"expression", "lineage", "biomarker", "therapy"},
    "ESS_HG": {"expression", "lineage", "biomarker", "therapy"},
    "PCN": {"expression", "lineage", "biomarker", "therapy"},
}


def _leaf_codes_with_coverage():
    """Return a dict ``{code: {field: bool}}`` for every leaf entry."""
    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]

    pan = pan_cancer_expression()
    pan_codes = {
        c.replace("FPKM_", "")
        for c in pan.columns if c.startswith("FPKM_")
    }
    sub = subtype_deconvolved_expression()
    sub_codes = (
        set(sub["cancer_code"].dropna().unique()) if sub is not None else set()
    )

    # lineage
    ln = pd.read_csv("pirlygenes/data/lineage-genes.csv")
    ln_codes = {
        code for code, group in ln.groupby("Cancer_Type")
        if len(group) >= 5
    }

    # key-genes
    key = pd.read_csv("pirlygenes/data/cancer-key-genes.csv")
    biomarker_codes = set(key[key["role"] == "biomarker"]["cancer_code"].dropna())
    therapy_codes = set(key[key["role"] == "target"]["cancer_code"].dropna())

    out = {}
    for _, row in leaf.iterrows():
        code = row["code"]
        out[code] = {
            "expression": code in pan_codes or code in sub_codes,
            "lineage": code in ln_codes,
            "biomarker": code in biomarker_codes,
            "therapy": code in therapy_codes,
        }
    return out


def test_every_leaf_passes_minimum_or_is_tolerated():
    coverage = _leaf_codes_with_coverage()
    violations = []
    for code, fields in coverage.items():
        tolerated = _TOLERATED_GAPS.get(code, set())
        missing = {f for f, present in fields.items() if not present}
        unexpected_missing = missing - tolerated
        if unexpected_missing:
            violations.append(f"{code} missing {sorted(unexpected_missing)}")

    assert not violations, (
        "Registry-completeness violations:\n  "
        + "\n  ".join(violations)
        + "\n\nEither fix the gap (add expression / lineage / biomarker / "
        "therapy data) or extend ``_TOLERATED_GAPS`` with a justified "
        "entry in this test."
    )


def test_tolerated_gaps_only_list_real_codes():
    """Every entry in ``_TOLERATED_GAPS`` must point at a real
    registry code. Prevents silent drift if a code is renamed."""
    reg_codes = set(cancer_type_registry()["code"])
    unknown = set(_TOLERATED_GAPS) - reg_codes
    assert not unknown, (
        f"``_TOLERATED_GAPS`` references codes not in the registry: {sorted(unknown)}"
    )


def test_tolerated_fields_are_valid_names():
    valid = {"expression", "lineage", "biomarker", "therapy"}
    for code, fields in _TOLERATED_GAPS.items():
        bad = fields - valid
        assert not bad, (
            f"``_TOLERATED_GAPS[{code!r}]`` has invalid field names {bad}; "
            f"valid are {sorted(valid)}"
        )


def test_completeness_progress_report(capsys):
    """Informational — print the current gap distribution. Not a
    contract assertion; just a way to see progress in CI logs."""
    coverage = _leaf_codes_with_coverage()
    total = len(coverage)
    complete = sum(1 for fields in coverage.values() if all(fields.values()))
    with capsys.disabled():
        print(
            f"\n[completeness] {complete}/{total} leaf codes have the "
            "full minimum package (expression + lineage + biomarker + "
            "therapy). "
            f"{len(_TOLERATED_GAPS)} codes still in tolerated-gaps list."
        )
    assert complete >= 0  # always passes; smoke
