"""Registry-completeness contract (data-only fields).

Every **leaf** cancer-type code in ``cancer-type-registry.csv``
(``parent_code`` empty) must carry a minimum package of curated
gene-knowledge data:

1. **Lineage panel** — at least five genes registered in
   ``lineage-genes.csv``.
2. **Biomarker** — at least one row in ``cancer-key-genes.csv``
   with ``role=biomarker``.
3. **Therapy target** — at least one row in ``cancer-key-genes.csv``
   with ``role=target``.
4. **Therapy-response axis panel** — at least one row in
   ``therapy-response-signatures.csv`` whose ``cancer_context``
   names this code (not just ``pan_cancer``). Drives the
   per-cancer subtype-signature plot in downstream consumers.

Subtype rows (``parent_code`` set) are exempt — the parent carries the
minimum, subtype inherits via mixture-cohort / subtype_key.

Expression-data + matched-normal-reference contracts moved to
trufflepig along with the expression matrices. See
``trufflepig/tests/test_registry_completeness.py`` for the
expression-side coverage.

This test pins the contract. When a new entity is added to the
registry, this test flags the gap. The ``_TOLERATED_GAPS`` mapping
enumerates codes allowed to miss one or more fields; the goal is to
shrink it toward empty over time.
"""

from __future__ import annotations

from pirlygenes.gene_sets_cancer import cancer_type_registry
from pirlygenes.load_dataset import get_data


# Codes lacking a cancer-specific therapy-response axis panel (rows in
# ``therapy-response-signatures.csv`` mentioning the code in
# ``cancer_context``, beyond the ``pan_cancer`` fallback). Shrink as
# cancer-specific axis panels get curated per family.
_MISSING_THERAPY_AXIS = frozenset(
    {
        # Phase C: the bone / RMS / ESS sarcomas (SARC_OS, SARC_EWS,
        # SARC_CHON/CHOR, SARC_GCTB, SARC_ESS_*, SARC_RMS_*) are now parented
        # under SARC and inherit its therapy axis, so they are no longer gaps.
        # ACINIC/ADCC/CHOL/GBC/NET_LUNG/NET_MIDGUT/NET_PANCREAS/NET_RECTAL now
        # inherit a therapy axis through the parent links oncoref's registry
        # adds (e.g. NET_* -> NET, CHOL/GBC -> BTC), so they are no longer gaps.
        "ACC", "ATRT", "BL", "BLCA", "B_ALL", "CESC",
        "CLL", "CML", "CTCL", "DLBC", "ESCA",
        "FL", "GBM", "HCL", "HEPB",
        "HL", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
        "NEC_LUNG_LARGECELL", "MBL", "MCL", "MDS", "NEC_MERKEL",
        "MESO", "MM", "MPN", "MTC", "NPC", "NUTM",
        "OV", "PAAD", "PCPG", "RB",
        "RT", "SARC", "CRC",
        "SCLC", "STAD", "TGCT", "THCA", "THYM", "THYMCA", "T_ALL", "UCEC", "UCS",
        "UVM", "WILMS",
        # Re-export of oncoref's registry (pirlygenes#523) adds these codes; no
        # therapy-response signature is materialised for them yet.
        # (CRC_MSI dropped from this set in oncoref 1.8.95: it was reparented
        # under CRC as a molecular subtype (oncoref #322/#323), so it is no
        # longer a top-level code the completeness check evaluates.)
        "BTC", "NET", "NEN_G3_EXTRAPULMONARY", "NSCLC", "RCC_NCC",
        "SGC",
        # #294/#295 NCI-gap additions — curated registry entries; expression
        # data not yet built, so no therapy-axis panel is materialised yet.
        "BCC", "cSCC", "VSCC", "PENSCC", "VAGC", "URETH", "ANSC",
        "EPN", "CRANIO", "DIPG", "PITNET",
        # #358 CNS panel anchors — curated registry entries that anchor the
        # MENINGIOMA / CHOROID_PLEXUS family panels (#357); expression data and
        # therapy curation not yet built.
        "MENINGIOMA", "CHOROID_PLEXUS",
        # ALCL — curated entity (lineage/CD30+ALK biomarker+therapy present); no
        # therapy-response signature materialised yet.
        "ALCL",
        # ASTB — Astroblastoma (MN1-altered), new MN1::BEND2 entity (#458);
        # biomarker-only, no public expression cohort / therapy-response panel.
        "ASTB",
    }
)


# Explicit per-code gap allowance. Format: ``code: set of allowed-to-miss
# fields``. ``therapy_axis`` is seeded automatically from the baseline
# set above; this dict carries the lineage / biomarker / therapy gaps.
_TOLERATED_GAPS_EXPLICIT = {
    # Re-export of oncoref's registry (pirlygenes#523) adds these codes; their
    # pirlygenes gene-set panels (lineage/biomarker/therapy) aren't curated yet.
    "BTC": {"lineage", "biomarker", "therapy"},
    "CRC_MSI": {"lineage", "biomarker", "therapy"},
    "NET": {"lineage", "biomarker", "therapy"},
    "NET_NONPANCREATIC": {"lineage", "biomarker", "therapy"},
    "NEN_G3_EXTRAPULMONARY": {"lineage", "biomarker", "therapy"},
    "NSCLC": {"lineage", "biomarker", "therapy"},
    "RCC_NCC": {"lineage", "biomarker", "therapy"},
    "RCC_NCC_UNCLASSIFIED": {"lineage", "biomarker", "therapy"},
    "SGC": {"lineage", "biomarker", "therapy"},
    "SARC_MMNST": {"lineage", "biomarker", "therapy"},
    # Heme entries still awaiting curation
    "CLL": {"lineage"},
    "B_ALL": {"lineage", "biomarker", "therapy"},
    "MDS": {"lineage", "biomarker", "therapy"},
    "MPN": {"lineage", "biomarker", "therapy"},
    # NET axis — NET_PANCREAS shipped v4.47.0; rest pending
    "NEC_LUNG_LARGECELL": {"lineage", "biomarker", "therapy"},
    "NET_RECTAL": {"lineage", "biomarker", "therapy"},
    # Pediatric entities — lineage panels still need curation
    "NBL": {"lineage"},
    "WILMS": {"lineage"},
    "HEPB": {"lineage", "biomarker", "therapy"},
    "ATRT": {"lineage"},
    "RB": {"lineage", "biomarker", "therapy"},
    "MBL": {"lineage"},
    "RT": {"lineage", "biomarker", "therapy"},
    "SARC_OS": {"lineage"},
    "SARC_EWS": {"lineage"},
    "SARC_RMS_ERMS": {"lineage"},
    "SARC_RMS_ARMS": {"lineage"},
    "SARC_RMS_SSRMS": {"lineage", "biomarker", "therapy"},
    "NUTM": {"lineage"},
    # TGCT is chemo-dominant (BEP); no clinician-validated targeted
    # therapy exists, so the panel is intentionally empty.
    "TGCT": {"therapy"},
    # Rare entities — need dedicated curation
    "ACINIC": {"lineage"},
    "ADCC": {"lineage"},
    "NPC": {"lineage"},
    "SARC_CHOR": {"lineage"},
    "SARC_CHON": {"lineage"},
    "SARC_GCTB": {"lineage", "biomarker", "therapy"},
    "SARC_ESS_LG": {"lineage", "biomarker", "therapy"},
    "SARC_ESS_HG": {"lineage", "biomarker", "therapy"},
    # #294/#295 NCI-gap additions — curated registry entries; lineage panels
    # and (where noted) biomarker/therapy curation still pending. BCC/cSCC/
    # CRANIO carry curated biomarker+target; GBC carries a target; DIPG/EPN
    # carry biomarkers only.
    "BCC": {"lineage"},
    "cSCC": {"lineage"},
    "VSCC": {"lineage", "biomarker", "therapy"},
    "PENSCC": {"lineage", "biomarker", "therapy"},
    "VAGC": {"lineage", "biomarker", "therapy"},
    "URETH": {"lineage", "biomarker", "therapy"},
    "ANSC": {"lineage", "biomarker", "therapy"},
    "GBC": {"lineage", "biomarker"},
    "EPN": {"lineage", "therapy"},
    "CRANIO": {"lineage"},
    "DIPG": {"lineage", "therapy"},
    "PITNET": {"lineage", "biomarker", "therapy"},
    # #358/#369 CNS panel anchors — lineage panels + biomarkers now curated.
    # MENINGIOMA also has a target (SSTR2/177Lu-DOTATATE) so it's fully covered
    # (only therapy_axis remains, via _MISSING_THERAPY_AXIS). CHOROID_PLEXUS
    # carcinoma has no approved targeted therapy -> therapy gap stays tolerated.
    "CHOROID_PLEXUS": {"therapy"},
    # CRC: computed colorectal union (parent of COAD+READ); an abstract node
    # with no panels of its own — its children carry lineage/biomarker/therapy.
    "CRC": {"lineage", "biomarker", "therapy"},
    # THYMCA: thymic carcinoma split from thymoma (THYM); curated clinical
    # reference (TMB/ICI) only — lineage/biomarker/therapy panels not yet built.
    "THYMCA": {"lineage", "biomarker", "therapy"},
    # ASTB: Astroblastoma (MN1-altered), added with the MN1::BEND2 ontology
    # entry (#458). Carries the defining MN1 biomarker; no lineage panel and
    # no clinician-validated targeted therapy exist for this rare CNS tumor.
    "ASTB": {"lineage", "therapy"},
}


def _build_tolerated_gaps():
    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]
    out = {}
    for code in leaf["code"]:
        fields = set(_TOLERATED_GAPS_EXPLICIT.get(code, set()))
        if code in _MISSING_THERAPY_AXIS:
            fields.add("therapy_axis")
        if fields:
            out[code] = fields
    return out


_TOLERATED_GAPS = _build_tolerated_gaps()


def _leaf_codes_with_coverage():
    """Return ``{code: {field: bool}}`` for every leaf registry entry."""
    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]

    ln = get_data("lineage-genes")
    ln_codes = {code for code, group in ln.groupby("Cancer_Type") if len(group) >= 5}

    key = get_data("cancer-key-genes")
    biomarker_codes = set(key[key["role"] == "biomarker"]["cancer_code"].dropna())
    therapy_codes = set(key[key["role"] == "target"]["cancer_code"].dropna())

    ts = get_data("therapy-response-signatures")
    therapy_axis_codes = set()
    for value in ts["cancer_context"].dropna():
        for part in str(value).split(";"):
            part = part.strip()
            if part and part != "pan_cancer":
                therapy_axis_codes.add(part)

    out = {}
    for _, row in leaf.iterrows():
        code = row["code"]
        out[code] = {
            "lineage": code in ln_codes,
            "biomarker": code in biomarker_codes,
            "therapy": code in therapy_codes,
            "therapy_axis": code in therapy_axis_codes,
        }
    return out


def test_every_leaf_passes_minimum_or_is_tolerated():
    coverage = _leaf_codes_with_coverage()
    violations = []
    for code, fields in coverage.items():
        tolerated = _TOLERATED_GAPS.get(code, set())
        missing = {f for f, present in fields.items() if not present}
        unexpected = missing - tolerated
        if unexpected:
            violations.append(f"{code} missing {sorted(unexpected)}")

    assert not violations, (
        "Registry-completeness violations:\n  "
        + "\n  ".join(violations)
        + "\n\nEither add the missing data or extend ``_TOLERATED_GAPS_EXPLICIT``"
          " / ``_MISSING_THERAPY_AXIS`` with a justified entry."
    )


def test_tolerated_gaps_reference_real_codes():
    reg_codes = set(cancer_type_registry()["code"])
    unknown_explicit = set(_TOLERATED_GAPS_EXPLICIT) - reg_codes
    unknown_axis = set(_MISSING_THERAPY_AXIS) - reg_codes
    assert not unknown_explicit, (
        f"_TOLERATED_GAPS_EXPLICIT references unknown codes: {sorted(unknown_explicit)}"
    )
    assert not unknown_axis, (
        f"_MISSING_THERAPY_AXIS references unknown codes: {sorted(unknown_axis)}"
    )


def test_tolerated_fields_are_valid_names():
    valid = {"lineage", "biomarker", "therapy", "therapy_axis"}
    for code, fields in _TOLERATED_GAPS.items():
        bad = fields - valid
        assert not bad, (
            f"_TOLERATED_GAPS[{code!r}] has invalid field names {bad}; "
            f"valid are {sorted(valid)}"
        )


def test_therapy_axis_baseline_matches_current_data():
    """If the baseline claims a code lacks a therapy axis but the panel
    is actually present (or vice versa), the set has drifted."""
    coverage = _leaf_codes_with_coverage()
    missing_now = {c for c, f in coverage.items() if not f["therapy_axis"]}

    stale = set(_MISSING_THERAPY_AXIS) - missing_now
    new = missing_now - set(_MISSING_THERAPY_AXIS)

    assert not stale, (
        f"_MISSING_THERAPY_AXIS lists codes that NOW have a panel — shrink: {sorted(stale)}"
    )
    assert not new, (
        f"New leaf codes lack a therapy-axis panel but aren't in "
        f"_MISSING_THERAPY_AXIS: {sorted(new)}"
    )


def test_completeness_progress_report(capsys):
    """Informational — print current gap distribution for CI logs."""
    coverage = _leaf_codes_with_coverage()
    total = len(coverage)
    fields = ("lineage", "biomarker", "therapy", "therapy_axis")
    complete = sum(
        1 for cov in coverage.values() if all(cov[f] for f in fields)
    )
    per_field_missing = {
        f: sum(1 for cov in coverage.values() if not cov[f]) for f in fields
    }
    with capsys.disabled():
        print(
            f"\n[completeness] {complete}/{total} leaf codes pass the 4-field "
            f"data-only minimum (lineage + biomarker + therapy + therapy_axis). "
            f"{len(_TOLERATED_GAPS)} codes carry tolerated gaps."
        )
        print("  gaps by field:")
        for f, n in per_field_missing.items():
            print(f"    {f}: {n} codes")
    assert complete >= 0  # always passes; smoke
