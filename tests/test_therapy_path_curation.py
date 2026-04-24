"""Audit structured therapy-path curation for cancer-key-genes rows."""

import pandas as pd

from pirlygenes.reporting import (
    THERAPY_PATH_TIERS,
    therapy_path_context,
    therapy_path_rank,
    therapy_path_tier,
)


def _target_rows():
    df = pd.read_csv("pirlygenes/data/cancer-key-genes.csv").fillna("")
    return df[df["role"].astype(str).str.strip() == "target"].copy()


def test_target_rows_have_structured_treatment_path_curation():
    targets = _target_rows()
    assert len(targets) >= 300
    for column in ("treatment_path_tier", "line_of_therapy", "eligibility_note"):
        missing = targets[targets[column].astype(str).str.strip().eq("")]
        assert missing.empty, (
            f"{column} missing for target rows: "
            + ", ".join(
                f"{row.cancer_code}:{row.symbol}:{row.agent}"
                for row in missing.head(10).itertuples()
            )
        )

    invalid = targets[
        ~targets["treatment_path_tier"].astype(str).isin(THERAPY_PATH_TIERS)
    ]
    assert invalid.empty, (
        "invalid treatment_path_tier values: "
        + ", ".join(sorted(set(invalid["treatment_path_tier"].astype(str))))
    )


def test_treatment_path_tier_is_phase_compatible():
    targets = _target_rows()
    allowed = {
        "approved_standard": {"approved"},
        "approved_indication_matched": {"approved"},
        "approved_later_line": {"approved"},
        "late_clinical": {"phase_3"},
        "trial_follow_up": {"phase_1", "phase_2"},
        "preclinical": {"preclinical"},
        "off_label": {"off_label"},
    }
    bad = []
    for row in targets.itertuples():
        phase = str(row.phase)
        tier = str(row.treatment_path_tier)
        if phase not in allowed[tier]:
            bad.append(f"{row.cancer_code}:{row.symbol}:{row.agent}:{phase}->{tier}")
    assert not bad, "phase/tier mismatch: " + ", ".join(bad[:20])


def test_target_row_sources_are_present_for_most_curation_rows():
    targets = _target_rows()
    sources = targets["source"].astype(str).str.strip()
    assert sources.ne("").mean() >= 0.95
    assert sources[sources.ne("")].str.contains("PMID:", regex=False).all()


def test_reports_prefer_explicit_treatment_path_tier_over_rationale_text():
    later_line_row = {
        "symbol": "TEST1",
        "agent": "example ADC",
        "agent_class": "ADC",
        "phase": "approved",
        "indication": "example cancer",
        "rationale": "frontline standard backbone wording should not win",
        "treatment_path_tier": "approved_later_line",
        "eligibility_note": "confirm prior therapy",
    }
    assert therapy_path_tier(later_line_row) == "approved_later_line"
    assert therapy_path_rank(later_line_row) == 2
    assert "approved later-line pathway" in therapy_path_context(later_line_row)
    assert "guideline-standard" not in therapy_path_context(later_line_row)

    standard_row = {
        "symbol": "TEST2",
        "agent": "example antibody",
        "agent_class": "antibody",
        "phase": "approved",
        "indication": "example cancer",
        "rationale": "",
        "treatment_path_tier": "approved_standard",
        "eligibility_note": "confirm first-line eligibility",
    }
    assert therapy_path_tier(standard_row) == "approved_standard"
    assert therapy_path_rank(standard_row) == 0
    assert "guideline-standard approved pathway" in therapy_path_context(standard_row)


def test_treatment_path_context_dedupes_curated_note_prefix():
    row = {
        "symbol": "TEST3",
        "agent": "example TCE",
        "agent_class": "TCE",
        "phase": "phase_2",
        "indication": "example cancer",
        "rationale": "",
        "treatment_path_tier": "trial_follow_up",
        "eligibility_note": "clinical-trial follow-up; not default standard",
    }
    context = therapy_path_context(row)
    assert context == "clinical-trial follow-up; not default standard"
    assert "clinical-trial follow-up; clinical-trial follow-up" not in context
