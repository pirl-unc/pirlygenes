#!/usr/bin/env python3
"""Validate structured therapy-path curation in cancer-key-genes.csv."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import sys

from pirlygenes.reporting import THERAPY_PATH_TIERS


PHASE_BY_TIER = {
    "approved_standard": {"approved"},
    "approved_indication_matched": {"approved"},
    "approved_later_line": {"approved"},
    "late_clinical": {"phase_3"},
    "trial_follow_up": {"phase_1", "phase_2"},
    "preclinical": {"preclinical"},
    "off_label": {"off_label"},
}


def main() -> int:
    path = Path("pirlygenes/data/cancer-key-genes.csv")
    rows = list(csv.DictReader(path.open(newline="")))
    targets = [row for row in rows if (row.get("role") or "").strip() == "target"]
    errors = []
    counts = Counter()
    sourced_rows = 0
    pmid_sourced_rows = 0

    for idx, row in enumerate(targets, start=2):
        label = (
            f"line {idx} {row.get('cancer_code')}:{row.get('symbol')}:"
            f"{row.get('agent')}"
        )
        tier = (row.get("treatment_path_tier") or "").strip()
        line = (row.get("line_of_therapy") or "").strip()
        note = (row.get("eligibility_note") or "").strip()
        phase = (row.get("phase") or "").strip()

        if not tier:
            errors.append(f"{label}: missing treatment_path_tier")
            continue
        if tier not in THERAPY_PATH_TIERS:
            errors.append(f"{label}: invalid treatment_path_tier={tier!r}")
            continue
        if not line:
            errors.append(f"{label}: missing line_of_therapy")
        if not note:
            errors.append(f"{label}: missing eligibility_note")
        if phase not in PHASE_BY_TIER[tier]:
            errors.append(f"{label}: phase {phase!r} incompatible with tier {tier!r}")
        source = (row.get("source") or "").strip()
        if source:
            sourced_rows += 1
            if "PMID:" in source:
                pmid_sourced_rows += 1
        counts[tier] += 1

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Validated {len(targets)} target rows")
    print(
        f"Rows with source citations: {sourced_rows}/{len(targets)} "
        f"({sourced_rows / max(len(targets), 1):.1%})"
    )
    print(f"Rows with PMID citations: {pmid_sourced_rows}/{len(targets)}")
    for tier, count in sorted(counts.items()):
        print(f"{tier}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
