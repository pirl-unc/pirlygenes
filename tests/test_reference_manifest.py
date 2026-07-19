"""Regression fences for the lightweight reference-availability manifest."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import oncoref
import pandas as pd

from pirlygenes.expression import available_cancer_expression_references
from pirlygenes.expression import accessors


_ROOT = Path(__file__).resolve().parent.parent
_PUBLIC_COLUMNS = [
    "cancer_code",
    "source_cohort",
    "source_project",
    "source_version",
    "n_samples",
    "processing_pipeline",
    "tumor_origin",
    "metastasis_site",
]


def test_availability_never_loads_full_expression_frame(monkeypatch):
    accessors._load_available_reference_manifest.cache_clear()
    delegated_calls = []
    delegated = oncoref.cancer_reference_expression_availability
    monkeypatch.setattr(
        oncoref,
        "cancer_reference_expression_availability",
        lambda **kwargs: delegated_calls.append(kwargs) or delegated(**kwargs),
    )
    monkeypatch.setattr(
        accessors,
        "_load_cancer_reference_expression",
        lambda: (_ for _ in ()).throw(
            AssertionError("availability loaded the full expression frame")
        ),
    )
    monkeypatch.setattr(
        accessors,
        "_cohort_views_root",
        lambda: (_ for _ in ()).throw(
            AssertionError("availability consulted the local cohort-view sidecar")
        ),
    )

    result = available_cancer_expression_references()

    assert result.shape == (137, 8)
    assert result.columns.tolist() == _PUBLIC_COLUMNS
    assert delegated_calls == [{
        "normalize": "tpm_clean",
        "sample_qc": "all",
        "reference_source": "summary_rows_all",
        "all_sources": True,
    }]


def test_availability_keys_match_oncoref_all_source_manifest():
    delegated = oncoref.cancer_reference_expression_availability(
        normalize="tpm_clean",
        sample_qc="all",
        reference_source="summary_rows_all",
        all_sources=True,
    )
    result = available_cancer_expression_references()

    expected_keys = set(map(
        tuple,
        delegated[["cancer_code", "source_cohort"]]
        .astype(str)
        .itertuples(index=False, name=None),
    ))
    actual_keys = set(map(
        tuple,
        result[["cancer_code", "source_cohort"]]
        .astype(str)
        .itertuples(index=False, name=None),
    ))
    assert actual_keys == expected_keys


def test_availability_keys_match_the_pirlygenes_provenance_sidecar():
    sidecar = pd.read_parquet(
        accessors._cohort_views_root() / accessors._COHORT_VIEW_PROVENANCE_FILE,
        columns=["cancer_code", "source_cohort"],
    )
    result = available_cancer_expression_references()

    expected_keys = set(map(tuple, sidecar.astype(str).itertuples(index=False, name=None)))
    actual_keys = set(map(
        tuple,
        result[["cancer_code", "source_cohort"]]
        .astype(str)
        .itertuples(index=False, name=None),
    ))
    assert actual_keys == expected_keys


def test_availability_preserves_the_complete_legacy_manifest():
    result = available_cancer_expression_references().copy()
    for column in result:
        result[column] = result[column].map(
            lambda value: "<NA>" if pd.isna(value) else str(value)
        )
    payload = result.to_csv(index=False, lineterminator="\n").encode()

    # Captured from the former full-frame projection in data version 5.23.34.
    # This pins every public value and row order, while the readable cohort-label
    # test below makes the most drift-prone compatibility cases explicit.
    assert hashlib.sha256(payload).hexdigest() == (
        "02b4cd32de40981deccadfd689047e61fc2b70432d3cf2e3dc7bd8545a0af701"
    )


def test_availability_keeps_compatibility_only_and_recent_cohort_labels():
    result = available_cancer_expression_references()
    keys = set(map(
        tuple,
        result[["cancer_code", "source_cohort"]]
        .astype(str)
        .itertuples(index=False, name=None),
    ))
    expected = {
        (code, "TREEHOUSE_POLYA_25_01_MBL_SUBGROUP_MARKERS")
        for code in ("MBL_G3", "MBL_G4", "MBL_SHH", "MBL_WNT")
    }
    expected.update({
        ("MTC", "GSE32662_PRINGLE_2012_MTC"),
        ("NUTM", "TREEHOUSE_POLYA_25_01"),
        ("NUTM", "UNC_NUTM1"),
        ("NEC_MERKEL", "GSE235092_MERKEL_2024"),
        ("SARC_CHON", "GSE299759_MEIJER_2026"),
        ("SARC_CHOR", "GSE239531_VANOOST_2024"),
        ("SARC_PEC", "GSE328026_PECOMA_2026"),
    })
    assert expected <= keys


def test_availability_returns_a_defensive_copy():
    first = available_cancer_expression_references()
    expected = available_cancer_expression_references().iloc[0]["source_cohort"]
    first.loc[0, "source_cohort"] = "MUTATED"

    second = available_cancer_expression_references()

    assert second.iloc[0]["source_cohort"] == expected


def test_availability_sorts_categorical_origins_without_a_fill_sentinel():
    source = pd.DataFrame(
        {
            "cancer_code": ["AAA"] * 4,
            "source_cohort": ["unknown", "met", "mixed", "primary"],
            "tumor_origin": pd.Categorical(
                [pd.NA, "metastasis", "mixed", "primary"],
                categories=["primary", "mixed", "metastasis"],
            ),
        }
    )

    result = accessors._build_available_references(source)

    assert result["source_cohort"].tolist() == [
        "primary",
        "mixed",
        "met",
        "unknown",
    ]


def test_fresh_process_availability_peak_rss_stays_below_one_gb():
    code = """
import sys
from pathlib import Path
from pirlygenes.expression import available_cancer_expression_references

assert sys.gettrace() is None, 'memory probe inherited coverage/debug tracing'
result = available_cancer_expression_references()
assert result.shape == (137, 8)
if sys.platform.startswith('linux'):
    # getrusage().ru_maxrss retains the forked pytest parent's historical
    # high-water mark across exec on Linux. VmHWM belongs to this executable's
    # new memory map and therefore measures the intended fresh process.
    status = Path('/proc/self/status').read_text().splitlines()
    peak_kib = next(line for line in status if line.startswith('VmHWM:')).split()[1]
    peak_bytes = int(peak_kib) * 1024
else:
    import resource
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak_bytes)
"""
    env = os.environ.copy()
    # Do not make coverage/debug instrumentation part of the measured workload.
    for name in tuple(env):
        if name.startswith(("COV_CORE_", "COVERAGE_")):
            env.pop(name)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    peak_bytes = int(completed.stdout.strip().splitlines()[-1])
    assert peak_bytes < 1_000_000_000, (
        f"availability used {peak_bytes / 1e9:.2f} GB peak RSS; it must remain "
        "a gene-independent manifest read"
    )
