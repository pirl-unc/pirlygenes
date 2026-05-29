"""Fixture-based unit tests for the parser helpers in pirlygenes.builders.

These exercise three small, reusable helpers that the bigger build
scripts rely on:

- :func:`pirlygenes.builders.affy_gpl570.parse_series_matrix` — GEO
  ``series_matrix.txt.gz`` reader (probe-intensity matrix + per-sample
  metadata).
- :func:`pirlygenes.builders.affy_gpl570._parse_gpl570_annot` — GPL570
  ``annot.gz`` / ``acc.cgi`` platform-table reader (probe → HUGO).
- :func:`pirlygenes.builders.ncbi_gene_info.harmonize_entrez_via_ncbi`
  — Entrez-keyed matrix → ENSG-keyed matrix via the NCBI gene_info
  mapping. The test injects a tiny entrez→symbol fixture rather than
  calling out to NCBI.

Each test writes its fixture inputs to a tmp_path so the helpers go
through their real file-I/O paths.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest


SERIES_MATRIX_FIXTURE = """\
!Series_title\t"Tiny test series"
!Sample_title\t"sampleA"\t"sampleB"\t"sampleC"
!Sample_geo_accession\t"GSMfakeA"\t"GSMfakeB"\t"GSMfakeC"
!Sample_characteristics_ch1\t"tissue: peripheral blood"\t"tissue: peripheral blood"\t"tissue: lymph node"
!Sample_characteristics_ch1\t"diagnosis: HCL"\t"diagnosis: control"\t"diagnosis: HCL"
!series_matrix_table_begin
"ID_REF"\t"sampleA"\t"sampleB"\t"sampleC"
"1007_s_at"\t8.5\t7.2\t9.1
"1053_at"\t6.0\t6.2\t6.1
"117_at"\t4.5\t4.6\t4.4
!series_matrix_table_end
"""


GPL570_ANNOT_FIXTURE = """\
^PLATFORM = GPL570
!Platform_title = Affymetrix Human Genome U133 Plus 2.0 Array
!Platform_description = test fixture
!platform_table_begin
ID\tGB_ACC\tGene symbol\tENTREZ_GENE_ID
1007_s_at\tU48705\tDDR1 /// MIR4640\t780 /// 100616237
1053_at\tM87338\tRFC2\t5982
117_at\tX51757\t\t
!platform_table_end
"""


def _write_series_matrix(path: Path, content: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(content)
    return path


def _write_annot(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ─── parse_series_matrix ──────────────────────────────────────────────


def test_parse_series_matrix_returns_intensities_and_metadata(tmp_path):
    from pirlygenes.builders.affy_gpl570 import parse_series_matrix

    path = _write_series_matrix(
        tmp_path / "GSEfake_series_matrix.txt.gz",
        SERIES_MATRIX_FIXTURE,
    )
    matrix, meta = parse_series_matrix(path)

    # Matrix shape: 3 probes × 3 samples, float dtype, named index
    assert matrix.shape == (3, 3)
    assert matrix.index.name == "probe_id"
    assert list(matrix.columns) == ["sampleA", "sampleB", "sampleC"]
    assert matrix.loc["1007_s_at", "sampleA"] == pytest.approx(8.5)
    assert matrix.loc["117_at", "sampleC"] == pytest.approx(4.4)

    # Per-sample metadata: title + geo accession + char_* fields
    assert meta["sampleA"]["Sample_title"] == "sampleA"
    assert meta["sampleA"]["Sample_geo_accession"] == "GSMfakeA"
    assert meta["sampleA"]["char_tissue"] == "peripheral blood"
    assert meta["sampleA"]["char_diagnosis"] == "HCL"
    assert meta["sampleB"]["char_diagnosis"] == "control"
    assert meta["sampleC"]["char_tissue"] == "lymph node"


def test_parse_series_matrix_handles_missing_characteristics(tmp_path):
    """A sample-characteristic row with the wrong arity is skipped, not
    silently mis-aligned across samples."""
    from pirlygenes.builders.affy_gpl570 import parse_series_matrix

    bad = SERIES_MATRIX_FIXTURE.replace(
        '!Sample_characteristics_ch1\t"diagnosis: HCL"\t"diagnosis: control"\t"diagnosis: HCL"',
        '!Sample_characteristics_ch1\t"diagnosis: HCL"\t"diagnosis: control"',
    )
    path = _write_series_matrix(tmp_path / "GSEfake.txt.gz", bad)
    _matrix, meta = parse_series_matrix(path)

    # Mis-arity row dropped; the well-formed char_tissue row remains.
    assert "char_tissue" in meta["sampleA"]
    assert "char_diagnosis" not in meta["sampleA"]


# ─── _parse_gpl570_annot ──────────────────────────────────────────────


def test_parse_gpl570_annot_splits_multi_symbol(tmp_path):
    from pirlygenes.builders.affy_gpl570 import _parse_gpl570_annot

    path = _write_annot(tmp_path / "GPL570.platform_table.txt", GPL570_ANNOT_FIXTURE)
    df = _parse_gpl570_annot(path)

    # DDR1 /// MIR4640 → DDR1 (first symbol); RFC2 stays; empty/dashed row dropped.
    by_probe = df.set_index("probe_id")["gene_symbol"].to_dict()
    assert by_probe == {"1007_s_at": "DDR1", "1053_at": "RFC2"}


def test_parse_gpl570_annot_rejects_bad_file(tmp_path):
    from pirlygenes.builders.affy_gpl570 import _parse_gpl570_annot

    bad = "!platform_table_begin\nFOO\tBAR\nrow\tdata\n!platform_table_end\n"
    path = _write_annot(tmp_path / "bad.txt", bad)
    with pytest.raises(RuntimeError, match="missing probe/symbol cols"):
        _parse_gpl570_annot(path)


# ─── harmonize_entrez_via_ncbi ────────────────────────────────────────


def test_harmonize_entrez_via_ncbi_resolves_via_ncbi_map(monkeypatch):
    """Inject a tiny entrez→symbol map and confirm the helper routes
    Entrez-keyed counts through pyensembl to ENSG."""
    from pirlygenes.builders import ncbi_gene_info

    # Fake NCBI lookup: 780 → DDR1, 5982 → RFC2, 99999 → unmappable.
    fake_map = {"780": "DDR1", "5982": "RFC2", "99999": "ZZZZNOTAGENE"}
    monkeypatch.setattr(
        ncbi_gene_info, "cached_entrez_to_symbol", lambda: fake_map
    )

    counts = pd.DataFrame(
        {"sampleA": [100, 200, 300, 50], "sampleB": [110, 220, 330, 55]},
        index=["780", "5982", "99999", "notanumber"],
    )
    counts.index.name = "gene_id"

    mapping, by_ensg = ncbi_gene_info.harmonize_entrez_via_ncbi(
        counts, ensembl_release=112,
    )

    # The two resolvable Entrez IDs (780 = DDR1, 5982 = RFC2) should
    # land in the mapping with concrete ENSG IDs.
    resolved_symbols = set(mapping["Symbol"])
    assert "DDR1" in resolved_symbols
    assert "RFC2" in resolved_symbols

    # The matrix index should be ENSGs, not Entrez.
    assert all(idx.startswith("ENSG") for idx in by_ensg.index)
    # Sample columns preserved
    assert list(by_ensg.columns) == ["sampleA", "sampleB"]
    # Values for the two resolved rows match the input counts.
    ddr1_ensg = mapping.loc[mapping["Symbol"] == "DDR1", "Ensembl_Gene_ID"].iloc[0]
    assert by_ensg.loc[ddr1_ensg, "sampleA"] == 100


def test_harmonize_entrez_via_ncbi_returns_empty_on_no_matches(monkeypatch):
    """All-unresolvable input returns empty mapping + empty matrix —
    doesn't crash with a KeyError or NaN dataframe."""
    from pirlygenes.builders import ncbi_gene_info

    monkeypatch.setattr(
        ncbi_gene_info, "cached_entrez_to_symbol",
        lambda: {"99998": "ZZZBOGUS", "99999": "ZZZZNOTAGENE"},
    )

    counts = pd.DataFrame(
        {"sampleA": [1, 2], "sampleB": [3, 4]},
        index=pd.Index(["99998", "99999"], name="gene_id"),
    )
    mapping, by_ensg = ncbi_gene_info.harmonize_entrez_via_ncbi(
        counts, ensembl_release=112,
    )
    assert mapping.empty
    # Empty result still preserves the sample-column shape so consumers
    # can `.reindex().fillna(0)` without a structure surprise.
    assert list(by_ensg.columns) == ["sampleA", "sampleB"]
    assert by_ensg.empty
