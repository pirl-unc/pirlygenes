"""Preserve observation scope and missingness in the five-gene profiles."""
import numpy as np
import pandas as pd

from pirlygenes import placental_gene_profiles as profiles


def test_plot_matrix_keeps_missing_ihc_distinct_from_measured_zero():
    data = pd.DataFrame({"cancer": ["breast", "breast"],
                         "symbol": ["GCM1", "HTRA4"], "fraction": [0., .5]})
    matrix = profiles._matrix(data, "cancer", "fraction", ["breast", "lung"])
    assert np.isnan(matrix.loc["breast", "INSL4"])
    assert matrix.loc["breast", "GCM1"] == 0
    assert matrix.loc["breast", "HTRA4"] == .5
    assert matrix.loc["lung"].isna().all()
    assert list(matrix) == list(profiles.GENES)


def test_data_requests_rna_only_genes_and_preserves_separate_cohorts(monkeypatch):
    observed = {}
    ids = list(profiles.GENES.values())
    rna = pd.DataFrame({"gene_id": ids * 2, "cohort": ["TCGA"] * 5 + ["validation"] * 5})
    ihc = pd.DataFrame({"gene_id": ids[1:]})
    def get_rna(gene_ids):
        assert gene_ids == ids
        return rna.copy()
    def get_comparison(gene_ids, **kwargs):
        observed.update(kwargs)
        assert gene_ids == ids
        return pd.DataFrame({"gene_id": ids})
    monkeypatch.setattr(profiles.hpa_cancer, "hpa_cancer_rna_prevalence", get_rna)
    monkeypatch.setattr(profiles.hpa_cancer, "hpa_cancer_ihc_prevalence", lambda gene_ids: ihc.copy())
    monkeypatch.setattr(profiles.hpa_cancer, "hpa_cancer_rna_ihc_comparison", get_comparison)
    r, i, c = profiles.profile_data()
    assert set(r.cohort) == {"TCGA", "validation"}
    assert set(c.symbol) == set(profiles.GENES)
    assert "INSL4" not in set(i.symbol)
    assert observed == {"cohort": "TCGA", "threshold": 1., "include_missing_ihc": True}


def test_rounded_prevalence_does_not_turn_rare_detection_into_zero():
    values = [0, .001, .008, .2, .992, 1, np.nan]
    assert [profiles._fraction_label(v) for v in values] == [
        "0%", "<1%", "<1%", "20%", ">99%", "100%", "NA",
    ]
