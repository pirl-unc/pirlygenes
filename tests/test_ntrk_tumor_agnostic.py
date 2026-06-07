"""Tumor-agnostic NTRK/ROS1 fusion drugs must not carry organ-specific
indications (#297).

larotrectinib / entrectinib / repotrectinib are FDA-approved tumor-agnostically
for NTRK gene-fusion-positive solid tumors (entrectinib/repotrectinib also for
ROS1+ NSCLC). Surfacing "NTRK-fusion GIST / thyroid / cholangiocarcinoma" as
the indication misrepresents the approval. The drugs stay placed under specific
cancer_codes (so they surface in the right report), but the ``indication`` must
read tumor-agnostic."""

from pirlygenes.load_dataset import get_data

_AGNOSTIC = ("larotrectinib", "entrectinib", "repotrectinib")
# organ words that must not appear in a tumor-agnostic indication
_ORGAN_WORDS = ("gist", "thyroid", "cholangiocarcinoma", "lung", "colorectal",
                "sarcoma", "melanoma", "breast")


def _agnostic_target_rows():
    key = get_data("cancer-key-genes")
    agent = key["agent"].astype(str).str.lower()
    mask = (key["role"] == "target") & agent.isin(_AGNOSTIC)
    return key[mask]


def test_ntrk_agnostic_drugs_have_tumor_agnostic_indication():
    rows = _agnostic_target_rows()
    assert len(rows) >= 1, "expected at least one tumor-agnostic NTRK target row"
    for r in rows.itertuples():
        ind = str(r.indication).lower()
        assert "tumor-agnostic" in ind or "solid tumor" in ind, (
            f"{r.agent} under {r.cancer_code}: indication {r.indication!r} is not "
            "phrased tumor-agnostic"
        )
        assert not any(w in ind for w in _ORGAN_WORDS), (
            f"{r.agent} under {r.cancer_code}: indication {r.indication!r} names a "
            "specific organ/histology for a tumor-agnostic drug"
        )
