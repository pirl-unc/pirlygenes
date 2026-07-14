#!/usr/bin/env python
"""Generate the #266/#326 representative hard-case gate and its contrasts.

The downstream no-hint classifier gate currently has 13 cross-lineage misses.
They must remain visible while their remedies are split honestly between
discriminator-panel work, source QC, and mixture-aware scoring.  This script
owns two artifacts:

* ``cancer-discriminator-hard-cases.csv`` records the exact representatives,
  baseline attractors, required evidence, and resolution track.
* the corresponding rows in ``cancer-type-discriminators.csv`` provide cited
  positive and negative (``direction=low``) evidence for every hard pair.

The script preserves all unrelated discriminator rows and replaces only the
contrasts declared below, so it is safe to rerun after curation changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease


DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"
ISSUE_266 = "https://github.com/pirl-unc/pirlygenes/issues/266#issuecomment-4964643961"
ISSUE_326 = "https://github.com/pirl-unc/pirlygenes/issues/326#issuecomment-4964643967"


def _markers(*symbols: str) -> str:
    return ";".join(symbols)


HARD_CASES = [
    {
        "representative_id": "BRCA_rep02",
        "expected_code": "BRCA",
        "baseline_prediction": "SARC_EPITH",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("TRPS1", "SCGB2A2", "PIP"),
        "counter_lineage_markers": _markers("CD34", "CDH5", "ERG"),
        "notes": "Require a mammary program and audit the mesenchymal-heavy source vector.",
    },
    {
        "representative_id": "BRCA_Basal_rep02",
        "expected_code": "BRCA_Basal",
        "baseline_prediction": "SARC_EPITH",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("TRPS1", "SCGB2A2", "SOX10"),
        "counter_lineage_markers": _markers("CD34", "CDH5", "ERG"),
        "notes": "Require mammary-basal evidence rather than basal keratins alone.",
    },
    {
        "representative_id": "BRCA_Basal_rep05",
        "expected_code": "BRCA_Basal",
        "baseline_prediction": "SARC_OS",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("TRPS1", "SCGB2A2", "SOX10"),
        "counter_lineage_markers": _markers("ALPL", "IBSP", "RUNX2", "SPP1"),
        "notes": "Audit the osteoid-heavy source vector; do not override coherent osteoblast biology.",
    },
    {
        "representative_id": "GBM_rep03",
        "expected_code": "GBM",
        "baseline_prediction": "SARC_LPS_UNSPEC",
        "resolution_track": "panel",
        "expected_positive_markers": _markers("GFAP", "OLIG2", "AQP4"),
        "counter_lineage_markers": _markers("FABP4", "PPARG", "LPL"),
        "notes": "Retain glial-positive evidence while guarding against an adipocytic attractor.",
    },
    {
        "representative_id": "KICH_rep02",
        "expected_code": "KICH",
        "baseline_prediction": "SARC_UPS",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("FOXI1", "KIT", "PVALB", "EPCAM"),
        "counter_lineage_markers": _markers("EMP3", "VIM", "COL1A1"),
        "notes": "Require a chromophobe-renal epithelial program and audit weak renal lineage signal.",
    },
    {
        "representative_id": "NBL_rep03",
        "expected_code": "NBL",
        "baseline_prediction": "MTC",
        "resolution_track": "panel_and_proxy_qc",
        "expected_positive_markers": _markers("PHOX2B", "HAND2", "DBH"),
        "counter_lineage_markers": _markers("CALCA", "CALCB", "CEACAM5"),
        "notes": "Separate sympathetic noradrenergic identity from the MTC calcitonin program.",
    },
    {
        "representative_id": "NBL_MYCNnonamp_rep02",
        "expected_code": "NBL_MYCNnonamp",
        "baseline_prediction": "MTC",
        "resolution_track": "panel_and_proxy_qc",
        "expected_positive_markers": _markers("PHOX2B", "HAND2", "DBH"),
        "counter_lineage_markers": _markers("CALCA", "CALCB", "CEACAM5"),
        "notes": "Keep the subtype label while using the same NBL-vs-MTC lineage evidence.",
    },
    {
        "representative_id": "NET_MIDGUT_rep04",
        "expected_code": "NET_MIDGUT",
        "baseline_prediction": "SARC_LMS",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("CDX2", "TPH1", "INSM1"),
        "counter_lineage_markers": _markers("ACTG2", "MYH11", "LMOD1", "DES"),
        "notes": "Counts-to-TPM/source-QC sentinel; keep the sample even while smooth-muscle signal dominates.",
        "source_issue": f"{ISSUE_266};{ISSUE_326}",
    },
    {
        "representative_id": "RB_rep03",
        "expected_code": "RB",
        "baseline_prediction": "SARC_LPS_UNSPEC",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("CRX", "RCVRN", "RXRG"),
        "counter_lineage_markers": _markers("FABP4", "PPARG", "LPL"),
        "notes": "Retinal-lineage sentinel paired with the separately tracked all-fail RB QC policy.",
    },
    {
        "representative_id": "SARC_LPS_UNSPEC_rep02",
        "expected_code": "SARC_LPS_UNSPEC",
        "baseline_prediction": "READ_MSS",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("FABP4", "PPARG", "LPL"),
        "counter_lineage_markers": _markers("SATB2", "CDX2", "CDH17", "GUCY2C"),
        "notes": "Audit the colorectal-epithelial source signal before changing the classifier.",
    },
    {
        "representative_id": "SARC_SYN_rep02",
        "expected_code": "SARC_SYN",
        "baseline_prediction": "LAML",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("TLE1", "BCL2", "KRT19"),
        "counter_lineage_markers": _markers("PTPRC", "MPO", "ELANE"),
        "notes": "Require synovial-sarcoma evidence and audit the myeloid-heavy source vector.",
    },
    {
        "representative_id": "STAD_GS_rep02",
        "expected_code": "STAD_GS",
        "baseline_prediction": "BL",
        "resolution_track": "source_qc_and_panel",
        "expected_positive_markers": _markers("CLDN18", "TFF1", "GKN1"),
        "counter_lineage_markers": _markers("CD79A", "MS4A1", "PAX5"),
        "notes": "Audit the B-cell-dominant source vector before applying gastric lineage rescue.",
    },
    {
        "representative_id": "THYM_rep04",
        "expected_code": "THYM",
        "baseline_prediction": "CTCL",
        "resolution_track": "mixture_aware_panel",
        "expected_positive_markers": _markers("FOXN1", "PAX9", "KRT19"),
        "counter_lineage_markers": _markers("PTPRC", "CD3D", "TRAC"),
        "notes": "Score thymic epithelium within a T-cell-rich mixture; CD3 positivity alone must not exclude thymoma.",
    },
]


# Each marker tuple is (symbol, direction, tier).  Low-direction rows are
# deliberate anti-lineage evidence, not expression-absence claims in the source
# representative.  A source-QC hard case may currently violate them—that is the
# point of keeping the gate visible instead of overfitting the classifier.
CONTRASTS = {
    "BRCA_vs_SARC_EPITH": {
        "type_a": "BRCA",
        "type_b": "SARC_EPITH",
        "separability": "poor",
        "source": "PMID:23000897;PMID:9839166",
        "favors": {
            "BRCA": [
                ("TRPS1", "high", "primary"),
                ("SCGB2A2", "high", "primary"),
                ("PIP", "high", "supporting"),
                ("CD34", "low", "supporting"),
            ],
            "SARC_EPITH": [
                ("CD34", "high", "primary"),
                ("CDH5", "high", "primary"),
                ("ERG", "high", "supporting"),
                ("SCGB2A2", "low", "supporting"),
            ],
        },
    },
    "BRCA_Basal_vs_SARC_EPITH": {
        "type_a": "BRCA_Basal",
        "type_b": "SARC_EPITH",
        "separability": "poor",
        "source": "PMID:23000897;PMID:9839166",
        "favors": {
            "BRCA_Basal": [
                ("TRPS1", "high", "primary"),
                ("SCGB2A2", "high", "primary"),
                ("SOX10", "high", "supporting"),
                ("CD34", "low", "supporting"),
            ],
            "SARC_EPITH": [
                ("CD34", "high", "primary"),
                ("CDH5", "high", "primary"),
                ("ERG", "high", "supporting"),
                ("SCGB2A2", "low", "supporting"),
            ],
        },
    },
    "BRCA_Basal_vs_SARC_OS": {
        "type_a": "BRCA_Basal",
        "type_b": "SARC_OS",
        "separability": "moderate",
        "source": "PMID:23000897;PMID:32173717",
        "favors": {
            "BRCA_Basal": [
                ("TRPS1", "high", "primary"),
                ("SCGB2A2", "high", "primary"),
                ("SOX10", "high", "supporting"),
                ("ALPL", "low", "supporting"),
            ],
            "SARC_OS": [
                ("ALPL", "high", "primary"),
                ("RUNX2", "high", "primary"),
                ("IBSP", "high", "supporting"),
                ("SPP1", "high", "supporting"),
                ("SCGB2A2", "low", "supporting"),
            ],
        },
    },
    "GBM_vs_SARC_LPS_UNSPEC": {
        "type_a": "GBM",
        "type_b": "SARC_LPS_UNSPEC",
        "separability": "moderate",
        "source": "PMID:26061751;PMID:23511993",
        "favors": {
            "GBM": [
                ("GFAP", "high", "primary"),
                ("OLIG2", "high", "primary"),
                ("AQP4", "high", "supporting"),
                ("FABP4", "low", "supporting"),
            ],
            "SARC_LPS_UNSPEC": [
                ("FABP4", "high", "primary"),
                ("PPARG", "high", "primary"),
                ("LPL", "high", "supporting"),
                ("GFAP", "low", "supporting"),
            ],
        },
    },
    "KICH_vs_SARC_UPS": {
        "type_a": "KICH",
        "type_b": "SARC_UPS",
        "separability": "poor",
        "source": "PMID:23792563;PMID:41977488",
        "favors": {
            "KICH": [
                ("FOXI1", "high", "primary"),
                ("KIT", "high", "primary"),
                ("PVALB", "high", "supporting"),
                ("EPCAM", "high", "supporting"),
            ],
            "SARC_UPS": [
                ("EMP3", "high", "primary"),
                ("VIM", "high", "supporting"),
                ("COL1A1", "high", "supporting"),
                ("EPCAM", "low", "primary"),
                ("FOXI1", "low", "supporting"),
            ],
        },
    },
    "NBL_vs_MTC": {
        "type_a": "NBL",
        "type_b": "MTC",
        "separability": "moderate",
        "source": "PMID:28740262;PMID:28929017;PMID:26494386",
        "favors": {
            "NBL": [
                ("PHOX2B", "high", "primary"),
                ("HAND2", "high", "primary"),
                ("DBH", "high", "supporting"),
                ("CEACAM5", "low", "supporting"),
            ],
            "MTC": [
                ("CALCA", "high", "primary"),
                ("CALCB", "high", "primary"),
                ("CEACAM5", "high", "supporting"),
                ("HAND2", "low", "supporting"),
            ],
        },
    },
    "NBL_MYCNnonamp_vs_MTC": {
        "type_a": "NBL_MYCNnonamp",
        "type_b": "MTC",
        "separability": "moderate",
        "source": "PMID:28740262;PMID:28929017;PMID:26494386",
        "favors": {
            "NBL_MYCNnonamp": [
                ("PHOX2B", "high", "primary"),
                ("HAND2", "high", "primary"),
                ("DBH", "high", "supporting"),
                ("CEACAM5", "low", "supporting"),
            ],
            "MTC": [
                ("CALCA", "high", "primary"),
                ("CALCB", "high", "primary"),
                ("CEACAM5", "high", "supporting"),
                ("HAND2", "low", "supporting"),
            ],
        },
    },
    "NET_MIDGUT_vs_SARC_LMS": {
        "type_a": "NET_MIDGUT",
        "type_b": "SARC_LMS",
        "separability": "strong",
        "source": "PMID:15848904;PMID:19901961;PMID:28893210",
        "anchor": "pirlygenes#266 exact gate plus pirlygenes#326 source-QC follow-up",
        "favors": {
            "NET_MIDGUT": [
                ("CDX2", "high", "primary"),
                ("TPH1", "high", "primary"),
                ("INSM1", "high", "supporting"),
                ("ACTG2", "low", "primary"),
                ("MYH11", "low", "supporting"),
            ],
            "SARC_LMS": [
                ("ACTG2", "high", "primary"),
                ("MYH11", "high", "primary"),
                ("LMOD1", "high", "supporting"),
                ("DES", "high", "supporting"),
                ("TPH1", "low", "supporting"),
            ],
        },
    },
    "RB_vs_SARC_LPS_UNSPEC": {
        "type_a": "RB",
        "type_b": "SARC_LPS_UNSPEC",
        "separability": "moderate",
        "source": "PMID:41535675;PMID:23511993",
        "favors": {
            "RB": [
                ("CRX", "high", "primary"),
                ("RCVRN", "high", "primary"),
                ("RXRG", "high", "supporting"),
                ("FABP4", "low", "supporting"),
            ],
            "SARC_LPS_UNSPEC": [
                ("FABP4", "high", "primary"),
                ("PPARG", "high", "primary"),
                ("LPL", "high", "supporting"),
                ("RCVRN", "low", "supporting"),
            ],
        },
    },
    "SARC_LPS_UNSPEC_vs_READ_MSS": {
        "type_a": "SARC_LPS_UNSPEC",
        "type_b": "READ_MSS",
        "separability": "strong",
        "source": "PMID:23511993;PMID:22810696",
        "favors": {
            "SARC_LPS_UNSPEC": [
                ("FABP4", "high", "primary"),
                ("PPARG", "high", "primary"),
                ("LPL", "high", "supporting"),
                ("CDX2", "low", "primary"),
                ("EPCAM", "low", "supporting"),
            ],
            "READ_MSS": [
                ("SATB2", "high", "primary"),
                ("CDX2", "high", "primary"),
                ("CDH17", "high", "supporting"),
                ("GUCY2C", "high", "supporting"),
                ("FABP4", "low", "supporting"),
            ],
        },
    },
    "SARC_SYN_vs_LAML": {
        "type_a": "SARC_SYN",
        "type_b": "LAML",
        "separability": "strong",
        "source": "PMID:17255769;PMID:34673590;PMID:23634996",
        "favors": {
            "SARC_SYN": [
                ("TLE1", "high", "primary"),
                ("BCL2", "high", "primary"),
                ("KRT19", "high", "supporting"),
                ("PTPRC", "low", "primary"),
                ("MPO", "low", "supporting"),
            ],
            "LAML": [
                ("PTPRC", "high", "primary"),
                ("MPO", "high", "primary"),
                ("ELANE", "high", "supporting"),
                ("TLE1", "low", "supporting"),
                ("KRT19", "low", "supporting"),
            ],
        },
    },
    "STAD_GS_vs_BL": {
        "type_a": "STAD_GS",
        "type_b": "BL",
        "separability": "strong",
        "source": "PMID:25079317;PMID:31857451",
        "favors": {
            "STAD_GS": [
                ("CLDN18", "high", "primary"),
                ("TFF1", "high", "primary"),
                ("GKN1", "high", "supporting"),
                ("PTPRC", "low", "primary"),
                ("CD79A", "low", "supporting"),
            ],
            "BL": [
                ("CD79A", "high", "primary"),
                ("MS4A1", "high", "primary"),
                ("PAX5", "high", "supporting"),
                ("CLDN18", "low", "primary"),
                ("TFF1", "low", "supporting"),
            ],
        },
    },
    "THYM_vs_CTCL": {
        "type_a": "THYM",
        "type_b": "CTCL",
        "separability": "poor",
        "source": "PMID:28740671;PMID:32111619;PMID:39133553",
        "anchor": "mixture-aware gate: thymocyte signal alone must not exclude thymoma",
        "favors": {
            "THYM": [
                ("FOXN1", "high", "primary"),
                ("PAX9", "high", "primary"),
                ("KRT19", "high", "supporting"),
            ],
            "CTCL": [
                ("PTPRC", "high", "primary"),
                ("CD3D", "high", "primary"),
                ("TRAC", "high", "supporting"),
                ("FOXN1", "low", "primary"),
                ("PAX9", "low", "supporting"),
            ],
        },
    },
}


def _newest_release() -> EnsemblRelease:
    for release in range(115, 90, -1):
        try:
            genome = EnsemblRelease(release)
            genome.gene_ids()
            return genome
        except Exception:
            continue
    raise SystemExit("no installed GRCh38 Ensembl release")


def _resolve(symbol: str, genome: EnsemblRelease) -> str:
    ids = {gene.gene_id.split(".", 1)[0] for gene in genome.genes_by_name(symbol)}
    if len(ids) != 1:
        raise SystemExit(f"ambiguous/unknown ENSG for {symbol}: {sorted(ids)}")
    return ids.pop()


def main() -> int:
    hard_cases = pd.DataFrame(HARD_CASES)
    hard_cases["must_keep"] = True
    hard_cases["required_outcome"] = "lineage_correct"
    hard_cases["source_issue"] = hard_cases.get("source_issue", pd.Series(dtype=object))
    hard_cases["source_issue"] = hard_cases["source_issue"].fillna(ISSUE_266)
    hard_cases = hard_cases[
        [
            "representative_id",
            "expected_code",
            "baseline_prediction",
            "resolution_track",
            "expected_positive_markers",
            "counter_lineage_markers",
            "must_keep",
            "required_outcome",
            "source_issue",
            "notes",
        ]
    ]
    hard_cases.to_csv(DATA / "cancer-discriminator-hard-cases.csv", index=False)

    genome = _newest_release()
    path = DATA / "cancer-type-discriminators.csv"
    existing = pd.read_csv(path)
    existing = existing[~existing["contrast"].isin(CONTRASTS)].copy()
    rows = []
    default_anchor = (
        "pirlygenes#266 exact representative hard-case gate; "
        "positive and negative lineage evidence"
    )
    for contrast, spec in CONTRASTS.items():
        for favored_code, markers in spec["favors"].items():
            for symbol, direction, tier in markers:
                rows.append(
                    {
                        "contrast": contrast,
                        "type_a": spec["type_a"],
                        "type_b": spec["type_b"],
                        "favors": favored_code,
                        "Symbol": symbol,
                        "Ensembl_Gene_ID": _resolve(symbol, genome),
                        "direction": direction,
                        "tier": tier,
                        "separability": spec["separability"],
                        "source": spec["source"],
                        "support_type": "hard_case_discriminator_literature",
                        "source_anchor": spec.get("anchor", default_anchor),
                    }
                )
    generated = pd.DataFrame(rows, columns=existing.columns)
    pd.concat([existing, generated], ignore_index=True).to_csv(path, index=False)
    print(f"hard-case gate: {len(hard_cases)} representatives")
    print(f"hard-case contrasts: {len(rows)} rows / {len(CONTRASTS)} contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
