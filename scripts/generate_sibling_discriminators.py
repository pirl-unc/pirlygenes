#!/usr/bin/env python
"""Generate general sibling-entity discriminator programs for #266.

The panels here describe reusable disease programs, not individual reference
samples.  Molecular/risk children inherit their parent's program downstream;
we therefore curate ``LAML`` once rather than copying the same genes onto every
ELN label.  Contrasts whose separation is intrinsically limited in bulk RNA
are marked ``poor`` instead of being made decisive with narrower thresholds.

COAD and READ are the important counterexample: their expression programs do
not support a leaf distinction.  Their existing degenerate-pair row is kept
marker-free and extended to the MSS children so anatomy remains the tiebreaker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease


DATA = Path(__file__).resolve().parent.parent / "pirlygenes" / "data"


# Each marker tuple is (symbol, direction, tier).  ``low`` is used only where
# absence of the counter-program is biologically meaningful.  The two
# within-myeloid contrasts intentionally rely on positive programs: forcing an
# absolute negative marker there would misrepresent overlapping differentiation
# states.
CONTRASTS = {
    "CML_vs_LAML": {
        "type_a": "CML",
        "type_b": "LAML",
        "separability": "poor",
        "source": "PMID:32405436;PMID:23634996;PMID:31857451",
        "favors": {
            "CML": [
                ("CEBPE", "high", "primary"),
                ("FCGR3B", "high", "primary"),
                ("CTSG", "high", "supporting"),
                ("AZU1", "high", "supporting"),
            ],
            "LAML": [
                ("FLT3", "high", "primary"),
                ("KIT", "high", "primary"),
                ("LILRB4", "high", "supporting"),
                ("CD34", "high", "supporting"),
            ],
        },
        "anchor": (
            "literature-curated blast-versus-granulocytic maturation programs; "
            "BCR::ABL1 remains required to establish CML"
        ),
    },
    "MPN_vs_LAML": {
        "type_a": "MPN",
        "type_b": "LAML",
        "separability": "poor",
        "source": "PMID:39820365;PMID:23634996;PMID:31857451",
        "favors": {
            "MPN": [
                ("PF4", "high", "primary"),
                ("PPBP", "high", "primary"),
                ("GP9", "high", "supporting"),
                ("ITGA2B", "high", "supporting"),
            ],
            "LAML": [
                ("MPO", "high", "primary"),
                ("ELANE", "high", "primary"),
                ("FLT3", "high", "supporting"),
                ("KIT", "high", "supporting"),
            ],
        },
        "anchor": (
            "literature-curated megakaryocytic-versus-acute-myeloid programs; "
            "driver and marrow evidence remain required for an MPN diagnosis"
        ),
    },
    "B_ALL_vs_LAML": {
        "type_a": "B_ALL",
        "type_b": "LAML",
        "separability": "strong",
        "source": "PMID:30046113;PMID:23634996;PMID:31857451",
        "favors": {
            "B_ALL": [
                ("DNTT", "high", "primary"),
                ("VPREB1", "high", "primary"),
                ("CD19", "high", "supporting"),
                ("PAX5", "high", "supporting"),
                ("MPO", "low", "primary"),
                ("ELANE", "low", "supporting"),
            ],
            "LAML": [
                ("MPO", "high", "primary"),
                ("ELANE", "high", "primary"),
                ("KIT", "high", "supporting"),
                ("ITGAM", "high", "supporting"),
                ("DNTT", "low", "primary"),
                ("VPREB1", "low", "supporting"),
            ],
        },
    },
    "FL_vs_LAML": {
        "type_a": "FL",
        "type_b": "LAML",
        "separability": "strong",
        "source": "PMID:33569544;PMID:23634996;PMID:31857451",
        "favors": {
            "FL": [
                ("MS4A1", "high", "primary"),
                ("CD19", "high", "primary"),
                ("CD79A", "high", "supporting"),
                ("BCL6", "high", "supporting"),
                ("MPO", "low", "primary"),
                ("ELANE", "low", "supporting"),
            ],
            "LAML": [
                ("MPO", "high", "primary"),
                ("ELANE", "high", "primary"),
                ("KIT", "high", "supporting"),
                ("ITGAM", "high", "supporting"),
                ("PAX5", "low", "primary"),
                ("MS4A1", "low", "supporting"),
            ],
        },
    },
    "STAD_vs_ESCA": {
        "type_a": "STAD",
        "type_b": "ESCA",
        "separability": "poor",
        "source": "PMID:25079317;PMID:28052061",
        "favors": {
            "STAD": [
                ("PGC", "high", "primary"),
                ("TFF2", "high", "primary"),
                ("MUC6", "high", "supporting"),
                ("CLDN18", "high", "supporting"),
                ("TP63", "low", "primary"),
                ("KRT5", "low", "supporting"),
            ],
            "ESCA": [
                ("TP63", "high", "primary"),
                ("KRT5", "high", "primary"),
                ("SOX2", "high", "supporting"),
                ("DSG3", "high", "supporting"),
                ("PGC", "low", "primary"),
                ("TFF2", "low", "supporting"),
            ],
        },
        "anchor": (
            "gastric versus squamous-esophageal programs; poor separability "
            "preserves the known esophageal-adenocarcinoma overlap"
        ),
    },
    "STAD_vs_CHOL": {
        "type_a": "STAD",
        "type_b": "CHOL",
        "separability": "moderate",
        "source": "PMID:25079317;PMID:28622513",
        "favors": {
            "STAD": [
                ("PGC", "high", "primary"),
                ("TFF2", "high", "primary"),
                ("MUC6", "high", "supporting"),
                ("CLDN18", "high", "supporting"),
                ("KRT7", "low", "primary"),
            ],
            "CHOL": [
                ("KRT7", "high", "primary"),
                ("EPCAM", "high", "primary"),
                ("KRT19", "high", "supporting"),
                ("SOX9", "high", "supporting"),
                ("PGC", "low", "primary"),
                ("TFF2", "low", "supporting"),
            ],
        },
    },
    "ESCA_vs_CHOL": {
        "type_a": "ESCA",
        "type_b": "CHOL",
        "separability": "poor",
        "source": "PMID:28052061;PMID:28622513",
        "favors": {
            "ESCA": [
                ("TP63", "high", "primary"),
                ("KRT5", "high", "primary"),
                ("SOX2", "high", "supporting"),
                ("DSG3", "high", "supporting"),
                ("KRT7", "low", "primary"),
            ],
            "CHOL": [
                ("KRT7", "high", "primary"),
                ("EPCAM", "high", "primary"),
                ("KRT19", "high", "supporting"),
                ("SOX9", "high", "supporting"),
                ("TP63", "low", "primary"),
                ("KRT5", "low", "supporting"),
            ],
        },
        "anchor": (
            "biliary versus squamous-esophageal programs; poor separability "
            "preserves the esophageal-adenocarcinoma caveat"
        ),
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


def _update_discriminators(genome: EnsemblRelease) -> int:
    path = DATA / "cancer-type-discriminators.csv"
    existing = pd.read_csv(path)
    existing = existing[~existing["contrast"].isin(CONTRASTS)].copy()
    rows = []
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
                        "support_type": "sibling_discriminator_literature",
                        "source_anchor": spec.get(
                            "anchor",
                            "pirlygenes#266 general sibling-entity discriminator program",
                        ),
                    }
                )
    generated = pd.DataFrame(rows, columns=existing.columns)
    pd.concat([existing, generated], ignore_index=True).to_csv(path, index=False)
    return len(rows)


def _update_colorectal_degeneracy() -> None:
    path = DATA / "degenerate-subtype-pairs.csv"
    pairs = pd.read_csv(path)
    mask = pairs["pair_id"] == "COAD_vs_READ"
    if mask.sum() != 1:
        raise SystemExit("expected exactly one COAD_vs_READ degenerate-pair row")
    pairs.loc[mask, "members"] = "COAD;COAD_MSS;READ;READ_MSS"
    pairs.loc[mask, "refs"] = "PMID:22810696"
    pairs.loc[mask, "source_anchor"] = (
        "TCGA colon/rectal molecular equivalence; anatomy remains the tiebreaker"
    )
    pairs.to_csv(path, index=False)


def main() -> int:
    rows = _update_discriminators(_newest_release())
    _update_colorectal_degeneracy()
    print(f"sibling discriminators: {rows} rows / {len(CONTRASTS)} contrasts")
    print("COAD/READ degeneracy: parent and MSS children, site-resolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
