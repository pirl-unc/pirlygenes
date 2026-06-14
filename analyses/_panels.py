"""Central registry of the semantically-meaningful gene / cohort panels used by
the anti-PD-1 / ICI analysis scripts.

ONE home so the panels are discoverable and can't drift between scripts, and ONE
general mechanism for paralog/proteoform handling: every gene panel is folded
onto the matrix's proteoform-ID columns via the package rollup
(:func:`pirlygenes.expression.protein_groups.fold_to_cdna_canonical_symbol`), so
a curated member symbol like ``CTAG1B`` resolves to the matrix's ``CTAG1A/B``
column instead of silently dropping — no per-script special-case lists.

The curated, literature-grounded *signatures* still live in the package CSV
(``data/therapy-response-signatures.csv``) — already a single source of truth;
this module holds only the analysis-layer (non-CSV) panels.

Display convention (one canonical form per layer, never interchangeably):
  * DATA / keys / matrix columns / CSV headers use the **structural** identifier
    — the proteoform ID for a fold (``CTAG1A/B``) or the raw symbol otherwise.
  * USER-FACING labels (plot axes, titles, annotations, markdown tables) go
    through :func:`pirlygenes.gene_names.display_name` — the sole display
    authority — which maps both the single locus ``CTAG1B`` and the folded
    proteoform ``CTAG1A/B`` to ``NY-ESO-1``. Re-exported here as
    :func:`display_label` so every analysis script shares one render boundary.
"""
from __future__ import annotations

from pirlygenes.expression.protein_groups import fold_to_cdna_canonical_symbol
from pirlygenes.gene_names import display_name as display_label
from pirlygenes.gene_sets_cancer import CTA_gene_names

__all__ = ["fold", "display_label", "cta_antigen_panel", "GENE_PANELS",
           "mechanism_controls", "GYN_COLD", "HOT"]


def fold(genes) -> list[str]:
    """The one general paralog/proteoform mechanism for panels: fold symbols onto
    the matrix's proteoform-ID columns (idempotent; single-copy genes pass
    through, de-duplicated)."""
    return fold_to_cdna_canonical_symbol(genes)


def cta_antigen_panel() -> list[str]:
    """The full curated cancer-testis-antigen panel (``CTA_gene_names`` — the
    package source of truth), proteoform-folded. Replaces the old hand-picked
    'representative MAGEs' list with the general panel."""
    return fold(CTA_gene_names())


# ── analysis-layer gene panels (folded on access via GENE_PANELS) ─────────────
_RAW_GENE_PANELS = {
    # secreted immune-EXCLUSION genes (ici_landscape suppression term)
    "secreted_inhibitory": ["TGFB1", "WNT5A", "WNT11", "IL10"],
    # exclusion-mechanism composites (exclusion_vs_apd1)
    "tgfb_ligand_caf": ["TGFB1", "TGFB3", "FAP", "CXCL12", "LRRC15", "POSTN",
                        "COL11A1"],
    "pge2_cox2": ["PTGS2", "PTGES"],
    "angiogenic": ["VEGFA"],
    "myeloid_cxcr2": ["CSF1", "CXCL1", "CXCL5"],
    "gyn_tolerance": ["VTCN1", "MUC1", "FOLR1"],
    "exclusion_pool_extras": ["GAS6", "ISLR", "PMEPA1", "TGFB2", "ACTA2",
                              "CTNNB1"],
    # mechanism-screen NON-curated controls / known failures (apd1_mechanism_screen)
    "erv_annotated": ["ERV3-1", "ERVK-28", "ERVK3-1", "ERVMER34-1", "ERVW-1",
                      "ERVFRD-1", "ERVH-1", "ERVV-1"],
    "hypoxia": ["CA9", "SLC2A1", "LDHA", "VEGFA", "PGK1"],
    "myeloid_tolerance": ["VSIG4", "MARCO", "CD163", "C1QA", "C1QB", "ARG1",
                          "ALDH1A1", "IL10"],
}
GENE_PANELS = {name: fold(genes) for name, genes in _RAW_GENE_PANELS.items()}


def mechanism_controls() -> dict:
    """NON-curated mechanism-screen controls as
    ``{name: (folded_genes, expected_sign, circularity_tag, rationale)}``. The
    CTA panel is the full package CTA panel folded (general mechanism), not a
    one-off representative list."""
    return {
        "antigen:CTA": (
            cta_antigen_panel(), +1, "causal",
            "cancer-testis antigens (full curated CTA panel, proteoform-folded; "
            "panel owned by tsarina)"),
        "antigen:ERV_annotated": (
            GENE_PANELS["erv_annotated"], +1, "causal",
            "Ensembl-annotated ERVs (mostly placental syncytins - FAILS)"),
        "exclude:hypoxia": (
            GENE_PANELS["hypoxia"], -1, "borderline",
            "hypoxia (CA9 is also a ccRCC lineage marker - confounded, NOT curated)"),
        "exclude:myeloid_tolerance": (
            GENE_PANELS["myeloid_tolerance"], -1, "borderline",
            "resident tolerogenic myeloid (WRONG-SIGN: indexes infiltrate; NOT curated)"),
    }


# ── shared cohort (cancer-type) panels ────────────────────────────────────────
# genuinely COLD gyn cohorts (UCEC_MSI is MSI-H/hot -> excluded). Shared by
# exclusion_vs_apd1 + apd1_causal_factors so the archetype set can't drift.
GYN_COLD = ["OV", "BRCA_Basal", "UCEC_CNL", "UCEC_CNH"]
HOT = ["SKCM"]
