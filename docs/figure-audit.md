# Figure-set audit

Inventory + purpose + similar-purpose groupings for every PNG/PDF
emitted by `pirlygenes analyze`.

**Round 1** (v4.45.0 baseline, shipped 2026-04-21): verdict format below was
✓ keep / ~ review / ✗ drop / + missing. Actions taken based on that
round:

- C2 subtype-attribution figures shipped (PR #190 → v4.46.0)
- C3 redundant tumor-attribution plots dropped (PR #189 → v4.45.1)
- C4 overview strip plots retired (PR #191 → v4.45.2)

**Round 2** (v4.46.0, this doc): pure enumeration + similar-purpose
groupings — **no verdicts, no removals in this round**. The goal is a
shared reference the reader can audit later to decide if any cluster
should be consolidated. 45 files baseline after QC-plot additions.

---

## Step-0 tissue composition + QC

| File | Purpose | Verdict |
|---|---|---|
| `sample-sample-summary.png` | One-panel overview — cancer type, purity, top tissues | ✓ keep — the "first glance" frame readers want |
| `sample-sample-context.png` | Library-prep / preservation / degradation diagnostic axes | ✓ keep — used by step-0 reasoning |
| `sample-expression-concentration-qc.png` | Dominant-gene + cumulative TPM concentration QC, including semantic classes such as rRNA pseudogene or mtDNA | ✓ keep — explains denominator-distorting artifacts |
| `sample-qc-reference-mtdna-rrna.png` | Sample mtDNA, nuclear rRNA-like, and rRNA-pseudogene fractions against TCGA/HPA reference-column distributions | ✓ keep — shows whether technical RNA burden is out of reference range and separates pseudogene dominance from intact rRNA |
| `sample-qc-reference-technical-rna-burden.png` | Combined mtDNA+rRNA-like fraction against TCGA/HPA reference-column distributions | ✓ keep — one-axis summary of the same QC risk |
| `sample-degradation-index.png` | Long/short transcript length-pair scatter | ✓ keep — concrete preservation signal |
| `sample-background-tissues.png` | Top HPA-tissue correlations for the sample | ~ review — overlaps with the narrative "Top tissue matches" in summary.md; consider drop if always redundant |

## Cancer-type identification

| File | Purpose | Verdict |
|---|---|---|
| `sample-cancer-hypotheses.png` | Candidate cancer-type ranking chart | ✓ keep — the single best "why this call" visualization |
| `sample-reference-mds.png` | MDS embedding: sample among cancer, subtype, and normal references | ✓ keep — independent view, catches ambiguous calls |
| `sample-vs-cancer.pdf` | Per-category scatter vs reference cohort (multi-page) | ~ review — large PDF, only opened occasionally; consider lazy / on-demand generation |
| `sample-subtype-signature.png` | Therapy-response axes (AR/ER/HER2/NE/EMT/hypoxia/IFN) | ✓ keep — drives disease-state narrative |

## Purity

| File | Purpose | Verdict |
|---|---|---|
| `sample-purity.png` | Tumor purity detail — CI, components, integration | ✓ keep |
| `sample-purity-methods.png` | All purity-estimation methods on one axis + direct/derived separator (polished in v4.40.1) | ✓ keep |
| `sample-purity-targets.png` | Tumor-expression ranges for therapeutic targets | ✓ keep |
| `sample-purity-ctas.png` | Same for CTAs | ✓ keep |
| `sample-purity-surface.png` | Same for surface proteins | ✓ keep |

## Decomposition

| File | Purpose | Verdict |
|---|---|---|
| `sample-decomposition-composition.png` | Standalone composition bar (tumor + TME) — best hypothesis | ✓ keep — the "what's in the sample" bar |
| `sample-decomposition-components.png` | Standalone TME cell-type breakdown | ✓ keep |
| `sample-decomposition-candidates.png` | Per-candidate composition across top decomposition hypotheses | ✓ keep |

## Target landscape (present + attribution)

| File | Purpose | Verdict |
|---|---|---|
| `sample-target-tissues.pdf` | Therapy-target TPM across host tissues | ~ review — PDF; consider PNG-by-target or drop |
| `sample-target-safety.png` | Therapy-target normal-tissue expression — on-target/off-tissue safety | ✓ keep |
| `sample-target-attribution-targets.png` | Per-gene attribution stacked bars (targets) | ✓ keep |
| `sample-target-attribution-ctas.png` | Same for CTAs | ✓ keep |
| `sample-target-attribution-surface.png` | Same for surface | ✓ keep |
| `sample-tumor-attribution-targets.png` | Per-gene tumor-attributed TPM (targets) | ~ review — heavy overlap with `target-attribution-*`; two names for near-identical views. **Merge or drop one.** |
| `sample-tumor-attribution-cta.png` | Same for CTAs | ~ review (same as above) |
| `sample-targets-deep-dive.png` | Top-N actionable targets, detailed | ✓ keep |
| `sample-cta-deep-dive.png` | Top-N CTA deep dive | ✓ keep |
| `sample-matched-normal-targets.png` | Matched-normal vs tumor for each target | ✓ keep — answers the #131 "matched-normal over-predicted" story |
| `sample-matched-normal-ctas.png` | Same for CTAs | ✓ keep |
| `sample-matched-normal-surface.png` | Same for surface | ✓ keep |

## Gene-set strip plots (10 files — Cancer_surfaceome, CTAs, DNA_repair, Growth_receptors, Immune_checkpoints, Interferon_response, MHC1_presentation, Oncogenes, TLR, Tumor_suppressors)

| Verdict | Rationale |
|---|---|
| ~ review as a batch | One PNG per gene-set category. Useful for curated-panel review but the 10-file flood overwhelms the figures directory. Consider: (a) consolidate into a single multi-panel `sample-genesets.pdf` with one page per category, OR (b) only emit the top-5 most clinically-relevant categories (Oncogenes, Tumor_suppressors, Immune_checkpoints, MHC1_presentation, Growth_receptors) by default + a `--all-genesets` flag for the rest. |

## Legacy / overlapping

| File | Purpose | Verdict |
|---|---|---|
| `sample-immune.png` | Immune gene-set expression overview | ~ review — heavy overlap with the strip plots above |
| `sample-tumor.png` | Tumor gene-set expression overview | ~ review — same |
| `sample-antigens.png` | CTA + surfaceome expression overview | ~ review — same (dominated by the CTAs.png + Cancer_surfaceome.png strip plots) |
| `sample-treatments.png` | Therapy-target expression by modality | ✓ keep — modality axis isn't on any other figure |
| `sample-mhc-expression.png` | HLA-A/B/C + B2M bar | ✓ keep — drives the MHC-I status line in reports |

## Narrative / provenance

| File | Purpose | Verdict |
|---|---|---|
| `sample-provenance.png` | Sample composition stacked bar (#106) | ✓ keep — paired with evidence.md |
| `sample-therapy-pathway-state.png` | Dumbbell of therapy-response axes (up/down-panel fold) | ✓ keep (polished in v4.40.1) |

---

## Missing / deferred

- **`subtype-attribution-{cat}.png`** — per-gene before/after delta bars for the CAF / TAM / MDSC / exhausted-T refinement introduced in #56 / #58. Referenced in both issue bodies; provenance columns (`subtype_refined`, `tme_tpm_before_subtype_refinement`) already populate the TSV but no visualization has landed yet. Low/medium effort — would fit the per-category-PNG pattern the user prefers.
- **`subtype-attribution-summary.png`** — single PNG ranking the top-N most-affected genes across all refined compartments. Useful to see "these 12 genes had their tumor-attribution corrected by X TPM" at a glance.
- **Composition subtype-split bar** — #56 issue calls out showing the CAF-vs-generic-fibroblast / TAM-vs-monocyte split inside the composition figure. Currently the partition helper exists (`partition_compartment`) but the display layer doesn't consume it.

## Top redundancies (if reducing figure count)

1. **`sample-tumor-attribution-*` vs `sample-target-attribution-*`** — near-identical per-gene attribution views for the same categories. Drop one set.
2. **`sample-immune.png` + `sample-tumor.png` + `sample-antigens.png`** — overview strip plots duplicating the curated gene-set strip plots. Either retire the overviews or retire the per-category PNGs; pick one axis of organization.
3. **`sample-background-tissues.png`** — if the tissue-match line in the narrative is enough for clinicians, the standalone PNG is optional.

## Top missing

The subtype-attribution-{cat}.png figures are the single biggest gap — the #56/#58 refinement mechanism now runs on every sample but a reader can't see its per-gene effect visually, only in the TSV.

## Recommended next steps (PRs to open)

1. **`subtype-attribution-{cat}.png`** emitter for CAF / TAM / MDSC / exhausted-T / tumor-EC / TLS-B / TI-plasma — addresses the single biggest gap in the current figure set. **Shipped via PR #190 (v4.46.0).**
2. **Collapse gene-set strip plots** into one `sample-genesets.pdf` with curated-categories-only-by-default + `--all-genesets` opt-in. **Deferred.**
3. **Dedupe attribution plots** — pick one of ``sample-target-attribution-*`` or ``sample-tumor-attribution-*`` and drop the other. **Shipped via PR #189 (v4.45.1).**
4. **Retire the three overview panels** (`sample-immune.png`, `sample-tumor.png`, `sample-antigens.png`) once the strip-plot consolidation lands. **Shipped via PR #191 (v4.45.2).**

---

## Round-2 enumeration (v4.46.0+) — 45 files

Pure enumeration + similar-purpose groupings. No removals in this
round. Each figure has a one-line purpose. Groups collect figures
that answer overlapping questions — the reader can later decide if
any group deserves consolidation.

### Full inventory

| # | File | Purpose |
|---|---|---|
| 1 | `sample-sample-summary.png` | One-panel overview — cancer call, purity, top tissues |
| 2 | `sample-sample-context.png` | Library-prep / preservation / degradation diagnostic axes |
| 3 | `sample-expression-concentration-qc.png` | Dominant-gene and cumulative TPM concentration QC |
| 4 | `sample-qc-reference-mtdna-rrna.png` | mtDNA, nuclear rRNA-like, and rRNA-pseudogene fractions against reference-column distributions |
| 5 | `sample-qc-reference-technical-rna-burden.png` | Combined mtDNA+rRNA-like burden against reference-column distributions |
| 6 | `sample-degradation-index.png` | Long/short transcript length-pair scatter |
| 7 | `sample-background-tissues.png` | Top HPA-tissue correlations for the sample |
| 8 | `sample-cancer-hypotheses.png` | Candidate cancer-type ranking chart |
| 9 | `sample-reference-mds.png` | MDS embedding: sample among cancer, subtype, and normal references |
| 10 | `sample-vs-cancer.pdf` | Per-category scatter vs reference cohort (multi-page) |
| 11 | `sample-subtype-signature.png` | Therapy-response axes — AR/ER/HER2/NE/EMT/hypoxia/IFN |
| 12 | `sample-purity.png` | Tumor purity detail — CI + component contributions |
| 13 | `sample-purity-methods.png` | All purity-estimation methods on one axis |
| 14 | `sample-purity-targets.png` | Tumor-expression ranges (targets) |
| 15 | `sample-purity-ctas.png` | Tumor-expression ranges (CTAs) |
| 16 | `sample-purity-surface.png` | Tumor-expression ranges (surface proteins) |
| 17 | `sample-decomposition-composition.png` | Composition bar — tumor + TME for best hypothesis |
| 18 | `sample-decomposition-components.png` | TME cell-type breakdown for best hypothesis |
| 19 | `sample-decomposition-candidates.png` | Per-candidate composition across top hypotheses |
| 20 | `sample-target-tissues.pdf` | Therapy-target TPM across host tissues |
| 21 | `sample-target-safety.png` | Therapy-target normal-tissue expression (safety view) |
| 22 | `sample-target-attribution-targets.png` | Per-gene stacked attribution (targets) |
| 23 | `sample-target-attribution-ctas.png` | Per-gene stacked attribution (CTAs) |
| 24 | `sample-target-attribution-surface.png` | Per-gene stacked attribution (surface) |
| 25 | `sample-matched-normal-targets.png` | Matched-normal vs tumor per target (targets) |
| 26 | `sample-matched-normal-ctas.png` | Same (CTAs) |
| 27 | `sample-matched-normal-surface.png` | Same (surface) |
| 28 | `sample-subtype-attribution-targets.png` | #56/#58 before/after reference swap (targets) |
| 29 | `sample-subtype-attribution-surface.png` | Same (surface) |
| 30 | `sample-targets-deep-dive.png` | Top actionable targets, detailed |
| 31 | `sample-cta-deep-dive.png` | Top CTAs, detailed |
| 32 | `sample-mhc-expression.png` | HLA-A/B/C + B2M bar |
| 33 | `sample-provenance.png` | 5-step attribution chain stacked bar |
| 34 | `sample-therapy-pathway-state.png` | Dumbbell of therapy-response axes fold-change |
| 35 | `sample-treatments.png` | Therapy-target expression by modality (ADC/TCR-T/…) |
| 36 | `Cancer_surfaceome.png` | Strip plot — surfaceome gene set |
| 37 | `CTAs.png` | Strip plot — CTA gene set |
| 38 | `DNA_repair.png` | Strip plot — DNA-repair gene set |
| 39 | `Growth_receptors.png` | Strip plot — growth-receptor gene set |
| 40 | `Immune_checkpoints.png` | Strip plot — immune-checkpoint gene set |
| 41 | `Interferon_response.png` | Strip plot — IFN-response gene set |
| 42 | `MHC1_presentation.png` | Strip plot — MHC-I presentation gene set |
| 43 | `Oncogenes.png` | Strip plot — oncogene set |
| 44 | `TLR.png` | Strip plot — TLR gene set |
| 45 | `Tumor_suppressors.png` | Strip plot — tumor-suppressor gene set |

### Similar-purpose groupings

Groups collect figures that answer overlapping questions. Rough
"decision axis" per group noted so a future consolidation PR has a
concrete starting point. No figure is being removed here.

**G1: Sample-level snapshot**
Which kind of sample is this + QC? Candidates for a single-composite at glance:
- `sample-sample-summary.png` (#1)
- `sample-sample-context.png` (#2)
- `sample-expression-concentration-qc.png` (#3)
- `sample-qc-reference-mtdna-rrna.png` (#4)
- `sample-qc-reference-technical-rna-burden.png` (#5)
- `sample-degradation-index.png` (#6)
- `sample-background-tissues.png` (#7)

**G2: Cancer-type identification**
What cancer, how confident?
- `sample-cancer-hypotheses.png` (#8) — scalar ranking
- `sample-reference-mds.png` (#9) — 2-D embedding
- `sample-vs-cancer.pdf` (#10) — per-category scatter
- `sample-subtype-signature.png` (#11) — therapy-response axes that disambiguate subtypes

**G3: Purity + tumor-expression ranges**
How confident is the purity estimate + per-target implied expression?
- `sample-purity.png` (#12) — overall purity detail
- `sample-purity-methods.png` (#13) — method-comparison view
- `sample-purity-targets.png` (#14)
- `sample-purity-ctas.png` (#15)
- `sample-purity-surface.png` (#16)
(The 3 per-category "purity-*" views share their format; could be a
single multi-panel PDF if a reader typically consults all three.)

**G4: Decomposition / composition**
What else is in the sample besides tumor?
- `sample-decomposition-composition.png` (#17) — tumor + TME bar
- `sample-decomposition-components.png` (#18) — TME cell-type bar
- `sample-decomposition-candidates.png` (#19) — per-candidate composition
- `sample-provenance.png` (#33) — 5-step attribution chain bar

**G5: Therapy-target landscape**
What can this sample be treated with + how reliable is each target?
Per-category triplets are the dominant pattern here:

- **Attribution**: `sample-target-attribution-targets.png` (#22) + `-ctas` (#23) + `-surface` (#24)
- **Matched-normal**: `sample-matched-normal-targets.png` (#25) + `-ctas` (#26) + `-surface` (#27)
- **Subtype-refined before/after**: `sample-subtype-attribution-targets.png` (#28) + `-surface` (#29)

Plus per-category singletons:
- `sample-target-tissues.pdf` (#20) — tissue-expression PDF
- `sample-target-safety.png` (#21) — normal-tissue safety
- `sample-targets-deep-dive.png` (#30) + `-cta-deep-dive.png` (#31)

Six files (#22-29) share the "per-gene stacked bar, one PNG per
category" idiom. If a reader generally opens one category's
attribution / matched-normal / subtype-refined views together, a
per-category composite might read better than three separate PNGs.

**G6: Therapy program / modality view**
- `sample-therapy-pathway-state.png` (#34) — dumbbell of AR/ER/HER2/… axes
- `sample-treatments.png` (#35) — targets by modality (ADC/TCR-T/…)
- `sample-subtype-signature.png` (#11) — **cross-listed with G2** (it's also a therapy-axis summary)

**G7: Curated gene-set strip plots (10 files)**
Each organised by a single gene-set category:
- `Oncogenes.png` (#43), `Tumor_suppressors.png` (#45) — tumor biology
- `Immune_checkpoints.png` (#40), `MHC1_presentation.png` (#42), `Interferon_response.png` (#41), `TLR.png` (#44) — immune
- `CTAs.png` (#37), `Cancer_surfaceome.png` (#36) — antigens
- `Growth_receptors.png` (#39), `DNA_repair.png` (#38) — therapy-axis support

All follow the same strip-plot idiom. Candidate for a single
`sample-genesets.pdf` one-page-per-category + `--all-genesets`
opt-in, OR keep split for per-panel inspection.

**G8: MHC / immune surface**
- `sample-mhc-expression.png` (#32)
- `MHC1_presentation.png` (#42) — cross-listed with G7
(These answer the same question from different angles: the
observed MHC-I loaded protein TPM vs the curated MHC-I presentation
panel's gene-set pattern.)

### Groups most likely to benefit from consolidation

| Group | Why | Consolidation idea |
|---|---|---|
| G5 per-category triplets | 6-9 files sharing a layout, read per-category | Three composites: `{attribution+matched-normal+subtype}-{targets|ctas|surface}.png` |
| G7 gene-set strip plots | 10 files with the same idiom; readers skim not compare | `sample-genesets.pdf` with one page per category |
| G8 MHC | Two files, same biology | Fold the curated panel into the MHC bar figure as a second row |

### Groups where the split is load-bearing

- G2 (cancer-type ID) — each view answers a different question (rank vs embedding vs per-category scatter); splitting matters for reader drill-down
- G3 per-category purity — per-category is the expected drill-down
- G4 decomposition — each figure is a different slice
- G6 therapy program — different axes of "treatment story"

---

## TODO — deferred work (not being acted on in this audit)

- **G5 composite per-category** (attribution + matched-normal + subtype combined PNG per category) — would drop 3 files per category if the reader actually consumes them together.
- **G7 consolidation** to a single `sample-genesets.pdf` — removes 10 files; retrievable via `--all-genesets`.
- **G8 MHC fold-in** — second row on the MHC bar showing the MHC-I presentation panel strip; -1 file.
- **FN1-EDB transcript-lookup release iteration** — should early-exit after the first release that returns exon data, rather than iterating all installed releases (one of which may have a legacy schema; see PR #192). Not a figure issue — tracked here as a related TODO.
