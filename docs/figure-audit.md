# Figure-set audit (Track C1 of PLAN-post-v4.44-series.md)

Inventory + purpose + redundancy notes for every PNG/PDF emitted by
`pirlygenes analyze` as of v4.45.0. Baseline sample: the CRPC PRAD
real-world test case run at `/tmp/pirlygenes-sanity/s3i/`.

Format: ✓ keep, ~ review / improve, ✗ drop, + missing.

---

## Step-0 tissue composition + QC

| File | Purpose | Verdict |
|---|---|---|
| `sample-sample-summary.png` | One-panel overview — cancer type, purity, top tissues | ✓ keep — the "first glance" frame readers want |
| `sample-sample-context.png` | Library-prep / preservation / degradation diagnostic axes | ✓ keep — used by step-0 reasoning |
| `sample-degradation-index.png` | Long/short transcript length-pair scatter | ✓ keep — concrete preservation signal |
| `sample-background-tissues.png` | Top HPA-tissue correlations for the sample | ~ review — overlaps with the narrative "Top tissue matches" in summary.md; consider drop if always redundant |

## Cancer-type identification

| File | Purpose | Verdict |
|---|---|---|
| `sample-cancer-hypotheses.png` | Candidate cancer-type ranking chart | ✓ keep — the single best "why this call" visualization |
| `sample-mds-tme.png` | MDS embedding: sample among TCGA cohorts (TME-low gene space) | ✓ keep — independent view, catches ambiguous calls |
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
| `sample-provenance.png` | Sample composition stacked bar (#106) | ✓ keep — paired with provenance.md |
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

1. **`subtype-attribution-{cat}.png`** emitter for CAF / TAM / MDSC / exhausted-T / tumor-EC / TLS-B / TI-plasma — addresses the single biggest gap in the current figure set.
2. **Collapse gene-set strip plots** into one `sample-genesets.pdf` with curated-categories-only-by-default + `--all-genesets` opt-in.
3. **Dedupe attribution plots** — pick one of ``sample-target-attribution-*`` or ``sample-tumor-attribution-*`` and drop the other.
4. **Retire the three overview panels** (`sample-immune.png`, `sample-tumor.png`, `sample-antigens.png`) once the strip-plot consolidation lands.
