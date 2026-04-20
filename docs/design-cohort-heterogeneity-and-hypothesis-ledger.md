# Design: cohort heterogeneity + auditable hypothesis ledger

Two related design problems pirlygenes needs to solve to grow past its
current "one median per cohort, each stage picks the top candidate" shape.

1. **Cohort heterogeneity** — not every TCGA cohort is a biologically
   coherent category. SARC is a union of at least eight lineage-distinct
   subtypes; BRCA is one lineage with modifier subtypes; GBM sits in
   between. The ranking / purity / lineage math assumes a single median
   is meaningful, and breaks when that assumption doesn't hold.
2. **Auditable hypothesis trace** — the pipeline narrows hypotheses
   across stages but the narrowing is implicit (each stage picks top-k
   of its predecessor without recording what was rejected or why). The
   reader can see "final call = X" but can't reconstruct the path.

This document lays out a conceptual framework for both and proposes a
concrete shape for the code changes that would implement them.

---

## Part 1 — cohort heterogeneity

### Three kinds of cohort

Cancer-type cohorts aren't all the same shape. For ranking / purity
purposes we need to distinguish:

**Type A — Convergent subtypes** (shared root biology + modifier axes)

Examples: `BRCA`, `PRAD`, `COAD`, `LIHC`, `UCEC`.

- A shared lineage program defines the cohort (breast epithelium,
  prostate luminal, colonic enterocyte, hepatocyte).
- Subtypes (Luminal A/B, HER2+, TNBC for BRCA; CRPC vs castration-
  sensitive for PRAD) are **modifiers** on that shared root — they
  amplify, suppress, or re-weight the root program.
- Cohort-level median IS biologically meaningful: it captures the
  shared root.
- Subtype distinctions come from:
  - Drivers (ERBB2 amp, ESR1 / PGR expression, BRCA1/2 mutations)
  - Therapeutic-state axes (ADT exposure, AR activity)
  - Morphology (ductal vs lobular — transcriptomically subtle)
- Handling: **one curated lineage panel** for the root + separate
  signature / therapy-axis panels for modifiers.

**Type B — Disjoint mixture** (union of biologically distinct tumors)

Examples: `SARC`, partially `DLBC` (ABC vs GCB), maybe `PCPG`.

- The cohort label is nosological / clinical, not biological. TCGA
  SARC lumps together LMS (smooth-muscle program), DDLPS (MDM2/CDK4
  amplicon), WDLPS (lipogenic), MYXLPS (FUS-DDIT3), UPS (undiff-
  mesenchymal), synovial sarcoma (SS18-SSX fusion), RMS
  (skeletal-muscle program), DSRCT (EWS-WT1), GIST (KIT/PDGFRA).
- Each subtype has a **different tissue of origin** and a different
  driver program. No shared lineage program bridges them beyond a
  loose "mesenchymal" axis.
- Cohort-level median is NOT biologically meaningful: arithmetic
  average over divergent entities. MYOD1 (RMS-specific) sits at
  TCGA-SARC median ≈ 0 because only ~12% of the cohort is RMS.
- Handling: **union of per-subtype lineage panels**. Each subtype
  gets its own panel (LMS: MYH11, CNN1, TAGLN, SMTN; RMS: MYOD1,
  MYOG, MYF5, DES; DDLPS: MDM2, CDK4, HMGA2, FRS2; ...). Cohort-
  level "lineage score" = MAX over subtype scores with a
  "matched-subtype" hypothesis attached.

**Type C — Molecular subtypes along a continuum**

Examples: `GBM` (classical / mesenchymal / proneural / neural),
`LGG` (oligo / astro / IDHmut-non-codel), possibly `UCEC`
(endometrioid vs serous).

- Same anatomical tumor type + same tissue of origin.
- Molecular subtypes have distinct transcriptomic programs on top
  of a shared root.
- Between Type A and Type B: most samples express the core panel,
  but there's a clean secondary axis separating subtypes.
- Handling: **one shared lineage panel** + subtype-axis scoring for
  the signature component (not lineage).

### How to diagnose which type a cohort is

Don't hand-annotate — derive from the data:

1. **Lineage-panel coherence across samples**. For each candidate
   lineage marker, what fraction of cohort samples express it above
   a threshold? Type A: every panel gene is expressed in > 70% of
   samples. Type B: each panel gene is expressed in < 40% of
   samples (different subsets for different genes).
2. **Subtype-vs-cohort-median divergence**. When subtype
   annotations exist (TCGA provides them for most cohorts), compute
   the correlation between each subtype median and the cohort
   median. Type A: all subtype medians ≥ 0.8 ρ with cohort median.
   Type B: subtype medians < 0.5 ρ with cohort median (and with
   each other).
3. **Proliferation-panel dispersion**. Type A: tight distribution
   around cohort median. Type B: multiple modes.

### Implementation plan

Three phases, each independently valuable:

**Phase 1** — register a `heterogeneity_class` flag per cohort.
Annotate each TCGA code as `atomic` / `convergent` / `mixture` /
`molecular_axis` in the registry. No scoring changes yet — just
the tag so downstream stages can make heterogeneity-aware choices.

**Phase 2** — mixture-aware lineage scoring. When a cohort is tagged
`mixture`, its lineage panel is defined as a **union of subtype
panels** (dict: subtype_id → panel genes). The lineage estimator
produces one purity estimate per subtype; cohort-level lineage
score = max over subtypes. Classifier output carries the winning
subtype hypothesis:

    Cancer call: SARC (subtype: rhabdomyosarcoma-consistent;
    MYOD1 / MYOG / DES / MYF5 pattern match 0.84)

**Phase 3** — convergent / molecular-axis support. For `convergent`
cohorts, continue using single median (no change needed today). For
`molecular_axis` cohorts, add a subtype-axis signature without
splitting the lineage panel.

### Acceptance

- Every cohort in the registry has a `heterogeneity_class`
  annotation with a justification.
- A real-sarcoma validation sample with a muscle-lineage program
  classifies as `SARC` with a subtype hypothesis that names the
  RMS panel.
- 33/33 median-battery test still passes.
- BRCA / PRAD / COAD (convergent cohorts) ranking unchanged.

---

## Part 2 — auditable hypothesis ledger

### The gap today

pirlygenes has stage-by-stage output, but **no unified hypothesis
trajectory**. If you read an analysis.md, you can see:

- Stage 0 emitted a `cancer_hint` + a `reasoning_trace` (which
  Stage-0 rule fired).
- Stage 1 emitted a `candidate_trace` (top-k cancer types with
  scores).
- Stage 2 emitted a `purity` dict (final estimate + components).
- Stage 3 emitted a `decomp_results` list (per-candidate template
  fits).

What you CAN'T see from the final report:

- **Which hypotheses were considered then rejected** at each stage,
  and why. Stage 1 might consider 8 cohorts and narrow to top-5, but
  the three rejected ones disappear from the trace.
- **Whether a given stage contradicted or corroborated** the prior
  stage's preferred hypothesis. Stage 0 says "tumor-consistent";
  Stage 1 picks LUAD; Stage 3 decomposition finds low tumor-
  fraction under the LUAD template. Is that a contradiction or a
  low-purity artifact? The reader has to splice the story together.
- **What the "second-best" hypothesis was at any stage** — the
  runner-up hypothesis is lost beyond Stage 1's top-k.
- **The ordering of constraints**. Stage 2 refines purity under a
  fixed cancer-type hypothesis; it doesn't know it's exploring
  multiple hypotheses still alive from Stage 1.

The effect: contested calls (e.g. a SARC sample being called THYM
with lineage concordance 0) fall through because no stage
explicitly asks "is the top hypothesis consistent with everything
we've accumulated so far?" The #169 confidence tier is a band-aid
— the architectural fix is per-hypothesis bookkeeping.

### What the ledger is actually tracking

Reasoning about a sample is never just "which cancer type?". At every
stage we're jointly refining several orthogonal hypothesis spaces:

1. **Cancer-type** — which TCGA cohort (and, when the cohort is a
   Type-B mixture, which subtype: SARC_RMS vs SARC_LMS vs …).
2. **Site / template** — primary vs metastatic vs pure population vs
   heme (solid_primary / met_bone / met_liver / met_lymph_node /
   pure_population / heme_marrow / …). A cancer-type hypothesis
   can exist under multiple site hypotheses; decomposition chooses.
3. **Purity** — starts as an unbounded [0, 1] range at Stage 0,
   narrows to a CI at Stage 2, further sharpened by Stage-3
   decomposition anchoring.
4. **Sample composition** — the fraction dict over compartments
   `{tumor, fibroblast, T_cell, B_cell, macrophage, endothelial,
   matched_normal_<tissue>, host_specific_<...>}`. Stage 3 fits
   the NNLS; earlier stages don't decompose but do note tissue
   contamination hints (top-HPA at Stage 0, site-indeterminacy at
   Stage 3).
5. **Activation state** — therapy-response axis scores (AR active /
   suppressed, IFN active, hypoxia active, EMT, MAPK, glycolysis,
   DDR). These aren't standalone hypotheses; they annotate the
   surviving cancer-type + site + purity hypothesis and explain
   non-cancer components of the expression signal.

These are **coupled**. Knowing the cancer type tells you which
matched-normal template to fit. Fitting the template tells you
whether the purity estimate is consistent. Knowing the site
(primary vs met) constrains which matched-normal compartments are
plausible. Knowing the activation state explains observed surface-
target TPMs that aren't actually tumor-specific (IFN-driven HLA-F
up). Rejecting a cancer-type hypothesis can release a composition
hypothesis (the matched-normal choice is no longer forced).

### Proposed structure

Five hypothesis dataclasses sharing one `HypothesisLedger` so their
joint trajectories can be cross-referenced at render time:

```python
@dataclass
class CancerTypeHypothesis:
    id: str                     # "SARC" or "SARC:RMS" for subtype
    family: str | None          # "MESENCHYMAL", "SQUAMOUS", ...
    score_by_stage: dict[str, float]
    evidence_by_stage: dict[str, dict]
    alive: bool = True
    rejection: tuple[str, str] | None = None  # (stage, reason)

@dataclass
class SiteHypothesis:
    id: str                     # "solid_primary", "met_liver", ...
    under_cancer_type: str      # links to a CancerTypeHypothesis.id
    template_name: str
    score_by_stage: dict[str, float]
    evidence_by_stage: dict[str, dict]
    alive: bool = True
    rejection: tuple[str, str] | None = None

@dataclass
class PurityHypothesis:
    under_cancer_type: str      # per-candidate purity
    estimate: float
    lower: float                # CI bounds
    upper: float
    source_by_stage: dict[str, str]  # stage -> "signature" | "lineage" | "decomposed" | ...
    confidence_tier: str        # "degenerate" | "low" | "moderate" | "high"

@dataclass
class CompositionHypothesis:
    under_cancer_type: str
    under_site: str
    fractions: dict[str, float] # {tumor, fibroblast, T_cell, ...}
    score: float
    warnings: list[str]
    alive: bool = True
    rejection: tuple[str, str] | None = None

@dataclass
class ActivationSignature:
    axis: str                   # "AR", "IFN", "hypoxia", "EMT", ...
    state: str                  # "active", "suppressed", "intact"
    up_fold: float
    down_fold: float | None
    explains: list[str]         # genes this axis explains away
    therapy_context: str | None # "post-ADT", "IFNg-driven inflation", ...

@dataclass
class StageTransition:
    stage: str
    space: str                  # which hypothesis space: "cancer" | "site" | "purity" | ...
    action: str                 # "introduce" | "score" | "promote" | "reject"
    touched_ids: list[str]
    summary: str                # human-readable
    evidence: dict              # the numbers that drove the action

class HypothesisLedger:
    cancer: dict[str, CancerTypeHypothesis]
    sites: dict[str, SiteHypothesis]
    purities: dict[str, PurityHypothesis]
    compositions: dict[str, CompositionHypothesis]
    activations: list[ActivationSignature]
    transitions: list[StageTransition]

    # Methods for each space follow a common shape:
    # .introduce_<kind>(...), .score_<kind>(...), .reject_<kind>(...)
    # .narrate() renders the full joint trail.
```

### How each stage uses the ledger (broader scope)

- **Stage 0 — coarsest "is this cancer, and what family".**
  Introduces cancer-FAMILY hypotheses (epithelial / mesenchymal /
  lymphoid / melanocytic / glial / germ-cell). Writes a first
  purity hypothesis that's nothing more than a Bernoulli-style
  prior: `% cancer > 0` with a coarse range (e.g. [0.1, 0.9] for
  tumor-consistent, [0, 0.3] for healthy-dominant). Records top
  HPA-normal tissue matches as preliminary composition hints (not
  yet decomposed).
- **Stage 1 — TCGA-cohort level + coarse site.** Promotes
  surviving family hypotheses to TCGA-cohort codes; introduces
  preliminary site hypotheses (solid_primary default, plus met_*
  when Stage-0 top HPA is from a typical met site). Refines purity
  per-cohort hypothesis. Rejects cohorts whose signature / lineage
  / concordance fall out of the top-k. Writes per-cohort purity
  hypotheses with CIs.
- **Stage 2 — reconcile cohort with purity.** For each alive
  cohort × site pair, combine three orthogonal purity estimators
  (signature / ESTIMATE / lineage). Hypotheses whose three
  estimates diverge materially get downgraded ("lineage purity
  contradicts signature" → `low` confidence). Purity CIs tighten
  when the three agree.
- **Stage 3 — decomposition: composition becomes explicit.** For
  each alive (cohort, site) pair, NNLS-fit yields a
  `CompositionHypothesis` with per-compartment fractions. This is
  where **site refinement actually happens** — if the solid_primary
  template fits better than met_liver, the met_liver site is
  rejected; if the matched_normal_colon compartment dominates, the
  tumor-fraction falls. Per-subtype fits for Type-B mixture
  cohorts (SARC) happen here as well.
- **Stage 4 — activation annotations.** Score therapy-response
  axes under the winning (cancer, site, composition). These
  annotate rather than select — but they EXPLAIN composition
  features ("MHC-I 5× cohort is IFN-driven, not tumor-specific").
  Rejections happen only when an activation signature contradicts
  the call (e.g. AR-axis suppressed contradicts a PRAD hypothesis
  in a treatment-naive context; but in an ADT-treated patient it
  confirms).
- **Stage 5 — per-gene tumor-attributed TPM** under the surviving
  joint hypothesis. The ledger isn't adding new hypotheses here;
  the per-gene attributions derive from the composition.
- **Stage 6 — render.** Read the ledger and narrate.

### Ladder of refinement

| Stage | Cancer-type space | Purity | Site | Composition | Activation |
|---|---|---|---|---|---|
| 0 | ~6 families (binary cancer-present + coarse family) | [0.1, 0.9] or [0, 0.3] | unknown | top-HPA-tissue hints only | deferred |
| 1 | ≤ 8 TCGA cohorts (top-k + family members) | per-cohort CI, 20–40 pp wide | primary default + met candidates when relevant | unchanged | deferred |
| 2 | ≤ 4 cohorts (after purity-consistency check) | tightened CI, 10–25 pp | unchanged | unchanged | deferred |
| 3 | ≤ 2 cohort × site pairs | decomposed-anchor CI, 5–15 pp | rejected templates removed | per-compartment fractions (tumor + TME + matched_normal) | deferred |
| 4 | 1 winning hypothesis | unchanged | unchanged | unchanged | all axes annotated |
| 5 | 1 | unchanged | unchanged | per-gene attribution | unchanged |
| 6 | render | render | render | render | render |

### The markdown output — joint trail

Analysis.md gains a **Reasoning trail** section that walks through
each stage's movement in every hypothesis space, then a **Final
synthesis** that reconciles them:

```markdown
## Reasoning trail

### Stage 0 — coarsest read
- Cancer-family hypotheses introduced: epithelial, mesenchymal,
  lymphoid, melanocytic, glial, germ-cell, unknown-normal.
- Epithelial kept (top HPA nTPM_colon ρ 0.79; top TCGA FPKM_COAD
  ρ 0.77 — clean epithelial pattern).
- Rejected: mesenchymal (smooth_muscle ρ 0.31 < 0.55), lymphoid
  (lymph_node ρ 0.28), unknown-normal (aggregate-tumor-evidence
  fired, score 3.72).
- First purity range: [0.10, 0.85] — cancer-present likely, but
  % unquantified at this stage.
- First composition hints: top-HPA-tissue colon (0.79),
  smooth_muscle (0.31, plausible muscularis stromal contribution).

### Stage 1 — TCGA cohort + preliminary site
- Cancer-type hypotheses introduced (under epithelial): COAD,
  READ, STAD, PAAD, LUAD, BLCA, KIRC, KIRP.
- Alive after Stage 1: COAD, READ, STAD (top-3 by geomean).
- Rejected: KIRC/KIRP (signature < 0.3 vs CRC ≥ 0.7); BLCA (wrong
  family — urothelial divergent); PAAD / LUAD (lineage
  concordance < 0.4).
- Site hypotheses introduced per alive cohort:
    COAD / solid_primary — default
    COAD / met_liver — liver ρ 0.35 in HPA; follow up
    COAD / met_peritoneal — plausible given muscle hint
    READ / solid_primary — default
    STAD / solid_primary — default
- Purity per (cohort, site): 36% / 34% / 45% — wide CI estimates.

### Stage 2 — purity reconciliation
- Three orthogonal estimators run per alive hypothesis:
    COAD / solid_primary: sig 0.30 ± 0.12, lin 0.50 ± 0.18, est
      0.22 → overall 36% (CI 17–60%, moderate)
    READ / solid_primary: overall 34% (CI 16–58%, moderate)
    STAD / solid_primary: overall 45% (CI 22–68%, low — span 46pp)
- None rejected; all three estimators agree within a CI.

### Stage 3 — decomposition + site refinement
- NNLS-fit under each alive (cohort, site):
    COAD / solid_primary: score 0.145 — fractions {tumor 0.36,
      fibroblast 0.18, T_cell 0.04, matched_normal_colon 0.22,
      endothelial 0.08, other 0.12}. ✓ top.
    READ / solid_primary: score 0.134 — similar composition.
    COAD / met_liver: score 0.088 — REJECTED.
      matched_normal_liver absorbs 62% of signal; primary-site
      support dominates.
    COAD / met_peritoneal: score 0.071 — REJECTED.
      Peritoneal-mesothelial compartment fits poorly.
    STAD / solid_primary: score 0.120 — REJECTED.
      Lineage concordance dropped to 0.31 on this sample
      (MUC6 / CDX2 subtype-specific pattern not matching);
      hypothesis no longer supported.
- Subtype narrative: cohort COAD is Type-A convergent; no subtype
  refinement needed. (For a Type-B mixture cohort like SARC the
  fit would run per-subtype and promote SARC:RMS / SARC:LMS / …
  based on max per-subtype lineage match.)
- Composition pinned: tumor 36%, matched_normal_colon 22%,
  fibroblast 18%, endothelial 8%, T-cell 4%, remainder 12%.
- Purity tightened by decomposition: overall 36% (CI 28–44%,
  moderate — narrows from 17–60% after NNLS anchor).

### Stage 4 — activation state
- IFN axis: **active** (up-panel 5.1× cohort). Explains the
  HLA-A/B/F, HLA-E, IFIT1/3, OAS1 TPM elevation that was initially
  flagged as tumor-specific — not a cancer-cell-specific signal,
  just IFN-driven inflation.
- MAPK / EGFR axis: **active** (3.0× cohort). Consistent with
  COAD receptor-signaling activity; no therapy context inferred.
- Hypoxia (CA9): 2.3× cohort — moderate; consistent with tumor-
  core presence.
- AR, ER, HER2, EMT: intact (within 0.8–1.2× cohort).
- (In a CRPC PRAD sample this section would flag AR-axis
  suppressed as evidence of ADT exposure; in a HER2+ BRCA it
  would confirm ERBB2 amp drives the signature; and so on.)

### Stage 5 — per-gene tumor-attributed TPM
Under the pinned (COAD, solid_primary, purity 0.36, composition
above) hypothesis, per-gene attribution yields 9-point TPM ranges
for every target. The Stage-4 activation annotations feed into
per-target confidence — IFN-active samples auto-downgrade MHC / ISG
surface targets to "not tumor-specific".

## Final synthesis

**Cancer call:** COAD (Colon Adenocarcinoma), solid primary.
**Purity:** 36% (CI 28–44%, moderate).
**Sample composition:** tumor 36%, matched-normal colon 22%,
fibroblast 18%, endothelial 8%, T-cell 4%, other / unassigned 12%.
**Activation state:** IFN active (5.1× cohort), MAPK/EGFR active
(3.0×), moderate hypoxia (2.3×); AR / ER / HER2 / EMT intact.
**Confidence:** moderate — runner-up READ within 8% on geomean;
no late-stage rejection casts doubt on COAD specifically; IFN
activity explains the HLA-family uplift as non-tumor-specific.
```

### What the trail lets a reader audit

At any row, you can ask "why is this hypothesis still alive /
rejected?" and the ledger has the specific evidence. For an
ambiguous sarcoma sample that historically miscalled as THYM, the
trail would have shown THYM entering at Stage 1 with concordance
= 0, and being rejected at Stage 2 with "lineage concordance 0
below minimum 0.2" — the banner would carry that reason rather
than emitting a clean THYM call.

More importantly, **cross-space reconciliation is visible**. When
the HLA-A observation is explained by IFN axis (Stage 4) rather
than tumor-specific signature (Stage 1), the reader sees both
pieces of the chain and can judge whether the HLA-A target is
actionable.

### Implementation phases

**Phase 1 — ledger skeleton + cancer-type + site spaces.**
Introduce `HypothesisLedger` with the five dataclasses.
Retro-populate it from existing `analysis["candidate_trace"]` and
`analysis["decomp_results"]` — no behavioral change, just bookkeep
what the stages already decide. Add a "Reasoning trail" section to
analysis.md that renders the ledger. Covers cancer-type + site +
preliminary purity / composition.

**Phase 2 — joint refinement + composition + activation.** Stage
2's purity estimator runs under every alive cohort hypothesis (not
just the top one) and writes per-hypothesis purity records. Stage
3 writes composition hypotheses with fraction dicts. Stage 4 emits
`ActivationSignature` records that annotate (not replace) the
surviving cancer-type hypothesis. At this point the ledger is the
source of truth; the old `analysis[...]` keys become thin views.

**Phase 3 — late-stage rejection propagates back.** When Stage 3
rejects a hypothesis (e.g. STAD on a COAD sample for lineage-
concordance collapse), re-score the surviving set and re-rank; the
re-rank can demote a hypothesis that previously "won" Stage 1 on a
thin margin. Produces a proper constraint-satisfaction flavour:
every stage's evidence can override earlier preliminary choices as
long as the supporting evidence is surfaced.

**Phase 4 — heterogeneity-class + mixture-subtype integration
(closes this doc's Part 1).** Type-B mixture cohorts (SARC) expand
into per-subtype cancer-type hypotheses. Ledger carries the
subtype-level hypothesis through all stages; the final narrative
names the subtype.

### Acceptance

- analysis.md includes a **Reasoning trail** section structured by
  stage, covering all five hypothesis spaces (cancer / site /
  purity / composition / activation).
- A historical sarcoma-miscalled-as-THYM case would have shown the
  THYM hypothesis entering at Stage 1 with concordance = 0 and
  being rejected at Stage 2 or Stage 3, with the rejection reason
  carried forward to the banner.
- A CRC sample with active IFN axis surfaces the activation
  annotation that explains HLA-family uplift as non-tumor-specific,
  so the brief's therapy recommendations don't pitch HLA-A as an
  actionable target.
- Every stage records the evidence behind each score action in a
  round-trippable form (the ledger can be serialised, replayed,
  and compared across runs).
- For a Type-B mixture cohort (SARC), the final call names the
  matched subtype ("SARC, rhabdomyosarcoma-consistent") rather
  than just "SARC".

---

## Combined roll-out

The two parts interleave. The simplest order that keeps each step
independently shippable:

1. **Part 1 Phase 1** — `heterogeneity_class` registry flag on
   every cohort. Pure data annotation; no scoring change. Sets up
   the discriminator Phase 4 will use.
2. **Part 2 Phase 1** — ledger skeleton + retro-populated from
   existing `analysis[...]` keys + Reasoning-trail section in
   analysis.md. Highest-leverage-for-transparency. No behavior
   change yet; just makes the hidden reasoning visible.
3. **Part 2 Phase 2** — ledger becomes source of truth; per-
   hypothesis purity + composition + activation records; stages
   write ledger-first. This is where the Reasoning trail actually
   starts including the richer information (multi-hypothesis
   purity, composition fractions, activation annotations).
4. **Part 1 Phase 2** — mixture-aware lineage scoring. SARC
   expands into per-subtype hypotheses. Requires Part 2 Phase 2
   (the ledger has to carry subtype hypotheses as first-class
   members).
5. **Part 2 Phase 3** — late-stage rejection propagates back.
   Biggest refactor; produces full constraint-satisfaction
   behaviour. Only now can a Stage-3 rejection revive a
   previously-rejected runner-up.

### Dependency between the two parts

Cohort heterogeneity (Part 1) and the hypothesis ledger (Part 2)
are independent but mutually useful:

- Phase 1 of Part 1 (registry flag) can land without the ledger.
- Phase 2 of Part 1 (mixture-aware lineage) benefits a lot from the
  ledger (each subtype becomes a first-class hypothesis in Stage 1's
  alive set) but doesn't require it.
- The ledger benefits from Part 1 because `mixture` cohorts surface
  multiple subtype hypotheses — more per-stage rejection / promotion
  drama to narrate.

Suggested roll-out order:

1. Part 1 Phase 1 — registry annotation (quick; mostly data + a
   small property on `CANCER_TYPES`).
2. Part 2 Phase 1 — ledger drop-in + analysis.md Reasoning-trail
   section (medium; touches every stage but optional).
3. Part 1 Phase 2 — mixture-aware lineage scoring (bigger; requires
   per-subtype panel curation for SARC first, then generalisation).
4. Part 2 Phase 2 — stages become ledger-first (bigger refactor).
5. Part 2 Phase 3 — late-stage rejection propagates back (the
   full hypothesis-space reasoning; largest change).

Each phase is independently shippable and valuable on its own.
