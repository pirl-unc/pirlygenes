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
- pfo004 (real rhabdomyosarcoma-ish sarcoma) classifies as `SARC`
  with a subtype hypothesis that names the RMS panel.
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

The effect: contested calls (pfo004 → THYM with concordance 0) fall
through because no stage explicitly asks "is the top hypothesis
consistent with everything we've accumulated so far?" The #169
confidence tier is a band-aid — the architectural fix is
per-hypothesis bookkeeping.

### Proposed structure: `HypothesisLedger`

One dataclass-backed object that threads through all stages:

```python
@dataclass
class Hypothesis:
    id: str                          # stable: "COAD/solid_primary"
    label: str                       # human: "Colon Adenocarcinoma / primary site"
    introduced_by: str               # stage that introduced it
    score_by_stage: dict[str, float] # stage -> confidence score
    evidence_by_stage: dict[str, dict] # stage -> per-stage signals
    alive: bool = True
    rejection_stage: str | None = None
    rejection_reason: str | None = None

@dataclass
class StageTransition:
    stage: str
    alive_before: list[str]          # hypothesis ids
    alive_after: list[str]
    introduced: list[str]
    rejected: list[tuple[str, str]]  # (id, reason)
    summary: str                     # human-readable stage-outcome sentence

class HypothesisLedger:
    hypotheses: dict[str, Hypothesis]
    transitions: list[StageTransition]

    def introduce(self, h: Hypothesis) -> None: ...
    def score(self, hid: str, stage: str, score: float, evidence: dict) -> None: ...
    def reject(self, hid: str, stage: str, reason: str) -> None: ...
    def alive(self) -> list[Hypothesis]: ...
    def narrate(self) -> str: ...      # renders the reasoning trail
```

### How each stage uses the ledger

- **Stage 0** introduces tissue-composition hypotheses (e.g.
  `"tumor-consistent"`, `"lymphoid-ambiguous"`) and scores them via
  the rule runner. Every rule's outcome is a score entry on the
  winning hypothesis.
- **Stage 1** introduces cancer-type hypotheses (`"COAD"`, `"READ"`,
  `"STAD"`, ...). Candidates not in the top-k are introduced and
  immediately rejected with the per-candidate geomean / family-
  factor reason.
- **Stage 2** consumes the alive cancer-type hypotheses, computes
  a purity under each, and records the score. Hypotheses that fall
  outside a purity-consistent window can be rejected here ("purity
  estimated < 5% under this hypothesis — no tumor-fraction support").
- **Stage 3** runs decomposition per alive (cancer_type, template)
  combination. Decomposition-fit failures (no TME compartments,
  tumor-fraction collapses to 0) become rejection events.
- **Stage 4+5+6** read the final alive list and the full transition
  history, and render the narration.

### The markdown output

Analysis.md gains a **Reasoning trail** section:

```markdown
## Reasoning trail

Stage 0 (tissue composition) considered 3 broad possibilities:
  - tumor-consistent — chosen (aggregate-tumor-evidence fired,
    score 3.72)
  - possibly-tumor — rejected (no structural-ambiguity pattern)
  - healthy-dominant — rejected (proliferation panel 4.1 log2-TPM > 3.5)

Stage 1 (cancer type) introduced 8 hypotheses:
  Alive after: COAD, READ, STAD (3 candidates; all CRC-family)
  Rejected: KIRC, KIRP (signature < 0.3 vs CRC ≥ 0.7)
           BLCA (wrong family)
           PAAD, LUAD (lineage concordance < 0.4)

Stage 2 (purity) refined under the 3 alive hypotheses:
  COAD / solid_primary: 36% (CI 17%–60%, moderate)
  READ / solid_primary: 34% (CI 16%–58%, moderate)
  STAD / solid_primary: 45% (CI 22%–68%, wide)

Stage 3 (decomposition) ranked by template fit:
  COAD / solid_primary: score 0.145 ✓ top
  READ / solid_primary: score 0.134
  STAD / solid_primary: score 0.120 — lineage concordance
    dropped to 0.3 on this sample; rejected

Final call: COAD (Colon Adenocarcinoma / solid primary)
  Confidence: moderate — runner-up READ within 8% on geomean
```

This tells the reader a story they can audit: **at each stage, what
was on the table, what survived, what was rejected and why**.

### Implementation phases

**Phase 1** — introduce the ledger as a parallel structure the
stages optionally write to. No behavioral change; analysis.md gets
a new section. Low-risk drop-in.

**Phase 2** — stages become ledger-first. Each stage's public
function takes a ledger argument and emits events; the `analysis`
dict becomes a thin view over the ledger for back-compat.

**Phase 3** — stages use the ledger to constrain reasoning. Stage
2's purity estimator refines purity under all alive hypotheses, not
just the top one. Stage 3's decomposition fits every alive
hypothesis rather than just top-k. Late-stage rejection events
trigger a re-ranking propagated back through the ledger's score
history.

### Acceptance

- analysis.md includes a Reasoning trail section that lists, per
  stage, `{introduced, alive_after, rejected_with_reason}`.
- The pfo004 → THYM miscall case (historical) would have shown the
  THYM hypothesis entering at Stage 1 and being rejected at Stage 2
  or Stage 3 for concordance = 0, with the ledger carrying the
  rejection reason forward to the final banner.
- Every stage that scores a hypothesis records the evidence behind
  the score in a round-trippable form.

---

## Dependency between the two parts

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
