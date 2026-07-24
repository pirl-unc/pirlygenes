# Documentation map

`pirlygenes` is the curated data and reference-expression layer for cancer RNA
analysis. Start with the current public contracts below; use the audit,
operations, and historical documents only when you need their additional
detail.

## Use the package

- [Available gene sets and lists](gene-sets.md) — public panels, expression
  references, identifiers, and the APIs that expose them.
- [Gene canonicalization contract](gene-canonicalization.md) — stable gene keys,
  symbol metadata, collision handling, and proteoform reductions.
- [Cancer-type classification ontology](cancer-classification-ontology.md) —
  how compartment, lineage, family, subtype, and marker role fit together.
- [Embedding gene selection](embedding-gene-selection.md) — why the reference
  embedding uses bottleneck and pan-reference gene sets.
- [CTA curation](cta-curation.md) — ownership, inclusion logic, evidence, and
  generated curation figures.

## Understand expression data

- [Reference-expression delegation parity](reference-expression-parity.md) —
  the current pirlygenes/oncoref ownership boundary, compatibility transforms,
  and validation strategy.
- [Normalization and mapping audit](normalization-and-mapping-audit.md) — the
  conclusions first, followed by unit conversion, identifier mapping, and
  multi-source evidence.
- [recount3 integration](recount3-integration.md) — why selected GEO sources use
  recount3, what it covers, and how oncoref owns the rebuild path.
- [Expression-data refresh plan](expression-data-refresh-plan.md) — historical
  implementation log and the resulting current ownership model.

## Operate and audit releases

- [Data-bundle deploy checklist](phase-c-deploy-checklist.md) — release ordering,
  validation, and publication.
- [Reference-expression full audit report](reference-expression-delegation-557.md)
  and [per-code CSV](reference-expression-delegation-557.csv) — generated parity
  evidence for the pinned oncoref release.
- [Figure-set audit](figure-audit.md) — historical inventory and consolidation
  recommendations for the retired pirlygenes sample runner.

## Historical sample-analysis design

Per-sample analysis now belongs to
[`trufflepig`](https://github.com/pirl-unc/trufflepig). These documents remain
for migration context and for the pirlygenes-owned data contracts they mention:

- [Legacy analyze command](analyze-command.md)
- [Analyze API boundary](analyze-api.md)
- [Reasoning pipeline](reasoning-pipeline.md)
- [Step-0 reasoning](step0-reasoning.md)
- [Lineage-aware decomposition proposal](lineage-aware-decomposition.md)

Within each document, read the summary or current-status section first. Later
sections contain algorithms, inventories, historical milestones, or operational
procedures.
