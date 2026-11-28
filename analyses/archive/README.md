# Archived docs (historical snapshots)

These are **point-in-time** audit / planning documents whose work has since
shipped. They predate the Phase-C cancer-type renames (e.g. `EWS`→`SARC_EWS`,
`OS`→`SARC_OS`, `PANNET`→`NET_PANCREAS`, `MID_NET`→`NET_MIDGUT`,
`LUNG_NET_LCNEC`→`NEC_LUNG_LARGECELL`, `NBL_MYCN_*`→`NBL_MYCN*`,
`LAML_ELN_*`→`LAML_ELN*`, `HNSC_HPV_pos`→`HNSC_HPVpos`) and the later
registry/source growth, so their counts and code names are **stale** — kept only
as a record of how the current state was reached.

For **current** values, query the live API (the single source of truth):

```python
from pirlygenes.gene_sets_cancer import cancer_type_registry, CTA_gene_names
from pirlygenes.expression import (
    available_cancer_expression_references,   # packaged per-cohort references
    cancer_expression_reference_status,       # per-code source/built/candidate status
    cancer_expression_source_candidates,      # acquisition targets for gaps
)

cancer_type_registry()                  # registry codes (canonical)
available_cancer_expression_references() # what expression data is packaged
cancer_expression_reference_status("SARC_OS")  # per-code status incl. candidate gaps
```

## Contents
- `expression-sources-per-code-2026-05.md` — per-code expression-source manifest
  (superseded by `cancer_expression_reference_status()`).
- `expression-data-audit-2026-05.md` — bucket-level expression-data audit /
  work-queue for the refresh project (implemented).
- `cancer-type-ontology-audit-2026-05.md` — Phase-C registry ontology audit +
  restructure plan (implemented).
