# Gene Canonicalization Contract

Pirlygenes has two related identifier spaces. They are intentionally separate.

## Canonical Gene Space

The canonical gene space is the row identity space for gene-level expression
tables.

- The key is `Ensembl_Gene_ID`.
- Values are unversioned human Ensembl stable gene IDs such as
  `ENSG00000141510`.
- The authority release is `pirlygenes.CANONICAL_ENSEMBL_RELEASE`, currently
  Ensembl release 112, because the packaged references claim to have been
  harmonized there.
- `Symbol` is display metadata only. It is never part of a join key.
- A canonicalized expression table has at most one row per
  `(Ensembl_Gene_ID, context...)`, where context is the caller's declared
  cohort/source/normalization grain.
- If multiple source rows map to the same canonical gene inside one context,
  linear expression values are summed with `min_count=1`; all-missing stays
  missing.
- Version suffixes (`.17`), raw source IDs, IDs outside the authority release,
  and synthetic proteoform keys are not valid gene-space keys.
- Source rows that cannot be mapped into the authority release are quarantined
  by default by `canonicalize_gene_table(..., drop_unmapped=True)`. Keeping them
  requires an explicit opt-out and should be followed by
  `canonical_gene_space_report(...)`.

The runtime API lives in `pirlygenes.gene_canonicalization`:

- `canonical_gene_id(identifier, source_version=None, symbol_hint=None)` maps
  source identifiers into the canonical gene space. Pass `symbol_hint` when
  resolving a table row that has both an Ensembl ID and a symbol; this rescues
  retired IDs whose stable-ID history is incomplete but whose symbol maps
  uniquely into release 112.
- `canonicalize_gene_table(...)` rewrites and collapses a table into that space.
- `validate_canonical_gene_table(...)` raises when a table violates the contract.
- `canonical_gene_space_report(...)` returns counts and examples without raising.
- `canonical_gene_id_map()` exposes the versioned bundled alias map.

`source_version` is part of the API even when the current bundled runtime map does
not need it for every lookup. Callers should pass it, because release-specific
stable-ID history is the natural authority for retire/rename/merge/split events.

## Reduced Proteoform Space

The reduced proteoform space is layered on top of canonical genes for analyses
that need a protein-abundance proxy.

- If one canonical gene maps uniquely to one protein, the proteoform key remains
  the canonical ENSG.
- If multiple genes encode a byte-identical protein, the proteoform key is a
  synthetic summed key such as `CTAG1A/B`.
- Synthetic proteoform keys are valid only in a declared proteoform-space table,
  never in an ordinary gene-space table.
- Collapses happen in linear expression space before log or percentile transforms.

The public helpers are:

- `canonical_proteoform_id(ensg, kind="protein")`
- `canonical_proteoform_id_map(kind="protein")`

Use `kind="cdna"` for the read-recovery cDNA-identical space and
`kind="protein"` for byte-identical protein abundance.

## Invariant Checks

Before joining expression data across cohorts or sources, run:

```python
from pirlygenes.gene_canonicalization import validate_canonical_gene_table

validate_canonical_gene_table(
    df,
    context_cols=["cancer_code", "source_cohort", "normalization"],
)
```

For diagnostics without raising:

```python
from pirlygenes.gene_canonicalization import canonical_gene_space_report

report = canonical_gene_space_report(df, context_cols=["cancer_code"])
print(report)
```

Set `forbid_symbol_fallback_ids=True` when raw `ENSG...` symbols should be
treated as a hard failure instead of a reported warning.
