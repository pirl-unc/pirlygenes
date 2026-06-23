# Gene Canonicalization Contract

Pirlygenes has two related identifier conventions. They are intentionally
separate.

## Canonical Gene Tables

Canonical gene tables use one row identity convention for gene-level expression
data.

- The key is `Ensembl_Gene_ID`.
- Values are unversioned human Ensembl stable gene IDs such as
  `ENSG00000141510`.
- The authority release is `pirlygenes.CANONICAL_ENSEMBL_RELEASE`, currently
  Ensembl release 112, because the packaged references claim to have been
  harmonized there. It is loaded from a bundled offline snapshot
  (`pirlygenes/data/canonical-gene-reference.csv.gz`: id, symbol, contig,
  biotype), so canonicalization does not depend on which pyensembl releases are
  installed at runtime. `canonical_authority_release()` reports the release
  actually in effect.
- `Symbol` is display metadata only. It is never part of a join key.
- A canonicalized expression table has at most one row per
  `(Ensembl_Gene_ID, context...)`, where context is the caller's declared
  cohort/source/normalization grain.
- If multiple source rows map to the same canonical gene inside one context,
  linear expression values are summed with `min_count=1`; all-missing stays
  missing.
- Version suffixes (`.17`), non-Ensembl / malformed ids, and synthetic
  proteoform keys are not valid gene-table keys. A well-formed unversioned ENSG
  *is* valid even when absent from the authority release (see keep-as-self
  below); the authority governs symbol rescue and merge targets, not row
  admissibility.
- Keep-as-self: every well-formed ENSG resolves to a stable canonical id rather
  than being dropped. Genes outside the authority release (alt-haplotype,
  multi-copy ncRNA, retired-without-successor) survive with the ENSG itself as
  their display symbol.
- Multiple representations of one sequence collapse to a single id. Byte-
  identical-cDNA loci (alt-haplotype copies and multi-copy ncRNA families like
  `U6` / `Y_RNA`) plus the curated alt-haplotype/retired aliases form one
  equivalence class (union-find over
  `scripts/generate_sequence_identical_gene_groups.py` output and the bundled
  alias table); members sum their TPM onto one representative.

The runtime API lives in `pirlygenes.gene_canonicalization`:

- `canonical_gene_id(identifier, source_version=None, symbol_hint=None)` maps
  source identifiers into the canonical gene table convention. Pass `symbol_hint` when
  resolving a table row that has both an Ensembl ID and a symbol; this rescues
  retired IDs whose stable-ID history is incomplete but whose symbol maps
  uniquely into release 112.
- `canonicalize_gene_table(...)` rewrites and collapses a table into that
  convention.
- `validate_canonical_gene_table(...)` raises when a table violates the contract.
- `gene_table_validation_report(...)` returns counts and examples without raising.
- `canonical_gene_id_map()` exposes the versioned bundled static maps — the
  alt-haplotype/retired alias table and the sequence-identity groups, resolved
  to the same terminal IDs as `canonical_gene_id()`. If both bundled sources
  mention a source ID, their `mapping_source` values are combined.
- `canonical_gene_biotype(ensg)` returns the authority biotype offline, for a
  `protein_coding`-only filter without a live pyensembl install.

`source_version` is part of the API even when the current bundled runtime map does
not need it for every lookup. Callers should pass it, because release-specific
stable-ID history is the natural authority for retire/rename/merge/split events.

## Reduced Proteoform Tables

Reduced proteoform tables are layered on top of canonical genes for analyses
that need a protein-abundance proxy.

- If one canonical gene maps uniquely to one protein, the proteoform key remains
  the canonical ENSG.
- If multiple genes encode a byte-identical protein, the proteoform key is a
  synthetic summed key such as `CTAG1A/B`.
- Synthetic proteoform keys are valid only in a declared proteoform table,
  never in an ordinary gene table.
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
from pirlygenes.gene_canonicalization import gene_table_validation_report

report = gene_table_validation_report(df, context_cols=["cancer_code"])
print(report)
```

Set `forbid_symbol_fallback_ids=True` when raw `ENSG...` symbols should be
treated as a hard failure instead of a reported warning.
