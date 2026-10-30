# OncoRef and Tsarina compatibility

PirlyGenes 6.0.20 supports the tested OncoRef range **1.23.1–1.24.0** and
Tsarina **1.36.0 or newer**. The bounded owner range lets current consumers
resolve a mutually compatible release while retaining an explicit upper bound.
Both owner endpoints preserve the historical 624-gene CTA panel and expression
data identities used by these compatibility APIs.

OncoRef retains expression data version 5.23.26 and source-matrix version
5.22.14. PirlyGenes retains data version 6.0.13. Existing materialized files keep
their original builder version; this package upgrade does not relabel or
regenerate them.

## One historical policy across the compatibility APIs

`CTA_gene_ids()` and its related helpers continue to use `oncoref.cta`.
`CTA_evidence()` uses the same historical HPA table and joins only Tsarina's
independent `ms_*` columns by exact gene ID. Ambiguous or missing join identities
raise an error. Unknown observations remain missing, and `filtered` remains an
exact alias of `passes_filters`, including nulls.

All ID, symbol and DataFrame partition views derive from one historical
partition. They retain the existing public result containers. The expressed and
never-expressed members come directly from OncoRef; the background is every
other coding gene in the requested sequence annotation. No local membership
snapshot or biological decision rules are copied.

Tsarina 1.36 deliberately changed its defaults to OncoRef's current
`cta_selection` policy. Its 558 retained loci are a different population from
the historical 624-gene panel. Mixing the historical direct helper with the
current Tsarina partition caused #664. Adopting the current selection as a
PirlyGenes default remains the separate policy decision in #655.

## Current owner references and evidence

OncoRef 1.23.1 adds human Ensembl 115 transcript/protein mappings bound to
verified sequence assets. OncoRef 1.24.0 adds clinical-evidence explanations to
its separate research selection, with exact peptide, HLA and product limits.
That selection retains 522 protein groups and 558 member loci. Its regenerated
receipt now uses the full Ensembl 115 sequence-reference identity that 1.23.1
already exposed through the common antigen API. Preserve the new reference,
policy and source-snapshot IDs; the older gene-only reference is a distinct
identity. See the [owner's release notes](https://github.com/pirl-unc/oncoref/releases/tag/v1.24.0).

## Joint installation and release checks

The current-stack CI job pins the latest Vaxrank (including its `cta` extra),
Topiary, Hitlist, Isovar and Trufflepig releases. It upgrades shared dependencies
within those consumers' declared bounds and records installed versions. For
example, current consumers exclude Varcode 11 until their candidate traversal
migrations land; the resolver must respect that API boundary.

Validate both supported owner endpoints with current Tsarina, focused CTA/MS
and peptide-exclusion tests, full serial tests, lint and `pip check`. CI also
checks installed matrix reuse and historical/current policy separation. Publish
only after exact-head CI and the normal clean-main deployment pass; verify
public dependency metadata, wheel/sdist content and the release tag.

This update consolidates #658, #659, #663 and #664.
