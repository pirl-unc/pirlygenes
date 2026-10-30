# OncoRef 1.23 compatibility

Update the exact owner pin together with Vaxrank so Isovar's versioned CTA
catalogue API can be installed in the same environment. OncoRef 1.23.1 keeps
its historical CTA compatibility APIs separate from `cta_selection`; this
dependency update does not silently change PirlyGenes panel membership or
adopt the new selection policy tracked in #655.

The 1.23.1 patch adds pinned Ensembl 115 transcript/protein reference mappings
for human CTA snapshots and the optional sequence-asset verification API.
Historical CTA membership and expression/source data identities are unchanged.
Vaxrank 3.48.0 accepts OncoRef 1.23.1, including its CTA extra.

OncoRef retains data version 5.23.26 and source-matrix version 5.22.14.
Keep PirlyGenes data version 6.0.13 and its original materialized build
provenance. Run the CTA identity/peptide tests,
data compatibility tests, full serial suite, lint, and joint installation checks.
Publish only after downstream pins and their release gates are coordinated.
