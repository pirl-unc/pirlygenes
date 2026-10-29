# OncoRef 1.22 compatibility

Update the exact owner pin together with Vaxrank so Isovar's versioned CTA
catalogue API can be installed in the same environment. OncoRef 1.22.0 keeps
its historical CTA compatibility APIs separate from `cta_selection`; this
dependency update does not silently change PirlyGenes panel membership or
adopt the new selection policy tracked in #655.

Preserve materialized data build provenance and verify the actual owner data
identity before retaining existing assets. Run the CTA identity/peptide tests,
data compatibility tests, full serial suite, lint, and joint installation checks.
Publish only after downstream pins and their release gates are coordinated.
