# OncoRef 1.23 compatibility

Update the exact owner pin together with Vaxrank so Isovar's versioned CTA
catalogue API can be installed in the same environment. OncoRef 1.23.0 keeps
its historical CTA compatibility APIs separate from `cta_selection`; this
dependency update does not silently change PirlyGenes panel membership or
adopt the new selection policy tracked in #655.

OncoRef retains data version 5.23.26 and source-matrix version 5.22.14.
Keep PirlyGenes data version 6.0.13 and its original materialized build
provenance. Run the CTA identity/peptide tests,
data compatibility tests, full serial suite, lint, and joint installation checks.
Publish only after downstream pins and their release gates are coordinated.

## OncoRef 1.23.1

PirlyGenes 6.0.20 pins OncoRef 1.23.1, which adds human Ensembl 115
transcript/protein mappings bound to verified sequence assets. This permits
the published `vaxrank[cta]` consumer to install with current Topiary. The
historical CTA panel, owner data/source versions and PirlyGenes data version
remain unchanged. Existing materialized files keep their original builder
version; this package upgrade does not relabel or regenerate them.

The current-stack CI job pins the latest Vaxrank (including its `cta` extra),
Topiary, Hitlist, Isovar and Trufflepig releases. It upgrades shared dependencies
within those consumers' declared bounds and records the installed versions.
For example, current consumers exclude Varcode 11 until their candidate
traversal migrations land. Selecting the latest compatible Varcode under those
bounds preserves the declared API contract; pinning its advertised latest major
would instead create an unsupported environment. `pip check`, historical CTA
and peptide-exclusion tests, and the full serial suite remain required.

This coordination resolves #658 and the validation-policy defect in #659.
Adopting a different biological CTA selection policy remains separate in #655.

## OncoRef 1.24.0

PirlyGenes 6.0.21 pins OncoRef 1.24.0. Coordinate this update with a Vaxrank
release whose tested dependency bounds permit both 1.23.1 and 1.24.0. The
historical 624-gene CTA panel, expression/source data versions, sequence-reference
assets and PirlyGenes data version remain unchanged; existing materialized
artifacts retain their original build provenance.

The owner's versioned research selection now records reviewed TCR-T evidence
and baseline/effective RNA ceilings in its explanations. Its 522 protein groups
and 558 member loci are unchanged. That separate policy carries exact
peptide, HLA and product limits; it does not establish clinical clearance of
every isoform or experimental translation of a reconstructed RNA sequence.
See the [owner's release notes](https://pirl-unc.github.io/oncoref/cta-release-1.24.0/)
for policy and source-snapshot provenance. This dependency update resolves #663;
selection-policy adoption remains tracked separately in #655.
