# OncoRef 1.8.208 consumer adoption

PirlyGenes 6.0.10 pins the public OncoRef 1.8.208 wheel for
[#634](https://github.com/pirl-unc/pirlygenes/issues/634). This is a code-only
release. OncoRef's default/candidate CTA sets remain 624/2,532; its data/source
versions remain 5.23.25/5.22.14. Every shipped `oncoref/data/` file is byte-identical
to the public 1.8.207 wheel. PirlyGenes retains data bundle 6.0.5 and its original
OncoRef 1.8.204 build provenance; the existing bundle passes the data-identity
compatibility guard without modifying its metadata.

## Consumer behavior

`analyses/cta_patient_counts.py` now delegates peptide counts to
`oncoref.cta_peptides.cta_specific_9mer_counts`. `_apd1_factors.py` uses that same
public owner API for peptide weights. The local canonical-ID-only background
calculation and both readers of `outputs/_cache/cta_specific_9mers.csv` are gone.
Historical CSVs and previously generated figures remain untouched and are not
asserted to contain the new identity evidence. Regenerating an analysis obtains
current owner counts; it does not rewrite historical outputs.

OncoRef selects the newest usable installed human annotation. The analysis
helper retains `k` and `refresh`, but no longer accepts its old `ensembl_release`
keyword: the public owner count API has no explicit-annotation parameter.
Its cache includes k, annotation release, canonical CTA membership, contract-v1
identity decisions and mapping-reference hashes. An unavailable or incomplete
proteome raises the owner's error instead of falling back to old CSV counts.

Canonical CTA output IDs, expression source/cohort identities, cDNA/protein
collapse rules and member-ID provenance are unchanged. Verified annotation
aliases affect peptide exclusions; symbol equality or identical protein sequence
does not create a gene alias. The downstream folded weight remains the maximum
member count once per expression proteoform, preserving the existing grouping.

## Public-wheel audit, 2026-10-08

The isolated Python 3.12 environment installed OncoRef from PyPI, with no editable
owner checkout. [Audit evidence](audits/oncoref-1.8.208.json) records both public
wheel hashes, annotation input hashes, mapping decisions, retained source
transcripts/proteins, count-table hashes and unchanged expression build metadata.

| Full human annotation | PRAME distinct 9-mers | Old background matches | Retained non-CTA matches | Specific 9-mers |
| --- | ---: | ---: | ---: | ---: |
| Ensembl 93 | 501 | 501 | 14 | 487 |
| Ensembl 112 | 501 | 501 | 14 | 487 |

The audit scanned all annotated protein-coding transcripts and retained genuine
PRAMEF-family sources. In each annotation, ENSG00000275013 is a verified alternate
source for canonical PRAME ENSG00000185686. The PirlyGenes analysis wrapper over
the public owner API returned 624 canonical CTA rows and PRAME 501/487. Clearing
the in-process cache and disabling the builder verified disk-cache reuse for
both releases. Only annotation discovery was constrained in the audit to exercise
each release; counting, membership, identity validation and cache keys came from
the installed public wheel.

Consumer regressions exercise verified alternate-locus exclusion, annotation
conflicts, unknown same-symbol loci, identical non-CTA sequences, retained shared
peptides, annotation-driven cache invalidation, parameter forwarding, historical
CSV preservation and downstream folded weights. Existing protein-group and
source-identity tests remain part of the full serial suite.

## Coordinated installation

[Vaxrank #592](https://github.com/openvax/vaxrank/issues/592) owns Vaxrank's
admission, negative-reference and native-replay adoption. Vaxrank 3.40.0 still
requires `oncoref==1.8.207`; it cannot share an environment with PirlyGenes 6.0.10.
The shared OpenVax environment must wait for a compatible Vaxrank release. This
adoption was validated in an isolated repository release environment and does not
force the shared installation, downgrade any consumer, or refresh unrelated
packages.
