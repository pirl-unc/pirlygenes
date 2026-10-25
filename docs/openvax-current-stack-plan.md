# Current OpenVax stack and OncoRef adoption

## Objective

Resolve PirlyGenes #652 and make the active OpenVax stack use the newest
mutually compatible stable releases. Validate released packages together in an
isolated environment before updating the shared environment. Preserve editable
source locations, local work, unrelated notebook/AI packages, and all existing
reference caches.

## Initial audit (24 October 2026)

- Published PirlyGenes 6.0.17 pins OncoRef 1.13.0; published Vaxrank 3.46.11
  pins OncoRef 1.19.2. This is an unsatisfiable joint installation.
- Vaxrank's `hitlist` extra requires Hitlist below 2, while current Hitlist is
  2.0.35. Audit the portable evidence consumer and validate current producer
  output before broadening that constraint in a separate Vaxrank release.
- Other current releases include PyEnsembl 2.32.3, DataCache 1.29.3,
  GTFParse 3.0.5, Varcode 10.18.5, Isovar 1.59.22, Topiary 5.95.3,
  MHCtools 4.4.3, MHCflurry 2.3.16, MHCgnomes 3.64.6, Pepdata 1.2.4,
  Serializable 1.3.2, Tsarina 1.35.1, Weirdo 2.2.0 and Trufflepig 1.30.3.
- Shared-environment metadata is stale for some editable packages, but other
  checkouts genuinely contain older feature branches or unpublished work.
  ORFdata 0.36.0 and Varlens 1.0.0 are ahead of their published releases and
  must not be downgraded. Determine each case from source and Git state.
- PirlyGenes' published code uses data bundle 6.0.13. OncoRef's expression
  data and source versions, artifact hashes and compatibility APIs determine
  whether that bundle remains valid; package version differences alone do not
  justify regeneration or relabeling original build provenance.

## Implementation sequence

1. Inventory package metadata, editable paths, source versions, Git state and
   current PyPI requirements. Save the existing shared `pip check` result and
   package inventory. Resolve the full requested package set, including the
   Hitlist and Weirdo integrations, in isolation.
2. File newly found incompatibilities in their owning repositories. For
   Vaxrank/Hitlist, exercise real portable evidence production, loading,
   serialization/replay and invalid-input handling with current Hitlist.
   Change only the supported constraint and any genuinely necessary adapter
   code; retain backward compatibility with previously supported evidence.
3. Advance PirlyGenes' exact OncoRef pin to the current validated stable
   release and bump its code version. Compare owner data/source identities,
   existing materialized matrix metadata, availability snapshots, delegated
   public accessors and CTA identity behavior. Keep DATA_VERSION, original
   build provenance and all bundled data unchanged when these remain valid.
4. Add behavioral regression coverage for unchanged installed-bundle reads
   and current owner interoperability. Validate the joint current stack via
   the real resolver and `pip check`; avoid tests that merely repeat version
   declarations. Record the exact validated versions and remaining constraints.
5. Run repository lint, the full serial PirlyGenes suite, appropriate
   downstream producer/consumer tests, package checks and CI. Use disposable
   or copy-on-write cache copies for integration tests; no repairs, rebuilds,
   pruning, moves or downloads into existing user cache directories.
6. Merge feature PRs and deploy from clean main using each repository's
   release script. Verify public wheels/sdists and release tags. Recheck stable
   releases before finalizing the coordinated pins.
7. Rehearse shared-environment changes in an isolated copy. Refresh stale
   editable metadata from its existing source. Advance clean checkouts only
   when a fast-forward to the tested release preserves local work; never
   replace an editable checkout with a wheel, reset a branch, discard changes,
   invent a source version, or downgrade unpublished work. Report any source
   constraint that prevents a current release. Apply the validated changes
   together, then run `pip check` and relevant integration checks. Verify
   unrelated packages and editable source locations remain unchanged.

## Acceptance evidence

- Current stable PirlyGenes and Vaxrank install together with one OncoRef.
- Current Hitlist is supported by the Vaxrank integration, or a concrete
  reproducible incompatibility is reported and filed.
- Existing PirlyGenes data remains readable with truthful original provenance;
  integration reads preserve copied cache contents and file metadata.
- All required tests and CI pass, and changed packages are published.
- The shared environment passes `pip check`, preserves editable work and
  unrelated packages, and has an explicit per-package account of any version
  that cannot safely advance to the latest release.

## Bundle compatibility evidence

Before/after checks against the existing 6.0.13 bundle compare OncoRef 1.13.0
and 1.19.2. Owner data 5.23.26 and source matrices 5.22.14 remain unchanged.
TPM, cleaned TPM, biological matrices, provenance, the availability manifest,
624 CTA IDs and 2,533 candidate IDs match. Original build metadata remains
OncoRef 1.8.210, and copied cache bytes, inode, permissions and timestamps
are unchanged. No new data release or cache rebuild is required.

A new regression reads an installed, read-only synthetic matrix bundle through
the public accessor and rejects any attempt to rebuild or install it. The 69
focused owner, matrix and manifest tests pass. Existing and freshly generated
Hitlist evidence also pass Vaxrank interoperability checks. Tests use an
isolated environment and copied caches; OncoRef shard-cache configuration is
tracked separately in pirl-unc/oncoref#626.

The full serial suite passes (1,091 tests). Cold/offline validation identified
an independent provenance-manifest download requirement, tracked in #654. The
cache-lifecycle fixture supplies that independent manifest explicitly; all three
owner-adoption tests then pass with network access disabled and empty owner
caches. Real-bundle before/after checks remain unchanged.
