# Audit: expression normalization, clean-TPM, and symbol mapping

Scope: current ownership and consistency of native-unit conversion,
**clean-TPM**, and identifier mapping. The source-level findings were first
audited in June 2026; ownership moved completely to oncoref under #528.

## Executive summary

- Oncoref is the single ingestion and source-gene mapping owner. Pirlygenes
  consumes its matrices and summaries and carries no builder fallback.
- Native counts and recount3 coverage are length-normalized exactly once;
  FPKM, RPKM, TPM, microarray proxies, and 3′/UMI pseudobulk are not
  incorrectly length-normalized a second time.
- Every source reaches the canonical 16/9/75 clean-TPM contract.
- Ensembl IDs are unversioned and canonical across cohorts.
- Multiple assay/source cohorts remain separate. They are selectable and
  visible in provenance rather than being averaged across incompatible scales.

The sections below provide the evidence and remaining follow-ups behind those
conclusions.

## Detailed findings

### 1. Native unit → TPM → clean TPM

Oncoref's public expression-builder layer is the unit-conversion authority.
Pirlygenes' `expression.normalize.clean_tpm_matrix` is a compatibility wrapper
over the same clean-TPM policy, not an ingestion path. Clean TPM follows the
canonical 16/9/75 compartment contract: oncoref's
`clean-tpm-censored-genes` rows
with `category == "ribosomal_protein"` receive 16% of each sample's 1e6 budget,
rows with `category == "technical"` receive 9%, and all other biological rows
receive 75%. `technical_rna_mask` is the strict technical subset used by lower
level masking/QC paths, not the clean-TPM compartment contract.

| native unit | sources | length-normalized? | path |
| --- | --- | --- | --- |
| raw counts / HTSeq | FL, MCL, MPN, NET(old), heme | **yes** — counts ÷ gene length | oncoref source builder |
| recount3 coverage gene-sums | NET, MDS, PANNET-prim, HL | **yes** — coverage ÷ exonic bp_length | `oncoref.expression_builders.recount3_gene_sums_to_tpm` |
| FPKM | HL(old), pan-cancer | no (already length-normalized) | renormalize to 1e6 |
| RPKM | CML | no (already length-normalized) | renormalize to 1e6 |
| TPM | GDC STAR (`tpm_unstranded`), CLL-map, most GEO | no (already TPM) | renormalize to 1e6 |
| log2(TPM+1) | Treehouse PolyA / RiboD | no | inverse `2^x−1` → renormalize |
| microarray intensity | MTC, LPS | **no — correct** (intensity ∝ concentration, not length×conc.) | probe-max → anti-log2 → sum-to-1e6 (TPM-*proxy*) |
| scRNA pseudobulk nTPM | CTCL | **no — correct** (UMI/3′ counts are length-agnostic) | pseudobulk → counts-per-million |

#### Unit-conversion conclusions
- **Counts are length-normalized; FPKM/RPKM/TPM are not re-length-normalized** (they already are — re-dividing would double-count). Correct.
- **Microarrays are *not* length-normalized** — correct: a probe measures transcript concentration directly, so the array TPM-*proxy* needs no length term. (It is *not* absolute-comparable to RNA-seq TPM; flagged in the `processing_pipeline` tag and surfaced by `pirlygenes data sources`.)
- **scRNA pseudobulk is not length-normalized** — correct for UMI/3′ data.
- **There is no CPM-unit source.** (The only "CPM" in the tree is the gene *Carboxypeptidase M*.) The CTCL scRNA path is the one CPM-like quantity, and it correctly skips length normalization.
- Source clean TPM and the compatibility-layer clean TPM use the same canonical
  `clean_tpm_matrix` path. Lower-level technical-RNA zeroing still uses the
  narrower strict technical mask when explicitly requested.

**Minor / follow-ups**
- Registry rows without a complete executable upstream build route are tracked
  in [oncoref #450](https://github.com/pirl-unc/oncoref/issues/450).
- Missing structured accessions/provenance fields are tracked in
  [oncoref #451](https://github.com/pirl-unc/oncoref/issues/451).

### 2. Symbol / synonym mapping

Oncoref's source canonicalization is authoritative: it resolves source symbols
and identifiers, follows retired-locus mappings, and emits unversioned canonical
ENSG keys. Pirlygenes delegates its lowest-tier synonym lookup to
`oncoref.resolve_symbol` and retains only display aliases and compatibility
lookups over installed Ensembl releases.

The original migration recovered retired symbols such as
`HIST1H1T`→`H1-6` and `GNB2L1`→`RACK1`. With the local builder fleet removed,
there is no second source-level resolver that can drift from oncoref.

### 3. Multi-source cohorts — semantics and visibility

A cancer code can have **multiple sources** (e.g. NET_PANCREAS = liver-met
recount3 + primary recount3; SARC_DDLPS = Treehouse RNA-seq + GEO RNA-seq +
microarray). These are **kept as separate `source_cohort` rows and are
NOT averaged/merged** — different assays and quantification scales
(microarray TPM-proxy is not comparable in magnitude to RNA-seq TPM).
Consumers select or compare explicitly; e.g. the CTA heatmaps pick the
most-gene-rich source per code.

Now visible via:

    pirlygenes data sources [CODE] [--multi]

which lists, per cancer code, each source with its **n_samples**, **gene
count**, and **native unit** (derived from the `processing_pipeline`
provenance tag). Example:

    SARC_DDLPS  (multi-source — kept separate, not merged)
        TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY  n=48  genes=34571  RSEM log2(TPM+1)
        GSE75885_DELESPAUL_2017            n=19  genes=18499  TPM
        GSE30929_SINGER_2007_LPS           n=40  genes=13654  microarray intensity (TPM-proxy)
