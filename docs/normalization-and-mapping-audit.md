# Audit: expression normalization, clean-TPM, and symbol mapping

Scope: how every cohort builder converts its native quantification to the
comparable **clean-TPM** reference values, and whether identifier/synonym
mapping is uniform. Done June 2026 alongside the recount3 integration.

## 1. Native unit → TPM → clean TPM

There is **one** unit dispatcher,
`pirlygenes.builders.geo_matrix.normalize_to_tpm`, and **one** clean-TPM
transform, `expression.normalize.clean_tpm_matrix` + `technical_rna_mask`
(zero mtDNA / rRNA-like / mt-like pseudogene / polyA-bias lncRNA, renormalize
each sample column to 1e6; ribosomal-protein mRNA kept). Every per-sample
builder imports the *same* clean-TPM helper.

| native unit | sources | length-normalized? | path |
| --- | --- | --- | --- |
| raw counts / HTSeq | FL, MCL, MPN, NET(old), heme | **yes** — counts ÷ gene length | `normalize_to_tpm(unit="raw_counts", gene_lengths_kb=…)` |
| recount3 coverage gene-sums | NET, MDS, PANNET-prim, HL | **yes** — coverage ÷ exonic bp_length | `recount3.gene_sums_to_tpm` → delegates to the same `raw_counts` path |
| FPKM | HL(old), pan-cancer | no (already length-normalized) | renormalize to 1e6 |
| RPKM | CML | no (already length-normalized) | renormalize to 1e6 |
| TPM | GDC STAR (`tpm_unstranded`), CLL-map, most GEO | no (already TPM) | renormalize to 1e6 |
| log2(TPM+1) | Treehouse PolyA / RiboD | no | inverse `2^x−1` → renormalize |
| microarray intensity | MTC, LPS | **no — correct** (intensity ∝ concentration, not length×conc.) | probe-max → anti-log2 → sum-to-1e6 (TPM-*proxy*) |
| scRNA pseudobulk nTPM | CTCL | **no — correct** (UMI/3′ counts are length-agnostic) | pseudobulk → counts-per-million |

**Conclusions**
- **Counts are length-normalized; FPKM/RPKM/TPM are not re-length-normalized** (they already are — re-dividing would double-count). Correct.
- **Microarrays are *not* length-normalized** — correct: a probe measures transcript concentration directly, so the array TPM-*proxy* needs no length term. (It is *not* absolute-comparable to RNA-seq TPM; flagged in the `processing_pipeline` tag and surfaced by `pirlygenes data sources`.)
- **scRNA pseudobulk is not length-normalized** — correct for UMI/3′ data.
- **There is no CPM-unit source.** (The only "CPM" in the tree is the gene *Carboxypeptidase M*.) The CTCL scRNA path is the one CPM-like quantity, and it correctly skips length normalization.
- The same technical-RNA group set drives both the builder clean-TPM and the analysis-layer transform (`_DEFAULT_NORMALIZE_REMOVE_GROUPS = _TECHNICAL_RNA_GROUPS`).

**Minor / follow-ups**
- recount3 now delegates its length-normalization to the shared
  `normalize_to_tpm` `raw_counts` path (was a private copy) — one route.
- `normalize_to_tpm`'s `log2(TPM+1)` branch inverse-transforms but does not
  re-renormalize; only the Treehouse builder uses log2 input and it
  renormalizes downstream, so no live source is affected. Harmless but worth
  tightening if a log2 GEO-matrix source is ever added.

## 2. Symbol / synonym mapping — NOT yet uniform

Identifier mapping has a single intended home,
`pirlygenes.builders.gene_mapping.resolve_symbol` (direct pyensembl →
Entrez chain → NCBI-synonym + curated-alias rescue), which recovers renamed
symbols (`HIST1H1T`→`H1-6`, `GNB2L1`→`RACK1`). It is used by:
- the generic GEO-matrix builder (`geo_matrix._harmonize_by_symbol`), and
- the Treehouse builder (`builders/treehouse.py`).

But several **older symbol-keyed builders still call raw
`genome.genes_by_name(symbol)`** with no synonym rescue:
`build_geo_heme_reference_expression.py` (CML), `build_cllmap_*` (CLL),
`build_ctcl_*` (CTCL), `import_cancer_specific_expression.py` (summary
imports), and the standalone `build_treehouse_reference_expression.py`.
The GDC builders also call `genes_by_name`, but they are **ENSG-keyed**
(STAR `tpm_unstranded` by Ensembl ID), so synonym rescue doesn't apply to
their aggregation — lower risk.

**Impact:** in those symbol-keyed cohorts, a gene whose source uses a
retired HGNC symbol absent from Ensembl 112 is silently dropped (becomes
`not_measurable`) instead of rescued — the same class of gap the recount3
work fixed for NET. Mostly rare-cancer cohorts (CLL/CML/CTCL) and small
summary imports.

**Recommended fix (follow-up):** route every symbol-keyed builder through
`resolve_symbol`, then rebuild the affected cohorts. ENSG-keyed (GDC,
recount3) and already-migrated (Treehouse, GEO-matrix) paths need no change.

## 3. Multi-source cohorts — semantics & visibility

A cancer code can have **multiple sources** (e.g. PANNET = liver-met
recount3 + primary recount3; SARC_DDLPS = Treehouse RNA-seq + GEO RNA-seq +
microarray). These are **kept as separate `source_cohort` shards and are
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
        TREEHOUSE_POLYA_25_01_TCGA_SUBSET  n=48  genes=34571  RSEM log2(TPM+1)
        GSE75885_DELESPAUL_2017            n=19  genes=18499  TPM
        GSE30929_SINGER_2007_LPS           n=40  genes=13654  microarray intensity (TPM-proxy)
