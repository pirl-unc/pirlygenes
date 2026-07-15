# recount3 as a coverage-complete drop-in for spotty GEO sources

Several of our GEO RNA-seq sources ship a *processed* matrix with an
incomplete gene universe — HTSeq counts keyed to a stale Entrez/GTF set,
author-chosen FPKM/TPM tables — leaving genuine coverage holes. The
motivating case: **GSE98894 (SRP107025)** HTSeq counts never annotated the
near-identical CTA paralogs `XAGE1A`/`XAGE1B`, so they are *absent*
(`not_measurable`), not zero, in NET_MIDGUT / NET_PANCREAS / NET_RECTAL.

[recount3](https://rna.recount.bio/) (Wilks 2021) uniformly re-quantified
~750k public SRA/GTEx/TCGA RNA-seq runs with Monorail (STAR → base coverage)
and summarized to genes against **Gencode v26** (`G026`, ~63.8k genes). Where
it covers a study, swapping our source for the recount3 gene-sums replaces
spotty coverage with the full Gencode universe on one consistent scale.
Confirmed for SRP107025: `XAGE1A` goes from *absent* to max 218 TPM (measured
in 20/212 samples).

## How many of our sources can use it

recount3's human SRA release is a frozen **~2019 snapshot**, and it is
RNA-seq only. Probing every source (GSE→SRP via NCBI elink, then the
recount3 S3 path) gives **4 of 15 GEO RNA-seq sources** covered today:

| source-id | SRP | codes | recount3 |
| --- | --- | --- | --- |
| `gse98894-midnet` | SRP107025 | NET_MIDGUT, NET_PANCREAS, NET_RECTAL | ✅ 212 samples |
| `gse114922-mds` | SRP149374 | MDS | ✅ |
| `gse118014-pannet` | SRP156049 | NET_PANCREAS (primary) | ✅ |
| `gse120328-hl` | SRP162356 | HL | ✅ |
| `gse100026-cml` | SRP109177 | CML | ❌ not in release |
| `gse142334-fl` | SRP238213 | FL | ❌ past snapshot |
| 2023–2026 GEO series¹ | — | various sarcoma/heme | ❌ post-snapshot |
| `gse171811-ctcl` | — | CTCL | ❌ scRNA, not bulk |

¹ `gse271664-mcl`, `gse283710-mpn`, `gse248751-sarc-ccs`,
`gse241095-sarc-ks-skin`, `gse328026-sarc-pec`, `gse299759-chon`.

**Not candidates (no coverage gap to fill):**
- **GDC sources** (CGCI-BLGSP, MMRF, TARGET-*, all 32 `tcga-*`): already
  STAR-counts with the complete ~60k-gene GENCODE v36 universe. recount3's
  TCGA would be redundant *and* older (v26).
- **Treehouse** (polya / ribod): symbol-keyed RSEM compendium; its gap is the
  HUGO→ENSG collapse (largely fixed), and its samples are a TCGA/TARGET/GTEx/
  GEO mix, not a single SRP.
- **Microarray** (`gse32662-mtc`, `gse30929-lps`): not RNA-seq.
- **Already-uniform** GEO (`drmetrics-lnen-2020`: STAR+featureCounts GENCODE
  v33; `cllmap`: GENCODE v19 TPM): complete coverage already.

## Normalization: gene-sums → clean TPM

recount3 `gene_sums` are **coverage sums** (Σ per-base read depth over a
gene's disjoint exonic bases), *not* read counts. The transform is owned by
`oncoref.expression_builders`:

1. **Length-normalize to TPM** (per sample column `j`):
   `rate[g,j] = gene_sums[g,j] / bp_length[g]`, then
   `TPM[g,j] = 1e6 · rate[g,j] / Σ_h rate[h,j]`.
   `bp_length` is the gene's **exonic** length under Gencode v26 (the score
   column of recount3's annotation GTF) — the exact bases the coverage was
   summed over, *not* the gene span. The per-sample read-length and
   library-size factors cancel in the renormalization, so **no AUC/library
   metadata is needed** for TPM.
2. **Harmonize IDs**: strip `.<version>` / `_PAR_Y`, sum collisions → one row
   per unversioned ENSG.
3. **Clean TPM**: apply the canonical 16/9/75 clean-TPM transform through
   `expression.normalize.clean_tpm_matrix`. Rows in
   `clean-tpm-censored-genes.csv` with `category == "ribosomal_protein"` are
   pinned to 16% of the 1e6 budget, rows with `category == "technical"` are
   pinned to 9%, and all other biological rows receive the remaining 75%.

Result: ENSG × sample clean TPM on the same scale and technical-RNA
treatment as our GDC/Treehouse/GEO shards, ready for the usual
per-(gene, cancer_code) median/q1/q3/n summarization.

**Caveats.** STAR splits multimapping reads across near-identical paralogs,
so per-paralog values (XAGE1A vs XAGE1B) are approximate — but *measured*
rather than absent, which is the meaningful gain. Gencode v26 (recount3) vs
GENCODE v36 (GDC) vs RSEM (Treehouse) all mix acceptably as "RNA-seq TPM" for
cohort-level summaries, exactly as the sources we already combine.

## Ownership and rebuilds

oncoref is the sole owner of recount3 ingestion. Its registry records the
study-specific routing and expected sample counts, while
`build_recount3_source_matrices` fetches the annotation, gene sums, and SRA
metadata; aggregates runs into biological samples; canonicalizes gene IDs;
and writes per-code matrices with mapping, parse, QC, and summary sidecars.
pirlygenes consumes those published matrices and summaries and retains only
the `recount3_srp` and `source_cohort` provenance in its registry.

To rebuild one source in the oncoref development environment:

```python
from pathlib import Path

from oncoref.expression_builders import (
    build_recount3_source_matrices,
    recount3_source_from_registry,
)

source = recount3_source_from_registry("gse98894-midnet")
build_recount3_source_matrices(
    source,
    cache_dir=Path.home() / ".cache" / "oncoref" / "recount3",
)
```

The resulting source keeps the same canonical `source_cohort` tag, so the
published oncoref artifact replaces the coverage-incomplete source in place.
