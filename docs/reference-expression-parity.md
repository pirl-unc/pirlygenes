# Reference-expression parity: pirlygenes vs oncoref (#207)

pirlygenes ships pre-built per-`(gene, cancer_code, source_cohort)` summary rows
in `cancer-reference-expression.csv.gz`. oncoref computes the *same* rows on
demand from its source-matrix artifact (`oncoref.cancer_reference_expression`).
Before pirlygenes retires its own builders and becomes a pure consumer of
`oncoref.expression_builders`, we need to know exactly where the two agree and
where they diverge. This harness measures that.

## Running it

```bash
python scripts/parity_reference_expression.py                  # every code
python scripts/parity_reference_expression.py --codes PRAD MBL # a subset
```

Writes `parity_by_code.csv` + `parity_report.md` under
`analyses/outputs/reference_expression_parity/`. The core lives in
`pirlygenes.expression.parity` (`parity_for_code`, `parity_report`,
`format_markdown`) so notebooks and tests can call it directly.

The comparison joins the two frames on `(cancer_code, Ensembl_Gene_ID)` at
`tpm_clean` and reports, per code: reference-sample-count agreement, the
relative-delta distribution of the median `expression` for genes above a 1 TPM
floor, and gene-universe deltas (genes on only one side). Each cohort is read
under the QC policy its oncoref artifact was baked with (`pass`, falling back to
`pass_or_warn` / `all`).

oncoref serves exactly one canonical `source_cohort` per code; pirlygenes' bundle
sometimes carries several (e.g. `SARC_DDLPS` spans three cohorts). The harness
pairs each code against the pg cohort oncoref actually used — matched by
reference-sample count — so multi-cohort codes are compared apples-to-apples
rather than as a many-to-many blur.

## Headline

Sweep of all 120 pirlygenes cancer_codes (oncoref 1.8.98, bundle 5.23.2):

- **116/120** served by both sides; 4 (`MBL_G3/G4/SHH/WNT`) return no oncoref rows.
- **116/116** agree on `n_samples` exactly (once multi-cohort codes are paired to
  oncoref's chosen cohort).
- Median relative delta across codes: **0.15%**.
- For ~90 of the codes, median delta is ~0.05% and p95 ~0.13% — i.e. **parity to
  float/rounding noise**. Every large-cohort TCGA type (BRCA 1099, CLL 708, KIRC
  531, LGG 523, THCA 512) is in this bucket.

So the common case is already parity. The value of the harness is the tail.

## Divergences that need attention

Ordered roughly by severity. The largest class turns out to be a **pirlygenes**
problem the handoff itself fixes (a stale bundle over-counting technical genes);
the remainder are an oncoref coverage gap and a shared gene-universe question.

### Compositional scale shifts from technical-gene multimapping (pg bundle is stale)

| code | n_samp pg/on | rel median | on/pg ratio (real genes) |
| --- | --- | --- | --- |
| `MBL` | 125/125 | **421%** (p95 1026%) | ~5.3x |
| `MM` | 764/764 | **133%** | ~2.3x |
| `SARC_LGFMS` | 2/2 | **49%** (p95 180%) | — |
| `HL` | 5/5 | 45% | — |
| `SARC_CCS` | 5/5 | 22% | — |
| `NPC` | 4/4 | 18% | — |

Same sample set, but every real gene's median is inflated on the oncoref side by
a near-constant factor. The cause is **not** a normalization bug in oncoref — it
is the compositional (sum-to-a-million) nature of TPM reacting to a handful of
**technical genes that pirlygenes' *current* bundle over-counts**:

- `MBL`: pg reports `RNU1-28P` (a spliceosomal-snRNA pseudogene) at **228,608
  TPM** — 23% of the entire transcriptome from one gene — plus large piles on
  `RN7SL1`/`RN7SK`/`RNU4-2`/`RNU2-2`. These are classic multimapping-magnet
  loci; oncoref's builder values `RNU1-28P` at ~85 TPM.
- `MM`: pg reports `IGKC` (a rearranged immunoglobulin constant segment) at
  **234,899 TPM**; oncoref ~7,400.

`classify_gene_qc` tags every one of these `small_ncrna` or `immune_receptor` —
the technical classes a clean-TPM is meant to exclude. When oncoref drops/deflates
them, the freed compositional budget re-inflates all real genes, so the median
ratio tracks how much of pg's TPM those technical piles consumed (~80% for MBL,
~60% for MM). **oncoref is the more-correct side here.** The pg bundle predates
the builder→oncoref delegation (#527); rebuilding it from `oncoref.expression_builders`
(the handoff this harness gates) is what resolves these codes — no oncoref fix
needed. `SARC_LGFMS`/`HL`/`SARC_CCS`/`NPC` are the same effect at smaller cohorts.

### Coverage gaps

- `MBL_G3`, `MBL_G4`, `MBL_SHH`, `MBL_WNT` — oncoref returns no rows; pirlygenes
  ships these medulloblastoma molecular subgroups (44/39/25/17 samples). oncoref
  computes only the pooled `MBL`.

### Systematic gene-universe gap

Nearly every TCGA-scale code shows **~1686 genes present in pirlygenes but not
oncoref**, and **~225 in oncoref but not pirlygenes**. The pg-only set is
dominated by the ncRNA/lncRNA/miRNA tail (e.g. `MIR3179-2`, `LINC01422`) — genes
that gene-universe filtering and sequence-identity canonicalization treat
differently between the two artifacts. This is the source of most per-code
`divergent` gene counts and is a deliberate-choice question, not necessarily a bug.

## Regression guard

`tests/test_reference_expression_parity.py` locks in the *shape* of parity on
`PRAD`/`LUAD` (exact `n_samples`, median delta < 1%, p95 < 5%, >20k shared genes)
and the `pass_or_warn` fallback on `MTC`. It skips when oncoref's artifact is
unavailable. The loose ceilings catch a structural break (wrong join, a
unit/scale regression, a vanished cohort) without pinning float noise.
