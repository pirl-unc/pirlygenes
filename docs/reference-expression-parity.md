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

Ordered roughly by severity. These are the items to resolve (mostly on the
oncoref side) before trusting a full handoff.

### Scale / normalization mismatches (same samples, wrong values)

| code | n_samp pg/on | rel median | note |
| --- | --- | --- | --- |
| `MBL` | 125/125 | **421%** (p95 1026%) | identical sample set, medians on a different scale — a per-source normalization/unit bug |
| `MM` | 764/764 | **133%** | same — large cohort, so not a small-n artifact |
| `SARC_LGFMS` | 2/2 | **49%** (p95 180%) | ~6.3k of ~18.8k scored genes divergent — a per-source scale issue |
| `HL` | 5/5 | 45% | small cohort, but the shift is systematic |
| `SARC_CCS` | 5/5 | 22% | |
| `NPC` | 4/4 | 18% | |

These matter most: the sample sets match but the summarized expression does not,
which points at how the source matrix was unit-converted/normalized on one side.

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
