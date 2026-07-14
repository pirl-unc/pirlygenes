# Reference-expression delegation parity (#557)

`pirlygenes.cancer_reference_expression()` is a compatibility wrapper over
`oncoref.cancer_reference_expression()`. Oncoref owns the empirical rows,
provenance, source selection, and downloadable summary artifact; pirlygenes
preserves its historical schema, normalization labels, source-union view,
gene-ID space, pooling, and proteoform-collapse surface.

The wrapper requests oncoref's all-sample `summary_rows_all` product because the
pirlygenes API exposes every source cohort and supports `source_kind=`,
`source_cohort=`, proxy exclusion, and heterogeneity-safe pooling. It never
falls back to a pirlygenes expression artifact. `DataFrame.attrs` records the
delegation target, availability/missing requests, and every compatibility
transform.

## Runtime ownership

`cancer-reference-expression` is no longer a pirlygenes download-bundle member,
and both `get_data()` and `load_all_dataframes()` route its runtime reads to
oncoref. The in-repository `pirlygenes/data/cancer-reference-expression/` shards
remain only as builder/audit inputs while the local ingestion fleet is retired
under #528; no supported public read path selects them.

## Compatibility transforms

The adapter performs four deterministic operations over delegated rows:

- maps oncoref normalization labels back to `TPM`, `TPM_clean`,
  `TPM_log1p`, and `TPM_clean_log1p`;
- expands legacy pirlygenes gene aliases before the delegated filter and derives
  both log views with `numpy.log1p` from oncoref's delegated linear summaries;
- maps the DDLPS/WDLPS rows from oncoref's stale generic TCGA-subset storage
  label to the dedicated SARC-histology label advertised by oncoref's registry,
  including translation of canonical `source_cohort=` filters;
- preserves pirlygenes proteoform IDs and applies identical-locus collapse to
  delegated linear rows while oncoref's summary-union collapse retains its
  `n_detected` provenance bug. This exception is exposed in
  `attrs["compatibility_transforms"]` and has no local-data fallback.

## Validation

`tests/test_reference_expression_delegation.py` compares the wrapper directly
against oncoref for five required reference classes:

- common TCGA (`LUAD`);
- non-TCGA heme (`CLL`);
- microarray proxy (`MTC`);
- molecular subtype (`BRCA_Basal`);
- computed union (`SARC`).

It also proves the legacy-symbol and raw-log adapters cannot invoke the former
local summary-frame loader. The existing expression, source-union, pooling,
legacy-Ensembl-ID, and proteoform-collapse tests continue to gate the historical
public contract.

## Full audit report

Run:

```bash
python scripts/parity_reference_expression.py
```

The committed [Markdown report](reference-expression-delegation-557.md) and
[per-code CSV](reference-expression-delegation-557.csv) were generated with
oncoref 1.8.124. Headline results:

- 128 source-union cancer codes audited;
- 124 served by both the compatibility and canonical selected/QC-aware views;
- 124/124 exact reference-sample-count agreement;
- median per-code relative expression delta: 0.129%.

This sweep intentionally compares two different oncoref products. Large value
outliers such as MBL, HL, and MM reflect all-sample source-union rows versus the
default pass-QC selected artifact; they are not adapter distortion. The four
unserved canonical comparisons are the MBL molecular subgroups. Exact
adapter-to-source value parity is covered separately by the five-class test.
